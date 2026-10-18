#include "ds41/hotpath/kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
namespace ds41e::hotpath {
namespace {
using B = __nv_bfloat16;
constexpr int H = 5120;
__global__ void embed(const B *table, const int *tokens, B *h, float *pre) {
  int row = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 4 * H) h[row * 4 * H + i] = table[tokens[row] * H + i % H];
  if (i < 4) pre[row * 4 + i] = i == 0 ? 1.f : 0.f;
}
__global__ void gather(const uint8_t *x, const uint8_t *s, const float *w,
                       const int *source, const int *slots, uint8_t *out,
                       uint8_t *os, float *ow) {
  int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < H) out[r * H + i] = x[source[r] * H + i];
  if (i < H / 32) os[r * (H / 32) + i] = s[source[r] * (H / 32) + i];
  if (!i) ow[r] = w[slots[r]];
}
__global__ void scatter(const B *in, const int *slots, B *out) {
  int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < H) out[slots[r] * H + i] = in[r * H + i];
}
__global__ void capture(const B *h, B *out, int layer) {
  int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H) return;
  h += r * 4 * H;
  float sum = __fadd_rn(__fadd_rn(__fadd_rn(__bfloat162float(h[i]),
      __bfloat162float(h[H+i])), __bfloat162float(h[2*H+i])),
      __bfloat162float(h[3*H+i]));
  out[(r * 3 + layer) * H + i] = __float2bfloat16_rn(__fmul_rn(sum, .25f));
}
__global__ void rms(const B *x,const B *w,B *y,int k) {
  __shared__ float sh[512];
  int r=blockIdx.x,t=threadIdx.x;
  float v[4]={0,0,0,0};
  for(int i=t*4;i<k;i+=blockDim.x*4)
    for(int j=0;j<4;++j) if(i+j<k) {
      float z=__bfloat162float(x[r*k+i+j]);
      v[j]=__fadd_rn(v[j],__fmul_rn(z,z));
    }
  float sum=__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]);
  sh[t]=sum; __syncthreads();
  for(int d=blockDim.x/2;d>=32;d>>=1) {
    if(t<d) { sum=__fadd_rn(sum,sh[t+d]); sh[t]=sum; }
    __syncthreads();
  }
  if(t<32) {
    for(int d=16;d;d>>=1) sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));
    if(!t) sh[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/k),1e-20f));
  }
  __syncthreads();
  for(int i=t;i<k;i+=blockDim.x) y[r*k+i]=__float2bfloat16_rn(
      __fmul_rn(__fmul_rn(__bfloat162float(x[r*k+i]),sh[0]),__bfloat162float(w[i])));
}
__global__ void activation(const B *gate,const B *up,const float *weights,
                           const int *rows,uint8_t *q,uint8_t *s,int groups) {
  int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
  if(group>=groups) return;
  int row=rows[blockIdx.y],col=group*32+lane,off=row*2304+col;
  float g=fminf(__bfloat162float(gate[off]),10.f);
  float u=fminf(10.f,fmaxf(-10.f,__bfloat162float(up[off])));
  float z=__fmul_rn(g/(1.f+expf(-g)),u);
  z=__fmul_rn(weights[row],z);
  z=__bfloat162float(__float2bfloat16_rn(z));
  float mx=fabsf(z);
  for(int d=16;d;d>>=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(mx,1e-4f),1.f/448.f));
  unsigned e=(bits>>23)+((bits&0x7fffff)!=0);
  float inv=__uint_as_float((254-e)<<23);
  q[off]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,__fmul_rn(z,inv))),__NV_SATFINITE,__NV_E4M3);
  if(!lane) s[row*72+group]=uint8_t(e);
}
__global__ void frequencies(const float *table,float *main,float *compressed,
                            const int64_t *position,int rows,int context) {
  int i=threadIdx.x+blockIdx.x*blockDim.x;
  if(i>=rows*64) return;
  int r=i/64,d=i%64;int64_t p=*position+r;
  main[r*64+d]=table[p*64+d];
  main[(6+r)*64+d]=table[(context+p)*64+d];
  int64_t cp=p>0?p-1:0;
  compressed[r*64+d]=table[(context+cp)*64+d];
}
__global__ void store(const uint8_t *records,uint8_t *global,const int64_t *position,
                      int rows,int bytes,int ratio) {
  int i=threadIdx.x+blockIdx.x*blockDim.x;
  if(i>=rows*bytes) return;
  int r=i/bytes;int64_t p=*position+r;
  if((p+1)%ratio==0) global[((p+1)/ratio-1)*bytes+i%bytes]=records[i];
}
__global__ void aux(const int *visible,int *out,int rows,int divisor,int fixed) {
  int r=threadIdx.x;
  if(r<rows) out[r]=fixed>=0?fixed:(visible[r]+divisor-1)/divisor;
}
}
void embedding(const void *table, const int *tokens, void *h, float *pre,
               int rows, cudaStream_t s) {
  embed<<<dim3(80, rows), 256, 0, s>>>((const B *)table, tokens, (B *)h, pre);
}
void gather_routes(const uint8_t *x, const uint8_t *s, const float *w,
                   const int *source, const int *slots, uint8_t *out,
                   uint8_t *os, float *ow, int count, cudaStream_t st) {
  gather<<<dim3(20, count), 256, 0, st>>>(x,s,w,source,slots,out,os,ow);
}
void scatter_routes(const void *in, const int *slots, void *out, int count,
                    cudaStream_t s) {
  scatter<<<dim3(20,count),256,0,s>>>((const B *)in,slots,(B *)out);
}
void capture_rows(const void *h, void *out, int rows, int layer, cudaStream_t s) {
  capture<<<dim3(20,rows),256,0,s>>>((const B *)h,(B *)out,layer);
}
void rms_decode(const void *x,const void *w,void *out,int rows,int cols,cudaStream_t s) {
  int threads=1; while(threads*2<=cols/4 && threads*2<=512) threads*=2;
  rms<<<rows,threads,0,s>>>((const B*)x,(const B*)w,(B*)out,cols);
}
void activate_routes(const void *gate,const void *up,const float *weights,
                     const int *rows,uint8_t *q,uint8_t *s,int count,cudaStream_t st) {
  activation<<<dim3(18,count),128,0,st>>>((const B*)gate,(const B*)up,weights,rows,q,s,72);
}
void stage_frequencies(const float *table,float *main,float *compressed,
                       const int64_t *position,int rows,int context,cudaStream_t s) {
  frequencies<<<(rows*64+255)/256,256,0,s>>>(table,main,compressed,position,rows,context);
}
void store_groups(const void *packed,void *global,const int64_t *position,
                   int rows,int bytes,int ratio,cudaStream_t s) {
  store<<<(rows*bytes+255)/256,256,0,s>>>((const uint8_t*)packed,(uint8_t*)global,position,rows,bytes,ratio);
}
void lengths_aux(const int *v,int *out,int rows,int divisor,int fixed,cudaStream_t s) {
  aux<<<1,32,0,s>>>(v,out,rows,divisor,fixed);
}
} // namespace ds41e::hotpath
extern "C" int ds41_hotpath_rms(const void *x,const void *w,void *out,int rows,
                                int cols,void *stream) {
  ds41e::hotpath::rms_decode(x,w,out,rows,cols,(cudaStream_t)stream);
  return int(cudaGetLastError());
}
extern "C" int ds41_hotpath_activate(const void *gate,const void *up,const float *weights,
  const int *rows,uint8_t *q,uint8_t *s,int count,void *stream) {
  ds41e::hotpath::activate_routes(gate,up,weights,rows,q,s,count,(cudaStream_t)stream);
  return int(cudaGetLastError());
}
extern "C" int ds41_hotpath_frequencies(const float *table,float *main,float *compressed,
    const int64_t *position,int rows,int context,void *stream) {
  ds41e::hotpath::stage_frequencies(table,main,compressed,position,rows,context,(cudaStream_t)stream);
  return int(cudaGetLastError());
}
extern "C" int ds41_hotpath_store_groups(const void *records,void *global,const int64_t *position,
    int rows,int bytes,int ratio,void *stream) {
  ds41e::hotpath::store_groups(records,global,position,rows,bytes,ratio,(cudaStream_t)stream);
  return int(cudaGetLastError());
}
