#include "ds41/attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <math_constants.h>
using bf=__nv_bfloat16;
__device__ float e8(uint8_t c) { return __uint_as_float(unsigned(c)<<23); }
__device__ float f8(uint8_t c) { return __half2float(__nv_cvt_fp8_to_halfraw(c,__NV_E4M3)); }
__device__ float f4(uint8_t c) { return __half2float(__nv_cvt_fp4_to_halfraw(c,__NV_E2M1)); }
__device__ float powceil(float v) {
 unsigned u=__float_as_uint(v); unsigned ex=(u>>23)&255;
 return __uint_as_float((ex+((u&0x7fffff)!=0))<<23);
}
template<int F, bool RT> __global__ void packer(const bf* x, uint8_t* out, bf* rt, int rows,int cols) {
 constexpr int G=F==1?16:32;
 const int group=blockIdx.x*8+threadIdx.x/32, lane=threadIdx.x%32;
 const int groups=rows*(cols/G); if(group>=groups)return;
 const int row=group/(cols/G), gg=group%(cols/G);
 float a= lane<G?__bfloat162float(x[row*cols+gg*G+lane]):0.f;
 float am=fabsf(a);
 for(int d=G/2;d;d>>=1) am=fmaxf(am,__shfl_xor_sync(0xffffffff,am,d));
 float s;
 uint8_t sc;
 if constexpr(F==0) {s=powceil(__fmul_rn(fmaxf(am,1e-4f),1.f/448.f));sc=__float_as_uint(s)>>23;}
 else if constexpr(F==1) {s=__fmul_rn(fmaxf(am,6.f*0x1p-9f),1.f/6.f);sc=__nv_cvt_float_to_fp8(s,__NV_SATFINITE,__NV_E4M3);s=f8(sc);}
 else {s=powceil(__fmul_rn(fmaxf(am,6.f*0x1p-126f),1.f/6.f));sc=__float_as_uint(s)>>23;}
 if(lane>=G)return;
 float z=__fdiv_rn(a,s);
 if constexpr(F==0){
  uint8_t v=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,z)),__NV_SATFINITE,__NV_E4M3);
  if constexpr(RT)rt[row*cols+gg*G+lane]=__float2bfloat16_rn(__fmul_rn(f8(v),s));
  else{out[row*(cols+cols/G)+gg*G+lane]=v;if(lane==0)out[row*(cols+cols/G)+cols+gg]=sc;}
 }else{
  uint8_t v=__nv_cvt_float_to_fp4(fminf(6.f,fmaxf(-6.f,z)),__NV_E2M1,cudaRoundNearest);
  if constexpr(RT)rt[row*cols+gg*G+lane]=__float2bfloat16_rn(__fmul_rn(f4(v),s));
  else{unsigned other=__shfl_xor_sync(G==16?0x0000ffff:0xffffffff,unsigned(v),1);if(!(lane&1))out[row*(cols/2+cols/G)+gg*G/2+lane/2]=v|(other<<4);if(lane==0)out[row*(cols/2+cols/G)+cols/2+gg]=sc;}
 }
}
template<int F> __global__ void unpacker(const uint8_t* p,bf* x,int count,int cols){
 int i=blockIdx.x*256+threadIdx.x;if(i>=count)return;int row=i/cols,d=i%cols;
 if constexpr(F==0){p+=row*(cols+cols/32);x[i]=__float2bfloat16_rn(__fmul_rn(f8(p[d]),e8(p[cols+d/32])));}
 else {constexpr int G=F==1?16:32;p+=row*(cols/2+cols/G);uint8_t b=p[d/2];float sc=F==1?f8(p[cols/2+d/G]):e8(p[cols/2+d/G]);x[i]=__float2bfloat16_rn(__fmul_rn(f4((b>>(4*(d&1)))&15),sc));}
}
extern "C" int ds41_attn_pack(const void*x,void*out,int rows,int f,void*st){
 if(rows<1||f<0||f>2||!x||!out)return cudaErrorInvalidValue;int cols=f==2?128:512;cudaStream_t s=(cudaStream_t)st;
 if(f==0)packer<0,false><<<(rows*(cols/32)+7)/8,256,0,s>>>((const bf*)x,(uint8_t*)out,nullptr,rows,cols);
 if(f==1)packer<1,false><<<(rows*(cols/16)+7)/8,256,0,s>>>((const bf*)x,(uint8_t*)out,nullptr,rows,cols);
 if(f==2)packer<2,false><<<(rows*(cols/32)+7)/8,256,0,s>>>((const bf*)x,(uint8_t*)out,nullptr,rows,cols);
 return cudaGetLastError();
}
extern "C" int ds41_attn_unpack(const void*x,void*out,int rows,int f,void*st){
 if(rows<1||f<0||f>2||!x||!out)return cudaErrorInvalidValue;int cols=f==2?128:512;cudaStream_t s=(cudaStream_t)st;
 if(f==0)unpacker<0><<<(rows*cols+255)/256,256,0,s>>>((const uint8_t*)x,(bf*)out,rows*cols,cols);
 if(f==1)unpacker<1><<<(rows*cols+255)/256,256,0,s>>>((const uint8_t*)x,(bf*)out,rows*cols,cols);
 if(f==2)unpacker<2><<<(rows*cols+255)/256,256,0,s>>>((const uint8_t*)x,(bf*)out,rows*cols,cols);
 return cudaGetLastError();
}
extern "C" int ds41_attn_roundtrip(void*x,int rows,int cols,int f,void*st){
 if(rows<1||cols<32||cols%32||f<0||f>2||!x)return cudaErrorInvalidValue;cudaStream_t s=(cudaStream_t)st;
 if(f==0)packer<0,true><<<(rows*(cols/32)+7)/8,256,0,s>>>((bf*)x,nullptr,(bf*)x,rows,cols);
 if(f==1)packer<1,true><<<(rows*(cols/16)+7)/8,256,0,s>>>((bf*)x,nullptr,(bf*)x,rows,cols);
 if(f==2)packer<2,true><<<(rows*(cols/32)+7)/8,256,0,s>>>((bf*)x,nullptr,(bf*)x,rows,cols);
 return cudaGetLastError();
}
