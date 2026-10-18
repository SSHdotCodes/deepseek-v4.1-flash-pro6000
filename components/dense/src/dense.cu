#include "ds41/dense.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
using bf16=__nv_bfloat16;
__device__ __forceinline__ float e8(uint8_t x){return x==0?__uint_as_float(0x00400000):x==255?__uint_as_float(0x7fffffff):__uint_as_float(uint32_t(x)<<23);}
__global__ void quant(const bf16* x,uint8_t* a,uint8_t* s,int groups){
 int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;if(g>=groups)return;
 float v=__bfloat162float(x[g*32+lane]),vmax=fabsf(v);
 for(int d=16;d;d>>=1)vmax=fmaxf(vmax,__shfl_xor_sync(0xffffffff,vmax,d));
 uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(vmax,1e-4f),1.f/448.f));
 unsigned exponent=(bits>>23)+((bits&0x7fffff)!=0);float scale=__uint_as_float(exponent<<23);
 a[g*32+lane]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,v/scale)),__NV_SATFINITE,__NV_E4M3);
 if(!lane)s[g]=exponent;
}
__device__ __forceinline__ void mma(float (&c)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void mma_scaled(float (&c)[4],const uint32_t (&a)[4],const uint32_t (&b)[2],uint32_t sa,uint32_t sb){
 asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}
__global__ void basic(const uint8_t* x,const uint8_t* xs,const uint8_t* w,const uint8_t* ws,bf16* y,int M,int N,int K){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3,row=blockIdx.x*64+warp*16+g,ng=K/32;
 float acc[4]={0,0,0,0};
 for(int j=0;j<ng;++j){uint32_t a[4]={0,0,0,0},b[2]={0,0};
  for(int hr=0;hr<2;++hr)if(row+hr*8<N){const uint8_t* v=w+(size_t)(row+hr*8)*K+j*32+t*4;a[hr]=*(const uint32_t*)v;a[hr+2]=*(const uint32_t*)(v+16);}
  if(g<M){const uint8_t* v=x+(size_t)g*K+j*32+t*4;b[0]=*(const uint32_t*)v;b[1]=*(const uint32_t*)(v+16);}
  float c[4]={0,0,0,0};mma(c,a,b);
  for(int hr=0;hr<2;++hr)for(int cc=0;cc<2;++cc){int token=t*2+cc;float sx=token<M?e8(xs[token*ng+j]):0.f,sw=row+hr*8<N?e8(ws[((row+hr*8)/32)*ng+j]):0.f;acc[hr*2+cc]=__fadd_rn(acc[hr*2+cc],__fmul_rn(__fmul_rn(c[hr*2+cc],sx),sw));}
 }
 for(int hr=0;hr<2;++hr)for(int cc=0;cc<2;++cc){int token=t*2+cc;if(row+hr*8<N&&token<M)y[token*N+row+hr*8]=__float2bfloat16_rn(acc[hr*2+cc]);}
}
// Every k32 dot is retained individually. The warp carry is serial in ascending K, never split-K sums.
template<int K,int Warps=8,bool Native=false,int NativeCols=8>
__global__ void __launch_bounds__(Warps*32,1) parallel(const uint8_t* x,const uint8_t* xs,const uint8_t* w,const uint8_t* ws,bf16* y,int M,int N){
 constexpr int ng=K/32,G=(ng+Warps-1)/Warps,Cols=Native?NativeCols:16,Count=Native?2:4;
 __shared__ float carry[Warps][32*Count];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3,n0=blockIdx.x*Cols;
 float dots[G][Count];
 #pragma unroll
 for(int j=0;j<G;++j){int group=warp*G+j;
  #pragma unroll
  for(int c=0;c<Count;++c)dots[j][c]=0;
  if(group<ng){uint32_t a[4]={0,0,0,0},b[2]={0,0};
   if constexpr(Native){
    if(g<M){const uint8_t*v=x+(size_t)g*K+group*32+t*4;a[0]=*(const uint32_t*)v;a[2]=*(const uint32_t*)(v+16);}
    if(g<Cols&&n0+g<N){const uint8_t*v=w+(size_t)(n0+g)*K+group*32+t*4;b[0]=*(const uint32_t*)v;b[1]=*(const uint32_t*)(v+16);}
    uint32_t sa=(g<M&&!(t&1))?xs[g*ng+group]:127,sb=g<Cols&&n0+g<N?ws[((n0+g)/32)*ng+group]:127;
    float c[4]={0,0,0,0};mma_scaled(c,a,b,sa,sb);dots[j][0]=c[0];dots[j][1]=c[1];
   }else{
    for(int hr=0;hr<2;++hr)if(n0+g+hr*8<N){const uint8_t*v=w+(size_t)(n0+g+hr*8)*K+group*32+t*4;a[hr]=*(const uint32_t*)v;a[hr+2]=*(const uint32_t*)(v+16);}
    if(g<M){const uint8_t*v=x+(size_t)g*K+group*32+t*4;b[0]=*(const uint32_t*)v;b[1]=*(const uint32_t*)(v+16);}
    float c[4]={0,0,0,0};mma(c,a,b);
    for(int hr=0;hr<2;++hr)for(int cc=0;cc<2;++cc){int token=t*2+cc;float sx=token<M?e8(xs[token*ng+group]):0.f,sw=n0+g+hr*8<N?e8(ws[((n0+g+hr*8)/32)*ng+group]):0.f;dots[j][hr*2+cc]=__fmul_rn(__fmul_rn(c[hr*2+cc],sx),sw);}
   }
  }
 }
 __syncthreads();
 #pragma unroll
 for(int turn=0;turn<Warps;++turn){if(warp==turn){float acc[Count];
  #pragma unroll
  for(int c=0;c<Count;++c)acc[c]=turn?carry[turn-1][lane*Count+c]:0.f;
  #pragma unroll
  for(int j=0;j<G;++j)if(turn*G+j<ng){
   #pragma unroll
   for(int c=0;c<Count;++c)acc[c]=__fadd_rn(acc[c],dots[j][c]);
  }
  #pragma unroll
  for(int c=0;c<Count;++c)carry[turn][lane*Count+c]=acc[c];
  if(turn==Warps-1){if constexpr(Native){if(g<M){if(t*2<Cols&&n0+t*2<N)y[g*N+n0+t*2]=__float2bfloat16_rn(acc[0]);if(t*2+1<Cols&&n0+t*2+1<N)y[g*N+n0+t*2+1]=__float2bfloat16_rn(acc[1]);}}
   else for(int hr=0;hr<2;++hr)for(int cc=0;cc<2;++cc){int token=t*2+cc;if(n0+g+hr*8<N&&token<M)y[token*N+n0+g+hr*8]=__float2bfloat16_rn(acc[hr*2+cc]);}}
 }__syncthreads();}
}
__global__ void dequant(const uint8_t*w,const uint8_t*ws,bf16*y,int N,int K,bool activation){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)N*K)return;
 __nv_fp8_e4m3 f;f.__x=w[i];int row=i/K,col=i%K,si=(activation?row:row/32)*(K/32)+col/32;y[i]=__float2bfloat16_rn(__fmul_rn(float(f),e8(ws[si])));
}
}
extern "C" int ds41_dense_quant(const void*x,void*a,void*s,int M,int K,cudaStream_t stream){if(!x||!a||!s||M<1||K<32||K%32)return cudaErrorInvalidValue;quant<<<(M*K+127)/128,128,0,stream>>>((const bf16*)x,(uint8_t*)a,(uint8_t*)s,M*K/32);return cudaGetLastError();}
extern "C" int ds41_dense_fp8(const void*x,const void*xs,const void*w,const void*ws,void*y,int M,int N,int K,int variant,cudaStream_t stream){
 if(variant==-1){variant=K==8192||K==6144||(N==2304&&K==5120)?4:K==15360||(N==512&&K==5120)?5:K==2304||(N==4096&&K==1280)?3:2;}
 if(!x||!xs||!w||!ws||!y||M<1||M>8||N<32||N%32||K<32||K%32||variant<0||variant>6)return cudaErrorInvalidValue;
 auto a=(const uint8_t*)x;auto as=(const uint8_t*)xs;auto b=(const uint8_t*)w;auto bs=(const uint8_t*)ws;auto c=(bf16*)y;
 if(variant==0){basic<<<(N+63)/64,128,0,stream>>>(a,as,b,bs,c,M,N,K);return cudaGetLastError();}
 #define LAUNCH(KV) case KV: if(variant==2)parallel<KV,8,true><<<(N+7)/8,256,0,stream>>>(a,as,b,bs,c,M,N);else if(variant==3)parallel<KV,4,true><<<(N+7)/8,128,0,stream>>>(a,as,b,bs,c,M,N);else if(variant==4)parallel<KV,4,true,4><<<(N+3)/4,128,0,stream>>>(a,as,b,bs,c,M,N);else if(variant==5)parallel<KV,8,true,4><<<(N+3)/4,256,0,stream>>>(a,as,b,bs,c,M,N);else if(variant==6)parallel<KV,4,true,2><<<(N+1)/2,128,0,stream>>>(a,as,b,bs,c,M,N);else parallel<KV,8,false><<<(N+15)/16,256,0,stream>>>(a,as,b,bs,c,M,N);break
 switch(K){LAUNCH(128);LAUNCH(256);LAUNCH(512);LAUNCH(1280);LAUNCH(2304);LAUNCH(4096);LAUNCH(5120);LAUNCH(6144);LAUNCH(8192);LAUNCH(15360);default:basic<<<(N+63)/64,128,0,stream>>>(a,as,b,bs,c,M,N,K);}
 #undef LAUNCH
 return cudaGetLastError();
}
extern "C" int ds41_dense_dequant(const void*w,const void*s,void*y,int N,int K,cudaStream_t stream){if(!w||!s||!y||N<1||N%32||K<32||K%32)return cudaErrorInvalidValue;dequant<<<((size_t)N*K+255)/256,256,0,stream>>>((const uint8_t*)w,(const uint8_t*)s,(bf16*)y,N,K,false);return cudaGetLastError();}
extern "C" int ds41_dense_roundtrip(const void*w,const void*s,void*y,int M,int K,cudaStream_t stream){if(!w||!s||!y||M<1||K<32||K%32)return cudaErrorInvalidValue;dequant<<<((size_t)M*K+255)/256,256,0,stream>>>((const uint8_t*)w,(const uint8_t*)s,(bf16*)y,M,K,true);return cudaGetLastError();}
