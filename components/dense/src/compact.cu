#include "ds41/dense.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
namespace {
using bf16=__nv_bfloat16;
__device__ __forceinline__ float e8(uint8_t x){return x==0?__uint_as_float(0x00400000):x==255?__uint_as_float(0x7fffffff):__uint_as_float(uint32_t(x)<<23);}
// Vectorized SIMT GEMV: each BF16 weight is fetched once and shared across M rows.
// All inputs remain FP32. Independent FP32 FMA chains followed by a fixed warp tree.
template<int M,int V=8,int W=4>
__global__ void head(const float* __restrict__ x,const bf16* __restrict__ w,float* __restrict__ y,int N,int K){
 int lane=threadIdx.x&31,row=blockIdx.x*W+(threadIdx.x>>5);if(row>=N)return;
 float acc[M][4]={};
 for(int k=lane*V;k<K;k+=32*V){
  uint4 v=*(const uint4*)(w+(size_t)row*K+k);bf16 ww[V];*(uint4*)ww=v;
  #pragma unroll
  for(int m=0;m<M;++m){
   float4 a=*(const float4*)(x+m*K+k),b=*(const float4*)(x+m*K+k+4);float xx[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
   #pragma unroll
   for(int j=0;j<V;++j)acc[m][j&3]=__fmaf_rn(__bfloat162float(ww[j]),xx[j],acc[m][j&3]);
  }
 }
 #pragma unroll
 for(int m=0;m<M;++m){float sum=__fadd_rn(__fadd_rn(acc[m][0],acc[m][1]),__fadd_rn(acc[m][2],acc[m][3]));for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(!lane)y[m*N+row]=sum;}
}
template<int M,int W=4>
__global__ void woa(const bf16* __restrict__ x,const uint8_t* __restrict__ w,const uint8_t* __restrict__ s,bf16* __restrict__ y,int N,int K,int G){
 int lane=threadIdx.x&31,row=blockIdx.x*W+(threadIdx.x>>5),group=blockIdx.y;if(row>=N)return;
 float acc[M][4]={};
 for(int k=lane*8;k<K;k+=256){
  uint2 packed=*(const uint2*)(w+((size_t)group*N+row)*K+k);uint8_t vals[8];*(uint2*)vals=packed;
  float scale=e8(s[((size_t)group*N+row)/32*(K/32)+k/32]);float weights[8];
  #pragma unroll
  for(int j=0;j<8;++j){__nv_fp8_e4m3 q;q.__x=vals[j];weights[j]=__bfloat162float(__float2bfloat16_rn(__fmul_rn(float(q),scale)));}
  #pragma unroll
  for(int m=0;m<M;++m){uint4 vv=*(const uint4*)(x+((size_t)m*G+group)*K+k);bf16 xx[8];*(uint4*)xx=vv;
   #pragma unroll
   for(int j=0;j<8;++j)acc[m][j&3]=__fmaf_rn(weights[j],__bfloat162float(xx[j]),acc[m][j&3]);
  }
 }
 #pragma unroll
 for(int m=0;m<M;++m){float sum=__fadd_rn(__fadd_rn(acc[m][0],acc[m][1]),__fadd_rn(acc[m][2],acc[m][3]));for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(!lane)y[((size_t)m*G+group)*N+row]=__float2bfloat16_rn(sum);}
}
}
extern "C" int ds41_dense_head_compact(const float*x,const void*w,float*y,int M,int N,int K,cudaStream_t st){
 if(!x||!w||!y||M<1||M>8||N<1||K<256||K%256)return cudaErrorInvalidValue;
 #define L(MM) case MM:head<MM><<<(N+3)/4,128,0,st>>>(x,(const bf16*)w,y,N,K);break
 switch(M){L(1);L(2);L(3);L(4);L(5);L(6);L(7);L(8);}
 #undef L
 return cudaGetLastError();
}
extern "C" int ds41_dense_wo_a_compact(const void*x,const void*w,const void*s,void*y,int M,int N,int K,int G,cudaStream_t st){
 if(!x||!w||!s||!y||M<1||M>8||N<32||N%32||K<256||K%256||G<1)return cudaErrorInvalidValue;
 #define L(MM) case MM:woa<MM><<<dim3((N+3)/4,G),128,0,st>>>((const bf16*)x,(const uint8_t*)w,(const uint8_t*)s,(bf16*)y,N,K,G);break
 switch(M){L(1);L(2);L(3);L(4);L(5);L(6);L(7);L(8);}
 #undef L
 return cudaGetLastError();
}
#include <cstdlib>
#include <cstring>
extern "C" int ds41_dense_policy_from_env(void){
 const char* ref=std::getenv("DS41_DENSE_REFERENCE");if(ref&&std::strcmp(ref,"0")!=0)return 0;
 const char* compact=std::getenv("DS41_DENSE_COMPACT");return compact&&std::strcmp(compact,"1")==0?1:0;
}
extern "C" int ds41_dense_head_dispatch(void*handle,const float*x,const void*bf,const float*f32,float*y,int M,int N,int K,int policy,cudaStream_t st){
 if(policy==0)return ds41_dense_cublas(handle,x,f32,y,M,N,K,1,1,st);
 if(policy==1)return ds41_dense_head_compact(x,bf,y,M,N,K,st);
 return cudaErrorInvalidValue;
}
extern "C" int ds41_dense_wo_a_dispatch(void*handle,const void*x,const void*fp8,const void*s,const void*bf,void*y,int M,int N,int K,int G,int policy,cudaStream_t st){
 if(policy==0)return ds41_dense_wo_a(handle,x,bf,y,M,N,K,G,st);
 if(policy==1)return ds41_dense_wo_a_compact(x,fp8,s,y,M,N,K,G,st);
 return cudaErrorInvalidValue;
}
