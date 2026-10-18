// Research only. Inferred gemvx specialization6 lane/FMA/tree mechanism must
// be reviewed and tested before any E0 claim or production selection.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
namespace {
template<int M> __global__ void half_tree(const float *x,const __nv_bfloat16 *w,
    float *y,int N,int K) {
  int lane=threadIdx.x,row=blockIdx.x*4+threadIdx.y;
  float acc[M]={};
  for(int k=lane;k<K;k+=16) {
    float weight=row<N?__bfloat162float(w[size_t(row)*K+k]):0.f;
    #pragma unroll
    for(int m=0;m<M;++m) acc[m]=__fmaf_rn(weight,x[size_t(m)*K+k],acc[m]);
  }
  #pragma unroll
  for(int m=0;m<M;++m) {
    for(int d=8;d;d>>=1)acc[m]=__fadd_rn(acc[m],__shfl_down_sync(0xffffffff,acc[m],d,16));
    if(!lane&&row<N)y[size_t(m)*N+row]=__fadd_rn(acc[m],0.f);
  }
}
}
extern "C" int ds41_fuse_head_half_tree(const float *x,const void *w,float *y,
    int M,int N,int K,cudaStream_t st) {
  if(!x||!w||!y||M<1||M>6||N<1||K<1)return cudaErrorInvalidValue;
  #define L(V) case V:half_tree<V><<<(N+3)/4,dim3(16,4),0,st>>>(x,static_cast<const __nv_bfloat16*>(w),y,N,K);break
  switch(M){L(1);L(2);L(3);L(4);L(5);L(6);}
  #undef L
  return cudaGetLastError();
}
