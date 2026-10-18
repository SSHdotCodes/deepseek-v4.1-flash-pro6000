// Research only. Inferred gemvx specialization6 lane/FMA/tree mechanism must
// be reviewed and tested before any E0 claim or production selection.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
namespace {
template<int M,int ROWS,int UNROLL> __global__ void half_tree(const float *x,const __nv_bfloat16 *w,
    float *y,int N,int K) {
  int lane=threadIdx.x,row=blockIdx.x*ROWS+threadIdx.y;
  float acc[M]={};
  #pragma unroll UNROLL
  for(int k=lane;k<5120;k+=16) {
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
template<int ROWS,int UNROLL> int launch(const float*x,const void*w,float*y,int M,int N,cudaStream_t st) {
  #define L(V) case V:half_tree<V,ROWS,UNROLL><<<(N+ROWS-1)/ROWS,dim3(16,ROWS),0,st>>>(x,static_cast<const __nv_bfloat16*>(w),y,N,5120);break
  switch(M){L(1);L(2);L(3);L(4);L(5);L(6);}
  #undef L
  return cudaGetLastError();
}
extern "C" int ds41_fuse_head_half_tune(const float*x,const void*w,float*y,int M,int N,int K,int variant,cudaStream_t st) {
  if(!x||!w||!y||M<1||M>6||N<1||K!=5120||variant<0||variant>11)return cudaErrorInvalidValue;
  #define V(ID,R,U) case ID:return launch<R,U>(x,w,y,M,N,st)
  switch(variant){V(0,4,1);V(1,4,4);V(2,4,8);V(3,4,16);V(4,8,1);V(5,8,4);V(6,8,8);V(7,8,16);V(8,16,1);V(9,16,4);V(10,16,8);V(11,16,16);}
  #undef V
  return cudaErrorInvalidValue;
}
