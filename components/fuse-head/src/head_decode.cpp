#include "ds41/fuse_head.h"
extern "C" int ds41_fuse_head_half_tree(const float*,const void*,float*,int,int,int,cudaStream_t);
extern "C" int ds41_fuse_head_half_tune(const float*,const void*,float*,int,int,int,int,cudaStream_t);
extern "C" int ds41_fuse_head_decode(const float*x,const void*w,float*y,
    int M,int N,int K,cudaStream_t st) {
  if(!x||!w||!y||M<1||M>6||N!=129280||K!=5120)return cudaErrorInvalidValue;
  if(M==6)return ds41_fuse_head_half_tree(x,w,y,M,N,K,st);
  constexpr int variant[5]={1,2,3,2,1};
  return ds41_fuse_head_half_tune(x,w,y,M,N,K,variant[M-1],st);
}
