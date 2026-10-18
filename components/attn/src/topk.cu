#include "ds41/attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
using bf=__nv_bfloat16;
__device__ unsigned ordered(bf v){unsigned u=__bfloat16_as_ushort(v);if((u&0x7fff)==0)u=0;if((u&0x7fff)>0x7f80)return 65535;return u&0x8000?(~u)&65535:u^0x8000;}
struct Work{int *hi,*lo,*meta,*counts,*offsets;};
Work work(void*s,int rows,int n){int*t=(int*)s;Work w;w.hi=t;t+=rows*256;w.lo=t;t+=rows*256;w.meta=t;t+=rows*4;w.counts=t;t+=rows*((n+255)/256)*2;w.offsets=t;return w;}
extern "C" size_t ds41_attn_topk_scratch(int rows,int n){if(rows<1||n<1)return 0;return size_t(rows)*(512+4+4*((n+255)/256))*4;}
template<bool LOW> __global__ void hist(const bf*s,int n,Work w){
 __shared__ int bins[256];int t=threadIdx.x,row=blockIdx.y;bins[t]=0;__syncthreads();
 for(int i=blockIdx.x*256+t;i<n;i+=gridDim.x*256){unsigned u=ordered(s[row*n+i]);if(!LOW||int(u>>8)==w.meta[row*4])atomicAdd(bins+(LOW?(u&255):(u>>8)),1);}
 __syncthreads();if(bins[t])atomicAdd((LOW?w.lo:w.hi)+row*256+t,bins[t]);
}
template<bool LOW> __global__ void threshold(Work w,int k){
 int row=blockIdx.x;if(threadIdx.x)return;int*hist=(LOW?w.lo:w.hi)+row*256;int rem=LOW?w.meta[row*4+1]:k,above=0,b=255;
 for(;b>0;--b){if(above+hist[b]>=rem)break;above+=hist[b];}
 if(LOW){w.meta[row*4+2]=(w.meta[row*4]<<8)|b;w.meta[row*4+3]=k-w.meta[row*4+1]+above;}
 else{w.meta[row*4]=b;w.meta[row*4+1]=k-above;}
}
__global__ void tilecount(const bf*s,int n,Work w){
 using R=cub::BlockReduce<unsigned long long,256>;__shared__ typename R::TempStorage temp;
 int row=blockIdx.y,tile=blockIdx.x,i=tile*256+threadIdx.x,nb=(n+255)/256;unsigned u=i<n?ordered(s[row*n+i]):0;unsigned thr=w.meta[row*4+2];
 unsigned long long z=i<n?((u>thr?1ull:0ull)<<32)|(u==thr?1ull:0ull):0ull;
 z=R(temp).Sum(z);if(!threadIdx.x){w.counts[(row*nb+tile)*2]=z>>32;w.counts[(row*nb+tile)*2+1]=unsigned(z);}
}
__global__ void offsets(Work w,int n,int k){
 using S=cub::BlockScan<unsigned long long,256>;__shared__ typename S::TempStorage temp;
 int row=blockIdx.x,nb=(n+255)/256,budget=k-w.meta[row*4+3];unsigned long long carry=0;
 for(int base=0;base<nb;base+=256){
  int t=base+threadIdx.x,at=(row*nb+t)*2;
  unsigned long long v=t<nb?((static_cast<unsigned long long>(w.counts[at])<<32)|unsigned(w.counts[at+1])):0ull;
  unsigned long long prefix,total;S(temp).ExclusiveSum(v,prefix,total);prefix+=carry;
  if(t<nb){int above=prefix>>32,ties=unsigned(prefix);w.offsets[at]=above+min(ties,budget);w.offsets[at+1]=ties;}
  carry+=total;__syncthreads();
 }
}
__global__ void emit(const bf*s,const int*visible,int*out,int n,int k,Work w){
 using S=cub::BlockScan<int,256>;__shared__ typename S::TempStorage temp;
 int row=blockIdx.y,tile=blockIdx.x,i=tile*256+threadIdx.x,nb=(n+255)/256;unsigned u=i<n?ordered(s[row*n+i]):0,thr=w.meta[row*4+2];
 int eq=i<n&&u==thr,pref;S(temp).ExclusiveSum(eq,pref);__syncthreads();
 int at=(row*nb+tile)*2;bool take=i<n&&(u>thr||(eq&&(pref+w.offsets[at+1]<k-w.meta[row*4+3])));
 int rank;S(temp).ExclusiveSum(int(take),rank);
 if(take)out[row*k+w.offsets[at]+rank]=i<visible[row]?i:-1;
}
extern "C" int ds41_attn_topk(const void*s,const int*vis,int*out,int rows,int n,int k,void*scratch,void*st){
 if(!s||!vis||!out||!scratch||rows<1||n<1||k<1||k>n||k>2048)return cudaErrorInvalidValue;cudaStream_t stream=(cudaStream_t)st;Work w=work(scratch,rows,n);
 cudaError_t e=cudaMemsetAsync(w.hi,0,size_t(rows)*512*4,stream);if(e)return e;
 dim3 grid(min((n+255)/256,128),rows);hist<false><<<grid,256,0,stream>>>((const bf*)s,n,w);threshold<false><<<rows,32,0,stream>>>(w,k);
 hist<true><<<grid,256,0,stream>>>((const bf*)s,n,w);threshold<true><<<rows,32,0,stream>>>(w,k);
 dim3 tiles((n+255)/256,rows);tilecount<<<tiles,256,0,stream>>>((const bf*)s,n,w);offsets<<<rows,256,0,stream>>>(w,n,k);emit<<<tiles,256,0,stream>>>((const bf*)s,vis,out,n,k,w);return cudaGetLastError();
}
