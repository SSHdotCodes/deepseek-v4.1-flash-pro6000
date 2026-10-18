#include "ds41/attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <mma.h>
#include <math_constants.h>
using bf=__nv_bfloat16;
using namespace nvcuda;
__device__ float ixscale(uint8_t s){return __uint_as_float(unsigned(s)<<23);}
__device__ bf ixval(const uint8_t*p,int d){float v=__half2float(__nv_cvt_fp4_to_halfraw((p[d/2]>>(4*(d&1)))&15,__NV_E2M1));return __float2bfloat16_rn(__fmul_rn(v,ixscale(p[64+d/32])));}
__device__ uint16_t ixbit(unsigned c,unsigned scale){
 unsigned m=c&7,sign=(c&8)<<12;if(!m)return sign;
 int mag=(m==1?0x3f00:0x3f00+(m<<6))+(int(scale)-127)*128;
 if(mag>=0x80&&mag<0x7f80)return sign|mag;
 float v=__half2float(__nv_cvt_fp4_to_halfraw(c,__NV_E2M1));return __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(v,ixscale(scale))));
}
__global__ void indexscore(const bf*q,const uint8_t*key,const bf*w,const int*ids,const int*visible,bf*out,int n){
 __shared__ __align__(32) bf qs[32*128];
 __shared__ __align__(32) bf ks[64*128];
 __shared__ __align__(32) float dot[32*64];
 int row=blockIdx.y,base=blockIdx.x*64,t=threadIdx.x,warp=t/32;
 if(!ids && base>=visible[row]){if(base+t<n && t<64)out[row*n+base+t]=__float2bfloat16(-CUDART_INF_F);return;}
 for(int i=t;i<512;i+=128)((uint4*)qs)[i]=((const uint4*)(q+row*4096))[i];
 for(int v=t;v<1024;v+=128){
  int j=base+v/16,id=j<n?(ids?ids[row*n+j]:j):-1;uint4 result=make_uint4(0,0,0,0);
  if(id>=0&&id<visible[row]){
   const uint8_t* rec=key+int64_t(id)*68;int d=(v%16)*8;
   unsigned packed=*(const unsigned*)(rec+d/2),scale=rec[64+d/32];unsigned pair[4];
   #pragma unroll
   for(int k=0;k<4;++k){unsigned a=ixbit((packed>>(k*8))&15,scale),b=ixbit((packed>>(k*8+4))&15,scale);pair[k]=a|(b<<16);}
   result=make_uint4(pair[0],pair[1],pair[2],pair[3]);
  }
  ((uint4*)ks)[v]=result;
 }
 __syncthreads();
 #pragma unroll
 for(int h=0;h<32;h+=16){
  wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0.f);
  #pragma unroll
  for(int d=0;d<128;d+=16){
   wmma::fragment<wmma::matrix_a,16,16,16,bf,wmma::row_major> a;
   wmma::fragment<wmma::matrix_b,16,16,16,bf,wmma::col_major> b;
   wmma::load_matrix_sync(a,qs+h*128+d,128);wmma::load_matrix_sync(b,ks+warp*16*128+d,128);
   wmma::mma_sync(c,a,b,c);
  }
  wmma::store_matrix_sync(dot+h*64+warp*16,c,64,wmma::mem_row_major);
 }
 __syncthreads();
 // Keep the BF16 dot, product, and final head-sum boundaries distinct.
 // PyTorch Reduce.cuh vt0=4: four independent ascending h%4 sums, then
 // ((partial0+partial1)+partial2)+partial3. N=1 is contiguous warp reduction.
 for(int j=t;j<64;j+=128){
  int col=base+j;if(col>=n)continue;int id=ids?ids[row*n+col]:col;
  float acc[32];
  for(int h=0;h<32;++h){bf v=__float2bfloat16_rn(dot[h*64+j]);float a=fmaxf(__bfloat162float(v),0.f);bf b=__float2bfloat16_rn(__fmul_rn(a,__bfloat162float(w[row*32+h])));acc[h]=__bfloat162float(b);}
  float sum;
  if(n==1){for(int stride=16;stride;stride>>=1)for(int h=0;h<stride;++h)acc[h]=__fadd_rn(acc[h],acc[h+stride]);sum=acc[0];}
  else {float partial[4]={0,0,0,0};for(int h=0;h<32;++h)partial[h%4]=__fadd_rn(partial[h%4],acc[h]);sum=__fadd_rn(__fadd_rn(__fadd_rn(partial[0],partial[1]),partial[2]),partial[3]);}
  out[row*n+col]=__float2bfloat16_rn(id>=0&&id<visible[row]?sum:-CUDART_INF_F);
 }
}
extern "C" int ds41_attn_index_scores(const void*q,const void*k,const void*w,const int*ids,const int*visible,void*out,int rows,int n,void*st){
 if(!q||!k||!w||!visible||!out||rows<1||n<1)return cudaErrorInvalidValue;
 indexscore<<<dim3((n+63)/64,rows),128,0,(cudaStream_t)st>>>((const bf*)q,(const uint8_t*)k,(const bf*)w,ids,visible,(bf*)out,n);return cudaGetLastError();
}
__global__ void blocks(const bf*s,const int*vis,bf*out,int n){
 int b=blockIdx.x*256+threadIdx.x,row=blockIdx.y,nb=(n+7)/8;if(b>=nb)return;float v=-CUDART_INF_F;
 for(int i=0;i<8;++i)if(b*8+i<n)v=fmaxf(v,__bfloat162float(s[row*n+b*8+i]));
 if(b==(vis[row]-1)/8&&vis[row]>0)v=CUDART_INF_F;out[row*nb+b]=__float2bfloat16_rn(v);
}
extern "C" int ds41_attn_block_scores(const void*s,const int*vis,void*out,int rows,int n,void*st){
 if(!s||!vis||!out||rows<1||n<1)return cudaErrorInvalidValue;blocks<<<dim3(((n+7)/8+255)/256,rows),256,0,(cudaStream_t)st>>>((const bf*)s,vis,(bf*)out,n);return cudaGetLastError();
}
__global__ void expand(const int*block,int*out,int nb,int n){int i=blockIdx.x*256+threadIdx.x,row=blockIdx.y;if(i>=nb*8)return;int b=block[row*nb+i/8];int id=b<0?-1:b*8+i%8;out[row*nb*8+i]=id<n?id:-1;}
extern "C" int ds41_attn_expand_blocks(const int*blocks,int*out,int rows,int nb,int n,void*st){if(!blocks||!out||rows<1||nb<1||n<1)return cudaErrorInvalidValue;expand<<<dim3((nb*8+255)/256,rows),256,0,(cudaStream_t)st>>>(blocks,out,nb,n);return cudaGetLastError();}
__global__ void remap(const int*local,const int*ids,int*out,int k,int n){int i=blockIdx.x*256+threadIdx.x,row=blockIdx.y;if(i>=k)return;int j=local[row*k+i];out[row*k+i]=j>=0&&j<n?ids[row*n+j]:-1;}
extern "C" int ds41_attn_remap_topk(const int*local,const int*ids,int*out,int rows,int k,int n,void*st){if(!local||!ids||!out||rows<1||k<1||n<1)return cudaErrorInvalidValue;remap<<<dim3((k+255)/256,rows),256,0,(cudaStream_t)st>>>(local,ids,out,k,n);return cudaGetLastError();}
