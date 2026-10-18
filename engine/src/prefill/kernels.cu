#include "ds41/prefill/kernels.cuh"
#include "ds41/dense.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
namespace ds41e::prefill {
namespace {
constexpr int H=5120;
__global__ void select(const float* biased,const float* unbiased,int* ids,float* weights){
  if(threadIdx.x)return;
  for(int k=0;k<6;++k){int best=-1;
    for(int e=0;e<384;++e){bool seen=false;for(int j=0;j<k;++j)seen|=ids[j]==e;
      if(!seen&&(best<0||biased[e]>biased[best]||(biased[e]==biased[best]&&e<best)))best=e;}
    ids[k]=best;
  }
  // Accepted sequential M1 route reduction, including route-rank order.
  float v[4]={0,0,0,0};for(int k=0;k<6;++k)v[k%4]=__fadd_rn(v[k%4],unbiased[ids[k]]);
  float sum=__fadd_rn(__fadd_rn(v[0],v[2]),__fadd_rn(v[1],v[3]));
  sum=__fadd_rn(sum,1e-20f);
  for(int k=0;k<6;++k)weights[k]=__fmul_rn(unbiased[ids[k]]/sum,1.5f);
}
__global__ void gather(const unsigned char*x,const unsigned char*s,const float*w,
    const int*slots,unsigned char*out,unsigned char*os,float*ow){
  int r=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x,slot=slots[r],src=slot/6;
  if(i<H)out[size_t(r)*H+i]=x[size_t(src)*H+i];
  if(i<H/32)os[size_t(r)*(H/32)+i]=s[size_t(src)*(H/32)+i];
  if(!i)ow[r]=w[slot];
}
__global__ void scatter(const __nv_bfloat16*x,const int*slots,__nv_bfloat16*out){
  int r=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<H)out[size_t(slots[r])*H+i]=x[size_t(r)*H+i];
}
}
void select_routes(const float*b,const float*u,int*i,float*w,cudaStream_t s){select<<<1,32,0,s>>>(b,u,i,w);}
void gather_routes(const unsigned char*x,const unsigned char*s,const float*w,const int*slots,
    unsigned char*out,unsigned char*os,float*ow,int rows,cudaStream_t st){
  gather<<<dim3(20,rows),256,0,st>>>(x,s,w,slots,out,os,ow);
}
void scatter_routes(const void*x,const int*slots,void*out,int rows,cudaStream_t s){
  scatter<<<dim3(20,rows),256,0,s>>>((const __nv_bfloat16*)x,slots,(__nv_bfloat16*)out);
}
}
// ---- Batched prefill (--prefill-chunk) row kernels. The RMS, mHC promote/stats and Sinkhorn bodies are
// copied from libds41_dense (src/ops.cu) without its M<=8 validation; launches keep the M=1 reduction
// geometry, so every row reduces exactly as the per-row path does.
namespace ds41e::prefill {
namespace rows_detail {
using bf16=__nv_bfloat16;
__device__ __forceinline__ float blocksum(float sum,float* sh,int width){int tid=threadIdx.x;sh[tid]=sum;__syncthreads();
 for(int d=width/2;d>=32;d>>=1){if(tid%width<d){sum=__fadd_rn(sum,sh[tid+d]);sh[tid]=sum;}__syncthreads();}
 for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(tid%width==0)sh[tid]=sum;__syncthreads();
 for(int d=(blockDim.x/width)/2;d;d>>=1){if(tid%(width)==0&&tid/width<d){sum=__fadd_rn(sum,sh[tid+d*width]);sh[tid]=sum;}__syncthreads();}
 return sh[0];
}
int geometry(int M,int K,int& width){int height=1;while(height*2<=M&&height*2<=16)height*=2;int n=1;while(n*2<=K/4&&n*2<=512)n*=2;width=n<512/height?n:512/height;return K/width>=height*16?width*height:width;}
__global__ void rms(const bf16*x,const bf16*w,bf16*y,int M,int K,float eps,int width){
 __shared__ float sums[512];int tid=threadIdx.x,row=blockIdx.x;if(row>=M)return;
 const bf16*xr=x+size_t(row)*K;bf16*yr=y+size_t(row)*K;
 float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4){
  #pragma unroll
  for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(xr[i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 }
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sums,width);
 if(!tid)sums[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int i=tid;i<K;i+=blockDim.x)yr[i]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(xr[i]),sums[0]),__bfloat162float(w[i])));
}
__device__ float sum4(float v){v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,2));return __fadd_rn(v,__shfl_xor_sync(0xffffffff,v,1));}
__device__ float colsum4(float v){v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,8));return __fadd_rn(v,__shfl_xor_sync(0xffffffff,v,4));}
__global__ void sinkhorn(const float*m,const float*s,const float*b,float*pre,float*post,float*comb,int M,float eps,const float* rstd){
 int row=blockIdx.x,lane=threadIdx.x;if(row>=M)return;
 float z=lane<24?m[row*24+lane]:0.f;if(rstd)z=__fmul_rn(z,rstd[row]);
 if(lane<4){pre[row*4+lane]=__fadd_rn(1.f/(1.f+expf(-__fmaf_rn(z,s[0],b[lane]))),eps);float p=m[row*24+4+lane];if(rstd)p=__fmul_rn(p,rstd[row]);post[row*4+lane]=__fmul_rn(2.f,1.f/(1.f+expf(-__fmaf_rn(p,s[1],b[4+lane]))));}
 float q=lane<16?m[row*24+8+lane]:0.f;if(rstd)q=__fmul_rn(q,rstd[row]);float v=__fmaf_rn(q,s[2],b[8+(lane&15)]);
 float mx=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
 v=expf(v-mx);v=__fadd_rn(v/sum4(v),eps);v=v/__fadd_rn(colsum4(v),eps);
 for(int it=1;it<20;++it){v=v/__fadd_rn(sum4(v),eps);v=v/__fadd_rn(colsum4(v),eps);}
 if(lane<16)comb[row*16+lane]=v;
}
__global__ void promote_stats(const bf16*x,float*y,float*rstd,int M,int K,float eps,int width){__shared__ float sh[512];int row=blockIdx.x,tid=threadIdx.x;float v[4]={0,0,0,0};
 const bf16*xr=x+size_t(row)*K;float*yr=y+size_t(row)*K;
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float a=__bfloat162float(xr[i+j]);yr[i+j]=a;v[j]=__fadd_rn(v[j],__fmul_rn(a,a));}
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sh,width);if(!tid)rstd[row]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));
}
__global__ void select_rows(const float* biased,const float* unbiased,int* ids,float* weights,int rows){
  int r=blockIdx.x;if(r>=rows||threadIdx.x)return;
  biased+=size_t(r)*384;unbiased+=size_t(r)*384;ids+=r*6;weights+=r*6;
  for(int k=0;k<6;++k){int best=-1;
    for(int e=0;e<384;++e){bool seen=false;for(int j=0;j<k;++j)seen|=ids[j]==e;
      if(!seen&&(best<0||biased[e]>biased[best]||(biased[e]==biased[best]&&e<best)))best=e;}
    ids[k]=best;
  }
  float v[4]={0,0,0,0};for(int k=0;k<6;++k)v[k%4]=__fadd_rn(v[k%4],unbiased[ids[k]]);
  float sum=__fadd_rn(__fadd_rn(v[0],v[2]),__fadd_rn(v[1],v[3]));
  sum=__fadd_rn(sum,1e-20f);
  for(int k=0;k<6;++k)weights[k]=__fmul_rn(unbiased[ids[k]]/sum,1.5f);
}
}
int rms_rows(const void* x,const void* w,void* y,int M,int K,float eps,cudaStream_t s){
  if(!x||!w||!y||M<1||K<128||K%4)return cudaErrorInvalidValue;
  int width,threads=rows_detail::geometry(1,K,width);
  rows_detail::rms<<<M,threads,0,s>>>((const rows_detail::bf16*)x,(const rows_detail::bf16*)w,(rows_detail::bf16*)y,M,K,eps,width);
  return cudaGetLastError();
}
int hc_mixes_rows(void* blas,const void* h,const float* fn,const float* scale,const float* base,float* pre,float* post,
    float* comb,float* promoted,float* mixes,float* rstd,int M,int K,float neps,float heps,cudaStream_t s){
  if(!blas||!h||M<1||K<32||K%32)return cudaErrorInvalidValue;
  int width,threads=rows_detail::geometry(1,K*4,width);
  rows_detail::promote_stats<<<M,threads,0,s>>>((const rows_detail::bf16*)h,promoted,rstd,M,K*4,neps,width);
  int rc=cudaGetLastError();if(rc)return rc;
  rc=ds41_dense_cublas(blas,promoted,fn,mixes,M,24,K*4,1,1,s);if(rc)return rc;
  rows_detail::sinkhorn<<<M,32,0,s>>>(mixes,scale,base,pre,post,comb,M,heps,rstd);
  return cudaGetLastError();
}
void select_routes_rows(const float* biased,const float* unbiased,int* ids,float* weights,int rows,cudaStream_t s){
  rows_detail::select_rows<<<rows,32,0,s>>>(biased,unbiased,ids,weights,rows);
}
// Token-major grouped BF16 einsum x[M,G,K] . w[G,N,K]^T -> y[M,G,N] (ds41_dense_wo_a without its M<=8 cap).
int wo_a_rows(void* blas,const void* x,const void* w,void* y,int M,int N,int K,int G,cudaStream_t s){
  auto h=(cublasHandle_t)blas;auto status=cublasSetStream(h,s);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
  status=cublasSetMathMode(h,CUBLAS_DEFAULT_MATH);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
  float alpha=1.f,beta=0.f;
  status=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,CUDA_R_16BF,K,(long long)N*K,x,CUDA_R_16BF,G*K,K,
      &beta,y,CUDA_R_16BF,G*N,N,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
  return status==CUBLAS_STATUS_SUCCESS?0:10000+status;
}
}
