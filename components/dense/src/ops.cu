#include "ds41/dense.h"
#include <cuda_bf16.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
namespace {
using bf16=__nv_bfloat16;
// Mirror PyTorch Reduce.cuh's contiguous FP32 mean: four independent accumulators,
// sequential combine, block halving, then descending warp shuffles. M controls its launch geometry.
__device__ __forceinline__ float blocksum(float sum,float* sh,int width){int tid=threadIdx.x;sh[tid]=sum;__syncthreads();
 for(int d=width/2;d>=32;d>>=1){if(tid%width<d){sum=__fadd_rn(sum,sh[tid+d]);sh[tid]=sum;}__syncthreads();}
 for(int d=16;d;d>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,d));if(tid%width==0)sh[tid]=sum;__syncthreads();
 for(int d=(blockDim.x/width)/2;d;d>>=1){if(tid%(width)==0&&tid/width<d){sum=__fadd_rn(sum,sh[tid+d*width]);sh[tid]=sum;}__syncthreads();}
 return sh[0];
}
int geometry(int M,int K,int& width){int height=1;while(height*2<=M&&height*2<=16)height*=2;int n=1;while(n*2<=K/4&&n*2<=512)n*=2;width=n<512/height?n:512/height;return K/width>=height*16?width*height:width;}
__global__ void rms(const bf16*x,const bf16*w,bf16*y,int M,int K,float eps,int width){
 __shared__ float sums[512];int tid=threadIdx.x,row=blockIdx.x;if(row>=M)return;
 float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4){
  #pragma unroll
  for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(x[row*K+i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 }
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sums,width);
 if(!tid)sums[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int i=tid;i<K;i+=blockDim.x)y[row*K+i]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+i]),sums[0]),__bfloat162float(w[i])));
}
// Fused RMS -> explicit BF16 rounding -> optional FP8 quant or RoPE. The reduction
// geometry and every arithmetic boundary remain identical to the standalone ops.
__global__ void rmsrope(const bf16*x,const bf16*w,bf16*y,const float*f,int M,int K,int RD,float eps,int width,int inverse){
 __shared__ float sums[512];int tid=threadIdx.x,row=blockIdx.x;float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4){
  #pragma unroll
  for(int j=0;j<4;++j)if(i+j<K){float z=__bfloat162float(x[row*K+i+j]);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 }
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sums,width);
 if(!tid)sums[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int k=tid*2;k<K;k+=blockDim.x*2){
  bf16 aa=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+k]),sums[0]),__bfloat162float(w[k])));
  bf16 bb=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(x[row*K+k+1]),sums[0]),__bfloat162float(w[k+1])));
  if(k>=K-RD){int pair=k-(K-RD);float a=__bfloat162float(aa),b=__bfloat162float(bb),c=f[row*RD+pair],si=f[row*RD+pair+1];if(inverse)si=-si;aa=__float2bfloat16_rn(__fmaf_rn(a,c,-__fmul_rn(b,si)));bb=__float2bfloat16_rn(__fmaf_rn(b,c,__fmul_rn(a,si)));}
  y[row*K+k]=aa;y[row*K+k+1]=bb;
 }
}
__global__ void hcpre(const bf16*h,const float*p,bf16*y,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*K)return;int row=i/K,k=i%K;float v[4];
 for(int j=0;j<4;++j)v[j]=__fmul_rn(p[row*4+j],__bfloat162float(h[(row*4+j)*K+k]));
 y[i]=__float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3]));
}
__global__ void hcpost(const bf16*y,const bf16*h,const float*p,const float*c,bf16*out,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*4*K)return;int row=i/(4*K),j=(i/K)%4,k=i%K;float v[4];
 for(int a=0;a<4;++a)v[a]=__fmul_rn(c[row*16+a*4+j],__bfloat162float(h[(row*4+a)*K+k]));
 float sum=__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3]);out[i]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(p[row*4+j],__bfloat162float(y[row*K+k])),sum));
}
__device__ float sum4(float v){v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,2));return __fadd_rn(v,__shfl_xor_sync(0xffffffff,v,1));}
__device__ float colsum4(float v){v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,8));return __fadd_rn(v,__shfl_xor_sync(0xffffffff,v,4));}
__global__ void sinkhorn(const float*m,const float*s,const float*b,float*pre,float*post,float*comb,int M,float eps,const float* rstd=nullptr){
 int row=blockIdx.x,lane=threadIdx.x;if(row>=M)return;
 float z=lane<24?m[row*24+lane]:0.f;if(rstd)z=__fmul_rn(z,rstd[row]);
 if(lane<4){pre[row*4+lane]=__fadd_rn(1.f/(1.f+expf(-__fmaf_rn(z,s[0],b[lane]))),eps);float p=m[row*24+4+lane];if(rstd)p=__fmul_rn(p,rstd[row]);post[row*4+lane]=__fmul_rn(2.f,1.f/(1.f+expf(-__fmaf_rn(p,s[1],b[4+lane]))));}
 float q=lane<16?m[row*24+8+lane]:0.f;if(rstd)q=__fmul_rn(q,rstd[row]);float v=__fmaf_rn(q,s[2],b[8+(lane&15)]);
 float mx=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
 v=expf(v-mx);v=__fadd_rn(v/sum4(v),eps);v=v/__fadd_rn(colsum4(v),eps);
 for(int it=1;it<20;++it){v=v/__fadd_rn(sum4(v),eps);v=v/__fadd_rn(colsum4(v),eps);}
 if(lane<16)comb[row*16+lane]=v;
}
template<bool BF>
__global__ void f32linear(const float*x,const void*w,float*y,int M,int N,int K){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp;float acc[8]={0,0,0,0,0,0,0,0};
 if(row>=N)return;
 for(int k=lane;k<K;k+=32){float weight;if constexpr(BF)weight=__bfloat162float(((const bf16*)w)[(size_t)row*K+k]);else weight=((const float*)w)[(size_t)row*K+k];
  for(int m=0;m<M;++m)acc[m]=__fmaf_rn(weight,x[m*K+k],acc[m]);}
 for(int m=0;m<M;++m){for(int d=16;d;d>>=1)acc[m]=__fadd_rn(acc[m],__shfl_down_sync(0xffffffff,acc[m],d));if(!lane)y[m*N+row]=acc[m];}
}
__global__ void router(const float*logits,const float*bias,float*u,float*b,int count,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float z=logits[i],v=sqrtf(z>20.f?z:log1pf(expf(z)));u[i]=v;b[i]=__fadd_rn(v,bias[i%N]);}
__global__ void rope(bf16*x,const float*freq,int rows,int heads,int dim,int rd,int inverse){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads*rd/2)return;int pair=i%(rd/2),head=(i/(rd/2))%heads,row=i/(heads*(rd/2));int off=(row*heads+head)*dim+dim-rd+pair*2;
 float a=__bfloat162float(x[off]),b=__bfloat162float(x[off+1]),c=freq[row*rd+pair*2],s=freq[row*rd+pair*2+1];if(inverse)s=-s;
 // PyTorch complex64 contracts the first product for real and second for imag.
 x[off]=__float2bfloat16_rn(__fmaf_rn(a,c,-__fmul_rn(b,s)));x[off+1]=__float2bfloat16_rn(__fmaf_rn(b,c,__fmul_rn(a,s)));
}
__global__ void engram(const bf16*h,const bf16*key,const bf16*value,const bf16*qw,const bf16*kw,const uint8_t*mask,bf16*out,int M,int K,float eps){
 __shared__ float sh[3][512];int row=blockIdx.x,tid=threadIdx.x;float v[3][4]={{0,0,0,0},{0,0,0,0},{0,0,0,0}};
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){int k=i+j;float a=__bfloat162float(h[row*K+k]),b=__bfloat162float(key[row*K+k]);float w=__fmul_rn(__bfloat162float(qw[(row%4)*K+k]),__bfloat162float(kw[(row%4)*K+k]));
  v[0][j]=__fadd_rn(v[0][j],__fmul_rn(a,a));v[1][j]=__fadd_rn(v[1][j],__fmul_rn(b,b));v[2][j]=__fadd_rn(v[2][j],__fmul_rn(__fmul_rn(a,w),b));}
 float sum[3];for(int r=0;r<3;++r){sum[r]=__fadd_rn(__fadd_rn(__fadd_rn(v[r][0],v[r][1]),v[r][2]),v[r][3]);sh[r][tid]=sum[r];}__syncthreads();
 for(int d=blockDim.x/2;d>=32;d>>=1){if(tid<d)for(int r=0;r<3;++r){sum[r]=__fadd_rn(sum[r],sh[r][tid+d]);sh[r][tid]=sum[r];}__syncthreads();}
 if(tid<32){for(int d=16;d;d>>=1)for(int r=0;r<3;++r)sum[r]=__fadd_rn(sum[r],__shfl_down_sync(0xffffffff,sum[r],d));
  if(!tid){float rstd=__fmul_rn(rsqrtf(__fadd_rn(__fmul_rn(sum[0],1.f/K),eps)),rsqrtf(__fadd_rn(__fmul_rn(sum[1],1.f/K),eps)));float dot=__fmul_rn(__fmul_rn(sum[2],rstd),1.f/sqrtf(float(K)));float gate=1.f/(1.f+expf(-copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot)));sh[0][0]=(mask&&!mask[row/4])?0.f:gate;}}
 __syncthreads();for(int k=tid;k<K;k+=blockDim.x)out[row*K+k]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(h[row*K+k]),__fmul_rn(sh[0][0],__bfloat162float(value[(row/4)*K+k]))));
}
__device__ __forceinline__ float mixvalue(const bf16*x,const float*pre,int row,int k,int K){if(!pre)return __bfloat162float(x[row*K+k]);float v[4];for(int j=0;j<4;++j)v[j]=__fmul_rn(pre[row*4+j],__bfloat162float(x[(row*4+j)*K+k]));return __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f,v[0]),v[1]),v[2]),v[3])));}
__global__ void normquant(const bf16*x,const float*pre,const bf16*w,bf16*out,uint8_t*a,uint8_t*s,int M,int K,float eps,int width){
 extern __shared__ bf16 normalized[];__shared__ float sh[512];int row=blockIdx.x,tid=threadIdx.x;float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float z=mixvalue(x,pre,row,i+j,K);v[j]=__fadd_rn(v[j],__fmul_rn(z,z));}
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sh,width);
 if(!tid)sh[0]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));__syncthreads();
 for(int i=tid;i<K;i+=blockDim.x){bf16 z=__float2bfloat16_rn(__fmul_rn(__fmul_rn(mixvalue(x,pre,row,i,K),sh[0]),__bfloat162float(w[i])));normalized[i]=z;if(out)out[row*K+i]=z;}__syncthreads();
 int lane=tid&31,warp=tid>>5;for(int g=warp;g<K/32;g+=blockDim.x/32){float z=__bfloat162float(normalized[g*32+lane]),mx=fabsf(z);for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(mx,1e-4f),1.f/448.f));unsigned exponent=(bits>>23)+((bits&0x7fffff)!=0);float inv=__uint_as_float((254-exponent)<<23);a[row*K+g*32+lane]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,__fmul_rn(z,inv))),__NV_SATFINITE,__NV_E4M3);if(!lane)s[row*(K/32)+g]=exponent;}
}
__global__ void promote_stats(const bf16*x,float*y,float*rstd,int M,int K,float eps,int width){__shared__ float sh[512];int row=blockIdx.x,tid=threadIdx.x;float v[4]={0,0,0,0};
 for(int i=tid*4;i<K;i+=blockDim.x*4)for(int j=0;j<4;++j)if(i+j<K){float a=__bfloat162float(x[row*K+i+j]);y[row*K+i+j]=a;v[j]=__fadd_rn(v[j],__fmul_rn(a,a));}
 float sum=blocksum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]),sh,width);if(!tid)rstd[row]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/K),eps));
}
__global__ void promote(const bf16*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__bfloat162float(x[i]);}
__global__ void shared_act(const bf16*g,const bf16*u,bf16*z,uint8_t*a,uint8_t*s,int groups){int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;if(group>=groups)return;int i=group*32+lane;
 float gate=fminf(__bfloat162float(g[i]),10.f),up=fminf(10.f,fmaxf(-10.f,__bfloat162float(u[i])));float value=__fmul_rn(gate/(1.f+expf(-gate)),up);bf16 zz=__float2bfloat16_rn(value);if(z)z[i]=zz;
 value=__bfloat162float(zz);float mx=fabsf(value);for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(mx,1e-4f),1.f/448.f));unsigned exponent=(bits>>23)+((bits&0x7fffff)!=0);float inv=__uint_as_float((254-exponent)<<23);a[i]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,__fmul_rn(value,inv))),__NV_SATFINITE,__NV_E4M3);if(!lane)s[group]=exponent;
}
__global__ void routerweights(const float*s,const int32_t*ids,float*w,int M,int N,int topk,float scale){int row=blockIdx.x,lane=threadIdx.x;float v[6]={0,0,0,0,0,0};if(lane>=topk)return;for(int i=0;i<topk;++i)v[i]=s[row*N+ids[row*topk+i]];float sum=topk==6?__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[4]),v[2]),__fadd_rn(__fadd_rn(v[1],v[5]),v[3])):__fadd_rn(__fadd_rn(v[0],v[2]),v[1]);w[row*topk+lane]=__fmul_rn(v[lane]/__fadd_rn(sum,1e-20f),scale);}
__global__ void hcmean(const bf16*h,bf16*y,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M*K)return;int row=i/K,k=i%K;float sum=0.f;for(int j=0;j<4;++j)sum=__fadd_rn(sum,__bfloat162float(h[(row*4+j)*K+k]));y[i]=__float2bfloat16_rn(__fmul_rn(sum,.25f));}
}
extern "C" int ds41_dense_rms(const void*x,const void*w,void*y,int M,int K,float eps,cudaStream_t st){if(!x||!w||!y||M<1||M>8||K<128||K%4)return cudaErrorInvalidValue;
 int width,threads=geometry(M,K,width);rms<<<M,threads,0,st>>>((const bf16*)x,(const bf16*)w,(bf16*)y,M,K,eps,width);return cudaGetLastError();}
extern "C" int ds41_dense_hc_pre(const void*h,const float*p,void*y,int M,int K,cudaStream_t st){if(!h||!p||!y||M<1||K<1)return cudaErrorInvalidValue;hcpre<<<(M*K+255)/256,256,0,st>>>((const bf16*)h,p,(bf16*)y,M,K);return cudaGetLastError();}
extern "C" int ds41_dense_hc_post(const void*y,const void*h,const float*p,const float*c,void*out,int M,int K,cudaStream_t st){if(!y||!h||!p||!c||!out||M<1||K<1)return cudaErrorInvalidValue;hcpost<<<(M*4*K+255)/256,256,0,st>>>((const bf16*)y,(const bf16*)h,p,c,(bf16*)out,M,K);return cudaGetLastError();}
extern "C" int ds41_dense_hc_sinkhorn(const float*m,const float*s,const float*b,float*p,float*post,float*c,int M,float eps,cudaStream_t st){if(!m||!s||!b||!p||!post||!c||M<1)return cudaErrorInvalidValue;sinkhorn<<<M,32,0,st>>>(m,s,b,p,post,c,M,eps);return cudaGetLastError();}
extern "C" int ds41_dense_f32_bf16(const float*x,const void*w,float*y,int M,int N,int K,cudaStream_t st){if(!x||!w||!y||M<1||M>8||N<1||K<1)return cudaErrorInvalidValue;f32linear<true><<<(N+3)/4,128,0,st>>>(x,w,y,M,N,K);return cudaGetLastError();}
extern "C" int ds41_dense_f32(const float*x,const float*w,float*y,int M,int N,int K,cudaStream_t st){if(!x||!w||!y||M<1||M>8||N<1||K<1)return cudaErrorInvalidValue;f32linear<false><<<(N+3)/4,128,0,st>>>(x,w,y,M,N,K);return cudaGetLastError();}
extern "C" int ds41_dense_router_scores(const float*l,const float*bias,float*u,float*b,int M,int N,cudaStream_t st){if(!l||!bias||!u||!b||M<1||N<1)return cudaErrorInvalidValue;router<<<(M*N+255)/256,256,0,st>>>(l,bias,u,b,M*N,N);return cudaGetLastError();}
extern "C" int ds41_dense_rope(void*x,const float*f,int R,int H,int D,int RD,int inv,cudaStream_t st){if(!x||!f||R<1||H<1||RD<2||RD%2||D<RD)return cudaErrorInvalidValue;rope<<<(R*H*RD/2+255)/256,256,0,st>>>((bf16*)x,f,R,H,D,RD,inv);return cudaGetLastError();}
extern "C" int ds41_dense_engram_gate(const void*h,const void*k,const void*v,const void*qw,const void*kw,const uint8_t*mask,void*out,int M,int K,float eps,cudaStream_t st){if(!h||!k||!v||!qw||!kw||!out||M<1||M>8||K<128||K%4)return cudaErrorInvalidValue;int height=1;while(height*2<=M*4&&height*2<=16)height*=2;int width=512/height;engram<<<M*4,width,0,st>>>((const bf16*)h,(const bf16*)k,(const bf16*)v,(const bf16*)qw,(const bf16*)kw,mask,(bf16*)out,M,K,eps);return cudaGetLastError();}
extern "C" int ds41_dense_cublas(void* raw,const void*x,const void*w,void*y,int M,int N,int K,int f32,int groups,cudaStream_t st){if(!raw||!x||!w||!y||M<1||N<1||K<1||groups<1)return cudaErrorInvalidValue;
 auto h=(cublasHandle_t)raw;cublasStatus_t status=cublasSetStream(h,st);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
 // Set the math mode explicitly: TF32 is never enabled for these reference exceptions.
 status=cublasSetMathMode(h,CUBLAS_DEFAULT_MATH);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;
 float alpha=1.f,beta=0.f;auto dt=f32?CUDA_R_32F:CUDA_R_16BF;
 if(groups==1)status=cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,dt,K,x,dt,K,&beta,y,dt,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
 else status=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,dt,K,(long long)N*K,x,dt,K,(long long)M*K,&beta,y,dt,N,(long long)M*N,groups,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
 return status==CUBLAS_STATUS_SUCCESS?0:10000+status;
}
extern "C" int ds41_dense_norm_quant(const void*x,const float*pre,const void*w,void*out,void*a,void*s,int M,int K,float eps,cudaStream_t st){if(!x||!w||!a||!s||M<1||M>8||K<128||K%32||K>20480)return cudaErrorInvalidValue;int width,threads=geometry(M,K,width);
 normquant<<<M,threads,K*sizeof(bf16),st>>>((const bf16*)x,pre,(const bf16*)w,(bf16*)out,(uint8_t*)a,(uint8_t*)s,M,K,eps,width);return cudaGetLastError();}
extern "C" int ds41_dense_hc_mixes(void* handle,const void*h,const float*fn,const float*s,const float*b,float*pre,float*post,float*comb,float*x,float*mixes,float*rstd,int M,int K,float neps,float heps,cudaStream_t st){if(!handle||!h||!fn||!s||!b||!pre||!post||!comb||!x||!mixes||!rstd||M<1||M>8||K<32||K%32)return cudaErrorInvalidValue;int width,threads=geometry(M,K*4,width);promote_stats<<<M,threads,0,st>>>((const bf16*)h,x,rstd,M,K*4,neps,width);int rc=cudaGetLastError();if(rc)return rc;rc=ds41_dense_cublas(handle,x,fn,mixes,M,24,K*4,1,1,st);if(rc)return rc;sinkhorn<<<M,32,0,st>>>(mixes,s,b,pre,post,comb,M,heps,rstd);return cudaGetLastError();}
extern "C" int ds41_dense_shared_act(const void*g,const void*u,void*z,void*a,void*s,int M,int K,cudaStream_t st){if(!g||!u||!a||!s||M<1||K<32||K%32)return cudaErrorInvalidValue;shared_act<<<(M*K+127)/128,128,0,st>>>((const bf16*)g,(const bf16*)u,(bf16*)z,(uint8_t*)a,(uint8_t*)s,M*K/32);return cudaGetLastError();}
extern "C" int ds41_dense_promote(const void*x,float*y,int n,cudaStream_t st){if(!x||!y||n<1)return cudaErrorInvalidValue;promote<<<(n+255)/256,256,0,st>>>((const bf16*)x,y,n);return cudaGetLastError();}
extern "C" int ds41_dense_wo_a(void*raw,const void*x,const void*w,void*y,int M,int N,int K,int G,cudaStream_t st){if(!raw||!x||!w||!y||M<1||M>8||N<1||K<1||G<1)return cudaErrorInvalidValue;auto h=(cublasHandle_t)raw;auto status=cublasSetStream(h,st);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;status=cublasSetMathMode(h,CUBLAS_DEFAULT_MATH);if(status!=CUBLAS_STATUS_SUCCESS)return 10000+status;float alpha=1.f,beta=0.f;
 status=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,CUDA_R_16BF,K,(long long)N*K,x,CUDA_R_16BF,G*K,K,&beta,y,CUDA_R_16BF,G*N,N,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);return status==CUBLAS_STATUS_SUCCESS?0:10000+status;}
extern "C" int ds41_dense_router_weights(const float*s,const int32_t*ids,float*w,int M,int N,int topk,float scale,cudaStream_t st){if(!s||!ids||!w||M<1||N<topk||(topk!=3&&topk!=6))return cudaErrorInvalidValue;routerweights<<<M,32,0,st>>>(s,ids,w,M,N,topk,scale);return cudaGetLastError();}
extern "C" int ds41_dense_hc_mean(const void*h,void*y,int M,int K,cudaStream_t st){if(!h||!y||M<1||K<1)return cudaErrorInvalidValue;hcmean<<<(M*K+255)/256,256,0,st>>>((const bf16*)h,(bf16*)y,M,K);return cudaGetLastError();}

extern "C" int ds41_dense_rms_rope(const void*x,const void*w,void*y,const float*f,int M,int K,int RD,float eps,int inverse,cudaStream_t st){
 if(!x||!w||!y||!f||M<1||M>8||K<128||K%4||RD<2||RD%2||RD>K)return cudaErrorInvalidValue;
 int width,threads=geometry(M,K,width);rmsrope<<<M,threads,0,st>>>((const bf16*)x,(const bf16*)w,(bf16*)y,f,M,K,RD,eps,width,inverse);return cudaGetLastError();
}
