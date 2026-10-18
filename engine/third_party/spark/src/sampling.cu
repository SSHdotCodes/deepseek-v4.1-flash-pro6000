#include "ds41/spark.h"
#include "ds41/spark_rng.h"
#include <cuda_runtime.h>
#include <math.h>
namespace {
__device__ double reduce(double v,double*sh){int t=threadIdx.x;sh[t]=v;__syncthreads();for(int d=128;d;d>>=1){if(t<d)sh[t]+=sh[t+d];__syncthreads();}double result=sh[0];__syncthreads();return result;}
__device__ double mass(const float*p,int n,double*sh){double z=0;for(int i=threadIdx.x;i<n;i+=256){double x=p[i];if(!isfinite(x)||x<0)x=NAN;z+=x;}return reduce(z,sh);}
__device__ int argmax(const float*p,int n,int*is,float*fs){int t=threadIdx.x,b=-1;float v=-INFINITY;for(int i=t;i<n;i+=256)if(p[i]>v||(p[i]==v&&(b<0||i<b))){b=i;v=p[i];}is[t]=b;fs[t]=v;__syncthreads();for(int d=128;d;d>>=1){if(t<d&&(fs[t+d]>fs[t]||(fs[t+d]==fs[t]&&is[t+d]<is[t]))){fs[t]=fs[t+d];is[t]=is[t+d];}__syncthreads();}int result=is[0];__syncthreads();return result;}
/* Contiguous per-thread intervals make inverse-CDF selection deterministic.
 * Both q sampling and p/q acceptance use normalized versions of the same
 * actual FP32 row. Residual subtracts those measures in FP64. */
__device__ int draw(const float*p,const float*q,int n,double pz,double qz,double u,double*sh,int*answer){
 int t=threadIdx.x,chunk=(n+255)/256,begin=t*chunk,end=min(n,begin+chunk);double z=0;
 for(int i=begin;i<end;++i)z+=q?fmax(0.,double(p[i])/pz-double(q[i])/qz):double(p[i])/pz;
 sh[t]=z;__syncthreads();if(!t){double sum=0;for(int j=0;j<256;++j){double v=sh[j];sh[j]=sum;sum+=v;}sh[256]=sum;*answer=n;}
 __syncthreads();double total=sh[256],target=u*total;
 if(total>0&&target>=sh[t]&&(t==255||target<sh[t+1])){
  double c=sh[t];int last=-1;for(int i=begin;i<end;++i){double v=q?fmax(0.,double(p[i])/pz-double(q[i])/qz):double(p[i])/pz;if(v>0)last=i;c+=v;if(target<c){atomicMin(answer,i);break;}}
  if(*answer==n&&last>=0)atomicMin(answer,last);
 }
 __syncthreads();int result=*answer==n?-1:*answer;__syncthreads();return result;
}
__device__ double random_at(uint64_t seed,uint64_t address,int slot,uint32_t purpose,int v2){return v2?spark_uniform_v2(seed,address,slot,purpose):spark_uniform(seed,address+slot,purpose);}
__global__ void sample_kernel(const float*p,int n,uint64_t seed,uint64_t pos,uint32_t purpose,int*out,int slot=0,int v2=0){seed+=blockIdx.x;out+=blockIdx.x;__shared__ double sh[257];__shared__ int answer;double z=mass(p,n,sh);if(!(z>0)||!isfinite(z)){if(!threadIdx.x)*out=-1;return;}int v=draw(p,nullptr,n,z,1,random_at(seed,pos,slot,purpose,v2),sh,&answer);if(!threadIdx.x)*out=v;}
__global__ void verify_kernel(const int*d,const float*q,const float*p,int count,int n,int greedy,uint64_t seed,uint64_t pos,ds41_spark_result*out,int many=0,const ds41_spark_result*first=nullptr,int v2=0){
 if(many){seed+=blockIdx.x;d+=blockIdx.x;out+=blockIdx.x;if(first&&((min(first[blockIdx.x].tokens[0],8))%2))p+=size_t(2)*n;}
 __shared__ double sh[257];__shared__ int answer,ids[256];__shared__ float fs[256];__shared__ ds41_spark_result result;
 if(!threadIdx.x)result={};__syncthreads();
 for(int k=0;k<=count;++k){const float*pk=p+size_t(k)*n;double pz=mass(pk,n,sh);if(!(pz>0)||!isfinite(pz)){if(!threadIdx.x){result.emitted=-1;*out=result;}return;}
  if(k==count){int v=greedy?argmax(pk,n,ids,fs):draw(pk,nullptr,n,pz,1,random_at(seed,pos,k,DS41_SPARK_TARGET,v2),sh,&answer);if(!threadIdx.x)result.tokens[result.emitted++]=v;break;}
  int token=d[k];if(token<0||token>=n){if(!threadIdx.x){result.emitted=-1;*out=result;}return;}
  const float*qk=q?q+size_t(k)*n:nullptr;double qz=greedy?1:mass(qk,n,sh);if(!greedy&&(!(qz>0)||!isfinite(qz)||!(qk[token]>0))){if(!threadIdx.x){result.emitted=-1;*out=result;}return;}
  bool accept=greedy?token==argmax(pk,n,ids,fs):random_at(seed,pos,k,DS41_SPARK_ACCEPT,v2)<fmin(1.,(double(pk[token])/pz)/(double(qk[token])/qz));
  if(accept){if(!threadIdx.x){result.tokens[result.emitted++]=token;++result.accepted_drafts;}}
  else{int v=greedy?argmax(pk,n,ids,fs):draw(pk,qk,n,pz,qz,random_at(seed,pos,k,DS41_SPARK_RESIDUAL,v2),sh,&answer);if(!threadIdx.x){result.tokens[result.emitted++]=v;result.rejected=1;}break;}
  __syncthreads();
 }
 __syncthreads();if(!threadIdx.x){result.accepted_input_rows=result.accepted_drafts+1;*out=result;}
}
/* Fixed-distribution QA specialization. Preserve the production draw's
 * per-thread chunk sums, serial chunk prefixes, element prefixes, total and
 * last-positive fallback exactly, rather than replacing them with a new scan.
 * Thus every finite uniform selects the same token, including boundaries.
 * Five tables: q,p,p-q,rotated-p,rotated-p-q. Each table stores n element
 * prefixes,257 chunk prefixes,256 last-positive IDs,and the original masses. */
__global__ void fixed_cdf_build(const float*ps,const float*q,int n,double*tables){
 __shared__ double sh[257];int which=blockIdx.x,t=threadIdx.x;const float*p=which>=3?ps+size_t(2)*n:ps;double pz=mass(p,n,sh),qz=mass(q,n,sh);
 bool residual=which==2||which==4,isq=which==0;int chunk=(n+255)/256,begin=t*chunk,end=min(n,begin+chunk),last=-1;double z=0;
 for(int i=begin;i<end;++i){double v=isq?double(q[i])/qz:residual?fmax(0.,double(p[i])/pz-double(q[i])/qz):double(p[i])/pz;z+=v;if(v>0)last=i;}
 sh[t]=z;__syncthreads();if(!t){double sum=0;for(int j=0;j<256;++j){double v=sh[j];sh[j]=sum;sum+=v;}sh[256]=sum;}__syncthreads();
 double*out=tables+size_t(which)*(n+515);out[n+t]=sh[t];out[n+257+t]=last;if(!t){out[n+256]=sh[256];out[n+513]=pz;out[n+514]=qz;}
 double c=sh[t];for(int i=begin;i<end;++i){double v=isq?double(q[i])/qz:residual?fmax(0.,double(p[i])/pz-double(q[i])/qz):double(p[i])/pz;c+=v;out[i]=c;}
}
__device__ int fixed_draw(const double*cdf,int n,double u){const double*prefix=cdf+n;double target=u*prefix[256];int low=0,high=255;while(low<high){int middle=(low+high)/2;if(prefix[middle+1]<=target)low=middle+1;else high=middle;}int t=low,chunk=(n+255)/256;low=t*chunk;high=min(n,low+chunk);int end=high;while(low<high){int middle=(low+high)/2;if(cdf[middle]<=target)low=middle+1;else high=middle;}return low<end?low:int(prefix[257+t]);}
__global__ void fixed_cdf_sweep(const float*p,const float*q,int n,int draws,uint64_t seed,const double*tables,int*out){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=draws)return;seed+=row;int stride=n+515;const double*qc=tables,*pc=tables+stride,*rc=tables+2*stride;
 double qz=qc[n+514],pz=pc[n+513];int proposal=fixed_draw(qc,n,spark_uniform_v2(seed,100,0,1));bool accepted=spark_uniform_v2(seed,100,0,2)<fmin(1.,(double(p[proposal])/pz)/(double(q[proposal])/qz));int first=accepted?proposal:fixed_draw(rc,n,spark_uniform_v2(seed,100,0,3));out[row*2]=first;
 bool rotated=(min(first,8)%2)!=0;pc=tables+(rotated?3:1)*stride;rc=tables+(rotated?4:2)*stride;const float*p2=p+(rotated?size_t(2)*n:0);pz=pc[n+513];proposal=fixed_draw(qc,n,spark_uniform_v2(seed,101,0,1));accepted=spark_uniform_v2(seed,101,0,2)<fmin(1.,(double(p2[proposal])/pz)/(double(q[proposal])/qz));out[row*2+1]=accepted?proposal:fixed_draw(rc,n,spark_uniform_v2(seed,101,0,3));
}
}
extern "C" int ds41_spark_sample_gpu(const float*p,int n,uint64_t seed,uint64_t pos,uint32_t purpose,int32_t*out,void*stream){if(!p||!out||n<1||purpose>3)return cudaErrorInvalidValue;sample_kernel<<<1,256,0,(cudaStream_t)stream>>>(p,n,seed,pos,purpose,out);return cudaGetLastError();}
extern "C" int ds41_spark_verify_gpu(const int32_t*d,const float*q,const float*p,int count,int n,int greedy,uint64_t seed,uint64_t pos,ds41_spark_result*out,void*stream){if(!out||!p||count<0||count>5||n<1||(count&&(!d||(!q&&!greedy)))||pos>UINT64_MAX-5)return cudaErrorInvalidValue;verify_kernel<<<1,256,0,(cudaStream_t)stream>>>(d,q,p,count,n,greedy,seed,pos,out);return cudaGetLastError();}
/* Statistical sweep over the actual production kernels, fixed p/q and one
 * proposal. Batch index enters only the request seed. Test-only export. */
extern "C" int ds41_spark_test_sweep(const float*p,const float*q,int n,int draws,uint64_t seed,uint64_t pos,int32_t*proposals,ds41_spark_result*out,const ds41_spark_result*first,void*stream){
 if(!p||!q||!proposals||!out||n<1||draws<1)return cudaErrorInvalidValue;
 sample_kernel<<<draws,256,0,(cudaStream_t)stream>>>(q,n,seed,pos,1,proposals,0,1);int e=cudaGetLastError();if(e)return e;
 verify_kernel<<<draws,256,0,(cudaStream_t)stream>>>(proposals,q,p,1,n,0,seed,pos,out,1,first,1);return cudaGetLastError();
}
extern "C" int ds41_spark_test_fixed_cdf(const float*p,const float*q,int n,int draws,uint64_t seed,double*scratch,int32_t*out,void*stream){if(!p||!q||!scratch||!out||n<1||draws<1)return cudaErrorInvalidValue;fixed_cdf_build<<<5,256,0,(cudaStream_t)stream>>>(p,q,n,scratch);int e=cudaGetLastError();if(e)return e;fixed_cdf_sweep<<<(draws+255)/256,256,0,(cudaStream_t)stream>>>(p,q,n,draws,seed,scratch,out);return cudaGetLastError();}

extern "C" int ds41_spark_sample_gpu_v2(const float*p,int n,uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,int32_t*out,void*stream){if(!p||!out||n<1||purpose>3||slot>5)return cudaErrorInvalidValue;sample_kernel<<<1,256,0,(cudaStream_t)stream>>>(p,n,seed,step,purpose,out,slot,1);return cudaGetLastError();}
extern "C" int ds41_spark_verify_gpu_v2(const int32_t*d,const float*q,const float*p,int count,int n,int greedy,uint64_t seed,uint64_t step,ds41_spark_result*out,void*stream){if(!out||!p||count<0||count>5||n<1||(count&&(!d||(!q&&!greedy))))return cudaErrorInvalidValue;verify_kernel<<<1,256,0,(cudaStream_t)stream>>>(d,q,p,count,n,greedy,seed,step,out,0,nullptr,1);return cudaGetLastError();}
