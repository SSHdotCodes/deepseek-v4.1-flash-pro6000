#include "ds41/attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <math_constants.h>
using bf=__nv_bfloat16;
__device__ float ex8(uint8_t c){return __uint_as_float(unsigned(c)<<23);}
__device__ float exf8(uint8_t c){return __half2float(__nv_cvt_fp8_to_halfraw(c,__NV_E4M3));}
__device__ float exf4(uint8_t c){return __half2float(__nv_cvt_fp4_to_halfraw(c,__NV_E2M1));}
__global__ void ringwrite(const uint8_t* in,uint8_t* ring,int bytes,int64_t start,int rows,int cap){
 int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
 if(i<int64_t(bytes)*rows)ring[((start+i/bytes)%cap)*bytes+i%bytes]=in[i];
}
extern "C" int ds41_attn_ring_write(const void*in,void*out,int bytes,int64_t start,int rows,int cap,void*st){
 if(!in||!out||bytes<1||start<0||rows<1||cap<rows)return cudaErrorInvalidValue;
 ringwrite<<<(int64_t(rows)*bytes+255)/256,256,0,(cudaStream_t)st>>>((const uint8_t*)in,(uint8_t*)out,bytes,start,rows,cap);return cudaGetLastError();
}
__global__ void pool(const float*kv,const float*sc,bf*out,int64_t start,int rows,int cap){
 int row=blockIdx.x,d=threadIdx.x;int64_t pos=start+row;
 for(;d<512;d+=blockDim.x){float y=0;if(pos&1){int a=((pos-1)%cap)*512+d,b=(pos%cap)*512+d;float mx=fmaxf(sc[a],sc[b]);float p0=expf(sc[a]-mx),p1=expf(sc[b]-mx);float denom=__fadd_rn(p0,p1);p0=__fdiv_rn(p0,denom);p1=__fdiv_rn(p1,denom);y=__fadd_rn(__fmul_rn(kv[a],p0),__fmul_rn(kv[b],p1));}out[row*512+d]=__float2bfloat16_rn(y);}
}
extern "C" int ds41_attn_pool2(const float*kv,const float*sc,void*out,int64_t start,int rows,int cap,void*st){
 if(!kv||!sc||!out||start<0||rows<1||cap<rows+1)return cudaErrorInvalidValue;
 pool<<<rows,256,0,(cudaStream_t)st>>>(kv,sc,(bf*)out,start,rows,cap);return cudaGetLastError();
}
__global__ void gather(const uint8_t*swa,int cap,const uint8_t*global,int gcap,const int*ids,int k,int64_t start,int rows,int ratio,bf*out,int*local,const int64_t* dynamic_start,int width,bool prefill){
 int slot=blockIdx.x, row=blockIdx.y,d=threadIdx.x;
 int count=width+k;int64_t pos=(dynamic_start?*dynamic_start:start)+row;int id=-1;const uint8_t*r=nullptr;bool sw=slot<width;
 if(sw){int64_t p=prefill?max(int64_t(0),pos-127)+slot:pos-127+slot;if(p>=0&&p<=pos){id=slot;r=swa+(p%cap)*528;}}
 else {int j=ids[row*k+slot-width];if(j>=0&&j<gcap&&int64_t(j+1)*ratio<=pos+1){id=slot;r=global+int64_t(j)*288;}}
 if(d==0)local[row*count+slot]=id;
 for(;d<512;d+=blockDim.x){float v=0;if(r){if(sw)v=__fmul_rn(exf8(r[d]),ex8(r[512+d/32]));else{int c=(r[d/2]>>(4*(d&1)))&15;v=__fmul_rn(exf4(c),exf8(r[256+d/16]));}}out[(int64_t(row)*count+slot)*512+d]=__float2bfloat16_rn(v);}
}
extern "C" int ds41_attn_gather(const void*swa,int cap,const void*g,int gcap,const int*ids,int k,int64_t start,int rows,int ratio,void*out,int*local,void*st){
 if(!swa||!out||!local||cap<128+rows||start<0||rows<1||k<0||k>512||gcap<0||(k&&(!g||!ids||ratio<1)))return cudaErrorInvalidValue;
 gather<<<dim3(128+k,rows),128,0,(cudaStream_t)st>>>((const uint8_t*)swa,cap,(const uint8_t*)g,gcap,ids,k,start,rows,ratio,(bf*)out,local,nullptr,128,false);return cudaGetLastError();
}

extern "C" int ds41_attn_gather_at(const void*swa,int cap,const void*g,int gcap,const int*ids,int k,const int64_t*start,int rows,int ratio,void*out,int*local,void*st){
 if(!swa||!out||!local||!start||cap<128+rows||rows<1||k<0||k>512||gcap<0||(k&&(!g||!ids||ratio<1)))return cudaErrorInvalidValue;
 gather<<<dim3(128+k,rows),128,0,(cudaStream_t)st>>>((const uint8_t*)swa,cap,(const uint8_t*)g,gcap,ids,k,0,rows,ratio,(bf*)out,local,start,128,false);return cudaGetLastError();
}
__global__ void lengths(const int64_t*start,int*one,int*two,int rows,int n){int i=blockIdx.x*256+threadIdx.x;if(i<rows){int64_t p=*start+i+1;one[i]=min(int64_t(n),p);two[i]=min(int64_t(n/2),p/2);}}
extern "C" int ds41_attn_lengths(const int64_t*start,int*one,int*two,int rows,int n,void*st){if(!start||!one||!two||rows<1||n<1)return cudaErrorInvalidValue;lengths<<<(rows+255)/256,256,0,(cudaStream_t)st>>>(start,one,two,rows,n);return cudaGetLastError();}

__global__ void ring_at(const uint8_t*in,uint8_t*out,int bytes,const int64_t*start,int rows,int cap,bool read){
 int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;if(i>=int64_t(bytes)*rows)return;
 int64_t at=(((*start)+i/bytes)%cap)*bytes+i%bytes;
 if(read)out[i]=in[at];else out[at]=in[i];
}
extern "C" int ds41_attn_ring_write_at(const void*in,void*out,int bytes,const int64_t*start,int rows,int cap,void*st){
 if(!in||!out||!start||bytes<1||rows<1||cap<rows)return cudaErrorInvalidValue;ring_at<<<(int64_t(rows)*bytes+255)/256,256,0,(cudaStream_t)st>>>((const uint8_t*)in,(uint8_t*)out,bytes,start,rows,cap,false);return cudaGetLastError();
}
extern "C" int ds41_attn_ring_read_at(const void*in,void*out,int bytes,const int64_t*start,int rows,int cap,void*st){
 if(!in||!out||!start||bytes<1||rows<1||cap<rows)return cudaErrorInvalidValue;ring_at<<<(int64_t(rows)*bytes+255)/256,256,0,(cudaStream_t)st>>>((const uint8_t*)in,(uint8_t*)out,bytes,start,rows,cap,true);return cudaGetLastError();
}

extern "C" int ds41_attn_gather_prefill_at(const void*swa,int cap,const void*g,int gcap,const int*ids,int k,const int64_t*start,int rows,int ratio,int width,void*out,int*local,void*st){
 if(!swa||!out||!local||!start||width<1||width>128||cap<width||rows<1||k<0||k>512||gcap<0||(k&&(!g||!ids||ratio<1)))return cudaErrorInvalidValue;
 gather<<<dim3(width+k,rows),128,0,(cudaStream_t)st>>>((const uint8_t*)swa,cap,(const uint8_t*)g,gcap,ids,k,0,rows,ratio,(bf*)out,local,start,width,true);return cudaGetLastError();
}
__global__ void poolprefill(const float*kv,const float*sc,bf*out,int pairs){
 int pair=blockIdx.x,d=threadIdx.x;for(;d<512;d+=blockDim.x){int a=pair*1024+d,b=a+512;float mx=fmaxf(sc[a],sc[b]);float p0=expf(sc[a]-mx),p1=expf(sc[b]-mx),denom=__fadd_rn(p0,p1);float v=__fadd_rn(__fmul_rn(kv[a],__fdiv_rn(p0,denom)),__fmul_rn(kv[b],__fdiv_rn(p1,denom)));out[pair*512+d]=__float2bfloat16_rn(v);}
}
extern "C" int ds41_attn_pool2_prefill(const float*kv,const float*sc,void*out,int rows,void*st){if(!kv||!sc||!out||rows<2)return cudaErrorInvalidValue;poolprefill<<<rows/2,256,0,(cudaStream_t)st>>>(kv,sc,(bf*)out,rows/2);return cudaGetLastError();}
