#include "ds41/attention.h"
#include "split_attention.cuh"
extern "C" int ds41_attn_sparse_split(const void*q,const void*kv,const float*sink,const int*idx,void*out,float*scores,int rows,int k,int heads,void*st){
 if(!q||!kv||!sink||!idx||!out||!scores||rows<1||k<1||k>640||(heads!=4&&heads!=8&&heads!=16))return cudaErrorInvalidValue;
 cudaError_t e=cudaFuncSetAttribute(ds41_qk_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,81920);if(e)return e;
 e=cudaFuncSetAttribute(ds41_pv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,69632);if(e)return e;
 ds41_qk_kernel<<<dim3(rows,64/heads,(k+63)/64),256,81920,(cudaStream_t)st>>>((const bfloat16_t*)q,(const bfloat16_t*)kv,idx,scores,heads,rows,k,k);
 e=cudaGetLastError();if(e)return e;
 ds41_pv_kernel<<<dim3(rows,64/heads),256,69632,(cudaStream_t)st>>>(sink,(const bfloat16_t*)kv,(bfloat16_t*)out,(const bfloat16_t*)q,idx,heads,rows,k,k,scores);return cudaGetLastError();
}
