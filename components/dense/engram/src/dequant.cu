#include "ds41/engram.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
namespace {
__global__ void unpack(const uint8_t* x,__nv_bfloat16* y,int rows){int row=blockIdx.x,k=threadIdx.x;if(row>=rows)return;__nv_fp8_e4m3 q;q.__x=x[row*264+k];uint8_t e=x[row*264+256+k/32];float s=e==0?__uint_as_float(0x00400000):e==255?__uint_as_float(0x7fffffff):__uint_as_float(uint32_t(e)<<23);y[row*256+k]=__float2bfloat16_rn(__fmul_rn(float(q),s));}
}
extern "C" int ds41_engram_dequant(const void*x,void*y,int rows,void*stream){if(!x||!y||rows<1)return cudaErrorInvalidValue;unpack<<<rows,256,0,(cudaStream_t)stream>>>((const uint8_t*)x,(__nv_bfloat16*)y,rows);return cudaGetLastError();}
