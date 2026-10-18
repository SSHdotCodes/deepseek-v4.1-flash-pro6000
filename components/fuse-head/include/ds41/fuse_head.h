#pragma once
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
// Unqualified native-tree candidate, restricted to the DS41 target head.
// x[M,5120] FP32, original weights[129280,5120] BF16, out[M,129280] FP32.
// Caller owns all storage; no allocation, synchronization or state changes.
int ds41_fuse_head_decode(const float *x,const void *weights_bf16,float *out,
    int M,int N,int K,cudaStream_t stream);
#ifdef __cplusplus
}
#endif
