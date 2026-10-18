#pragma once
#include "ds41/store.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ds41 {
enum class Projection : int { Gate, Up, Down };
struct GemmJob {
    ExpertId expert;
    int tokens;                         // 1..8, dense rows for this expert
    const uint8_t *x, *x_scale;          // E4M3 [M,K], E8M0 [M,K/32]
    __nv_bfloat16 *output;              // [M,N], mandatory BF16 boundary
};
cudaError_t quantize_mxfp8(const __nv_bfloat16* x, uint8_t* q, uint8_t* scales, int rows, int cols, cudaStream_t stream);
// Strict per-32-group FP32 accumulation, then BF16. Tensor core multiplication
// uses E4M3 representations of the exact E2M1 values, as the official reference.
cudaError_t grouped_mxfp4(const DeviceExpert* table, const GemmJob* jobs, int count,
                         Projection projection, int n, int k, cudaStream_t stream);
// Candidate native block-scaled MMA, independently fidelity-gated; not default.
cudaError_t grouped_mxfp4_native(const DeviceExpert* table,const GemmJob* jobs,int count,
                                Projection projection,int n,int k,cudaStream_t stream);
cudaError_t grouped_mxfp4_parallel(const DeviceExpert* table,const GemmJob* jobs,int count,
                                  Projection projection,int n,int k,cudaStream_t stream);
cudaError_t grouped_mxfp4_native_parallel(const DeviceExpert* table,const GemmJob* jobs,int count,
                                         Projection projection,int n,int k,cudaStream_t stream);
cudaError_t grouped_mxfp4_native_tiled(const DeviceExpert* table,const GemmJob* jobs,int count,
                                      Projection projection,int n,int k,cudaStream_t stream);
// BF16 gate/up -> FP32 clamp/silu/product/routing-weight -> BF16, BEFORE w2 quant.
cudaError_t swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up, const float* route_weight,
                   __nv_bfloat16* out, int rows, int cols, float limit, cudaStream_t stream);
// Expert outputs [M,topk,H], logical ids [M,topk], shared [M,H].
// No atomic reduction or physical-slot ordering: ascending logical id, shared last.
cudaError_t reduce_experts(const __nv_bfloat16* routed,const int32_t* ids,const __nv_bfloat16* shared,
                           __nv_bfloat16* out,int rows,int topk,int hidden,cudaStream_t stream);
} // namespace ds41

// Fixture/torch integration: raw device pointers, caller-owned stream and output.
// These allocate no scratch, descriptors, or CUDA context beyond the caller.
extern "C" int ds41_mxfp4_raw(const uint8_t* weight,const uint8_t* weight_scale,
                              const uint8_t* activation,const uint8_t* activation_scale,
                              void* output_bf16,int m,int n,int k,void* stream);
extern "C" int ds41_quantize_mxfp8_raw(const void* input_bf16,uint8_t* q,uint8_t* scales,
                                      int rows,int cols,void* stream);
extern "C" int ds41_mxfp4_native_raw(const uint8_t* weight,const uint8_t* weight_scale,
                                     const uint8_t* activation,const uint8_t* activation_scale,
                                     void* output_bf16,int m,int n,int k,void* stream);
extern "C" int ds41_mxfp4_parallel_raw(const uint8_t* weight,const uint8_t* weight_scale,
                                       const uint8_t* activation,const uint8_t* activation_scale,
                                       void* output_bf16,int m,int n,int k,void* stream);
extern "C" int ds41_swiglu_raw(const void* gate_bf16,const void* up_bf16,const float* route_weight,
                               void* output_bf16,int rows,int cols,float limit,void* stream);
extern "C" int ds41_mxfp4_native_parallel_raw(const uint8_t* weight,const uint8_t* weight_scale,
                                              const uint8_t* activation,const uint8_t* activation_scale,
                                              void* output_bf16,int m,int n,int k,void* stream);
