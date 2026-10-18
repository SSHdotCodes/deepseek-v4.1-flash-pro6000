#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace ds41e {
using Bf16 = __nv_bfloat16;
void bf16_float(const void *, float *, int, cudaStream_t);
void float_bf16(const float *, void *, int, cudaStream_t);
void expand_embedding(const void *, void *, float *, int, cudaStream_t);
void scale_bf16(void *, int, float, cudaStream_t);
void normalize_mixes(const void *, float *, int, cudaStream_t);
void route_weights(const float *, const int *, float *, int, cudaStream_t);
void engram_dequant(const uint8_t *, const uint8_t *, void *, int,
                    cudaStream_t);
void capture_mean(const void *, void *, int, cudaStream_t);
} // namespace ds41e
