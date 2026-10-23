#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace ds41e::hotpath {
void embedding(const void *table, const int *tokens, void *h, float *pre,
               int rows, cudaStream_t);
void gather_routes(const uint8_t *x, const uint8_t *scales,
                   const float *weights, const int *source, const int *slots,
                   uint8_t *out, uint8_t *out_scales, float *out_weights,
                   int count, cudaStream_t);
// BF16 rows for EXL3 experts (x: [rows, 5120] BF16; out: [count, 5120] BF16).
void gather_routes_bf16(const void *x, const float *weights, const int *source,
                        const int *slots, void *out, float *out_weights,
                        int count, cudaStream_t);
void scatter_routes(const void *compact, const int *slots, void *out,
                    int count, cudaStream_t);
void capture_rows(const void *h, void *out, int rows, int layer_slot,
                  cudaStream_t);
// Batch rows while retaining the sequential decoder's PyTorch M1 reduction
// geometry. The general dense RMS ABI intentionally matches official batched M.
void rms_decode(const void *x,const void *weight,void *out,int rows,int cols,
                cudaStream_t);
void activate_routes(const void *gate,const void *up,const float *weights,
                     const int *work_rows,uint8_t *quant,uint8_t *scales,
                     int count,cudaStream_t);
// activate_routes with BF16 output rows (row stride 2304) for EXL3 down projections.
void activate_routes_bf16(const void *gate,const void *up,const float *weights,
                          const int *work_rows,void *out,int count,cudaStream_t);
void stage_frequencies(const float *table,float *main,float *compressed,
                       const int64_t *position,int rows,int context,cudaStream_t);
void store_groups(const void *packed,void *global,const int64_t *position,
                   int rows,int bytes,int ratio,cudaStream_t);
void lengths_aux(const int *visible,int *out,int rows,int divisor,int fixed,
                  cudaStream_t);
} // namespace ds41e::hotpath
