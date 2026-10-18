#pragma once
#include <stdint.h>
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
// All buffers are caller-owned, contiguous, on the device, and live through stream completion.
// Returns cudaError_t as int, no allocations, synchronization, default stream or hidden precision casts.
// BF16 and FP8 arguments use void* to keep this header usable from C and ggml adapters.
int ds41_dense_quant(const void* x_bf16, void* a_fp8, void* as_ue8m0, int M, int K, cudaStream_t stream);
int ds41_dense_fp8(const void* a_fp8, const void* as_ue8m0, const void* w_fp8,
                   const void* ws_ue8m0, void* y_bf16, int M, int N, int K,
                   int variant, cudaStream_t stream);
// variant -1: fixed per-shape dispatch; 0: portable strict k32; 1: strict k32 with parallel dot and serial carry;
// variant 2: native block-scaled MMA with parallel dot and serial carry (fixture-gated separately).
// variants 3/4/5/6: native (warps, output columns) = (4,8)/(4,4)/(8,4)/(4,2).
// M in 1..8, N/K positive multiples of 32. Weight scales [N/32,K/32], activation scales [M,K/32].
int ds41_dense_dequant(const void* w_fp8, const void* ws_ue8m0, void* w_bf16, int N,int K,cudaStream_t stream);
int ds41_dense_roundtrip(const void* a_fp8,const void* as_ue8m0,void* x_bf16,int M,int K,cudaStream_t stream);
int ds41_dense_rms(const void* x_bf16,const void* weight_bf16,void* y_bf16,int M,int K,float eps,cudaStream_t stream);
int ds41_dense_hc_pre(const void* h_bf16,const float* pre,void* y_bf16,int M,int K,cudaStream_t stream);
int ds41_dense_hc_post(const void* y_bf16,const void* h_bf16,const float* post,const float* comb,void* out_bf16,int M,int K,cudaStream_t stream);
int ds41_dense_hc_sinkhorn(const float* mixes,const float* scale,const float* base,float* pre,float* post,float* comb,int M,float eps,cudaStream_t stream);
// Strict FP32 arithmetic candidate, BF16 weights promoted exactly, float activations unchanged.
// Fixed reduction tree is batch-invariant; equality with cuBLAS is NOT assumed.
int ds41_dense_f32_bf16(const float* x,const void* w_bf16,float* y,int M,int N,int K,cudaStream_t stream);
int ds41_dense_f32(const float* x,const float* w,float* y,int M,int N,int K,cudaStream_t stream);
int ds41_dense_router_scores(const float* logits,const float* bias,float* unbiased,float* biased,int M,int N,cudaStream_t stream);
int ds41_dense_rope(void* x_bf16,const float* cos_sin,int rows,int heads,int dim,int rope_dim,int inverse,cudaStream_t stream);
int ds41_dense_engram_gate(const void* h_bf16,const void* key_bf16,const void* value_bf16,const void* qw_bf16,const void* kw_bf16,const uint8_t* token_mask,void* out_bf16,int M,int K,float eps,cudaStream_t stream);
// cuBLAS reference-compatible escape hatch. Caller supplies a cublasHandle_t, owns lifetime/workspace.
// Weights/activations must already have the stated dtype. f32 preserves FP32 activation and disables TF32.
// Return value: CUDA success 0 or 10000+cublasStatus. No persistent allocation or implicit conversion.
int ds41_dense_cublas(void* handle,const void* x,const void* w,void* y,int M,int N,int K,int f32,int groups,cudaStream_t stream);
// Fuse optional hc_pre (pre=null => x [M,K]) + RMSNorm + explicit BF16 boundary + quant.
// norm_bf16 may be null; a/s always required. Matches unfused boundaries, not an algebraic reassociation.
int ds41_dense_norm_quant(const void* x,const float* pre,const void* norm_weight,void* norm_bf16,void* a_fp8,void* as_ue8m0,int M,int K,float eps,cudaStream_t stream);
// Full HC coefficient path, native FP32 projection and Sinkhorn20.
// Scratch sizes: promoted_x[M,4*K] FP32, mixes[M,24] FP32, rstd[M] FP32.
int ds41_dense_hc_mixes(void* blas_handle,const void* h_bf16,const float* fn,const float* scale,const float* base,float* pre,float* post,float* comb,float* promoted_x,float* mixes,float* rstd,int M,int K,float norm_eps,float hc_eps,cudaStream_t stream);
int ds41_dense_shared_act(const void* gate_bf16,const void* up_bf16,void* z_bf16,void* a_fp8,void* as_ue8m0,int M,int K,cudaStream_t stream);
int ds41_dense_promote(const void* x_bf16,float* x_f32,int elements,cudaStream_t stream);
// wo_a grouped BF16 einsum, token-major x[M,G,K], y[M,G,N], w[G,N,K].
// Weights must be exact native FP8->BF16 dequant. Input has already had its FP8 roundtrip.
int ds41_dense_wo_a(void* blas_handle,const void* x_bf16,const void* w_bf16,void* y_bf16,int M,int N,int K,int groups,cudaStream_t stream);
// Top-k selection remains engine-owned; ids must be PyTorch-compatible order.
int ds41_dense_router_weights(const float* unbiased,const int32_t* ids,float* weights,int M,int N,int topk,float route_scale,cudaStream_t stream);
int ds41_dense_hc_mean(const void* h_bf16,void* mean_bf16,int M,int K,cudaStream_t stream);
// Exact RMS->BF16->RoPE for KV/compressor rows; same M-dependent reduction as rms.
// x/y [M,K], cos_sin [M,rope_dim], adjacent tail pairs; one kernel, supports in-place.
int ds41_dense_rms_rope(const void* x,const void* weight,void* y,const float* cos_sin,int M,int K,int rope_dim,float eps,int inverse,cudaStream_t stream);
// Additive round-2 class N APIs. Explicit opt-in, not promoted by this library.
// Head BF16 storage, FP32 input/output and FP32 FMA accumulation. M1..8,K%256==0.
int ds41_dense_head_compact(const float* x,const void* w_bf16,float* y,int M,int N,int K,cudaStream_t stream);
// Native wo_a [G*N,K] FP8 + [G*N/32,K/32] E8M0 scales, exact BF16 dequant in registers.
// Input [M,G,K] is already official FP8-roundtripped, output [M,G,N] BF16. N%32,K%256.
int ds41_dense_wo_a_compact(const void* x_bf16,const void* w_fp8,const void* ws_ue8m0,void* y_bf16,int M,int N,int K,int groups,cudaStream_t stream);
// Call once at engine init, then pass a fixed policy into captured dispatch calls.
// Default=0 reference. DS41_DENSE_COMPACT=1 requests class N; DS41_DENSE_REFERENCE!=0 forces0.
// Caller must enforce QUALITY acceptance before selecting class N for production.
int ds41_dense_policy_from_env(void);
int ds41_dense_head_dispatch(void* blas_handle,const float* x,const void* w_bf16,const float* w_f32,float* y,int M,int N,int K,int policy,cudaStream_t stream);
int ds41_dense_wo_a_dispatch(void* blas_handle,const void* x,const void* w_fp8,const void* ws,const void* w_bf16,void* y,int M,int N,int K,int groups,int policy,cudaStream_t stream);
#ifdef __cplusplus
}
#endif
