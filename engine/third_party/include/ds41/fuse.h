#pragma once
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
// Caller-owned contiguous device buffers. All functions are allocation-free and
// asynchronous on the supplied stream. Decode M1 reduction geometry for all M.
// pre=null reads x[M,K]; otherwise x[M,4,K] and pre[M,4]. At least one
// output is required; FP8 data/scales must be supplied together. K%32==0.
int ds41_fuse_norm_quant_decode(const void *x,const float *pre,const void *weight,
    void *norm_bf16,float *norm_f32,void *fp8,void *scales,int M,int K,float eps,
    cudaStream_t stream);
// Full HC projection retains M separate cuBLAS M1 calls, in original row order.
// Scratch: promoted_h[M,4*K], mixes[M,24], rstd[M], all FP32. Outputs:
// current pre/post[M,4], comb[M,16]; normalization uses INCOMING pre[M,4],
// not the newly projected pre. This distinction is part of official HC semantics.
int ds41_fuse_hc_prepare(void *blas,const void *h,const float *fn,
    const float *scale,const float *base,const float *incoming_pre,
    const void *norm_weight,float *pre,float *post,float *comb,
    float *promoted_h,float *mixes,float *rstd,void *norm_bf16,float *norm_f32,
    void *fp8,void *fp8_scales,int M,int K,float norm_eps,float hc_eps,
    cudaStream_t stream);
// Same outputs, with two completion kernels. The tail M floats of promoted_h
// are reused for normalization statistics after projection; scratch is opaque.
int ds41_fuse_hc_prepare_split(void *blas,const void *h,const float *fn,
    const float *scale,const float *base,const float *incoming_pre,
    const void *norm_weight,float *pre,float *post,float *comb,
    float *promoted_h,float *mixes,float *rstd,void *norm_bf16,float *norm_f32,
    void *fp8,void *fp8_scales,int M,int K,float norm_eps,float hc_eps,
    cudaStream_t stream);
// RMS -> BF16 -> interleaved tail RoPE with the decode M1 reduction tree.
int ds41_fuse_rms_rope_decode(const void *x,const void *weight,void *out,
    const float *frequencies,int M,int K,int rope_dim,float eps,int inverse,
    cudaStream_t stream);
#ifdef __cplusplus
}
#endif
