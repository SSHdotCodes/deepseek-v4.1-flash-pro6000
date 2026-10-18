#include "ds41/ops.cuh"
#include <cuda_fp8.h>
namespace ds41e {
namespace {
__global__ void cast_up(const Bf16 *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __bfloat162float(x[i]);
}
__global__ void cast_down(const float *x, Bf16 *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __float2bfloat16_rn(x[i]);
}
__global__ void embed(const Bf16 *x, Bf16 *h, float *pre, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k * 4)
    h[i] = x[i % k];
  if (i < 4)
    pre[i] = i == 0 ? 1.f : 0.f;
}
__global__ void scale(Bf16 *x, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(x[i]), factor));
}
// TEMP: contiguous single-row PyTorch mean reduction geometry. Kept explicit
// until the component ABI exports this exact hc_mixes stage and fixture gate.
__global__ void mixnorm(const Bf16 *h, float *mix, int k) {
  __shared__ float sh[512];
  int t = threadIdx.x;
  float v[4] = {0, 0, 0, 0};
  for (int i = t * 4; i < k; i += 2048)
    for (int j = 0; j < 4; ++j)
      if (i + j < k) {
        float x = __bfloat162float(h[i + j]);
        v[j] = __fadd_rn(v[j], __fmul_rn(x, x));
      }
  float z = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[1]), v[2]), v[3]);
  sh[t] = z;
  __syncthreads();
  for (int d = 256; d >= 32; d >>= 1) {
    if (t < d) {
      z = __fadd_rn(z, sh[t + d]);
      sh[t] = z;
    }
    __syncthreads();
  }
  if (t < 32) {
    for (int d = 16; d; d >>= 1)
      z = __fadd_rn(z, __shfl_down_sync(0xffffffff, z, d));
    if (!t)
      sh[0] = rsqrtf(__fadd_rn(__fmul_rn(z, 1.f / k), 1e-20f));
  }
  __syncthreads();
  if (t < 24)
    mix[t] = __fmul_rn(mix[t], sh[0]);
}
__global__ void route(const float *s, const int *ids, float *w, int k) {
  // T=1 torch Reduce.cuh: block width last_pow2(K), vt0=4, descending
  // shuffles. K=3 uses two lanes; K=6 uses four lanes. The addition order
  // matters even though all router scores are positive.
  if (threadIdx.x)
    return;
  float v[4] = {0, 0, 0, 0};
  for (int i = 0; i < k; ++i)
    v[i % 4] = __fadd_rn(v[i % 4], s[ids[i]]);
  float sum = k == 3 ? __fadd_rn(__fadd_rn(v[0], v[2]), v[1])
                     : __fadd_rn(__fadd_rn(v[0], v[2]), __fadd_rn(v[1], v[3]));
  sum = __fadd_rn(sum, 1e-20f);
  for (int i = 0; i < k; ++i)
    w[i] = __fmul_rn(s[ids[i]] / sum, 1.5f);
}
__global__ void rows(const uint8_t *values, const uint8_t *scales, Bf16 *out,
                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 256)
    return;
  uint8_t raw = values[i];
  __nv_fp8_e4m3 f;
  f.__x = raw;
  float x = float(f);
  uint8_t exponent = scales[i / 32];
  float s = __uint_as_float(exponent ? uint32_t(exponent) << 23 : 0x00400000u);
  out[i] = __float2bfloat16_rn(__fmul_rn(x, s));
}
__global__ void capture(const Bf16 *h, Bf16 *out, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k)
    return;
  float sum =
      __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f, __bfloat162float(h[i])),
                                    __bfloat162float(h[k + i])),
                          __bfloat162float(h[2 * k + i])),
                __bfloat162float(h[3 * k + i]));
  out[i] = __float2bfloat16_rn(__fmul_rn(sum, .25f));
}
} // namespace
void bf16_float(const void *x, float *y, int n, cudaStream_t s) {
  cast_up<<<(n + 255) / 256, 256, 0, s>>>((const Bf16 *)x, y, n);
}
void float_bf16(const float *x, void *y, int n, cudaStream_t s) {
  cast_down<<<(n + 255) / 256, 256, 0, s>>>(x, (Bf16 *)y, n);
}
void expand_embedding(const void *x, void *h, float *p, int k, cudaStream_t s) {
  embed<<<(k * 4 + 255) / 256, 256, 0, s>>>((const Bf16 *)x, (Bf16 *)h, p, k);
}
void scale_bf16(void *x, int n, float factor, cudaStream_t s) {
  scale<<<(n + 255) / 256, 256, 0, s>>>((Bf16 *)x, n, factor);
}
void normalize_mixes(const void *h, float *m, int k, cudaStream_t s) {
  mixnorm<<<1, 512, 0, s>>>((const Bf16 *)h, m, k);
}
void route_weights(const float *s, const int *ids, float *w, int k,
                   cudaStream_t stream) {
  route<<<1, 32, 0, stream>>>(s, ids, w, k);
}
void engram_dequant(const uint8_t *v, const uint8_t *s, void *out, int n,
                    cudaStream_t stream) {
  rows<<<n, 256, 0, stream>>>(v, s, (Bf16 *)out, n);
}
void capture_mean(const void *h, void *out, int k, cudaStream_t s) {
  capture<<<(k + 255) / 256, 256, 0, s>>>((const Bf16 *)h, (Bf16 *)out, k);
}
} // namespace ds41e
extern "C" int ds41_engine_route_weights(const float *scores, const int *ids,
                                         float *out, int k, void *stream) {
  ds41e::route_weights(scores, ids, out, k, (cudaStream_t)stream);
  return int(cudaGetLastError());
}
extern "C" int ds41_engine_capture_mean(const void *h, void *out, int k,
                                        void *stream) {
  ds41e::capture_mean(h, out, k, (cudaStream_t)stream);
  return int(cudaGetLastError());
}
extern "C" int ds41_engine_engram_dequant(const uint8_t *values,
                                          const uint8_t *scales, void *out,
                                          int rows, void *stream) {
  ds41e::engram_dequant(values, scales, out, rows, (cudaStream_t)stream);
  return int(cudaGetLastError());
}
