// Vision tower + aligner for image prompts (official inference/vision.py and image_processor.py).
// Arithmetic follows the reference's BF16 module graph: every linear multiplies BF16 operands with FP32
// accumulation and rounds once (bias added in FP32 first), RMSNorm/RoPE/SiLU/GELU/softmax run in FP32 and
// round to BF16 where the reference's tensors are BF16. Attention is two cuBLAS GEMMs around an FP32
// softmax per head (and per query block for large images) instead of a fused kernel, so it sums in a
// different order than PyTorch's SDPA.
#include "ds41/vision.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
namespace ds41e {
namespace {
using Bf16 = __nv_bfloat16;
constexpr int D = 1024, HEADS = 16, HD = 64, INTER = 2816, LAYERS = 32, PATCH = 14, PDIM = 3 * PATCH * PATCH,
              LLM = 5120, ALIGN_IN = D * 9;
constexpr float NORM_EPS = 1e-6f, ROPE_THETA = 10000.f;
constexpr size_t ATTN_ELEMENTS = size_t(32) << 20; // S (FP32) and P (BF16) capacity per pass: 192 MiB

void ck(cudaError_t e, const char *what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("vision ") + what + ": " + cudaGetErrorString(e));
}
void ckb(cublasStatus_t s, const char *what) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("vision cuBLAS ") + what + ": " + std::to_string(int(s)));
}
size_t align(size_t n) { return (n + 255) & ~size_t(255); }

__device__ __forceinline__ float bf(Bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ Bf16 rn(float x) { return __float2bfloat16_rn(x); }

// Every thread returns the block-wide reduction. blockDim.x is a multiple of 32.
template <class Op> __device__ float block_reduce(float v, Op op) {
  __shared__ float shared[32];
  for (int o = 16; o; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffff, v, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  __syncthreads();
  if (!lane) shared[warp] = v;
  __syncthreads();
  v = lane < warps ? shared[lane] : Op::identity();
  for (int o = 16; o; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
struct Sum {
  __device__ float operator()(float a, float b) const { return __fadd_rn(a, b); }
  __device__ static float identity() { return 0.f; }
};
struct Max {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
  __device__ static float identity() { return -INFINITY; }
};

// load_image: float32 pixels / 255, then (x - 0.5) / 0.5, then BF16; patch features ordered (c, py, px).
__global__ void patchify(const uint8_t *__restrict__ px, Bf16 *__restrict__ out, int nw) {
  const int n = blockIdx.x, i = n / nw, j = n % nw, width = nw * PATCH;
  for (int f = threadIdx.x; f < PDIM; f += blockDim.x) {
    const int c = f / (PATCH * PATCH), r = f % (PATCH * PATCH), py = r / PATCH, pxl = r % PATCH;
    float x = __fdiv_rn(float(px[(size_t(i * PATCH + py) * width + (j * PATCH + pxl)) * 3 + c]), 255.f);
    x = __fdiv_rn(__fsub_rn(x, .5f), .5f);
    out[size_t(n) * PDIM + f] = rn(x);
  }
}
// out = BF16(acc + bias)
__global__ void bias_cast(const float *__restrict__ acc, const Bf16 *__restrict__ bias, Bf16 *__restrict__ out,
                          size_t total, int n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x)
    out[i] = rn(bias ? __fadd_rn(acc[i], bf(bias[i % n])) : acc[i]);
}
// x = BF16(x + BF16(acc + bias)): the reference rounds the linear's output before the residual add.
__global__ void bias_residual(const float *__restrict__ acc, const Bf16 *__restrict__ bias, Bf16 *__restrict__ x,
                              size_t total, int n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const float t = bf(rn(bias ? __fadd_rn(acc[i], bf(bias[i % n])) : acc[i]));
    x[i] = rn(__fadd_rn(bf(x[i]), t));
  }
}
// RMSNorm(dim 1024, eps 1e-6): BF16(w * (x * rsqrt(mean(x^2) + eps))), all FP32.
__global__ void rms_norm(const Bf16 *__restrict__ x, const Bf16 *__restrict__ w, Bf16 *__restrict__ y) {
  const Bf16 *row = x + size_t(blockIdx.x) * D;
  float v[D / 256], ss = 0;
#pragma unroll
  for (int k = 0; k < D / 256; ++k) {
    v[k] = bf(row[threadIdx.x + k * 256]);
    ss = __fadd_rn(ss, __fmul_rn(v[k], v[k]));
  }
  ss = block_reduce(ss, Sum());
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, float(D)), NORM_EPS));
#pragma unroll
  for (int k = 0; k < D / 256; ++k) {
    const int c = threadIdx.x + k * 256;
    y[size_t(blockIdx.x) * D + c] = rn(__fmul_rn(bf(w[c]), __fmul_rn(v[k], r)));
  }
}
// qkv [N, 3072] -> q, k rotated (2D RoPE: x1*cos - x2*sin, x2*cos + x1*sin over the two 32-wide halves) and
// v, each [16, N, 64].
__global__ void rope_split(const Bf16 *__restrict__ qkv, const float *__restrict__ cosv, const float *__restrict__ sinv,
                           Bf16 *__restrict__ q, Bf16 *__restrict__ k, Bf16 *__restrict__ v, int N) {
  const int n = blockIdx.x, h = threadIdx.x / 32, d = threadIdx.x % 32;
  const float c = cosv[n * 32 + d], s = sinv[n * 32 + d];
  const Bf16 *row = qkv + size_t(n) * 3 * D + h * HD;
  const size_t o = (size_t(h) * N + n) * HD;
  for (int part = 0; part < 2; ++part) {
    const Bf16 *src = row + part * D;
    Bf16 *dst = (part ? k : q) + o;
    const float x1 = bf(src[d]), x2 = bf(src[32 + d]);
    dst[d] = rn(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
    dst[32 + d] = rn(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
  }
  v[o + d] = row[2 * D + d];
  v[o + 32 + d] = row[2 * D + 32 + d];
}
// One row of already scaled scores -> BF16 probabilities.
__global__ void softmax_rows(const float *__restrict__ S, Bf16 *__restrict__ P, int cols) {
  const float *s = S + size_t(blockIdx.x) * cols;
  Bf16 *p = P + size_t(blockIdx.x) * cols;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, s[i]);
  m = block_reduce(m, Max());
  float sum = 0;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sum = __fadd_rn(sum, expf(__fsub_rn(s[i], m)));
  sum = block_reduce(sum, Sum());
  for (int i = threadIdx.x; i < cols; i += blockDim.x) p[i] = rn(__fdiv_rn(expf(__fsub_rn(s[i], m)), sum));
}
// MLP: h = BF16(x @ w1^T) split into gate | up; BF16(BF16(silu(gate)) * up).
__global__ void swiglu(const float *__restrict__ acc, Bf16 *__restrict__ out, size_t rows) {
  const size_t total = rows * INTER;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / INTER, c = i % INTER;
    const float g = bf(rn(acc[r * 2 * INTER + c])), u = bf(rn(acc[r * 2 * INTER + INTER + c]));
    const float s = bf(rn(__fdiv_rn(g, __fadd_rn(1.f, expf(-g)))));
    out[i] = rn(__fmul_rn(s, u));
  }
}
// Aligner input: zero-pad the [nh, nw] patch grid to multiples of 3 and unfold 3x3 blocks; features are
// ordered (channel, ki, kj) like F.unfold, blocks in reading order.
__global__ void unfold3(const Bf16 *__restrict__ x, Bf16 *__restrict__ u, int nh, int nw, int lw) {
  const int b = blockIdx.x, bi = b / lw, bj = b % lw;
  for (int f = threadIdx.x; f < ALIGN_IN; f += blockDim.x) {
    const int c = f / 9, kk = f % 9, r = bi * 3 + kk / 3, col = bj * 3 + kk % 3;
    u[size_t(b) * ALIGN_IN + f] = r < nh && col < nw ? x[(size_t(r) * nw + col) * D + c] : rn(0.f);
  }
}
// BF16(gelu(BF16(acc + bias))), exact (erf) GELU.
__global__ void bias_gelu(const float *__restrict__ acc, const Bf16 *__restrict__ bias, Bf16 *__restrict__ out,
                          size_t total, int n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const float x = bf(rn(__fadd_rn(acc[i], bf(bias[i % n]))));
    out[i] = rn(__fmul_rn(__fmul_rn(x, .5f), __fadd_rn(1.f, erff(__fmul_rn(x, 0.70710678118654752440f)))));
  }
}
int grid(size_t total) { return int(std::min<size_t>((total + 255) / 256, 65535)); }

struct Region { size_t offset = 0, bytes = 0; };
} // namespace

struct VisionEncoder::Weights {
  // Pinned host image of every vision tensor, grouped so one copy uploads a stage.
  void *host = nullptr;
  size_t total = 0;
  Region stem, blocks[LAYERS], aligner;
  // Byte offsets of each tensor inside its stage.
  size_t pe_w, pe_b, final_norm;
  size_t n1, qkv_w, qkv_b, wo_w, wo_b, n2, w1, w2;
  size_t a1_w, a1_b, a2_w, a2_b;
  ~Weights() { if (host) cudaFreeHost(host); }
};

VisionEncoder::VisionEncoder(const Store &source, cudaStream_t stream)
    : weights_(std::make_unique<Weights>()), stream_(stream) {
  auto &W = *weights_;
  struct Item { std::string name; std::vector<uint64_t> shape; Region *stage; size_t offset; };
  struct Spec { std::string name; std::vector<uint64_t> shape; size_t *slot; };
  std::vector<Item> items;
  // Each stage is one contiguous upload; every block has the same internal layout.
  auto stage = [&](Region &r, const std::vector<Spec> &tensors) {
    size_t at = 0;
    for (const auto &t : tensors) {
      *t.slot = at;
      items.push_back({t.name, t.shape, &r, at});
      size_t elements = 1;
      for (auto s : t.shape) elements *= s;
      at = align(at + elements * 2);
    }
    r.bytes = at;
  };
  stage(W.stem, {{"vision.patch_embed.proj.weight", {D, PDIM}, &W.pe_w},
                 {"vision.patch_embed.proj.bias", {D}, &W.pe_b},
                 {"vision.norm.weight", {D}, &W.final_norm}});
  for (int b = 0; b < LAYERS; ++b) {
    const std::string p = "vision.blocks." + std::to_string(b) + ".";
    stage(W.blocks[b], {{p + "norm1.weight", {D}, &W.n1},
                        {p + "attn.wqkv.weight", {3 * D, D}, &W.qkv_w},
                        {p + "attn.wqkv.bias", {3 * D}, &W.qkv_b},
                        {p + "attn.wo.weight", {D, D}, &W.wo_w},
                        {p + "attn.wo.bias", {D}, &W.wo_b},
                        {p + "norm2.weight", {D}, &W.n2},
                        {p + "mlp.w1.weight", {2 * INTER, D}, &W.w1},
                        {p + "mlp.w2.weight", {D, INTER}, &W.w2}});
  }
  stage(W.aligner, {{"aligner.w1.weight", {LLM, ALIGN_IN}, &W.a1_w},
                    {"aligner.w1.bias", {LLM}, &W.a1_b},
                    {"aligner.w2.weight", {LLM, LLM}, &W.a2_w},
                    {"aligner.w2.bias", {LLM}, &W.a2_b}});
  size_t at = 0;
  W.stem.offset = at; at += W.stem.bytes;
  for (auto &r : W.blocks) { r.offset = at; at += r.bytes; }
  W.aligner.offset = at; at += W.aligner.bytes;
  W.total = at;
  ck(cudaHostAlloc(&W.host, W.total, cudaHostAllocDefault), "pinned weights");
  for (const auto &item : items) {
    const auto &t = source.tensor(item.name);
    if (t.dtype != "BF16" || t.shape != item.shape)
      throw std::runtime_error("vision tensor dtype/shape mismatch: " + item.name);
    auto bytes = source.read(t);
    if (bytes.size() != t.bytes) throw std::runtime_error("vision tensor read: " + item.name);
    std::memcpy(static_cast<char *>(W.host) + item.stage->offset + item.offset, bytes.data(), bytes.size());
  }
  ckb(cublasCreate(&blas_), "create");
  ckb(cublasSetStream(blas_, stream_), "stream");
}
VisionEncoder::~VisionEncoder() {
  if (blas_) cublasDestroy(blas_);
}

size_t VisionEncoder::workspace_bytes(const ImageSpan &image) {
  const size_t N = size_t(image.n_vit_h) * image.n_vit_w, L = size_t(image.n_llm_h) * image.n_llm_w;
  size_t block = 0;
  // Largest stage: the aligner (two linears into 5120).
  block = align(size_t(LLM) * ALIGN_IN * 2) + align(size_t(LLM) * 2) * 2 + align(size_t(LLM) * LLM * 2);
  const size_t f32 = std::max({N * 2 * INTER, L * LLM, N * D}) * 4;
  return align(N * PDIM) + align(N * PDIM * 2) + 2 * align(N * D * 2) + align(N * 3 * D * 2) + 3 * align(N * D * 2) +
         align(N * D * 2) + align(f32) + align(N * INTER * 2) + 2 * align(N * 32 * 4) + align(L * ALIGN_IN * 2) +
         align(L * LLM * 2) + align(ATTN_ELEMENTS * 6) + block + (2 << 20) /* stem */ + align(L * LLM * 2) /* output */;
}

void *VisionEncoder::encode(const ImageSpan &image) {
  const auto started = std::chrono::steady_clock::now();
  auto &W = *weights_;
  const int nh = image.n_vit_h, nw = image.n_vit_w, N = nh * nw, lh = image.n_llm_h, lw = image.n_llm_w, L = lh * lw;
  if (nh < 1 || nw < 1 || lh != (nh + 2) / 3 || lw != (nw + 2) / 3 ||
      image.pixels.size() != size_t(nh) * PATCH * nw * PATCH * 3)
    throw std::invalid_argument("vision: image geometry/pixel count mismatch");
  // One arena per encode; nothing stays allocated except the returned rows.
  struct Arena {
    char *base = nullptr;
    ~Arena() { if (base) cudaFree(base); }
  } arena;
  std::vector<std::pair<char **, size_t>> plan;
  char *pixels, *patches, *x, *xn, *qkv, *q, *k, *v, *o, *f32, *act, *cosv, *sinv, *unf, *mid, *S, *P, *wstage, *stem;
  const size_t stage_bytes = std::max(W.aligner.bytes, W.blocks[0].bytes);
  const size_t f32_elems = std::max({size_t(N) * 2 * INTER, size_t(L) * LLM, size_t(N) * D});
  plan = {{&pixels, size_t(N) * PDIM}, {&patches, size_t(N) * PDIM * 2}, {&x, size_t(N) * D * 2}, {&xn, size_t(N) * D * 2},
          {&qkv, size_t(N) * 3 * D * 2}, {&q, size_t(N) * D * 2}, {&k, size_t(N) * D * 2}, {&v, size_t(N) * D * 2},
          {&o, size_t(N) * D * 2}, {&f32, f32_elems * 4}, {&act, size_t(N) * INTER * 2}, {&cosv, size_t(N) * 32 * 4},
          {&sinv, size_t(N) * 32 * 4}, {&unf, size_t(L) * ALIGN_IN * 2}, {&mid, size_t(L) * LLM * 2},
          {&S, ATTN_ELEMENTS * 4}, {&P, ATTN_ELEMENTS * 2}, {&wstage, stage_bytes}, {&stem, W.stem.bytes}};
  size_t total = 0;
  for (auto &item : plan) total += align(item.second);
  ck(cudaMalloc(&arena.base, total), "workspace (free VRAM too low for this image)");
  size_t at = 0;
  for (auto &item : plan) { *item.first = arena.base + at; at += align(item.second); }
  void *out = nullptr;
  ck(cudaMalloc(&out, size_t(L) * LLM * 2), "output rows");
  try {
    auto host = [&](const Region &r) { return static_cast<const char *>(W.host) + r.offset; };
    // 2D RoPE tables (get_vision_cos_sin): rope dim 32, inv_freq = 1 / theta^(i/32) for even i; the first 16
    // frequencies rotate by the patch row, the next 16 by its column.
    std::vector<float> cs(size_t(N) * 32), sn(size_t(N) * 32);
    float inv[16];
    for (int i = 0; i < 16; ++i) inv[i] = 1.f / std::pow(ROPE_THETA, float(2 * i) / 32.f);
    for (int r = 0; r < nh; ++r)
      for (int c = 0; c < nw; ++c)
        for (int i = 0; i < 32; ++i) {
          const float f = float(i < 16 ? r : c) * inv[i % 16];
          cs[(size_t(r) * nw + c) * 32 + i] = float(std::cos(double(f)));
          sn[(size_t(r) * nw + c) * 32 + i] = float(std::sin(double(f)));
        }
    ck(cudaMemcpyAsync(cosv, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice, stream_), "cos");
    ck(cudaMemcpyAsync(sinv, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice, stream_), "sin");
    ck(cudaMemcpyAsync(pixels, image.pixels.data(), image.pixels.size(), cudaMemcpyHostToDevice, stream_), "pixels");
    ck(cudaMemcpyAsync(stem, host(W.stem), W.stem.bytes, cudaMemcpyHostToDevice, stream_), "stem weights");
    auto gemm = [&](const void *X, const char *Wt, void *Y, cudaDataType yt, int M, int Nn, int K) {
      const float one = 1.f, zero = 0.f;
      ckb(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, Nn, M, K, &one, Wt, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y, yt,
                       Nn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "linear");
    };
    auto B = [](const char *p) { return reinterpret_cast<const Bf16 *>(p); };
    patchify<<<N, 256, 0, stream_>>>(reinterpret_cast<uint8_t *>(pixels), reinterpret_cast<Bf16 *>(patches), nw);
    gemm(patches, stem + W.pe_w, f32, CUDA_R_32F, N, D, PDIM);
    bias_cast<<<grid(size_t(N) * D), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), B(stem + W.pe_b),
                                                        reinterpret_cast<Bf16 *>(x), size_t(N) * D, D);
    // Attention blocking: whole heads when the N x N scores fit, else query blocks of one head.
    const size_t NN = size_t(N) * N;
    const int hb = NN <= ATTN_ELEMENTS ? int(std::min<size_t>(HEADS, ATTN_ELEMENTS / NN)) : 1;
    const int qb = NN <= ATTN_ELEMENTS ? N : int(std::max<size_t>(64, ATTN_ELEMENTS / N / 64 * 64));
    for (int b = 0; b < LAYERS; ++b) {
      ck(cudaMemcpyAsync(wstage, host(W.blocks[b]), W.blocks[b].bytes, cudaMemcpyHostToDevice, stream_), "block weights");
      rms_norm<<<N, 256, 0, stream_>>>(B(x), B(wstage + W.n1), reinterpret_cast<Bf16 *>(xn));
      gemm(xn, wstage + W.qkv_w, f32, CUDA_R_32F, N, 3 * D, D);
      bias_cast<<<grid(size_t(N) * 3 * D), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), B(wstage + W.qkv_b),
                                                             reinterpret_cast<Bf16 *>(qkv), size_t(N) * 3 * D, 3 * D);
      rope_split<<<N, 512, 0, stream_>>>(B(qkv), reinterpret_cast<float *>(cosv), reinterpret_cast<float *>(sinv),
                                          reinterpret_cast<Bf16 *>(q), reinterpret_cast<Bf16 *>(k),
                                          reinterpret_cast<Bf16 *>(v), N);
      for (int h0 = 0; h0 < HEADS; h0 += hb) {
        const int heads = std::min(hb, HEADS - h0);
        for (int q0 = 0; q0 < N; q0 += qb) {
          const int rows = std::min(qb, N - q0);
          const float scale = 0.125f, one = 1.f, zero = 0.f; // 1/sqrt(64)
          const long long head = (long long)N * HD, block = (long long)rows * N;
          ckb(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, HD, &scale,
                                         k + size_t(h0) * N * HD * 2, CUDA_R_16BF, HD, head,
                                         q + (size_t(h0) * N + q0) * HD * 2, CUDA_R_16BF, HD, head, &zero, S, CUDA_R_32F,
                                         N, block, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
              "scores");
          softmax_rows<<<heads * rows, 256, 0, stream_>>>(reinterpret_cast<float *>(S), reinterpret_cast<Bf16 *>(P), N);
          ckb(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_N, CUBLAS_OP_N, HD, rows, N, &one, v + size_t(h0) * N * HD * 2,
                                         CUDA_R_16BF, HD, head, P, CUDA_R_16BF, N, block, &zero,
                                         o + (size_t(q0) * D + size_t(h0) * HD) * 2, CUDA_R_16BF, D, HD, heads,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
              "values");
        }
      }
      gemm(o, wstage + W.wo_w, f32, CUDA_R_32F, N, D, D);
      bias_residual<<<grid(size_t(N) * D), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), B(wstage + W.wo_b),
                                                             reinterpret_cast<Bf16 *>(x), size_t(N) * D, D);
      rms_norm<<<N, 256, 0, stream_>>>(B(x), B(wstage + W.n2), reinterpret_cast<Bf16 *>(xn));
      gemm(xn, wstage + W.w1, f32, CUDA_R_32F, N, 2 * INTER, D);
      swiglu<<<grid(size_t(N) * INTER), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), reinterpret_cast<Bf16 *>(act),
                                                          size_t(N));
      gemm(act, wstage + W.w2, f32, CUDA_R_32F, N, D, INTER);
      bias_residual<<<grid(size_t(N) * D), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), nullptr,
                                                             reinterpret_cast<Bf16 *>(x), size_t(N) * D, D);
    }
    rms_norm<<<N, 256, 0, stream_>>>(B(x), B(stem + W.final_norm), reinterpret_cast<Bf16 *>(xn));
    unfold3<<<L, 256, 0, stream_>>>(B(xn), reinterpret_cast<Bf16 *>(unf), nh, nw, lw);
    ck(cudaMemcpyAsync(wstage, host(W.aligner), W.aligner.bytes, cudaMemcpyHostToDevice, stream_), "aligner weights");
    gemm(unf, wstage + W.a1_w, f32, CUDA_R_32F, L, LLM, ALIGN_IN);
    bias_gelu<<<grid(size_t(L) * LLM), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), B(wstage + W.a1_b),
                                                         reinterpret_cast<Bf16 *>(mid), size_t(L) * LLM, LLM);
    gemm(mid, wstage + W.a2_w, f32, CUDA_R_32F, L, LLM, LLM);
    bias_cast<<<grid(size_t(L) * LLM), 256, 0, stream_>>>(reinterpret_cast<float *>(f32), B(wstage + W.a2_b),
                                                         static_cast<Bf16 *>(out), size_t(L) * LLM, LLM);
    ck(cudaGetLastError(), "kernels");
    ck(cudaStreamSynchronize(stream_), "encode");
  } catch (...) {
    cudaStreamSynchronize(stream_);
    cudaFree(out);
    throw;
  }
  last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return out;
}
} // namespace ds41e
