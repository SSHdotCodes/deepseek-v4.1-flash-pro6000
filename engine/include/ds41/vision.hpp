#pragma once
#include "ds41/engine.hpp"
#include "ds41/native_store.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
namespace ds41e {
// The checkpoint's vision tower and aligner (official inference/vision.py): a 32-block ViT (dim 1024, 16
// heads, 2D RoPE, full attention over one image) and a 3x3 unfold + two linears into the LLM's 5120-dim
// embedding space. Weights stay in pinned host memory (970 MB, loaded at the first image) and stream to
// the GPU one block at a time, so an encode borrows about 1 GB of VRAM and returns it.
class VisionEncoder {
public:
  VisionEncoder(const Store &source, cudaStream_t stream);
  ~VisionEncoder();
  VisionEncoder(const VisionEncoder &) = delete;
  VisionEncoder &operator=(const VisionEncoder &) = delete;
  // BF16 [n_llm_h * n_llm_w, 5120] aligner rows in reading order, in a cudaMalloc'd buffer the caller frees
  // with cudaFree. Enqueued on the constructor's stream; synchronizes before returning.
  void *encode(const ImageSpan &image);
  // Smallest free VRAM an encode of this image needs.
  static size_t workspace_bytes(const ImageSpan &image);
  double last_ms() const { return last_ms_; }

private:
  struct Weights;
  std::unique_ptr<Weights> weights_;
  cudaStream_t stream_;
  cublasHandle_t blas_ = nullptr;
  double last_ms_ = 0;
};
} // namespace ds41e
