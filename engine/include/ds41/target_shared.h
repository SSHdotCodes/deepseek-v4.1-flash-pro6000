#ifndef DS41_TARGET_SHARED_H
#define DS41_TARGET_SHARED_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed device pointers owned by the target engine instance. Never free,
 * resize, write, or retain them beyond that instance's lifetime. CUDA stream
 * is explicit: no default-stream work or hidden synchronization. */
typedef struct ds41_target_weights_v2 {
  uint32_t abi_version;
  uint32_t struct_bytes;
  int32_t vocab_size, hidden_size;
  int64_t embedding_stride_elements, head_stride_elements;
  const void *embedding_bf16; /* [129280,5120], native BF16 */
  const float *head_f32;      /* [129280,5120], native BF16 promoted exactly */
  void *cuda_stream;
  void *blas_handle; /* Borrowed cuBLAS handle, serialized with target work. */
  const float *rope_f32; /* Exact theta10000 table [rope_rows,64], real/imag. */
  int64_t rope_rows;
} ds41_target_weights_v2;
/* Matches spark-api.md. Routed experts only; output is BF16[rows,3,5120]
 * in selected top-k slot order. Never includes shared experts or reduction.
 * All callbacks return zero on success, never throw across this C boundary.
 * A failure poisons the target instance. error() is borrowed until next call.
 */
typedef int (*ds41_target_experts_fn)(void *user, int stage, const void *x_bf16,
                                      const int32_t *ids, const float *weights,
                                      void *out_bf16, int rows, void *stream);
typedef int (*ds41_target_rollback_fn)(void *user, int64_t committed_inputs,
                                       void *stream);
typedef struct ds41_target_services_v1 {
  uint32_t abi_version;
  void *user;
  ds41_target_experts_fn experts;
  ds41_target_rollback_fn rollback;
  const char *(*error)(void *user);
} ds41_target_services_v1;
/* BF16 concatenation of mean_hc(INPUT of layer37), layer38, layer39. Storage
 * is stable; CONTENTS are overwritten by the next verify(). Enqueue ingest of
 * committed_rows on cuda_stream before that next verify. Never ingest rejects.
 */
typedef struct ds41_target_capture_v1 {
  uint32_t abi_version;
  int32_t rows, columns, committed_rows;
  int64_t begin_position;
  size_t row_stride_bytes;
  const void *hidden_bf16; /* [rows,15360], capacity six rows */
  void *cuda_stream;
} ds41_target_capture_v1;
typedef int (*ds41_target_capture_sink)(void *user,
                                        const ds41_target_capture_v1 *capture);
enum ds41_target_query_kind {
  DS41_TARGET_WEIGHTS = 0,
  DS41_TARGET_SERVICES = 1,
  DS41_TARGET_CAPTURES = 2
};
/* The supported component boundary is this size/version-negotiated C query.
 * target is an opaque handle supplied by the engine, valid for its lifetime.
 * Unknown versions or unequal output sizes return -1 WITHOUT writing output.
 * Unavailable resources return -2. Never use the internal C++ vtable as an ABI.
 * WEIGHTS requires version2; SERVICES and CAPTURES require version1. */
int ds41_target_query(void *target, uint32_t kind, uint32_t version,
                      void *output, size_t output_bytes);
enum ds41_target_dtype {
  DS41_TARGET_BF16 = 0,
  DS41_TARGET_F32 = 1,
  DS41_TARGET_FP8 = 2,
  DS41_TARGET_U8 = 3
};
typedef struct ds41_target_tensor_v1 {
  uint32_t abi_version, struct_bytes;
  int32_t dtype;
  int64_t rows, columns;
  const void *data;
} ds41_target_tensor_v1;
/* Named borrowed views use the official normalized names, matching spark_bind.
 * One-dimensional tensors are [elements,1]. Returns -2 for unavailable names.
 * Visit with prefix="mtp." for nonexpert MTP bindings; shared embed/head and
 * exact theta10000 "rope" are queried separately. Enable --load-mtp at init.
 * name/view passed to a visitor are valid only during that call; data remains
 * borrowed for the target lifetime. The visitor must copy the name/descriptor.
 */
int ds41_target_tensor(void *target, const char *name, uint32_t version,
                       void *output, size_t output_bytes);
typedef int (*ds41_target_tensor_visitor)(void *user, const char *name,
                                          const ds41_target_tensor_v1 *view);
int ds41_target_visit_tensors(void *target, const char *prefix,
                              uint32_t version,
                              ds41_target_tensor_visitor visitor, void *user);
#ifdef __cplusplus
}
#endif
#endif
