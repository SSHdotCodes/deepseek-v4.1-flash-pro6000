#ifndef DS41_ATTENTION_H
#define DS41_ATTENTION_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Device pointers throughout; BF16 uses uint16 storage. stream is cudaStream_t.
 * Functions return cudaError_t as int; asynchronous errors require stream sync.
 * Quantizers accept finite BF16 inputs. No projection, norm, or RoPE here. */
enum ds41_kv_format { DS41_SWA_FP8=0, DS41_GLOBAL_FP4=1, DS41_INDEX_FP4=2 };
/* AoS records: 528=(512+16), 288=(256+32), 68=(64+4) bytes.
 * Payload first, scales second. FP4 low nibble first. No extra row padding. */
int ds41_attn_pack(const void* x, void* records, int rows, int format, void* stream);
int ds41_attn_unpack(const void* records, void* x, int rows, int format, void* stream);
/* In-place BF16 roundtrip; cols multiple of32. format0 FP8, format1 FP4/16,
 * format2 FP4/32. Primarily attention output and index Q. */
int ds41_attn_roundtrip(void* x, int rows, int cols, int format, void* stream);
/* Raw ratio2 pooling: row-addressed FP32 values/scores in ring>=8. Write all
 * verify rows before call; each completed pair reads absolute positions. Output
 * has rows entries, odd positions emit pooled BF16, even positions zero. */
int ds41_attn_pool2(const float* kv_ring, const float* score_ring, void* pooled,
                   int64_t start_pos, int rows, int ring_capacity, void* stream);
/* Cold-prefill direct pairs [rows,512] -> [floor(rows/2),512]. The caller
 * retains an odd trailing raw row for the next chunk. */
int ds41_attn_pool2_prefill(const float* kv, const float* score, void* pooled,
                          int rows, void* stream);
/* Absolute positions in [start,start+rows) stored at pos%capacity. Capacity
 * >=128+max_verify_width protects accepted predecessors against rejected writes.
 * Rollback only rewinds committed length; future rows are invisible. */
int ds41_attn_ring_write(const void* records, void* ring, int bytes_per_row,
                        int64_t start, int rows, int capacity, void* stream);
int ds41_attn_ring_write_at(const void* records, void* ring, int bytes_per_row,
                           const int64_t* start, int rows, int capacity, void* stream);
int ds41_attn_ring_read_at(const void* ring, void* records, int bytes_per_row,
                          const int64_t* start, int rows, int capacity, void* stream);
/* top_idx[p,k] is global entry index, -1 masked; query position start+i.
 * Window always 128 slots, leading -1 while filling, then oldest valid first,
 * followed by k globals. Cold prefill has a separate trailing-mask entry point.
 * ring capacity>=134; comp_len per query derives from position and ratio.
 * gather outputs BF16[p,128+k,512], idx[p,128+k] local or -1. */
int ds41_attn_gather(const void* swa_ring, int capacity, const void* global_kv,
                    int global_capacity, const int32_t* top_idx, int k,
                    int64_t start, int rows, int ratio, void* gathered,
                    int32_t* local_idx, void* stream);
/* Graph-friendly counterpart: start is a stable device int64 scalar. */
int ds41_attn_gather_at(const void* swa_ring, int capacity, const void* global_kv,
                       int global_capacity, const int32_t* top_idx, int k,
                       const int64_t* start, int rows, int ratio, void* gathered,
                       int32_t* local_idx, void* stream);
/* Full-prefill reference uses trailing masked slots for its early windows.
 * width=min(full_prompt_length,128), different from cached decode leading masks. */
int ds41_attn_gather_prefill_at(const void* swa, int capacity, const void* global_kv,
                              int global_capacity, const int32_t* top_idx, int k,
                              const int64_t* start, int rows, int ratio, int width,
                              void* gathered, int32_t* local_idx, void* stream);
int ds41_attn_lengths(const int64_t* start, int32_t* ratio1_lengths,
                     int32_t* ratio2_lengths, int rows, int max_context, void* stream);
/* q/o BF16[p,64,512]; gathered BF16[p,topk,512]; local_idx[p,topk].
 * tile_heads=4,8,16 controls independent head grouping. Reference 64-entry tiles,
 * online FP32 softmax, BF16 P, FP32 sink denominator, BF16 result. */
int ds41_attn_sparse(const void* q, const void* gathered, const float* sink,
                    const int32_t* local_idx, void* o, int rows, int topk,
                    int tile_heads, void* stream);
/* Optional two-stage exact QK tiles + unchanged serial softmax/PV. scores is
 * caller-owned FP32[rows,64,topk] scratch. Both symbols pass independent frozen
 * fixtures. Recommended for global decode/verify rows1..6 with tile_heads16. */
int ds41_attn_sparse_split(const void* q, const void* gathered, const float* sink,
                          const int32_t* local_idx, void* o, float* scores,
                          int rows, int topk, int tile_heads, void* stream);
/* Index q BF16[p,32,128] already MXFP4 roundtripped. k records68B, weights
 * BF16[p,32] includes 1/64 scaling. out BF16[p,nscan]. key_ids may be NULL
 * for full scan; otherwise int32[p,nscan] absolute compressed indices.
 * visible[p] counts complete groups. Output masked with -inf. */
int ds41_attn_index_scores(const void* q, const void* key_records,
                           const void* weights, const int32_t* key_ids,
                           const int32_t* visible, void* out, int rows,
                           int nscan, void* stream);
/* score topk (k<=2048), output sorted ascending original row index.
 * lower-index tie selection qualified against pinned BF16 CUDA torch fixtures.
 * This does not establish FP32 router top-k ordering.
 * scratch queried below; output width k, slots beyond visible become -1. */
size_t ds41_attn_topk_scratch(int rows, int n);
int ds41_attn_topk(const void* scores, const int32_t* visible, int32_t* out,
                   int rows, int n, int k, void* scratch, void* stream);
/* Block8 candidate max with newest visible block pinned +inf. For the following
 * top-k call, block visibility must be ceil(visible_positions/8), not total blocks. */
int ds41_attn_block_scores(const void* scores, const int32_t* visible,
                           void* out, int rows, int n, void* stream);
/* Sorted block ids -> sorted compressed positions, then compact topk remap. */
int ds41_attn_expand_blocks(const int32_t* blocks, int32_t* ids, int rows,
                            int nblocks, int npositions, void* stream);
int ds41_attn_remap_topk(const int32_t* local, const int32_t* ids, int32_t* out,
                        int rows, int k, int nscan, void* stream);
#ifdef __cplusplus
}
#endif
#endif
