#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define DS41_ENGRAM_ABI_VERSION 1
#define DS41_ENGRAM_ROW_BYTES 264
#define DS41_ENGRAM_HASH_COLUMNS 24
#define DS41_ENGRAM_LAYERS 2
typedef struct ds41_engram_store ds41_engram_store;
typedef struct ds41_engram_state ds41_engram_state;
typedef struct ds41_engram_ticket ds41_engram_ticket;
// Fixed ABI: set struct_bytes=sizeof(config), abi_version=1. Strings copied on open.
typedef struct {
 uint32_t struct_bytes,abi_version;
 const char *weight_root,*index_json,*hash_json,*integrity_json;
 uint64_t cache_rows; // default 65536 (16.5 MiB packed rows), zero disables cache
 uint32_t queue_depth,max_batch_rows,pin_memory,reserved; // default QD48,batch4096
} ds41_engram_config;
typedef struct {
 uint64_t submitted_rows,cache_hits,unique_misses,disk_pages,disk_bytes,batches;
 uint64_t cache_capacity_rows,cache_live_rows;
 double io_ms,total_ms;
} ds41_engram_stats;
// Host APIs return 0 success, -1 failure; thread-local error string. No fallback from direct I/O.
const char *ds41_engram_error(void);
int ds41_engram_open(const ds41_engram_config*,ds41_engram_store**);
void ds41_engram_close(ds41_engram_store*); // drains worker, all tickets must be released first
int ds41_engram_get_stats(ds41_engram_store*,ds41_engram_stats*);
int ds41_engram_clear_cache(ds41_engram_store*); // requires idle store
int ds41_engram_state_create(ds41_engram_store*,ds41_engram_state**);
int ds41_engram_state_clone(const ds41_engram_state*,ds41_engram_state**); // drafts fork history
void ds41_engram_state_free(ds41_engram_state*);
int ds41_engram_state_truncate(ds41_engram_state*,uint64_t start_pos);
// Sequential append or replacement at start_pos<=history length. Replacement truncates suffix.
// mask=null means all text. A zero mask blocks all lookback at/before the dead token.
// rows layout [count,2,24], layer ordering [1,14]. History survives arbitrary chunk boundaries.
int ds41_engram_hash(ds41_engram_state*,const int32_t* tokens,const uint8_t* mask,
 uint64_t start_pos,size_t count,int64_t* rows);
// Submit immediately when tokens are known, before layer1. Rows copied on submission.
// Requests are deduplicated, served in bounded chunks, and cached by (layer,row).
// Row bytes are native FP8[256] followed by native E8M0[8], no dequantization.
int ds41_engram_submit(ds41_engram_store*,int layer,const int64_t* rows,size_t count,
 ds41_engram_ticket**);
int ds41_engram_ready(ds41_engram_ticket*); // 0 pending,1 done,-1 failed
int ds41_engram_wait(ds41_engram_ticket*); // blocks host only; not a CUDA-graph operation
int ds41_engram_data(ds41_engram_ticket*,const void** data,size_t* count); // after ready/wait
void ds41_engram_ticket_free(ds41_engram_ticket*); // waits if pending; data dies here
// Copy interleaved ticket rows to caller arrays, after wait. Handy for stable graph input buffers.
int ds41_engram_copy(ds41_engram_ticket*,void* fp8_rows,void* e8m0_scales);
// CUDA graph-capturable: interleaved native rows on device -> BF16 [rows,256].
// No allocation, host synchronization, I/O, or default stream. Stream passed as opaque cudaStream_t.
int ds41_engram_dequant(const void* interleaved,void* bf16_rows,int rows,void* stream);
#ifdef __cplusplus
}
#endif
