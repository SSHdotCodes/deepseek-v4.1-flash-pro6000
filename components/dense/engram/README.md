# Native Engram store (C ABI v1)

`include/ds41/engram.h` is the public interface. `libds41_engram.so` contains a host
hash/I/O/cache service and a separately callable CUDA graph-compatible row dequantizer.
The checkpoint, its safetensors index, the engine's official hash metadata export,
and sol-ref's complete matching integrity receipt are inputs. No model repack,
conversion, buffered-I/O fallback, or network access is involved.

## Lifetime and use

1. Zero-initialize `ds41_engram_config`, set `struct_bytes`, `abi_version=1`, the four
   paths, `queue_depth=48`, `max_batch_rows=4096`, `cache_rows=65536`, and
   `pin_memory=1`. Open once. The cache is 17,301,504 bytes plus its LRU/map metadata.
   Direct-I/O staging is bounded at 64 MiB for batch4096. Unpinned mode is useful
   for CPU-only hashing/fixtures. Pinned mode requires a valid CUDA runtime and
   must be opened outside graph capture, under the team's heavy lock.
2. Create one hash state per live sequence. `hash` accepts arbitrary chunk sizes,
   maintaining full compressed-token history. `start_pos` may append or overwrite;
   overwrite discards the suffix, so a later draft cannot see stale future tokens.
   A false token mask stores DEAD and prevents any n-gram from crossing that token.
   The exact official normalization, primes, offsets and RNG multipliers come
   from `engine/generated/engram.json`, exported by unchanged official `engram.py`.
3. When token IDs are known, hash immediately, then submit layer1 before layer14.
   Hash output is `[tokens,2,24]`; submit takes a contiguous array of IDs for one
   layer. A request is also a prefetch: it fills the shared cache independently
   of whether its output is consumed. Fork the history with `state_clone` for
   speculative tokens; truncate/replace the committed history only on acceptance.
4. Work can proceed while a ticket is pending. `ready` polls without blocking;
   `wait` is a host wait. After success, `data` returns the immutable requested
   rows in their original order, including duplicates. Each row is exactly 264
   bytes: FP8[256], then E8M0[8]. The ticket owns that memory. `pin_memory=1`
   makes it suitable for async H2D; retain the ticket until its copy event finishes.
5. Copy interleaved rows into a stable device buffer and call
   `ds41_engram_dequant(raw, bf16, rows, stream)`. This call allocates nothing and
   is graph-capturable. The 24 rows flatten to the official 6144-wide wkv input.
   Alternatively, `copy` splits native values/scales into caller-owned host arrays.
6. Free tickets, then close the store. Close drains queued work. A ticket can be
   freed while pending, but that operation waits. Public store methods are safe
   from multiple host threads; a hash state belongs to one caller at a time.
   Concurrent close with other public calls is unsupported. Cache clear requires
   an idle store. Stats are cumulative and synchronized.

## Prefill and resource limits

Each request deduplicates all rows globally, then sorts unique misses for locality,
shares 4 KiB pages within each bounded batch, and reads with O_DIRECT/io_uring at
configured QD (48 by default). Completion order does not change logical row order.
Hot rows use a bounded LRU cache keyed by layer plus logical row ID. Duplicate rows
within a request read once even when the request exceeds cache capacity.

Large prefill requests are supported up to 16,777,216 rows per ticket. Output costs
264 bytes per requested row, plus ID/dedup metadata; two 150K-token tickets cost
about 1.9 GB of pinned output. The consumer can submit smaller token chunks to bound
its output retention. Global dedup then applies per chunk; the bounded cache carries
reuse across chunks. Avoid submitting unbounded draft/prefill tickets ahead of an
urgent decode: one store has one FIFO worker, and running requests are not preempted.
Use a separate store with a small cache if independent concurrent requests require
latency isolation. No numerical state depends on cache residency or queue order.

Errors return -1, with `ds41_engram_error()` on the calling thread. Worker errors
propagate through ready/wait/data/copy. All I/O is read-only; unsupported direct I/O
and partial/corrupt reads fail explicitly. Open validates tensor dtype/shape/range,
file size+mtime against the complete externally SHA-verified receipt, and hash layout.
It does not redundantly rehash 203 GB at startup.

## Reproduce

Run the native test behind `~/ai/ds41/bin/heavy.sh` using `ref/python.sh`:

```sh
ref/python.sh dense/engram/tests/qualify.py --lib dense/build-r2c/libds41_engram.so --output dense/results/engram-final.json --long
```

The CPU-only `--pin 0 --hash-only` path loads metadata/headers only and creates no
CUDA context. It compares official hashes for monolithic/dead-token, irregular chunk,
and draft-clone/rollback cases. The full run compares raw rows with the independent
native safetensors reader, checks cache hits do no disk I/O, and measures decode1..6
and prefill8K..150K. Prefill inputs are real agentic token replay for I/O distribution,
not a full model prefill or TTFT benchmark.
