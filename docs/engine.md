# The ds41d engine

`ds41d` serves DeepSeek-V4.1-Flash-1017 from its downloaded checkpoint on one RTX PRO 6000 (96 GB) with 128 GB of RAM. It is a single C++/CUDA process speaking a line-delimited JSON protocol on a Unix socket. `api/` (FastAPI) turns that into the OpenAI and Anthropic APIs, applying the model's official prompt encoding and parsing thinking and tool calls.

## The model

| | |
|---|---|
| layers | 40, hidden size 5,120 |
| routed experts | 384 per layer, 6 active per token, plus 1 shared expert |
| MTP layers | 3 (DSpark draft layers), 128 routed experts each, 3 active |
| expert weights | MXFP4 (FP4 values, one UE8M0 scale per 32), 18.8 MB per expert with its scales |
| dense weights | FP8, 32 x 32 blocks with UE8M0 scales |
| attention | 64 heads of 512 dimensions with one shared KV head: sliding-window layers (128 tokens) and compressed layers, whose KV blocks are chosen by an indexer (top 512) |
| Engram | n-gram embedding tables for layers 1 and 14, read by token hash |
| context | 1,048,576 tokens (YaRN, factor 16) |

The checkpoint is 510 GB, of which 296 GB are the 15,744 routed experts (15,360 in the 40 layers, 384 in the MTP layers). Compressed KV takes about 890 bytes per token: a full 1M-token context needs under 1 GiB.

## Expert residency

Every token routes to 6 experts in each of 40 layers, plus 3 in each MTP layer when drafting. Experts live in three tiers:

| Tier | Size | Experts | Policy |
|---|---|---:|---|
| VRAM | 4,452 slots, 77.99 GiB | 28% | 3,240 pinned by `profiles/p80-mtp48.placement`, the rest an LFU cache with per-layer budgets (48 for each MTP layer) |
| pinned RAM | 104 GiB | about 38% | LFU (recent window 2, counts decayed every 256 steps), seeded with the placement's 5,480 `host` experts; exclusive of VRAM: an expert evicted from VRAM is demoted to RAM |
| NVMe | the rest | about 34% | read on demand in 2 MiB `O_DIRECT` chunks; with GDS, straight into the VRAM slot through cuFile P2PDMA |

The placement ranks experts by routing frequency over prompts from many sessions. It pins 3,240 experts and leaves the other 1,212 slots to the dynamic cache; pinning half as many, or none, measured slower.

A RAM hit costs about 0.17 ms per expert and an NVMe miss about 1.19 ms. Host DRAM (DDR5-3600, about 48 GB/s) is shared by GPU uploads, NVMe writes into RAM and the CPU; GDS P2P keeps NVMe reads out of it. Under concurrent uploads it cut an NVMe miss from 2.76 to 1.69 ms and raised upload bandwidth from 34 to 45 GB/s.

## A decode step

1. **Draft.** The three MTP layers propose up to 4 tokens. Before drafting token *k*, the engine estimates its acceptance from the drafter's top probability and its confidence head (a logistic model, `DS41_SPARK_STOP_MODEL`) and stops when the expected number of extra accepted tokens no longer pays for another verify row (`DS41_SPARK_STOP_Q`, scaled per position by `DS41_SPARK_STOP_SCALE`). The stop decision uses only information available before token *k* is drawn, so it cannot bias the output.
2. **Verify.** The target runs all rows layer by layer. Each layer's routed experts are gathered once for every row, uploaded or read once, and multiplied in batched MXFP4 kernels.
3. **Accept.** Rejection sampling at the request's temperature and top-p accepts a prefix of the draft and samples the next token from the target's residual distribution. The output distribution is the target's own; with the same seed, the tokens do not depend on cache state, the RAM tier size or GDS.
4. **Prefetch.** While layer *l* runs, layer *l+1*'s router is scored on layer *l*'s input and the predicted RAM-resident experts start uploading (`--prefetch-lookahead`; top 4 per row for the first 2 rows). A prediction only moves data: routing uses the real router.

The step is transfer-bound: the GPU is busy for about 22 ms of an 80 ms verify step, and a layer costs about 0.158 + 0.171 x (RAM hits) + 1.191 x (NVMe misses) ms. Decode steps are replayed as CUDA graphs.

## Prompts

- **Batched prefill.** 256 rows per dense and attention launch, 8 rows per FP8 projection launch. Experts for a chunk are read once for all of its rows.
- **Segments.** Long prompts split into segments that fit the VRAM the expert cache leaves free, so the 4,452 slots never shrink for a prompt. A cold 32K-token prompt takes 33 seconds.
- **Prefix reuse.** The engine keeps the state of the latest conversation. A request that extends it, whatever its conversation ID (`--prefix-cache --prefix-any-conversation`), prefills only its new tokens: a follow-up turn on an 8K conversation starts answering in 0.5 s.
- **Warm RAM.** After a prompt of 256+ tokens, the experts its last 256 rows routed to are read from NVMe into RAM (up to 16 GiB), so the answer starts with them close.
- **Drafter restore.** After a long appended turn, the drafter restarts from the last 128 rows of the append.

## Quality

The target computes in the checkpoint's own formats: MXFP4 experts, FP8 dense weights, BF16 head. Nothing is requantized. It is not bit-identical to DeepSeek's reference implementation: batched prefill and the compact FP8 `wo_a` kernel add in a different order, which shifts logits slightly and occasionally flips an MoE route. The official reference differs from itself between its prefill and decode layouts by about the same amount.

The project's quality gate on the measured build ([`benchmarks/quality-gate-20261018.json`](../benchmarks/quality-gate-20261018.json); run with smaller caches, which do not change outputs):

- **Against the reference.** A 16-token greedy continuation agreed with the reference at all 16 positions, with a mean per-position KL divergence of 0.015 (max 0.093). On a 64-position chat the largest KL was 0.51, and the greedy continuation diverged from the reference's after 10 tokens.
- **Exactness of the serving machinery.** Bit for bit: verify widths 2 and 6 against width 1; earlier rows unaffected by later draft rows (causality); greedy output with and without DSpark drafting; repeated runs, seeded T=1 runs and runs after a restart.
- **Recorded as failures, by design.** The served path's log-probabilities differ from teacher-forced scoring at 2 of 48 positions, and an appended turn differs from pure decoding. Both compare batched prefill with one-row decode, which sum in different orders.
- **Tasks.** Tool calls, GSM8K and HumanEval samples pass. That is a smoke test, not an accuracy evaluation.

Speculative decoding is exact: see *Accept* above. The fused LM head (`libds41_fuse_head`) computes the BF16 head bit for bit like the pinned cuBLAS head.

`--allow-unqualified` overrides a guard from the engine's first native-forward milestone, which refuses to serve until a full-model "gold" fidelity check is recorded. That check needs reference outputs the project does not have for agentic prompts (`fidelity_gold: SKIP` in the gate summary), so the measured service runs with the flag, as does this repo.

## How the speed was found

Measured during development with `scripts/benchmark.py`'s method on the six-prompt set (raw data for the last row only, in `benchmarks/`):

| Version | Decode (tok/s) | Cold 8K prompt | Follow-up on 8K | What changed |
|---|---:|---:|---:|---|
| v1 | ~21 | 112 s | 114 s | native forward, VRAM + RAM + NVMe expert tiers |
| v2 | ~26 | 14.4 s | 0.85 s | batched prefill, prefix reuse, chunked NVMe uploads, drift-aware cache policy, fused head, parallel exact sampler |
| v3 | ~27.4 | | | DSpark speculative decoding with the checkpoint's MTP layers |
| v4, v4.1 | 29.5, 29.9 | 13.9 s | 0.71 s | adaptive draft length, compact FP8 `wo_a`, MTP-layer VRAM budget, multi-row lookahead prefetch, drafter restore after long appends, chunk-aware prefill segments, batched zero-copy uploads |
| v4.1, 104 GiB RAM tier | **31.4** | 13.5 s | 0.71 s | RAM tier 96 → 104 GiB: 11% fewer NVMe reads, identical tokens |

Rows are means over 3 or 4 seeds; a change that alters the sampled trajectory varies about ±10% per prompt, so single runs mislead. This repo's build measures 31.6 tok/s on seeds 21 and 22, where the measured service measured 31.7 with identical tokens ([benchmarks](benchmarks.md)).

Measured and not adopted: a two-lane verify, a fused expert library (bitwise, no gain), pinning 50% or 0% of the placement, lookahead over more than 2 rows, 100 to 150 more VRAM slots (+1.6% decode, but 24K prompts then needed two prefill segments), wider drafts, and stricter stop rules than the fitted one.

The largest remaining gains are in hardware: more RAM (96 to 104 GiB of tier gave 3.3%), a second Gen5 NVMe drive, faster DRAM. In software, the drafter's fixed cost of about 4 ms per step is the next target.

## Source layout

| Path | What |
|---|---|
| `engine/` | `ds41d`: checkpoint store and receipt checks, native forward, DSpark drafter (`third_party/spark`), batched prefill, socket server |
| `components/moe` | expert store: VRAM slots, RAM tier, NVMe reader (`O_DIRECT` and cuFile), MXFP4 kernels |
| `components/dense` | FP8 dense projections and the Engram store |
| `components/attn` | sliding-window, compressed and indexer attention kernels (built against TileLang's headers) |
| `components/fuse-head` | the fused BF16 LM head |
| `components/spark-policy` | the draft policy sidecar |
| `api/` | the HTTP server; `ds41_api/encoding.py` is DeepSeek's official encoding, unchanged |

Each tree is extracted unchanged from the commit named in `sources.lock.json`, apart from the build files listed there. Code comments mention the work streams that wrote each component (`sol-moe`, `sol-attn` and so on).

### Reproducing the binaries

`scripts/build.sh` uses the same compiler flags as the measured service. A build from a fresh clone matches the service's 11 runtime files in every kernel's instructions (`cuobjdump -sass`), the host code (`objdump`) and the exported symbols, once build paths are normalized ([`benchmarks/clean-clone-20261018/binary-comparison.json`](../benchmarks/clean-clone-20261018/binary-comparison.json); `scripts/compare_binaries.py` compares any two builds). The bytes differ where paths are recorded: the names nvcc gives anonymous namespaces, `-lineinfo` data and `assert` messages. `auxiliary.json` authenticates the Engram library by SHA-256, so `scripts/build.sh` binds the assets to the library it built.

`engine/tests/*.py` are the development harness's reference checks; they need the project's reference environment, which is not part of this repo. The CTest suite that `scripts/build.sh` runs is self-contained.
