# 3-bit experts (EXL3 K3)

With `"experts": "k3"` in `config.local.json`, the 15,360 routed experts of the 40 target layers run from a bank of 3-bit [EXL3](https://github.com/turboderp-org/exllamav3) weights instead of the checkpoint's MXFP4. Everything else stays native: the FP8 dense layers, the shared experts, the attention, the head and the DSpark drafter with its 384 MTP-layer experts.

An expert shrinks from 18.8 to 13.3 MB. The same VRAM then holds 5,745 target experts instead of 4,308, and the RAM tier and NVMe have proportionally less to move, which is what decode waits on. On the benchmark prompts decode runs at 49.1 instead of 31.7 tok/s (1.55×), and an 8K-token prompt reaches its first token in 10.7 instead of 13.3 seconds. The cost is a small loss of fidelity: perplexity is 2.1% higher on the model's own outputs, task accuracy did not change measurably, and on code the model thinks about 15% longer.

Native MXFP4 remains the default and the reference: all of the repo's other numbers are for it.

## Use it

```bash
# 191 GiB into models/k3-bank (default "k3_bank"), about 35 minutes at 100 MB/s; resumable.
.venv/bin/python scripts/k3_bank.py fetch models/k3-bank
# in config.local.json: "experts": "k3"
python3 scripts/run.py start
```

`k3_bank.py fetch` reads only the expert tensors it needs from two public EXL3 quants with HTTP range reads and writes them straight into the bank layout. Each finished layer must match a SHA-256 pinned in [`scripts/assets/k3-bank.json`](../scripts/assets/k3-bank.json), so every download is byte-identical to the measured bank. It needs 191 GiB of free space next to the checkpoint's 510 GB, ideally on the same fast NVMe drive, because decode reads the bank on demand. `ds41d` refuses to start if a bank file's size or modification time differs from its receipt; after copying or moving the bank, run `scripts/k3_bank.py verify DIR`, which recomputes every digest and stamps new receipts.

## Results

### Speed

`scripts/benchmark.py --seeds 21 22` on a fresh clone of this repo, both profiles one after the other on the same machine ([`results-k3-20261023.json`](../benchmarks/results-k3-20261023.json), [`bench-native.json`](../benchmarks/k3-20261023/bench-native.json), [`run.log`](../benchmarks/k3-20261023/run.log)):

| Prompt | native, seed 21 | seed 22 | K3, seed 21 | seed 22 | K3 draft acceptance |
|---|---:|---:|---:|---:|---:|
| chat-000 | 28.04 | 28.39 | 43.57 | 44.06 | 0.84, 0.88 |
| agentic-000 | 31.51 | 35.28 | 51.74 | 52.70 | 0.84, 0.86 |
| mt-000 | 27.38 | 30.69 | 45.73 | 47.31 | 0.80, 0.88 |
| chat-001 | 34.34 | 32.50 | 48.81 | 50.79 | 0.80, 0.82 |
| chat-002 | 38.03 | 35.67 | 51.72 | 52.93 | 0.81, 0.93 |
| agentic-001 | 28.20 | 29.77 | 49.84 | 50.34 | 0.93, 0.94 |
| **mean (tok/s)** | **31.65** | | **49.13** | | |

Every prompt is faster on both seeds, by 36% to 77%. Over the 14 requests the K3 run read 8,729 experts from NVMe instead of 21,594 and uploaded 60,866 from RAM instead of 85,234. Draft acceptance averaged 0.86 against 0.87: the drafter still predicts the native model. The native run's 14 token sequences equal the published reference ([`results-20261018.json`](../benchmarks/results-20261018.json)), so the change leaves native decoding as it was; with `"experts": "k3"`, `benchmark.py` compares the tokens with `results-k3-20261023.json` instead.

Through the API (`scripts/api_check.py`, 15 of 15 checks passed; [`api-check.json`](../benchmarks/k3-20261023/api-check.json)), against the native clean-clone run of 2026-10-18 ([benchmarks](benchmarks.md#clean-clone)):

| Thinking off, up to 64 output tokens | Prompt tokens | native | K3 |
|---|---:|---:|---:|
| cold prompt | 2,047 | 9.0 s | 6.5 s |
| cold prompt | 8,192 | 13.3 s | 10.7 s |
| follow-up turn on that conversation (8,255 tokens reused) | 8,268 | 0.48 s | 0.47 s |
| cold prompt | 32,765 | 33.4 s | 28.4 s |

The image checks (`scripts/vision_e2e.py`) passed 7 of 7 on K3 ([`vision-e2e.json`](../benchmarks/k3-20261023/vision-e2e.json)).

### Fidelity

Both engines scored the same token sequences teacher-forced: the native build's prompts and its own sampled continuations from the benchmark cases (`score` operation, verify width 6; [`kl-native-vs-k3.json`](../benchmarks/k3-20261023/kl-native-vs-k3.json)).

| Positions | n | KL(native ‖ K3) | top-1 agreement | perplexity, K3 / native |
|---|---:|---:|---:|---:|
| the model's own outputs | 1,503 | 0.035 | 93.3% | 1.021 |
| prompts | 1,754 | 0.36 | 82.0% | 0.935 |

The prompt rows are dominated by agent tool output (JSON, page dumps), where the distributions are flat and chaotic: in such a window, the reference study below found a KL of 0.76 between the native computation and an exact FP32 recompute of the same experts. On a natural multi-turn English prompt the KL was 0.014. The generated rows are the meaningful number, and slightly pessimistic for K3, because those tokens were sampled from the native model.

A reference-implementation study before the build measured the same thing over 27,700 positions (its own thinking samples and three 4K-token windows of prose, code and agent traffic): all-K3 experts raised perplexity by 1.8% on chat and 0.9% on hard reasoning, with top-1 agreement of 93.8% and 94.4% against a 95.4–96.0% noise floor.

### Tasks and output length

Through the API, thinking on, temperature 1.0, top-p 1.0, seed 42, up to 8,192 output tokens per request ([`tasks.json`](../benchmarks/k3-20261023/tasks.json); harness not included):

| | native | K3 |
|---|---:|---:|
| GSM8K, first 100 | 99 | 100 |
| HumanEval, first 82, pass@1 | 76 (92.7%) | 81 (98.8%) |
| tool-call checks (44 that the API accepts) | 38 | 40 |

GSM8K is scored leniently (the reference answer appears in the reply's last 300 characters); the harness's strict extraction scored 91 and 86 because it often takes the wrong number from a correct answer. Of the 9 items where the strict scores disagreed, both builds answered 8 correctly; in the ninth, native answered 12 where the reference is 8. At temperature 1.0, differences of this size are within sampling noise: the measurement shows no loss, not a gain.

K3 against native, paired by item: geometric mean of the per-request ratios with a 95% bootstrap interval. HumanEval/32 hits the 8,192-token limit in both builds and is left out.

| | output tokens | decode time per answer | median decode rate, K3 / native |
|---|---:|---:|---:|
| GSM8K | 1.01 (0.95–1.07) | 0.69 (0.65–0.74) | 69.7 / 48.8 tok/s |
| HumanEval | 1.15 (1.02–1.29) | 0.77 (0.69–0.87) | 62.2 / 41.7 tok/s |
| tool calls | 1.08 (0.96–1.22) | 0.63 (0.54–0.76) | 62.0 / 37.1 tok/s |

On code, K3 thinks measurably longer, and an answer still takes about a quarter less decode time.

## The bank

Sixteen public 1017 EXL3 quants were compared by relative output error against the native expert, on 64 experts in each of 8 layers with real routed activations. The native FP8-activation path itself has 4.0%. The best K3 tensors were [LibertAIDAI's 2.9 bpw quant](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-exl3-2.9bpw) at 18.0% (others: 18.1–18.5%). An uncalibrated quant measured 38.9%, K2 29–41%, and K4 8–12% while saving only 6% of the bytes. The 2.9 bpw quant stores layers 18–22 at K2, so those five layers come from the same author's [3.5 bpw quant](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-exl3-3.5bpw), whose experts there are K3. Both are MIT-licensed and pinned to a revision in `scripts/assets/k3-bank.json`.

`bank-LL.bin` holds layer LL's 384 experts in order, one 13,316,096-byte record each: for w1 (gate), w3 (up) and w2 (down), the trellis (int16, `[k/16, n/16, 48]`), then FP16 `suh[k]` and `svh[n]`, then zero padding to a 4 KiB multiple. `records-k3.tsv` is the store's ordinary manifest over these files: the gate/up/down views point at the trellises, the scale views at `suh` (with `svh` right after). `receipts.json` holds each file's SHA-256, size and modification time.

## How ds41d runs it

- **Two stores.** `--exl3-experts records-k3.tsv` makes the main expert store (VRAM slots, RAM tier, NVMe reads) hold only the 15,360 K3 records. A store sizes every slot by its largest record, so mixing the formats would give the saving back; they live in separate stores. `--experts` still names the native manifest, and its 384 MTP-layer records go into a second store with every expert pinned in VRAM (7.2 GB). The drafter's math is unchanged, and it never misses.
- **Placement.** `profiles/k3-p80.placement` keeps the native placement's 3,240 VRAM pins and 5,480 RAM seeds and scales the target layers' dynamic budgets from 1,068 to 2,505 slots. 5,745 × 13.3 MB plus the drafter store is the native profile's expert VRAM. With the 104 GiB RAM tier, about 92% of the target experts are resident.
- **Kernels.** `components/exl3` implements `grouped_exl3` (decode and verify, 1–8 rows per expert) and `grouped_exl3_prefill` (up to 256 rows) behind the same job and device-table interface as the MXFP4 kernels, including the store's ready flags. A CTA scales its input rows by `suh` and rotates them with the 128-point Hadamard transform, decodes the trellis to FP16 with exllamav3's mul1 codebook, accumulates in FP32 MMA, then applies the output rotation and `svh` in FP32 and rounds once to BF16. The tile shape and the reduction order never depend on the row count, the job count or the row's slot, so results are batch-invariant and deterministic, and the launches can be captured in CUDA graphs. EXL3 takes BF16 activations, so the expert paths gather and activate BF16 rows instead of quantizing them to FP8.
- **Paths.** Width-1 decode, batched and streamed verify, and prefill (including experts staged from RAM or NVMe) all switch to these kernels. Without the flag, `ds41d` runs exactly as before: on the benchmark cases it returns the same tokens as the native build.

## Kernel checks

Against FP32 matrix products with the exactly reconstructed weights, on 48 real experts from layers 3, 20 and 38, every projection, every row count (1–8 decode, 1–256 prefill): relative error at most 0.20% (mean 0.19%), no NaN or Inf. Bit for bit: every row count against the full-size result, the same row in every slot, reordered rows and jobs, 1–4 jobs of mixed sizes, every prefill bound, CUDA graph replays with changed expert ids and table contents, and readiness waits. Compute Sanitizer memcheck: no errors.

Cold-weight timings (384 distinct experts, more than the L2 cache), against the MXFP4 kernels: decode and verify jobs take 0.91–0.93× the time, prefill 1.10–1.22×. Prefill computes more per byte, but prompts still start sooner end to end, because less is read from RAM and NVMe.

## Limits

- Measured on one machine (see [benchmarks](benchmarks.md)); the bank's speed gain depends on how much of the native profile's time goes to RAM and NVMe transfers.
- The drafter still predicts the native model: draft acceptance is about 1 point lower. A drafter tuned to the K3 target could recover that.
- The placement's pins come from the native profile; they were not re-tuned for the larger slot count.
