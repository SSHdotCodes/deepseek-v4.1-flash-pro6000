# Benchmarks

Every number on this page has its raw data in [`benchmarks/`](../benchmarks/), except where marked. All are from one machine:

- RTX PRO 6000 Blackwell Workstation Edition (96 GB, 580 W limit), driver 615.71.09, CUDA 13.4.0;
- AMD Ryzen 9 9950X3D, 128 GB DDR5-3600 (four DIMMs), 120.7 GiB visible to Linux;
- one PCIe 5.0 NVMe drive with ext4, GPUDirect Storage P2P enabled (`gds/enable-p2p.sh`);
- Ubuntu 26.04.1, kernel 7.0.0-38.

## Method

`scripts/benchmark.py` starts `ds41d` with the service's exact command (`scripts/run.py command`) and sends the prompts in `benchmarks/cases` over its socket, in this order, once per seed:

| # | Prompt | Tokens | Output |
|---|---|---:|---:|
| 0 | `agentic-001`: warm-up right after loading, not counted | 8,029 | 64 |
| 1 | `chat-000`: explain how a transformer processes a sentence | 74 | 256 |
| 2 | `agentic-000`: browser-agent turn (system prompt, tools, history) | 7,999 | 256 |
| 3 | `mt-000`: a multi-turn conversation, follow-up question | 2,335 | 256 |
| 4 | `chat-001`: what happens when you type a URL | 73 | 256 |
| 5 | `chat-002`: mixture-of-experts versus dense layers | 77 | 256 |
| 6 | `agentic-001`: browser-agent turn | 8,029 | 256 |

The prompts are pre-encoded with the official encoding, thinking `high`, temperature 1.0, top-p 1.0. Decode rate is (tokens − 1) / (last token time − first token time), so it includes thinking tokens and excludes prompt processing. The headline is the mean over prompts 1–6 and the seeds. The engine stays up across the prompts of a seed, as in the service, so caches carry over.

A change that alters which tokens are sampled moves single prompts by up to ±10%, so compare means over several seeds. A change that keeps the tokens (a RAM tier size, GDS, a rebuild) can be compared prompt by prompt, and the token SHA-256 values prove the tokens stayed the same.

## Source build

Seeds 21 and 22: this repo's `scripts/build.sh` output against the measured service's binaries, one after the other ([`results-20261018.json`](../benchmarks/results-20261018.json)):

| Prompt | seed 21 | seed 22 | service, seed 21 | service, seed 22 | draft acceptance |
|---|---:|---:|---:|---:|---:|
| chat-000 | 27.99 | 28.36 | 28.11 | 28.41 | 0.87, 0.90 |
| agentic-000 | 31.51 | 35.25 | 31.47 | 35.34 | 0.92, 0.83 |
| mt-000 | 27.34 | 30.77 | 27.33 | 30.77 | 0.86, 0.91 |
| chat-001 | 34.14 | 32.57 | 34.25 | 32.54 | 0.88, 0.83 |
| chat-002 | 38.01 | 35.59 | 38.03 | 35.80 | 0.89, 0.83 |
| agentic-001 | 28.19 | 29.75 | 28.17 | 29.85 | 0.82, 0.91 |
| **mean (tok/s)** | **31.62** | | **31.67** | | |

All 14 token sequences, the warm-ups included, are identical between the two builds. Run through `scripts/benchmark.py` and `scripts/run.py`, the source build measured 31.62 tok/s again, with the same 14 token hashes ([`benchmark-py-20261018.json`](../benchmarks/benchmark-py-20261018.json)).

Time to first token: 2.0–2.3 s for the 73–77-token chats, 9.4 s for the 2,335-token follow-up, 13.3–14.4 s for the 8K-token agent turns. The warm-up decodes at 22–24 tok/s: its 64 tokens start on a cold expert cache.

## RAM tier

The measured service with a 104 GiB and a 96 GiB RAM tier, seeds 11 to 14 interleaved:

| | 104 GiB | 96 GiB |
|---|---:|---:|
| decode, mean of 24 prompts (tok/s) | **31.37** | 30.38 |
| per seed | 31.90, 31.18, 29.87, 32.55 | 30.84, 30.15, 28.94, 31.59 |
| experts read from NVMe | 38,929 | 43,940 |
| experts uploaded from RAM | 159,928 | 154,195 |

All 24 paired prompts were faster with 104 GiB (+1.4% to +5.0%), with identical tokens. Through systemd, the 104 GiB service peaked at 105.8 GiB of anonymous memory, and the system kept at least 5.5 GiB available during 8K–29K-token prompts (memory log not included).

## Clean clone

A fresh clone of this repo, following the README on the same machine (`benchmarks/clean-clone-20261018/`):

- **Build.** `scripts/build.sh` took 7 minutes, 6 of them hashing the 510 GB checkpoint; all 7 engine tests passed. A second run took 38 seconds: unchanged shards are not hashed again ([`run.log`](../benchmarks/clean-clone-20261018/run.log)).
- **Binaries.** Compared with the measured service's 11 runtime files, every kernel's instructions (`cuobjdump -sass`), the host code (`objdump`) and the exported symbols are identical once build paths are normalized ([`binary-comparison.json`](../benchmarks/clean-clone-20261018/binary-comparison.json)). The bytes differ where paths are recorded: nvcc's anonymous-namespace names, `-lineinfo` and `assert` messages. The service's two fused-head kernel libraries also carry an unused sm_120 PTX copy.
- **Decode.** `scripts/benchmark.py --seeds 21 22`: 31.61 tok/s, all 14 token sequences identical to the reference run.
- **GDS.** The same run with `"gds": false`: 30.79 tok/s with identical tokens. All 12 paired prompts were faster with P2P (+0.4% to +5.0%, median +2.8%).
- **API.** `scripts/api_check.py`: 15 of 15 checks passed.

| Through the API, thinking off, up to 64 output tokens | Prompt tokens | Time to first token | Prompt rate |
|---|---:|---:|---:|
| cold prompt | 2,047 | 9.0 s | 228 tok/s |
| cold prompt | 8,192 | 13.3 s | 615 tok/s |
| follow-up turn on that conversation (8,255 tokens reused) | 8,268 | 0.48 s | |
| cold prompt | 32,765 | 33.4 s | 983 tok/s |

The cold prompts are slices of this repo's own C++ and CUDA sources, so nothing is reused between them. On this machine, time to first token is roughly 6.5 seconds plus 0.8 ms per prompt token.
