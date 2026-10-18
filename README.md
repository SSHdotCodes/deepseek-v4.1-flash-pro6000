# DeepSeek-V4.1-Flash-1017 on one RTX PRO 6000

**31.6 tok/s decode at native precision on one 96 GB card with 128 GB of RAM.** The model is DeepSeek's [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): a 510 GB checkpoint with native MXFP4 experts and FP8 dense weights. It runs here unconverted and unquantized, with:

- a 1,048,576-token context;
- thinking with `low` / `high` / `max` reasoning effort;
- tool calls;
- speculative decoding with the checkpoint's own MTP layers;
- OpenAI Chat Completions, Anthropic Messages and OpenAI Responses APIs.

`ds41d` is a purpose-built engine. 296 GB of the checkpoint are routed experts, three times the card's memory. Each MoE layer's 384 experts are split three ways: the most used stay in VRAM, the next tier in pinned host RAM, and the rest are read from NVMe on demand, straight into VRAM with GPUDirect Storage.

| temperature 1.0, top-p 1.0, thinking `high` | |
|---|---:|
| decode, mean of 3 short chats, a 2.3K-token multi-turn follow-up and two 8K-token agent turns | **31.6 tok/s** |
| decode per request | 22–38 tok/s |
| time to first token, cold 8K-token prompt | 13.5 s |
| time to first token, cold 24K / 29K-token prompt | 23.1 s / 24.6 s |
| follow-up turn on an 8K-token conversation (prefix reuse) | 0.7 s |
| greedy agreement with DeepSeek's reference implementation | 16/16 prompts, mean KL 0.015 |

Decode rates include thinking tokens and exclude prompt processing. Short chats decode slowest (28 tok/s) and long agent turns fastest (up to 38), because the drafts are accepted more often. A clean build from this repo returned the same tokens as the measured service on all 14 benchmark requests, at the same speed (31.62 against 31.67 tok/s). [Benchmarks](docs/benchmarks.md).

## What you need

- **GPU:** one RTX PRO 6000 Blackwell 96 GB, otherwise idle. The engine uses about 93 GB of VRAM.
- **RAM:** 128 GB. The service pins 104 GiB of host memory for experts; by default the tier is your total RAM minus 16 GiB, at most 104 GiB. 96 GiB measured 3.3% slower than 104 GiB; smaller tiers work but read more from NVMe.
- **Disk:** 520 GB on a fast NVMe drive for the checkpoint, read during decode (an expert is 18.8 MB). Measured on one PCIe 5.0 drive with ext4. Another 10 GB for the Python environments and the build.
- **Software:**
  - Linux with cgroup v2 and `systemctl --user`. The service runs with `MemorySwapMax=0`.
  - NVIDIA driver 615 (measured with 615.71.09).
  - **CUDA Toolkit 13.4.0** exactly. The engine checks that it runs with that release's `libcudart.so.13.4.92` and with the cuBLAS 13.1.1.3 pair the build downloads.
  - GCC 13 or newer (measured with 15.2), CMake 3.24 or newer.
  - [uv](https://docs.astral.sh/uv/) or Python 3.12.
- **Recommended:** GPUDirect Storage P2P, so NVMe reads skip host RAM. It takes one modprobe file and a reboot ([docs/gds.md](docs/gds.md)). Without it, reads go through a host bounce buffer.

## Install and run

```bash
git clone https://github.com/SSHdotCodes/deepseek-v4.1-flash-pro6000.git
cd deepseek-v4.1-flash-pro6000

# .venv (API, downloads, benchmarks) and .venv-assets (PyTorch 2.14, to export the RoPE and Engram tables).
scripts/setup.sh

# 510 GB at the pinned revision into models/; checks every file's size, and small files' hashes.
.venv/bin/python scripts/download.py

# Pinned cuBLAS and TileLang wheels (SHA-256 checked), the components, the engine assets (hashes all 510 GB
# against the Hub's LFS hashes once, about 6 minutes), ds41d and its tests. Output: build/dist.
scripts/build.sh

# Preflight (idle GPU, RAM, build), then a systemd user service with no swap. Loads in about a minute.
python3 scripts/run.py start
```

The server is at **http://127.0.0.1:18087/v1**, model ID **`deepseek-v4.1-flash-1017`**, loopback only. Other actions:

- `python3 scripts/run.py logs`: show the logs.
- `python3 scripts/run.py stop`: unload the model.
- `python3 scripts/run.py command`: print the exact engine and API command lines.

To change the checkpoint location, port, unit name, RAM tier or GDS mode, copy `config.example.json` to `config.local.json` ([operations](docs/operations.md)).

```bash
curl http://127.0.0.1:18087/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash-1017", "messages": [{"role": "user", "content": "Explain an LRU cache."}],
  "max_tokens": 4096, "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"}'
```

Port 18087 serves:

- OpenAI chat completions, with tools, streaming and reasoning split out (`reasoning_content`);
- Anthropic Messages, at `/v1/messages` and `/v1/messages/count_tokens`, with thinking blocks and tool use;
- OpenAI Responses, at `/v1/responses`.

Defaults:

- Thinking is on, at `high`. `reasoning_effort` takes `low`, `high` or `max` (the encoding's 50, 75 and 100); `"thinking": false` answers directly. Anthropic `thinking.budget_tokens` maps to the same three levels.
- Sampling is DeepSeek's recommendation for this model: temperature 1.0, top-p 1.0.
- A request may ask for up to 131,072 output tokens.
- One request runs at a time; up to 16 wait in a queue. A request that extends an earlier conversation prefills only its new tokens.

## Reproduce the measurements

```bash
python3 scripts/run.py stop
python3 scripts/benchmark.py --seeds 21 22 --out results/bench.json     # about 6 minutes
python3 scripts/run.py start
.venv/bin/python scripts/api_check.py --out results/api-check.json     # API checks and prompt timings
```

`benchmark.py` starts its own engine with the service's exact command and runs the six fixed prompts in `benchmarks/cases`, plus a warm-up, once per seed. It records time to first token, decode rate and a SHA-256 of every token sequence. Sampling is seeded and exact, so on the same GPU, driver and build the tokens should equal [ours](benchmarks/results-20261018.json); the script reports how many match. `api_check.py` checks every API surface (chat, streaming, reasoning effort, tool calls and round trips, Anthropic messages and tools, `count_tokens`, Responses) and times cold 2K, 8K and 32K-token prompts and a follow-up turn. GitHub CI runs the repository tests, the API tests and script checks; CUDA and model checks need the GPU.

## How it works

- **Three expert tiers.** Every routed expert is 18.8 MB of MXFP4 weights and scales, 15,744 of them across 40 MoE layers and the 3 MTP layers.
  - VRAM holds 4,452 of them (78 GiB). 3,240 are pinned by a placement ranked by routing frequency; the rest are a per-layer LFU cache.
  - The RAM tier holds about 5,900 more in 104 GiB of pinned memory, also LFU, and exclusive of VRAM: an expert evicted from VRAM moves to RAM.
  - The remaining third is read from NVMe when routed, in 2 MiB chunks, straight into its VRAM slot through cuFile.
- **Speculative decoding (DSpark).** The checkpoint's three MTP layers draft up to four tokens, and the target verifies them in one pass that loads each layer's experts once for all rows. Before each draft token the engine estimates its acceptance from the drafter's confidence and stops when another row would not pay for itself. Rejection sampling keeps the output distribution the target's own.
- **Prefetching.** While layer *l* runs, layer *l+1*'s router is scored on its input and the predicted RAM-resident experts start copying to VRAM. After a long prompt, the experts it used are read from NVMe into RAM for the answer.
- **Prompts.** Prefill runs 256 rows per launch in segments sized to the VRAM the expert cache leaves free, and reuses any earlier conversation's prefix.

Decode is bound by expert transfers, not compute: the GPU is busy for about 22 ms of an 80 ms verify step. Each layer costs about 0.16 ms plus 0.17 ms per expert from RAM and 1.19 ms per expert from NVMe. More RAM, a second NVMe drive or faster DRAM would help most. [Engine details](docs/engine.md), [asset pipeline](docs/assets.md).

## Credits and licenses

Thanks to **DeepSeek** for DeepSeek-V4.1-Flash-1017, its encoding and its reference implementation, and to the **TileLang** and **NVIDIA CUTLASS / cuBLAS / cuFile** teams.

Repo code is MIT. The model, its prompt encoding and inference code are MIT; no weights are included in this repository. The benchmark prompts come from this project, WildChat-1M (ODC-BY 1.0) and betterwright-agent-traces (Apache-2.0). See [THIRD_PARTY.md](THIRD_PARTY.md).
