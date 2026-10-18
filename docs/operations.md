# Operations

## Configuration

Copy `config.example.json` to the ignored `config.local.json` and change what you need. Relative paths are resolved against the repo. Every script reads the same file.

```json
{
  "weights": "models/DeepSeek-V4.1-Flash-1017",
  "assets": "generated",
  "build": "build",
  "port": 18087,
  "unit": "ds41-pro6000",
  "model_id": "deepseek-v4.1-flash-1017",
  "host_gib": "auto",
  "gds": "auto",
  "max_context": 1048576,
  "max_tokens_limit": 131072,
  "max_queue": 16
}
```

- `weights`: the checkpoint directory. `scripts/download.py` downloads there. The engine assets record this path, so after moving the checkpoint, run `scripts/build.sh` again.
- `host_gib`: the pinned RAM tier in GiB. `auto` is your RAM minus 16 GiB, at most 104 (the measured value). At least 96: the placement seeds 5,480 experts (96.0 GiB) into the tier. Larger values are untested.
- `gds`: `auto` uses GPUDirect Storage when the NVIDIA driver has the P2P options from `gds/enable-p2p.sh` and CUDA's `libcufile.so.0` is present; `true` or `false` forces it ([GDS](gds.md)).
- `max_context`: the context limit, up to 1,048,576 tokens. It sets the RoPE rows the engine checks, not a memory reservation: compressed KV is about 890 bytes per token.
- `max_tokens_limit`: the largest `max_tokens` a request may ask for.

`python3 scripts/run.py command` prints the engine's environment and both command lines. Every flag and `DS41_*` variable in `scripts/run.py` is part of the measured profile, with a comment on what it does.

## Service lifecycle

```bash
python3 scripts/run.py start     # preflight, then a transient systemd user unit; waits for /health
python3 scripts/run.py status
python3 scripts/run.py logs
python3 scripts/run.py stop
```

Preflight checks for exactly one idle RTX PRO 6000, enough available RAM for the tier plus 4 GiB, cgroup v2 and a build whose assets match the configured checkpoint. The unit runs `scripts/run.py serve` with `MemoryMax` = tier + 12 GiB and `MemorySwapMax=0`; `serve` refuses to pin memory unless swap is off for its cgroup. It drops the checkpoint's page cache, starts `ds41d` on a Unix socket in `$XDG_RUNTIME_DIR/<unit>/`, waits for it to load (about a minute), then starts the API. If either process exits, the service stops.

To start at boot, create a regular user unit that runs `python3 /path/to/repo/scripts/run.py serve` with the same `MemoryMax`, `MemorySwapMax=0` and `KillMode=control-group`, and enable lingering (`loginctl enable-linger`).

## Memory

| Where | What | Size |
|---|---|---|
| VRAM | 4,452 expert slots | 78.0 GiB |
| VRAM | attention, shared experts, routers, head, MTP drafter dense weights, KV cache, prefill scratch | rest; about 93 GB in use |
| pinned RAM | expert tier | `host_gib` (104 GiB) |
| RAM | engine and API processes | about 2 GiB |
| page cache | checkpoint reads | reclaimable |

The cgroup's `memory.peak` includes page cache, so it reads higher than the service needs (111.4 GiB at peak on the reference machine, against 105.8 GiB of anonymous memory). Judge headroom by `anon` in `memory.stat`.

Distributions that run `systemd-oomd` act on sustained memory pressure in the user slice. Pinning 104 GiB with a full page cache creates that pressure, which is why `serve` drops the checkpoint's page cache first.

## Clients

The server takes OpenAI and Anthropic requests on loopback; use temperature 1.0 and top-p 1.0.

- **Reasoning effort.** `reasoning_effort`: `low`, `high` (default) or `max`; also a number from 1 to 100. `"thinking": false` turns thinking off. Anthropic `thinking: {"type": "enabled", "budget_tokens": N}` maps N ≤ 2,048 to `low`, ≤ 8,192 to `high` and more to `max`.
- **One conversation at a time.** There is one generation slot and a queue of 16, and the engine keeps the state of the latest conversation only. A client that interleaves two conversations (for example an agent's background title request) makes the other one prefill again. Turn title generation off in agent frontends where you can.
- **Context.** The engine serves up to 1M tokens. For long agent sessions we configure harnesses with a 512K window and let them compact before it, which keeps prompts fast.
- **Mid-conversation system messages** (which Claude Code sends) become the model's native system turns.

Example OpenCode provider:

```json
{
  "provider": {
    "ds41": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:18087/v1"},
      "models": {"deepseek-v4.1-flash-1017": {"limit": {"context": 524288, "output": 131072}}}
    }
  }
}
```

Anthropic-compatible clients use `http://127.0.0.1:18087` as their base URL; the server ignores API keys and accepts any model name.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `native runtime family mismatch: cudart` | The engine found a cudart other than CUDA 13.4.0's. Install the 13.4.0 toolkit and set `CUDA_HOME`; `run.py` puts `$CUDA_HOME/targets/x86_64-linux/lib` on the library path. |
| `auxiliary identity mismatch` | The assets, the checkpoint or the Engram library changed after the build. Run `scripts/build.sh` again. |
| `receipt` or mtime errors at load | A shard was touched or replaced. Run `scripts/build.sh` again; it rehashes the changed shards. |
| `libcufile` / P2P errors | GDS is on but not usable. Set `"gds": false` or see [GDS](gds.md). |
| The GPU is in use | Another process holds VRAM; the profile needs nearly all 96 GB. |
| The unit was killed while loading | Check `journalctl --user` for `systemd-oomd`; free RAM or lower `host_gib`. |

## API server

`api/` is a FastAPI server around the engine socket: `ds41_api.server --backend socket --socket PATH --tokenizer tokenizer.json`. `--backend fake` runs it without a GPU for client development. Its tests run on any machine:

```bash
cd api && DS41_TOKENIZER=../models/DeepSeek-V4.1-Flash-1017/tokenizer.json ../.venv/bin/python -m pytest -q
```

`api/contracts/backend-api.md` documents the engine socket protocol.
