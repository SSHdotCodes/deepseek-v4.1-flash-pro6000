# ds41_api

The HTTP front end for `ds41d`: OpenAI Chat Completions, Anthropic Messages (with `count_tokens`) and OpenAI Responses on one port. It encodes requests with DeepSeek's official V4.1 encoding (`ds41_api/encoding.py`, unchanged from the model repository), streams tokens from the engine's Unix socket, and parses thinking and DSML tool calls back out.

`scripts/run.py` starts it; to run it alone:

```bash
../.venv/bin/python -m ds41_api.server --backend socket --socket "$XDG_RUNTIME_DIR/ds41-pro6000/engine.sock" \
  --tokenizer ../models/DeepSeek-V4.1-Flash-1017/tokenizer.json --port 18087 --max-tokens-limit 131072
```

`--backend fake` serves scripted output without an engine, for client development and the tests:

```bash
DS41_TOKENIZER=../models/DeepSeek-V4.1-Flash-1017/tokenizer.json ../.venv/bin/python -m pytest -q
```

Without `DS41_TOKENIZER` (or the downloaded checkpoint) the tests fall back to a character tokenizer, and the few that need the real one fail. `contracts/backend-api.md` documents the engine's socket protocol.
