# Backend API contract (DS41) — v1

Owner: glm-api. Consumers: api (this contract's client), the inference engine teams (llama / vllm / moe / ref).
Python reference implementation: `~/ai/ds41/api/ds41_api/backend.py` (ABC + event dataclasses),
`socket_backend.py` (Unix-socket JSON-lines transport), `fake_backend.py` (test double).
Run `python -m ds41_api.contract_check` for a machine-checkable summary of the wire types.

## Scope

The API server owns: OpenAI HTTP surface, chat encoding (prompt string + token ids via the
official `encoding/encoding.py`), streaming message parsing, tool-call/reasoning extraction,
usage accounting. The engine owns: model forward, sampling, KV cache reuse.

The hand-off is **token ids in, token-id events out**. The engine never sees text or OpenAI
JSON; the server never touches weights.

## Types

```python
SamplingParams:
    temperature: float = 1.0        # recommended sampling: 1.0
    top_p: float = 1.0              # recommended sampling: 1.0
    max_tokens: int = 4096          # hard cap on generated tokens
    stop_token_ids: sequence[int] = ()   # engine must stop when one is sampled
    seed: int | None = None        # deterministic decoding when set
    logprobs: int = 0              # 0 = off; k>0 = per-token logprob + top-k

VisionImage:                       # one image span of the prompt (ds41_api/vision.py)
    start: int                     # position of IMAGE_START in token_ids
    n_vit_h, n_vit_w: int          # ViT patch grid (14-pixel patches)
    n_llm_h, n_llm_w: int          # token grid, ceil(n_vit / 3)
    pixels: bytes                  # RGB, n_vit_h*14 rows of n_vit_w*14 pixels, resized and padded
                                   # like inference/image_processor.load_image (base64 on the wire)

Usage:
    prompt_tokens: int
    completion_tokens: int          # generated tokens, including reasoning tokens
```

## Operations

### `submit`

```
submit(token_ids: sequence[int],
       sampling: SamplingParams,
       images: sequence[VisionImage] | None = None,
       conversation_id: str | None = None,
       request_id: str | None = None
) -> BackendStream
```

- `token_ids`: full prompt already encoded by the official tokenizer, including BOS and the
  final assistant/thinking start marker. No server-side truncation is applied.
- `conversation_id`: optional hint for prefix/KV reuse. Same id implies a shared growing
  prefix; the engine may ignore it. Server may send the same id across turns of a session.
- `request_id`: chosen by the caller (UUID4 recommended); the engine MUST use it as the
  stream's identity so `cancel()` can target it. If omitted, the engine generates one.
- Returns immediately (does **not** wait for the first token). The stream object carries
  `request_id: str` and is an async iterator of events.
- Backpressure: the server pulls events; the engine should not buffer unboundedly.

### Event stream

Exactly zero or more token events, then exactly **one** final finish event (last).

```json
{"type": "token", "token_id": 123,
 "logprob": -0.012,                      // present when sampling.logprobs > 0
 "top_logprobs": [                        // up to sampling.logprobs entries
    {"token_id": 123, "logprob": -0.012}, ...
 ]}
```

```json
{"type": "finish",
 "finish_reason": "stop" | "length" | "cancelled" | "error",
 "usage": {"prompt_tokens": 812, "completion_tokens": 240},
 "error": "message or null"}             // set iff finish_reason == "error"
```

- `stop`: a `sampling.stop_token_ids` entry or EOS was sampled (token is not emitted).
- `length`: `sampling.max_tokens` reached.
- `cancelled`: ended early because `cancel(request_id)` was called.
- `error`: engine-side failure; the server surfaces HTTP 502 and `error` in the body.
- Reasoning tokens are **not** counted separately by the engine; the server derives them
  (tokens before the thinking-end delimiter) for `usage.completion_tokens_details`.

### `cancel`

```
cancel(request_id: str) -> bool
```

Best effort; returns True if a live request was found. The associated stream must still
terminate with a `finish` event (`cancelled`, or the already-produced `stop`/`length`).
Idempotent; unknown ids return False.

## Unix-socket JSON-lines transport

One request per stream, newline-delimited JSON both ways (UTF-8, no trailing spaces).
The server dials the socket path; the engine may serve many connections concurrently.
Request id is chosen by the client (UUID4 recommended).

```
-> {"op": "submit", "request_id": "...", "token_ids": [...], "sampling": {...},
    "images": [{"start": 33, "n_vit_h": 43, "n_vit_w": 58, "n_llm_h": 15, "n_llm_w": 20,
                "pixels": "<base64>"}], "conversation_id": "..."}
<- {"type": "token", ...}
<- {"type": "finish", ...}
-> {"op": "cancel", "request_id": "..."}
```

- The engine may send `{"type": "finish", ...}` at any time (e.g. engine error).
- A connection close without a finish event is a transport error; clients must surface it.
- `images` may be omitted/null. Empty `token_ids` is invalid.
- Each image's span, `[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]`, holds
  the image token id (129264) at every position of `token_ids`, as `prepare_vl_inputs` builds it. Spans are
  sorted and disjoint. The engine rejects images in `score` requests.

## Versioning

v1 is frozen once the first engine binds to it. Additions are additive new optional fields
with defaults; clients and engines must ignore unknown fields.
