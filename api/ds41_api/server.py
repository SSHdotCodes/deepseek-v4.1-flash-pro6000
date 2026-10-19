"""OpenAI/Anthropic-compatible HTTP server for DS41.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions     (OpenAI, streaming and non-streaming)
  POST /v1/messages              (Anthropic Messages API, streaming and not)
  POST /v1/messages/count_tokens (Anthropic token counting)
  POST /v1/responses             (OpenAI Responses API, streaming and not)

The server encodes requests with the official reference encoding, submits
token ids to a Backend (engine), and renders protocol responses including
reasoning and tool calls. Streaming uses the incremental StreamParser
state machine with delimiter hold-back.

Robustness (ServerConfig):
  - max_body_bytes: requests with a larger Content-Length get 413.
  - max_concurrent / max_queue / queue_timeout: the engine is treated as
    single-stream by default; concurrent requests queue (429 when the
    queue is full or the wait exceeds queue_timeout).
  - max_tokens_limit: hard cap on generation length (400 above).

Every response's usage carries a `ds41_timing` extension object
(prefill/decode ms, decode tok/s, queue ms, cache and draft stats when the
engine reports them).

CLI (demo/testing):
  python -m ds41_api.server --tokenizer <tokenizer.json> [--port 8000]
runs with a FakeBackend that echoes scripts from --script-file (one JSON per
line: {"text": "..."}). Bind address is 127.0.0.1 by default and refuses
non-loopback hosts unless --allow-remote is passed.
"""

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import encoding as enc
from .backend import Backend, FinishEvent, SamplingParams, TokenEvent
from .encoder import CharTokenizer, DEFAULT_MODEL_ID, Encoder
from .fake_backend import FakeBackend
from .parser import StreamParser
from .schemas import ChatCompletionRequest, resolve_thinking_mode
from .socket_backend import SocketBackend
from .timing import TimingCollector


@dataclass
class ServerConfig:
    max_body_bytes: int = 10 * 1024 * 1024
    # The engine (ds41d) is single-stream: one generation at a time, others
    # queue. 429 when the queue is full or a request waits too long.
    max_concurrent: int = 1
    max_queue: int = 64
    queue_timeout: float = 120.0
    max_tokens_limit: int = 32768


class QueueFullError(Exception):
    """Admission control rejected the request (HTTP 429)."""


class RequestGate:
    """Concurrency admission for backend generations."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._sem = asyncio.Semaphore(config.max_concurrent)
        self._waiting = 0

    async def acquire(self):
        """Wait for a generation slot. Returns (slot, waited_seconds).

        Raises QueueFullError when the queue is at capacity or the wait
        exceeds queue_timeout.
        """
        if self._waiting >= self.config.max_queue:
            raise QueueFullError(
                f"request queue full ({self.config.max_queue} waiting); "
                "engine is single-stream, retry later"
            )
        self._waiting += 1
        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                self._sem.acquire(), timeout=self.config.queue_timeout
            )
        except asyncio.TimeoutError:
            raise QueueFullError(
                f"queued longer than {self.config.queue_timeout:.0f}s; retry later"
            ) from None
        finally:
            self._waiting -= 1
        waited = time.perf_counter() - start
        return self._sem, waited


def _error_body(status: int, message: str, err_type: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "code": status}}


def _json_error(status: int, message: str, err_type: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=_error_body(status, message, err_type))


def create_app(
    backend: Backend,
    encoder: Encoder,
    model_id: str = DEFAULT_MODEL_ID,
    config: Optional[ServerConfig] = None,
) -> FastAPI:
    cfg = config or ServerConfig()
    gate = RequestGate(cfg)
    app = FastAPI(title="DS41 API", version="0.2.0")

    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                size = int(cl)
            except ValueError:
                size = None
            if size is not None and size > cfg.max_body_bytes:
                return _json_error(
                    413,
                    f"request body of {size} bytes exceeds the "
                    f"{cfg.max_body_bytes} byte limit",
                    "request_too_large",
                )
        return await call_next(request)

    def _validate_max_tokens(value: int) -> None:
        if value > cfg.max_tokens_limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"max_tokens {value} exceeds the server limit "
                    f"{cfg.max_tokens_limit}"
                ),
            )

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": _ts(),
                    "owned_by": "deepseek",
                }
            ],
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "model": model_id}

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, exc: QueueFullError):
        if request.url.path.startswith("/v1/messages"):
            # Anthropic error shape for Anthropic clients.
            return JSONResponse(
                status_code=429,
                content={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": str(exc)},
                },
            )
        return _json_error(429, str(exc), "rate_limit_error")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        try:
            thinking_mode = resolve_thinking_mode(request.thinking)
            _validate_max_tokens(request.resolved_max_tokens())
            prompt, images = encoder.build_prompt(request)
        except (ValueError, AssertionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        slot, waited = await gate.acquire()
        collector = TimingCollector()
        collector.waited_in_queue(waited)

        token_ids = encoder.encode_prompt(prompt)
        eos_id = encoder.token_id(enc.eos_token)
        stop_token_ids = tuple(
            tid for tid in (eos_id,) if tid is not None
        )
        sampling = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.resolved_max_tokens(),
            stop_token_ids=stop_token_ids,
            seed=request.seed,
            logprobs=request.resolved_logprobs(),
        )
        collector.submit_started()
        stream = await backend.submit(
            token_ids,
            sampling,
            images=images or None,
            conversation_id=request.conversation_id,
            request_id=uuid.uuid4().hex,
        )

        if request.stream:
            return StreamingResponse(
                _sse_stream(request, stream, encoder, thinking_mode, model_id,
                            prompt_tokens=len(token_ids), collector=collector,
                            slot=slot),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            return await _complete(
                request, stream, encoder, thinking_mode, model_id,
                prompt_tokens=len(token_ids), collector=collector,
            )
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"backend output malformed: {exc}") from exc
        finally:
            slot.release()

    def _chunk_base(model: str) -> Dict[str, Any]:
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion.chunk",
            "created": _ts(),
            "model": model,
        }

    async def _sse_stream(
        request: ChatCompletionRequest,
        stream,
        encoder: Encoder,
        thinking_mode: str,
        model: str,
        prompt_tokens: int,
        collector: TimingCollector,
        slot,
    ) -> AsyncIterator[str]:
        parser = StreamParser(thinking_mode, request.stop_sequences())
        detok = encoder.detokenizer()
        include_usage = bool((request.stream_options or {}).get("include_usage"))
        want_logprobs = request.resolved_logprobs() > 0
        finish_reason: Optional[str] = None
        backend_finish = "stop"
        saw_tool_calls = False
        completion_tokens = 0
        reasoning_tokens = 0
        usage = None
        request_id = stream.request_id
        try:
            first = _chunk_base(model)
            first["choices"] = [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ]
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

            async for event in stream:
                if isinstance(event, TokenEvent):
                    collector.token()
                    completion_tokens += 1
                    token_text = None
                    if want_logprobs:
                        token_text = encoder.decode_one(event.token_id)
                    if parser.state == "thinking":
                        reasoning_tokens += 1
                    text = detok.feed(event.token_id)
                    if not text:
                        continue
                    deltas = parser.feed(text)
                    for delta in deltas:
                        if "tool_calls" in delta:
                            saw_tool_calls = True
                        chunk = _chunk_base(model)
                        choice: Dict[str, Any] = {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                        if want_logprobs and "content" in delta and token_text:
                            choice["logprobs"] = {
                                "content": [
                                    _logprob_entry(
                                        token_text, event.logprob, event.top_logprobs
                                    )
                                ]
                            }
                        chunk["choices"] = [choice]
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    if parser.stopped_on_stop:
                        await _cancel(backend, request_id)
                        finish_reason = "stop"
                        break
                elif isinstance(event, FinishEvent):
                    backend_finish = event.finish_reason
                    usage = event.usage
                    if event.finish_reason == "error":
                        chunk = _chunk_base(model)
                        chunk["choices"] = [
                            {"index": 0, "delta": {}, "finish_reason": None}
                        ]
                        chunk["error"] = {"message": event.error}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    break

            collector.finish()
            if finish_reason is None:
                finish_reason = {
                    "stop": "stop",
                    "length": "length",
                    "cancelled": "stop",
                }.get(backend_finish, "stop")
            # Flush anything held back (e.g. stream cut at max_tokens).
            for delta in parser.finish_truncated():
                if "tool_calls" in delta:
                    saw_tool_calls = True
                chunk = _chunk_base(model)
                chunk["choices"] = [
                    {"index": 0, "delta": delta, "finish_reason": None}
                ]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            if saw_tool_calls and finish_reason == "stop":
                finish_reason = "tool_calls"
            chunk = _chunk_base(model)
            chunk["choices"] = [
                {"index": 0, "delta": {}, "finish_reason": finish_reason}
            ]
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            if include_usage:
                chunk = _chunk_base(model)
                chunk["choices"] = []
                chunk["usage"] = _usage_block(
                    usage, prompt_hint=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    timing=collector.stats(usage),
                )
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Fire-and-forget so the engine cancel survives a client
            # disconnect (generator cancellation); idempotent either way.
            import asyncio as _aio

            _aio.get_running_loop().create_task(_cancel(backend, request_id))
            slot.release()

    async def _complete(
        request: ChatCompletionRequest,
        stream,
        encoder: Encoder,
        thinking_mode: str,
        model: str,
        prompt_tokens: int,
        collector: TimingCollector,
    ) -> Dict[str, Any]:
        parser = StreamParser(thinking_mode, request.stop_sequences())
        detok = encoder.detokenizer()
        want_logprobs = request.resolved_logprobs() > 0
        content_logprobs: List[Dict[str, Any]] = []
        completion_tokens = 0
        reasoning_tokens = 0
        finish_event: Optional[FinishEvent] = None

        async for event in stream:
            if isinstance(event, TokenEvent):
                collector.token()
                completion_tokens += 1
                if parser.state == "thinking":
                    reasoning_tokens += 1
                text = detok.feed(event.token_id)
                if want_logprobs:
                    token_text = encoder.decode_one(event.token_id)
                    if parser.state in ("content",) and token_text:
                        content_logprobs.append(
                            _logprob_entry(
                                token_text, event.logprob, event.top_logprobs
                            )
                        )
                if text:
                    parser.feed(text)
                if parser.stopped_on_stop:
                    await _cancel(backend, stream.request_id)
                    break
            else:
                finish_event = event
                break

        collector.finish()
        parser.finish_truncated()
        message = parser.message()
        finish_reason: str
        if parser.stopped_on_stop:
            finish_reason = "stop"
        elif finish_event is None:
            finish_reason = "stop"
        elif finish_event.finish_reason == "error":
            raise HTTPException(
                status_code=502,
                detail=f"backend error: {finish_event.error}",
            )
        elif finish_event.finish_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"
        if finish_reason == "stop" and message.get("tool_calls"):
            finish_reason = "tool_calls"

        usage = finish_event.usage if finish_event is not None else None
        response: Dict[str, Any] = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": _ts(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage_block(
                usage,
                prompt_hint=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                timing=collector.stats(usage),
            ),
        }
        if want_logprobs:
            response["choices"][0]["logprobs"] = {"content": content_logprobs}
        return response

    async def _cancel(backend: Backend, request_id: str) -> None:
        try:
            await backend.cancel(request_id)
        except Exception:
            pass

    def _logprob_entry(
        token_text: str, logprob: Optional[float], top_logprobs: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "token": token_text,
            "bytes": list(token_text.encode("utf-8")),
            "logprob": logprob,
        }
        if top_logprobs:
            entry["top_logprobs"] = [
                {
                    "token": encoder.decode_one(t.get("token_id", 0)),
                    "bytes": list(
                        (encoder.decode_one(t.get("token_id", 0)) or "").encode("utf-8")
                    ),
                    "logprob": t.get("logprob"),
                }
                for t in top_logprobs
            ]
        return entry

    def _usage_block(
        usage,
        prompt_hint: int,
        completion_tokens: int,
        reasoning_tokens: int,
        timing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt_tokens = usage.prompt_tokens if usage is not None else prompt_hint
        completion = (
            usage.completion_tokens
            if usage is not None and usage.completion_tokens
            else completion_tokens
        )
        block: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "prompt_tokens_details": {
                "cached_tokens": usage.cached_prompt_tokens or 0
                if usage is not None
                else 0
            },
        }
        if timing:
            block["ds41_timing"] = timing
        return block

    # -- Anthropic Messages API + OpenAI Responses API ------------------
    from .anthropic import register_anthropic
    from .responses import register_responses

    register_anthropic(
        app,
        backend=backend,
        encoder=encoder,
        model_id=model_id,
        gate=gate,
        cfg=cfg,
        usage_block=_usage_block,
    )
    register_responses(
        app,
        backend=backend,
        encoder=encoder,
        model_id=model_id,
        gate=gate,
        cfg=cfg,
        usage_block=_usage_block,
    )

    return app


def _ts() -> int:
    return int(time.time())


def main() -> None:
    parser = argparse.ArgumentParser(description="DS41 OpenAI-compatible demo server")
    parser.add_argument("--tokenizer", default=None, help="path to tokenizer.json")
    parser.add_argument("--backend", choices=["fake", "socket"], default="fake")
    parser.add_argument("--socket", default=None, help="unix socket path (socket backend)")
    parser.add_argument(
        "--script-file",
        default=None,
        help="JSONL file of fake completions: {\"text\": \"...\"} per line",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="explicitly allow non-loopback bind (default refuses)",
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--max-queue", type=int, default=64)
    parser.add_argument("--max-tokens-limit", type=int, default=32768,
                        help="hard cap on max_tokens per request (400 above)")
    parser.add_argument("--no-remote-images", action="store_true",
                        help="refuse http(s) image URLs (data: URLs and base64 still work)")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        parser.error(
            f"refusing to bind {args.host}: pass --allow-remote to serve off-loopback"
        )

    if args.tokenizer:
        encoder = Encoder.from_file(args.tokenizer, allow_remote_images=not args.no_remote_images)
    else:
        print("WARNING: no tokenizer.json path given; using CharTokenizer fallback")
        encoder = Encoder(CharTokenizer())

    if args.backend == "socket":
        assert args.socket, "--socket required for the socket backend"
        backend = SocketBackend(args.socket)
    else:
        fake = FakeBackend(tokenizer=encoder.tokenizer)
        if args.script_file:
            with open(args.script_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "text" in obj:
                        fake.add_script(obj["text"])
                    elif "token_ids" in obj:
                        fake.add_script(obj["token_ids"])
        else:
            fake.add_script(
                "This is the FakeBackend demo completion."
                + enc.eos_token
            )
        backend = fake

    import uvicorn

    config = ServerConfig(
        max_concurrent=max(1, args.max_concurrent),
        max_queue=max(0, args.max_queue),
        max_tokens_limit=max(1, args.max_tokens_limit),
    )
    app = create_app(backend, encoder, model_id=args.model, config=config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
