"""Anthropic Messages API for DS41 (POST /v1/messages, /v1/messages/count_tokens).

Maps the Anthropic wire format onto the official DeepSeek-V4.1 reference
encoding via the same ChatCompletionRequest pipeline the OpenAI endpoint
uses, then renders Anthropic responses and SSE event streams.

Mapping notes (documented deliberately, reference-faithful):
  - system: string or {type:"text"} blocks -> one system message.
  - user blocks: text -> text; image -> reference image block (engine owns
    vision); tool_result -> reference tool_result block inside the user
    message content_blocks (the encoder renders <tool_result> and orders
    results by the preceding assistant's call order itself).
  - assistant blocks: text -> content; thinking/redacted_thinking ->
    reasoning_content (the reference drops thinking before the last user
    message and keeps it for trailing assistant turns); tool_use ->
    OpenAI tool_calls.
  - tools: {name, description, input_schema} -> OpenAI function tools.
  - tool_choice: auto/none supported; any/tool would require forcing, which
    the reference encoder cannot express -> 400.
  - thinking: {"type":"enabled","budget_tokens":N} enables thinking; the
    budget is mapped onto the reference numeric reasoning effort as
    <=2k -> "low", <=8k -> "high", else "max" (heuristic, documented).
  - stop_reason: end_turn | max_tokens | stop_sequence | tool_use.
  - usage carries the ds41_timing extension object.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backend import Backend
from .core import _safe_cancel, build_submission, run_generation
from .encoder import Encoder
from .schemas import ChatCompletionRequest


class AnthropicError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type


# ----------------------------------------------------------------------
# request mapping


def _budget_to_effort(budget: Optional[int]) -> Optional[int]:
    """Heuristic mapping of Anthropic budget_tokens onto the reference's
    numeric reasoning effort (1-100; low=50, high=75, max=100)."""
    if budget is None:
        return None
    if budget <= 2048:
        return 50
    if budget <= 8192:
        return 75
    return 100


def _system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if not isinstance(block, dict) or block.get("type") != "text":
                raise AnthropicError(400, "system blocks must be {type:'text'}")
            parts.append(block.get("text", ""))
        return "\n\n".join(parts)
    raise AnthropicError(400, "system must be a string or a list of text blocks")


def _tool_result_content(content: Any) -> Any:
    """tool_result content: string, or list of text/image blocks."""
    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, list):
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise AnthropicError(400, "tool_result content must be blocks")
            btype = block.get("type")
            if btype == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                blocks.append({"type": "image", "source": block.get("source")})
            else:
                raise AnthropicError(400, f"unsupported tool_result block type {btype!r}")
        return blocks
    raise AnthropicError(400, "tool_result content must be a string or block list")


def _convert_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise AnthropicError(400, "messages must be a non-empty list")

    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise AnthropicError(400, "each message must be an object")
        role = msg.get("role")
        if role == "system":
            # Mid-conversation system messages (newer Claude Code sends them)
            # map onto V4.1's native <｜System｜> turns.
            text = _system_text(msg.get("content"))
            if text:
                out.append({"role": "system", "content": text})
            continue
        if role not in ("user", "assistant"):
            raise AnthropicError(400, f"unsupported message role {role!r}")
        content = msg.get("content")
        if content is None:
            content = []

        if isinstance(content, str):
            if not content:
                raise AnthropicError(400, "message content must not be empty")
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise AnthropicError(400, "content must be a string or block list")

        if role == "user":
            blocks: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    raise AnthropicError(400, "content blocks must be objects")
                btype = block.get("type")
                if btype == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    source = block.get("source")
                    if not isinstance(source, dict):
                        raise AnthropicError(400, "image block needs a source object")
                    blocks.append({"type": "image", "source": source})
                elif btype == "tool_result":
                    if "tool_use_id" not in block:
                        raise AnthropicError(400, "tool_result block needs tool_use_id")
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": _tool_result_content(block.get("content")),
                        }
                    )
                else:
                    raise AnthropicError(400, f"unsupported user block type {btype!r}")
            if not blocks:
                raise AnthropicError(400, "user message has no content blocks")
            out.append({"role": "user", "content_blocks": blocks})
        else:  # assistant
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            reasoning: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    raise AnthropicError(400, "content blocks must be objects")
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype in ("thinking", "redacted_thinking"):
                    # Reference-faithful: reasoning before the last user
                    # message is dropped by the encoder anyway.
                    reasoning.append(block.get("thinking", "") or "")
                elif btype == "tool_use":
                    if "id" not in block or "name" not in block:
                        raise AnthropicError(400, "tool_use block needs id and name")
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
                else:
                    raise AnthropicError(400, f"unsupported assistant block type {btype!r}")
            message: Dict[str, Any] = {"role": "assistant"}
            if text_parts:
                message["content"] = "\n\n".join(text_parts)
            if reasoning:
                message["reasoning_content"] = "\n".join(reasoning)
            if tool_calls:
                message["tool_calls"] = tool_calls
            out.append(message)
    if all(m["role"] == "system" for m in out):
        raise AnthropicError(400, "messages must include a user or assistant message")
    return out


def parse_anthropic_request(body: Dict[str, Any], max_tokens_limit: int) -> ChatCompletionRequest:
    if not isinstance(body, dict):
        raise AnthropicError(400, "request body must be a JSON object")

    if "max_tokens" not in body:
        raise AnthropicError(400, "max_tokens is required")
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise AnthropicError(400, "max_tokens must be a positive integer")
    if max_tokens > max_tokens_limit:
        raise AnthropicError(400, f"max_tokens {max_tokens} exceeds the server limit {max_tokens_limit}")

    messages = _convert_messages(body.get("messages"))

    system = _system_text(body.get("system"))
    if system:
        messages.insert(0, {"role": "system", "content": system})

    thinking = body.get("thinking")
    thinking_enabled = False
    reasoning_effort: Optional[int] = None
    if thinking is not None:
        if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "disabled"):
            raise AnthropicError(400, "thinking must be {type:'enabled'|'disabled', ...}")
        thinking_enabled = thinking.get("type") == "enabled"
        if thinking_enabled:
            if not isinstance(body.get("temperature", 1.0), (int, float)) or body.get("temperature", 1.0) != 1:
                raise AnthropicError(
                    400, "temperature must be 1 when thinking is enabled", "invalid_request_error"
                )
            budget = thinking.get("budget_tokens")
            if budget is not None and (not isinstance(budget, int) or budget < 1024):
                raise AnthropicError(400, "budget_tokens must be an integer >= 1024")
            reasoning_effort = _budget_to_effort(budget)

    tools = None
    if body.get("tools") is not None:
        if not isinstance(body["tools"], list):
            raise AnthropicError(400, "tools must be a list")
        tools = []
        for tool in body["tools"]:
            if not isinstance(tool, dict) or "name" not in tool:
                raise AnthropicError(400, "each tool needs a name")
            function: Dict[str, Any] = {
                "name": tool["name"],
                "description": tool.get("description") or "",
            }
            if tool.get("input_schema") is not None:
                function["parameters"] = tool["input_schema"]
            tools.append({"type": "function", "function": function})

    tool_choice = "auto"
    tc = body.get("tool_choice")
    if tc is not None:
        if isinstance(tc, dict):
            tc_type = tc.get("type")
            if tc_type == "auto":
                tool_choice = "auto"
            elif tc_type == "none":
                tool_choice = "none"
            else:
                # any / tool: forced tool use; the reference encoder cannot
                # express forcing.
                raise AnthropicError(
                    400,
                    "tool_choice 'any'/'tool' (forced tool use) is not supported; "
                    "the reference encoding has no forcing mechanism",
                )
        elif isinstance(tc, str):
            if tc not in ("auto", "none"):
                raise AnthropicError(400, f"unsupported tool_choice {tc!r}")
            tool_choice = tc
        else:
            raise AnthropicError(400, "tool_choice must be a string or object")

    # The reference encoder renders tool schemas only for system-role
    # messages. Without a system prompt we synthesize an empty one so tool
    # schemas are never silently dropped (documented deviation, this
    # endpoint only; /v1/chat/completions keeps reference behavior).
    if tools and (not messages or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": ""})

    stop = body.get("stop_sequences")
    if stop is not None and not isinstance(stop, list):
        raise AnthropicError(400, "stop_sequences must be a list of strings")

    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 1.0)
    if not isinstance(temperature, (int, float)) or not isinstance(top_p, (int, float)):
        raise AnthropicError(400, "temperature/top_p must be numbers")

    # DS41 extension (documented): explicit prefix/KV-reuse hint for the
    # engine. metadata.user_id is deliberately NOT used for this (it is not
    # a conversation identity).
    conversation_id = body.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise AnthropicError(400, "conversation_id must be a string")

    return ChatCompletionRequest(
        model=str(body.get("model") or ""),
        messages=messages,
        thinking=thinking_enabled,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        stop=stop,
        max_tokens=max_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        stream=bool(body.get("stream")),
        conversation_id=conversation_id,
    )


# ----------------------------------------------------------------------
# response rendering


def _msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def _block_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


def _content_blocks(message: Dict[str, Any], thinking_mode: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (ValueError, TypeError):
            args = {"_raw": tc["function"]["arguments"]}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or _block_id(),
                "name": tc["function"]["name"],
                "input": args,
            }
        )
    return blocks


def _stop_reason(reason: str, stop_sequence: Optional[str], has_tool_calls: bool) -> Tuple[str, Optional[str]]:
    if stop_sequence:
        return "stop_sequence", stop_sequence
    if has_tool_calls:
        return "tool_use", None
    if reason == "length":
        return "max_tokens", None
    return "end_turn", None


def _usage(usage, timing: Dict[str, Any], prompt_hint: int, completion_tokens: int) -> Dict[str, Any]:
    input_tokens = usage.prompt_tokens if usage is not None and usage.prompt_tokens else prompt_hint
    output = (
        usage.completion_tokens
        if usage is not None and usage.completion_tokens
        else completion_tokens
    )
    block: Dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output}
    if timing:
        block["ds41_timing"] = timing
    return block


# ----------------------------------------------------------------------
# endpoint registration


def register_anthropic(
    app: FastAPI,
    backend: Backend,
    encoder: Encoder,
    model_id: str,
    gate,
    cfg,
    usage_block,
) -> None:
    from .schemas import resolve_thinking_mode

    @app.exception_handler(AnthropicError)
    async def anthropic_error_handler(request: Request, exc: AnthropicError):
        return JSONResponse(
            status_code=exc.status,
            content={"type": "error", "error": {"type": exc.err_type, "message": exc.message}},
        )

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise AnthropicError(400, "request body must be valid JSON")
        chat_request = parse_anthropic_request(body, cfg.max_tokens_limit)
        thinking_mode = resolve_thinking_mode(chat_request.thinking)

        slot, waited = await gate.acquire()
        try:
            stream, _, token_ids = await build_submission(backend, encoder, chat_request)
        except (ValueError, AssertionError) as exc:
            slot.release()
            raise AnthropicError(400, str(exc)) from exc

        if chat_request.stream:
            return StreamingResponse(
                _sse(chat_request, stream, thinking_mode, model_id,
                     prompt_tokens=len(token_ids), slot=slot, queue_wait_s=waited),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await _non_stream(chat_request, stream, thinking_mode, model_id,
                                     prompt_tokens=len(token_ids), queue_wait_s=waited)
        finally:
            slot.release()

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise AnthropicError(400, "request body must be valid JSON")
        try:
            # Same shape as /v1/messages (no generation happens here).
            chat_request = parse_anthropic_request({**body, "max_tokens": 1, "stream": False},
                                                   cfg.max_tokens_limit)
            prompt, _ = encoder.build_prompt(chat_request)
            token_ids = encoder.encode_prompt(prompt)
        except AnthropicError:
            raise
        except (ValueError, AssertionError) as exc:
            raise AnthropicError(400, str(exc)) from exc
        # Approximation: the encoded prompt includes the assistant
        # generation header the model appends (a handful of tokens).
        return {"input_tokens": len(token_ids)}

    # -- non-streaming ---------------------------------------------------

    async def _non_stream(chat_request, stream, thinking_mode, model, prompt_tokens, queue_wait_s=0.0):
        finish: Optional[Dict[str, Any]] = None
        usage = None
        async for event in run_generation(backend, encoder, chat_request, stream,
                                          queue_wait_s=queue_wait_s):
            if event["type"] == "finish":
                finish = event
                usage = event["usage"]
            elif event["type"] == "error":
                raise AnthropicError(502, event["message"], "api_error")
        if finish is None:
            raise AnthropicError(502, "backend stream ended without a finish event", "api_error")
        message = finish["message"]
        blocks = _content_blocks(message, thinking_mode)
        stop_reason, stop_sequence = _stop_reason(
            finish["reason"], finish.get("stop_sequence"), bool(message.get("tool_calls"))
        )
        msg_id = _msg_id()
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": _usage(usage, finish["timing"], prompt_tokens, finish["completion_tokens"]),
        }

    # -- streaming ---------------------------------------------------------

    async def _sse(chat_request, stream, thinking_mode, model, prompt_tokens, slot, queue_wait_s=0.0):
        """Anthropic SSE: `event: <type>` + `data: <json>` per spec."""
        request_id = stream.request_id
        msg_id = _msg_id()
        usage = None
        finish: Optional[Dict[str, Any]] = None

        def sse(event_type: str, data: Dict[str, Any]) -> str:
            data = dict(data)
            data.setdefault("type", event_type)
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            yield sse("message_start", {
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                }
            })

            open_blocks: Dict[int, str] = {}  # parser call index -> block index
            block_index = -1
            text_block_open = False
            thinking_block_open = False
            open_call_idx: Optional[int] = None

            async for event in run_generation(backend, encoder, chat_request, stream,
                                              queue_wait_s=queue_wait_s):
                kind = event["type"]
                if kind == "thinking":
                    if not thinking_block_open:
                        block_index += 1
                        thinking_block_open = True
                        yield sse("content_block_start", {
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    yield sse("content_block_delta", {
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": event["text"]},
                    })
                elif kind == "text":
                    if thinking_block_open:
                        yield sse("content_block_delta", {
                            "index": block_index,
                            "delta": {"type": "signature_delta", "signature": ""},
                        })
                        yield sse("content_block_stop", {"index": block_index})
                        thinking_block_open = False
                    if not text_block_open:
                        block_index += 1
                        text_block_open = True
                        yield sse("content_block_start", {
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield sse("content_block_delta", {
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": event["text"]},
                    })
                elif kind == "tool_call_start":
                    if thinking_block_open:
                        yield sse("content_block_delta", {
                            "index": block_index,
                            "delta": {"type": "signature_delta", "signature": ""},
                        })
                        yield sse("content_block_stop", {"index": block_index})
                        thinking_block_open = False
                    if text_block_open:
                        yield sse("content_block_stop", {"index": block_index})
                        text_block_open = False
                    if open_call_idx is not None:
                        yield sse("content_block_stop", {"index": open_blocks[open_call_idx]})
                        open_call_idx = None
                    block_index += 1
                    open_blocks[event["index"]] = block_index
                    open_call_idx = event["index"]
                    yield sse("content_block_start", {
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": event["id"],
                            "name": event["name"],
                            "input": {},
                        },
                    })
                elif kind == "tool_call_delta":
                    if event["fragment"]:
                        yield sse("content_block_delta", {
                            "index": open_blocks[event["index"]],
                            "delta": {"type": "input_json_delta", "partial_json": event["fragment"]},
                        })
                elif kind == "error":
                    yield sse("error", {"error": {"type": "api_error", "message": event["message"]}})
                    return
                elif kind == "finish":
                    finish = event
                    usage = event["usage"]

            if thinking_block_open:
                yield sse("content_block_delta", {
                    "index": block_index,
                    "delta": {"type": "signature_delta", "signature": ""},
                })
                yield sse("content_block_stop", {"index": block_index})
            if text_block_open:
                yield sse("content_block_stop", {"index": block_index})
            if open_call_idx is not None:
                yield sse("content_block_stop", {"index": open_blocks[open_call_idx]})
                open_call_idx = None

            if finish is None:
                yield sse("error", {"error": {"type": "api_error", "message": "stream ended early"}})
                return
            stop_reason, stop_sequence = _stop_reason(
                finish["reason"], finish.get("stop_sequence"),
                bool((finish["message"] or {}).get("tool_calls")),
            )
            yield sse("message_delta", {
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": _usage(usage, finish["timing"], prompt_tokens, finish["completion_tokens"]),
            })
            yield sse("message_stop", {})
        finally:
            # Fire-and-forget: the generator can be cancelled by a client
            # disconnect; the engine cancel must survive that.
            import asyncio

            asyncio.get_running_loop().create_task(_safe_cancel(backend, request_id))
            slot.release()
