"""OpenAI Responses API for DS41 (POST /v1/responses).

A protocol adapter over the same generation core as the other endpoints,
for Codex-style clients. Statelesss: `store` is ignored (no response
persistence), `previous_response_id` is rejected with 400.

Input item mapping:
  - {type:"message", role: system|developer}       -> system message
  - {type:"message", role: user|assistant}        -> chat message
      content parts: input_text/output_text -> text, input_image -> image
  - {type:"function_call", call_id, name, args}   -> assistant tool_calls
  - {type:"function_call_output", call_id, output}-> tool result message
  - {type:"reasoning", ...}                       -> dropped (the reference
    encoder drops prior-turn thinking before the last user message anyway)
  - {type:"item_reference"}                       -> 400 (store unsupported)

Output items: reasoning (summary-free), message (output_text content part)
and function_call items; SSE events response.created, output_item.added,
response.reasoning_text.delta, response.output_text.delta,
response.function_call_arguments.delta, output_item.done, response.completed.
Usage carries input/output tokens, reasoning token details and the
ds41_timing extension object.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backend import Backend
from .core import _safe_cancel, build_submission, run_generation
from .encoder import Encoder
from .schemas import ChatCompletionRequest


class ResponsesError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type


_EFFORT_MAP = {
    "minimal": 50,
    "low": 50,
    "medium": 60,
    "high": 75,
}


def _content_text(content: Any, role: str) -> Union[str, List[Dict[str, Any]]]:
    """message content: string, or parts list -> text or content_blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesError(400, "message content must be a string or parts array")
    has_image = any(
        isinstance(part, dict) and part.get("type") == "input_image" for part in content
    )
    if not has_image:
        return "\n\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("input_text", "output_text")
        )
    blocks: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ResponsesError(400, "content parts must be objects")
        ptype = part.get("type")
        if ptype in ("input_text", "output_text"):
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "input_image":
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            elif isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                raise ResponsesError(400, "input_image part needs image_url")
            blocks.append({"type": "image", "url": url})
        else:
            raise ResponsesError(400, f"unsupported content part type {ptype!r}")
    return blocks


def _convert_input(input_value: Any) -> List[Dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if not isinstance(input_value, list):
        raise ResponsesError(400, "input must be a string or an array of items")

    out: List[Dict[str, Any]] = []
    for item in input_value:
        if not isinstance(item, dict):
            raise ResponsesError(400, "input items must be objects")
        itype = item.get("type")

        if itype == "message":
            role = item.get("role", "user")
            content = _content_text(item.get("content"), role)
            if role in ("system", "developer"):
                if not isinstance(content, str):
                    raise ResponsesError(400, "system message content must be text")
                out.append({"role": "system", "content": content})
            elif role in ("user", "assistant"):
                if isinstance(content, list):
                    out.append({"role": role, "content_blocks": content})
                else:
                    out.append({"role": role, "content": content})
            else:
                raise ResponsesError(400, f"unsupported message role {role!r}")

        elif itype == "function_call":
            out.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments") or "{}",
                    },
                }],
            })

        elif itype == "function_call_output":
            output = item.get("output")
            if isinstance(output, dict):
                output = json.dumps(output.get("content", output))
            if not isinstance(output, str):
                raise ResponsesError(400, "function_call_output.output must be a string or object")
            out.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or "",
                "content": output,
            })

        elif itype == "reasoning":
            # Dropped: reference-faithful (prior-turn thinking is dropped).
            continue

        elif itype == "item_reference":
            raise ResponsesError(
                400,
                "item_reference requires response storage; this server is "
                "stateless (store unsupported)",
            )

        else:
            raise ResponsesError(400, f"unsupported input item type {itype!r}")
    return out


def parse_responses_request(body: Dict[str, Any], max_tokens_limit: int) -> ChatCompletionRequest:
    if not isinstance(body, dict):
        raise ResponsesError(400, "request body must be a JSON object")

    if body.get("previous_response_id"):
        raise ResponsesError(400, "previous_response_id is unsupported (stateless server)")

    messages = _convert_input(body.get("input"))
    if not messages:
        raise ResponsesError(400, "input must not be empty")

    instructions = body.get("instructions")
    if instructions:
        messages.insert(0, {"role": "system", "content": str(instructions)})

    tools = None
    if body.get("tools") is not None:
        if not isinstance(body["tools"], list):
            raise ResponsesError(400, "tools must be a list")
        tools = []
        for tool in body["tools"]:
            if not isinstance(tool, dict):
                raise ResponsesError(400, "tools entries must be objects")
            ttype = tool.get("type", "function")
            if ttype != "function":
                raise ResponsesError(400, f"unsupported tool type {ttype!r}")
            function = tool.get("function") or {}
            name = tool.get("name") or function.get("name")
            if not name:
                raise ResponsesError(400, "function tools need a name")
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or function.get("description") or "",
                    "parameters": tool.get("parameters")
                    or function.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            })

    tool_choice = "auto"
    tc = body.get("tool_choice")
    if tc is not None:
        if isinstance(tc, str):
            if tc not in ("auto", "none"):
                # "required" and named forcing are not expressible in the
                # reference encoding.
                raise ResponsesError(
                    400, f"tool_choice {tc!r} is not supported (no forcing mechanism)"
                )
            tool_choice = tc
        elif isinstance(tc, dict) and tc.get("type") in ("auto", "none"):
            tool_choice = tc["type"]
        else:
            raise ResponsesError(400, "unsupported tool_choice")

    thinking_enabled = True
    reasoning_effort: Optional[Union[int, str]] = None
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            raise ResponsesError(400, "reasoning must be an object")
        effort = reasoning.get("effort")
        if effort is not None:
            if effort not in _EFFORT_MAP:
                raise ResponsesError(400, f"unsupported reasoning effort {effort!r}")
            reasoning_effort = _EFFORT_MAP[effort]
        # reasoning.summary is accepted and ignored: the model streams raw
        # reasoning (response.reasoning_text.delta), not summaries.

    max_output = body.get("max_output_tokens")
    if max_output is None:
        max_output = 4096
    if not isinstance(max_output, int) or isinstance(max_output, bool) or max_output <= 0:
        raise ResponsesError(400, "max_output_tokens must be a positive integer")
    if max_output > max_tokens_limit:
        raise ResponsesError(400, f"max_output_tokens {max_output} exceeds the server limit {max_tokens_limit}")

    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 1.0)
    if not isinstance(temperature, (int, float)) or not isinstance(top_p, (int, float)):
        raise ResponsesError(400, "temperature/top_p must be numbers")

    # The reference encoder renders tool schemas / response formats only for
    # system-role messages. Without a system message we synthesize an empty
    # one so neither is silently dropped (documented deviation, this
    # endpoint only; /v1/chat/completions keeps reference behavior).
    if (tools or body.get("text")) and (
        not messages or messages[0].get("role") != "system"
    ):
        messages.insert(0, {"role": "system", "content": ""})

    response_format = None
    rf = body.get("text") or {}
    if isinstance(rf, dict) and isinstance(rf.get("format"), dict):
        fmt = rf["format"]
        if fmt.get("type") == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name") or "response",
                    "schema": fmt.get("schema") or {},
                },
            }
        elif fmt.get("type") == "json_object":
            response_format = {"type": "json_object"}
        elif fmt.get("type") != "text":
            raise ResponsesError(400, f"unsupported text.format type {fmt.get('type')!r}")

    # DS41 extension (documented): explicit prefix/KV-reuse hint.
    conversation_id = body.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise ResponsesError(400, "conversation_id must be a string")

    return ChatCompletionRequest(
        model=str(body.get("model") or ""),
        messages=messages,
        thinking=thinking_enabled,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        stop=body.get("stop") if isinstance(body.get("stop"), list) else None,
        max_tokens=max_output,
        temperature=float(temperature),
        top_p=float(top_p),
        stream=bool(body.get("stream")),
        conversation_id=conversation_id,
    )


# ----------------------------------------------------------------------
# response rendering


def _resp_id() -> str:
    return "resp_" + uuid.uuid4().hex[:24]


def _item_id(prefix: str) -> str:
    return f"{prefix}_" + uuid.uuid4().hex[:24]


def _output_items(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if message.get("reasoning_content"):
        items.append({
            "type": "reasoning",
            "id": _item_id("rs"),
            "summary": [],
        })
    if message.get("content"):
        items.append({
            "type": "message",
            "id": _item_id("msg"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": message["content"], "annotations": []}],
        })
    for tc in message.get("tool_calls") or []:
        items.append({
            "type": "function_call",
            "id": _item_id("fc"),
            "call_id": tc.get("id"),
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
            "status": "completed",
        })
    return items


def _usage(usage, timing: Dict[str, Any], prompt_hint: int, completion_tokens: int,
           reasoning_tokens: int) -> Dict[str, Any]:
    input_tokens = usage.prompt_tokens if usage is not None and usage.prompt_tokens else prompt_hint
    output = (
        usage.completion_tokens
        if usage is not None and usage.completion_tokens
        else completion_tokens
    )
    block: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cached_prompt_tokens or 0 if usage is not None else 0
        },
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output,
    }
    if timing:
        block["ds41_timing"] = timing
    return block


def _response_shell(resp_id: str, model: str, status: str = "in_progress",
                    output: Optional[List[Dict[str, Any]]] = None,
                    incomplete: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp: Dict[str, Any] = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output or [],
        "usage": None,
    }
    if incomplete:
        resp["incomplete_details"] = incomplete
    return resp


# ----------------------------------------------------------------------
# endpoint registration


def register_responses(
    app: FastAPI,
    backend: Backend,
    encoder: Encoder,
    model_id: str,
    gate,
    cfg,
    usage_block,
) -> None:
    from .schemas import resolve_thinking_mode

    @app.exception_handler(ResponsesError)
    async def responses_error_handler(request: Request, exc: ResponsesError):
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"message": exc.message, "type": exc.err_type, "code": exc.status}},
        )

    @app.post("/v1/responses")
    async def responses(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise ResponsesError(400, "request body must be valid JSON")
        chat_request = parse_responses_request(body, cfg.max_tokens_limit)
        thinking_mode = resolve_thinking_mode(chat_request.thinking)

        slot, waited = await gate.acquire()
        try:
            stream, _, token_ids = await build_submission(backend, encoder, chat_request)
        except (ValueError, AssertionError) as exc:
            slot.release()
            raise ResponsesError(400, str(exc)) from exc
        except BaseException:
            # Any other failure (or cancellation) before the response owns the slot must free it.
            slot.release()
            raise

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
                raise ResponsesError(502, event["message"], "api_error")
        if finish is None:
            raise ResponsesError(502, "backend stream ended without a finish event", "api_error")

        message = finish["message"]
        items = _output_items(message)
        status = "incomplete" if finish["reason"] == "length" else "completed"
        incomplete = {"reason": "max_output_tokens"} if status == "incomplete" else None
        resp = _response_shell(_resp_id(), model, status, items, incomplete)
        resp["usage"] = _usage(usage, finish["timing"], prompt_tokens,
                               finish["completion_tokens"], finish["reasoning_tokens"])
        return resp

    # -- streaming ---------------------------------------------------------

    async def _sse(chat_request, stream, thinking_mode, model, prompt_tokens, slot, queue_wait_s=0.0):
        request_id = stream.request_id
        resp_id = _resp_id()

        def sse(event_type: str, data: Dict[str, Any]) -> str:
            data = dict(data)
            data.setdefault("type", event_type)
            data.setdefault("response_id", resp_id)
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def close_current() -> Optional[str]:
            if current_item is None:
                return None
            return sse("response.output_item.done", {
                "output_index": output_index,
                "item": current_item,
            })

        try:
            yield sse("response.created", {"response": _response_shell(resp_id, model)})
            yield sse("response.in_progress", {"response": _response_shell(resp_id, model)})

            output_index = -1
            current_kind: Optional[str] = None
            current_item: Optional[Dict[str, Any]] = None
            message_text = ""
            open_calls: Dict[int, int] = {}  # parser call index -> output_index
            finish: Optional[Dict[str, Any]] = None
            usage = None

            async for event in run_generation(backend, encoder, chat_request, stream,
                                              queue_wait_s=queue_wait_s):
                kind = event["type"]
                if kind == "thinking":
                    if current_kind != "reasoning":
                        ev = close_current()
                        if ev:
                            yield ev
                        output_index += 1
                        current_kind = "reasoning"
                        current_item = {
                            "type": "reasoning",
                            "id": _item_id("rs"),
                            "summary": [],
                        }
                        yield sse("response.output_item.added", {
                            "output_index": output_index,
                            "item": current_item,
                        })
                    yield sse("response.reasoning_text.delta", {
                        "output_index": output_index,
                        "content_index": 0,
                        "item_id": current_item["id"],
                        "delta": event["text"],
                    })
                elif kind == "text":
                    if current_kind != "message":
                        ev = close_current()
                        if ev:
                            yield ev
                        output_index += 1
                        current_kind = "message"
                        message_text = ""
                        current_item = {
                            "type": "message",
                            "id": _item_id("msg"),
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        }
                        yield sse("response.output_item.added", {
                            "output_index": output_index,
                            "item": current_item,
                        })
                    message_text += event["text"]
                    yield sse("response.output_text.delta", {
                        "output_index": output_index,
                        "content_index": 0,
                        "item_id": current_item["id"],
                        "delta": event["text"],
                    })
                elif kind == "tool_call_start":
                    ev = close_current()
                    if ev:
                        yield ev
                    output_index += 1
                    current_kind = "function_call"
                    open_calls[event["index"]] = output_index
                    current_item = {
                        "type": "function_call",
                        "id": _item_id("fc"),
                        "call_id": event["id"],
                        "name": event["name"],
                        "arguments": "",
                        "status": "in_progress",
                    }
                    yield sse("response.output_item.added", {
                        "output_index": output_index,
                        "item": current_item,
                    })
                elif kind == "tool_call_delta":
                    if event["fragment"]:
                        if current_item is not None and current_kind == "function_call":
                            current_item["arguments"] += event["fragment"]
                        yield sse("response.function_call_arguments.delta", {
                            "output_index": open_calls[event["index"]],
                            "item_id": None,
                            "delta": event["fragment"],
                        })
                elif kind == "error":
                    yield sse("error", {"message": event["message"]})
                    return
                elif kind == "finish":
                    finish = event
                    usage = event["usage"]

            if current_item is not None:
                if current_kind == "message":
                    current_item["status"] = "completed"
                    current_item["content"] = [
                        {"type": "output_text", "text": message_text, "annotations": []}
                    ]
                elif current_kind == "function_call":
                    current_item["status"] = "completed"
                yield sse("response.output_item.done", {
                    "output_index": output_index,
                    "item": current_item,
                })

            if finish is None:
                yield sse("error", {"message": "stream ended early"})
                return
            message = finish["message"]
            items = _output_items(message)
            status = "incomplete" if finish["reason"] == "length" else "completed"
            incomplete = {"reason": "max_output_tokens"} if status == "incomplete" else None
            resp = _response_shell(resp_id, model, status, items, incomplete)
            resp["usage"] = _usage(usage, finish["timing"], prompt_tokens,
                                   finish["completion_tokens"], finish["reasoning_tokens"])
            yield sse("response.completed", {"response": resp})
        finally:
            # Fire-and-forget: the generator can be cancelled by a client
            # disconnect; the engine cancel must survive that.
            import asyncio

            asyncio.get_running_loop().create_task(_safe_cancel(backend, request_id))
            slot.release()
