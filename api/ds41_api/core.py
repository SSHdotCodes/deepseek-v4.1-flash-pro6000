"""Protocol-agnostic generation core shared by the Anthropic and Responses
endpoints.

run_generation() submits a ChatCompletionRequest-shaped request to the
backend and yields normalized events any wire protocol can render:

    {"type": "thinking", "text": str}
    {"type": "text", "text": str}
    {"type": "tool_call_start", "index": int, "id": str, "name": str}
    {"type": "tool_call_delta", "index": int, "fragment": str}
    {"type": "finish",
     "reason": "stop" | "length",
     "stop_sequence": str | None,     # user stop sequence that fired
     "message": {...},                # final parsed message (OpenAI shape)
     "usage": Usage | None,
     "timing": {...}}                 # ds41_timing object (TimingCollector)

Backend errors surface as {"type": "error", "message": str} followed by the
end of iteration. The final parsed message always matches the reference
parse_message_from_completion_text semantics via StreamParser accumulation.
"""

from typing import Any, AsyncIterator, Dict, Optional
import uuid

from . import encoding as enc
from .backend import Backend, FinishEvent, SamplingParams, TokenEvent
from .encoder import Encoder
from .parser import StreamParser
from .schemas import ChatCompletionRequest
from .timing import TimingCollector


class GenerationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


async def build_submission(
    backend: Backend,
    encoder: Encoder,
    request: ChatCompletionRequest,
    request_id: Optional[str] = None,
):
    """Encode the request and submit it; returns (stream, sampling, token_ids)."""
    prompt, images = encoder.build_prompt(request)
    token_ids = encoder.encode_prompt(prompt)
    eos_id = encoder.token_id(enc.eos_token)
    stop_token_ids = tuple(tid for tid in (eos_id,) if tid is not None)
    sampling = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.resolved_max_tokens(),
        stop_token_ids=stop_token_ids,
        seed=request.seed,
        logprobs=request.resolved_logprobs(),
    )
    stream = await backend.submit(
        token_ids,
        sampling,
        images=images or None,
        conversation_id=request.conversation_id,
        request_id=request_id or uuid.uuid4().hex,
    )
    return stream, sampling, token_ids


async def run_generation(
    backend: Backend,
    encoder: Encoder,
    request: ChatCompletionRequest,
    stream,
    queue_wait_s: float = 0.0,
) -> AsyncIterator[Dict[str, Any]]:
    """Consume a BackendStream; yield normalized events (see module docstring).

    queue_wait_s: time this request spent waiting for its generation slot;
    reported as queue_ms in the finish event's timing object."""
    from .schemas import resolve_thinking_mode

    thinking_mode = resolve_thinking_mode(request.thinking)
    parser = StreamParser(thinking_mode, request.stop_sequences())
    detok = encoder.detokenizer()
    timing = TimingCollector()
    timing.waited_in_queue(queue_wait_s)
    timing.submit_started()
    usage = None
    finish_event: Optional[FinishEvent] = None
    reasoning_tokens = 0

    def tool_call_delta(tc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        idx = tc["index"]
        if "id" in tc or tc.get("function", {}).get("name"):
            return {
                "type": "tool_call_start",
                "index": idx,
                "id": tc.get("id") or parser.tool_calls[idx]["id"],
                "name": tc["function"]["name"],
            }
        return {
            "type": "tool_call_delta",
            "index": idx,
            "fragment": tc["function"].get("arguments", ""),
        }

    async for event in stream:
        if isinstance(event, TokenEvent):
            timing.token()
            if parser.state == "thinking":
                reasoning_tokens += 1
            text = detok.feed(event.token_id)
            if not text:
                continue
            for delta in parser.feed(text):
                if "reasoning_content" in delta:
                    yield {"type": "thinking", "text": delta["reasoning_content"]}
                elif "content" in delta:
                    yield {"type": "text", "text": delta["content"]}
                elif "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        ev = tool_call_delta(tc)
                        if ev:
                            yield ev
            if parser.stopped_on_stop:
                await _safe_cancel(backend, stream.request_id)
                break
        elif isinstance(event, FinishEvent):
            finish_event = event
            if event.finish_reason == "error":
                yield {"type": "error", "message": event.error or "backend error"}
                return
            break

    timing.finish()
    for delta in parser.finish_truncated():
        if "tool_calls" in delta:
            for tc in delta["tool_calls"]:
                ev = tool_call_delta(tc)
                if ev:
                    yield ev

    reason = "stop"
    if parser.stopped_on_stop:
        reason = "stop"
    elif finish_event is None or finish_event.finish_reason == "cancelled":
        reason = "stop"
    elif finish_event.finish_reason == "length":
        reason = "length"
    usage = finish_event.usage if finish_event is not None else None
    yield {
        "type": "finish",
        "reason": reason,
        "stop_sequence": parser.matched_stop,
        "message": parser.message(),
        "usage": usage,
        "completion_tokens": timing.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "timing": timing.stats(usage),
    }


async def _safe_cancel(backend: Backend, request_id: str) -> None:
    try:
        await backend.cancel(request_id)
    except Exception:
        pass
