"""Backend contract: the inference engine interface the API server sits in front of.

See ~/ai/ds41/contracts/backend-api.md for the full contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class SamplingParams:
    """Sampling knobs passed to the engine. Defaults follow the DS41 recommended
    sampling (temperature 1.0, top_p 1.0)."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096
    stop_token_ids: Sequence[int] = ()
    seed: Optional[int] = None
    # 0 = no logprobs; k>0 = return logprob of each sampled token plus the
    # top k token logprobs per position.
    logprobs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop_token_ids": list(self.stop_token_ids),
            "seed": self.seed,
            "logprobs": self.logprobs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingParams":
        return cls(
            temperature=d.get("temperature", 1.0),
            top_p=d.get("top_p", 1.0),
            max_tokens=d.get("max_tokens", 4096),
            stop_token_ids=tuple(d.get("stop_token_ids", ())),
            seed=d.get("seed"),
            logprobs=d.get("logprobs", 0),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # --- optional engine-reported fields (contract delta, additive) ---
    # Prompt tokens served from the prefix/KV cache (conversation reuse).
    cached_prompt_tokens: Optional[int] = None
    # Engine-measured phases in ms. When absent the server falls back to
    # wall-clock: prefill ~ submit->first token, decode ~ first token->finish.
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    # Speculative decoding: drafted tokens and how many were accepted.
    draft_tokens: Optional[int] = None
    accepted_draft_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        for key in (
            "cached_prompt_tokens",
            "prefill_ms",
            "decode_ms",
            "draft_tokens",
            "accepted_draft_tokens",
        ):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Usage":
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            cached_prompt_tokens=d.get("cached_prompt_tokens"),
            prefill_ms=d.get("prefill_ms"),
            decode_ms=d.get("decode_ms"),
            draft_tokens=d.get("draft_tokens"),
            accepted_draft_tokens=d.get("accepted_draft_tokens"),
        )


@dataclass
class TokenEvent:
    """One generated token. token_id is the engine token id; the server is
    responsible for detokenization."""

    token_id: int
    logprob: Optional[float] = None
    # List of {token_id, logprob} dicts, up to sampling.logprobs entries.
    top_logprobs: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "token",
            "token_id": self.token_id,
            "logprob": self.logprob,
            "top_logprobs": self.top_logprobs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenEvent":
        return cls(
            token_id=d["token_id"],
            logprob=d.get("logprob"),
            top_logprobs=d.get("top_logprobs"),
        )


@dataclass
class FinishEvent:
    """Terminal event for one request. Always exactly one per stream, last."""

    finish_reason: str  # "stop" | "length" | "cancelled" | "error"
    usage: Usage = field(default_factory=Usage)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "finish",
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FinishEvent":
        return cls(
            finish_reason=d["finish_reason"],
            usage=Usage.from_dict(d.get("usage", {})),
            error=d.get("error"),
        )


BackendEvent = Union[TokenEvent, FinishEvent]


def event_from_dict(d: Dict[str, Any]) -> BackendEvent:
    if d.get("type") == "token":
        return TokenEvent.from_dict(d)
    if d.get("type") == "finish":
        return FinishEvent.from_dict(d)
    raise ValueError(f"Unknown backend event: {d!r}")


@dataclass
class ImagePayload:
    """Image passed through to the engine. Exactly one of `data` (raw bytes,
    base64-encoded on the wire) or `url` is set. The server does no vision
    preprocessing; the engine owns the image pipeline."""

    data: Optional[bytes] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        import base64

        return {
            "data": base64.b64encode(self.data).decode("ascii") if self.data else None,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImagePayload":
        import base64

        data = d.get("data")
        return cls(
            data=base64.b64decode(data) if data else None,
            url=d.get("url"),
        )


class BackendStream:
    """Handle for one submitted request: request_id plus the event stream.

    The stream yields TokenEvent instances and exactly one final FinishEvent.
    Iteration ends after the FinishEvent.
    """

    def __init__(self, request_id: str, events: AsyncIterator[BackendEvent]):
        self.request_id = request_id
        self._events = events

    def __aiter__(self) -> AsyncIterator[BackendEvent]:
        return self._events.__aiter__()

    async def __anext__(self) -> BackendEvent:
        return await self._events.__anext__()


class Backend(ABC):
    """Inference engine interface.

    submit() never blocks on generation: it returns immediately with a
    BackendStream; events are produced as the engine decodes. When the caller
    supplies request_id, the engine MUST use it (cancel() targets it).
    cancel() asks the engine to stop generating for request_id; the stream
    then finishes with finish_reason "cancelled" (or "stop" if it had already
    finished). Returns True if a live request was cancelled.
    """

    @abstractmethod
    async def submit(
        self,
        token_ids: Sequence[int],
        sampling: SamplingParams,
        images: Optional[Sequence[ImagePayload]] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> BackendStream: ...

    async def cancel(self, request_id: str) -> bool:
        return False

    async def close(self) -> None:
        """Release resources (connections, workers)."""
