"""OpenAI-compatible request/response schemas (subset used by DS41).

Messages are kept as raw dicts: they follow the OpenAI chat format plus the
DeepSeek extensions the reference encoder understands (content_blocks,
reasoning_content, task, latest_reminder, context, wo_eos, namespace tools).
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


def resolve_thinking_mode(thinking: Optional[Union[bool, str, Dict[str, Any]]]) -> str:
    """Map the `thinking` request field to a reference thinking_mode.

    Accepts: True/False/"enabled"/"disabled"/{"type": "enabled"|"disabled"}.
    Default (None) is thinking, matching the DS41 recommended sampling.
    """
    if thinking is None:
        return "thinking"
    if isinstance(thinking, bool):
        return "thinking" if thinking else "chat"
    if isinstance(thinking, str):
        t = thinking.strip().lower()
        if t in ("enabled", "on", "true", "thinking"):
            return "thinking"
        if t in ("disabled", "off", "false", "chat"):
            return "chat"
        raise ValueError(f"Invalid thinking value: {thinking!r}")
    if isinstance(thinking, dict):
        t = str(thinking.get("type", "")).strip().lower()
        if t in ("enabled", "thinking"):
            return "thinking"
        if t in ("disabled", "chat"):
            return "chat"
        raise ValueError(f"Invalid thinking value: {thinking!r}")
    raise ValueError(f"Invalid thinking value: {thinking!r}")


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions body."""

    model: str = "deepseek-v4.1-flash-1017"
    messages: List[Dict[str, Any]]
    # DS41 extension: prior turns kept verbatim (reference `context` argument).
    context: Optional[List[Dict[str, Any]]] = None

    thinking: Optional[Union[bool, str, Dict[str, Any]]] = None
    # int in [1, 100] or "low"/"high"/"max"; None -> reference default ("high").
    reasoning_effort: Optional[Union[int, str]] = None

    tools: Optional[List[Dict[str, Any]]] = None
    # "auto" (default) | "none". "required" and named forcing are not
    # supported by the reference encoder; rejected with 400.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    response_format: Optional[Dict[str, Any]] = None

    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = 1.0
    top_p: float = 1.0
    seed: Optional[int] = None

    logprobs: bool = False
    top_logprobs: int = 0
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    # DS41 extension: hint for engine-side prefix/KV reuse across turns.
    conversation_id: Optional[str] = None

    def stop_sequences(self) -> List[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)

    def resolved_max_tokens(self, default: int = 4096) -> int:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return default

    def resolved_logprobs(self) -> int:
        """Number of top logprobs the backend should return (0 = off)."""
        if self.logprobs:
            return max(1, self.top_logprobs)
        return 0
