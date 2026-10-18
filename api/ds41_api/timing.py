"""Request timing and usage accounting shared by all protocol endpoints.

Every response (OpenAI chat, Anthropic messages, OpenAI responses) gets a
`ds41_timing` extension object inside its usage block. Values the engine
reports win; server-side measurements are the fallback:

- prefill_ms: engine-reported, else time from submit to first token event
  (minus queue wait), x1000.
- decode_ms: engine-reported, else first-token -> finish wall time x1000.
- decode_tok_s: completion_tokens / decode_ms * 1000.
- queue_ms: time spent waiting for the request gate (concurrency queue).
- cached_prompt_tokens / reused prompt tokens: engine-reported only.
- draft acceptance stats: engine-reported only (speculative decoding).

Contract delta for the engine (backend-api.md v1 is additive-extensible):
finish.usage may optionally include cached_prompt_tokens, prefill_ms,
decode_ms, draft_tokens, accepted_draft_tokens; all default to absent.
"""


import time
from typing import Any, Dict, Optional

from .backend import Usage


class TimingCollector:
    """Wall-clock measurement of one request's lifecycle.

    The API server calls submit_started() once the request leaves the queue,
    token() for every TokenEvent, finish() when the finish event arrives.
    """

    def __init__(self):
        self.queue_ms = 0.0
        self.submit_at: Optional[float] = None
        self.first_token_at: Optional[float] = None
        self.finish_at: Optional[float] = None
        self.completion_tokens = 0

    def waited_in_queue(self, seconds: float) -> None:
        self.queue_ms = max(0.0, seconds) * 1000.0

    def submit_started(self) -> None:
        if self.submit_at is None:
            self.submit_at = time.perf_counter()

    def token(self) -> None:
        self.completion_tokens += 1
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def finish(self) -> None:
        if self.finish_at is None:
            self.finish_at = time.perf_counter()

    # ------------------------------------------------------------------

    def prefill_ms(self, usage: Optional[Usage]) -> Optional[float]:
        if usage is not None and usage.prefill_ms is not None:
            return usage.prefill_ms
        if self.submit_at is not None and self.first_token_at is not None:
            return (self.first_token_at - self.submit_at) * 1000.0
        return None

    def decode_ms(self, usage: Optional[Usage]) -> Optional[float]:
        if usage is not None and usage.decode_ms is not None:
            return usage.decode_ms
        if self.first_token_at is not None and self.finish_at is not None:
            return max(0.0, (self.finish_at - self.first_token_at) * 1000.0)
        return None

    def stats(self, usage: Optional[Usage]) -> Dict[str, Any]:
        """Build the ds41_timing extension object."""
        out: Dict[str, Any] = {}
        if self.queue_ms:
            out["queue_ms"] = round(self.queue_ms, 3)
        prefill = self.prefill_ms(usage)
        if prefill is not None:
            out["prefill_ms"] = round(prefill, 3)
        decode = self.decode_ms(usage)
        if decode is not None:
            out["decode_ms"] = round(decode, 3)
        completion = (
            usage.completion_tokens
            if usage is not None and usage.completion_tokens
            else self.completion_tokens
        )
        if decode is not None and decode > 0 and completion:
            out["decode_tok_s"] = round(completion / (decode / 1000.0), 3)
        if usage is not None:
            if usage.cached_prompt_tokens:
                out["cached_prompt_tokens"] = usage.cached_prompt_tokens
                if usage.prompt_tokens:
                    out["cached_prompt_fraction"] = round(
                        usage.cached_prompt_tokens / usage.prompt_tokens, 4
                    )
            if usage.draft_tokens:
                out["draft_tokens"] = usage.draft_tokens
                out["accepted_draft_tokens"] = usage.accepted_draft_tokens or 0
                out["draft_acceptance_rate"] = round(
                    (usage.accepted_draft_tokens or 0) / usage.draft_tokens, 4
                )
        return out
