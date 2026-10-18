"""FakeBackend: scripted in-process engine for tests and load smoke runs.

Echoes scripted completion texts token by token with a controllable per-token
delay, honoring sampling.max_tokens / stop_token_ids / logprobs and cancel().
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backend import (
    Backend,
    BackendStream,
    FinishEvent,
    SamplingParams,
    TokenEvent,
    Usage,
)


class FakeBackend(Backend):
    def __init__(self, tokenizer: Any = None, delay: float = 0.0, loop_scripts: bool = True,
                 usage_extra: Optional[Dict[str, Any]] = None):
        """tokenizer: anything with .encode(text, add_special_tokens=False).ids
        (a real tokenizers.Tokenizer or the CharTokenizer test fallback).
        Without one, scripts must be token id lists.
        usage_extra: optional fields merged into every finish event's Usage
        (engine-reported timing/cache/draft stats, for tests)."""
        self.tokenizer = tokenizer
        self.delay = delay
        self.loop_scripts = loop_scripts
        self.usage_extra = usage_extra
        self._scripts: List[Sequence[int]] = []
        self._script_pos = 0
        self._live: Dict[str, bool] = {}
        self.submitted: List[dict] = []  # test introspection

    # -- scripting -------------------------------------------------------

    def add_script(self, script: Any) -> None:
        """Add a completion: a text string (tokenized via self.tokenizer) or
        a list of token ids."""
        if isinstance(script, str):
            if self.tokenizer is None:
                raise ValueError("text script needs a tokenizer")
            script = list(self.tokenizer.encode(script, add_special_tokens=False).ids)
        self._scripts.append(list(script))

    def _next_script(self) -> Sequence[int]:
        if not self._scripts:
            raise RuntimeError("FakeBackend has no scripts")
        script = self._scripts[self._script_pos]
        if self._script_pos + 1 < len(self._scripts):
            self._script_pos += 1
        elif self.loop_scripts:
            self._script_pos = 0
        return script

    # -- Backend ---------------------------------------------------------

    async def submit(
        self,
        token_ids: Sequence[int],
        sampling: SamplingParams,
        images: Optional[Sequence] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> BackendStream:
        request_id = request_id or uuid.uuid4().hex
        script = list(self._next_script())[: max(0, sampling.max_tokens)]
        self.submitted.append(
            {
                "request_id": request_id,
                "token_ids": list(token_ids),
                "sampling": sampling,
                "images": list(images or []),
                "conversation_id": conversation_id,
                "script": script,
            }
        )
        self._live[request_id] = True
        stream = self._run(request_id, script, sampling, len(token_ids))
        return BackendStream(request_id, stream)

    def _usage(self, prompt_len: int, completion_len: int) -> Usage:
        base = Usage(prompt_tokens=prompt_len, completion_tokens=completion_len)
        if self.usage_extra:
            merged = base.to_dict()
            merged.update(self.usage_extra)
            return Usage.from_dict(merged)
        return base

    async def _run(
        self,
        request_id: str,
        script: Sequence[int],
        sampling: SamplingParams,
        prompt_len: int,
    ):
        try:
            for pos, token_id in enumerate(script):
                if not self._live.get(request_id):
                    yield FinishEvent(
                        finish_reason="cancelled",
                        usage=self._usage(prompt_len, 0),
                    )
                    return
                if token_id in sampling.stop_token_ids:
                    yield FinishEvent(
                        finish_reason="stop",
                        usage=self._usage(prompt_len, pos),
                    )
                    return
                logprob = None
                top_logprobs = None
                if sampling.logprobs > 0:
                    # Deterministic pseudo-logprobs (seeded by position).
                    base = -0.001 * (pos + 1)
                    logprob = base
                    top_logprobs = [
                        {"token_id": token_id + j, "logprob": base - 0.01 * j}
                        for j in range(sampling.logprobs)
                    ]
                yield TokenEvent(
                    token_id=token_id, logprob=logprob, top_logprobs=top_logprobs
                )
                if self.delay:
                    await asyncio.sleep(self.delay)
            reason = "length" if len(script) >= max(1, sampling.max_tokens) else "stop"
            yield FinishEvent(
                finish_reason=reason,
                usage=self._usage(prompt_len, len(script)),
            )
        finally:
            self._live.pop(request_id, None)

    async def cancel(self, request_id: str) -> bool:
        if request_id in self._live:
            self._live[request_id] = False
            return True
        return False
