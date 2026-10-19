"""Encoding layer: OpenAI-style request -> reference prompt -> token ids,
and completion text -> parsed message.

The reference encoder lives in ds41_api.encoding (an unchanged copy of the
official encoding/encoding.py from the weights dir). This module is a thin,
non-invasive wrapper around it.
"""

import base64
import binascii
import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import encoding as enc
from .backend import ImagePayload
from .schemas import ChatCompletionRequest, resolve_thinking_mode

# Model id reported by the server and matched in requests.
DEFAULT_MODEL_ID = "deepseek-v4.1-flash-1017"


class CharTokenizer:
    """Fallback tokenizer used only for tests/CI when tokenizer.json is not
    available. Character-level: encode/decode are exact inverses and special
    tokens round-trip as ordinary text."""

    def _encode(self, text: str):
        ids = [ord(c) for c in text]
        return _Encoding(ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._encode(text)

    def encode_batch(self, texts, add_special_tokens: bool = False):
        return [self._encode(t) for t in texts]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


class _Encoding:
    def __init__(self, ids: List[int]):
        self.ids = ids
        self.tokens: List[str] = []


class Encoder:
    """Tokenizes and detokenizes. Wraps a `tokenizers.Tokenizer` loaded from
    tokenizer.json (same dir as the weights); CharTokenizer for tests."""

    def __init__(self, tokenizer: Any, vision: Optional[Any] = None):
        self.tokenizer = tokenizer
        # ds41_api.vision.VisionConfig when the checkpoint has a vision tower (image prompts), else None.
        self.vision = vision

    @classmethod
    def from_file(cls, tokenizer_path: str, allow_remote_images: bool = True) -> "Encoder":
        from pathlib import Path

        from tokenizers import Tokenizer

        from .vision import VisionConfig

        vision = VisionConfig.from_weights(Path(tokenizer_path).parent, allow_remote=allow_remote_images)
        return cls(Tokenizer.from_file(tokenizer_path), vision)

    # -- text <-> ids ---------------------------------------------------

    def encode_prompt(self, prompt: str) -> List[int]:
        """Prompt string (already contains all special tokens, including BOS)
        -> token ids. add_special_tokens=False keeps the encoder from adding
        another BOS."""
        enc_obj = self.tokenizer.encode(prompt, add_special_tokens=False)
        return list(enc_obj.ids)

    def decode_ids(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def decode_one(self, token_id: int) -> str:
        """Decode a single token id (may be a partial UTF-8 fragment)."""
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def token_id(self, text: str) -> Optional[int]:
        """Token id for a special token string, or None if not in the vocab."""
        tok = self.tokenizer
        if hasattr(tok, "token_to_id"):
            return tok.token_to_id(text)
        if isinstance(tok, CharTokenizer) and len(text) == 1:
            return ord(text)
        return None

    def detokenizer(self) -> "IncrementalDetokenizer":
        return IncrementalDetokenizer(self.tokenizer)

    # -- request -> prompt ----------------------------------------------

    def build_prompt(
        self, request: ChatCompletionRequest
    ) -> Tuple[str, List[ImagePayload]]:
        """OpenAI-style request -> (reference prompt string, image payloads).

        Byte-identical to encoding/encode_case on the reference test cases:
        tools/response_format are attached to messages[0] exactly like
        load_cases() in the reference test file.
        """
        thinking_mode = resolve_thinking_mode(request.thinking)
        tool_choice = request.tool_choice
        if tool_choice not in (None, "auto", "none"):
            raise ValueError(
                "tool_choice must be 'auto' or 'none'; the reference encoder "
                f"does not support forced tool choice (got {tool_choice!r})"
            )

        messages = copy.deepcopy(request.messages)
        attach_tools = request.tools is not None and tool_choice != "none"
        # The reference encoder renders tools/response_format only on a
        # system-role messages[0] and silently drops them otherwise. The
        # official production toolkit (deepseek-recipe 0.1.1) renders them
        # as if an empty system message were present (token-identical), so
        # synthesize one, like /v1/messages and /v1/responses (lead decision).
        if (attach_tools or request.response_format is not None) and (
            not messages or messages[0].get("role") != "system"
        ):
            messages.insert(0, {"role": "system", "content": ""})
        if messages:
            if attach_tools:
                messages[0]["tools"] = request.tools
            if request.response_format is not None:
                messages[0]["response_format"] = request.response_format

        prompt, media = enc.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            context=request.context,
            reasoning_effort=request.reasoning_effort,
            return_multi_modal_data=True,
        )
        return prompt, [image_payload(img) for img in media["images"]]

    # -- completion -> message ------------------------------------------

    @staticmethod
    def parse_completion(text: str, thinking_mode: str) -> Dict[str, Any]:
        """Completion text (including delimiter tokens) -> message dict with
        role/content/reasoning_content/tool_calls (OpenAI tool call format)."""
        return enc.parse_message_from_completion_text(text, thinking_mode)


_DATA_URI_RE = re.compile(r"^data:[^;,]*?(;base64)?,")


def image_payload(record: Dict[str, Any]) -> ImagePayload:
    """Convert a reference encoder image record into a backend ImagePayload.

    data: URIs, plain base64 in `data` and Anthropic base64 sources are decoded
    to raw bytes; http(s) URLs pass through for ds41_api.vision to fetch
    (public addresses only). Fields of the wrong type are a ValueError.
    """
    # Anthropic blocks carry a source object: {"type": "base64", "data": ...} or {"type": "url", "url": ...}
    # (official image_processor.load_image_bytes reads them the same way).
    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            if not isinstance(source["data"], str):
                raise ValueError("image source data must be a base64 string")
            try:
                return ImagePayload(data=base64.b64decode(source["data"]))
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"undecodable base64 image source: {exc}") from exc
        if source.get("url"):
            if not isinstance(source["url"], str):
                raise ValueError("image source url must be a string")
            record = {"url": source["url"]}
    for key in ("url", "data"):
        if record.get(key) is not None and not isinstance(record[key], (str, bytes)):
            raise ValueError(f"image {key} must be a string")
    data = record.get("data")
    if isinstance(data, str) and data:
        try:
            return ImagePayload(data=base64.b64decode(data))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"undecodable base64 image data: {exc}") from exc
    url = record.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        m = _DATA_URI_RE.match(url)
        if m:
            payload = url[m.end():]
            try:
                if m.group(1):  # ;base64
                    return ImagePayload(data=base64.b64decode(payload))
                from urllib.parse import unquote_to_bytes

                return ImagePayload(data=unquote_to_bytes(payload))
            except (binascii.Error, ValueError):
                # Undecodable data URI: pass the whole URI to the engine.
                return ImagePayload(url=url)
    if url is not None:
        return ImagePayload(url=url)
    if "data" in record and record["data"]:
        return ImagePayload(data=bytes(record["data"]))
    if "source" in record and record["source"]:
        return ImagePayload(url=str(record["source"]))
    raise ValueError("Image record without url/source/data")


class IncrementalDetokenizer:
    """Incremental text decoder over a token-id stream.

    Feeds ids one at a time; emits only complete text. A token whose decode
    still contains the Unicode replacement character (a UTF-8 sequence split
    across tokens) is held back until the sequence completes.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._pending: List[int] = []

    def feed(self, token_id: int) -> str:
        """Add one token id, return any newly completed text ('' if none)."""
        self._pending.append(token_id)
        text = self.tokenizer.decode(self._pending, skip_special_tokens=False)
        if "\ufffd" in text:
            return ""
        self._pending = []
        return text

    def flush(self) -> str:
        """Return any remaining partial text (should be empty after EOS)."""
        if not self._pending:
            return ""
        text = self.tokenizer.decode(self._pending, skip_special_tokens=False)
        self._pending = []
        return text
