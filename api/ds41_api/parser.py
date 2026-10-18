"""Incremental streaming parser for DeepSeek-V4.1 completions.

State machine: thinking -> content -> tool calls -> done, with strict
hold-back of partial delimiter text so nothing that later turns out to be a
delimiter is ever emitted as message content.

Delimiters are built from the reference constants (imported from
ds41_api.encoding, never hardcoded). The final message mirrors
encoding.parse_message_from_completion_text.

feed() returns OpenAI-style delta dicts:
    {"reasoning_content": str}
    {"content": str}
    {"tool_calls": [{"index": i, "id": ..., "type": "function",
                     "function": {"name": n, "arguments": ""}}]}
    {"tool_calls": [{"index": i, "function": {"arguments": fragment}}]}
Clients concatenate deltas in order; the concatenated arguments of a call
are valid JSON once the call closes.
"""


import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import encoding as enc

_TH_END = enc.thinking_end_token
_EOS = enc.eos_token
_D = enc.dsml_token
_TC_START = "\n\n<" + _D + enc.tool_calls_block_name
_TC_END = "</" + _D + enc.tool_calls_block_name + ">"
_INVOKE_START = "<" + _D + enc.tool_call_tag_name
# NOTE: the reference stop token has no trailing ">" (parse_tool_calls): the
# ">" of the closing tag is consumed by the next ">\n" separator check.
_INVOKE_END = "</" + _D + enc.tool_call_tag_name
_PARAM_START = "<" + _D + enc.tool_parameter_tag_name
_PARAM_END = "/" + _D + enc.tool_parameter_tag_name

_NAME_RE = re.compile(r'^\s*name="(.*?)">\n$', flags=re.DOTALL)
_PARAM_RE = re.compile(r'^ name="(.*?)" string="(true|false)">(.*?)<$', flags=re.DOTALL)

# states: thinking | content | tc_pre | tc_name | tc_param | tc_param_tail
#         | after_tc | done


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


class StreamParser:
    def __init__(self, thinking_mode: str, stop_sequences: Optional[List[str]] = None):
        assert thinking_mode in ("thinking", "chat")
        self.thinking_mode = thinking_mode
        self.stops = [s for s in (stop_sequences or []) if s]
        self.state = "thinking" if thinking_mode == "thinking" else "content"
        self.buffer = ""
        self.stopped_on_stop = False
        # The user stop sequence that fired (None unless stopped_on_stop).
        self.matched_stop: Optional[str] = None
        # Final message accumulation (reference shape).
        self.reasoning_content = ""
        self.content = ""
        self.tool_calls: List[Dict[str, Any]] = []  # {id, name, arguments}
        self._call_index = -1
        self._args_fragments: List[str] = []
        self._emitted_args_len = 0
        self._open = False  # a tool call is currently open

    # ------------------------------------------------------------------
    # string helpers

    @staticmethod
    def _holdback(buf: str, candidates: List[str]) -> int:
        """Longest suffix of buf that is a proper prefix of a candidate."""
        best = 0
        for c in candidates:
            limit = min(len(buf), len(c) - 1)
            for k in range(limit, best, -1):
                if buf.endswith(c[:k]):
                    best = k
                    break
        return best

    @staticmethod
    def _earliest(buf: str, candidates: List[str]) -> Tuple[Optional[int], Optional[str]]:
        best_pos, best = len(buf), None
        for c in candidates:
            pos = buf.find(c)
            if pos != -1 and pos < best_pos:
                best_pos, best = pos, c
        if best is None:
            return None, None
        return best_pos, best

    # ------------------------------------------------------------------
    # driving

    def feed(self, text: str) -> List[Dict[str, Any]]:
        """Feed detokenized completion text; return OpenAI delta dicts."""
        self.buffer += text
        deltas: List[Dict[str, Any]] = []
        while True:
            progressed, new_deltas = self._step()
            deltas.extend(new_deltas)
            if not progressed:
                break
        return deltas

    def _step(self) -> Tuple[bool, List[Dict[str, Any]]]:
        buf = self.buffer
        state = self.state

        if state == "thinking":
            pos, delim = self._earliest(buf, [_TH_END])
            if delim is not None:
                return self._consume_reasoning(
                    buf[:pos], buf, pos + len(_TH_END), "content"
                )
            keep = self._holdback(buf, [_TH_END])
            if keep < len(buf):
                return self._consume_reasoning(
                    buf[: len(buf) - keep], buf, len(buf) - keep, "thinking"
                )
            return False, []

        if state == "content":
            candidates = [_TC_START, _EOS] + self.stops
            pos, delim = self._earliest(buf, candidates)
            if delim is not None:
                out = buf[:pos]
                self.content += out
                self.buffer = buf[pos + len(delim):]
                if delim == _TC_START:
                    self.state = "tc_pre"
                elif delim == _EOS:
                    self.state = "done"
                else:
                    self.stopped_on_stop = True
                    self.matched_stop = delim
                    self.state = "done"
                return True, ([{"content": out}] if out else [])
            keep = self._holdback(buf, candidates)
            if keep < len(buf):
                out = buf[: len(buf) - keep]
                self.buffer = buf[len(buf) - keep:]
                self.content += out
                return True, [{"content": out}]
            return False, []

        if state == "tc_pre":
            pos, delim = self._earliest(buf, [_INVOKE_START, _TC_END])
            if delim is not None:
                prefix = buf[:pos]
                if prefix != ">\n":
                    raise ValueError(
                        f"Tool call format error: expected '>\\n' but got {prefix!r}"
                    )
                self.buffer = buf[pos + len(delim):]
                if delim == _TC_END:
                    self.state = "after_tc"
                else:
                    self._open_call()
                    self.state = "tc_name"
                return True, []
            return False, []

        if state == "tc_name":
            pos, delim = self._earliest(buf, [_PARAM_START, _INVOKE_END])
            if delim is not None:
                header = buf[:pos]
                m = _NAME_RE.findall(header)
                if len(m) != 1:
                    raise ValueError(f"Tool name format error: {header!r}")
                name = m[0]
                self._current_call()["name"] = name
                self.buffer = buf[pos + len(delim):]
                delta = {
                    "tool_calls": [
                        {
                            "index": self._call_index,
                            "id": self._current_call()["id"],
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                }
                if delim == _INVOKE_END:
                    close_delta = self._close_call()
                    self.state = "tc_pre"
                    return True, [delta, close_delta] if close_delta else [delta]
                self.state = "tc_param"
                return True, [delta]
            return False, []

        if state == "tc_param":
            pos = buf.find(_PARAM_END)
            if pos != -1:
                param_text = buf[:pos]
                frag = self._param_fragment(param_text)
                self._args_fragments.append(frag)
                self.buffer = buf[pos + len(_PARAM_END):]
                self.state = "tc_param_tail"
                return True, [self._args_delta()]
            return False, []

        if state == "tc_param_tail":
            pos, delim = self._earliest(buf, [_PARAM_START, _INVOKE_END])
            if delim is not None:
                prefix = buf[:pos]
                if prefix != ">\n":
                    raise ValueError(
                        f"Parameter format error: expected '>\\n' but got {prefix!r}"
                    )
                self.buffer = buf[pos + len(delim):]
                if delim == _INVOKE_END:
                    close_delta = self._close_call()
                    self.state = "tc_pre"
                    return True, [close_delta] if close_delta else []
                self.state = "tc_param"
                return True, []
            return False, []

        if state == "after_tc":
            pos = buf.find(_EOS)
            if pos != -1:
                self.buffer = buf[pos + len(_EOS):]
                self.state = "done"
                return True, []
            hold = self._holdback(buf, [_EOS])
            junk = buf[: len(buf) - hold].strip("\n")
            if junk:
                raise ValueError(f"Unexpected content after tool calls: {junk!r}")
            return False, []

        return False, []  # done: swallow the rest

    # ------------------------------------------------------------------
    # emission helpers

    def _consume_reasoning(self, out: str, buf: str, new_pos: int, next_state: str):
        self.buffer = buf[new_pos:]
        self.state = next_state
        if out:
            self.reasoning_content += out
            return True, [{"reasoning_content": out}]
        return True, []

    def _open_call(self) -> None:
        self._call_index += 1
        self.tool_calls.append({"id": _new_call_id(), "name": None, "arguments": ""})
        self._args_fragments = []
        self._emitted_args_len = 0
        self._open = True

    def _current_call(self) -> Dict[str, Any]:
        return self.tool_calls[-1]

    def _args_so_far(self) -> str:
        return "{" + ", ".join(self._args_fragments)

    def _args_delta(self) -> Dict[str, Any]:
        """Delta with only the not-yet-emitted part of the arguments."""
        args = self._args_so_far()
        frag = args[self._emitted_args_len:]
        self._emitted_args_len = len(args)
        self._current_call()["arguments"] = args
        return {
            "tool_calls": [{"index": self._call_index, "function": {"arguments": frag}}]
        }

    def _close_call(self) -> Optional[Dict[str, Any]]:
        """Close the open call; return the final arguments fragment delta."""
        args = self._args_so_far() + "}"
        frag = args[self._emitted_args_len:]
        self._emitted_args_len = len(args)
        self._current_call()["arguments"] = args
        self._open = False
        if not frag:
            return None
        return {
            "tool_calls": [{"index": self._call_index, "function": {"arguments": frag}}]
        }

    @staticmethod
    def _param_fragment(param_text: str) -> str:
        m = _PARAM_RE.findall(param_text)
        if len(m) != 1:
            raise ValueError(f"Parameter format error: {param_text!r}")
        key, string, value = m[0]
        if string == "true":
            value = enc.to_json(value)
        return f'{enc.to_json(key)}: {value}'

    # ------------------------------------------------------------------
    # completion

    def finish_truncated(self) -> List[Dict[str, Any]]:
        """End of stream without EOS (length/cancel): flush and return final
        deltas, closing any dangling tool call so client JSON stays valid."""
        deltas: List[Dict[str, Any]] = []
        buf = self.buffer
        if buf:
            if self.state == "thinking":
                self.reasoning_content += buf
            elif self.state == "content":
                self.content += buf
            elif self.state in ("tc_param", "tc_param_tail"):
                # Malformed/truncated parameter: do not emit its text as args.
                pass
            self.buffer = ""
        if self._open:
            delta = self._close_call()
            if delta:
                deltas.append(delta)
        self.state = "done"
        return deltas

    def message(self) -> Dict[str, Any]:
        """Final message in the reference parser's shape, plus OpenAI tool
        call ids (the reference parse has no ids; the API needs them)."""
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in self.tool_calls
        ]
        return {
            "role": "assistant",
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": tool_calls,
        }
