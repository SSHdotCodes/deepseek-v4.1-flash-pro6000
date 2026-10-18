"""Streaming parser state machine tests.

Key property: feed the same completion text in arbitrary chunk sizes and the
concatenated deltas must reproduce the reference parser's message (and never
leak delimiter text into content/reasoning).
"""

import json
import random

import ds41_api.encoding as enc
from ds41_api.encoder import Encoder
from ds41_api.parser import StreamParser

D = enc.dsml_token
TH_END = enc.thinking_end_token
EOS = enc.eos_token


def invoke(name: str, params: dict) -> str:
    from ds41_api.encoder import Encoder as _  # noqa: F401

    args = enc.encode_arguments_to_dsml(
        {"name": name, "arguments": params}
    )
    return enc.tool_call_template.format(
        dsml_token=D,
        tool_call_tag_name=enc.tool_call_tag_name,
        name=name,
        arguments=args,
    )


def calls_block(*invokes: str) -> str:
    return enc.tool_calls_template.format(
        dsml_token=D,
        tool_calls="\n".join(invokes),
        tc_block_name=enc.tool_calls_block_name,
    )


def replay_random_chunks(text: str, thinking_mode: str, seed: int = 0):
    rng = random.Random(seed)
    parser = StreamParser(thinking_mode)
    deltas = []
    i = 0
    while i < len(text):
        n = rng.randint(1, 5)
        deltas.extend(parser.feed(text[i : i + n]))
        i += n
    parser.finish_truncated()
    return parser, deltas


def apply_deltas(deltas) -> dict:
    """OpenAI client-side accumulation of deltas -> message."""
    msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
    for d in deltas:
        if "reasoning_content" in d:
            msg["reasoning_content"] += d["reasoning_content"]
        if "content" in d:
            msg["content"] += d["content"]
        for tc in d.get("tool_calls", []):
            idx = tc["index"]
            call = msg["tool_calls"].setdefault(
                idx, {"id": None, "name": None, "arguments": ""}
            )
            if "id" in tc:
                call["id"] = tc["id"]
                call["name"] = tc["function"]["name"]
            call["arguments"] += tc["function"].get("arguments", "")
    return msg


COMPLETIONS = {
    "thinking_plain": (
        "I need to greet.\nChecking tone."
        + TH_END
        + "Hello! How can I help?"
        + EOS,
        "thinking",
    ),
    "thinking_tools": (
        "User wants weather. Call the tool."
        + TH_END
        + "\n\n"
        + "<" + D + enc.tool_calls_block_name + ">\n"
        + invoke("get_weather", {"city": "Paris", "days": 3})
        + "\n</" + D + enc.tool_calls_block_name + ">"
        + EOS,
        "thinking",
    ),
    "chat_plain": ("Just answer plainly." + EOS, "chat"),
    "chat_tools": (
        "\n\n"
        + "<" + D + enc.tool_calls_block_name + ">\n"
        + invoke("search", {"q": "deepseek v4.1", "limit": 5})
        + "\n</" + D + enc.tool_calls_block_name + ">"
        + EOS,
        "chat",
    ),
    "chat_two_tools": (
        "Two calls needed."
        + "\n\n"
        + "<" + D + enc.tool_calls_block_name + ">\n"
        + invoke("a", {"x": "one"})
        + "\n"
        + invoke("b", {"y": 2})
        + "\n</" + D + enc.tool_calls_block_name + ">"
        + EOS,
        "chat",
    ),
}


def test_parser_matches_reference_on_fixed_completions():
    for name, (text, mode) in COMPLETIONS.items():
        ref = Encoder.parse_completion(text, mode)
        parser = StreamParser(mode)
        deltas = []
        for i in range(len(text)):
            deltas.extend(parser.feed(text[i]))
        parser.finish_truncated()
        got = parser.message()
        assert got["content"] == ref["content"], name
        assert got["reasoning_content"] == ref["reasoning_content"], name
        assert len(got["tool_calls"]) == len(ref["tool_calls"]), name
        for g, r in zip(got["tool_calls"], ref["tool_calls"]):
            assert g["function"]["name"] == r["function"]["name"], name
            assert json.loads(g["function"]["arguments"]) == json.loads(
                r["function"]["arguments"]
            ), name


def test_parser_random_chunks_stream_semantics():
    for name, (text, mode) in COMPLETIONS.items():
        ref = Encoder.parse_completion(text, mode)
        for seed in range(3):
            parser, deltas = replay_random_chunks(text, mode, seed)
            acc = apply_deltas(deltas)
            assert acc["content"] == ref["content"], (name, seed)
            assert acc["reasoning_content"] == ref["reasoning_content"], (name, seed)
            assert len(acc["tool_calls"]) == len(ref["tool_calls"]), (name, seed)
            for idx, call in acc["tool_calls"].items():
                r = ref["tool_calls"][idx]
                assert call["name"] == r["function"]["name"], (name, seed)
                assert json.loads(call["arguments"]) == json.loads(
                    r["function"]["arguments"]
                ), (name, seed)


def test_no_delimiter_leak_into_content():
    # A completion whose content *starts* to look like the tool-call delimiter
    # but resolves into plain text.
    text = "Here is a fake tag: \n\n<" + D + enc.tool_calls_block_name[:-1] + "x... done" + EOS
    parser = StreamParser("chat")
    deltas = []
    for ch in text:
        deltas.extend(parser.feed(ch))
    parser.finish_truncated()
    acc = apply_deltas(deltas)
    assert acc["content"] == text[: -len(EOS)]
    assert parser.tool_calls == []


def test_user_stop_sequence():
    text = "one two three four" + EOS
    parser = StreamParser("chat", stop_sequences=["three"])
    deltas = []
    for ch in text:
        if parser.state == "done":
            break
        deltas.extend(parser.feed(ch))
    parser.finish_truncated()
    acc = apply_deltas(deltas)
    assert acc["content"] == "one two "
    assert parser.stopped_on_stop


def test_finish_truncated_mid_tool_call_keeps_json_valid():
    full = invoke("get_weather", {"city": "Paris", "days": 3})
    text = "\n\n" + "<" + D + enc.tool_calls_block_name + ">\n" + full[: len(full) // 2]
    parser = StreamParser("chat")
    deltas = []
    for ch in text:
        deltas.extend(parser.feed(ch))
    deltas.extend(parser.finish_truncated())
    acc = apply_deltas(deltas)
    args = acc["tool_calls"][0]["arguments"]
    # Must be balanced JSON (possibly truncated values dropped).
    json.loads(args)
    assert acc["tool_calls"][0]["name"] == "get_weather"


def test_thinking_never_emits_th_end_text():
    text = "reason" + TH_END + "answer" + EOS
    parser = StreamParser("thinking")
    for ch in text:
        for d in parser.feed(ch):
            assert TH_END not in d.get("reasoning_content", "")
            assert TH_END not in d.get("content", "")
    parser.finish_truncated()
