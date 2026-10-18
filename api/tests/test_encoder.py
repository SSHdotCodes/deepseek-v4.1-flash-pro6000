"""Checkpoint 1: the wrapper's prompt construction is byte-identical to the
reference golden outputs on every reference test case."""

import os
import json
from pathlib import Path

import pytest

import ds41_api.encoding as enc_ref  # unchanged reference copy
from ds41_api.encoder import CharTokenizer, Encoder
from ds41_api.schemas import ChatCompletionRequest, resolve_thinking_mode

FIXTURES = Path(__file__).resolve().parent.parent / "ds41_api" / "tests"
WEIGHTS_TOKENIZER = Path(os.environ.get("DS41_TOKENIZER", Path(__file__).resolve().parents[2] / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))

try:
    from tokenizers import Tokenizer  # noqa: F401

    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False


def make_encoder() -> Encoder:
    if HAVE_TOKENIZERS and WEIGHTS_TOKENIZER.exists():
        return Encoder.from_file(str(WEIGHTS_TOKENIZER))
    return Encoder(CharTokenizer())


def wrap_request(case: dict) -> ChatCompletionRequest:
    """Reference case -> the ChatCompletionRequest the server would receive.

    load_cases() attaches case["tools"] to messages[0]; in the server flow the
    tools arrive as a top-level field, so move them back there.
    """
    messages = json.loads(json.dumps(case["messages"]))
    tools = None
    if messages and "tools" in messages[0]:
        tools = messages[0].pop("tools")
    return ChatCompletionRequest(
        messages=messages,
        context=case.get("context"),
        thinking=case.get("thinking_mode") == "thinking",
        reasoning_effort=case.get("reasoning_effort"),
        tools=tools,
    )


CASE_IDS = [p.stem.split("_")[-1] for p in sorted(FIXTURES.glob("test_input_*.json"))]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_byte_identical_prompt_via_wrapper(case_id):
    input_file = FIXTURES / f"test_input_{case_id}.json"
    golden = (FIXTURES / f"test_output_{case_id}.txt").read_text(encoding="utf-8")

    for case in enc_ref.load_cases(str(input_file)):
        request = wrap_request(case)
        thinking_mode = resolve_thinking_mode(request.thinking)
        prompt, _images = make_encoder().build_prompt(request)
        assert prompt == golden, (
            f"prompt differs from test_output_{case_id}.txt "
            f"(thinking_mode={thinking_mode})"
        )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_reference_direct_byte_identical(case_id):
    """Reference encode_case path against the same goldens (sanity: fixtures
    are not stale)."""
    input_file = FIXTURES / f"test_input_{case_id}.json"
    golden = (FIXTURES / f"test_output_{case_id}.txt").read_text(encoding="utf-8")
    for case in enc_ref.load_cases(str(input_file)):
        # Reference golden files are generated with a "chat" fallback.
        prompt, _ = enc_ref.encode_case(case, thinking_mode="chat")
        assert prompt == golden


def test_token_ids_roundtrip():
    enc_obj = make_encoder()
    prompt, _ = enc_obj.build_prompt(
        ChatCompletionRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            thinking=True,
        )
    )
    ids = enc_obj.encode_prompt(prompt)
    assert ids, "empty token ids"
    assert enc_obj.decode_ids(ids) == prompt, "token ids must round-trip"


def test_incremental_detokenizer_matches_full_decode():
    enc_obj = make_encoder()
    prompt, _ = enc_obj.build_prompt(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "héllo wörld 你好 emoji \U0001F600"}],
            thinking=True,
        )
    )
    ids = enc_obj.encode_prompt(prompt)
    detok = enc_obj.detokenizer()
    streamed = "".join(detok.feed(i) for i in ids) + detok.flush()
    assert streamed == prompt


def _dsml_invoke(name: str, params: str) -> str:
    return (
        f"<{enc_ref.dsml_token}{enc_ref.tool_call_tag_name} name=\"{name}\">\n"
        f"{params}\n"
        f"</{enc_ref.dsml_token}{enc_ref.tool_call_tag_name}>"
    )


def test_parse_completion_thinking_with_tool_call():
    param = (
        f"<{enc_ref.dsml_token}{enc_ref.tool_parameter_tag_name} "
        'name="city" string="true">Paris</'
        f"{enc_ref.dsml_token}{enc_ref.tool_parameter_tag_name}>"
    )
    param2 = (
        f"<{enc_ref.dsml_token}{enc_ref.tool_parameter_tag_name} "
        'name="days" string="false">3</'
        f"{enc_ref.dsml_token}{enc_ref.tool_parameter_tag_name}>"
    )
    text = (
        "I should check the weather."
        + enc_ref.thinking_end_token
        + "\n\n<"
        + enc_ref.dsml_token
        + enc_ref.tool_calls_block_name
        + ">\n"
        + _dsml_invoke("get_weather", param + "\n" + param2)
        + "\n</"
        + enc_ref.dsml_token
        + enc_ref.tool_calls_block_name
        + ">"
        + enc_ref.eos_token
    )
    msg = Encoder.parse_completion(text, "thinking")
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "I should check the weather."
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris", "days": 3}


def test_parse_completion_chat_plain():
    text = "Hello there!" + enc_ref.eos_token
    msg = Encoder.parse_completion(text, "chat")
    assert msg["content"] == "Hello there!"
    assert msg["reasoning_content"] == ""
    assert msg["tool_calls"] == []


def test_tool_choice_validation():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="required",
    )
    with pytest.raises(ValueError):
        make_encoder().build_prompt(request)


def test_tool_choice_none_drops_tools():
    request = ChatCompletionRequest(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="none",
    )
    prompt, _ = make_encoder().build_prompt(request)
    assert enc_ref.dsml_token not in prompt
