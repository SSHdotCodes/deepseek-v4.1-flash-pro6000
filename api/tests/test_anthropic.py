"""Anthropic Messages API tests (POST /v1/messages, /v1/messages/count_tokens)."""

import json

import httpx

import ds41_api.encoding as enc

D = enc.dsml_token
TH_END = enc.thinking_end_token
EOS = enc.eos_token


def invoke(name: str, params: dict) -> str:
    args = enc.encode_arguments_to_dsml({"name": name, "arguments": params})
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


TOOL_COMPLETION = "\n\n" + calls_block(
    invoke("get_weather", {"city": "Paris", "days": 3})
) + EOS

TOOLS = [{
    "name": "get_weather",
    "description": "Get the weather",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    },
}]


def post_messages(server, body):
    return httpx.post(f"{server.base}/v1/messages", json=body, timeout=30.0)


def base_body(messages, **extra):
    body = {"model": "claude-3-test", "max_tokens": 128, "messages": messages}
    body.update(extra)
    return body


# ----------------------------------------------------------------------
# non-streaming


def test_plain_chat(server_factory):
    handle = server_factory()
    handle.script("Hello there." + EOS)
    resp = post_messages(handle, base_body([{"role": "user", "content": "hi"}]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "Hello there."}]
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None
    assert body["usage"]["input_tokens"] == len(handle.backend.submitted[-1]["token_ids"])
    assert body["usage"]["output_tokens"] > 0
    assert body["usage"]["ds41_timing"]["decode_tok_s"] > 0


def test_system_prompt_and_multi_turn(server_factory):
    handle = server_factory()
    handle.script("Second answer." + EOS)
    resp = post_messages(handle, base_body(
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        system="You are terse.",
    ))
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "Second answer."
    prompt = handle.decoded_prompt()
    assert "You are terse." in prompt
    assert "first question" in prompt
    assert "first answer" in prompt
    assert "second question" in prompt


def test_mid_conversation_system_message(server_factory):
    handle = server_factory()
    handle.script("Done." + EOS)
    resp = post_messages(handle, base_body(
        [
            {"role": "user", "content": "first question"},
            {"role": "system", "content": [{"type": "text", "text": "Working directory: /tmp/x"}]},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        system="You are terse.",
    ))
    assert resp.status_code == 200
    prompt = handle.decoded_prompt()
    assert prompt.count(enc.SYSTEM_SP_TOKEN) >= 2
    assert enc.SYSTEM_SP_TOKEN + "Working directory: /tmp/x" in prompt
    count = httpx.post(f"{handle.base}/v1/messages/count_tokens", timeout=30.0, json=base_body(
        [{"role": "user", "content": "q"}, {"role": "system", "content": "note"}]))
    assert count.status_code == 200 and count.json()["input_tokens"] > 0

def test_thinking_blocks(server_factory):
    handle = server_factory()
    handle.script("Let me think." + TH_END + "The answer is 42." + EOS)
    resp = post_messages(handle, base_body(
        [{"role": "user", "content": "meaning of life?"}],
        thinking={"type": "enabled", "budget_tokens": 2048},
    ))
    assert resp.status_code == 200
    body = resp.json()
    assert [b["type"] for b in body["content"]] == ["thinking", "text"]
    assert body["content"][0]["thinking"] == "Let me think."
    assert body["content"][1]["text"] == "The answer is 42."
    assert body["stop_reason"] == "end_turn"
    # budget 2048 maps to the reference effort tier "low" (50)
    assert "Reasoning Effort: 50" in handle.decoded_prompt()


def test_thinking_budget_high(server_factory):
    handle = server_factory()
    handle.script("ok" + TH_END + "fine" + EOS)
    resp = post_messages(handle, base_body(
        [{"role": "user", "content": "hi"}],
        thinking={"type": "enabled", "budget_tokens": 16384},
    ))
    assert resp.status_code == 200
    assert "Reasoning Effort: 100" in handle.decoded_prompt()


def test_tool_call_round_trip(server_factory):
    handle = server_factory()
    handle.script(TOOL_COMPLETION)
    resp = post_messages(handle, base_body(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "weather in Paris?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "get_weather",
                     "input": {"city": "Paris", "days": 3}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01",
                     "content": "sunny, 22C"},
                ],
            },
        ],
        tools=TOOLS,
    ))
    assert resp.status_code == 200
    body = resp.json()
    blocks = body["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["name"] == "get_weather"
    assert blocks[0]["input"] == {"city": "Paris", "days": 3}
    assert body["stop_reason"] == "tool_use"
    prompt = handle.decoded_prompt()
    assert "## Tools" in prompt  # tool schema rendered (no system prompt in
    assert "get_weather" in prompt  # this request, so one was synthesized)
    assert "<tool_result>sunny, 22C</tool_result>" in prompt


def test_stop_sequence(server_factory):
    handle = server_factory()
    handle.script("one two three four" + EOS)
    resp = post_messages(handle, base_body(
        [{"role": "user", "content": "count"}],
        stop_sequences=[" two"],
    ))
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == " two"
    assert body["content"][0]["text"] == "one"


def test_max_tokens_stop_reason(server_factory):
    handle = server_factory()
    handle.script("The quick brown fox jumps over the lazy dog and keeps going." + EOS)
    resp = post_messages(handle, base_body(
        [{"role": "user", "content": "story"}],
        max_tokens=5,
    ))
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "max_tokens"
    assert body["content"][0]["text"]


# ----------------------------------------------------------------------
# errors


def test_error_shape_missing_max_tokens(server_factory):
    handle = server_factory()
    resp = httpx.post(
        f"{handle.base}/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in body["error"]["message"]


def test_error_tool_choice_forced(server_factory):
    handle = server_factory()
    resp = post_messages(handle, base_body(
        [{"role": "user", "content": "hi"}],
        tools=TOOLS,
        tool_choice={"type": "any"},
    ))
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    assert "not supported" in body["error"]["message"]


def test_error_budget_too_small(server_factory):
    handle = server_factory()
    resp = post_messages(handle, base_body(
        [{"role": "user", "content": "hi"}],
        thinking={"type": "enabled", "budget_tokens": 10},
    ))
    assert resp.status_code == 400


def test_error_bad_role(server_factory):
    handle = server_factory()
    resp = post_messages(handle, base_body(
        [{"role": "developer", "content": "hi"}],
    ))
    assert resp.status_code == 400
    resp = post_messages(handle, base_body(
        [{"role": "system", "content": "hi"}],
    ))
    assert resp.status_code == 400


# ----------------------------------------------------------------------
# count_tokens


def test_count_tokens(server_factory):
    handle = server_factory()
    resp = httpx.post(
        f"{handle.base}/v1/messages/count_tokens",
        json={
            "model": "claude-3-test",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "And its population?"},
            ],
        },
        timeout=30.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["input_tokens"], int)
    assert body["input_tokens"] > 10
    # No generation happened.
    assert handle.backend.submitted == []


# ----------------------------------------------------------------------
# streaming


def read_events(handle, body):
    payload = dict(body)
    payload["stream"] = True
    events = []
    with httpx.stream(
        "POST",
        f"{handle.base}/v1/messages", json=payload, timeout=30.0
    ) as resp:
        assert resp.status_code == 200
        event_type = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((event_type, json.loads(line[len("data: "):])))
                event_type = None
    return events


def test_stream_plain_chat(server_factory):
    handle = server_factory()
    handle.script("Hello there." + EOS)
    events = read_events(handle, base_body([{"role": "user", "content": "hi"}]))
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types

    text = "".join(
        data["delta"]["text"]
        for t, data in events
        if t == "content_block_delta" and data["delta"]["type"] == "text_delta"
    )
    assert text == "Hello there."
    delta = [data for t, data in events if t == "message_delta"][0]["delta"]
    assert delta["stop_reason"] == "end_turn"
    assert delta["stop_sequence"] is None


def test_stream_thinking_and_tool_use(server_factory):
    handle = server_factory()
    handle.script("Plan the call." + TH_END + TOOL_COMPLETION)
    events = read_events(handle, base_body(
        [{"role": "user", "content": "weather?"}],
        tools=TOOLS,
        thinking={"type": "enabled", "budget_tokens": 2048},
    ))
    types = [t for t, _ in events]

    assert types[0] == "message_start"
    # thinking block first, then a tool_use block
    starts = [data for t, data in events if t == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[1]["content_block"]["type"] == "tool_use"
    assert starts[1]["content_block"]["name"] == "get_weather"

    thinking = "".join(
        data["delta"]["thinking"]
        for t, data in events
        if t == "content_block_delta" and data["delta"]["type"] == "thinking_delta"
    )
    assert thinking == "Plan the call."

    args = "".join(
        data["delta"]["partial_json"]
        for t, data in events
        if t == "content_block_delta" and data["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(args) == {"city": "Paris", "days": 3}

    delta = [data for t, data in events if t == "message_delta"][0]["delta"]
    assert delta["stop_reason"] == "tool_use"

    stops = [data["index"] for t, data in events if t == "content_block_stop"]
    assert stops == sorted(stops) and len(stops) == len(set(stops))


def test_stream_stop_sequence(server_factory):
    handle = server_factory()
    handle.script("one two three four" + EOS)
    events = read_events(handle, base_body(
        [{"role": "user", "content": "count"}],
        stop_sequences=[" three"],
    ))
    delta = [data for t, data in events if t == "message_delta"][0]["delta"]
    assert delta["stop_reason"] == "stop_sequence"
    assert delta["stop_sequence"] == " three"
    text = "".join(
        data["delta"]["text"]
        for t, data in events
        if t == "content_block_delta" and data["delta"]["type"] == "text_delta"
    )
    assert text == "one two"
