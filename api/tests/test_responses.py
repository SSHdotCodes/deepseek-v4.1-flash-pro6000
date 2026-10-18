"""OpenAI Responses API tests (POST /v1/responses)."""

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
    invoke("get_weather", {"city": "Paris"})
) + EOS


def post_responses(server, body):
    return httpx.post(f"{server.base}/v1/responses", json=body, timeout=30.0)


def base_body(input_value, **extra):
    body = {"model": "gpt-5-test", "input": input_value}
    body.update(extra)
    return body


# ----------------------------------------------------------------------
# non-streaming


def test_plain_input_string(server_factory):
    handle = server_factory()
    handle.script("Sure." + TH_END + "Hello there." + EOS)
    resp = post_responses(handle, base_body("hi", instructions="Be terse."))
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    messages = [i for i in body["output"] if i["type"] == "message"]
    assert len(messages) == 1
    item = messages[0]
    assert item["role"] == "assistant"
    assert item["status"] == "completed"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "Hello there."
    assert body["usage"]["output_tokens"] > 0
    assert body["usage"]["ds41_timing"]["decode_tok_s"] > 0
    prompt = handle.decoded_prompt()
    assert "Be terse." in prompt
    assert "hi" in prompt


def test_reasoning_item_and_effort(server_factory):
    handle = server_factory()
    handle.script("Think it through." + TH_END + "Final: yes." + EOS)
    resp = post_responses(handle, base_body(
        "decide?",
        reasoning={"effort": "high", "summary": "auto"},
    ))
    assert resp.status_code == 200
    body = resp.json()
    kinds = [item["type"] for item in body["output"]]
    assert kinds == ["reasoning", "message"]
    assert body["usage"]["output_tokens_details"]["reasoning_tokens"] > 0
    assert "Reasoning Effort: 75" in handle.decoded_prompt()


def test_function_tool_round_trip(server_factory):
    handle = server_factory()
    handle.script("Call the tool." + TH_END + TOOL_COMPLETION)
    resp = post_responses(handle, base_body(
        [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "weather in Paris?"}]},
            {"type": "function_call", "call_id": "call_01", "name": "get_weather",
             "arguments": json.dumps({"city": "Paris"})},
            {"type": "function_call_output", "call_id": "call_01", "output": "sunny"},
            {"type": "reasoning", "summary": []},
        ],
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
    ))
    assert resp.status_code == 200
    body = resp.json()
    fcs = [i for i in body["output"] if i["type"] == "function_call"]
    assert len(fcs) == 1
    fc = fcs[0]
    assert fc["status"] == "completed"
    assert fc["name"] == "get_weather"
    assert json.loads(fc["arguments"]) == {"city": "Paris"}
    assert fc["call_id"].startswith("call_")
    prompt = handle.decoded_prompt()
    assert "## Tools" in prompt  # schema rendered; no system message was
    assert "get_weather" in prompt  # supplied, so one was synthesized
    assert "<tool_result>sunny</tool_result>" in prompt


def test_max_output_tokens_incomplete(server_factory):
    handle = server_factory()
    handle.script("The quick brown fox jumps over the lazy dog and continues." + EOS)
    resp = post_responses(handle, base_body("story", max_output_tokens=5))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_errors(server_factory):
    handle = server_factory()
    # previous_response_id needs storage we do not have
    resp = post_responses(handle, base_body("hi", previous_response_id="resp_123"))
    assert resp.status_code == 400
    assert "stateless" in resp.json()["error"]["message"]

    resp = post_responses(handle, base_body([
        {"type": "item_reference", "id": "msg_1"},
    ]))
    assert resp.status_code == 400

    resp = post_responses(handle, base_body(
        "hi", tools=[{"type": "function", "name": "f", "parameters": {}}],
        tool_choice="required",
    ))
    assert resp.status_code == 400

    resp = post_responses(handle, base_body("hi", reasoning={"effort": "ultra"}))
    assert resp.status_code == 400

    resp = post_responses(handle, {"model": "m", "input": []})
    assert resp.status_code == 400


def test_text_format_json_object(server_factory):
    handle = server_factory()
    handle.script("brief." + TH_END + '{"a": 1}' + EOS)
    resp = post_responses(handle, base_body(
        "give me json",
        text={"format": {"type": "json_object"}},
    ))
    assert resp.status_code == 200
    assert "## Response Format:" in handle.decoded_prompt()


# ----------------------------------------------------------------------
# streaming


def read_events(handle, body):
    payload = dict(body)
    payload["stream"] = True
    events = []
    with httpx.stream(
        "POST",
        f"{handle.base}/v1/responses", json=payload, timeout=30.0
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


def test_stream_plain(server_factory):
    handle = server_factory()
    handle.script("Sure." + TH_END + "Hello there." + EOS)
    events = read_events(handle, base_body("hi"))
    types = [t for t, _ in events]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert "response.output_item.added" in types
    assert "response.output_text.delta" in types
    assert "response.output_item.done" in types
    assert types[-1] == "response.completed"

    text = "".join(
        data["delta"] for t, data in events if t == "response.output_text.delta"
    )
    assert text == "Hello there."

    completed = [data for t, data in events if t == "response.completed"][0]
    resp_obj = completed["response"]
    assert resp_obj["status"] == "completed"
    messages = [i for i in resp_obj["output"] if i["type"] == "message"]
    assert messages[0]["content"][0]["text"] == "Hello there."
    assert resp_obj["usage"]["output_tokens"] > 0
    assert resp_obj["usage"]["ds41_timing"]["decode_tok_s"] > 0


def test_stream_thinking_and_function_call(server_factory):
    handle = server_factory()
    handle.script("Plan." + TH_END + TOOL_COMPLETION)
    events = read_events(handle, base_body("weather?"))
    types = [t for t, _ in events]
    assert "response.reasoning_text.delta" in types
    assert "response.function_call_arguments.delta" in types

    added = [data for t, data in events if t == "response.output_item.added"]
    kinds = [data["item"]["type"] for data in added]
    assert kinds == ["reasoning", "function_call"]

    args = "".join(
        data["delta"]
        for t, data in events if t == "response.function_call_arguments.delta"
    )
    assert json.loads(args) == {"city": "Paris"}

    completed = [data for t, data in events if t == "response.completed"][0]
    resp_obj = completed["response"]
    assert resp_obj["status"] == "completed"
    fc_items = [i for i in resp_obj["output"] if i["type"] == "function_call"]
    assert len(fc_items) == 1
    assert json.loads(fc_items[0]["arguments"]) == {"city": "Paris"}
