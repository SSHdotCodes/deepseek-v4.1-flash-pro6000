"""End-to-end tests: FastAPI server + FakeBackend over real HTTP on 127.0.0.1.

Covers plain chat, thinking, tool calls (streamed + non-streamed), multi-turn
with tool results, logprobs, stop sequences, usage and /v1/models.
Each test gets a fresh server process bound to 127.0.0.1 (ephemeral port),
stopped afterwards.
"""

import os
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest

import ds41_api.encoding as enc
from ds41_api.encoder import CharTokenizer, Encoder
from ds41_api.fake_backend import FakeBackend
from ds41_api.server import create_app

D = enc.dsml_token
TH_END = enc.thinking_end_token
EOS = enc.eos_token

WEIGHTS_TOKENIZER = Path(os.environ.get("DS41_TOKENIZER", Path(__file__).resolve().parents[2] / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))


def make_encoder() -> Encoder:
    if WEIGHTS_TOKENIZER.exists():
        try:
            return Encoder.from_file(str(WEIGHTS_TOKENIZER))
        except Exception:
            pass
    return Encoder(CharTokenizer())


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


# The model emits "\n\n" before the tool calls block (reference
# tool_calls_start_token includes it).
TOOL_COMPLETION = "\n\n" + calls_block(invoke("get_weather", {"city": "Paris", "days": 3})) + EOS


class ServerHandle:
    def __init__(self, port: int, backend: FakeBackend):
        self.port = port
        self.backend = backend
        self.base = f"http://127.0.0.1:{port}"

    def script(self, text: str) -> None:
        self.backend.add_script(text)

    def post(self, payload: dict, stream: bool = False):
        if stream:
            return httpx.post(
                f"{self.base}/v1/chat/completions", json=payload, timeout=30.0
            )
        return httpx.post(f"{self.base}/v1/chat/completions", json=payload, timeout=30.0)


@pytest.fixture()
def server():
    backend = FakeBackend(tokenizer=make_encoder().tokenizer, delay=0.0)
    app = create_app(backend, make_encoder())
    import uvicorn

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvi = uvicorn.Server(config)
    thread = threading.Thread(target=uvi.run, daemon=True)
    thread.start()
    for _ in range(200):
        if uvi.started:
            break
        time.sleep(0.05)
    assert uvi.started, "server did not start"
    handle = ServerHandle(port, backend)
    yield handle
    uvi.should_exit = True
    thread.join(timeout=10)
    assert not thread.is_alive(), "test server did not stop"


def accumulate_sse(resp) -> Dict[str, Any]:
    """OpenAI client-side accumulation of an SSE stream."""
    acc: Dict[str, Any] = {
        "content": "",
        "reasoning_content": "",
        "tool_calls": {},
        "finish_reason": None,
        "usage": None,
    }
    got_done = False
    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            got_done = True
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            acc["usage"] = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "reasoning_content" in delta:
                acc["reasoning_content"] += delta["reasoning_content"]
            if "content" in delta:
                acc["content"] += delta["content"]
            for tc in delta.get("tool_calls", []):
                call = acc["tool_calls"].setdefault(
                    tc["index"], {"id": None, "name": None, "arguments": ""}
                )
                if tc.get("id"):
                    call["id"] = tc["id"]
                    call["name"] = tc["function"]["name"]
                call["arguments"] += tc["function"].get("arguments", "")
            if choice.get("finish_reason"):
                acc["finish_reason"] = choice["finish_reason"]
    assert got_done, "stream must end with [DONE]"
    return acc


# ----------------------------------------------------------------------
# tests


def test_models(server: ServerHandle):
    resp = httpx.get(f"{server.base}/v1/models", timeout=10.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["object"] == "model"


def test_plain_chat_non_streaming(server: ServerHandle):
    server.script("Hello! How can I help you today?" + EOS)
    resp = server.post(
        {"messages": [{"role": "user", "content": "Hi"}], "thinking": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["content"] == "Hello! How can I help you today?"
    assert msg["reasoning_content"] == ""
    assert msg["tool_calls"] == []
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


def test_thinking_non_streaming(server: ServerHandle):
    server.script("Consider the question." + TH_END + "The answer is 42." + EOS)
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "thinking": True,
        }
    )
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning_content"] == "Consider the question."
    assert msg["content"] == "The answer is 42."
    body = resp.json()
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0


def test_tool_call_non_streaming(server: ServerHandle):
    server.script(TOOL_COMPLETION)
    resp = server.post(
        {
            "messages": [
                {"role": "system", "content": "You can use tools."},
                {"role": "user", "content": "Weather in Paris for 3 days?"},
            ],
            "thinking": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "days": {"type": "integer"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    msg = body["choices"][0]["message"]
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"].startswith("call_")
    assert tc["function"]["name"] == "get_weather"
    args = json.loads(tc["function"]["arguments"])
    assert args == {"city": "Paris", "days": 3}
    # The submitted prompt must contain the rendered tool schema.
    submitted = server.backend.submitted[-1]
    encoder = make_encoder()
    prompt_text = encoder.decode_ids(submitted["token_ids"])
    assert "get_weather" in prompt_text
    assert enc.dsml_token in prompt_text


def test_tool_call_streaming(server: ServerHandle):
    server.script(TOOL_COMPLETION)
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "thinking": False,
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}}
            ],
            "stream": True,
        },
        stream=True,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    acc = accumulate_sse(resp)
    assert acc["content"] == ""
    assert acc["finish_reason"] == "tool_calls"
    assert len(acc["tool_calls"]) == 1
    call = acc["tool_calls"][0]
    assert call["name"] == "get_weather"
    assert call["id"].startswith("call_")
    assert json.loads(call["arguments"]) == {"city": "Paris", "days": 3}


def test_thinking_streaming_with_usage(server: ServerHandle):
    server.script("Think briefly." + TH_END + "Final words." + EOS)
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "Go"}],
            "thinking": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
    )
    assert resp.status_code == 200
    acc = accumulate_sse(resp)
    assert acc["reasoning_content"] == "Think briefly."
    assert acc["content"] == "Final words."
    assert acc["finish_reason"] == "stop"
    assert acc["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0


def test_multi_turn_with_tool_results(server: ServerHandle):
    server.script("The weather in Paris will be sunny." + EOS)
    resp = server.post(
        {
            "messages": [
                {"role": "system", "content": "You can use tools."},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Paris"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc123", "content": "22C sunny"},
                {
                    "role": "user",
                    "content": "Great, and in London?",
                },
            ],
            "thinking": False,
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "The weather in Paris will be sunny."
    # The encoded prompt must carry the tool result as a user content block.
    prompt_text = make_encoder().decode_ids(server.backend.submitted[-1]["token_ids"])
    assert "tool_result" in prompt_text
    assert "22C sunny" in prompt_text


def test_logprobs(server: ServerHandle):
    server.script("Sure thing." + EOS)
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": False,
            "logprobs": True,
            "top_logprobs": 2,
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    entries = body["choices"][0]["logprobs"]["content"]
    assert entries, "expected content logprobs"
    first = entries[0]
    assert isinstance(first["token"], str)
    assert isinstance(first["logprob"], (int, float))
    assert len(first.get("top_logprobs", [])) >= 1


def test_stop_sequences(server: ServerHandle):
    server.script("one two three four five" + EOS)
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "count"}],
            "thinking": False,
            "stop": ["three"],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "one two "
    assert body["choices"][0]["finish_reason"] == "stop"


def test_max_tokens_length(server: ServerHandle):
    server.script("a" * 100 + EOS)
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "go"}],
            "thinking": False,
            "max_tokens": 5,
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5


def test_tool_choice_rejected(server: ServerHandle):
    resp = server.post(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
        }
    )
    assert resp.status_code == 400
    assert "tool_choice" in resp.json()["detail"]


def test_default_sampling_reaches_backend(server: ServerHandle):
    server.script("ok" + EOS)
    resp = server.post({"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    sampling = server.backend.submitted[-1]["sampling"]
    assert sampling.temperature == 1.0
    assert sampling.top_p == 1.0
