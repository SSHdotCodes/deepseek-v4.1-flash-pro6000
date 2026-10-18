"""Round 3.1 follow-ups: protocol-aware 429, conversation_id extension,
and the engine contract smoke probe (run against the in-package socket
server implementation of the contract)."""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

import ds41_api.encoding as enc
from ds41_api.encoder import CharTokenizer
from ds41_api.fake_backend import FakeBackend
from ds41_api.server import ServerConfig
from ds41_api.socket_backend import serve_socket

EOS = enc.eos_token
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


# ----------------------------------------------------------------------
# protocol-aware 429


def _hold_slot_request(server):
    """Start a slow streaming generation that holds the single slot."""
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": False,
        "stream": True,
    }
    return httpx.stream(
        "POST",
        f"{server.base}/v1/chat/completions",
        json=payload,
        timeout=30.0,
    )


def test_429_anthropic_shape(server_factory):
    # queue capacity 0: the second concurrent request is rejected outright
    handle = server_factory(
        config=ServerConfig(max_concurrent=1, max_queue=0, queue_timeout=5.0),
        backend_kwargs={"delay": 0.02, "loop_scripts": False},
    )
    handle.script("a slow long stream " * 30 + EOS)
    with _hold_slot_request(handle):
        time.sleep(0.3)  # let the first request take the slot
        resp = httpx.post(
            f"{handle.base}/v1/messages",
            json={"model": "m", "max_tokens": 32,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=30.0,
        )
        assert resp.status_code == 429
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["message"]


def test_429_openai_shape_on_chat(server_factory):
    handle = server_factory(
        config=ServerConfig(max_concurrent=1, max_queue=0, queue_timeout=5.0),
        backend_kwargs={"delay": 0.02, "loop_scripts": False},
    )
    handle.script("a slow long stream " * 30 + EOS)
    with _hold_slot_request(handle):
        time.sleep(0.3)
        resp = httpx.post(
            f"{handle.base}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            timeout=30.0,
        )
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["type"] == "rate_limit_error"


# ----------------------------------------------------------------------
# conversation_id extension (and metadata.user_id NOT abused)


def test_messages_conversation_id_extension(server_factory):
    handle = server_factory()
    handle.script("ok" + EOS)
    resp = httpx.post(
        f"{handle.base}/v1/messages",
        json={"model": "m", "max_tokens": 32,
              "conversation_id": "sess-123",
              "metadata": {"user_id": "user-abc"},
              "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 200
    assert handle.backend.submitted[-1]["conversation_id"] == "sess-123"

    # without the extension, no conversation hint is sent
    handle.script("ok" + EOS)
    resp = httpx.post(
        f"{handle.base}/v1/messages",
        json={"model": "m", "max_tokens": 32,
              "metadata": {"user_id": "user-abc"},
              "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 200
    assert handle.backend.submitted[-1]["conversation_id"] is None


def test_responses_conversation_id_extension(server_factory):
    handle = server_factory()
    handle.script("ok" + EOS)
    resp = httpx.post(
        f"{handle.base}/v1/responses",
        json={"model": "m", "input": "hi", "conversation_id": "sess-456"},
        timeout=30.0,
    )
    assert resp.status_code == 200
    assert handle.backend.submitted[-1]["conversation_id"] == "sess-456"


def test_conversation_id_type_error(server_factory):
    handle = server_factory()
    resp = httpx.post(
        f"{handle.base}/v1/messages",
        json={"model": "m", "max_tokens": 32, "conversation_id": 7,
              "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 400


# ----------------------------------------------------------------------
# engine contract smoke probe (against the in-package socket server)


def test_engine_contract_smoke(tmp_path, capsys, monkeypatch):
    import os
    from pathlib import Path as _P

    tok_path = _P(os.environ.get("DS41_TOKENIZER", _P(__file__).resolve().parents[2] / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))
    if not tok_path.exists():
        pytest.skip("smoke probe needs the real tokenizer")
    from ds41_api.encoder import Encoder

    encoder = Encoder.from_file(str(tok_path))

    extra = {
        "cached_prompt_tokens": 2,
        "prefill_ms": 1.25,
        "decode_ms": 2.5,
        "draft_tokens": 16,
        "accepted_draft_tokens": 12,
    }
    backend = FakeBackend(
        tokenizer=encoder.tokenizer, delay=0.02, loop_scripts=False,
        usage_extra=extra,
    )
    backend.add_script("short answer")            # smoke-1 (<= 16 tokens)
    backend.add_script("a much longer completion here")  # smoke-2 (> 5 tokens)
    backend.add_script("streaming token " * 100)  # smoke-3 (cancelled)

    sock = tmp_path / "engine.sock"
    started = asyncio.Event()
    loop = asyncio.new_event_loop()
    holder = {}

    def run_server():
        asyncio.set_event_loop(loop)
        holder["task"] = asyncio.ensure_future(serve_socket(backend, sock, started))
        loop.run_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not sock.exists() and time.time() < deadline:
            time.sleep(0.05)

        sys.path.insert(0, str(TOOLS_DIR))
        import engine_contract_smoke

        monkeypatch.setattr(
            sys, "argv",
            ["engine_contract_smoke.py", "--socket", str(sock),
             "--request-timeout", "20"],
        )
        code = engine_contract_smoke.main()
        out = capsys.readouterr().out
        assert "SMOKE RESULT: PASS" in out, out
        assert code == 0
        # the optional usage extension fields were validated
        assert "PASS usage.cached_prompt_tokens" in out
        assert "PASS usage.prefill_ms" in out
        assert "PASS usage.draft_tokens" in out
        assert "PASS live cancel" in out
        # unknown sampling fields must have been ignored (submission worked)
        assert "PASS finish_reason" in out
    finally:
        task = holder.get("task")
        if task is not None:
            loop.call_soon_threadsafe(task.cancel)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
