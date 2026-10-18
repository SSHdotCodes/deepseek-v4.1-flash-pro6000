"""Robustness tests: body limits, max_tokens caps, the request queue,
client disconnect handling and the loopback bind guard."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

import ds41_api.encoding as enc
from ds41_api.server import ServerConfig

EOS = enc.eos_token
TH_END = enc.thinking_end_token
API_DIR = Path(__file__).resolve().parent.parent
VENV_PY = API_DIR / ".venv/bin/python"


def test_body_size_limit_413(server_factory):
    handle = server_factory(config=ServerConfig(max_body_bytes=1000))
    big = "x" * 5000
    resp = httpx.post(
        f"{handle.base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": big}]},
        timeout=30.0,
    )
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"]["type"] == "request_too_large"


def test_max_tokens_cap_400(server_factory):
    handle = server_factory(config=ServerConfig(max_tokens_limit=8))
    handle.script("word " * 20 + EOS)
    for payload in (
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 16},
    ):
        resp = httpx.post(
            f"{handle.base}/v1/chat/completions", json=payload, timeout=30.0
        )
        assert resp.status_code == 400
        assert "exceeds the server limit" in resp.json()["detail"]

    resp = httpx.post(
        f"{handle.base}/v1/messages",
        json={"model": "m", "max_tokens": 16,
              "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 400

    resp = httpx.post(
        f"{handle.base}/v1/responses",
        json={"model": "m", "input": "hi", "max_output_tokens": 16},
        timeout=30.0,
    )
    assert resp.status_code == 400


def test_queue_429_when_full(server_factory):
    # single-stream engine, room for exactly one waiting request
    config = ServerConfig(max_concurrent=1, max_queue=1, queue_timeout=10.0)
    handle = server_factory(config=config, backend_kwargs={"delay": 0.01})
    for _ in range(3):
        handle.script("streaming a long enough completion " * 30 + EOS)

    results = {}
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    def worker(name):
        try:
            with httpx.stream(
                "POST",
                f"{handle.base}/v1/chat/completions", json=payload, timeout=30.0
            ) as resp:
                # consume to completion
                for _ in resp.iter_lines():
                    pass
                results[name] = resp.status_code
        except httpx.HTTPStatusError as exc:
            results[name] = exc.response.status_code

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join(timeout=60)

    statuses = sorted(results.values())
    # One generates, one queues (both 200); the third is rejected with 429.
    assert 429 in statuses
    assert statuses.count(200) == 2


def test_client_disconnect_cancels_and_releases(server_factory):
    recorded_cancels = []

    handle = server_factory(config=ServerConfig(max_concurrent=1),
                            backend_kwargs={"delay": 0.02, "loop_scripts": False})

    original_cancel = handle.backend.cancel

    async def recording_cancel(request_id):
        recorded_cancels.append(request_id)
        return await original_cancel(request_id)

    handle.backend.cancel = recording_cancel

    handle.script("a long slow stream " * 40 + EOS)
    handle.script("next one." + EOS)
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": False,
        "stream": True,
    }

    with httpx.stream(
        "POST",
        f"{handle.base}/v1/chat/completions", json=payload, timeout=30.0
    ) as resp:
        assert resp.status_code == 200
        # read one event then drop the connection mid-stream
        for line in resp.iter_lines():
            assert line.startswith("data: ")
            break

    # The slot must be released and the engine cancel issued; poll briefly.
    deadline = time.time() + 10
    while time.time() < deadline:
        if recorded_cancels:
            break
        time.sleep(0.1)
    assert recorded_cancels, "backend cancel was not issued on disconnect"

    # gate must be free again: a new request succeeds (it queued behind
    # nothing, so 200 proves the disconnected stream released its slot)
    resp = httpx.post(
        f"{handle.base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "again"}], "thinking": False},
        timeout=30.0,
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "next one."


def test_anthropic_disconnect_releases_slot(server_factory):
    handle = server_factory(config=ServerConfig(max_concurrent=1),
                            backend_kwargs={"delay": 0.02, "loop_scripts": False})
    handle.script("a long slow stream " * 40 + EOS)
    handle.script("after." + EOS)
    body = {
        "model": "m",
        "max_tokens": 1024,
        "stream": True,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    with httpx.stream(
        "POST",
        f"{handle.base}/v1/messages", json=body, timeout=30.0
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            assert line.startswith(("event: ", "data: "))
            break

    # new request must go through (slot released after the disconnect);
    # a 200 here proves the disconnected stream released its generation slot
    deadline = time.time() + 15
    while time.time() < deadline:
        resp = httpx.post(
            f"{handle.base}/v1/messages",
            json={"model": "m", "max_tokens": 64,
                  "thinking": {"type": "disabled"},
                  "messages": [{"role": "user", "content": "again"}]},
            timeout=30.0,
        )
        if resp.status_code == 200:
            assert resp.json()["content"][0]["text"] == "after."
            return
        time.sleep(0.2)
    pytest.fail("anthropic stream did not release the gate after disconnect")


def test_cli_refuses_non_loopback_bind():
    if not VENV_PY.exists():
        pytest.skip("no venv python for CLI test")
    proc = subprocess.run(
        [str(VENV_PY), "-m", "ds41_api.server", "--host", "0.0.0.0", "--port", "0"],
        capture_output=True,
        text=True,
        cwd=str(API_DIR),
        timeout=30,
    )
    assert proc.returncode != 0
    assert "allow-remote" in proc.stderr
