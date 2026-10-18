"""Backend ABC, FakeBackend and Unix-socket JSON-lines transport tests."""

import asyncio
import json
from pathlib import Path

import pytest

from ds41_api.backend import FinishEvent, SamplingParams, TokenEvent
from ds41_api.encoder import CharTokenizer
from ds41_api.fake_backend import FakeBackend
from ds41_api.socket_backend import SocketBackend, serve_socket


async def collect(stream):
    events = []
    async for ev in stream:
        events.append(ev)
    return events


async def test_fake_backend_streams_script():
    tok = CharTokenizer()
    fb = FakeBackend(tokenizer=tok)
    fb.add_script("hello world")
    stream = await fb.submit([1, 2, 3], SamplingParams())
    events = await collect(stream)
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    finish = [e for e in events if isinstance(e, FinishEvent)]
    assert len(finish) == 1 and finish[-1] is events[-1]
    assert tok.decode([t.token_id for t in tokens]) == "hello world"
    assert finish[0].finish_reason == "stop"
    assert finish[0].usage.prompt_tokens == 3
    assert finish[0].usage.completion_tokens == len(tokens)


async def test_fake_backend_max_tokens_and_stop_ids():
    tok = CharTokenizer()
    fb = FakeBackend(tokenizer=tok)
    fb.add_script("abcdef")
    stream = await fb.submit([9], SamplingParams(max_tokens=3))
    events = await collect(stream)
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) == 3
    assert events[-1].finish_reason == "length"

    fb2 = FakeBackend(tokenizer=tok)
    fb2.add_script("abcdef")
    stop_id = ord("c")
    stream = await fb2.submit([9], SamplingParams(stop_token_ids=(stop_id,)))
    events = await collect(stream)
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert tok.decode([t.token_id for t in tokens]) == "ab"
    assert events[-1].finish_reason == "stop"
    assert events[-1].usage.completion_tokens == 2


async def test_fake_backend_logprobs():
    fb = FakeBackend(tokenizer=CharTokenizer())
    fb.add_script("ab")
    stream = await fb.submit([1], SamplingParams(logprobs=3))
    events = await collect(stream)
    token = events[0]
    assert token.logprob is not None
    assert len(token.top_logprobs) == 3
    stream = await fb.submit([1], SamplingParams())
    events = await collect(stream)
    assert events[0].logprob is None and events[0].top_logprobs is None


async def test_fake_backend_cancel():
    fb = FakeBackend(tokenizer=CharTokenizer(), delay=0.01)
    fb.add_script("x" * 100)
    stream = await fb.submit([1], SamplingParams())
    got = []
    async for ev in stream:
        got.append(ev)
        if len(got) == 3:
            ok = await fb.cancel(stream.request_id)
            assert ok
    assert got[-1].finish_reason == "cancelled"
    assert await fb.cancel("nope") is False


async def test_socket_roundtrip(tmp_path: Path):
    tok = CharTokenizer()
    fb = FakeBackend(tokenizer=tok, delay=0.001)
    fb.add_script("reasoning...\n\nanswer" + "\n\nfinal")
    sock = tmp_path / "backend.sock"
    started = asyncio.Event()
    server_task = asyncio.create_task(serve_socket(fb, sock, started))
    await started.wait()

    client = SocketBackend(sock)
    stream = await client.submit([5, 6, 7], SamplingParams(logprobs=2))
    events = await collect(stream)
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert tok.decode([t.token_id for t in tokens]) == "reasoning...\n\nanswer\n\nfinal"
    assert events[-1].finish_reason == "stop"
    assert tokens[0].logprob is not None

    # prompt usage reported by fake backend
    assert events[-1].usage.prompt_tokens == 3

    await client.close()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
    assert not sock.exists()


async def test_socket_cancel(tmp_path: Path):
    fb = FakeBackend(tokenizer=CharTokenizer(), delay=0.01)
    fb.add_script("y" * 200)
    sock = tmp_path / "cancel.sock"
    started = asyncio.Event()
    server_task = asyncio.create_task(serve_socket(fb, sock, started))
    await started.wait()

    client = SocketBackend(sock)
    stream = await client.submit([1], SamplingParams())
    got = []
    async for ev in stream:
        got.append(ev)
        if len(got) == 2:
            await client.cancel(stream.request_id)
    assert got[-1].finish_reason == "cancelled"

    await client.close()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


async def test_socket_error_on_empty_tokens(tmp_path: Path):
    fb = FakeBackend(tokenizer=CharTokenizer())
    sock = tmp_path / "err.sock"
    started = asyncio.Event()
    server_task = asyncio.create_task(serve_socket(fb, sock, started))
    await started.wait()
    client = SocketBackend(sock)
    stream = await client.submit([], SamplingParams())
    events = await collect(stream)
    assert events[-1].finish_reason == "error"
    await client.close()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
