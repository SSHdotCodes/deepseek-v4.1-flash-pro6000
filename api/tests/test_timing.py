"""Tests for the ds41_timing usage extension and engine-reported usage fields."""

import asyncio
import time

import httpx

import ds41_api.encoding as enc
from ds41_api.backend import Usage
from ds41_api.encoder import Encoder, CharTokenizer
from ds41_api.timing import TimingCollector

EOS = enc.eos_token
TH_END = enc.thinking_end_token


# ----------------------------------------------------------------------
# unit: TimingCollector


def test_collector_engine_reported_fields():
    collector = TimingCollector()
    collector.submit_started()
    collector.token()
    collector.token()
    collector.finish()
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=2,
        cached_prompt_tokens=6,
        prefill_ms=12.5,
        decode_ms=20.0,
        draft_tokens=8,
        accepted_draft_tokens=6,
    )
    stats = collector.stats(usage)
    assert stats["prefill_ms"] == 12.5  # engine wins over wall clock
    assert stats["decode_ms"] == 20.0
    assert stats["cached_prompt_tokens"] == 6
    assert stats["cached_prompt_fraction"] == 0.6
    assert stats["draft_tokens"] == 8
    assert stats["accepted_draft_tokens"] == 6
    assert stats["draft_acceptance_rate"] == 0.75
    assert "queue_ms" not in stats  # no queue wait recorded


def test_collector_wall_clock_fallback():
    collector = TimingCollector()
    collector.waited_in_queue(0.004)
    collector.submit_started()
    collector.token()
    collector.token()
    collector.finish()
    stats = collector.stats(None)
    assert stats["queue_ms"] == 4.0
    assert stats["prefill_ms"] >= 0.0
    assert stats["decode_ms"] >= 0.0
    assert stats["decode_tok_s"] > 0


def test_collector_no_tokens():
    collector = TimingCollector()
    collector.submit_started()
    collector.finish()
    stats = collector.stats(None)
    # No first token: prefill/decode fall back to nothing reported.
    assert "decode_tok_s" not in stats


def test_usage_roundtrip_optional_fields():
    usage = Usage(prompt_tokens=5, completion_tokens=3, cached_prompt_tokens=4,
                  prefill_ms=1.5, decode_ms=2.5, draft_tokens=9, accepted_draft_tokens=3)
    d = usage.to_dict()
    assert d == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "cached_prompt_tokens": 4,
        "prefill_ms": 1.5,
        "decode_ms": 2.5,
        "draft_tokens": 9,
        "accepted_draft_tokens": 3,
    }
    assert Usage.from_dict({"prompt_tokens": 5, "completion_tokens": 3}) == Usage(5, 3)
    back = Usage.from_dict(d)
    assert back == usage


# ----------------------------------------------------------------------
# e2e: ds41_timing in the OpenAI usage block


def test_chat_nonstream_ds41_timing(server_factory):
    handle = server_factory()
    handle.script("Hello there." + EOS)
    resp = httpx.post(
        f"{handle.base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert "ds41_timing" in usage
    timing = usage["ds41_timing"]
    assert timing["prefill_ms"] >= 0.0
    assert timing["decode_ms"] >= 0.0
    assert timing["decode_tok_s"] > 0
    # FakeBackend stops at EOS: completion tokens exclude the EOS itself.
    assert usage["completion_tokens"] == len(
        handle.backend.submitted[-1]["script"]
    ) - 1


def test_chat_streaming_ds41_timing(server_factory):
    handle = server_factory()
    handle.script("Thinking." + TH_END + "Done." + EOS)
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    usage = None
    with httpx.stream(
        "POST",
        f"{handle.base}/v1/chat/completions", json=payload, timeout=30.0
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = __import__("json").loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
    assert usage is not None
    assert "ds41_timing" in usage
    assert usage["ds41_timing"]["decode_tok_s"] > 0
    assert usage["completion_tokens_details"]["reasoning_tokens"] > 0


def test_engine_reported_usage_passes_through(server_factory):
    extra = {
        "cached_prompt_tokens": 42,
        "prefill_ms": 3.25,
        "decode_ms": 8.5,
        "draft_tokens": 100,
        "accepted_draft_tokens": 64,
    }
    handle = server_factory(backend_kwargs={"usage_extra": extra})
    handle.script("Reuse the cache." + EOS)
    resp = httpx.post(
        f"{handle.base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    usage = body["usage"]
    assert usage["prompt_tokens_details"]["cached_tokens"] == 42
    timing = usage["ds41_timing"]
    assert timing["cached_prompt_tokens"] == 42
    assert timing["prefill_ms"] == 3.25
    assert timing["decode_ms"] == 8.5
    assert timing["draft_tokens"] == 100
    assert timing["accepted_draft_tokens"] == 64
    assert timing["draft_acceptance_rate"] == 0.64
