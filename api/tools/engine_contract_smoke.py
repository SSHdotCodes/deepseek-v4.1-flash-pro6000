"""backend-api.md conformance smoke for the DS41 engine (ds41d).

Connects to a live engine Unix socket with the raw JSON-lines protocol and
checks the contract items an engine MUST honor, plus the optional timing /
cache / draft usage extensions. Read-only from the API side: it submits a
few short generations and (optionally) one cancel.

Usage:
  .venv/bin/python tools/engine_contract_smoke.py --socket /tmp/ds41d.sock \
      [--tokenizer <tokenizer.json>] [--request-timeout 120]

Exit code 0 = all mandatory checks pass (warnings allowed), 1 otherwise.
Each check prints PASS / FAIL / WARN / INFO with details.
"""

import os
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_ROOT))

WEIGHTS_TOKENIZER = Path(os.environ.get("DS41_TOKENIZER", Path(__file__).resolve().parents[2] / "models/DeepSeek-V4.1-Flash-1017/tokenizer.json"))


class Result:
    def __init__(self):
        self.fails = 0

    def pass_(self, name: str, detail: str = "") -> None:
        print(f"PASS {name}" + (f": {detail}" if detail else ""))

    def fail(self, name: str, detail: str = "") -> None:
        self.fails += 1
        print(f"FAIL {name}" + (f": {detail}" if detail else ""))

    def warn(self, name: str, detail: str = "") -> None:
        print(f"WARN {name}" + (f": {detail}" if detail else ""))

    def info(self, name: str, detail: str = "") -> None:
        print(f"INFO {name}" + (f": {detail}" if detail else ""))


async def request(
    socket_path: str,
    op: Dict[str, Any],
    timeout: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """One connection per request (contract). Returns (events, seconds).

    Events are raw JSON dicts in arrival order. Raises TimeoutError if no
    finish event arrives within `timeout` seconds.
    """
    start = time.perf_counter()
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(socket_path), timeout=timeout
    )
    try:
        writer.write((json.dumps(op) + "\n").encode())
        await writer.drain()
        events: List[Dict[str, Any]] = []
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                raise ConnectionError("connection closed without a finish event")
            obj = json.loads(line)
            events.append(obj)
            if obj.get("type") == "finish":
                return events, time.perf_counter() - start
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def cancel(socket_path: str, request_id: str, timeout: float) -> Any:
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(socket_path), timeout=timeout
    )
    try:
        writer.write((json.dumps({"op": "cancel", "request_id": request_id}) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def check_stream_shape(res: Result, events: List[Dict[str, Any]], name: str) -> bool:
    ok = True
    for i, ev in enumerate(events[:-1]):
        if ev.get("type") != "token":
            res.fail(f"{name}: non-token event before finish", str(ev))
            ok = False
    if not events or events[-1].get("type") != "finish":
        res.fail(f"{name}: no terminal finish event", str(events[-1] if events else None))
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--tokenizer", default=str(WEIGHTS_TOKENIZER))
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args()

    res = Result()

    if Path(args.tokenizer).exists():
        from tokenizers import Tokenizer

        import ds41_api.encoding as enc

        tok = Tokenizer.from_file(args.tokenizer)
        ids = tok.encode("Hello!", add_special_tokens=False).ids
        eos_id = tok.token_to_id(enc.eos_token)
    else:
        res.warn("tokenizer unavailable; using placeholder token ids")
        ids = [1, 2, 3]
        eos_id = None

    # ---- 1. submit returns immediately, request_id honored --------------
    request_id = "smoke-1"
    submit_op = {
        "op": "submit",
        "request_id": request_id,
        "token_ids": list(ids),
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 16,
            "stop_token_ids": [eos_id] if eos_id is not None else [],
            "seed": 42,
            "logprobs": 0,
            # versioning: engines must ignore unknown fields
            "future_unknown_field": {"x": 1},
        },
        "conversation_id": "smoke-conv",
    }
    try:
        events, elapsed = asyncio.run(request(args.socket, submit_op, args.request_timeout))
    except Exception as exc:  # noqa: BLE001
        res.fail("submit round-trip", repr(exc))
        return 1
    check_stream_shape(res, events, "submit")
    finish = events[-1] if events and events[-1].get("type") == "finish" else {}
    if finish.get("finish_reason") in ("stop", "length"):
        res.pass_("finish_reason", finish.get("finish_reason"))
    else:
        res.fail("finish_reason", repr(finish.get("finish_reason")))
    usage = finish.get("usage") or {}
    res.info("usage", json.dumps(usage))

    # ---- 2. optional usage extension fields ------------------------------
    for field in (
        "cached_prompt_tokens",
        "prefill_ms",
        "decode_ms",
        "draft_tokens",
        "accepted_draft_tokens",
    ):
        if field in usage:
            value = usage[field]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                res.pass_(f"usage.{field}", str(value))
            else:
                res.fail(f"usage.{field} has non-numeric value", repr(value))
        else:
            res.info(f"usage.{field} not reported (optional)")

    # ---- 3. max_tokens is a hard cap --------------------------------------
    submit_op = {
        "op": "submit",
        "request_id": "smoke-2",
        "token_ids": list(ids),
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 5,
            "stop_token_ids": [eos_id] if eos_id is not None else [],
            "seed": 42,
            "logprobs": 0,
        },
    }
    try:
        events, _ = asyncio.run(request(args.socket, submit_op, args.request_timeout))
    except Exception as exc:  # noqa: BLE001
        res.fail("max_tokens probe", repr(exc))
        events = []
    tokens = [e for e in events if e.get("type") == "token"]
    finish = events[-1] if events and events[-1].get("type") == "finish" else {}
    if len(tokens) > 5:
        res.fail("max_tokens cap", f"emitted {len(tokens)} tokens for max_tokens=5")
    else:
        res.pass_("max_tokens cap", f"{len(tokens)} tokens, finish {finish.get('finish_reason')}")
    if len(tokens) == 5 and finish.get("finish_reason") != "length":
        res.warn("max_tokens=5 emitted 5 tokens but finish_reason is not 'length'",
                 repr(finish.get("finish_reason")))

    # ---- 4. cancel is idempotent and unknown ids return false ------------
    try:
        reply = asyncio.run(cancel(args.socket, "smoke-nonexistent", args.request_timeout))
        if reply is None:
            res.warn("cancel(unknown id): connection closed without a reply")
        elif reply.get("cancelled") is False or reply.get("ok") is False or reply == {}:
            res.pass_("cancel(unknown id) returns false/unknown", json.dumps(reply))
        else:
            # engines may answer differently; only fail on a true claim
            if reply.get("cancelled") is True:
                res.fail("cancel(unknown id) claims a cancellation", json.dumps(reply))
            else:
                res.info("cancel(unknown id) reply shape", json.dumps(reply))
    except Exception as exc:  # noqa: BLE001
        res.warn("cancel(unknown id)", repr(exc))

    # ---- 5. live cancel: slow generation should finish 'cancelled' ------
    async def live_cancel_probe() -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(args.socket), timeout=args.request_timeout
        )
        try:
            op = {
                "op": "submit",
                "request_id": "smoke-3",
                "token_ids": list(ids),
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 4096,
                    "stop_token_ids": [eos_id] if eos_id is not None else [],
                    "seed": 42,
                    "logprobs": 0,
                },
            }
            writer.write((json.dumps(op) + "\n").encode())
            await writer.drain()

            # give the engine a moment to be mid-generation, then cancel
            # (backpressure keeps the request connection from filling)
            await asyncio.sleep(min(2.0, args.request_timeout / 10))
            reply = await cancel(args.socket, "smoke-3", args.request_timeout)
            res.info("cancel reply", json.dumps(reply) if reply is not None else "closed")

            # the request stream must still terminate with a finish event
            events: List[Dict[str, Any]] = []
            while True:
                line = await asyncio.wait_for(
                    reader.readline(), timeout=args.request_timeout
                )
                if not line:
                    res.fail("live cancel: connection closed without a finish event")
                    return
                events.append(json.loads(line))
                if events[-1].get("type") == "finish":
                    break
            reason = events[-1].get("finish_reason")
            if reason == "cancelled":
                res.pass_("live cancel", "stream finished with 'cancelled'")
            elif reason in ("stop", "length"):
                res.info(
                    "live cancel: generation had already completed when cancel landed",
                    f"finish_reason={reason}",
                )
            else:
                res.fail("live cancel: unexpected finish_reason", repr(reason))
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    try:
        asyncio.run(live_cancel_probe())
    except Exception as exc:  # noqa: BLE001
        res.fail("live cancel probe", repr(exc))

    print()
    if res.fails:
        print(f"SMOKE RESULT: FAIL ({res.fails} failures)")
        return 1
    print("SMOKE RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
