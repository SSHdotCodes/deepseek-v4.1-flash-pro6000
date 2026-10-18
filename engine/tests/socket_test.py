import json
import hashlib
import os
from pathlib import Path
import select
import socket
import subprocess
import struct
import sys
import tempfile
import time

binary = sys.argv[1]
with tempfile.TemporaryDirectory(prefix="ds41-socket-") as tmp:
    root = Path(tmp)
    path = root / "backend.sock"
    fixture = root / "test.json"
    fixture_rows = [[0, 3, 1, -20]+[-1000]*508,
                    [0, 4, 2, -20]+[-1000]*508,
                    [1, 6, 0, -20]+[-1000]*508]
    fixture.write_text(json.dumps({"schema": "ds41-replay-test-v1", "eos": 3,
                                   "logits": fixture_rows, "delay_ms": 5}))
    process = subprocess.Popen([binary, "serve", "--socket", str(path), "--replay-test", str(fixture), "--prefix-cache"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def connect():
        sock = socket.socket(socket.AF_UNIX)
        sock.settimeout(5)
        sock.connect(str(path))
        return sock

    def request(obj):
        sock = connect()
        sock.sendall((json.dumps(obj) + "\n").encode())
        stream = sock.makefile("rb")
        events = []
        for line in stream:
            events.append(json.loads(line))
        stream.close()
        sock.close()
        assert sum(x.get("type") == "finish" for x in events) == 1
        assert events[-1]["type"]=="finish",events
        return events

    try:
        deadline = time.monotonic() + 5
        while not path.exists():
            assert time.monotonic() < deadline, process.poll()
            time.sleep(.01)
        assert path.stat().st_mode & 0o777 == 0o600
        submit = {"op": "submit", "request_id": "test", "token_ids": [0, 1],
                  "sampling": {"temperature": 0, "max_tokens": 4, "logprobs": 2, "seed": 42}}
        events = request(submit)
        assert [x["token_id"] for x in events[:-1]] == [1] * 4, events
        usage = events[-1]["usage"]
        assert usage["prompt_tokens"] == 2 and usage["completion_tokens"] == 4
        assert usage["cached_prompt_tokens"] == 0
        assert usage["draft_tokens"] == usage["accepted_draft_tokens"] == 0
        assert usage["prefill_ms"] == events[-1]["stats"]["prefill_ms"]
        assert usage["decode_ms"] == events[-1]["stats"]["generation_ms"]
        assert usage["decode_ms"] >= events[-1]["stats"]["decode_ms"] >= 0
        assert events[-1]["finish_reason"] == "length"
        assert all(x["logprob"] < 0 and x["top_logprobs"][0]["token_id"] == 1 for x in events[:-1])
        expected = hashlib.sha256(struct.pack("<512f", *fixture_rows[1])).hexdigest()
        assert events[0]["logits_sha256"] == expected
        assert isinstance(events[0]["logsumexp"], float)
        assert events[-1]["stats"]["engine_mode"] == "REPLAY_TEST_ONLY"
        for bad in ([], [-1], [999], [True], [0.5]):
            result = request(dict(submit, token_ids=bad))
            assert result[-1]["finish_reason"] == "error", result
        for bad in ({"temperature": -1}, {"top_p": 0}, {"logprobs": -1}, {"seed": -1}, {"max_tokens": True}):
            assert request(dict(submit, sampling=bad))[-1]["finish_reason"] == "error"
        assert request(dict(submit, sampling={"temperature": 0, "stop_token_ids": [1]}))[-1]["usage"]["completion_tokens"] == 0
        assert len(request(dict(submit, sampling={"max_tokens": 0}))) == 1
        a = request(dict(submit, sampling={"max_tokens": 20, "seed": 123}))
        b = request(dict(submit, sampling={"max_tokens": 20, "seed": 123}))
        assert [x.get("token_id") for x in a] == [x.get("token_id") for x in b]
        score = request({"op": "score", "token_ids": [0, 1, 2], "top": 2})
        assert len(score) == 4 and score[0]["next_token_id"] == 1
        score_op = {"op":"score", "token_ids":list(range(31)), "top":2,
                    "score_from":3, "return_logits":True}
        baseline = request(score_op)
        for widths in (1,2,3,4,5,6,[1,2,3,4,5,6],[6,1,3]):
            wide = request(dict(score_op, verify_width=widths))
            assert wide[:-1] == baseline[:-1], widths
            assert wide[-1]["stats"]["committed"] == 31
        for event in baseline[:-1]:
            assert event["logits_sha256"] == hashlib.sha256(struct.pack("<512f",*event["logits"])).hexdigest()
        for widths in (0,7,True,1.5,[],[1,0],[True],[1]*65):
            assert request(dict(score_op,verify_width=widths))[-1]["finish_reason"] == "error", widths
        assert request(dict(submit,draft="off"))[-1]["finish_reason"] == "length"
        assert request(dict(submit,draft="on"))[-1]["error"].startswith("unsupported:")
        # One live growing conversation can reuse only consumed KV. Three
        # outputs consume two inputs; the last sampled output is not cached.
        turn = dict(submit,conversation_id="growing",sampling={"temperature":0,"max_tokens":3,"logprobs":2})
        first_turn = request(turn)
        assert first_turn[-1]["stats"]["committed"] == 4
        next_prompt = [0,1,1,1,1,2]
        followup = dict(turn,token_ids=next_prompt,sampling={"temperature":0,"max_tokens":1,"logprobs":2})
        reused = request(followup)
        assert reused[-1]["stats"]["reused_prompt_tokens"] == 4
        assert reused[-1]["usage"]["cached_prompt_tokens"] == 4
        assert reused[-1]["stats"]["prefill_input_tokens"] == 2
        assert reused[-1]["stats"]["verify_calls"] == 2
        repeated = request(followup)
        assert repeated[:-1] == reused[:-1]
        assert repeated[-1]["stats"]["reused_prompt_tokens"] == 6
        assert repeated[-1]["stats"]["verify_calls"] == 0
        cold = request(dict(followup,conversation_id="other"))
        assert cold[:-1] == reused[:-1]
        assert cold[-1]["stats"]["reused_prompt_tokens"] == 0
        assert cold[-1]["stats"]["verify_calls"] == 6
        # Edited and shortened prompts rebuild, even under the same id.
        edited=request(dict(followup,conversation_id="other",token_ids=[2]+next_prompt[1:]))
        assert edited[-1]["stats"]["reused_prompt_tokens"] == 0
        short=request(dict(followup,conversation_id="other",token_ids=[2,1]))
        assert short[-1]["stats"]["reused_prompt_tokens"] == 0
        request({"op":"score","token_ids":[1,2,3]})
        after_score=request(dict(followup,conversation_id="other",token_ids=[2,1]))
        assert after_score[-1]["stats"]["reused_prompt_tokens"] == 0
        tail=request({"op":"score","token_ids":[0,511],"top":512})
        assert isinstance(tail[0]["next_logprob"],(float,int)) and tail[0]["next_logprob"] < -1000
        assert all(isinstance(v,(float,int)) for v in tail[0]["top_logprob"])
        tail=request(dict(submit,sampling={"temperature":0,"max_tokens":1,"logprobs":512}))
        assert all(isinstance(v["logprob"],(float,int)) for v in tail[0]["top_logprobs"])
        stream_sock = connect()
        stream_sock.sendall((json.dumps(dict(submit, request_id="cancel-me", sampling={"max_tokens": 1000})) + "\n").encode())
        stream = stream_sock.makefile("rb")
        assert json.loads(stream.readline())["type"] == "token"
        control = connect()
        control_file = control.makefile("rb")
        for rid, expected in (("cancel-me", True), ("unknown", False)):
            control.sendall((json.dumps({"op": "cancel", "request_id": rid}) + "\n").encode())
            assert json.loads(control_file.readline()) == {"ok": expected}
        while True:
            event = json.loads(stream.readline())
            if event["type"] == "finish":
                assert event["finish_reason"] == "cancelled", event
                break
        control_file.close()
        control.close()
        stream.close()
        stream_sock.close()
        # An unread peer must not hold the serial model after cancellation.
        blocked=connect()
        blocked.sendall((json.dumps(dict(submit,request_id="blocked",sampling={"temperature":0,"max_tokens":2000,"logprobs":512}))+"\n").encode())
        blocked_file=blocked.makefile("rb")
        assert json.loads(blocked_file.readline())["type"]=="token"
        time.sleep(.2)
        control=connect(); control_file=control.makefile("rb")
        control.sendall(b'{"op":"cancel","request_id":"blocked"}\n')
        assert json.loads(control_file.readline())=={"ok":True}
        started=time.monotonic()
        assert request(submit)[-1]["finish_reason"]=="length"
        assert time.monotonic()-started<3.5
        control_file.close();control.close();blocked_file.close();blocked.close()
        # Existing socket ownership must be preserved.
        other = subprocess.run([binary, "serve", "--socket", str(path), "--replay-test", str(fixture)], capture_output=True)
        assert other.returncode == 1 and path.exists()
        assert request(submit)[-1]["finish_reason"] == "length"
        # A writable active stream gets a final cancelled event on shutdown.
        reader=connect(); reader.sendall((json.dumps(dict(submit,request_id="shutdown",sampling={"max_tokens":1000}))+"\n").encode())
        read_file=reader.makefile("rb")
        assert json.loads(read_file.readline())["type"]=="token"
        process.terminate()
        terminal=[json.loads(line) for line in read_file]
        assert terminal[-1]["type"]=="finish" and terminal[-1]["finish_reason"]=="cancelled",terminal
        assert sum(x["type"]=="finish" for x in terminal)==1
        read_file.close();reader.close()
        print("socket: EOF/cardinality, stable tails, blocked cancellation, graceful shutdown, protocol passed")
    finally:
        process.terminate()
        process.communicate(timeout=10)
        assert not path.exists()
