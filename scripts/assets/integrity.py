"""Resumable full-file verification, preserving the original receipt wire format.

Adapted from ref/verify_weights.py. Expected hashes come from the unchanged
public HF LFS inventory, never from the local checkpoint being verified.
"""
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
METADATA_SHA256 = "3bbb443b5a991673c0cc00b69f1dce94be81060e20146474c6d6334c009429d2"


def sha256(path, progress=None):
    h = hashlib.sha256()
    seen = 0
    last_report = time.monotonic()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
            seen += len(block)
            if progress is not None and time.monotonic() - last_report >= 10:
                progress(seen)
                last_report = time.monotonic()
    return h.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def inventory(metadata):
    if sha256(metadata) != METADATA_SHA256:
        raise ValueError("Bundled public HF inventory has changed")
    meta = json.loads(metadata.read_text())
    if (meta["repo"], meta["revision"]) != (REPO, REVISION):
        raise ValueError("Unexpected HF inventory identity")
    files = [f for f in meta["files"] if f["path"].endswith(".safetensors")]
    if len(files) != 48 or len({f["path"] for f in files}) != 48:
        raise ValueError("Expected exactly 48 native safetensors shards")
    return files


def identity(stat):
    # ctime/device/inode strengthen the restart cache; only mtime/size are
    # part of ds41d's public receipt schema.
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def validate_receipt(path, weights, files):
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "ds41-native-integrity-v1"
            or receipt.get("root") != str(weights)
            or receipt.get("repo") != REPO or receipt.get("revision") != REVISION
            or receipt.get("metadata_sha256") != METADATA_SHA256
            or receipt.get("scope") != "all-safetensors"
            or receipt.get("complete") is not True):
        raise ValueError("Receipt must be complete and match this weight root/revision/inventory")
    if len(receipt.get("files", [])) != len(files):
        raise ValueError("Receipt shard count differs")
    for got, expected in zip(receipt["files"], files):
        stat = (weights / expected["path"]).stat()
        for key, value in {
            "path": expected["path"], "expected_sha256": expected["sha256"],
            "expected_size": expected["size"], "ok": True,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": expected["sha256"],
        }.items():
            if got.get(key) != value:
                raise ValueError(f"Stale/invalid receipt: {expected['path']} field {key}")
        if stat.st_size != expected["size"]:
            raise ValueError(f"Wrong shard size: {expected['path']}")
    return receipt


def prepare(weights, out, work, metadata, log, skip=False, existing=None):
    files = inventory(metadata)
    target = out / "full-integrity.json"
    if skip:
        source = existing or target
        if not source.is_file():
            raise ValueError("--skip-integrity requires --integrity-receipt FILE or an existing output receipt")
        validate_receipt(source, weights, files)
        # Preserve the supplied receipt byte for byte, including its original time.
        if source.resolve() != target.resolve():
            atomic_write(target, source.read_text())
        log("Reused complete matching receipt; payload hashing was explicitly skipped")
        return

    checkpoint = work / "integrity-progress.json"
    state = {"root": str(weights), "metadata_sha256": METADATA_SHA256, "files": {}}
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text())
        if (cached.get("root"), cached.get("metadata_sha256")) == (str(weights), METADATA_SHA256):
            state = cached
    total = sum(f["size"] for f in files)
    done = 0
    results = []
    log(f"Integrity: {len(files)} shards, {total:,} bytes; completed shards resume from {checkpoint}")
    for number, expected in enumerate(files, 1):
        path = weights / expected["path"]
        before = path.stat()
        if before.st_size != expected["size"]:
            raise ValueError(f"Wrong shard size: {path}")
        saved = state["files"].get(expected["path"], {})
        resumed = (saved.get("identity") == identity(before)
                   and saved.get("sha256") == expected["sha256"])
        log(f"[{number:02d}/{len(files)}] {'resume' if resumed else 'hash'} {expected['path']} ({before.st_size:,} bytes)")
        def progress(seen):
            log(f"  hashing {expected['path']}: {seen:,}/{before.st_size:,} bytes; total {(done + seen) / total:.1%}")
        digest = saved["sha256"] if resumed else sha256(path, progress=progress)
        after = path.stat()
        if identity(before) != identity(after) or digest != expected["sha256"]:
            raise ValueError(f"Changed/corrupt shard: {path}")
        state["files"][expected["path"]] = {"identity": identity(after), "sha256": digest}
        atomic_write(checkpoint, json.dumps(state, indent=2) + "\n")
        # Preserve insertion order and indent from the working reference exporter.
        results.append(dict(path=expected["path"], expected_sha256=expected["sha256"],
                            expected_size=expected["size"], ok=True, size=after.st_size,
                            mtime_ns=after.st_mtime_ns, sha256=digest))
        done += after.st_size
        log(f"  verified {done / total:.1%} ({done:,}/{total:,} bytes)")
    receipt = dict(schema="ds41-native-integrity-v1", root=str(weights),
                   repo=REPO, revision=REVISION, metadata_sha256=METADATA_SHA256,
                   files=results, scope="all-safetensors", complete=True,
                   time=datetime.now(timezone.utc).isoformat())
    # A full revalidation can preserve an old receipt's original attestation
    # timestamp and exact formatting only when EVERY other field is equal.
    prior = existing or (target if target.is_file() else None)
    if prior is not None:
        old = json.loads(prior.read_text())
        if {k: v for k, v in old.items() if k != "time"} == {k: v for k, v in receipt.items() if k != "time"}:
            validate_receipt(prior, weights, files)
            atomic_write(target, prior.read_text())
            log("Full revalidation passed; preserved the supplied receipt's original attestation time/bytes")
            return
    atomic_write(target, json.dumps(receipt, indent=2))
