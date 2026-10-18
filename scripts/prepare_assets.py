#!/usr/bin/env python3
"""Regenerate ds41d's selected assets from the official pinned checkpoint.

See docs/assets.md for dependency installation, receipt reuse, resource needs,
and the required ordering relative to CMake configuration.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from assets import auxiliary, expert_bank, integrity

HELPERS = Path(__file__).resolve().parent / "assets"
OUTPUTS = ("full-integrity.json", "records.tsv", "engram.json", "rope.bin", "rope.bin.json", "auxiliary.json")
GOLDEN = {
    "engram.json": "edebf1b3ed13ce1ca4124e6c844624db66f045c46c8d04db37e56e1cb276f73d",
    "rope.bin": "5476db91830804ed4289b9b23ad0e475d863d65f73c246aed88157b22ef0d575",
    "rope.bin.json": "bd0d6ca6bbee782f628c5bfec06c37cad3bd2262bba154ef1b12324d59b5d53d",
}
PINS = {"torch": "2.14.0", "numpy": "2.5.3", "transformers": "5.18.0", "tokenizers": "0.23.2",
        "sympy": "1.14.0", "tilelang": "0.1.12", "apache-tvm-ffi": "0.1.11", "Pillow": "12.3.0"}


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Official revision directory, containing inference/ and tokenizer files")
    parser.add_argument("--out", type=Path, required=True, help="Asset directory; must be outside the weights")
    parser.add_argument("--skip-integrity", action="store_true", help="Reuse a complete receipt after checking root, revision, hashes, size and mtimes; do not rehash payloads")
    parser.add_argument("--integrity-receipt", type=Path, help="Existing full receipt to reuse, or fully revalidate while preserving original attestation time/bytes")
    parser.add_argument("--engram-library", type=Path, help="Authenticate this built library instead of recording the selected frozen library identity")
    args = parser.parse_args()
    weights, out = args.weights.resolve(strict=True), args.out.resolve()
    if out == weights or weights in out.parents:
        parser.error("--out must be outside the read-only weights directory")
    if args.integrity_receipt:
        args.integrity_receipt = args.integrity_receipt.resolve(strict=True)
    if args.engram_library:
        args.engram_library = args.engram_library.resolve(strict=True)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".prepare-assets"
    work.mkdir(exist_ok=True)
    # Serialize writers to a given output. A cancelled invocation can be resumed.
    import fcntl
    with (work / "lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another prepare_assets process is writing this output")
        execute(args, weights, out, work)


def execute(args, weights, out, work):
    # Qualified on CPython 3.12.3; any 3.12 works because the outputs are checked against the selected digests.
    if sys.version_info[:2] != (3, 12):
        raise ValueError(f"Use Python 3.12 for the pinned environment, got {sys.version.split()[0]}")
    versions = {name: importlib.metadata.version(name) for name in PINS}
    for name, version in versions.items():
        if version.split("+")[0] != PINS[name]:
            raise ValueError(f"{name}=={PINS[name]} required; found {version}. Install requirements-assets.txt")
    # Verify the official small inputs before executing checkpoint Python code.
    expected = json.loads((HELPERS / "input-identities.json").read_text())
    for relative, identity in expected.items():
        path = weights / relative
        if path.stat().st_size != identity["bytes"] or integrity.sha256(path) != identity["sha256"]:
            raise ValueError(f"Wrong official input/revision: {relative}")
    files = integrity.inventory(HELPERS / "hf-lfs.json")
    actual_shards = sorted(set(json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"].values()))
    if actual_shards != sorted(f["path"] for f in files):
        raise ValueError("Weight index shard inventory differs from the official LFS inventory")
    if args.skip_integrity:
        source = args.integrity_receipt or out / "full-integrity.json"
        if not source.is_file():
            raise ValueError("--skip-integrity requires --integrity-receipt FILE or an existing output receipt")
        integrity.validate_receipt(source, weights, files)
    log(f"Python {sys.version.split()[0]}; pinned packages {json.dumps(versions, sort_keys=True)}")
    log(f"Weights: {weights}; output: {out}")
    started = time.perf_counter()
    timings = {}

    def step(name, action):
        log(f"START {name}")
        begin = time.perf_counter()
        action()
        timings[name] = round(time.perf_counter() - begin, 3)
        log(f"DONE {name}: {timings[name]:.3f} s")

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for name, subdir in (("XDG_CACHE_HOME", "cache"), ("HF_HOME", "cache/hf"),
                         ("TILELANG_CACHE_DIR", "cache/tilelang"), ("TVM_FFI_CACHE_DIR", "cache/tvm-ffi"),
                         ("TORCHINDUCTOR_CACHE_DIR", "cache/torchinductor")):
        env[name] = str(work / subdir)

    def export(tool, *arguments):
        subprocess.run([sys.executable, "-B", str(HELPERS / tool), "--weights", str(weights), *map(str, arguments)], env=env, check=True)

    step("full-integrity", lambda: integrity.prepare(weights, out, work, HELPERS / "hf-lfs.json", log,
                                                      args.skip_integrity, args.integrity_receipt))

    def records():
        manifest = expert_bank.inspect(weights, weights / "model.safetensors.index.json", work / "headers")
        expert_bank.export_tsv(manifest, weights, out / "records.tsv.partial")
        (out / "records.tsv.partial").replace(out / "records.tsv")
        log(f"Expert records: {len(manifest['records']):,} (target and MTP); headers only")
    step("expert-records", records)
    step("engram", lambda: export("export-engram.py", "--output", out / "engram.json"))
    # The installed service uses 1M rows, not the bringup tool's 8192 default.
    step("rope-cuda", lambda: export("export-rope.py", "--rows", 1048576, "--output", out / "rope.bin"))
    for name, digest in GOLDEN.items():
        if integrity.sha256(out / name) != digest:
            raise ValueError(f"{name} is not byte-identical to the selected export; check CUDA/PyTorch and tokenizer environment")
    if args.engram_library is None:
        log("Auxiliary uses the recorded frozen Engram library identity (not built by this command); see --engram-library")
    step("auxiliary", lambda: auxiliary.prepare(weights, out, args.engram_library))
    assets = {name: {"bytes": (out / name).stat().st_size, "sha256": integrity.sha256(out / name)} for name in OUTPUTS}
    report = dict(schema="ds41-assets-run-v1", completed_at=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, dependencies=versions, weights=str(weights),
                  timings_seconds=timings, total_seconds=round(time.perf_counter() - started, 3),
                  integrity_payload_hashing=not args.skip_integrity, assets=assets)
    integrity.atomic_write(work / "report.json", json.dumps(report, indent=2) + "\n")
    log(f"COMPLETE: {report['total_seconds']:.3f} s; build ds41d against {out / 'auxiliary.json'}")
    for name, entry in assets.items():
        print(f"{entry['sha256']}  {name}  ({entry['bytes']:,} bytes)", flush=True)
    log(f"Step timings and checksums: {work / 'report.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as error:
        print(f"prepare_assets: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
