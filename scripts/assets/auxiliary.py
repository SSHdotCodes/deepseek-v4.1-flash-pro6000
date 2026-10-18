"""Path-parameterized port of engine tools/export-auxiliary.py.

The selected 1M-row receipt is identical to applying
export-prefill-auxiliary.py to the short receipt. Build ds41d AFTER this step:
its compiled identity includes the complete bytes of auxiliary.json.
"""
import json
from .integrity import REVISION, atomic_write, sha256

FROZEN_ENGRAM = {"bytes": 361360, "sha256": "82d2c1aba2806d001a6213731ad4be953447456ceb743707bedb41bdcf6fef5c"}


def prepare(weights, out, library=None):
    roles = {
        "weight_index": weights / "model.safetensors.index.json",
        "config": weights / "config.json", "inference_config": weights / "inference/config.json",
        "tokenizer": weights / "tokenizer.json", "official_model": weights / "inference/model.py",
        "official_kernel": weights / "inference/kernel.py", "official_engram": weights / "inference/engram.py",
        "official_generate": weights / "inference/generate.py", "engram_export": out / "engram.json",
        "rope_export": out / "rope.bin", "rope_metadata": out / "rope.bin.json",
        "expert_manifest": out / "records.tsv",
    }
    if library is not None:
        roles["engram_library"] = library
    files = {key: dict(bytes=path.stat().st_size, sha256=sha256(path)) for key, path in roles.items()}
    if library is None:
        # This is a selected component identity, not a claim to have built or
        # hashed a binary. ds41d authenticates the actual library at startup.
        files["engram_library"] = dict(FROZEN_ENGRAM)
    engram = json.loads(roles["engram_export"].read_text())
    for name, key in (("inference/engram.py", "official_engram"),
                      ("inference/config.json", "inference_config"), ("tokenizer.json", "tokenizer")):
        if engram["provenance"][name] != files[key]["sha256"]:
            raise ValueError(f"Stale Engram provenance: {name}")
    rope = json.loads(roles["rope_metadata"].read_text())
    if (rope["source_sha256"] != files["official_model"]["sha256"]
            or rope["sha256"] != files["rope_export"]["sha256"]
            or rope["dtype"] != "F32"
            or rope["layout"] != "[2,rows,32,2] pure-SWA then compressed"
            or files["rope_export"]["bytes"] != 2 * rope["rows"] * 64 * 4):
        raise ValueError("Stale RoPE provenance/geometry")
    manifest = dict(schema="ds41-auxiliary-v1", revision=REVISION, weight_root=str(weights), files=files)
    atomic_write(out / "auxiliary.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
