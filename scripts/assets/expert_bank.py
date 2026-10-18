#!/usr/bin/env python3
"""Inspect native safetensors and make bounded, byte-exact expert banks.

No tensor framework, dtype conversion, or mmap of the full checkpoint is used.
Headers can be fetched independently from Hugging Face with HTTP Range requests.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import urllib.request

ALIGN = 4096
FIELDS = [f"{w}.{t}" for w in ("w1", "w3", "w2") for t in ("weight", "scale")]
EXPERT = re.compile(r"^((?:layers|mtp)\.\d+\.ffn\.experts\.\d+)\.(w[123]\.(?:weight|scale))$")


def aligned(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def range_get(url, start, length):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start+length-1}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f"server ignored range for {url}: {response.status}")
        result = response.read(length + 1)
    if len(result) != length:
        raise RuntimeError(f"short or excessive range: expected {length}, got {len(result)}")
    return result


def header(root, cache, shard, base_url=None):
    local = root / shard
    saved = cache / (shard + ".header.json")
    if saved.exists():
        raw = saved.read_bytes()
    elif local.exists():
        with local.open("rb") as f:
            first = f.read(8)
            if len(first) != 8:
                raise RuntimeError(f"incomplete shard {local}")
            n, = struct.unpack("<Q", first)
            if n > 64 * 1024 * 1024:
                raise RuntimeError("invalid safetensors header size")
            raw = f.read(n)
            if len(raw) != n:
                raise RuntimeError(f"incomplete header {local}")
    elif base_url:
        url = base_url.rstrip("/") + "/" + shard
        n, = struct.unpack("<Q", range_get(url, 0, 8))
        if n > 64 * 1024 * 1024:
            raise RuntimeError("invalid remote header size")
        raw = range_get(url, 8, n)
    else:
        raise FileNotFoundError(local)
    cache.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(raw)
    h = json.loads(raw)
    end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
    return h, len(raw) + 8, len(raw) + 8 + end


def inspect(root, index, cache, base_url=None):
    weight_map = json.loads(index.read_text())["weight_map"]
    by_shard = collections.defaultdict(list)
    groups = {}
    for key, shard in weight_map.items():
        match = EXPERT.match(key)
        if match:
            by_shard[shard].append((key, *match.groups()))
    for shard, keys in sorted(by_shard.items()):
        h, data_start, expected_size = header(root, cache, shard, base_url)
        for key, expert, field in keys:
            tensor = h[key]
            lo, hi = tensor["data_offsets"]
            groups.setdefault(expert, {})[field] = dict(
                shard=shard, offset=data_start + lo, size=hi-lo,
                dtype=tensor["dtype"], shape=tensor["shape"], expected_shard_size=expected_size,
            )
    records = []
    for expert, tensors in sorted(groups.items()):
        if set(tensors) != set(FIELDS):
            raise RuntimeError(f"incomplete expert {expert}: {list(tensors)}")
        # Coalesce only adjacent tensor ranges. Scale and weight planes remain distinct.
        spans = []
        for t in sorted(tensors.values(), key=lambda t: (t["shard"], t["offset"])):
            if spans and spans[-1]["shard"] == t["shard"] and spans[-1]["offset"] + spans[-1]["size"] == t["offset"]:
                spans[-1]["size"] += t["size"]
            else:
                spans.append(dict(shard=t["shard"], offset=t["offset"], size=t["size"]))
        for span in spans:
            span["read_offset"] = span["offset"] // ALIGN * ALIGN
            span["prefix"] = span["offset"] - span["read_offset"]
            span["read_size"] = aligned(span["prefix"] + span["size"])
        records.append(dict(expert=expert, tensors=tensors, native_spans=spans))
    return dict(version=1, alignment=ALIGN, records=records)


def read_tensor(root, tensor, base_url=None):
    src = root / tensor["shard"]
    if src.exists() and src.stat().st_size == tensor["expected_shard_size"]:
        with src.open("rb") as f:
            f.seek(tensor["offset"])
            data = f.read(tensor["size"])
    elif base_url:
        data = range_get(base_url.rstrip("/")+"/"+tensor["shard"],tensor["offset"],tensor["size"])
    else:
        raise RuntimeError(f"shard incomplete: {src}")
    if len(data) != tensor["size"]:
        raise RuntimeError(f"short tensor {src}")
    return data


def repack(manifest, root, output, selected, max_bytes, base_url=None):
    records = [r for r in manifest["records"] if r["expert"] in selected]
    if {r["expert"] for r in records} != selected:
        raise ValueError("unknown selected experts")
    total = sum(ALIGN + sum(aligned(t["size"]) for t in r["tensors"].values()) for r in records)
    if total > max_bytes:
        raise ValueError(f"requested {total} bytes exceeds --max-bytes {max_bytes}")
    if output.exists():
        raise FileExistsError(output)
    entries = []
    with output.open("xb") as bank:
        for record in records:
            start = bank.tell()
            bank.write(bytes(ALIGN))
            fields = {}
            for field in FIELDS:
                tensor = record["tensors"][field]
                offset = bank.tell() - start
                data = read_tensor(root,tensor,base_url)
                bank.write(data)
                bank.write(bytes(aligned(len(data)) - len(data)))
                fields[field] = dict(offset=offset, size=len(data), sha256=hashlib.sha256(data).hexdigest(), dtype=tensor["dtype"], shape=tensor["shape"])
            entry = dict(expert=record["expert"], offset=start, size=bank.tell()-start, fields=fields)
            raw = json.dumps(dict(magic="DS41EX01", **entry), separators=(",", ":")).encode()
            if len(raw) + 8 > ALIGN:
                raise RuntimeError("record header overflow")
            end = bank.tell()
            bank.seek(start); bank.write(struct.pack("<Q", len(raw))); bank.write(raw); bank.seek(end)
            entries.append(entry)
        bank.flush(); os.fsync(bank.fileno())
    result = dict(magic="DS41BANK01", alignment=ALIGN, records=entries)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(manifest, root, bank_path, base_url=None):
    original = {r["expert"]: r for r in manifest["records"]}
    packed = json.loads(bank_path.with_suffix(bank_path.suffix + ".json").read_text())
    checked = 0
    with bank_path.open("rb") as bank:
        for record in packed["records"]:
            bank.seek(record["offset"])
            n, = struct.unpack("<Q", bank.read(8))
            if n > ALIGN - 8:
                raise ValueError("invalid bank header")
            embedded = json.loads(bank.read(n))
            if {k: v for k, v in embedded.items() if k != "magic"} != record or embedded["magic"] != "DS41EX01":
                raise ValueError("bank manifest/header mismatch")
            for field, t in record["fields"].items():
                src = original[record["expert"]]["tensors"][field]
                expected = read_tensor(root,src,base_url)
                bank.seek(record["offset"]+t["offset"]); actual = bank.read(t["size"])
                if expected != actual or hashlib.sha256(actual).hexdigest() != t["sha256"]:
                    raise ValueError(f"bit mismatch: {record['expert']} {field}")
                checked += len(actual)
    return dict(experts=len(packed["records"]), bit_exact_bytes=checked)


def export_tsv(manifest, root, out, selected=None, bank_path=None):
    """Portable C++ manifest, keeping native aligned read buffers as-is."""
    records = manifest["records"]
    if bank_path:
        records = json.loads(bank_path.with_suffix(bank_path.suffix + ".json").read_text())["records"]
    lines = ["# DS41MOE01: R id bytes (tensor offset bytes)x6; E id path file_offset bytes dst required expected_file_bytes"]
    for r in records:
        if selected and r["expert"] not in selected:
            continue
        kind, layer, _, _, expert = r["expert"].split(".")
        layer, expert = int(layer), int(expert)
        id_ = layer * 384 + expert if kind == "layers" else 15360 + layer * 128 + expert
        views = {}
        extents = []
        if bank_path:
            size = r["size"]
            extents.append((bank_path.resolve(), r["offset"], size, 0, size, bank_path.stat().st_size))
            views = {k: (v["offset"], v["size"]) for k, v in r["fields"].items()}
        else:
            size = 0
            for span in r["native_spans"]:
                expected = next(t["expected_shard_size"] for t in r["tensors"].values() if t["shard"] == span["shard"])
                extents.append(((root/span["shard"]).resolve(), span["read_offset"], span["read_size"], size, span["prefix"]+span["size"], expected))
                for key, t in r["tensors"].items():
                    if t["shard"] == span["shard"] and span["offset"] <= t["offset"] and t["offset"]+t["size"] <= span["offset"]+span["size"]:
                        views[key] = (size + t["offset"] - span["read_offset"], t["size"])
                size += span["read_size"]
        lines.append("R " + " ".join(map(str,[id_,size]+[n for k in FIELDS for n in views[k]])))
        for path, offset, n, dst, required, expected in extents:
            lines.append(f"E {id_} {json.dumps(str(path))} {offset} {n} {dst} {required} {expected}")
    out.write_text("\n".join(lines)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["inspect", "repack", "verify", "export-native", "export-bank"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--index", type=Path)
    p.add_argument("--headers", type=Path, default=Path("evidence"))
    p.add_argument("--base-url")
    p.add_argument("--manifest", type=Path, default=Path("evidence/layout.json"))
    p.add_argument("--bank", type=Path, default=Path("sample.bank"))
    p.add_argument("--expert", action="append", default=[])
    p.add_argument("--max-bytes", type=int, default=512*1024**2)
    p.add_argument("--output", type=Path, default=Path("evidence/records.tsv"))
    args = p.parse_args()
    if args.command == "inspect":
        m = inspect(args.root, args.index, args.headers, args.base_url)
        args.manifest.write_text(json.dumps(m, separators=(",", ":")))
        print(json.dumps(dict(experts=len(m["records"]), spans=dict(collections.Counter(len(r["native_spans"]) for r in m["records"])), drafter=sum(r["expert"].startswith("mtp.") for r in m["records"]))))
    else:
        m = json.loads(args.manifest.read_text())
        if args.command.startswith("export-"):
            export_tsv(m,args.root,args.output,set(args.expert) or None,args.bank if args.command == "export-bank" else None)
            print(str(args.output))
        elif args.command == "repack":
            if not args.expert:
                p.error("explicit --expert selections required; no accidental full bank")
            print(json.dumps(repack(m, args.root, args.bank, set(args.expert), args.max_bytes,args.base_url)))
        else:
            print(json.dumps(verify(m, args.root, args.bank,args.base_url)))


if __name__ == "__main__":
    main()
