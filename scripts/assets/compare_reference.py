#!/usr/bin/env python3
"""Compare all six generated assets against an installation, without exclusions."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from integrity import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-generated", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--reference-experts", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sources = {"full-integrity.json": args.reference_receipt, "records.tsv": args.reference_experts}
    for name in ("engram.json", "rope.bin", "rope.bin.json", "auxiliary.json"):
        sources[name] = args.reference_generated / name
    rows = []
    print("| Asset | Reference SHA-256 | Regenerated SHA-256 | Match |", flush=True)
    print("| --- | --- | --- | --- |", flush=True)
    for name, reference in sources.items():
        actual = args.out / name
        ref_hash, actual_hash = sha256(reference), sha256(actual)
        same = ref_hash == actual_hash and reference.stat().st_size == actual.stat().st_size
        rows.append(dict(file=name, reference_sha256=ref_hash, regenerated_sha256=actual_hash,
                         bytes=actual.stat().st_size, match=same))
        print(f"| `{name}` | `{ref_hash}` | `{actual_hash}` | {'PASS' if same else 'FAIL'} |", flush=True)
    report = dict(schema="ds41-assets-comparison-v1", excluded_fields=[],
                  all_match=all(row["match"] for row in rows), files=rows)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["all_match"] else 1)


if __name__ == "__main__":
    main()
