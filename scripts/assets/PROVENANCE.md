# Asset generator provenance

These helpers retain the working installation's algorithms and byte formats.
The downloaded model's Python source remains unchanged and is imported in
place, with bytecode writes disabled. The official model is MIT-licensed;
its source and license come with the public model download.

| Helper/input | Origin | SHA-256 of unchanged copy |
| --- | --- | --- |
| `export-engram.py` | engine `tools/export-engram.py` (`bb64ef8` lineage) | `05519c7df86026a44563a3d56e5a23a9a6bcef8d7e884c71e7bf595c19b91a3d` |
| `export-rope.py` | engine `tools/export-rope.py` (`bb64ef8` lineage) | `ece63815f882221edea6be9e77853cc2e0c2a518628a7e1befcb8899e8537075` |
| `expert_bank.py` | MoE `tools/expert_bank.py` (`415aa68` lineage) | `d79a52d881c64988307f1531f81d3bbeb1b4a5f2284d19f07a3e810f97a83a31` |
| `hf-lfs.json` | public Hugging Face LFS/tree inventory, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` | `3bbb443b5a991673c0cc00b69f1dce94be81060e20146474c6d6334c009429d2` |

`integrity.py` ports reference `verify_weights.py`: same expected-LFS hash
check, stable-file verification and JSON receipt ordering. It adds atomic
per-shard restart state, ctime/inode/device checks for restart reuse, strict
receipt reuse checks, and preservation of an already verified receipt's
original attestation bytes after complete revalidation.

`auxiliary.py` ports engine `export-auxiliary.py` and the selected result of
`export-prefill-auxiliary.py`: identical named roles, source-provenance checks,
geometry checks, sorted two-space JSON with terminal newline. Weights/output
paths are CLI parameters. The selected frozen Engram library byte count/hash
is explicit metadata; `--engram-library` hashes a new component without
loading it. No binary is claimed to be produced from the weights.

`input-identities.json` contains byte counts/SHA-256 of the public official
small inputs at the pinned revision. It contains no local filesystem roots.
`prepare_assets.py` orchestrates the original operations, exports the selected
1M-row RoPE table directly and records timings and digests.
