# Engine assets

`ds41d` reads the downloaded checkpoint as it is: the native MXFP4 experts and FP8 dense weights straight from the 48 safetensors shards. Besides the shards it needs six small files derived from the checkpoint, which `scripts/prepare_assets.py` writes. `scripts/build.sh` runs it for you; this page explains what it does.

The checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Keep the downloaded directory unchanged, including `inference/`, `encoding/`, the tokenizer and config files and `model.safetensors.index.json`. Nothing is quantized, repacked or rewritten.

## Files

| File | What it contains | Used as |
| --- | --- | --- |
| `full-integrity.json` | SHA-256 of all 510,296,708,312 shard bytes, checked against the Hub's LFS hashes, with each shard's size and mtime | `--receipt`: before loading, the engine checks that every shard is still the verified one (size and mtime) |
| `records.tsv` | Every routed expert's tensor views and file extents (target and MTP layers), from the safetensors headers | `--experts`: the expert store reads experts from NVMe with these extents |
| `engram.json` | The Engram tokenizer map and per-layer hash tables, exported by the official `inference/engram.py` | `--engram` |
| `rope.bin` | 1,048,576 rows of the official rotary frequencies for both attention kinds, little-endian F32 `[2, rows, 32, 2]`, 512 MiB, computed on the GPU by the official `inference/model.py` | `--rope`: attention up to 1M tokens of context |
| `rope.bin.json` | Shape, dtype and the hashes of `rope.bin` and its source | read next to `rope.bin` |
| `auxiliary.json` | Sizes and SHA-256 of all of the above, of the official config, tokenizer, index and inference sources, and of the Engram library built by `scripts/build.sh` | `--auxiliary`; CMake also compiles its SHA-256 into `ds41d` |

The engine refuses to start if any of these disagree with each other, with the checkpoint, or with the Engram library it loads.

## Running it

```bash
.venv-assets/bin/python scripts/prepare_assets.py --weights models/DeepSeek-V4.1-Flash-1017 --out generated \
  --engram-library build/components/lib/libds41_engram.so
```

`.venv-assets` comes from `scripts/setup.sh` (Python 3.12, PyTorch 2.14.0 with CUDA 13.0 wheels and the other exact versions in `requirements-assets.txt`). The steps:

1. Check the small official inputs (config, tokenizer, index, `inference/*.py`) against `scripts/assets/input-identities.json` before importing any of the checkpoint's Python code, and the shard list against the pinned LFS inventory `scripts/assets/hf-lfs.json`.
2. Hash all 48 shards and compare them with the LFS SHA-256 values: about 6 minutes for 510 GB at 1.47 GB/s on a PCIe 5.0 drive. Progress is saved per shard; running the same command again resumes, rehashing only shards whose device, inode, size, mtime or ctime changed.
3. Read the expert tensor headers (under a second).
4. Export the Engram tables on the CPU (about 5 seconds).
5. Export the RoPE table on the GPU (about 3 seconds, 1.5 GiB of VRAM; no weights are loaded).
6. Check `engram.json`, `rope.bin` and `rope.bin.json` against the SHA-256 values of the measured build. They do not depend on where the files live, so any difference means a different PyTorch, CUDA or tokenizer environment, and the script stops.
7. Write `auxiliary.json`.

Peak host memory is 1.7 GiB. Caches and the restart journal stay in `generated/.prepare-assets/`, with a report of step timings and checksums in `report.json`.

`--skip-integrity` reuses an existing `full-integrity.json` for the same files instead of hashing 510 GB again; it still checks the receipt's root, revision, hashes, sizes and mtimes, and fails on any mismatch. `scripts/build.sh` uses it when it only needs to bind a newly built Engram library. Run without it after the files change.

## Location-dependent fields

Three files record where the checkpoint is, and the engine uses those paths:

- `full-integrity.json`: the root, each shard's mtime and the time of verification;
- `records.tsv`: the absolute shard path in each extent row;
- `auxiliary.json`: the root and the hash of `records.tsv`, plus the Engram library's hash.

So moving the checkpoint or the assets means running `scripts/build.sh` again, which regenerates them and rebuilds the engine against the new `auxiliary.json`. Shard hashes, expert offsets, `engram.json` and `rope.bin` are the same everywhere.

## Verification on the reference machine

Run against the checkpoint directory and Engram library of the measured service, the script reproduced all six files of that service byte for byte, including the original integrity receipt, with no fields excluded from the comparison ([`scripts/assets/reference-proof.json`](../scripts/assets/reference-proof.json)):

| File | SHA-256 |
| --- | --- |
| `full-integrity.json` | `4e003f2ab741ca0783db94d23e7bcfd3fa005f7c5cac771416ad295b349b8367` |
| `records.tsv` | `085e4dd314ccfd11eb2864c5d5850538efc3356ba5dbfd22f1d7598beb7a3b51` |
| `engram.json` | `edebf1b3ed13ce1ca4124e6c844624db66f045c46c8d04db37e56e1cb276f73d` |
| `rope.bin` | `5476db91830804ed4289b9b23ad0e475d863d65f73c246aed88157b22ef0d575` |
| `rope.bin.json` | `bd0d6ca6bbee782f628c5bfec06c37cad3bd2262bba154ef1b12324d59b5d53d` |
| `auxiliary.json` | `2f57e890915fc2050039f0cc1061e57ee7ce8a50260dd9692d3503731112ec0b` |

| Step | Full run (s) | Reusing the receipt (s) |
| --- | ---: | ---: |
| integrity | 346.4 | 0.0 |
| expert records | 0.7 | 0.7 |
| Engram | 5.7 | 3.8 |
| RoPE (CUDA) | 2.8 | 2.5 |
| auxiliary | 0.2 | 0.2 |
| total, with the final checksums | 356.2 | 7.6 |

The Engram library this repo builds differs from the measured service's in one string: the source path an `assert` names, which `scripts/build.sh` maps to a path relative to the repo so the library has the same bytes wherever the repo is cloned. Its `auxiliary.json` therefore differs from the one above in the Engram entry (and in the paths, on another machine). The engine built from this repo returned the same tokens as the measured service on every benchmark request ([benchmarks](benchmarks.md#source-build)).

## Provenance

[`scripts/assets/PROVENANCE.md`](../scripts/assets/PROVENANCE.md) lists where each helper comes from and the hashes of the unchanged copies. `hf-lfs.json` is the Hub's public per-file size and hash list for the pinned revision.
