#!/usr/bin/env python3
"""Download deepseek-ai/DeepSeek-V4.1-Flash-1017 at the pinned revision (510 GB, 83 files) and check every file.

Sizes are checked for all files and git blob hashes for the small ones. The SHA-256 of all 48 weight shards is
checked against the Hub's LFS hashes by scripts/prepare_assets.py, which scripts/build.sh runs: hashing 510 GB
once is enough. The download resumes when run again.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from run import ROOT, configuration

INVENTORY = ROOT / 'scripts/assets/hf-lfs.json'


def git_blob_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, help='destination directory (default: "weights" in config.local.json)')
    p.add_argument('--verify-only', action='store_true', help='check an existing download without fetching')
    args = p.parse_args()
    lock = json.loads(INVENTORY.read_text())
    dest = (args.weights or Path(configuration()['weights'])).expanduser().resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        dest.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {lock["repo"]} @ {lock["revision"]} into {dest}', flush=True)
        snapshot_download(lock['repo'], revision=lock['revision'], local_dir=dest, max_workers=8)
    problems = []
    for item in lock['files']:
        path = dest / item['path']
        if not path.is_file():
            problems.append(f'missing {item["path"]}')
        elif path.stat().st_size != item['size']:
            problems.append(f'size differs: {item["path"]}')
        elif item['sha256'] is None and git_blob_sha1(path) != item['blob_id']:
            problems.append(f'content differs: {item["path"]}')
    if problems:
        raise SystemExit('Download check failed:\n  ' + '\n  '.join(problems))
    total = sum(item['size'] for item in lock['files'])
    print(f'Verified {len(lock["files"])} files ({total / 1e9:.1f} GB) at revision {lock["revision"]}.')


if __name__ == '__main__':
    main()
