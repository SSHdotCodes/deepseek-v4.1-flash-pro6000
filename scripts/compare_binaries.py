#!/usr/bin/env python3
"""Compare the runtime files of two builds, for example yours and another machine's build/dist.

sass: kernel names and instruction streams from cuobjdump -sass (fatbin headers and PTX are ignored).
host_text: objdump -d of .text. Both are normalized for what depends on where the sources were built:
addresses, RIP-relative displacements and the path hash nvcc puts into anonymous-namespace names.
dynamic_symbols: nm -D --defined-only.

  python3 scripts/compare_binaries.py build/dist /other/build/dist --out results/binary-comparison.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

FILES = ['ds41d', 'libengine_ops.so', 'libds41_spark.so', 'libds41_spark_policy_v2.so', 'libds41_moe.so',
         'libds41_dense.so', 'libds41_engram.so', 'libds41_attn.so', 'libds41_fuse_head.so',
         'libds41_fuse_head_half.so', 'libds41_fuse_head_half_tune.so']
ANON = re.compile(r'_GLOBAL__N__[0-9a-f]+_')
CUOBJDUMP = str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.4')) / 'bin/cuobjdump')


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def digest(lines):
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def sass(path):
    out = run(CUOBJDUMP, '-sass', str(path))
    if not out or 'Function :' not in out:
        return None
    return digest(re.sub(r'/\*[0-9a-f]{4,}\*/', '', ANON.sub('_GLOBAL__N__X_', l.strip()))
                  for l in out.splitlines() if 'Function :' in l or re.match(r'\s+/\*[0-9a-f]{4,}\*/', l))


def host_text(path):
    out = run('objdump', '-d', '--no-show-raw-insn', '--no-addresses', '-j', '.text', str(path))
    if not out:
        return None
    lines = []
    for l in out.splitlines()[3:]:
        s = ANON.sub('_GLOBAL__N__X_', l.strip())
        s = re.sub(r'#.*$', '', s)                      # objdump's resolved-address comments
        s = re.sub(r'-?0x[0-9a-f]+\(%rip\)', 'X(%rip)', s)
        s = re.sub(r'\b[0-9a-f]{4,}\s+<', '<', s)        # absolute branch and call targets
        lines.append(s.rstrip())
    return digest(lines)


def symbols(path):
    out = run('nm', '-D', '--defined-only', str(path))
    return None if not out else digest(sorted(ANON.sub('X', ' '.join(l.split()[1:])) for l in out.splitlines()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('build', type=Path)
    p.add_argument('reference', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    rows = []
    for name in FILES:
        a, b = args.build / name, args.reference / name
        row = {'file': name, 'bytes_identical': a.read_bytes() == b.read_bytes()}
        for key, fn in (('sass', sass), ('host_text', host_text), ('dynamic_symbols', symbols)):
            x, y = fn(a), fn(b)
            row[key] = 'no device code' if x is None and y is None else ('identical' if x == y else 'differs')
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'schema': 'ds41-binary-comparison-v2', 'method': __doc__.strip(),
                                        'rows': rows}, indent=1) + '\n')


if __name__ == '__main__':
    main()
