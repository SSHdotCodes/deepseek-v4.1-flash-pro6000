#!/usr/bin/env python3
"""Build or verify the EXL3 3-bit expert bank for `"experts": "k3"` (docs/exl3.md).

  fetch DIR    read the K3 routed-expert tensors of target layers 0..39 from the two pinned public EXL3 quants
               (scripts/assets/k3-bank.json) with HTTP range reads, straight into DIR/bank-LL.bin (5.1 GB per layer,
               191 GiB in all). Each finished layer must match its pinned SHA-256, so every download is byte-identical
               to the measured bank. Resumes: verified layers are kept. Then writes DIR/records-k3.tsv (the manifest
               ds41d --exl3-experts reads) and DIR/receipts.json.
  verify DIR   recompute every bank file's SHA-256 from disk against the pin and re-stamp its receipt. ds41d refuses
               a bank file whose size or mtime changed since its receipt, so run this after copying or moving a bank.

Only the Python standard library is required; huggingface_hub, when installed (.venv), supplies your token.
  .venv/bin/python scripts/k3_bank.py fetch models/k3-bank [--layers 0-39] [--threads 16]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'scripts/assets/k3-bank.json').read_text())
REC = LOCK['record_bytes']
H, I = 5120, 2304
# record layout: (projection, record view, (k, n)); trellis [k/16, n/16, 48] int16, then FP16 suh[k] + svh[n]
ORDER = (('w1', 'gate', (H, I)), ('w3', 'up', (H, I)), ('w2', 'down', (I, H)))


def layout():
    off, views = 0, {}
    for w, name, (k, n) in ORDER:
        tb = (k // 16) * (n // 16) * 48 * 2
        views[name] = (off, tb)
        views[name + '_scale'] = (off + tb, 2 * (k + n))
        off += tb + 2 * (k + n)
    assert off <= REC and REC % 4096 == 0
    return views


VIEWS = layout()


def source(layer):
    for s in LOCK['sources']:
        if layer in s['layers']:
            return s
    raise ValueError(f'no source for layer {layer}')


# ---- HTTP range reads from the Hugging Face Hub ----
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect)
_cdn, _cdn_lock = {}, threading.Lock()


def _auth():
    try:
        from huggingface_hub.utils import build_hf_headers
        return build_hf_headers()
    except Exception:
        return {}


def _backoff(err, attempt):
    wait = None
    value = err.headers.get('Retry-After') if err.headers else None
    if value:
        try:
            wait = float(value)
        except ValueError:
            pass
    time.sleep(min(120.0, (wait or 5.0 * (attempt + 1)) + 1.0))


def _resolve(url):
    """The signed CDN URL behind a Hub resolve URL, cached for 10 minutes (the Hub limits resolver calls)."""
    now = time.time()
    with _cdn_lock:
        hit = _cdn.get(url)
        if hit and hit[1] > now:
            return hit[0]
    for attempt in range(12):
        try:
            _opener.open(urllib.request.Request(url, method='HEAD', headers=_auth()), timeout=60)
            location = url
            break
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                location = urllib.parse.urljoin(url, e.headers['Location'])
                break
            if e.code == 429:
                _backoff(e, attempt)
                continue
            raise
        except OSError:
            if attempt == 11:
                raise
            time.sleep(2 * (attempt + 1))
    with _cdn_lock:
        _cdn[url] = (location, now + 600)
    return location


def get(url, first, last, tries=12):
    for attempt in range(tries):
        target = _resolve(url)
        headers = {'Range': f'bytes={first}-{last}'}
        if target.startswith('https://huggingface.co/'):
            headers.update(_auth())
        try:
            with urllib.request.urlopen(urllib.request.Request(target, headers=headers), timeout=180) as r:
                data = r.read()
            if len(data) != last - first + 1:
                raise OSError(f'short read {len(data)} for bytes {first}-{last}')
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _backoff(e, attempt)
            elif e.code in (400, 403, 404, 410) and target != url:
                with _cdn_lock:
                    _cdn.pop(url, None)  # expired signature: resolve again
            elif attempt == tries - 1:
                raise
            else:
                time.sleep(2 * (attempt + 1))
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise OSError(f'giving up on {url}')


class Repo:
    def __init__(self, repo, revision):
        self.base = f'https://huggingface.co/{repo}/resolve/{revision}/'
        self.name = f'{repo}@{revision[:8]}'
        index = json.loads(self._read('model.safetensors.index.json'))
        self.index, self.headers, self.lock = index['weight_map'], {}, threading.Lock()

    def _read(self, name):
        with urllib.request.urlopen(urllib.request.Request(self.base + name, headers=_auth()), timeout=120) as r:
            return r.read()

    def header(self, shard):
        with self.lock:
            if shard in self.headers:
                return self.headers[shard]
        n = struct.unpack('<Q', get(self.base + shard, 0, 7))[0]
        h = json.loads(get(self.base + shard, 8, 8 + n - 1))
        h['__base__'] = 8 + n
        with self.lock:
            self.headers[shard] = h
        return h

    def meta(self, name):
        """(shard, header entry), or None when the tensor is absent (some indexes list tensors a shard lacks)."""
        shard = self.index.get(name)
        if shard is None:
            return None
        entry = self.header(shard).get(name)
        return (shard, entry) if entry else None


def expert_record(repo, layer, e):
    """One expert's record: its nine tensors, checked (K3, mul1 codebook, shapes) and placed at their views."""
    p = f'layers.{layer}.ffn.experts.{e}'
    spans, record = [], bytearray(REC)
    for w, name, (k, n) in ORDER:
        if repo.meta(f'{p}.{w}.mul1') is None or repo.meta(f'{p}.{w}.mcg') is not None:
            raise ValueError(f'{repo.name} {p}.{w}: not the mul1 codebook')
        want = {'trellis': ('I16', [k // 16, n // 16, 48]), 'suh': ('F16', [k]), 'svh': ('F16', [n])}
        dest = {'trellis': VIEWS[name][0], 'suh': VIEWS[name + '_scale'][0], 'svh': VIEWS[name + '_scale'][0] + 2 * k}
        for t, (dtype, shape) in want.items():
            found = repo.meta(f'{p}.{w}.{t}')
            if found is None:
                raise ValueError(f'{repo.name} {p}.{w}.{t} missing')
            shard, entry = found
            if entry['dtype'] != dtype or entry['shape'] != shape:
                raise ValueError(f'{repo.name} {p}.{w}.{t}: {entry["dtype"]} {entry["shape"]}, '
                                 f'want {dtype} {shape} (K3)')
            a, b = entry['data_offsets']
            base = repo.header(shard)['__base__']
            spans.append((shard, base + a, base + b, dest[t]))
    spans.sort()
    i = 0
    while i < len(spans):  # one range read for tensors less than 1 MiB apart in the same shard
        j = i
        while j + 1 < len(spans) and spans[j + 1][0] == spans[i][0] and spans[j + 1][1] - spans[j][2] <= 1 << 20:
            j += 1
        raw = get(repo.base + spans[i][0], spans[i][1], spans[j][2] - 1)
        for shard, a, b, d in spans[i:j + 1]:
            record[d:d + b - a] = raw[a - spans[i][1]:b - spans[i][1]]
        i = j + 1
    return bytes(record)


def file_sha256(path):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        while chunk := os.read(fd, 64 << 20):
            digest.update(chunk)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return digest.hexdigest()


def receipt(path, layer):
    st = os.stat(path)
    s = source(layer)
    return {'file': path.name, 'bytes': st.st_size, 'sha256': LOCK['bank_sha256'][str(layer)],
            'mtime_ns': st.st_mtime_ns, 'source': f'{s["repo"]}@{s["revision"]}',
            'verified': time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def load_receipts(out):
    path = out / 'receipts.json'
    return json.loads(path.read_text()) if path.exists() else {}


def save_receipts(out, receipts):
    tmp = out / 'receipts.json.partial'
    tmp.write_text(json.dumps(receipts, indent=1) + '\n')
    os.replace(tmp, out / 'receipts.json')


def current(out, receipts, layer):
    """A layer is done when its receipt pins the expected digest and the file still has the receipt's size/mtime."""
    r, path = receipts.get(str(layer)), out / f'bank-{layer:02d}.bin'
    if not r or not path.exists() or r.get('sha256') != LOCK['bank_sha256'][str(layer)]:
        return False
    st = os.stat(path)
    return st.st_size == 384 * REC and st.st_mtime_ns == r.get('mtime_ns')


def manifest(out):
    """DS41MOE01 records: R id bytes (offset, bytes) x6 for gate/gate_scale/up/up_scale/down/down_scale, then one
    extent E id "path" file_offset bytes dst_offset required_bytes expected_file_bytes."""
    lines = ['# DS41MOE01 EXL3 K3 mul1 bank (scripts/k3_bank.py): gate/up/down -> trellis, *_scale -> suh then svh']
    v = VIEWS
    for layer in range(40):
        path = (out / f'bank-{layer:02d}.bin').resolve()
        for e in range(384):
            eid = layer * 384 + e
            lines.append('R %d %d %d %d %d %d %d %d %d %d %d %d %d %d' % (
                eid, REC, *v['gate'], *v['gate_scale'], *v['up'], *v['up_scale'], *v['down'], *v['down_scale']))
            lines.append('E %d "%s" %d %d 0 %d %d' % (eid, path, e * REC, REC, REC, 384 * REC))
    (out / 'records-k3.tsv').write_text('\n'.join(lines) + '\n')


def fetch(out, layers, threads):
    out.mkdir(parents=True, exist_ok=True)
    receipts, repos, t0, fetched = load_receipts(out), {}, time.time(), 0
    for layer in layers:
        if current(out, receipts, layer):
            continue
        s = source(layer)
        repo = repos.get(s['repo']) or repos.setdefault(s['repo'], Repo(s['repo'], s['revision']))
        path, tmp = out / f'bank-{layer:02d}.bin', out / f'bank-{layer:02d}.partial'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, 384 * REC)

            def one(e):
                os.pwrite(fd, expert_record(repo, layer, e), e * REC)
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(one, range(384)))
            os.fsync(fd)
        finally:
            os.close(fd)
        digest = file_sha256(tmp)
        if digest != LOCK['bank_sha256'][str(layer)]:
            raise SystemExit(f'layer {layer}: SHA-256 {digest} differs from the pin; {tmp} kept for inspection')
        os.replace(tmp, path)
        receipts[str(layer)] = receipt(path, layer)
        save_receipts(out, receipts)
        fetched += 384 * REC
        rate = fetched / 1e6 / max(1e-9, time.time() - t0)
        print(json.dumps({'layer': layer, 'source': repo.name, 'sha256': digest[:16], 'MB_s': round(rate, 1)}),
              flush=True)
    if all(current(out, receipts, layer) for layer in range(40)):
        manifest(out)
        print(f'bank complete: {out.resolve()}/records-k3.tsv', flush=True)
    else:
        print(f'layers missing: {[l for l in range(40) if not current(out, receipts, l)]}', flush=True)


def verify(out):
    receipts, bad = load_receipts(out), []

    def check(layer):
        path = out / f'bank-{layer:02d}.bin'
        if not path.exists():
            return layer, False
        before = os.stat(path).st_mtime_ns
        ok = file_sha256(path) == LOCK['bank_sha256'][str(layer)] and os.stat(path).st_mtime_ns == before
        return layer, ok and os.stat(path).st_size == 384 * REC
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(check, range(40)))
    bad = [layer for layer, ok in results if not ok]
    if bad:
        raise SystemExit(f'verify failed for layers {bad}; run fetch to rebuild them')
    for layer in range(40):
        receipts[str(layer)] = receipt(out / f'bank-{layer:02d}.bin', layer)
    save_receipts(out, receipts)
    manifest(out)
    print(f'verified 40 layers; receipts and records-k3.tsv rewritten in {out.resolve()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['fetch', 'verify'])
    p.add_argument('dir', type=Path)
    p.add_argument('--layers', default='0-39', help='fetch only: a layer or an inclusive range, e.g. 0-39 or 18')
    p.add_argument('--threads', type=int, default=16)
    a = p.parse_args()
    if a.action == 'verify':
        verify(a.dir)
        return
    lo, _, hi = a.layers.partition('-')
    layers = range(int(lo), int(hi or lo) + 1)
    if not layers or layers.start < 0 or layers.stop > 40:
        raise SystemExit('--layers must lie within 0-39')
    fetch(a.dir, layers, a.threads)


if __name__ == '__main__':
    sys.exit(main())
