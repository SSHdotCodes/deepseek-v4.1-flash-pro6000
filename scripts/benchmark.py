#!/usr/bin/env python3
"""Decode benchmark on the exact service engine command, the method behind the published numbers.

Starts its own ds41d (the service must be stopped) in a no-swap systemd scope, then submits the fixed cases in
benchmarks/cases over the engine socket, once per seed: an 8K-token agentic warm-up (64 tokens, not counted),
then three short chats, one 2.3K-token multi-turn follow-up and two 8K-token agentic prompts, 256 tokens each.
Prompts are pre-encoded with the official encoding and thinking at "high"; temperature 1.0, top-p 1.0.
Records TTFT, decode tok/s (first to last token) and a SHA-256 of every token sequence. The tokens depend only on
the seed: on the same GPU, driver and build they should equal benchmarks/results-20261018.json (with
"experts": "k3" in config.local.json, benchmarks/results-k3-20261023.json).

  python3 scripts/benchmark.py --seeds 21 22 --out results/bench.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import run

CASES = ['agentic-001:64', 'chat-000:256', 'agentic-000:256', 'mt-000:256', 'chat-001:256', 'chat-002:256',
         'agentic-001:256']
REFERENCE = run.ROOT / 'benchmarks/results-20261018.json'
# with "experts": "k3" (the EXL3 3-bit bank) the tokens are compared with this run instead
K3_REFERENCE = run.ROOT / 'benchmarks/results-k3-20261023.json'


def submit(path, request_id, case, sampling):
    op = {'op': 'submit', 'request_id': request_id, 'token_ids': case['token_ids'], 'sampling': sampling,
          'draft': 'on'}
    tokens, stamps = [], []
    with socket.socket(socket.AF_UNIX) as c:
        c.settimeout(3600)
        c.connect(str(path))
        begin = time.perf_counter()
        c.sendall((json.dumps(op) + '\n').encode())
        with c.makefile('r') as stream:
            for line in stream:
                now = time.perf_counter()
                event = json.loads(line)
                if event['type'] == 'token':
                    tokens.append(event['token_id'])
                    stamps.append(now)
                elif event['type'] == 'finish':
                    break
            else:
                raise RuntimeError('engine closed the socket before finishing')
    if event.get('error') or event['finish_reason'] not in ('stop', 'length'):
        raise RuntimeError(f'engine failed: {event.get("error")}')
    stats = event.get('stats', {})
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0
    tiers = stats.get('decode_tier_counters') or stats.get('tier_counters') or {}
    return {'id': case['id'], 'prompt_tokens': len(case['token_ids']), 'generated': len(tokens),
            'ttft_s': round(stamps[0] - begin, 3) if stamps else None,
            'decode_tok_s': round((len(tokens) - 1) / span, 3) if span else None,
            'tokens_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            'draft_acceptance': stats.get('acceptance'),
            'expert_host_hits': tiers.get('expert_host_hits'), 'expert_nvme_misses': tiers.get('expert_nvme_misses')}


def run_seed(cfg, seed, workdir):
    path = workdir / 'engine.sock'
    path.unlink(missing_ok=True)
    run.evict_page_cache(cfg['weights'])
    if cfg['experts'] == 'k3':
        run.evict_page_cache(cfg['k3_bank'])
    log = (workdir / f'engine-{seed}.log').open('w')
    start = time.perf_counter()
    engine = subprocess.Popen(run.command(cfg, path), stdout=log, stderr=subprocess.STDOUT,
                              env=run.environment(cfg))
    try:
        while not path.exists():
            if engine.poll() is not None:
                raise RuntimeError(f'ds41d exited while loading; see {log.name}')
            if time.perf_counter() - start > 900:
                raise RuntimeError('ds41d did not load within 15 minutes')
            time.sleep(0.2)
        load_s = round(time.perf_counter() - start, 1)
        rows = []
        for i, spec in enumerate(CASES):
            name, limit = spec.split(':')
            case = json.loads((run.ROOT / f'benchmarks/cases/{name}.json').read_text())
            sampling = dict(case['sampling'], max_tokens=int(limit), seed=seed, logprobs=0)
            row = submit(path, f'bench-{i}', case, sampling)
            row['warmup'] = i == 0
            rows.append(row)
            print(json.dumps({'seed': seed, **{k: row[k] for k in ('id', 'prompt_tokens', 'generated', 'ttft_s',
                                                                    'decode_tok_s')}}), flush=True)
        return {'seed': seed, 'load_s': load_s, 'cases': rows,
                'decode_tok_s_mean': round(statistics.mean(r['decode_tok_s'] for r in rows[1:]), 3)}
    finally:
        engine.terminate()
        try:
            engine.wait(timeout=60)
        except subprocess.TimeoutExpired:
            engine.kill()
        log.close()


def compare(results, path=None):
    path = path or REFERENCE
    if not path.exists():
        return None
    reference = {(r['seed'], i): c['tokens_sha256'] for r in json.loads(path.read_text())['runs']
                 for i, c in enumerate(r['cases'])}
    pairs = [(reference.get((r['seed'], i)), c['tokens_sha256']) for r in results for i, c in enumerate(r['cases'])]
    known = [(a, b) for a, b in pairs if a]
    return {'compared': len(known), 'identical': sum(a == b for a, b in known)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[21, 22])
    p.add_argument('--out', type=Path, default=run.ROOT / 'results/bench.json')
    args = p.parse_args()
    cfg = run.configuration()
    if os.environ.get('DS41_BENCH_SCOPE') != '1':
        # Same memory policy as the service: no swap, so the pinned tier cannot be paged out.
        run.preflight(cfg)
        limit = run.host_gib(cfg) + 12
        os.execvp('systemd-run', ['systemd-run', '--user', '--scope', '--quiet', f'--property=MemoryMax={limit}G',
                                  '--property=MemorySwapMax=0', '--setenv=DS41_BENCH_SCOPE=1', sys.executable,
                                  str(Path(__file__).resolve()), *sys.argv[1:]])
    workdir = Path(tempfile.mkdtemp(prefix='ds41-bench-'))
    results = [run_seed(cfg, seed, workdir) for seed in args.seeds]
    summary = {'decode_tok_s_mean': round(statistics.mean(r['decode_tok_s_mean'] for r in results), 3),
               'per_seed': {r['seed']: r['decode_tok_s_mean'] for r in results},
               'tokens_vs_reference': compare(results, K3_REFERENCE if cfg['experts'] == 'k3' else REFERENCE)}
    environment = {k: v for k, v in run.environment(cfg, render=False).items() if k.startswith(('DS41_', 'CUFILE_'))}
    report = {'schema': 'ds41-bench-v1', 'experts': cfg['experts'], 'host_gib': run.host_gib(cfg), 'gds': run.gds(cfg),
              'command': run.command(cfg, workdir / 'engine.sock'), 'environment': environment,
              'summary': summary, 'runs': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    shutil.rmtree(workdir, ignore_errors=True)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
