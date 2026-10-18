#!/usr/bin/env python3
"""Loopback DeepSeek-V4.1-Flash-1017 server: ds41d plus the OpenAI/Anthropic API, the measured profile, no swap."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
# VRAM expert-cache layout: which experts are pinned in VRAM or RAM, per layer, including the MTP layer the
# drafter uses (docs/engine.md#expert-residency).
PLACEMENT = ROOT / 'profiles/p80-mtp48.placement'
# 4,452 VRAM expert slots of 18.8 MB (77.99 GiB). With the KV cache, prefill scratch and dense weights this
# leaves about 3 GiB of the 96 GB card free.
EXPERT_SLOTS = 4452
# The measured host RAM tier. More RAM lowers NVMe reads (96 -> 104 GiB: +3.3% decode); larger tiers are untested.
MAX_HOST_GIB = 104
# RAM left to the OS, the API and page cache next to the pinned tier.
RESERVE_GIB = 16

# Every flag is part of the measured profile; see docs/engine.md for what each one does.
ENGINE_FLAGS = [
    # ds41d keeps a guard from its first native-forward milestone; the full-model checks it waited for have
    # passed since (docs/engine.md#quality).
    '--allow-unqualified',
    # decode and verify steps are replayed as CUDA graphs
    '--graphs',
    # routed experts run from the VRAM expert cache, every row of a verify step batched per layer
    '--streamed-experts',
    # the RAM tier is a cache too (LFU, recent window 2, decay every 256 steps), exclusive of VRAM
    '--dynamic-host', '--host-recent', '2', '--host-decay', '256', '--reset-cache-history',
    # a request that extends any earlier conversation prefills only its new tokens
    '--prefix-cache', '--prefix-any-conversation',
    # batched prefill: 256 rows per dense/attention launch, 8 FP8 projection rows per launch
    '--prefill', '--prefill-chunk', '256', '--prefill-dense-batch', '8',
    # after a prompt of 256+ rows, read the experts its last 256 rows used from NVMe into RAM (up to 16 GiB)
    '--prefill-warm-rows', '256', '--prefill-warm-gib', '16',
    # token embedding in host RAM and a 1 KiB CUDA stack: more VRAM for expert slots
    '--host-embedding', '--cuda-stack-bytes', '1024',
    # attention wo_a stays in its native FP8 (dequantized in registers, FP32 FMA) instead of a BF16 copy
    '--compact-woa',
    # DSpark speculative decoding with the checkpoint's own MTP layer: up to 4 drafted tokens per step,
    # verified by the target with exact rejection sampling
    '--spark', '--spark-width', '4',
    # score layer l+1's router on layer l's input and prefetch the predicted RAM-resident experts into VRAM
    # (a transfer hint: routing and outputs are unchanged)
    '--prefetch-lookahead',
]

ENGINE_ENV = {
    # dense projections on the reference (cuBLAS-compatible) path
    'DS41_DENSE_REFERENCE': '1',
    # NVMe misses are read in 2 MiB O_DIRECT chunks, each uploaded as soon as it lands
    'DS41_MOE_CHUNK_UPLOAD': '1', 'DS41_MOE_IO_CHUNK_KB': '2048',
    # expert-cache policy: LFU counts halve every 256 steps, drift-aware admission for width-1 decode,
    # constant-time eviction minima
    'DS41_MOE_DECAY256': '1', 'DS41_MOE_DRIFT_0_50': '1', 'DS41_MOE_TOKEN_MAJOR_WIDTH1': '1',
    'DS41_MOE_FAST_MINIMA': '1',
    # an expert evicted from VRAM moves to the RAM tier instead of being dropped
    'DS41_MOE_DEMOTE': '1',
    # DSpark adaptive draft length. Before drafting token k the engine estimates its acceptance from the
    # drafter's top probability and confidence head (a logistic model), and stops when the expected gain no
    # longer pays for another verify row. The decision uses only what is known before token k is drawn, so
    # sampling stays exact.
    'DS41_SPARK_STOP_Q': '0.65', 'DS41_SPARK_STOP_MODEL': '-2.049,2.987,-0.488,0.85,-0.063',
    'DS41_SPARK_STOP_SCALE': '1,1.05,1.42,1.79,2.16',
    # lookahead prefetch: top 4 predicted experts per row, for the first 2 rows of a step
    'DS41_LOOKAHEAD_K': '4', 'DS41_LOOKAHEAD_N': '2',
    # after a long appended turn the drafter restarts from the append's last 128 rows
    'DS41_SPARK_APPEND_RESTORE': '1',
    # long prompts split into prefill segments that fit the VRAM the expert cache leaves free
    'DS41_PREFILL_CHUNK_FIT': '1',
}
# GPUDirect Storage: NVMe misses during decode are read straight into their VRAM slot (docs/gds.md).
GDS_ENV = {'DS41_MOE_GDS': '1', 'CUFILE_USE_PCIP2PDMA': 'true'}


def configuration():
    cfg = json.loads((ROOT / 'config.example.json').read_text())
    local = ROOT / 'config.local.json'
    if local.exists():
        cfg.update(json.loads(local.read_text()))
    for key in ('weights', 'assets', 'build'):
        cfg[key] = str((ROOT / Path(cfg[key]).expanduser()).resolve())
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', cfg['unit']):
        raise ValueError('unit must contain only letters, digits, underscores or hyphens')
    if not isinstance(cfg['port'], int) or not 1024 <= cfg['port'] <= 65535:
        raise ValueError('port must be an unprivileged port number')
    if cfg['host_gib'] != 'auto' and not (isinstance(cfg['host_gib'], int) and 32 <= cfg['host_gib']):
        raise ValueError('host_gib must be "auto" or an integer of at least 32')
    if cfg['gds'] not in ('auto', True, False):
        raise ValueError('gds must be "auto", true or false')
    if not (isinstance(cfg['max_context'], int) and 4096 <= cfg['max_context'] <= 1048576):
        raise ValueError('max_context must be 4096..1048576')
    return cfg


def meminfo():
    return {l.split(':')[0]: int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()}


def host_gib(cfg):
    if cfg['host_gib'] != 'auto':
        return cfg['host_gib']
    return min(MAX_HOST_GIB, math.floor(meminfo()['MemTotal'] / GIB) - RESERVE_GIB)


def cuda_lib():
    return Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.4')) / 'targets/x86_64-linux/lib'


def p2p_enabled():
    """The nvidia module options that gds/enable-p2p.sh sets (static BAR1, no write-combined IO mappings)."""
    try:
        dwords = Path('/sys/module/nvidia/parameters/NVreg_RegistryDwords').read_text()
    except OSError:
        return False
    return 'RMForceStaticBar1=1' in dwords and (cuda_lib() / 'libcufile.so.0').exists()


def gds(cfg):
    return p2p_enabled() if cfg['gds'] == 'auto' else cfg['gds']


def mount_needs_shim(path):
    """cuFile requires data=ordered to be visible on ext4. The kernel prints data= only when it was passed at
    mount time, so an ext4 mount in the default (ordered) mode looks unsupported without the shim."""
    best, options = '', ''
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, _, right = line.partition(' - ')
        point = left.split()[4].replace('\\040', ' ')
        if (str(path) + '/').startswith(point.rstrip('/') + '/') and len(point) >= len(best):
            fstype, _, super_options = right.split()[:3]
            best, options = point, (fstype, left.split()[5] + ',' + super_options)
    return bool(options) and options[0] == 'ext4' and 'data=' not in options[1]


def runtime_dir(cfg):
    return Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')) / cfg['unit']


def cufile_config(cfg):
    """Render gds/cufile.json with a log directory; returns its path."""
    state = runtime_dir(cfg)
    state.mkdir(parents=True, exist_ok=True)
    out = state / 'cufile.json'
    out.write_text((ROOT / 'gds/cufile.json').read_text().replace('@CUFILE_LOG_DIR@', str(state)))
    return out


def command(cfg, socket):
    dist, assets = Path(cfg['build']) / 'dist', Path(cfg['assets'])
    return [str(dist / 'ds41d'), 'serve', '--socket', str(socket),
            '--weights', cfg['weights'], '--receipt', str(assets / 'full-integrity.json'),
            '--auxiliary', str(assets / 'auxiliary.json'), '--engram', str(assets / 'engram.json'),
            '--rope', str(assets / 'rope.bin'), '--experts', str(assets / 'records.tsv'),
            '--engram-lib', str(dist / 'libds41_engram.so'), '--fused-head', str(dist / 'libds41_fuse_head.so'),
            '--placement', str(PLACEMENT), '--expert-slots', str(EXPERT_SLOTS),
            '--host-gib', str(host_gib(cfg)), '--max-context', str(cfg['max_context'])] + ENGINE_FLAGS


def environment(cfg, render=True):
    dist = Path(cfg['build']) / 'dist'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DS41_', 'CUFILE_'))}
    env.pop('LD_PRELOAD', None)
    env.update(ENGINE_ENV)
    # the pinned cuBLAS pair (dist/blas) before CUDA 13.4's own libraries (cudart, cuFile)
    env['LD_LIBRARY_PATH'] = f'{dist}:{dist}/blas:{cuda_lib()}'
    if gds(cfg):
        env.update(GDS_ENV)
        env['CUFILE_ENV_PATH_JSON'] = str(cufile_config(cfg) if render else runtime_dir(cfg) / 'cufile.json')
        if mount_needs_shim(Path(cfg['weights'])):
            env['LD_PRELOAD'] = str(dist / 'mountinfo_shim.so')
    return env


def api_command(cfg, socket):
    return [str(ROOT / '.venv/bin/python'), '-m', 'ds41_api.server', '--backend', 'socket', '--socket', str(socket),
            '--tokenizer', str(Path(cfg['weights']) / 'tokenizer.json'), '--model', cfg['model_id'],
            '--host', '127.0.0.1', '--port', str(cfg['port']), '--max-concurrent', '1',
            '--max-queue', str(cfg['max_queue']), '--max-tokens-limit', str(cfg['max_tokens_limit'])]


def preflight(cfg):
    if sys.platform != 'linux':
        raise RuntimeError('This profile requires Linux, NVIDIA CUDA and cgroup v2')
    dist = Path(cfg['build']) / 'dist'
    if not (dist / 'ds41d').exists():
        raise RuntimeError('Build first: scripts/build.sh')
    if not (ROOT / '.venv/bin/python').exists():
        raise RuntimeError('Create the API environment first: scripts/setup.sh')
    aux = json.loads((Path(cfg['assets']) / 'auxiliary.json').read_text())
    if aux['weight_root'] != cfg['weights']:
        raise RuntimeError(f'The assets were prepared for {aux["weight_root"]}; run scripts/build.sh again')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1 or 'RTX PRO 6000 Blackwell' not in gpu[0] or int(gpu[0].split(',')[1]) < 97000:
        raise RuntimeError('This measured profile targets exactly one 96 GB RTX PRO 6000 Blackwell')
    if int(gpu[0].split(',')[2]) > 2048:
        raise RuntimeError('The GPU is in use; stop other GPU processes (the profile needs nearly all 96 GB)')
    tier, mem = host_gib(cfg), meminfo()
    if tier < 32:
        raise RuntimeError('Need at least 48 GiB of RAM for the pinned expert tier (128 GB measured)')
    if mem['MemAvailable'] < (tier + 4) * GIB:
        raise RuntimeError(f'Need {tier + 4} GiB available RAM for a {tier} GiB pinned expert tier; '
                           f'{mem["MemAvailable"] / GIB:.1f} GiB is available')
    if not Path('/sys/fs/cgroup/cgroup.controllers').exists():
        raise RuntimeError('cgroup v2 is required to enforce MemorySwapMax=0')
    mode = ('GDS P2P' + (' + mountinfo shim' if mount_needs_shim(Path(cfg['weights'])) else '')) if gds(cfg) \
        else 'host bounce (GDS off; see docs/gds.md)'
    print(f'Preflight passed: idle RTX PRO 6000, {tier} GiB RAM tier, NVMe reads: {mode}.', flush=True)


def evict_page_cache(path):
    """Drop the checkpoint's clean page cache before pinning the RAM tier, so pinning does not start with
    reclaim pressure (systemd-oomd acts on sustained pressure)."""
    for root, _, files in os.walk(path):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def serve(cfg):
    cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    if (cgroup / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Refusing to pin experts without effective memory.swap.max=0')
    socket = runtime_dir(cfg) / 'engine.sock'
    socket.parent.mkdir(parents=True, exist_ok=True)
    socket.unlink(missing_ok=True)
    evict_page_cache(cfg['weights'])
    argv = command(cfg, socket)
    print(shlex.join(argv), flush=True)
    engine = subprocess.Popen(argv, env=environment(cfg))
    api = None

    def stop(*_):
        for p in (api, engine):
            if p and p.poll() is None:
                p.terminate()
        for p in (api, engine):
            if p:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.time() + 900
    while not socket.exists():
        if engine.poll() is not None:
            raise SystemExit(f'ds41d exited while loading (rc={engine.returncode})')
        if time.time() > deadline:
            stop()
        time.sleep(1)
    argv = api_command(cfg, socket)
    print(shlex.join(argv), flush=True)
    api = subprocess.Popen(argv, cwd=ROOT / 'api')
    while True:
        for name, p in (('ds41d', engine), ('api', api)):
            if p.poll() is not None:
                print(f'{name} exited rc={p.returncode}; stopping the service', flush=True)
                stop()
        time.sleep(2)


def wait_ready(cfg):
    url = f'http://127.0.0.1:{cfg["port"]}/health'
    for _ in range(1200):
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', cfg['unit']], check=False).returncode:
            raise RuntimeError('Server exited; run scripts/run.py logs')
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f'Ready: http://127.0.0.1:{cfg["port"]}/v1  model {cfg["model_id"]}', flush=True)
                    return
        except (OSError, TimeoutError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError('Readiness timed out; inspect logs (the service is still managed by systemd)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'stop', 'status', 'logs', 'command', 'preflight', 'serve'])
    args = p.parse_args()
    cfg = configuration()
    if args.action == 'command':
        env = environment(cfg, render=False)
        socket = runtime_dir(cfg) / 'engine.sock'
        keys = sorted(k for k in env if k.startswith(('DS41_', 'CUFILE_'))) + ['LD_LIBRARY_PATH'] + \
            (['LD_PRELOAD'] if 'LD_PRELOAD' in env else [])
        print(' '.join(f'{k}={shlex.quote(env[k])}' for k in keys))
        print(shlex.join(command(cfg, socket)))
        print(shlex.join(api_command(cfg, socket)))
    elif args.action == 'preflight':
        preflight(cfg)
    elif args.action == 'start':
        preflight(cfg)
        # The pinned tier plus about 2 GiB of process memory; page cache fills the rest up to the limit.
        limit = host_gib(cfg) + 12
        subprocess.run(['systemd-run', '--user', '--collect', '--unit', cfg['unit'],
                        f'--property=MemoryMax={limit}G', '--property=MemorySwapMax=0',
                        '--property=KillMode=control-group', '--property=TimeoutStopSec=60',
                        '--', sys.executable, str(Path(__file__).resolve()), 'serve'], check=True)
        wait_ready(cfg)
    elif args.action == 'serve':
        serve(cfg)
    elif args.action == 'logs':
        subprocess.run(['journalctl', '--user', '-u', cfg['unit'], '-n', '100', '--no-pager'], check=True)
    else:
        subprocess.run(['systemctl', '--user', args.action, cfg['unit']], check=False)


if __name__ == '__main__':
    main()
