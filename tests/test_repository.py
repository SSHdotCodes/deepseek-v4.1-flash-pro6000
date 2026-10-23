import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark
import download
import run

REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


class SourceTests(unittest.TestCase):
    def test_component_trees_match_the_lock(self):
        lock = json.loads((ROOT / 'sources.lock.json').read_text())
        for c in lock['components']:
            files = {str(p.relative_to(ROOT / c['path'])): sha256(p)
                     for p in (ROOT / c['path']).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            self.assertEqual(len(files), c['files'], c['path'])
            self.assertEqual(hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                             c['tree_sha256'], c['path'])
            self.assertEqual(len(c['origin_commit']), 40)
            for name in c['release_build_files']:
                self.assertTrue(name.endswith(('CMakeLists.txt', 'build.sh')), name)
            # Source edits made in this repo (image input) are listed, never silent.
            for name in c.get('release_source_changes', []):
                self.assertIn(name, files, c['path'])
        for dep in lock['dependencies'].values():
            self.assertEqual(len(dep['sha256']), 64)

    def test_engine_pins_match_build_script(self):
        cmake = (ROOT / 'engine/CMakeLists.txt').read_text()
        build = (ROOT / 'scripts/build.sh').read_text()
        cudart = '7fb00519a24512bf677b59e87921e5c05aac8246ceb6a57435c0d9dfeef364f6'
        self.assertIn(cudart, cmake)
        self.assertIn(cudart, build)
        lock = json.loads((ROOT / 'sources.lock.json').read_text())
        for lib in ('25cde432108543626df85d887c730bf2cab96803ca16f9654e4abc8d2b45833e',
                    '1489aeb75d55ee55782798343ac66f57771d561cc1fe1e0491568deefa0e6515'):
            self.assertIn(lib, cmake)
            self.assertIn(lib[:8], lock['dependencies']['nvidia-cublas']['use'])
        # No checkout paths in assert messages; the assets bind the Engram library actually built.
        self.assertIn('-ffile-prefix-map=$ROOT/=', build)
        self.assertIn('--engram-library "$ENGRAM"', build)

    def test_upstream_copies_are_unchanged(self):
        inventory = {f['path']: f for f in json.loads((ROOT / 'scripts/assets/hf-lfs.json').read_text())['files']}
        copies = {'api/ds41_api/encoding.py': 'encoding/encoding.py',
                  'api/ds41_api/encoding_README.md': 'encoding/README.md',
                  'api/ds41_api/test_encoding.py': 'encoding/test_encoding.py',
                  'api/inference/examples/example.txt': 'inference/examples/example.txt',
                  'api/inference/examples/example_harmony.json': 'inference/examples/example_harmony.json',
                  'api/inference/examples/images/corn.jpeg': 'inference/examples/images/corn.jpeg'}
        for n in range(1, 6):
            for kind, ext in (('input', 'json'), ('output', 'txt')):
                copies[f'api/ds41_api/tests/test_{kind}_{n}.{ext}'] = f'encoding/tests/test_{kind}_{n}.{ext}'
        for local, upstream in copies.items():
            self.assertEqual(git_blob(ROOT / local), inventory[upstream]['blob_id'], local)

    def test_no_machine_paths(self):
        for p in ROOT.rglob('*'):
            # config.local.json is the user's own (gitignored) settings file and holds their checkpoint path.
            if (p.is_file() and '.git' not in p.parts and p.suffix in ('.py', '.sh', '.md', '.json', '.txt', '.c')
                    and p.relative_to(ROOT) != Path('config.local.json')
                    and not {'models', 'build', 'generated', '.venv', '.venv-assets', 'results'} & set(p.parts)):
                text = p.read_text(errors='ignore')
                for needle in ('/home/' + 'user', 'tail' + '3887', '/Us' + 'ers/'):
                    if needle in text:
                        self.fail(f'{needle} in {p.relative_to(ROOT)}')


class CheckpointTests(unittest.TestCase):
    def test_inventory_is_pinned_and_complete(self):
        lock = json.loads((ROOT / 'scripts/assets/hf-lfs.json').read_text())
        self.assertEqual(lock['repo'], 'deepseek-ai/DeepSeek-V4.1-Flash-1017')
        self.assertEqual(lock['revision'], REVISION)
        shards = [f for f in lock['files'] if f['path'].endswith('.safetensors')]
        self.assertEqual(len(lock['files']), 83)
        self.assertEqual(len(shards), 48)
        self.assertEqual(sum(f['size'] for f in shards), 510296708312)
        for f in shards:
            self.assertEqual(len(f['sha256']), 64)
        for f in lock['files']:
            self.assertNotIn('..', Path(f['path']).parts)

    def test_download_check_detects_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good, bad = b'{"a": 1}\n', b'{"a": 2}\n'
            (root / 'config.json').write_bytes(bad)
            (root / 'shard.safetensors').write_bytes(b'x' * 8)
            blob = hashlib.sha1(b'blob %d\0' % len(good) + good).hexdigest()
            lock = {'repo': 'r', 'revision': REVISION, 'files': [
                {'path': 'config.json', 'size': len(good), 'sha256': None, 'blob_id': blob},
                {'path': 'shard.safetensors', 'size': 8, 'sha256': '0' * 64, 'blob_id': None}]}
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps(lock))
            argv = ['download.py', '--verify-only', '--weights', str(root)]
            with patch.object(download, 'INVENTORY', inventory), patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit) as failed:
                    download.main()
                self.assertIn('content differs: config.json', str(failed.exception))
                (root / 'config.json').write_bytes(good)
                download.main()


class ProfileTests(unittest.TestCase):
    def config(self):
        cfg = run.configuration()
        cfg.update(weights='/data/ds41/weights', assets='/data/ds41/generated', build='/data/ds41/build')
        return cfg

    def test_engine_command_is_the_measured_profile(self):
        with patch.object(run, 'meminfo', return_value={'MemTotal': 126573804 * 1024}):
            cmd = run.command(self.config(), Path('/run/ds41/engine.sock'))
        for flag, value in {'--expert-slots': '4452', '--host-gib': '104', '--max-context': '1048576',
                            '--host-recent': '2', '--host-decay': '256', '--prefill-chunk': '256',
                            '--prefill-dense-batch': '8', '--prefill-warm-rows': '256', '--prefill-warm-gib': '16',
                            '--cuda-stack-bytes': '1024', '--spark-width': '4',
                            '--fused-head': '/data/ds41/build/dist/libds41_fuse_head.so',
                            '--placement': str(ROOT / 'profiles/p80-mtp48.placement')}.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)
        for flag in ('--allow-unqualified', '--graphs', '--streamed-experts', '--dynamic-host', '--reset-cache-history',
                     '--prefix-cache', '--prefix-any-conversation', '--prefill', '--host-embedding', '--spark',
                     '--compact-woa', '--prefetch-lookahead'):
            self.assertIn(flag, cmd)

    def test_k3_profile(self):
        cfg = dict(self.config(), experts='k3', k3_bank='/data/ds41/k3-bank')
        with patch.object(run, 'meminfo', return_value={'MemTotal': 126573804 * 1024}):
            cmd = run.command(cfg, Path('/run/ds41/engine.sock'))
            native = run.command(self.config(), Path('/run/ds41/engine.sock'))
        for flag, value in {'--expert-slots': '5745', '--placement': str(ROOT / 'profiles/k3-p80.placement'),
                            '--exl3-experts': '/data/ds41/k3-bank/records-k3.tsv',
                            '--exl3-receipts': '/data/ds41/k3-bank/receipts.json',
                            '--experts': '/data/ds41/generated/records.tsv'}.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)
        self.assertNotIn('--exl3-experts', native)
        # every other flag is the measured native profile
        strip = {'--expert-slots', '--placement', '--exl3-experts', '--exl3-receipts'}
        keep = lambda c: [x for i, x in enumerate(c) if x not in strip and (i == 0 or c[i - 1] not in strip)]
        self.assertEqual(keep(cmd), keep(native))
        # the same expert VRAM as the native profile: K3 slots plus the drafter's 384 native experts
        self.assertLessEqual(run.K3_EXPERT_SLOTS * 13316096 + 384 * 18808832, run.EXPERT_SLOTS * 18808832)

    def test_environment(self):
        cfg = dict(self.config(), gds=False)
        env = run.environment(cfg, render=False)
        self.assertEqual(env['DS41_SPARK_STOP_MODEL'], '-2.049,2.987,-0.488,0.85,-0.063')
        self.assertEqual(env['DS41_SPARK_STOP_SCALE'], '1,1.05,1.42,1.79,2.16')
        self.assertEqual(env['DS41_SPARK_STOP_Q'], '0.65')
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith('/data/ds41/build/dist:/data/ds41/build/dist/blas:'))
        self.assertNotIn('DS41_MOE_GDS', env)
        with patch.object(run, 'mount_needs_shim', return_value=True):
            env = run.environment(dict(cfg, gds=True), render=False)
        self.assertEqual(env['DS41_MOE_GDS'], '1')
        self.assertEqual(env['LD_PRELOAD'], '/data/ds41/build/dist/mountinfo_shim.so')

    def test_host_tier_follows_ram(self):
        cfg = self.config()
        for total_gib, tier in ((120.7, 104), (250, 104), (115, 99)):
            with patch.object(run, 'meminfo', return_value={'MemTotal': int(total_gib * run.GIB)}):
                self.assertEqual(run.host_gib(cfg), tier)
        self.assertEqual(run.host_gib(dict(cfg, host_gib=96)), 96)
        # the placement's RAM seeds must fit the tier
        seeds = sum(1 for l in (ROOT / 'profiles/p80-mtp48.placement').read_text().splitlines() if l.startswith('host '))
        self.assertLessEqual(seeds * 18808832, run.MIN_HOST_GIB * run.GIB)
        self.assertGreater(seeds * 18808832, (run.MIN_HOST_GIB - 1) * run.GIB)

    def test_mountinfo_shim_detection(self):
        mounts = ('32 2 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n'
                  '40 32 259:3 / /data rw,relatime shared:2 - ext4 /dev/nvme1n1p1 rw,data=ordered\n'
                  '41 32 259:4 / /xfs rw,relatime shared:3 - xfs /dev/nvme2n1p1 rw\n')
        with patch.object(Path, 'read_text', return_value=mounts):
            self.assertTrue(run.mount_needs_shim(Path('/home/u/models')))
            self.assertFalse(run.mount_needs_shim(Path('/data/models')))
            self.assertFalse(run.mount_needs_shim(Path('/xfs/models')))

    def test_placement(self):
        ops = {}
        for line in (ROOT / 'profiles/p80-mtp48.placement').read_text().splitlines():
            if line and not line.startswith('#'):
                ops[line.split()[0]] = ops.get(line.split()[0], 0) + 1
        self.assertEqual(ops, {'recent': 1, 'decay': 1, 'pools': 1, 'lfu': 1, 'host': 5480, 'vram': 3240,
                               'budget': 43, 'host_budget': 43})

    def test_k3_placement(self):
        def parse(name):
            lines = [l for l in (ROOT / 'profiles' / name).read_text().splitlines() if l and not l.startswith('#')]
            budgets = {int(l.split()[1]): int(l.split()[2]) for l in lines if l.startswith('budget ')}
            return [l for l in lines if not l.startswith(('budget ', 'host_budget '))], budgets, lines
        native, native_budgets, _ = parse('p80-mtp48.placement')
        k3, budgets, lines = parse('k3-p80.placement')
        self.assertEqual(native, k3)  # same pins, host seeds and policy
        pins = sum(1 for l in k3 if l.startswith('vram '))
        self.assertEqual(pins + sum(budgets.values()), run.K3_EXPERT_SLOTS)
        self.assertEqual([budgets[l] for l in (40, 41, 42)], [0, 0, 0])
        self.assertTrue(all(int(l.split()[1]) < 15360 for l in k3 if l.startswith(('vram ', 'host '))))
        self.assertTrue(all(l.split()[2] == '0' for l in lines if l.startswith('host_budget ') and int(l.split()[1]) >= 40))
        # the dynamic slots follow the native per-layer shares
        for l in range(40):
            self.assertLessEqual(abs(budgets[l] - native_budgets[l] * 2505 / 1068), 1, l)

    def test_k3_bank_lock(self):
        lock = json.loads((ROOT / 'scripts/assets/k3-bank.json').read_text())
        layers = sorted(l for s in lock['sources'] for l in s['layers'])
        self.assertEqual(layers, list(range(40)))
        for s in lock['sources']:
            self.assertEqual(len(s['revision']), 40)
            self.assertEqual(s['license'], 'MIT')
        self.assertEqual(sorted(map(int, lock['bank_sha256'])), list(range(40)))
        self.assertTrue(all(len(d) == 64 for d in lock['bank_sha256'].values()))
        import k3_bank
        v = k3_bank.VIEWS
        self.assertEqual([v[k] for k in ('gate', 'gate_scale', 'up', 'up_scale', 'down', 'down_scale')],
                         [(0, 4423680), (4423680, 14848), (4438528, 4423680), (8862208, 14848),
                          (8877056, 4423680), (13300736, 14848)])
        # the engine checks the same layout (validate_k3_records)
        forward = (ROOT / 'engine/src/forward.cu').read_text()
        self.assertIn('REC = 13316096, TB = 320 * 144 * 48 * 2, SB = 2 * (5120 + 2304)', forward)


class BenchmarkTests(unittest.TestCase):
    def test_cases(self):
        for spec in benchmark.CASES:
            case = json.loads((ROOT / f'benchmarks/cases/{spec.split(":")[0]}.json').read_text())
            self.assertEqual(len(case['token_ids']), case['prompt_tokens'])
            self.assertEqual(case['sampling']['temperature'], 1.0)
            self.assertEqual(case['sampling']['top_p'], 1.0)
            self.assertEqual(case['thinking'], 'high')

    def test_published_results(self):
        data = json.loads((ROOT / 'benchmarks/results-20261018.json').read_text())

        def mean(runs):
            return round(statistics.mean(c['decode_tok_s'] for r in runs for c in r['cases'][1:]), 2)
        s = data['summary']
        self.assertEqual(mean(data['runs']), s['source_build_seeds_21_22'])
        self.assertEqual(mean(data['installed_runs']), s['installed_build_seeds_21_22'])
        self.assertEqual(mean(data['ram_tier_comparison']['host_gib_104']), s['ram_tier_104_seeds_11_14'])
        self.assertEqual(mean(data['ram_tier_comparison']['host_gib_96']), s['ram_tier_96_seeds_11_14'])
        for a, b in zip(data['runs'], data['installed_runs']):
            self.assertEqual([c['tokens_sha256'] for c in a['cases']], [c['tokens_sha256'] for c in b['cases']])
        for a, b in zip(data['ram_tier_comparison']['host_gib_104'], data['ram_tier_comparison']['host_gib_96']):
            self.assertEqual([c['tokens_sha256'] for c in a['cases']], [c['tokens_sha256'] for c in b['cases']])
        for r in data['runs']:
            self.assertEqual([c['id'] for c in r['cases']],
                             [json.loads((ROOT / f'benchmarks/cases/{x.split(":")[0]}.json').read_text())['id']
                              for x in benchmark.CASES])

    def test_benchmark_script_reproduced_the_tokens(self):
        report = json.loads((ROOT / 'benchmarks/benchmark-py-20261018.json').read_text())
        with patch.object(benchmark, 'REFERENCE', ROOT / 'benchmarks/results-20261018.json'):
            self.assertEqual(benchmark.compare(report['runs']), {'compared': 14, 'identical': 14})
        self.assertEqual(report['summary']['tokens_vs_reference'], {'compared': 14, 'identical': 14})


class AssetTests(unittest.TestCase):
    def test_reference_proof(self):
        proof = json.loads((ROOT / 'scripts/assets/reference-proof.json').read_text())
        text = json.dumps(proof)
        for digest in ('edebf1b3ed13ce1ca4124e6c844624db66f045c46c8d04db37e56e1cb276f73d',
                       '5476db91830804ed4289b9b23ad0e475d863d65f73c246aed88157b22ef0d575',
                       '2f57e890915fc2050039f0cc1061e57ee7ce8a50260dd9692d3503731112ec0b'):
            self.assertIn(digest, text)
        prepare = (ROOT / 'scripts/prepare_assets.py').read_text()
        self.assertIn('edebf1b3ed13ce1ca4124e6c844624db66f045c46c8d04db37e56e1cb276f73d', prepare)


if __name__ == '__main__':
    unittest.main()
