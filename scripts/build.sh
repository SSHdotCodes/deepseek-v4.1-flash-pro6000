#!/usr/bin/env bash
# Build ds41d and every component from source into build/dist, the runtime directory scripts/run.py uses.
#   scripts/build.sh [--weights DIR] [--assets DIR] [--build DIR] [--jobs N]
# (defaults: "weights", "assets" and "build" in config.local.json or config.example.json)
# Order: pinned wheels -> components -> engine assets (scripts/prepare_assets.py, bound to the Engram library
# built here) -> engine -> engine tests -> build/dist. The engine compiles in the digest of auxiliary.json, so
# moving the weights or the assets directory means running this script again.
# Requirements: CUDA Toolkit 13.4.0 (CUDA_HOME, default /usr/local/cuda-13.4), GCC/G++ 13+, CMake 3.24+,
# the .venv-assets environment from scripts/setup.sh.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Defaults come from config.example.json and config.local.json, as for scripts/run.py.
mapfile -t paths < <(python3 "$ROOT/scripts/run.py" paths)
WEIGHTS="${paths[0]}"; ASSETS="${paths[1]}"; BUILD="${paths[2]}"; JOBS="$(nproc)"
while (( $# )); do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2;;
    --assets) ASSETS="$2"; shift 2;;
    --build) BUILD="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    -h|--help) sed -n '2,9p' "$0"; exit 0;;
    *) echo "build.sh: unknown argument $1" >&2; exit 2;;
  esac
done
WEIGHTS="$(realpath -m "$WEIGHTS")"; ASSETS="$(realpath -m "$ASSETS")"; BUILD="$(realpath -m "$BUILD")"
[[ -f "$WEIGHTS/model.safetensors.index.json" ]] || {
  echo "build.sh: no checkpoint at $WEIGHTS; run scripts/download.py or pass --weights" >&2; exit 2; }
ASSET_PYTHON="${ASSET_PYTHON:-$ROOT/.venv-assets/bin/python}"
[[ -x "$ASSET_PYTHON" ]] || { echo "build.sh: $ASSET_PYTHON not found; run scripts/setup.sh" >&2; exit 2; }
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.4}"
NVCC="$CUDA_HOME/bin/nvcc"
"$NVCC" --version | grep -q "release 13.4," || {
  echo "build.sh: CUDA 13.4 is required (the engine pins its cudart); set CUDA_HOME" >&2; exit 2; }
# ds41d refuses to start with any other cudart than CUDA 13.4.0's libcudart.so.13.4.92.
CUDART="$(realpath "$CUDA_HOME/targets/x86_64-linux/lib/libcudart.so.13")"
echo "7fb00519a24512bf677b59e87921e5c05aac8246ceb6a57435c0d9dfeef364f6  $CUDART" | sha256sum -c --quiet - || {
  echo "build.sh: $CUDART is not CUDA 13.4.0's cudart (13.4.92); install the 13.4.0 toolkit" >&2; exit 2; }
# The API venv from scripts/setup.sh has pip, which fetches the two pinned wheels.
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DEPS="$BUILD/deps"; COMP="$BUILD/components"; DIST="$BUILD/dist"
mkdir -p "$DEPS" "$COMP" "$DIST"
step() { printf '\n== %s\n' "$*"; }

# 1. Pinned binary dependencies.
pin() { "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1]))["dependencies"][sys.argv[2]]; print(d["version"], d["sha256"])' \
  "$ROOT/sources.lock.json" "$1"; }
fetch() {  # pip-name -> path of the verified wheel
  local name="$1" version sha whl
  read -r version sha < <(pin "$name")
  whl="$(ls "$DEPS/${name//-/_}-$version-"*.whl 2>/dev/null | head -1 || true)"
  if [[ -z "$whl" ]]; then
    "$PYTHON" -m pip download --quiet --no-deps --only-binary=:all: --python-version 3.12 \
      --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 "$name==$version" -d "$DEPS" >&2
    whl="$(ls "$DEPS/${name//-/_}-$version-"*.whl | head -1)"
  fi
  echo "$sha  $whl" | sha256sum -c --quiet - >&2 || { echo "build.sh: checksum mismatch for $whl" >&2; exit 1; }
  echo "$whl"
}
step "pinned dependencies"
tilelang_whl="$(fetch tilelang)"
cublas_whl="$(fetch nvidia-cublas)"
if [[ ! -d "$DEPS/tilelang" ]]; then
  tmp="$(mktemp -d)"; "$PYTHON" -m zipfile -e "$tilelang_whl" "$tmp"; mv "$tmp/tilelang" "$DEPS/tilelang"; rm -rf "$tmp"
fi
if [[ ! -f "$DEPS/blas/lib/libcublasLt.so.13" ]]; then
  tmp="$(mktemp -d)"; "$PYTHON" -m zipfile -e "$cublas_whl" "$tmp"; mkdir -p "$DEPS/blas/lib"
  cp "$tmp/nvidia/cu13/lib/libcublas.so.13" "$tmp/nvidia/cu13/lib/libcublasLt.so.13" "$DEPS/blas/lib/"; rm -rf "$tmp"
fi

# 2. Components, installed into one prefix (lib/ + include/).
# The dense/Engram libraries were built without a CMake build type (no -DNDEBUG); keep that to reproduce them.
# Their assert messages name source files: keep the checkout path out of them. (nvcc still derives
# anonymous-namespace symbol names from the source path, so the Engram library's bytes, which auxiliary.json
# authenticates, depend on where the repo is cloned; step 3 binds the library actually built.)
PREFIX_MAP="-ffile-prefix-map=$ROOT/="
for c in moe dense spark-policy fuse-head exl3; do
  step "component $c"
  build_type=Release; [[ $c == dense ]] && build_type=
  cmake -S "$ROOT/components/$c" -B "$BUILD/$c" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CXX_FLAGS="$PREFIX_MAP" -DCMAKE_CUDA_FLAGS="-Xcompiler=$PREFIX_MAP" \
    -DCMAKE_INSTALL_PREFIX="$COMP" > "$BUILD/$c.configure.log"
  cmake --build "$BUILD/$c" -j "$JOBS"
  cmake --install "$BUILD/$c" > /dev/null
done
step "component attn"
TILELANG_DIR="$DEPS/tilelang" OUT_DIR="$COMP" NVCC="$NVCC" "$ROOT/components/attn/build.sh"

# 3. Engine assets. auxiliary.json binds the checkpoint files, the generated assets and the Engram library
# built above. The first run hashes all 510 GB of shards (about 6 minutes from a PCIe 5.0 drive); later runs
# rehash only shards whose device, inode, size, mtime or ctime changed (a moved checkpoint is hashed again),
# and take about 10 seconds.
ENGRAM="$COMP/lib/libds41_engram.so"
step "engine assets $ASSETS"
"$ASSET_PYTHON" "$ROOT/scripts/prepare_assets.py" --weights "$WEIGHTS" --out "$ASSETS" --engram-library "$ENGRAM"

# 4. Engine (DSpark drafter and policy sidecar on, MoE source API 4, pinned cuBLAS pair).
step "engine"
cmake -S "$ROOT/engine" -B "$BUILD/engine" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DDS41_COMPONENT_DIR="$COMP" -DDS41_BLAS_RUNTIME_DIR="$DEPS/blas" -DDS41_AUXILIARY_FILE="$ASSETS/auxiliary.json" \
  -DDS41_MOE_SOURCE_API=4 -DDS41_SPARK_NATIVE=ON -DDS41_SPARK_POLICY_V2=ON > "$BUILD/engine.configure.log"
cmake --build "$BUILD/engine" -j "$JOBS"
ctest --test-dir "$BUILD/engine" --output-on-failure

# 5. Runtime directory: binary, libraries, cuBLAS pair, generated assets, GDS mountinfo shim.
step "runtime directory $DIST"
rm -rf "$DIST"; mkdir -p "$DIST/blas"
cp "$BUILD/engine/ds41d" "$BUILD/engine/libengine_ops.so" "$BUILD/engine/libds41_spark.so" "$DIST/"
cp "$COMP"/lib/*.so "$DIST/"
cp "$DEPS"/blas/lib/*.so.13 "$DIST/blas/"
ln -s "$ASSETS" "$DIST/generated"
gcc -O2 -shared -fPIC "$ROOT/gds/mountinfo_shim.c" -o "$DIST/mountinfo_shim.so" -ldl
"$PYTHON" - "$DIST" <<'PY'
import hashlib, json, sys
from pathlib import Path
dist = Path(sys.argv[1])
files = {p.name if p.parent == dist else f"blas/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(dist.rglob("*")) if p.is_file() and "generated" not in p.parts}
(dist / "manifest.json").write_text(json.dumps({"files": files}, indent=1) + "\n")
print(f"{len(files)} files")
PY
LD_LIBRARY_PATH="$DIST" ldd "$DIST/ds41d" | grep -E "not found|ds41|cublas" || true
echo "built $DIST"
