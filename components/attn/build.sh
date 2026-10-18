#!/usr/bin/env bash
# Builds libds41_attn.so with the same single nvcc invocation as the original attention build.
# Only addition: a soname, so ds41d records the library by name instead of by absolute path.
#   TILELANG_DIR=<tilelang 0.1.12 package dir> OUT_DIR=<prefix> [NVCC=...] components/attn/build.sh
set -euo pipefail
: "${TILELANG_DIR:?set TILELANG_DIR to the extracted tilelang 0.1.12 package (has src/ and 3rdparty/cutlass/include)}"
: "${OUT_DIR:?set OUT_DIR to the component install prefix}"
NVCC="${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p "$OUT_DIR/lib" "$OUT_DIR/include"
"$NVCC" -O3 -std=c++17 -arch=sm_120a -ftz=false -Xcompiler=-fPIC -DENABLE_BF16 -Iinclude \
  -I"$TILELANG_DIR/src" -I"$TILELANG_DIR/3rdparty/cutlass/include" -shared \
  src/quant.cu src/state.cu src/attention.cu src/index.cu src/topk.cu src/split.cu \
  -Xlinker -soname=libds41_attn.so -o "$OUT_DIR/lib/libds41_attn.so"
cp -r include/. "$OUT_DIR/include/"
