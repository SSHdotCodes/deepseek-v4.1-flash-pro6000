#!/usr/bin/env python3
"""Check the engine's image path against DeepSeek's reference code from the checkpoint.

1. Preprocessing: ds41_api.vision.load_image (the API's PIL resize/pad) against inference/image_processor.py's
   load_image: same grids and, after the engine's normalization (x/255, (x-0.5)/0.5, BF16), identical patches.
2. Vision tower + aligner: build/engine/vision_probe (the engine's CUDA encoder) against inference/vision.py's
   ViT and Aligner in PyTorch on the GPU. The reference runs twice, with the flash and the math attention
   backends, to show how much two correct BF16 implementations differ.

Run with the assets environment (torch + pillow), on a free GPU:
  .venv-assets/bin/python scripts/vision_check.py --weights models/<revision> --probe build/engine/vision_probe \\
      --out results/vision-check.json
"""
import argparse
import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from ds41_api import vision as api_vision  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_tensors(weights: Path, names):
    """BF16 tensors by name, read straight from the safetensors shards."""
    index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]
    out, headers = {}, {}
    for name in names:
        shard = weights / index[name]
        if shard not in headers:
            with open(shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                headers[shard] = (json.loads(f.read(n)), 8 + n)
        header, base = headers[shard]
        meta = header[name]
        assert meta["dtype"] == "BF16", (name, meta["dtype"])
        begin, end = meta["data_offsets"]
        with open(shard, "rb") as f:
            f.seek(base + begin)
            raw = bytearray(f.read(end - begin))
        out[name] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(meta["shape"]).clone()
    return out


def test_images(weights: Path):
    """The official example images plus synthetic pictures for the resize paths: upscaling, padding, the
    token-budget shrink and its single-row / single-column fallbacks (the checkpoint sets no max_wh_ratio)."""
    images = {}
    for path in sorted((weights / "inference" / "examples").rglob("*")):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            images[path.name] = path.read_bytes()
    rng = np.random.default_rng(0)
    for name, (w, h) in {
        "small-100x60": (100, 60),          # below min_pixels: upscaled
        "square-544": (544, 544),
        "photo-640x480": (640, 480),
        "odd-1023x767": (1023, 767),
        "large-4032x3024": (4032, 3024),    # over 1024 tokens: shrunk by solve_resize_ratio
        "wide-3000x200": (3000, 200),
        "tall-200x3000": (200, 3000),
        "extreme-wide-60000x20": (60000, 20),   # solve_resize_ratio: collapses to one token row
        "extreme-tall-20x60000": (20, 60000),   # solve_resize_ratio: collapses to one token column
    }.items():
        y, x = np.mgrid[0:h, 0:w]
        base = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256], -1)
        noise = rng.integers(0, 48, size=(h, w, 3))
        array = np.clip(base + noise, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        images[name] = buffer.getvalue()
    return images


def engine_patches(pixels: bytes, n_vit_h: int, n_vit_w: int, p: int) -> torch.Tensor:
    """The engine's patchify kernel in PyTorch: float32 x/255, (x-0.5)/0.5, BF16, features (c, py, px)."""
    x = torch.frombuffer(bytearray(pixels), dtype=torch.uint8).reshape(n_vit_h * p, n_vit_w * p, 3)
    x = x.to(torch.float32).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    return x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)


def compare(a: torch.Tensor, b: torch.Tensor):
    a, b = a.float(), b.float()
    diff = (a - b)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {
        "relative_error": float(diff.norm() / b.norm()),
        "max_abs_diff": float(diff.abs().max()),
        "mean_abs_diff": float(diff.abs().mean()),
        "min_row_cosine": float(cos.min()),
        "mean_row_cosine": float(cos.mean()),
        "reference_rms": float(b.pow(2).mean().sqrt()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    weights = args.weights.resolve()
    config = json.loads((weights / "inference" / "config.json").read_text())
    margs = SimpleNamespace(**config)
    cfg = api_vision.VisionConfig.from_weights(weights)
    processor = load_module("ds41_ref_image_processor", weights / "inference" / "image_processor.py")
    vision = load_module("ds41_ref_vision", weights / "inference" / "vision.py")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    vit, aligner = vision.ViT(margs).eval(), vision.Aligner(margs).eval()
    names = [f"vision.{k}" for k in vit.state_dict()] + [f"aligner.{k}" for k in aligner.state_dict()]
    tensors = read_tensors(weights, names)
    vit.load_state_dict({k[len("vision."):]: v for k, v in tensors.items() if k.startswith("vision.")})
    aligner.load_state_dict({k[len("aligner."):]: v for k, v in tensors.items() if k.startswith("aligner.")})

    rows, manifest = [], []
    work = Path(tempfile.mkdtemp(prefix="ds41-vision-"))
    for name, data in test_images(weights).items():
        ref_patches, rh, rw, rlh, rlw = processor.load_image({"data": data}, margs)
        pixels, nh, nw, lh, lw = api_vision.load_image(data, cfg)
        grids_equal = (rh, rw, rlh, rlw) == (nh, nw, lh, lw)
        patches = engine_patches(pixels, nh, nw, cfg.patch_size)
        patches_equal = grids_equal and torch.equal(patches.cpu(), ref_patches.cpu())
        (work / f"{name}.rgb").write_bytes(pixels)
        manifest.append({"pixels": str(work / f"{name}.rgb"), "n_vit_h": nh, "n_vit_w": nw,
                         "out": str(work / f"{name}.bf16")})
        rows.append({"image": name, "patch_grid": [nh, nw], "token_grid": [lh, lw],
                     "span_tokens": api_vision.num_image_tokens(lh, lw),
                     "grids_equal": grids_equal, "patches_equal": patches_equal})
    (work / "manifest.json").write_text(json.dumps(manifest))
    probe = subprocess.run([str(args.probe), str(weights), str(work / "manifest.json")],
                           capture_output=True, text=True)
    if probe.returncode:
        raise SystemExit(f"vision_probe failed: {probe.stderr.strip()}")
    timings = {}
    for line in probe.stdout.splitlines():
        out, *fields = line.split()
        timings[out] = dict(field.split("=") for field in fields)

    for row, item in zip(rows, manifest):
        nh, nw = row["patch_grid"]
        lh, lw = row["token_grid"]
        x = engine_patches(Path(item["pixels"]).read_bytes(), nh, nw, cfg.patch_size).cuda()
        refs = {}
        with torch.inference_mode():
            for backend in ("flash", "math"):
                kind = {"flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        "math": torch.nn.attention.SDPBackend.MATH}[backend]
                with torch.nn.attention.sdpa_kernel(kind):
                    refs[backend] = aligner(vit(x, nh, nw), nh, nw).float().cpu()
        engine = torch.frombuffer(bytearray(Path(item["out"]).read_bytes()), dtype=torch.bfloat16)
        engine = engine.reshape(lh * lw, 5120).float()
        row["engine_vs_flash"] = compare(engine, refs["flash"])
        row["engine_vs_math"] = compare(engine, refs["math"])
        row["math_vs_flash"] = compare(refs["math"], refs["flash"])
        row["engine_ms"] = float(timings[item["out"]]["ms"])
        print(json.dumps(row), flush=True)

    worst = max(r["engine_vs_flash"]["relative_error"] for r in rows)
    floor = max(r["math_vs_flash"]["relative_error"] for r in rows)
    summary = {
        "preprocessing_identical": all(r["patches_equal"] for r in rows),
        "worst_engine_relative_error": worst,
        "worst_reference_backend_relative_error": floor,
        "min_engine_row_cosine": min(r["engine_vs_flash"]["min_row_cosine"] for r in rows),
    }
    # Pass: identical patches, and the encoder no further from the reference than twice the reference's own
    # flash-vs-math spread (finite values only: NaN fails every comparison).
    summary["passed"] = bool(summary["preprocessing_identical"] and worst <= 2 * floor)
    print(json.dumps(summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": "ds41-vision-check-v1", "summary": summary, "images": rows},
                                       indent=1) + "\n")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
