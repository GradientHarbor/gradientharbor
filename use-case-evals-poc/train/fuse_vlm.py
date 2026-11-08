"""Merge a LoRA adapter trained with mlx-vlm into its (4-bit) base and save a standalone model folder.

    uv run --with "mlx-vlm>=0.7.11" python train/fuse_vlm.py --base LiquidAI/LFM3-1.7B-MLX-4bit \
        --adapter models/ft-lfm3-r2-adapter --out models/ft-lfm3-r2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=0,
                    help="re-quantize merged layers at this many bits (e.g. 8). Default 0 keeps the base's bits, "
                         "which rounds away much of a small LoRA delta at 4-bit.")
    args = ap.parse_args()

    base_dir = Path(args.base) if Path(args.base).exists() else Path(snapshot_download(args.base))
    model, _ = load(str(base_dir), adapter_path=args.adapter)
    import mlx.nn as nn
    fused = []
    for name, m in model.named_modules():
        if not (hasattr(m, "fuse") and hasattr(m, "lora_a")):
            continue
        if args.bits:
            lin = m.fuse(dequantize=True)
            gs = getattr(m.linear, "group_size", 64)
            fused.append((name, nn.QuantizedLinear.from_linear(lin, gs, args.bits)))
        else:
            fused.append((name, m.fuse()))
    model.update_modules(tree_unflatten(fused))
    print(f"fused {len(fused)} LoRA layers")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in base_dir.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        shutil.copy(f.resolve(), out / f.name)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
    if args.bits:  # record per-layer bits so the loader builds matching QuantizedLinear layers
        cfg = json.loads((out / "config.json").read_text())
        q = dict(cfg.get("quantization") or {})
        for name, layer in fused:
            q[name] = {"group_size": layer.group_size, "bits": layer.bits, "mode": "affine"}
        cfg["quantization"] = q
        if "quantization_config" in cfg:
            cfg["quantization_config"] = q
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "fused_from.json").write_text(json.dumps({"base": args.base, "adapter": args.adapter}, indent=2) + "\n")
    print(f"wrote {out} ({sum(f.stat().st_size for f in out.iterdir()) / 1e9:.2f} GB, {len(weights)} tensors)")


if __name__ == "__main__":
    main()
