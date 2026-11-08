"""Parse an mlx-lm training log and register the fused model + loss curve with the collector."""

import argparse
import json
import os
import re
from pathlib import Path

import httpx

ap = argparse.ArgumentParser()
for f in ("round", "size", "base", "fused", "adapter", "log"):
    ap.add_argument(f"--{f}", required=True)
a = ap.parse_args()
collector = os.environ.get("GH_COLLECTOR_URL", "http://localhost:8000")
loss = []
for line in Path(a.log).read_text(errors="replace").splitlines():
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    # mlx-lm <= 0.31: "Iter 10: Train loss 1.112" / "Iter 50: Val loss 0.9"
    # mlx-lm 0.32 table: "    10    1.112 ▼   24   2.4k" / "     1    val 2.033    37.07s"
    m = re.search(r"Iter (\d+): Val loss ([\d.]+)", line) or re.match(r"^\s*(\d+)\s+val\s+([\d.]+)", line)
    if m:
        loss.append({"iter": int(m.group(1)), "val_loss": float(m.group(2))})
        continue
    m = re.search(r"Iter (\d+): Train loss ([\d.]+)", line) or re.match(r"^\s*(\d+)\s+([\d.]+)\s", line)
    if m:
        loss.append({"iter": int(m.group(1)), "train_loss": float(m.group(2))})
model_id = f"ft-{a.size}-r{a.round}"
try:
    httpx.post(f"{collector}/v1/models", json={"id": model_id, "round": int(a.round), "base": a.base,
                                               "adapter_path": str(Path(a.adapter).resolve()),
                                               "fused_path": str(Path(a.fused).resolve()), "format": "mlx-4bit",
                                               "metrics": {"final_val_loss": next((x["val_loss"] for x in reversed(loss)
                                                                                    if "val_loss" in x), None)}})
    httpx.post(f"{collector}/v1/rounds", json={"round": int(a.round), "status": "trained", "model_id": model_id,
                                               "loss_log": loss})
    print(f"registered {model_id} ({len(loss)} loss points)")
except httpx.HTTPError as e:
    print(f"collector not reachable ({e}); model is at {a.fused}")
