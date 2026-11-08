"""Ask the collector to build an SFT dataset and copy it to train/data/r<round>/.

    uv run python train/build_sft.py --round 1 --run-ids datagen-r0-train --max-traces 250
    uv run python train/build_sft.py --round 2 --run-ids datagen-r0-train,hybrid-r1-traffic
    uv run python train/build_sft.py --round 3 --run-ids datagen-r0-train,<traffic runs> --with-shadow --with-ood

--with-shadow adds shadow-sample disagreements; --with-ood adds teacher relabels of out-of-distribution
traces (run train/label_ood.py first). Never pass eval run ids here: their shadow samples are held-out data.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--run-ids", required=True)
ap.add_argument("--max-traces", type=int)
ap.add_argument("--valid-frac", type=float, default=0.1)
ap.add_argument("--with-shadow", action="store_true", help="add shadow-sample disagreements")
ap.add_argument("--with-ood", action="store_true", help="add teacher relabels of OOD traces")
ap.add_argument("--collector", default=os.environ.get("GH_COLLECTOR_URL", "http://localhost:8000"))
a = ap.parse_args()
if any("eval" in r for r in a.run_ids.split(",")):
    ap.error("eval runs are held-out data; don't train on them")

res = httpx.post(f"{a.collector}/v1/datasets/build", timeout=120, json={
    "round": a.round, "run_ids": a.run_ids.split(","), "max_traces": a.max_traces, "valid_frac": a.valid_frac,
    "include_shadow": a.with_shadow, "include_ood": a.with_ood}).json()
out = HERE / "data" / f"r{a.round}"
out.mkdir(parents=True, exist_ok=True)
for name in ("train", "valid"):
    shutil.copy(Path(res["path"]) / f"{name}.jsonl", out / f"{name}.jsonl")
(out / "dataset.json").write_text(json.dumps(res, indent=2))
httpx.post(f"{a.collector}/v1/rounds", json={"round": a.round, "status": "dataset_ready", "dataset_id": res["id"]})
print(json.dumps(res["stats"], indent=1))
print(f"-> {out}")
