"""Have the teacher relabel out-of-distribution synced traces, so build_sft.py --with-ood can train on them.

    uv run python train/label_ood.py --run-ids hybrid-e4b-r1-traffic --dry-run     # score and list, no cost
    uv run python train/label_ood.py --run-ids hybrid-e4b-r1-traffic --limit 20

OOD = distance to the nearest teacher-datagen request above shared/router.json ood.max_distance. Every
teacher call is PII-scanned (fail closed) and written to the privacy ledger with reason "ood_label".
"""

import argparse
import json
import os

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--run-ids", required=True)
ap.add_argument("--limit", type=int, default=20, help="most distant OOD traces to relabel")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--collector", default=os.environ.get("GH_COLLECTOR_URL", "http://localhost:8000"))
a = ap.parse_args()
if any("eval" in r for r in a.run_ids.split(",")):
    ap.error("eval runs are held-out data; don't label them for training")

res = httpx.post(f"{a.collector}/v1/ood/label", timeout=1800, json={
    "run_ids": a.run_ids.split(","), "limit": a.limit, "dry_run": a.dry_run}).json()
traces = res.pop("traces", [])
print(json.dumps(res, indent=1))
for t in traces[:a.limit]:
    print(f"  {t['id']}  distance {t['distance']:.3f}  student steps {t['student_steps']}  labelled {t['labelled']}")
