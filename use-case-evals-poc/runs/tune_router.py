"""Tune the router's confidence thresholds on a calibration slice of train (never on held-out).

Runs the fine-tuned student alone over the slice, records the min / mean token logprob of its
tool-call steps and its validator results, then grid-searches thresholds to maximize expected
hybrid success (kept requests scored as the student did, escalated ones at the teacher's rate)
minus a price per escalation.

    uv run python runs/tune_router.py --model models/ft-e4b-r1 --from 400 --to 500 [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor_env.agent import read_jsonl, run_episode, write_jsonl  # noqa: E402
from harbor_env.paths import ROOT, RUNS, SHARED  # noqa: E402
from harbor_env.router import Router  # noqa: E402
from harbor_env.scoring import score_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--from", dest="lo", type=int, default=400)
    ap.add_argument("--to", dest="hi", type=int, default=500)
    ap.add_argument("--teacher-success", type=float, default=0.975, help="teacher task success on train (datagen run)")
    ap.add_argument("--escalation-cost", type=float, default=0.25,
                    help="penalty per escalated request, in units of task success (latency, cost, data leaving)")
    ap.add_argument("--write", action="store_true", help="write the chosen thresholds to shared/router.json")
    args = ap.parse_args()

    from harbor_env.models import MLXBackend

    scen = read_jsonl(ROOT / "scenarios" / "splits" / "train.jsonl")[args.lo: args.hi]
    st = MLXBackend(args.model, adapter=args.adapter, prompt="short", name="ft")
    router = Router(use_confidence=False)  # record gates, don't act on them
    rows = []
    for i, sc in enumerate(scen, 1):
        ep = run_episode(sc, student=st, router=router, mode="single")
        gates = [s["gate"] for s in ep["steps"] if s.get("gate") and s.get("action") and "tool" in s["action"]]
        vgates = [s["gate"] for s in ep["steps"] if s.get("gate")]
        rows.append({"id": sc["id"], "success": score_episode(sc, ep)["task_success"],
                     "validator_fail": any(g["reasons"] for g in vgates),
                     "min_lp": min((g["min_logprob"] for g in gates if g["min_logprob"] is not None), default=0.0),
                     "mean_lp": min((g["mean_logprob"] for g in gates if g["mean_logprob"] is not None), default=0.0)})
        if i % 20 == 0:
            print(f"  {i}/{len(scen)}", file=sys.stderr)
    out = RUNS / "router-calibration.jsonl"
    write_jsonl(out, rows)

    base_succ = sum(r["success"] for r in rows) / len(rows)
    best = None
    table = []
    for a in [-12, -10, -8, -6, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5]:
        for b in [-1.0, -0.6, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, -0.02]:
            kept = [r for r in rows if not (r["validator_fail"] or r["min_lp"] < a or r["mean_lp"] < b)]
            esc = 1 - len(kept) / len(rows)
            loc = sum(r["success"] for r in kept) / len(kept) if kept else 0.0
            table.append((a, b, esc, loc))
            # Expected hybrid success minus a price for every request that leaves the device.
            util = (sum(r["success"] for r in kept) + args.teacher_success * (len(rows) - len(kept))) / len(rows) \
                - args.escalation_cost * esc
            if best is None or util > best[4]:
                best = (a, b, esc, loc, util)
    print(f"student alone: task success {base_succ:.3f} on {len(rows)} calibration scenarios")
    print(f"validators alone: escalate {sum(r['validator_fail'] for r in rows) / len(rows):.3f}")
    print(f"chosen: min_token_logprob={best[0]} mean_token_logprob={best[1]} -> escalation {best[2]:.3f}, "
          f"local success {best[3]:.3f}, utility {best[4]:.3f}")
    if args.write:
        p = SHARED / "router.json"
        cfg = json.loads(p.read_text())
        cfg["confidence"].update(min_token_logprob=best[0], mean_token_logprob=best[1],
                                 tuned_on=f"train[{args.lo}:{args.hi}] with {Path(args.model).name}",
                                 objective=f"max expected success - {args.escalation_cost} x escalation rate")
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
