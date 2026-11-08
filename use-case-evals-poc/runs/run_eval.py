"""Run one configuration over one or more splits, score it, judge it, save it, post it to the backend.

    uv run python runs/run_eval.py --run teacher     --split heldout,novel,never_leave,policy
    uv run python runs/run_eval.py --run base-e4b    --split heldout,novel,never_leave,policy
    uv run python runs/run_eval.py --run ft-e4b      --split heldout,novel,never_leave,policy --model models/ft-e4b-r1
    uv run python runs/run_eval.py --run hybrid-e4b  --split heldout,novel,never_leave,policy --model models/ft-e4b-r1 --round 1
    uv run python runs/run_eval.py --run teacher-datagen --split train          # round-0 traces for training
    uv run python runs/run_eval.py --run hybrid-e4b --split traffic --model models/ft-e4b-r1 --round 1 --no-judge  # live traffic

Paid API calls (teacher, judge). Prints a cost estimate first; --max-cost stops a run.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor_env.agent import now_tag, read_jsonl, run_episode, write_jsonl  # noqa: E402
from harbor_env.paths import ROOT, RUNS, load_env  # noqa: E402
from harbor_env.router import Router  # noqa: E402
from harbor_env.scoring import score_episode  # noqa: E402

E4B = "lmstudio-community/gemma-4-E4B-it-MLX-4bit"
E2B = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
LFM3 = "LiquidAI/LFM3-1.7B-MLX-4bit"  # Liquid AI, Nov 2026; needs the mlx-vlm overlay (see harbor_env/models.py VLMBackend)
SPLITS = ROOT / "scenarios" / "splits"
COST_PER_TEACHER_EPISODE = 0.013  # measured on the smoke test (Sonnet 5.5, effort low, 2 steps)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)


def rate(xs: list[Any]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(bool(x) for x in xs) / len(xs), 4) if xs else None


def summarize(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    eps = [r["episode"] for r in rows]
    scores = [r["score"] for r in rows]
    local_lat = [s["latency_s"] for e in eps for s in e["steps"] if s.get("served_by") in ("student", "escalated")
                 and s.get("latency_s") is not None]
    cloud_lat = [s["latency_s"] for e in eps for s in e["steps"] if s.get("served_by") == "teacher"
                 and s.get("latency_s") is not None]
    judged = [r["judge"] for r in rows if r.get("judge")]
    by_cat: dict[str, list[bool]] = {}
    by_split: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["score"]["task_success"])
        by_split.setdefault(r["split"], []).append(r["score"]["task_success"])
    local_rows = [r for r in rows if not r["episode"]["escalated"]]
    nl = [r for r in rows if r["episode"]["never_leave"]]
    pol = [r for r in rows if r["category"] == "policy"]
    s = {
        "n": len(rows),
        "task_success": rate([x["task_success"] for x in scores]),
        "first_call_match": rate([x["first_call_match"] for x in scores]),
        "calls_ok": rate([x["calls_ok"] for x in scores]),
        "state_ok": rate([x["state_ok"] for x in scores]),
        "replied": rate([x["replied"] for x in scores]),
        "judge_correct": rate([j.get("correct") for j in judged]),
        "judge_grounded": rate([j.get("grounded") for j in judged]),
        "judge_score": round(statistics.mean([j["score"] for j in judged if j.get("score")]), 3)
        if any(j.get("score") for j in judged) else None,
        "judged": len(judged),
        "success_and_judged_correct": rate([x["task_success"] and (r.get("judge") or {}).get("correct", True)
                                            for x, r in zip(scores, rows)]),
        "by_category": {k: {"n": len(v), "task_success": rate(v)} for k, v in sorted(by_cat.items())},
        "by_split": {k: {"n": len(v), "task_success": rate(v)} for k, v in sorted(by_split.items())},
        "p50_step_latency_local_s": pct(local_lat, 0.5), "p95_step_latency_local_s": pct(local_lat, 0.95),
        "p50_step_latency_cloud_s": pct(cloud_lat, 0.5), "p95_step_latency_cloud_s": pct(cloud_lat, 0.95),
        "steps_per_episode": round(statistics.mean(len(e["steps"]) for e in eps), 2) if eps else None,
        "mean_prompt_tokens": round(statistics.mean([s.get("prompt_tokens") or 0 for e in eps for s in e["steps"]]), 1)
        if eps else None,
        "cost_usd": round(sum(c.get("cost_usd") or 0 for e in eps for c in e["cloud_calls"]), 4),
        "escalation_rate": rate([e["escalated"] for e in eps]),
        "left_device_rate": rate([e["left_device"] for e in eps]),
        "local_task_success": rate([r["score"]["task_success"] for r in local_rows]),
        "local_n": len(local_rows),
        "pii_fields_sent": sum(e["pii_fields_sent"] for e in eps),
        "blocked_by_scan": sum(1 for e in eps for c in e["cloud_calls"] if not c.get("sent")),
        "never_leave_n": len(nl),
        "never_leave_violations": sum(1 for r in nl if r["episode"]["left_device"] or any(
            c.get("reason") == "shadow" and c.get("sent") for c in r["episode"]["cloud_calls"])),
        "policy_n": len(pol),
        "policy_routed_to_teacher": rate([r["episode"]["escalated"] or r["episode"]["left_device"] for r in pol]) if pol else None,
        "shadow_n": sum(len(e["shadow"]) for e in eps),
        "shadow_agreement": rate([x["agree"] for e in eps for x in e["shadow"]]),
        "escalation_reasons": _count([x.split(":")[0] for e in eps for x in e["escalation_reasons"]]),
    }
    if kind in ("teacher", "teacher-datagen"):
        s["escalation_rate"] = s["left_device_rate"] = 1.0
    return s


def _count(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def build(args: argparse.Namespace):
    from harbor_env.models import TeacherBackend
    from harbor_env.models import local_backend as MLXBackend

    kind = args.run
    if kind in ("teacher", "teacher-datagen"):
        return dict(student=None, teacher=TeacherBackend(), router=None, mode="single"), "claude-sonnet-5-5"
    base = LFM3 if "lfm3" in kind else E2B if "e2b" in kind else E4B
    model = args.model or base
    if kind.startswith("base"):
        st = MLXBackend(model, prompt="full", few_shot=True, name=kind)
        return dict(student=st, teacher=None, router=None, mode="single"), model
    if kind.startswith("ft"):
        st = MLXBackend(model, adapter=args.adapter, prompt="short", name=kind)
        return dict(student=st, teacher=None, router=None, mode="single"), model
    if kind.startswith("hybrid"):
        st = MLXBackend(model, adapter=args.adapter, prompt="short", name=kind.replace("hybrid", "ft"))
        router = Router(min_logprob=args.min_logprob, mean_logprob=args.mean_logprob,
                        use_confidence=not args.no_confidence)
        return dict(student=st, teacher=TeacherBackend(), router=router, mode="hybrid"), model
    raise SystemExit(f"unknown run {kind}")


def trace_payload(run_id: str, kind: str, rnd: int | None, sc: dict[str, Any], ep: dict[str, Any],
                  score: dict[str, Any]) -> dict[str, Any]:
    served = {s.get("served_by") for s in ep["steps"]} - {None, "escalated"}
    final = next((m["action"].get("reply") for m in reversed(ep["messages"]) if m["role"] == "assistant"
                  and "reply" in m["action"]), None)
    return {
        "id": f"{run_id}:{sc['id']}", "task": "support-agent",
        "source": "teacher-datagen" if kind == "teacher-datagen" else ("traffic" if sc["split"] == "traffic" else "eval"),
        "run_id": run_id, "scenario_id": sc["id"], "category": sc["category"], "round": rnd,
        "served_by": served.pop() if len(served) == 1 else "mixed", "escalated": ep["escalated"],
        "escalation_reasons": ep["escalation_reasons"], "never_leave": ep["never_leave"],
        "request_redacted": ep["messages"][0]["content"], "messages": ep["messages"], "final_reply_redacted": final,
        "outcome": score, "latency_ms": round(sum(s.get("latency_s") or 0 for s in ep["steps"]) * 1000),
        "steps": len(ep["steps"]), "model": kind,
        "cloud_calls": [{k: v for k, v in c.items()} for c in ep["cloud_calls"]], "shadow": ep["shadow"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", help="mlx model path or HF id (fused fine-tuned model for ft/hybrid)")
    ap.add_argument("--adapter", help="LoRA adapter dir (instead of a fused model)")
    ap.add_argument("--round", type=int)
    ap.add_argument("--name")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--shadow-rate", type=float, default=0.0)
    ap.add_argument("--min-logprob", type=float)
    ap.add_argument("--mean-logprob", type=float)
    ap.add_argument("--no-confidence", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="parallel episodes for teacher-only runs")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--collector", default=os.environ.get("GH_COLLECTOR_URL", "http://localhost:8000"))
    args = ap.parse_args()
    load_env()

    scenarios = []
    for sp in args.split.split(","):
        rows = read_jsonl(SPLITS / f"{sp}.jsonl")
        scenarios += rows[: args.limit] if args.limit else rows
    kind = args.run
    teacherish = kind.startswith(("teacher", "hybrid"))
    est = len(scenarios) * COST_PER_TEACHER_EPISODE * (1.0 if kind.startswith("teacher") else 0.3) if teacherish else 0
    use_judge = not args.no_judge and kind != "teacher-datagen"
    print(f"{kind}: {len(scenarios)} scenarios from {args.split}; est. teacher cost ${est:.2f}"
          f"{'; + judge' if use_judge else ''}", file=sys.stderr)

    cfg, model_id = build(args)
    judge = None
    calib = None
    if use_judge:
        from judge.judge import Judge, calibrate

        judge = Judge()
        calib = calibrate(judge)
        print(f"judge calibration: {calib['cases'] - len(calib['misses'])}/{calib['cases']} ok", file=sys.stderr)
        if not calib["ok"]:
            print(json.dumps(calib["misses"], indent=1)[:2000], file=sys.stderr)
            if len(calib["misses"]) > 1:
                raise SystemExit("judge calibration failed; fix the rubric or rerun with --no-judge")

    run_id = args.name or f"{kind}-{now_tag()}"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)
    spent = 0.0
    rows: list[dict[str, Any]] = []

    def one(sc: dict[str, Any]) -> dict[str, Any]:
        ep = run_episode(sc, **cfg, shadow_rate=args.shadow_rate, rng=random.Random(hash(sc["id"]) & 0xFFFF))
        score = score_episode(sc, ep)
        j = None
        if judge and sc["split"] != "traffic":
            j = judge.grade(sc["message"], ep["calls"], ep["final_reply"], sc["expected"].get("reply_facts"),
                            ep.get("session"))
        return {"scenario_id": sc["id"], "split": sc["split"], "category": sc["category"], "message": sc["message"],
                "score": score, "judge": j, "episode": ep}

    t0 = time.time()
    parallel = kind.startswith("teacher")
    if parallel:
        with cf.ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(one, sc): sc for sc in scenarios}
            for i, f in enumerate(cf.as_completed(futs), 1):
                rows.append(f.result())
                spent += sum(c.get("cost_usd") or 0 for c in rows[-1]["episode"]["cloud_calls"])
                if i % 25 == 0:
                    print(f"  {i}/{len(scenarios)} ${spent:.2f} {time.time() - t0:.0f}s", file=sys.stderr)
                if spent > args.max_cost:
                    ex.shutdown(cancel_futures=True)
                    print("max cost reached; stopping", file=sys.stderr)
                    break
    else:
        for i, sc in enumerate(scenarios, 1):
            rows.append(one(sc))
            spent += sum(c.get("cost_usd") or 0 for c in rows[-1]["episode"]["cloud_calls"])
            if i % 10 == 0:
                ok = sum(r["score"]["task_success"] for r in rows)
                print(f"  {i}/{len(scenarios)} success {ok}/{i} ${spent:.2f} {time.time() - t0:.0f}s", file=sys.stderr)
            if spent > args.max_cost:
                print("max cost reached; stopping", file=sys.stderr)
                break
    order = {sc["id"]: i for i, sc in enumerate(scenarios)}
    rows.sort(key=lambda r: order[r["scenario_id"]])

    summary = summarize(rows, kind)
    summary.update(round=args.round, wall_s=round(time.time() - t0, 1), judge_model=judge.model if judge else None,
                   judge_calibration=calib and {"cases": calib["cases"], "misses": len(calib["misses"])})
    config = {"run": kind, "model": model_id, "adapter": args.adapter, "split": args.split, "limit": args.limit,
              "shadow_rate": args.shadow_rate,
              "router": vars(cfg["router"]) if cfg["router"] else None}
    write_jsonl(out_dir / "episodes.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps({"run_id": run_id, "config": config, "summary": summary}, indent=2,
                                                     default=str))
    for f in out_dir.iterdir():
        f.chmod(0o600)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))
    print(f"saved {out_dir}", file=sys.stderr)

    if not args.no_post:
        import httpx

        try:
            with httpx.Client(base_url=args.collector, timeout=60) as http:
                traces = [trace_payload(run_id, kind, args.round, next(s for s in scenarios if s["id"] == r["scenario_id"]),
                                        r["episode"], r["score"]) for r in rows]
                acc = rej = 0
                for i in range(0, len(traces), 50):
                    res = http.post("/v1/traces", json={"device_id": "laptop-harness", "traces": traces[i:i + 50]}).json()
                    acc, rej = acc + res["accepted"], rej + res["rejected"]
                http.post("/v1/eval-runs", json={
                    "id": run_id, "name": run_id, "kind": kind, "model": model_id, "split": args.split, "round": args.round,
                    "summary": summary, "config": config,
                    "results": [{"scenario_id": r["scenario_id"], "category": r["category"], "split": r["split"],
                                 "score": r["score"], "judge": r["judge"], "trace_id": f"{run_id}:{r['scenario_id']}"}
                                for r in rows]}).raise_for_status()
                print(f"posted to {args.collector}: traces accepted={acc} rejected={rej}", file=sys.stderr)
        except httpx.HTTPError as e:
            print(f"collector not reachable ({type(e).__name__}); results are saved locally", file=sys.stderr)


if __name__ == "__main__":
    main()
