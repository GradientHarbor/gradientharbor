"""GradientHarbor collector API.

    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

Clients: the eval harness (laptop), the iOS app (HarborKit, over the LAN), the SDK wrap(),
and the dashboard (frontend/). OpenAPI at /openapi.json for generating the Swift client.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import datasets, db, labels
from .shared import SHARED, router_config, scanner

ENV_FILE = db.ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API", ""))

app = FastAPI(title="GradientHarbor collector", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---- models -----------------------------------------------------------------------------------

class TraceIn(BaseModel):
    id: str | None = None
    task: str = "support-agent"
    source: str = "phone"
    run_id: str | None = None
    scenario_id: str | None = None
    category: str | None = None
    round: int | None = None
    served_by: str | None = None
    escalated: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)
    never_leave: bool = False
    request_redacted: str | None = None
    messages: list[dict[str, Any]] | None = None
    final_reply_redacted: str | None = None
    outcome: dict[str, Any] | None = None
    latency_ms: float | None = None
    steps: int | None = None
    model: str | None = None
    cloud_calls: list[dict[str, Any]] = Field(default_factory=list)
    shadow: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] | None = None  # SDK wrap() traces: request/response/usage

    model_config = {"extra": "allow"}


class TraceBatch(BaseModel):
    device_id: str = "unknown"
    traces: list[dict[str, Any]]


class TeacherIn(BaseModel):
    device_id: str = "unknown"
    trace_id: str | None = None
    reason: str = "escalation"
    messages: list[dict[str, Any]]
    effort: str = "low"


class DatasetIn(BaseModel):
    round: int
    run_ids: list[str] | None = None
    sources: list[str] | None = None
    valid_frac: float = 0.1
    require_success: bool = True
    max_traces: int | None = None
    include_shadow: bool = False  # add shadow-sample disagreements (teacher's action at that step)
    include_ood: bool = False  # add teacher relabels of out-of-distribution traces (POST /v1/ood/label first)


class OODLabelIn(BaseModel):
    run_ids: list[str] | None = None
    limit: int = 20  # most distant OOD traces to relabel
    dry_run: bool = False  # score and list only; no teacher calls
    effort: str = "low"


class ModelIn(BaseModel):
    id: str
    round: int
    base: str
    adapter_path: str | None = None
    fused_path: str | None = None
    format: str = "mlx-4bit"
    url: str | None = None
    metrics: dict[str, Any] | None = None


class RoundIn(BaseModel):
    round: int
    status: str = "planned"
    dataset_id: str | None = None
    model_id: str | None = None
    notes: str | None = None
    loss_log: list[dict[str, Any]] | None = None


class EvalRunIn(BaseModel):
    id: str | None = None
    name: str
    kind: str
    model: str
    split: str
    round: int | None = None
    summary: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)


# ---- traces + ledger --------------------------------------------------------------------------

def _trace_text(t: dict[str, Any]) -> str:
    # Scan what came from the device: the request, tool results and SDK request bodies. Model outputs
    # (assistant turns, final reply) were produced from redacted inputs and legitimately contain plan
    # terms like "$40 copay", so they are not scanned.
    parts = [t.get("request_redacted") or ""]
    for m in t.get("messages") or []:
        if m.get("role") == "user":
            parts.append(m.get("content", ""))
    if t.get("raw") or t.get("request"):
        parts.append(json.dumps(t.get("raw") or t.get("request")))
    return "\n".join(parts)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "db": str(db.DB_PATH), "shared": str(SHARED)}


@app.post("/v1/traces")
def post_traces(batch: TraceBatch) -> dict[str, Any]:
    sc = scanner()
    db.insert("devices", {"id": batch.device_id, "name": batch.device_id, "platform": None,
                          "created_at": db.now(), "last_seen": db.now()}, replace=True)
    accepted, rejected, results = 0, 0, []
    for raw in batch.traces:
        t = TraceIn(**raw).model_dump()
        tid = t["id"] or "tr_" + uuid.uuid4().hex[:20]
        findings = sc.scan(_trace_text(raw))
        ok = not findings
        db.insert("traces", {
            "id": tid, "device_id": batch.device_id, "task": t["task"], "source": t["source"], "run_id": t["run_id"],
            "scenario_id": t["scenario_id"], "category": t["category"], "round": t["round"],
            "served_by": t["served_by"], "escalated": t["escalated"], "escalation_reasons": t["escalation_reasons"],
            "never_leave": t["never_leave"], "request_redacted": t["request_redacted"] if ok else None,
            "messages": t["messages"] if ok else None, "final_reply_redacted": t["final_reply_redacted"] if ok else None,
            "outcome": t["outcome"], "latency_ms": t["latency_ms"], "steps": t["steps"], "model": t["model"],
            "raw": ({k: raw.get(k) for k in ("request", "response", "usage", "provider")} if raw.get("request") else None)
            if ok else None,
            "accepted": ok, "reject_reason": None if ok else "pii_scan:" + ",".join(sorted({f["type"] for f in findings})),
            "created_at": db.now()}, replace=True)
        for c in t["cloud_calls"]:
            db.insert("cloud_calls", {"trace_id": tid, "device_id": batch.device_id, "run_id": t["run_id"],
                                      "reason": c.get("reason"), "payload": c.get("payload"), "sent": c.get("sent", True),
                                      "findings": c.get("blocked_by_scan") or sc.scan(c.get("payload") or ""),
                                      "model": c.get("model"), "latency_ms": (c.get("latency_s") or 0) * 1000,
                                      "cost_usd": c.get("cost_usd"), "created_at": db.now()})
        for s in t["shadow"]:
            db.insert("shadow", {"trace_id": tid, "run_id": t["run_id"], "agree": s.get("agree"),
                                 "student": s.get("student"), "teacher": s.get("teacher"), "created_at": db.now()})
        accepted += ok
        rejected += not ok
        results.append({"id": tid, "accepted": ok, "findings": [f["type"] for f in findings]})
    return {"accepted": accepted, "rejected": rejected, "results": results}


@app.post("/v1/traces/{trace_id}/feedback")
def trace_feedback(trace_id: str, body: dict[str, Any]) -> dict[str, Any]:
    t = db.one("SELECT outcome FROM traces WHERE id = ?", [trace_id])
    if t is None:
        raise HTTPException(404, "trace not found")
    outcome = t.get("outcome") or {}
    outcome.update({"feedback_ok": bool(body.get("ok")), **{k: v for k, v in body.items() if k != "ok"}})
    db.execute("UPDATE traces SET outcome = ? WHERE id = ?", [json.dumps(outcome), trace_id])
    return {"ok": True}


@app.get("/v1/traces")
def list_traces(source: str | None = None, run_id: str | None = None, served_by: str | None = None,
                escalated: bool | None = None, accepted: bool | None = None, limit: int = Query(100, le=1000),
                offset: int = 0) -> dict[str, Any]:
    where, args = [], []
    for col, val in (("source", source), ("run_id", run_id), ("served_by", served_by)):
        if val:
            where.append(f"{col} = ?")
            args.append(val)
    if escalated is not None:
        where.append("escalated = ?")
        args.append(int(escalated))
    if accepted is not None:
        where.append("accepted = ?")
        args.append(int(accepted))
    w = ("WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(f"SELECT * FROM traces {w} ORDER BY created_at DESC LIMIT ? OFFSET ?", args + [limit, offset])
    total = db.one(f"SELECT COUNT(*) AS n FROM traces {w}", args)["n"]
    return {"total": total, "traces": rows}


@app.get("/v1/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    t = db.one("SELECT * FROM traces WHERE id = ?", [trace_id])
    if t is None:
        raise HTTPException(404, "trace not found")
    t["cloud_calls"] = db.query("SELECT * FROM cloud_calls WHERE trace_id = ?", [trace_id])
    return t


@app.get("/v1/ledger")
def ledger(run_id: str | None = None, limit: int = Query(200, le=2000)) -> dict[str, Any]:
    w, args = ("WHERE run_id = ?", [run_id]) if run_id else ("", [])
    rows = db.query(f"SELECT * FROM cloud_calls {w} ORDER BY id DESC LIMIT ?", args + [limit])
    sc = scanner()
    for r in rows:
        r["raw_pii_found"] = len(sc.scan(r.get("payload") or "")) if r.get("sent") else 0
    stats = db.one(f"SELECT COUNT(*) AS calls, SUM(sent) AS sent, SUM(CASE WHEN sent = 0 THEN 1 ELSE 0 END) AS blocked, "
                   f"SUM(cost_usd) AS cost_usd FROM cloud_calls {w}", args)
    stats["raw_pii_fields_sent"] = sum(r["raw_pii_found"] for r in rows)
    return {"stats": stats, "calls": rows}


# ---- teacher proxy ----------------------------------------------------------------------------

@app.post("/v1/teacher")
def teacher(body: TeacherIn) -> dict[str, Any]:
    from .teacher import call_teacher

    outbound = "\n".join(m.get("content", "") for m in body.messages if m.get("role") == "user")
    findings = scanner().scan(outbound)
    if findings:
        db.insert("cloud_calls", {"trace_id": body.trace_id, "device_id": body.device_id, "reason": body.reason,
                                  "payload": None, "sent": False, "findings": findings, "created_at": db.now()})
        raise HTTPException(422, {"error": "pii_scan_failed", "types": sorted({f["type"] for f in findings})})
    out = call_teacher(body.messages, effort=body.effort)
    db.insert("cloud_calls", {"trace_id": body.trace_id, "device_id": body.device_id, "reason": body.reason,
                              "payload": outbound, "sent": True, "findings": [], "model": out["model"],
                              "latency_ms": out["latency_ms"], "cost_usd": out["cost_usd"], "created_at": db.now()})
    return out


@app.post("/v1/shadow")
def post_shadow(body: dict[str, Any]) -> dict[str, Any]:
    db.insert("shadow", {"trace_id": body.get("trace_id"), "run_id": body.get("run_id"), "agree": body.get("agree"),
                         "student": body.get("student"), "teacher": body.get("teacher"), "created_at": db.now()})
    return {"ok": True}


# ---- datasets, rounds, models -----------------------------------------------------------------

@app.post("/v1/datasets/build")
def build_dataset(body: DatasetIn) -> dict[str, Any]:
    return datasets.build(body.round, body.run_ids, body.sources, body.valid_frac, body.require_success,
                          body.max_traces, body.include_shadow, body.include_ood)


@app.get("/v1/datasets")
def list_datasets() -> list[dict[str, Any]]:
    return db.query("SELECT * FROM datasets ORDER BY created_at DESC")


@app.get("/v1/datasets/{ds_id}/{name}.jsonl")
def download_dataset(ds_id: str, name: str) -> FileResponse:
    ds = db.one("SELECT path FROM datasets WHERE id = ?", [ds_id])
    if ds is None or name not in ("train", "valid"):
        raise HTTPException(404)
    return FileResponse(Path(ds["path"]) / f"{name}.jsonl", media_type="application/x-ndjson")


@app.post("/v1/rounds")
def upsert_round(body: RoundIn) -> dict[str, Any]:
    cur = db.one("SELECT * FROM rounds WHERE round = ?", [body.round]) or {}
    data = {**cur, **{k: v for k, v in body.model_dump().items() if v is not None}}
    data.setdefault("started_at", db.now())
    if body.status in ("trained", "deployed", "failed"):
        data["finished_at"] = db.now()
    db.insert("rounds", data, replace=True)
    return db.one("SELECT * FROM rounds WHERE round = ?", [body.round])


@app.get("/v1/rounds")
def list_rounds() -> list[dict[str, Any]]:
    return db.query("SELECT * FROM rounds ORDER BY round")


def _dir_size_and_hash(path: str | None) -> tuple[int, str | None]:
    if not path or not Path(path).exists():
        return 0, None
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    h = hashlib.sha256()
    size = 0
    for f in files:
        size += f.stat().st_size
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return size, h.hexdigest()


@app.post("/v1/models")
def register_model(body: ModelIn) -> dict[str, Any]:
    size, digest = _dir_size_and_hash(body.fused_path or body.adapter_path)
    db.insert("models", {**body.model_dump(), "size_bytes": size, "sha256": digest, "created_at": db.now()}, replace=True)
    return db.one("SELECT * FROM models WHERE id = ?", [body.id])


@app.get("/v1/models")
def list_models() -> list[dict[str, Any]]:
    return db.query("SELECT * FROM models ORDER BY round DESC, created_at DESC")


@app.get("/v1/models/latest")
def latest_model() -> dict[str, Any]:
    m = db.one("SELECT * FROM models ORDER BY round DESC, created_at DESC LIMIT 1")
    if m is None:
        raise HTTPException(404, "no model registered")
    return {"id": m["id"], "round": m["round"], "format": m["format"], "url": m["url"], "sha256": m["sha256"],
            "size_bytes": m["size_bytes"], "base": m["base"], "router": router_config()}


# ---- evals + metrics --------------------------------------------------------------------------

@app.post("/v1/eval-runs")
def post_eval_run(body: EvalRunIn) -> dict[str, Any]:
    run_id = body.id or f"run-{body.kind}-{int(db.now())}"
    db.execute("DELETE FROM eval_results WHERE run_id = ?", [run_id])
    db.insert("eval_runs", {"id": run_id, "name": body.name, "kind": body.kind, "model": body.model, "split": body.split,
                            "round": body.round, "created_at": db.now(), "summary": body.summary, "config": body.config},
              replace=True)
    db.insert_many("eval_results", [{"run_id": run_id, "scenario_id": r.get("scenario_id"), "category": r.get("category"),
                                     "split": r.get("split"), "score": r.get("score"), "judge": r.get("judge"),
                                     "trace_id": r.get("trace_id")} for r in body.results])
    return {"id": run_id, "results": len(body.results)}


@app.get("/v1/eval-runs")
def list_eval_runs() -> list[dict[str, Any]]:
    return db.query("SELECT * FROM eval_runs ORDER BY created_at DESC")


@app.get("/v1/eval-runs/{run_id}")
def get_eval_run(run_id: str) -> dict[str, Any]:
    run = db.one("SELECT * FROM eval_runs WHERE id = ?", [run_id])
    if run is None:
        raise HTTPException(404)
    run["results"] = db.query("SELECT * FROM eval_results WHERE run_id = ? ORDER BY id", [run_id])
    return run


@app.get("/v1/scenarios/{scenario_id}/episodes")
def scenario_episodes(scenario_id: str, run_ids: str = "") -> list[dict[str, Any]]:
    ids = [x for x in run_ids.split(",") if x]
    w, args = ("AND r.run_id IN (" + ",".join("?" * len(ids)) + ")", ids) if ids else ("", [])
    rows = db.query(f"SELECT r.*, t.messages, t.final_reply_redacted, t.served_by, t.escalation_reasons, t.latency_ms "
                    f"FROM eval_results r LEFT JOIN traces t ON t.id = r.trace_id WHERE r.scenario_id = ? {w}",
                    [scenario_id] + args)
    return rows


@app.get("/v1/metrics/overview")
def overview() -> dict[str, Any]:
    # "Latest" = the highest round, then the most recent run: re-running an older round's eval
    # (round 1 was re-run with the final router) must not replace the newer round on the dashboard.
    runs = db.query("SELECT * FROM eval_runs ORDER BY COALESCE(round, -1), created_at")
    latest: dict[str, dict[str, Any]] = {}
    for r in runs:
        latest[f"{r['kind']}:{r['split']}"] = r
    # Round metrics come from the eval splits only; live-traffic runs are training input, not evals.
    hybrid = [r for r in runs if r["kind"].startswith("hybrid") and not str(r["split"]).startswith("traffic")]
    by_round: dict[int, dict[str, Any]] = {0: {"round": 0, "escalation_rate": 1.0, "left_device": 1.0, "source": "definition"}}
    for r in hybrid:
        s = r["summary"] or {}
        rnd = r["round"] if r["round"] is not None else s.get("round")
        if rnd is None:
            continue
        by_round[int(rnd)] = {"round": int(rnd), "escalation_rate": s.get("escalation_rate"),
                              "left_device": s.get("left_device_rate"), "task_success": s.get("task_success"),
                              "local_accuracy": s.get("local_task_success"), "split": r["split"], "run_id": r["id"]}
    led = ledger(limit=2000)["stats"]
    traces = db.one("SELECT COUNT(*) AS n, SUM(accepted) AS accepted, SUM(CASE WHEN accepted = 0 THEN 1 ELSE 0 END) "
                    "AS rejected FROM traces")
    shadow = db.one("SELECT COUNT(*) AS n, AVG(agree) AS agree, SUM(CASE WHEN agree = 0 THEN 1 ELSE 0 END) "
                    "AS disagree FROM shadow")
    label_counts = db.query("SELECT source, COUNT(*) AS n, AVG(agree) AS agree FROM labels GROUP BY source")
    return {"rounds": sorted(by_round.values(), key=lambda x: x["round"]), "latest_runs": latest, "ledger": led,
            "traces": traces, "shadow": shadow, "labels": label_counts,
            "datasets": db.query("SELECT id, round, stats, created_at FROM datasets ORDER BY created_at"),
            "models": db.query("SELECT id, round, base, format, size_bytes, created_at, metrics FROM models ORDER BY round")}


@app.post("/v1/ood/score")
def ood_score(body: dict[str, Any]) -> dict[str, Any]:
    index, n = labels.ood_index()
    thr = router_config()["ood"]["max_distance"]
    out = []
    for t in body.get("texts", []):
        d = index.distance(t)
        out.append({"distance": round(d, 4), "ood": d > thr})
    return {"threshold": thr, "index_size": n, "scores": out}


@app.post("/v1/ood/label")
def ood_label(body: OODLabelIn) -> dict[str, Any]:
    """Relabel the student steps of out-of-distribution synced traces with the teacher (ledgered, fail closed)."""
    from . import teacher as teacher_mod

    return labels.label_ood(body.run_ids, body.limit, body.dry_run, teacher_mod.call_teacher, body.effort)


# ---- static eval report (use-case-evals-poc/runs/build_report.py) ------------------------------

REPORT = Path(__file__).resolve().parents[2] / "use-case-evals-poc" / "reports" / "eval-report.html"


@app.get("/report", include_in_schema=False)
def eval_report() -> FileResponse:
    if not REPORT.exists():
        raise HTTPException(404, "run use-case-evals-poc/runs/build_report.py first")
    return FileResponse(REPORT, media_type="text/html")


# ---- dashboard (built frontend) ---------------------------------------------------------------

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
