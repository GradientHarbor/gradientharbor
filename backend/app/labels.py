"""Teacher labels on student-served steps, from the two collection paths that aren't escalations.

- shadow: a random sample of student steps that the teacher also answered in the background. Where the two
  disagree, the teacher's action at that point becomes a training example. Read from the `shadow` table.
- ood: synced traces whose request is far from every training request. The teacher relabels each
  student-served step (DAgger-style: the prefix is the student's own trajectory). Stored in `labels`.

Escalations need neither: their teacher steps are already in the trace (served_by == "teacher").
"""

from __future__ import annotations

import json
from typing import Any, Callable

from . import db
from .ood import OODIndex
from .shared import router_config, scanner


def action_json(action: dict[str, Any]) -> str:
    # Must match harbor_env.prompts.action_json: it's the assistant format the student is trained on.
    return json.dumps(action, ensure_ascii=False, separators=(", ", ": "))


def _action(m: dict[str, Any]) -> dict[str, Any] | None:
    if m.get("action") is not None:
        return m["action"]
    try:
        return json.loads(m["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def find_student_step(messages: list[dict[str, Any]], student_action: dict[str, Any]) -> int | None:
    """Index of the first student-served assistant message carrying this action."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("served_by") == "student" and _action(m) == student_action:
            return i
    return None


def _run_filter(run_ids: list[str] | None, col: str = "t.run_id") -> tuple[str, list[str]]:
    if not run_ids:
        return "", []
    return f" AND {col} IN ({','.join('?' * len(run_ids))})", list(run_ids)


def shadow_labels(run_ids: list[str] | None) -> list[dict[str, Any]]:
    """Shadow disagreements on accepted, non-medical traces -> [{trace, index, action}]."""
    w, args = _run_filter(run_ids)
    rows = db.query("SELECT s.student, s.teacher, t.* FROM shadow s JOIN traces t ON t.id = s.trace_id "
                    "WHERE s.agree = 0 AND t.accepted = 1 AND t.messages IS NOT NULL AND COALESCE(t.never_leave, 0) = 0"
                    + w, args)
    out = []
    for r in rows:
        student, teacher = r.pop("student"), r.pop("teacher")
        if not isinstance(teacher, dict) or not ("tool" in teacher or "reply" in teacher):
            continue  # teacher refused or the row was malformed
        i = find_student_step(r["messages"], student)
        if i is not None:
            out.append({"trace": r, "index": i, "action": teacher})
    return out


def ood_labels(run_ids: list[str] | None) -> list[dict[str, Any]]:
    w, args = _run_filter(run_ids)
    rows = db.query("SELECT l.msg_index, l.action AS label_action, t.* FROM labels l JOIN traces t ON t.id = l.trace_id "
                    "WHERE l.source = 'ood' AND t.accepted = 1 AND t.messages IS NOT NULL" + w, args)
    out = []
    for r in rows:
        i, a = r.pop("msg_index"), r.pop("label_action")
        out.append({"trace": r, "index": i, "action": json.loads(a) if isinstance(a, str) else a})
    return out


# ---- OOD scoring + labelling --------------------------------------------------------------------

_index: dict[str, Any] = {}


def _request_text(request_redacted: str | None) -> str:
    # The request message is "[session] {...}\n\n<member's words>"; score only the member's words.
    return (request_redacted or "").split("\n\n", 1)[-1]


def ood_index() -> tuple[OODIndex, int]:
    n = db.one("SELECT COUNT(*) AS n FROM traces WHERE source = 'teacher-datagen' AND accepted = 1")["n"]
    if _index.get("n") != n:
        texts = [r["request_redacted"] for r in db.query(
            "SELECT request_redacted FROM traces WHERE source = 'teacher-datagen' AND accepted = 1")]
        _index.update(n=n, index=OODIndex([_request_text(t) for t in texts if t]))
    return _index["index"], n


def label_ood(run_ids: list[str] | None, limit: int, dry_run: bool, call_teacher: Callable[..., dict[str, Any]],
              effort: str = "low") -> dict[str, Any]:
    """Score synced traces, then have the teacher relabel every student step of the most distant OOD ones."""
    index, n = ood_index()
    thr = router_config()["ood"]["max_distance"]
    w, args = _run_filter(run_ids)
    rows = db.query("SELECT * FROM traces t WHERE t.accepted = 1 AND t.messages IS NOT NULL "
                    "AND COALESCE(t.never_leave, 0) = 0 AND t.source != 'teacher-datagen'" + w, args)
    scored = sorted(((index.distance(_request_text(r["request_redacted"])), r) for r in rows),
                    key=lambda x: -x[0])
    ood = [(d, r) for d, r in scored if d > thr]
    sc = scanner()
    stats = {"threshold": thr, "index_size": n, "scored": len(rows), "ood": len(ood), "traces_labelled": 0,
             "steps_labelled": 0, "steps_already_labelled": 0, "blocked_by_scan": 0, "teacher_failed": 0,
             "agree": 0, "dry_run": dry_run, "traces": []}
    for d, t in ood[:limit]:
        msgs = t["messages"]
        steps = [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("served_by") == "student"]
        done = {r["msg_index"] for r in db.query(
            "SELECT msg_index FROM labels WHERE trace_id = ? AND source = 'ood'", [t["id"]])}
        item = {"id": t["id"], "run_id": t["run_id"], "distance": round(d, 4), "student_steps": len(steps),
                "labelled": 0}
        stats["traces"].append(item)
        for i in steps:
            if i in done:
                stats["steps_already_labelled"] += 1
                continue
            if dry_run:
                continue
            prefix = msgs[:i]
            outbound = "\n".join(m.get("content", "") for m in prefix if m.get("role") == "user")
            findings = sc.scan(outbound)
            if findings:
                stats["blocked_by_scan"] += 1
                db.insert("cloud_calls", {"trace_id": t["id"], "device_id": t.get("device_id"), "run_id": t["run_id"],
                                          "reason": "ood_label", "payload": None, "sent": False, "findings": findings,
                                          "created_at": db.now()})
                continue
            try:
                out = call_teacher(prefix, effort=effort)
            except Exception:  # noqa: BLE001 - one bad step shouldn't stop the batch
                stats["teacher_failed"] += 1
                continue
            db.insert("cloud_calls", {"trace_id": t["id"], "device_id": t.get("device_id"), "run_id": t["run_id"],
                                      "reason": "ood_label", "payload": outbound, "sent": True, "findings": [],
                                      "model": out.get("model"), "latency_ms": out.get("latency_ms"),
                                      "cost_usd": out.get("cost_usd"), "created_at": db.now()})
            action = out.get("action")
            if not action:
                stats["teacher_failed"] += 1
                continue
            agree = action == _action(msgs[i])
            stats["agree"] += agree
            db.insert("labels", {"trace_id": t["id"], "run_id": t["run_id"], "source": "ood", "msg_index": i,
                                 "action": action, "agree": agree, "distance": d, "model": out.get("model"),
                                 "created_at": db.now()}, replace=True)
            item["labelled"] += 1
            stats["steps_labelled"] += 1
        stats["traces_labelled"] += item["labelled"] > 0
    return stats
