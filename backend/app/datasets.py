"""Dataset builder: accepted traces -> SFT examples in mlx-lm chat format.

Filters by outcome signal, keeps only teacher-labelled steps, strips reasoning (traces are
already in the action format), rejects any example the PII scanner flags, dedupes, and
splits train/valid by trace so steps of one conversation never straddle the split.

Optionally adds the teacher labels from the other two collection paths (labels.py): shadow-sample
disagreements and relabelled out-of-distribution traces. Those examples skip the outcome filter,
since the student's trajectory around them is what went wrong.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from . import db
from .labels import action_json, ood_labels, shadow_labels
from .shared import scanner, short_system_prompt

DATA_DIR = db.ROOT / "data" / "datasets"


def _examples(trace: dict[str, Any], system: str) -> list[list[dict[str, str]]]:
    msgs = trace["messages"] or []
    out = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("served_by") == "teacher":
            convo = [{"role": "system", "content": system}] + [
                {"role": x["role"], "content": x["content"]} for x in msgs[: i + 1]]
            out.append(convo)
    return out


def build(round_: int, run_ids: list[str] | None = None, sources: list[str] | None = None,
          valid_frac: float = 0.1, require_success: bool = True, max_traces: int | None = None,
          include_shadow: bool = False, include_ood: bool = False) -> dict[str, Any]:
    where, args = ["accepted = 1", "messages IS NOT NULL"], []
    if run_ids:
        where.append(f"run_id IN ({','.join('?' * len(run_ids))})")
        args += run_ids
    if sources:
        where.append(f"source IN ({','.join('?' * len(sources))})")
        args += sources
    traces = db.query(f"SELECT * FROM traces WHERE {' AND '.join(where)} ORDER BY scenario_id", args)
    if max_traces:
        traces = traces[:max_traces]
    system = short_system_prompt()
    sc = scanner()
    stats = {"traces": len(traces), "filtered_outcome": 0, "examples": 0, "rejected_pii": 0, "dupes": 0,
             "train": 0, "valid": 0, "by_tool": {}, "by_source": {}}
    seen: set[str] = set()
    train, valid = [], []

    def bucket_for(trace_id: str) -> list:
        return valid if int(hashlib.sha1(trace_id.encode()).hexdigest(), 16) % 1000 < valid_frac * 1000 else train

    candidates: list[tuple[str, str, list[dict[str, str]]]] = []  # (source, trace_id, convo)
    for t in traces:
        if require_success and not (t.get("outcome") or {}).get("task_success"):
            stats["filtered_outcome"] += 1
            continue
        candidates += [("escalation" if t.get("source") != "teacher-datagen" else "datagen", t["id"], c)
                       for c in _examples(t, system)]
    for src, labels in (("shadow", shadow_labels(run_ids) if include_shadow else []),
                        ("ood", ood_labels(run_ids) if include_ood else [])):
        for lab in labels:
            msgs = lab["trace"]["messages"]
            convo = [{"role": "system", "content": system}] + [
                {"role": x["role"], "content": x["content"]} for x in msgs[: lab["index"]]]
            convo.append({"role": "assistant", "content": action_json(lab["action"])})
            candidates.append((src, lab["trace"]["id"], convo))
    for src, trace_id, convo in candidates:
        bucket = bucket_for(trace_id)
        stats["examples"] += 1
        body = "\n".join(m["content"] for m in convo[1:])
        if sc.scan("\n".join(m["content"] for m in convo[1:] if m["role"] == "user")):
            stats["rejected_pii"] += 1
            continue
        h = hashlib.sha1(body.encode()).hexdigest()
        if h in seen:
            stats["dupes"] += 1
            continue
        seen.add(h)
        last = json.loads(convo[-1]["content"])
        key = last.get("tool", "reply")
        stats["by_tool"][key] = stats["by_tool"].get(key, 0) + 1
        stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
        bucket.append({"messages": convo})
    ds_id = f"ds-r{round_}-{int(db.now())}-{uuid.uuid4().hex[:4]}"  # suffix: two builds in one second
    path = DATA_DIR / ds_id
    path.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (path / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["train"], stats["valid"] = len(train), len(valid)
    db.insert("datasets", {"id": ds_id, "round": round_, "created_at": db.now(), "stats": stats, "path": str(path),
                           "sources": {"run_ids": run_ids, "sources": sources,
                                                   "include_shadow": include_shadow, "include_ood": include_ood}})
    return {"id": ds_id, "path": str(path), "stats": stats}
