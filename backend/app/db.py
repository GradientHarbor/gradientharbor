"""SQLite storage (WAL). One file: backend/data/harbor.db (override with GH_DB)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("GH_DB", ROOT / "data" / "harbor.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  id TEXT PRIMARY KEY, name TEXT, platform TEXT, created_at REAL, last_seen REAL
);
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY, device_id TEXT, task TEXT, source TEXT, run_id TEXT, scenario_id TEXT, category TEXT,
  round INTEGER, served_by TEXT, escalated INTEGER, escalation_reasons TEXT, never_leave INTEGER,
  request_redacted TEXT, messages TEXT, final_reply_redacted TEXT, outcome TEXT, latency_ms REAL, steps INTEGER,
  model TEXT, raw TEXT, accepted INTEGER, reject_reason TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS traces_source ON traces(source, round);
CREATE TABLE IF NOT EXISTS cloud_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, device_id TEXT, run_id TEXT, reason TEXT, payload TEXT,
  sent INTEGER, findings TEXT, model TEXT, latency_ms REAL, cost_usd REAL, created_at REAL
);
CREATE TABLE IF NOT EXISTS shadow (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, run_id TEXT, agree INTEGER, student TEXT, teacher TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, run_id TEXT, source TEXT, msg_index INTEGER, action TEXT,
  agree INTEGER, distance REAL, model TEXT, created_at REAL, UNIQUE(trace_id, source, msg_index)
);
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, round INTEGER, created_at REAL, stats TEXT, path TEXT, sources TEXT
);
CREATE TABLE IF NOT EXISTS rounds (
  round INTEGER PRIMARY KEY, status TEXT, dataset_id TEXT, model_id TEXT, started_at REAL, finished_at REAL,
  notes TEXT, loss_log TEXT
);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, round INTEGER, base TEXT, adapter_path TEXT, fused_path TEXT, format TEXT,
  size_bytes INTEGER, sha256 TEXT, url TEXT, created_at REAL, metrics TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY, name TEXT, kind TEXT, model TEXT, split TEXT, round INTEGER, created_at REAL,
  summary TEXT, config TEXT
);
CREATE TABLE IF NOT EXISTS eval_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, scenario_id TEXT, category TEXT, split TEXT,
  score TEXT, judge TEXT, trace_id TEXT
);
CREATE INDEX IF NOT EXISTS eval_results_run ON eval_results(run_id);
"""

JSON_COLS = {"action", "escalation_reasons", "messages", "outcome", "raw", "findings", "student", "teacher", "stats", "sources",
             "loss_log", "metrics", "summary", "config", "score", "judge"}

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.executescript(SCHEMA)
        _conn = c
    return _conn


def reset_for_tests(path: Path) -> None:
    global _conn, DB_PATH
    _conn = None
    DB_PATH = path


def _enc(k: str, v: Any) -> Any:
    if k in JSON_COLS and v is not None and not isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return int(v)
    return v


def row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        if isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


def insert(table: str, data: dict[str, Any], replace: bool = False) -> int:
    cols = list(data)
    sql = f"INSERT {'OR REPLACE ' if replace else ''}INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    with _lock:
        cur = conn().execute(sql, [_enc(k, data[k]) for k in cols])
        return cur.lastrowid


def insert_many(table: str, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    cols = list(rows[0])
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    with _lock:
        c = conn()
        c.execute("BEGIN")
        c.executemany(sql, [[_enc(k, r.get(k)) for k in cols] for r in rows])
        c.execute("COMMIT")


def query(sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with _lock:
        return [row(r) for r in conn().execute(sql, list(args)).fetchall()]


def one(sql: str, args: Iterable[Any] = ()) -> dict[str, Any] | None:
    rows = query(sql, args)
    return rows[0] if rows else None


def execute(sql: str, args: Iterable[Any] = ()) -> None:
    with _lock:
        conn().execute(sql, list(args))


def now() -> float:
    return time.time()
