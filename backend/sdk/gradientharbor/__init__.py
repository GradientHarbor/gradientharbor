"""GradientHarbor SDK.

One line around an existing client captures every call as a trace:

    import anthropic, gradientharbor
    client = gradientharbor.wrap(anthropic.Anthropic(), task="support-agent")

    resp = client.messages.create(...)            # unchanged call; a trace is queued
    gradientharbor.last_trace().feedback(ok=True)  # optional outcome signal

Traces are batched to the collector (GH_COLLECTOR_URL, default http://localhost:8000) in a
background thread. A trace that the PII scanner flags is rejected by the collector.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import uuid
from typing import Any

import httpx

__all__ = ["wrap", "TraceClient", "Trace", "last_trace"]

_local = threading.local()


class TraceClient:
    def __init__(self, url: str | None = None, device_id: str | None = None, batch: int = 20, flush_s: float = 2.0):
        self.url = (url or os.environ.get("GH_COLLECTOR_URL", "http://localhost:8000")).rstrip("/")
        self.device_id = device_id or os.environ.get("GH_DEVICE_ID", "laptop-harness")
        self.q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.batch, self.flush_s = batch, flush_s
        self.http = httpx.Client(timeout=10)
        self.sent = self.rejected = self.failed = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        atexit.register(self.close)

    def enqueue(self, trace: dict[str, Any]) -> None:
        self.q.put(trace)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        try:
            r = self.http.post(self.url + path, json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            self.failed += 1
            return None

    def _drain(self) -> list[dict[str, Any]]:
        items = []
        while len(items) < self.batch:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        return items

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.flush_s)
            self.flush()

    def flush(self) -> None:
        while True:
            items = self._drain()
            if not items:
                return
            res = self.post("/v1/traces", {"device_id": self.device_id, "traces": items})
            if res:
                self.sent += res.get("accepted", 0)
                self.rejected += res.get("rejected", 0)

    def close(self) -> None:
        self._stop.set()
        self.flush()


class Trace:
    def __init__(self, client: TraceClient, data: dict[str, Any]):
        self.client, self.data = client, data

    def feedback(self, ok: bool, **extra: Any) -> None:
        self.client.post(f"/v1/traces/{self.data['id']}/feedback", {"ok": ok, **extra})


def last_trace() -> Trace | None:
    return getattr(_local, "trace", None)


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


class _Wrapped:
    def __init__(self, target: Any, task: str, tc: TraceClient, path: tuple[str, ...] = ()):
        self._target, self._task, self._tc, self._path = target, task, tc, path

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = self._path + (name,)
        if callable(attr) and path[-1] == "create" and path[-2] in ("messages", "completions"):
            def create(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                resp = attr(*args, **kwargs)
                usage = _jsonable(getattr(resp, "usage", None))
                trace = {
                    "id": "tr_" + uuid.uuid4().hex[:20],
                    "task": self._task,
                    "source": "sdk",
                    "provider": "anthropic" if path[-2] == "messages" else "openai",
                    "model": kwargs.get("model"),
                    "request": {k: _jsonable(v) for k, v in kwargs.items() if k in ("system", "messages", "tools")},
                    "response": _jsonable(getattr(resp, "content", None) or getattr(resp, "choices", None)),
                    "usage": usage,
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                    "created_at": time.time(),
                }
                self._tc.enqueue(trace)
                _local.trace = Trace(self._tc, trace)
                return resp
            return create
        if hasattr(attr, "__dict__") and not callable(attr) or name in ("messages", "beta", "chat", "completions"):
            return _Wrapped(attr, self._task, self._tc, path)
        return attr


def wrap(client: Any, task: str, collector: TraceClient | None = None) -> Any:
    """Wrap an Anthropic or OpenAI(-compatible) client. Every create() call becomes a trace."""
    return _Wrapped(client, task, collector or TraceClient())
