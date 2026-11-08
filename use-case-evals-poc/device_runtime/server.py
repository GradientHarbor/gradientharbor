"""HTTP server for the Simulator app ("the phone", port 8100).

    uv run uvicorn device_runtime.server:app --host 0.0.0.0 --port 8100
    GH_DEVICE_MODEL=models/ft-e4b-r2   (default)   GH_COLLECTOR_URL=http://localhost:8000 (the cloud side)

Chat turns stream newline-delimited JSON events. A write tool pauses the stream with a
{"type": "confirm"} event; the app answers with POST /v1/device/chat/continue.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from harbor_env.agent import SCANNER
from harbor_env.paths import ROOT

from .live import Device

# Main student: Liquid AI LFM3-1.7B, LoRA merged at 6-bit (needs the mlx-vlm overlay, see .claude/launch.json).
# Gemma 4 E4B: GH_DEVICE_MODEL=models/ft-e4b-r2
MODEL = os.environ.get("GH_DEVICE_MODEL", str(ROOT / "models" / "ft-lfm3-r2"))
DEFAULT_MEMBER = os.environ.get("GH_DEMO_MEMBER", "BM-10200")

device = Device(MODEL, model_name=Path(MODEL).name)
app = FastAPI(title="GradientHarbor device runtime (Simulator)", version="0.1.0")


@app.on_event("startup")
def _load() -> None:
    threading.Thread(target=device.load, daemon=True).start()


def _model_info() -> dict[str, Any]:
    p = Path(MODEL)
    files = sorted(p.glob("*.safetensors"))
    size = sum(f.stat().st_size for f in files)
    h = hashlib.sha256("".join(f"{f.name}:{f.stat().st_size}" for f in files).encode()).hexdigest()[:12]
    rnd = next((c for c in p.name.split("-") if c.startswith("r") and c[1:].isdigit()), "r?")
    base = "Liquid LFM3-1.7B, MLX 6-bit" if "lfm3" in p.name.lower() else "Gemma 4 E4B, MLX 4-bit"
    return {"name": p.name, "round": int(rnd[1:]) if rnd[1:].isdigit() else None, "size_bytes": size, "version": h,
            "base": base}


@app.get("/v1/device/status")
def status() -> dict[str, Any]:
    s = device.settings
    return {"ready": device.student is not None, "error": device.load_error, "model": _model_info(),
            "settings": {"offline": s.offline, "shadow_rate": s.shadow_rate, "sync": s.sync, "backend": s.backend,
                         "device_id": s.device_id},
            "counters": dict(device.counters), "queued": len(device.queue), "runtime": "Mac (MLX) for the iOS Simulator"}


class SettingsIn(BaseModel):
    offline: bool | None = None
    shadow_rate: float | None = None
    sync: bool | None = None
    backend: str | None = None


@app.post("/v1/device/settings")
def settings(body: SettingsIn) -> dict[str, Any]:
    s = device.settings
    was_offline = s.offline
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(s, k, v)
    replayed = device.replay_offline_queue() if was_offline and not s.offline and device.queue else 0
    return {**status(), "replayed": replayed}


@app.get("/v1/device/members")
def members() -> list[dict[str, str]]:
    return device.members()


@app.get("/v1/device/state")
def state(member_id: str = DEFAULT_MEMBER) -> dict[str, Any]:
    try:
        return device.state(member_id)
    except StopIteration:
        raise HTTPException(404, "unknown member")


@app.post("/v1/device/reset")
def reset(body: dict[str, Any] | None = None) -> dict[str, Any]:
    device.reset((body or {}).get("member_id"))
    return {"ok": True}


@app.get("/v1/device/ledger")
def ledger() -> dict[str, Any]:
    rows = device.ledger
    sent = [r for r in rows if r["status"] == "sent"]
    raw = sum(len(SCANNER.scan(r.get("payload") or "")) for r in sent)
    c = device.counters
    return {"rows": rows, "stats": {"on_device": c.get("on_device", 0), "requests": c.get("requests", 0),
                                    "sent": len(sent), "blocked": sum(r["status"] == "blocked" for r in rows),
                                    "queued": len(device.queue), "raw_pii_fields_sent": raw,
                                    "cost_usd": round(sum(r.get("cost_usd") or 0 for r in sent), 4)}}


class ChatIn(BaseModel):
    text: str
    member_id: str = DEFAULT_MEMBER
    conversation_id: str | None = None


class ContinueIn(BaseModel):
    conversation_id: str
    approved: bool


def _drive(conv, gen, first: Any = None, started: bool = False) -> Iterator[bytes]:
    try:
        ev = gen.send(first) if started else next(gen)
        while True:
            ev = {**ev, "conversation_id": conv.id}
            yield (json.dumps(ev, ensure_ascii=False) + "\n").encode()
            if ev["type"] == "confirm":
                conv.pending = gen
                return
            ev = next(gen)
    except StopIteration:
        conv.pending = None
        yield (json.dumps({"type": "done", "conversation_id": conv.id}) + "\n").encode()
    except Exception as e:  # never leave the app hanging
        conv.pending = None
        yield (json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}", "conversation_id": conv.id}) + "\n").encode()


@app.post("/v1/device/chat")
def chat(body: ChatIn) -> StreamingResponse:
    conv = device.conversation(body.conversation_id, body.member_id)
    if conv.pending is not None:
        conv.pending.close()
        conv.pending = None
    return StreamingResponse(_drive(conv, device.run_turn(conv, body.text)), media_type="application/x-ndjson")


@app.post("/v1/device/chat/continue")
def chat_continue(body: ContinueIn) -> StreamingResponse:
    conv = device.convs.get(body.conversation_id)
    if conv is None or conv.pending is None:
        raise HTTPException(409, "nothing is waiting for confirmation")
    gen = conv.pending
    conv.pending = None
    return StreamingResponse(_drive(conv, gen, body.approved, started=True), media_type="application/x-ndjson")


@app.get("/v1/device/conversations/{conv_id}")
def conversation(conv_id: str) -> dict[str, Any]:
    """The redacted conversation exactly as the model saw it (for the 'what the model saw' sheet)."""
    conv = device.convs.get(conv_id)
    if conv is None:
        raise HTTPException(404, "unknown conversation")
    return {"id": conv.id, "messages": [{k: m[k] for k in ("role", "kind", "content", "served_by") if k in m}
                                        for m in conv.history + conv.messages]}
