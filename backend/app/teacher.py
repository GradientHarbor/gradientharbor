"""Teacher proxy: the phone never holds a provider key. It sends the redacted canonical
conversation here; the server scans it again (fail closed), calls Claude, and logs the call
to the privacy ledger."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .shared import prompt, tools

TEACHER_MODEL = os.environ.get("GH_TEACHER_MODEL", "claude-sonnet-5-5")
_client = None


def client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(max_retries=3)
    return _client


def to_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_tool_id = None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            action = m.get("action") or json.loads(m["content"])
            if m.get("native"):
                content = m["native"]
            elif "tool" in action:
                content = [{"type": "tool_use", "id": f"toolu_gh{i:03d}", "name": action["tool"], "input": action["args"]}]
            else:
                content = [{"type": "text", "text": action["reply"]}]
            last_tool_id = next((b["id"] for b in content if b.get("type") == "tool_use"), None)
            out.append({"role": "assistant", "content": content})
        elif m.get("kind") == "tool_result" and last_tool_id:
            body = m["content"].split(">", 1)[1].rsplit("</tool_result>", 1)[0]
            out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": last_tool_id, "content": body}]})
            last_tool_id = None
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


def call_teacher(messages: list[dict[str, Any]], effort: str = "low") -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = client().beta.messages.create(
        model=TEACHER_MODEL,
        max_tokens=4096,
        system=prompt("system_core.txt"),
        tools=[{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in tools()],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=to_native(messages),
        output_config={"effort": effort},
        cache_control={"type": "ephemeral"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    native = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
    tool = next((b for b in resp.content if b.type == "tool_use"), None)
    if resp.stop_reason == "refusal":
        action = None
    elif tool is not None:
        action = {"tool": tool.name, "args": dict(tool.input)}
    else:
        action = {"reply": " ".join(b.text for b in resp.content if b.type == "text").strip()}
    u = resp.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cost = (u.input_tokens * 2.0 + cache_write * 2.5 + cache_read * 0.2 + u.output_tokens * 10.0) / 1e6
    return {"action": action, "native": native, "model": resp.model, "latency_ms": latency_ms, "cost_usd": cost,
            "stop_reason": resp.stop_reason}
