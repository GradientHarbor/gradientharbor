"""System prompts and message rendering shared by every backend (and mirrored in HarborKit)."""

from __future__ import annotations

import json
import re
from typing import Any

from .env import compact, load_tools
from .paths import SHARED

CORE = (SHARED / "prompts" / "system_core.txt").read_text().strip()
ACTION_FORMAT = (SHARED / "prompts" / "action_format.txt").read_text().strip()

FEW_SHOT = """Examples:
User: [session] {"member_id":"[MEMBER_ID_1]","today":"2026-11-09",...}
Is acupuncture covered?
Assistant: {"tool": "check_coverage", "args": {"member_id": "[MEMBER_ID_1]", "service": "acupuncture"}}
User: <tool_result name="check_coverage">{"covered":false,...}</tool_result>
Assistant: {"reply": "Acupuncture isn't covered on your Bayside Bronze PPO plan."}

User: [session] {"member_id":"[MEMBER_ID_1]","upcoming_appointments":[{"id":"APT-7001","specialty":"dentist","datetime":"2026-11-10T13:30"}],...}
Move my dentist appointment to Thursday.
Assistant: {"tool": "reschedule_appointment", "args": {"member_id": "[MEMBER_ID_1]", "appointment_id": "APT-7001", "new_date": "2026-11-12"}}"""


def tool_block(kind: str) -> str:
    tools = load_tools()
    if kind == "short":
        lines = []
        for t in tools:
            props = t["input_schema"]["properties"]
            req = set(t["input_schema"].get("required", []))
            args = ", ".join(p if p in req else p + "?" for p in props)
            lines.append(f"- {t['name']}({args}): {t['short']}")
        return "Tools:\n" + "\n".join(lines)
    return "Tools (JSON Schema):\n" + "\n".join(
        f"- {t['name']}: {t['description']}\n  schema: {json.dumps(t['input_schema'], separators=(',', ':'))}" for t in tools)


def system_prompt(kind: str, few_shot: bool = False) -> str:
    """kind: 'teacher' (native tools), 'full' (base models), 'short' (fine-tuned student)."""
    if kind == "teacher":
        return CORE
    parts = [CORE, ACTION_FORMAT, tool_block("short" if kind == "short" else "full")]
    if few_shot:
        parts.append(FEW_SHOT)
    return "\n\n".join(parts)


def request_message(session: dict[str, Any], text: str) -> str:
    return f"[session] {compact(session)}\n\n{text}"


def tool_result_message(name: str, result: dict[str, Any]) -> str:
    return f'<tool_result name="{name}">{compact(result)}</tool_result>'


def action_json(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(", ", ": "))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_action(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a model's output into a valid action, or None."""
    s = text.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("tool"), str) and isinstance(obj.get("args", {}), dict):
        return {"tool": obj["tool"], "args": obj.get("args", {})}
    if isinstance(obj.get("reply"), str):
        return {"reply": obj["reply"]}
    return None
