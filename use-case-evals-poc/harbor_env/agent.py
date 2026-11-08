"""Agent loop: single-model runs and the hybrid (student + router + teacher) product path.

The model only ever sees redacted text (placeholders). Tool arguments are re-filled on
the device before execution, tool results are redacted before the model sees them, and
the final reply is re-filled before the member sees it. So the same conversation can go
to the cloud teacher without any further work, and every stored trace is train-ready.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from .env import HarborEnv, load_fixtures
from .paths import SHARED
from .prompts import action_json, request_message, tool_result_message
from .router import CONFIG as ROUTER_CONFIG
from .router import Gate, Router, is_never_leave

sys.path.insert(0, str(SHARED.parent / "backend" / "sdk"))
from gradientharbor.privacy import PIIScanner, Redactor, load_rules  # noqa: E402

RULES = load_rules(SHARED)
SCANNER = PIIScanner(RULES, load_fixtures()["members"])
MAX_STEPS = ROUTER_CONFIG["max_steps"]


def _local_handoff(env: HarborEnv, topic: str, why: str) -> dict[str, Any]:
    return {"tool": "escalate_to_human", "args": {"member_id": env.member_id, "topic": topic, "summary": why}}


def run_episode(scenario: dict[str, Any], student=None, teacher=None, router: Router | None = None,
                mode: str = "single", shadow_rate: float = 0.0, rng: random.Random | None = None) -> dict[str, Any]:
    """mode: 'single' (student only, or teacher only if student is None) or 'hybrid'."""
    rng = rng or random.Random(0)
    env = HarborEnv(scenario["member_id"])
    red = Redactor(RULES, env.known_pii())
    session_real = env.session_context()
    session = red.redact_obj(session_real)
    user_text = red.redact_text(scenario["message"])
    messages: list[dict[str, Any]] = [{"role": "user", "kind": "request", "content": request_message(session, user_text)}]
    never_leave = is_never_leave(scenario["message"])
    controller = "student" if student is not None else "teacher"
    steps: list[dict[str, Any]] = []
    cloud_calls: list[dict[str, Any]] = []
    shadow: list[dict[str, Any]] = []
    final_reply, error = None, None
    escalated_reason: list[str] = []
    cloud_sent_upto = 0  # index into messages already sent to the cloud (assistant turns came from there)

    def send_to_cloud(reason: str) -> tuple[Any, list[dict[str, str]]]:
        nonlocal cloud_sent_upto
        outbound = [m["content"] for m in messages[cloud_sent_upto:] if m["role"] == "user"]
        findings = SCANNER.scan("\n".join(outbound))
        if findings:
            cloud_calls.append({"step": len(steps), "reason": reason, "sent": False, "blocked_by_scan": findings})
            return None, findings
        out = teacher.step(messages)
        cloud_calls.append({"step": len(steps), "reason": reason, "sent": True,
                            "payload": "\n".join(outbound), "cost_usd": out.cost_usd, "latency_s": out.latency_s})
        cloud_sent_upto = len(messages)
        return out, []

    # One extra iteration so an escalation at the step limit still gets one teacher step (fixed 2026-11-08;
    # the published runs used range(MAX_STEPS + 1), so 1 of 25 round-2 escalations never reached the teacher).
    for step_i in range(MAX_STEPS + 2):
        record: dict[str, Any] = {"i": step_i}
        if controller == "student":
            out = student.step(messages)
            real_args = red.refill_obj(out.action["args"]) if out.action and "tool" in out.action else None
            context = [m["content"] for m in messages if m["role"] == "user"]
            gate = router.check(out.action, real_args, env, out.logprobs, context) if router else Gate(ok=True)
            if out.action and "tool" in out.action and step_i >= MAX_STEPS:
                gate.ok, gate.reasons = False, gate.reasons + ["max_steps"]
            record.update(model=student.name, served_by="student", raw=out.raw, latency_s=round(out.latency_s, 3),
                          prompt_tokens=out.prompt_tokens, output_tokens=out.output_tokens, gate=gate.to_dict())
            if mode == "hybrid" and not gate.ok:
                if never_leave:
                    # Never-leave: no cloud. Hand off to a human in the insurer's own system.
                    action = _local_handoff(env, "medical", "Member asked a medical question; needs a human.")
                    record.update(served_by="local_handoff", action=action)
                else:
                    controller = "teacher"
                    escalated_reason = gate.reasons
                    record.update(served_by="escalated", discarded_action=out.action)
                    steps.append(record)
                    continue
            else:
                action = out.action
                if action and "tool" in action and step_i >= MAX_STEPS:
                    action, error = None, "max_steps"  # student-only runs: the step limit ends the episode
                if (action is not None and mode == "hybrid" and shadow_rate > 0 and not never_leave and teacher is not None
                        and rng.random() < shadow_rate):
                    sh, blocked = send_to_cloud("shadow")
                    if sh is not None:
                        shadow.append({"step": step_i, "student": out.action, "teacher": sh.action,
                                       "agree": _same_action(out.action, sh.action)})
                        cloud_sent_upto = 0  # shadow call does not become part of the transcript
        else:
            if mode == "hybrid":
                out, blocked = send_to_cloud("escalation:" + ",".join(escalated_reason or ["teacher"]))
                if out is None:
                    action = _local_handoff(env, "other", "Request could not be sent safely; needs a human.")
                    record.update(model="local", served_by="blocked_by_pii_scan", action=action)
                    out = None
            else:
                out = teacher.step(messages)
                cloud_calls.append({"step": step_i, "reason": "teacher_only", "sent": True,
                                    "payload": messages[-1]["content"], "cost_usd": out.cost_usd,
                                    "latency_s": out.latency_s})
            if out is not None:
                action = out.action
                record.update(model=teacher.name, served_by="teacher", raw=out.raw, latency_s=round(out.latency_s, 3),
                              prompt_tokens=out.prompt_tokens, output_tokens=out.output_tokens,
                              cost_usd=round(out.cost_usd, 5), error=out.error)
            if action is not None and "tool" in action and step_i >= MAX_STEPS:
                action = None
                error = "max_steps"

        record["action"] = action
        if action is None:
            error = error or ("invalid_action" if out is None or not out.error else out.error)
            steps.append(record)
            break
        msg = {"role": "assistant", "content": action_json(action), "action": action,
               "served_by": record.get("served_by", "student"), "model": record.get("model")}
        if record.get("served_by") == "teacher" and out is not None and out.native:
            msg["native"] = out.native
        messages.append(msg)
        if "reply" in action:
            final_reply = red.refill_text(action["reply"])
            steps.append(record)
            break
        real_args = red.refill_obj(action["args"])
        result = env.execute(action["tool"], real_args)
        record["tool_result"] = result
        messages.append({"role": "user", "kind": "tool_result", "tool": action["tool"],
                         "content": tool_result_message(action["tool"], red.redact_obj(result))})
        steps.append(record)
        if record.get("served_by") in ("local_handoff", "blocked_by_pii_scan"):
            # Deterministic local reply: no model in the loop once we've handed off.
            final_reply = (f"I've passed this to a Bayside team member (ticket {result.get('ticket_id')}). "
                           f"They'll follow up {result.get('expected_response', 'soon')}.")
            messages.append({"role": "assistant", "content": action_json({"reply": final_reply}),
                             "action": {"reply": final_reply}, "served_by": "local_template", "model": "local"})
            break

    leaked = sum(len(SCANNER.scan(c["payload"])) for c in cloud_calls if c.get("sent"))
    return {
        "scenario_id": scenario["id"],
        "mode": mode,
        "never_leave": never_leave,
        "steps": steps,
        "calls": env.calls,
        "state_diff": env.state_diff(),
        "final_reply": final_reply,
        "error": error,
        "cloud_calls": cloud_calls,
        "shadow": shadow,
        "escalated": any(s.get("served_by") == "escalated" for s in steps),
        "escalation_reasons": escalated_reason,
        "left_device": any(c.get("sent") and c["reason"] != "shadow" for c in cloud_calls),
        "pii_fields_sent": leaked,
        "messages": [{k: v for k, v in m.items() if k != "native"} for m in messages],
        "redacted_request": user_text,
        "session": session_real,
    }


def _same_action(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    if "reply" in a or "reply" in b:
        return "reply" in a and "reply" in b
    from .scoring import norm_args
    return a["tool"] == b["tool"] and norm_args(a["args"]) == norm_args(b["args"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
