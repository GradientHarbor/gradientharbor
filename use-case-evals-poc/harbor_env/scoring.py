"""Programmatic scoring against each scenario's expected outcome (ground truth from the template)."""

from __future__ import annotations

from typing import Any


def _norm(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return round(float(s.replace("$", "").replace(",", "")), 2)
        except ValueError:
            return s
    return v


def norm_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {k: _norm(v) for k, v in (args or {}).items() if v not in (None, "")}
    if out.get("in_network_only") is True:
        out.pop("in_network_only")  # default
    for k in ("summary", "reason"):  # free text: not part of exact match
        out.pop(k, None)
    return out


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    a = norm_args(actual)
    for k, v in expected.items():
        if v == "*":
            if k not in a:
                return False
            continue
        if _norm(v) != a.get(k):
            return False
    return True


def call_matches(spec: dict[str, Any], call: dict[str, Any]) -> bool:
    if "any_of" in spec:
        return any(call_matches(s, call) for s in spec["any_of"])
    return spec["tool"] == call["tool"] and args_match(spec.get("args", {}), call["args"])


def state_matches(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    if len(expected) != len(actual):
        return False
    remaining = list(actual)
    for e in expected:
        hit = next((a for a in remaining if all(v == "*" or _norm(a.get(k)) == _norm(v) for k, v in e.items())), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def score_episode(scenario: dict[str, Any], ep: dict[str, Any]) -> dict[str, Any]:
    exp = scenario["expected"]
    calls = ep["calls"]
    required = exp.get("required_calls", [])
    calls_ok = all(any(call_matches(spec, c) for c in calls) for spec in required)
    forbidden = exp.get("forbid_tools", [])
    calls_ok = calls_ok and not any(c["tool"] in forbidden for c in calls)
    if "state_any_of" in exp:
        state_ok = any(state_matches(s, ep["state_diff"]) for s in exp["state_any_of"])
    else:
        state_ok = state_matches(exp.get("state", []), ep["state_diff"])
    if "first_call" in exp:
        first_spec = exp["first_call"]
    else:
        first_spec = required[0] if required else None
    first_action = next((s["action"] for s in ep["steps"] if s.get("action")), None)
    if first_spec is None:
        first_ok = bool(first_action and "reply" in first_action)
    elif first_action and "tool" in first_action:
        first_ok = call_matches(first_spec, {"tool": first_action["tool"], "args": first_action["args"]})
        # Booking or filing needs a provider_id the member never gives: looking it up first is the right first step.
        if not first_ok and first_spec.get("tool") in ("book_appointment", "file_claim"):
            first_ok = first_action["tool"] == "find_provider"
    else:
        first_ok = False
    replied = ep["final_reply"] is not None and ep["error"] is None
    return {
        "first_call_match": first_ok,
        "calls_ok": calls_ok,
        "state_ok": state_ok,
        "replied": replied,
        "task_success": calls_ok and state_ok and replied,
    }
