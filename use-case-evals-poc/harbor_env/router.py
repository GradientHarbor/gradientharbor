"""Router gates: validators + confidence + policy + never-leave (shared/router.json).

The iOS HarborKit Router implements the same checks with the same thresholds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import jsonschema

from .env import HarborEnv, load_tools
from .grounding import ungrounded
from .paths import SHARED

CONFIG = json.loads((SHARED / "router.json").read_text())
NEVER_LEAVE = re.compile(CONFIG["never_leave"]["regex"], re.IGNORECASE)
TOOLS = {t["name"]: t for t in load_tools()}


def is_never_leave(text: str) -> bool:
    return bool(NEVER_LEAVE.search(text or ""))


@dataclass
class Gate:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    min_logprob: float | None = None
    mean_logprob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": self.reasons, "min_logprob": self.min_logprob,
                "mean_logprob": self.mean_logprob}


class Router:
    def __init__(self, min_logprob: float | None = None, mean_logprob: float | None = None,
                 use_confidence: bool = True):
        conf = CONFIG["confidence"]
        self.min_lp = conf["min_token_logprob"] if min_logprob is None else min_logprob
        self.mean_lp = conf["mean_token_logprob"] if mean_logprob is None else mean_logprob
        self.use_confidence = use_confidence
        self.claim_threshold = CONFIG["policy"]["claim_review_threshold"]

    def validate(self, action: dict[str, Any] | None, real_args: dict[str, Any] | None, env: HarborEnv,
                 context: list[str] | None = None) -> list[str]:
        if action is None:
            return ["invalid_json"]
        if "reply" in action:
            if not action["reply"].strip():
                return ["empty_reply"]
            if context is not None and ungrounded(action["reply"], context):
                return ["reply_number_not_grounded"]
            return []
        tool = action["tool"]
        if tool not in TOOLS:
            return ["unknown_tool"]
        args = real_args or {}
        reasons = []
        try:
            jsonschema.validate(args, TOOLS[tool]["input_schema"])
        except jsonschema.ValidationError as e:
            reasons.append(f"schema:{e.validator}:{'.'.join(map(str, e.path)) or e.message[:40]}")
        if args.get("member_id") != env.member_id:
            reasons.append("member_id_mismatch")
        d = env.data
        if "provider_id" in args and not any(p["provider_id"] == args["provider_id"] for p in d["providers"]):
            reasons.append("unknown_provider_id")
        if "appointment_id" in args and not any(a["appointment_id"] == args["appointment_id"] and
                                                a["member_id"] == env.member_id for a in d["appointments"]):
            reasons.append("unknown_appointment_id")
        if args.get("claim_id") and not any(c["claim_id"] == args["claim_id"] and c["member_id"] == env.member_id
                                            for c in d["claims"]):
            reasons.append("unknown_claim_id")
        for key in ("date", "new_date"):
            if key in args:
                try:
                    if date.fromisoformat(str(args[key])[:10]) <= env.today:
                        reasons.append("booking_date_not_after_today")
                except ValueError:
                    reasons.append(f"bad_{key}")
        if "service_date" in args:
            try:
                if date.fromisoformat(str(args["service_date"])[:10]) > env.today:
                    reasons.append("service_date_in_future")
            except ValueError:
                reasons.append("bad_service_date")
        return reasons

    def policy(self, action: dict[str, Any] | None, real_args: dict[str, Any] | None) -> list[str]:
        if action and action.get("tool") == "file_claim":
            try:
                if float((real_args or {}).get("amount", 0)) > self.claim_threshold:
                    return ["policy_claim_over_threshold"]
            except (TypeError, ValueError):
                pass
        return []

    def check(self, action: dict[str, Any] | None, real_args: dict[str, Any] | None, env: HarborEnv,
              logprobs: list[float] | None, context: list[str] | None = None) -> Gate:
        reasons = self.validate(action, real_args, env, context) + self.policy(action, real_args)
        mn = mean = None
        if logprobs:
            mn, mean = min(logprobs), sum(logprobs) / len(logprobs)
            # Confidence gates tool calls only: reply text is naturally lower-probability and is
            # checked by validators, not by logprobs.
            is_tool = bool(action and "tool" in action)
            if self.use_confidence and is_tool and (mn < self.min_lp or mean < self.mean_lp):
                reasons.append("low_confidence")
        return Gate(ok=not reasons, reasons=reasons, min_logprob=mn, mean_logprob=mean)
