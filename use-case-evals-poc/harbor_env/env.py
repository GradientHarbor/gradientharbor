"""Bayside Mutual tool environment over the shared fixtures.

Each episode gets its own deep copy of the data, so write tools (book, reschedule,
file_claim, escalate) can be scored by the final-state diff. The iOS app implements
the same tools over the same JSON (shared/fixtures, shared/tools.json).
"""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

from .paths import SHARED

SLOT_TIMES = [f"{h:02d}:{m:02d}" for h in range(9, 17) for m in (0, 30)]  # 09:00 .. 16:30


@lru_cache(maxsize=1)
def load_fixtures() -> dict[str, Any]:
    fx = SHARED / "fixtures"
    return {name: json.loads((fx / f"{name}.json").read_text())
            for name in ["meta", "members", "policies", "coverage", "providers", "claims", "appointments"]}


@lru_cache(maxsize=1)
def load_tools() -> list[dict[str, Any]]:
    return json.loads((SHARED / "tools.json").read_text())["tools"]


def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


class ToolError(Exception):
    def __init__(self, message: str, **extra: Any):
        super().__init__(message)
        self.payload = {"error": message, **extra}


class HarborEnv:
    def __init__(self, member_id: str):
        fx = load_fixtures()
        self.meta = fx["meta"]
        self.today = date.fromisoformat(self.meta["sim_today"])
        self.data = copy.deepcopy({k: fx[k] for k in ["members", "policies", "providers", "claims", "appointments"]})
        self.coverage = fx["coverage"]
        self.data["tickets"] = []
        self.member = next(m for m in self.data["members"] if m["member_id"] == member_id)
        self.member_id = member_id
        self._initial = copy.deepcopy(self._mutable())
        self._seq = 0
        self.calls: list[dict[str, Any]] = []

    # ---- context -------------------------------------------------------------------------------
    def session_context(self) -> dict[str, Any]:
        m = self.member
        appts = sorted((a for a in self.data["appointments"]
                        if a["member_id"] == m["member_id"] and a["status"] == "scheduled"), key=lambda a: a["datetime"])
        claims = sorted((c for c in self.data["claims"] if c["member_id"] == m["member_id"]),
                        key=lambda c: c["service_date"], reverse=True)[:5]
        return {
            "today": self.today.isoformat(),
            "weekday": self.today.strftime("%A"),
            "member_id": m["member_id"],
            "first_name": m["first_name"],
            "city": m["city"],
            "plan": m["plan"],
            "policy_id": m["policy_id"],
            "upcoming_appointments": [{"id": a["appointment_id"], "provider": a["provider_name"],
                                       "specialty": a["specialty"], "datetime": a["datetime"]} for a in appts],
            "recent_claims": [{"id": c["claim_id"], "provider": c["provider_name"], "service_date": c["service_date"],
                               "status": c["status"]} for c in claims],
        }

    def known_pii(self) -> dict[str, str]:
        m = self.member
        return {"name": m["name"], "first_name": m["first_name"], "last_name": m["name"].split()[-1],
                "dob": m["dob"], "phone": m["phone"], "email": m["email"], "address": m["address"]}

    # ---- execution -----------------------------------------------------------------------------
    def execute(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"_t_{tool}", None)
        if fn is None:
            result = {"error": f"unknown tool {tool}"}
        else:
            try:
                if args.get("member_id") != self.member_id:
                    raise ToolError("member_id does not match the signed-in member")
                result = fn(**{k: v for k, v in args.items() if k != "member_id"})
            except ToolError as e:
                result = e.payload
            except TypeError as e:
                result = {"error": f"bad arguments: {e}"}
        self.calls.append({"tool": tool, "args": args, "result": result})
        return result

    def _next_id(self, prefix: str, base: int) -> str:
        self._seq += 1
        return f"{prefix}-{base + self._seq}"

    def _provider(self, provider_id: str) -> dict[str, Any]:
        p = next((p for p in self.data["providers"] if p["provider_id"] == provider_id), None)
        if p is None:
            raise ToolError(f"provider {provider_id} not found")
        return p

    def _policy(self) -> dict[str, Any]:
        return next(p for p in self.data["policies"] if p["member_id"] == self.member_id)

    def _parse_date(self, s: str, field: str) -> date:
        try:
            return date.fromisoformat(str(s)[:10])
        except ValueError:
            raise ToolError(f"{field} must be YYYY-MM-DD")

    def _free_times(self, provider: dict[str, Any], d: date, ignore_appt: str | None = None) -> list[str]:
        if d.weekday() not in provider["working_days"]:
            return []
        taken = {a["datetime"][11:16] for a in self.data["appointments"]
                 if a["provider_id"] == provider["provider_id"] and a["status"] == "scheduled"
                 and a["datetime"][:10] == d.isoformat() and a["appointment_id"] != ignore_appt}
        return [t for t in SLOT_TIMES if t not in taken]

    def _check_bookable_day(self, provider: dict[str, Any], d: date) -> None:
        if d <= self.today:
            raise ToolError("date must be after today", today=self.today.isoformat())
        if d > self.today + timedelta(days=60):
            raise ToolError("bookings open up to 60 days ahead")
        if d.weekday() not in provider["working_days"]:
            days = [date(2026, 1, 5 + i).strftime("%A") for i in provider["working_days"]]
            raise ToolError(f"{provider['name']} does not work on {d.strftime('%A')}s", working_days=days)

    def _t_get_policy(self) -> dict[str, Any]:
        p = self._policy()
        return {k: p[k] for k in ["policy_id", "plan_name", "status", "effective_date", "renewal_date", "deductible",
                                  "deductible_met", "out_of_pocket_max", "out_of_pocket_spent", "dependents"]}

    def _t_check_coverage(self, service: str) -> dict[str, Any]:
        plan = self.member["plan"]
        cov = self.coverage[plan].get(service)
        if cov is None:
            raise ToolError(f"unknown service {service}", services=sorted(self.coverage[plan]))
        p = self._policy()
        return {"service": service, "plan_name": p["plan_name"], **cov,
                "deductible_remaining": round(p["deductible"] - p["deductible_met"], 2)}

    def _t_get_claim_status(self, claim_id: str | None = None) -> dict[str, Any]:
        mine = [c for c in self.data["claims"] if c["member_id"] == self.member_id]
        if claim_id:
            c = next((c for c in mine if c["claim_id"] == claim_id), None)
            if c is None:
                raise ToolError(f"claim {claim_id} not found for this member")
            return {k: v for k, v in c.items() if k != "member_id"}
        mine.sort(key=lambda c: c["service_date"], reverse=True)
        return {"claims": [{k: c[k] for k in ["claim_id", "provider_name", "service", "service_date", "status",
                                              "amount_billed"]} for c in mine[:5]]}

    def _t_file_claim(self, provider_id: str, service_date: str, service: str, amount: float) -> dict[str, Any]:
        prov = self._provider(provider_id)
        d = self._parse_date(service_date, "service_date")
        if d > self.today:
            raise ToolError("service_date cannot be in the future")
        if d < self.today - timedelta(days=365):
            raise ToolError("claims must be filed within 365 days of service")
        if service not in self.coverage[self.member["plan"]]:
            raise ToolError(f"unknown service {service}")
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ToolError("amount must be a number")
        if amount <= 0:
            raise ToolError("amount must be positive")
        claim = {"claim_id": self._next_id("CLM", 90000), "member_id": self.member_id, "provider_id": provider_id,
                 "provider_name": prov["name"], "service": service, "service_date": d.isoformat(),
                 "submitted_date": self.today.isoformat(), "amount_billed": round(amount, 2), "amount_covered": None,
                 "status": "submitted", "denial_reason": None, "last_update": self.today.isoformat(),
                 "expected_decision_by": (self.today + timedelta(days=10)).isoformat()}
        self.data["claims"].append(claim)
        return {k: v for k, v in claim.items() if k != "member_id"}

    def _t_find_provider(self, specialty: str, city: str | None = None, in_network_only: bool = True) -> dict[str, Any]:
        plan = self.member["plan"]
        rows = [p for p in self.data["providers"] if p["specialty"] == specialty]
        if not rows:
            raise ToolError(f"unknown specialty {specialty}")
        if city:
            rows = [p for p in rows if p["city"].lower() == str(city).strip().lower()]
        if in_network_only:
            rows = [p for p in rows if plan in p["in_network_plans"]]
        rows.sort(key=lambda p: (not p["accepting_new_patients"], p["name"]))
        out = [{"provider_id": p["provider_id"], "name": p["name"], "clinic": p["clinic"], "city": p["city"],
                "in_network": plan in p["in_network_plans"], "accepting_new_patients": p["accepting_new_patients"],
                "working_days": [date(2026, 1, 5 + i).strftime("%a") for i in p["working_days"]]} for p in rows[:5]]
        return {"providers": out, "count": len(rows)}

    def _t_book_appointment(self, provider_id: str, date: str, time: str | None = None,
                            reason: str | None = None) -> dict[str, Any]:
        prov = self._provider(provider_id)
        d = self._parse_date(date, "date")
        self._check_bookable_day(prov, d)
        if not prov["accepting_new_patients"]:
            raise ToolError(f"{prov['name']} is not accepting new patients")
        free = self._free_times(prov, d)
        t = _norm_time(time) if time else (free[0] if free else None)
        if t is None or t not in free:
            raise ToolError("that time is not available", available_times=free[:8])
        appt = {"appointment_id": self._next_id("APT", 8000), "member_id": self.member_id, "provider_id": provider_id,
                "provider_name": prov["name"], "specialty": prov["specialty"], "datetime": f"{d.isoformat()}T{t}",
                "status": "scheduled", "reason": reason or ""}
        self.data["appointments"].append(appt)
        return {k: v for k, v in appt.items() if k != "member_id"}

    def _t_reschedule_appointment(self, appointment_id: str, new_date: str, new_time: str | None = None) -> dict[str, Any]:
        appt = next((a for a in self.data["appointments"] if a["appointment_id"] == appointment_id
                     and a["member_id"] == self.member_id and a["status"] == "scheduled"), None)
        if appt is None:
            raise ToolError(f"appointment {appointment_id} not found for this member")
        prov = self._provider(appt["provider_id"])
        d = self._parse_date(new_date, "new_date")
        self._check_bookable_day(prov, d)
        free = self._free_times(prov, d, ignore_appt=appointment_id)
        if new_time:
            t = _norm_time(new_time)
        else:
            orig = appt["datetime"][11:16]
            t = orig if orig in free else (free[0] if free else None)
        if t is None or t not in free:
            raise ToolError("that time is not available", available_times=free[:8])
        old = appt["datetime"]
        appt["datetime"] = f"{d.isoformat()}T{t}"
        return {"appointment_id": appointment_id, "provider_name": appt["provider_name"], "old_datetime": old,
                "new_datetime": appt["datetime"], "status": "scheduled"}

    def _t_escalate_to_human(self, topic: str, summary: str) -> dict[str, Any]:
        if topic not in {"claim_appeal", "complaint", "billing", "medical", "other"}:
            raise ToolError(f"unknown topic {topic}")
        ticket = {"ticket_id": self._next_id("HT", 4000), "topic": topic, "summary": summary,
                  "expected_response": "within 1 business day"}
        self.data["tickets"].append(ticket)
        return ticket

    # ---- scoring helpers -----------------------------------------------------------------------
    def _mutable(self) -> dict[str, Any]:
        return {"appointments": {a["appointment_id"]: a for a in self.data["appointments"]},
                "claims": {c["claim_id"]: c for c in self.data["claims"]},
                "tickets": {t["ticket_id"]: t for t in self.data["tickets"]}}

    def state_diff(self) -> list[dict[str, Any]]:
        now, before = self._mutable(), self._initial
        changes: list[dict[str, Any]] = []
        for aid, a in now["appointments"].items():
            if aid not in before["appointments"]:
                changes.append({"type": "appointment_booked", "provider_id": a["provider_id"], "date": a["datetime"][:10],
                                "time": a["datetime"][11:16]})
            elif a["datetime"] != before["appointments"][aid]["datetime"]:
                changes.append({"type": "appointment_rescheduled", "appointment_id": aid, "date": a["datetime"][:10],
                                "time": a["datetime"][11:16]})
        for cid, c in now["claims"].items():
            if cid not in before["claims"]:
                changes.append({"type": "claim_filed", "provider_id": c["provider_id"], "service_date": c["service_date"],
                                "service": c["service"], "amount": c["amount_billed"]})
        for tid, t in now["tickets"].items():
            if tid not in before["tickets"]:
                changes.append({"type": "ticket", "topic": t["topic"]})
        return changes


def _norm_time(t: str) -> str:
    s = str(t).strip().lower().replace(".", ":")
    pm = s.endswith("pm")
    s = s.removesuffix("pm").removesuffix("am").strip()
    try:
        if ":" in s:
            hh, mm = s.split(":")[:2]
        else:
            hh, mm = s, "00"
        h, m = int(hh), int(mm)
    except ValueError:
        return s
    if pm and h < 12:
        h += 12
    return f"{h:02d}:{m:02d}"
