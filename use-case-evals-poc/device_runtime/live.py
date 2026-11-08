"""Live, multi-turn version of harbor_env.agent.run_episode for the app.

Differences from the eval loop:
- conversations span several member turns; each turn starts with a fresh [session] line;
- write tools (book, reschedule, file_claim) pause for a confirmation card in the app;
- escalations go to the backend teacher proxy over HTTP (no provider key on the "phone");
- offline mode: if a gate fires and the cloud can't be reached, the member gets a best-effort
  answer marked "flagged for review" and the request joins the offline queue;
- shadow samples run in the background so they never slow the chat;
- an escalation at the step limit gives the teacher one last step (the eval loop ended there).

run_turn() is a generator of UI events. When it yields {"type": "confirm"}, the caller resumes
it later with gen.send(approved).
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Generator

import httpx

from harbor_env.agent import RULES, SCANNER, _same_action
from harbor_env.env import HarborEnv, load_fixtures
from harbor_env.prompts import action_json, request_message, tool_result_message
from harbor_env.router import CONFIG as ROUTER_CONFIG
from harbor_env.router import Router, is_never_leave
from gradientharbor.privacy import Redactor

MAX_STEPS = ROUTER_CONFIG["max_steps"]
WRITE_TOOLS = {"book_appointment", "reschedule_appointment", "file_claim"}
STATE_TOOLS = WRITE_TOOLS | {"escalate_to_human"}
LABELS = {
    "get_policy": "Checking your plan", "check_coverage": "Checking coverage", "get_claim_status": "Looking up claims",
    "file_claim": "Filing the claim", "find_provider": "Finding providers", "book_appointment": "Booking the appointment",
    "reschedule_appointment": "Moving the appointment", "escalate_to_human": "Creating a support ticket",
}
OFFLINE_REPLY = ("I can't finish this one without a connection. I've saved your request and will complete it "
                 "as soon as you're back online.")
DECLINED_REPLY = "Okay, I haven't changed anything. What would you like instead?"


def plain_reason(reasons: list[str]) -> str:
    r = " ".join(reasons)
    if "policy_claim_over_threshold" in r:
        return "claims over $1,000 are always reviewed"
    if "reply_number_not_grounded" in r:
        return "the answer had a number I couldn't verify"
    if "max_steps" in r:
        return "the request needed more steps"
    if "low_confidence" in r:
        return "unusual request (low confidence)"
    if "date" in r:
        return "a date didn't check out"
    if "unknown_" in r or "mismatch" in r:
        return "it referenced something that doesn't exist"
    if "schema:enum" in r:
        return "the on-device model picked an option that doesn't exist"
    if "schema:required" in r:
        return "the on-device model left out a required detail"
    if "invalid_json" in r:
        return "the on-device model's output wasn't a valid action"
    return "the on-device step didn't pass a check"


def nice_date(s: str) -> str:
    d = date.fromisoformat(str(s)[:10])
    return d.strftime("%a, %b ") + str(d.day)


def nice_time(t: str) -> str:
    h, m = (int(x) for x in str(t)[:5].split(":"))
    return f"{(h - 1) % 12 + 1}:{m:02d} {'PM' if h >= 12 else 'AM'}"


@dataclass
class Settings:
    offline: bool = False
    # Off by default so on-stage on-device requests leave the ledger empty; turn on in the app's Settings.
    shadow_rate: float = float(os.environ.get("GH_SHADOW_RATE", "0"))
    sync: bool = False
    backend: str = os.environ.get("GH_COLLECTOR_URL", "http://localhost:8000")
    device_id: str = os.environ.get("GH_DEVICE_ID", "sim-iphone-air")


@dataclass
class Conversation:
    id: str
    member_id: str
    red: Redactor
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending: Any = None  # suspended run_turn generator waiting for a confirmation
    history: list[dict[str, Any]] = field(default_factory=list)  # earlier turns, no longer in the model context


class Blocked(Exception):
    pass


class Offline(Exception):
    pass


class Device:
    def __init__(self, model_path: str, model_name: str):
        self.model_path, self.model_name = model_path, model_name
        self.student = None
        self.load_error: str | None = None
        self.lock = threading.Lock()
        self.router = Router()
        self.settings = Settings()
        self.envs: dict[str, HarborEnv] = {}
        self.convs: dict[str, Conversation] = {}
        self.ledger: list[dict[str, Any]] = []
        self.queue: list[dict[str, Any]] = []
        self.counters: Counter[str] = Counter()
        self.rng = random.Random(7)

    # ---- model ----------------------------------------------------------------------------------
    def load(self) -> None:
        try:
            from harbor_env.models import local_backend

            st = local_backend(self.model_path, prompt="short", name=self.model_name)
            with self.lock:  # warm up: compile kernels and the system-prompt path
                st.step([{"role": "user", "kind": "request", "content": "[session] {}\n\nhi"}])
            self.student = st
        except Exception as e:  # surfaced in /status
            self.load_error = f"{type(e).__name__}: {e}"

    # ---- data -----------------------------------------------------------------------------------
    def env(self, member_id: str) -> HarborEnv:
        if member_id not in self.envs:
            self.envs[member_id] = HarborEnv(member_id)
        return self.envs[member_id]

    def reset(self, member_id: str | None = None) -> None:
        if member_id:
            self.envs.pop(member_id, None)
            self.convs = {k: c for k, c in self.convs.items() if c.member_id != member_id}
        else:
            self.envs.clear()
            self.convs.clear()
            self.ledger.clear()
            self.queue.clear()
            self.counters.clear()

    def conversation(self, conv_id: str | None, member_id: str) -> Conversation:
        if conv_id and conv_id in self.convs and self.convs[conv_id].member_id == member_id:
            return self.convs[conv_id]
        env = self.env(member_id)
        c = Conversation(id=conv_id or uuid.uuid4().hex[:12], member_id=member_id, red=Redactor(RULES, env.known_pii()))
        self.convs[c.id] = c
        return c

    # ---- ledger + cloud -------------------------------------------------------------------------
    def _ledger(self, **row: Any) -> dict[str, Any]:
        row = {"id": uuid.uuid4().hex[:10], "at": datetime.now().isoformat(timespec="seconds"), **row}
        self.ledger.insert(0, row)
        return row

    def teacher(self, messages: list[dict[str, Any]], reason: str, conv_id: str) -> dict[str, Any]:
        if self.settings.offline:
            raise Offline()
        payload = "\n".join(m["content"] for m in messages if m["role"] == "user")
        findings = SCANNER.scan(payload)  # on-device scan, fail closed
        if findings:
            self._ledger(reason=reason, status="blocked", payload=None, conv_id=conv_id,
                         findings=sorted({f["type"] for f in findings}))
            raise Blocked()
        body = {"device_id": self.settings.device_id, "trace_id": conv_id, "reason": reason,
                "messages": [{k: m[k] for k in ("role", "kind", "tool", "content", "action", "native") if k in m}
                             for m in messages]}
        try:
            r = httpx.post(f"{self.settings.backend}/v1/teacher", json=body, timeout=120)
        except httpx.TransportError:
            raise Offline()
        if r.status_code == 422:
            self._ledger(reason=reason, status="blocked", payload=None, conv_id=conv_id, findings=["server_scan"])
            raise Blocked()
        if r.status_code >= 500:
            raise Offline()
        r.raise_for_status()
        out = r.json()
        a = out.get("action")
        req = next((m["content"].split("\n\n", 1)[-1] for m in messages if m.get("kind") == "request"), None)
        self._ledger(reason=reason, status="sent", payload=payload, conv_id=conv_id, request=req,
                     response=(action_json(a) if a else "(no action)")[:600],
                     latency_ms=round(out.get("latency_ms") or 0), cost_usd=round(out.get("cost_usd") or 0, 5),
                     model=out.get("model"))
        return out

    def _shadow(self, messages: list[dict[str, Any]], student_action: dict[str, Any], conv_id: str) -> None:
        def work() -> None:
            try:
                out = self.teacher(messages, "shadow", conv_id)
                agree = _same_action(student_action, out.get("action"))
                self.counters["shadow"] += 1
                self.counters["shadow_agree"] += int(agree)
                httpx.post(f"{self.settings.backend}/v1/shadow", timeout=10, json={
                    "trace_id": conv_id, "run_id": self.settings.device_id, "agree": agree,
                    "student": student_action, "teacher": out.get("action")})
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    def replay_offline_queue(self) -> int:
        items, self.queue = self.queue, []

        def work() -> None:
            for it in items:
                try:
                    self.teacher(it["messages"], "offline_queue", it["conv_id"])
                    for r in self.ledger:
                        if r["id"] == it.get("row_id"):
                            r["status"], r["note"] = "replayed", "sent for review after reconnecting"
                except (Offline, Blocked):
                    self.queue.append(it)
                except Exception:
                    pass

        threading.Thread(target=work, daemon=True).start()
        return len(items)

    def sync_trace(self, conv: Conversation, turn: dict[str, Any]) -> None:
        if not self.settings.sync:
            return
        msgs = [{k: m[k] for k in ("role", "kind", "tool", "content", "action", "served_by") if k in m}
                for m in conv.messages[turn["start"]:]]

        def work() -> None:
            try:
                httpx.post(f"{self.settings.backend}/v1/traces", timeout=20, json=[{
                    "source": "phone", "run_id": self.settings.device_id, "served_by": turn["served_by"],
                    "escalated": turn["escalated"], "escalation_reasons": turn["reasons"],
                    "never_leave": turn["never_leave"], "request_redacted": turn["redacted"], "messages": msgs,
                    "final_reply_redacted": turn["reply_redacted"], "latency_ms": turn["latency_s"] * 1000,
                    "steps": turn["steps"], "model": self.model_name}])
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    # ---- confirmation text ----------------------------------------------------------------------
    def summary(self, env: HarborEnv, tool: str, a: dict[str, Any]) -> str:
        try:
            if tool == "reschedule_appointment":
                appt = next((x for x in env.data["appointments"] if x["appointment_id"] == a.get("appointment_id")), None)
                kind = (appt.get("reason") or appt["specialty"].replace("_", " ") + " appointment").lower() if appt else ""
                what = f"your {kind} with {appt['provider_name']}" if appt else "your appointment"
                when = nice_date(a["new_date"]) + (f" at {nice_time(a['new_time'])}" if a.get("new_time") else
                                                   " (same time if it's free)")
                return f"Move {what} to {when}?"
            if tool == "book_appointment":
                p = next((x for x in env.data["providers"] if x["provider_id"] == a.get("provider_id")), None)
                when = nice_date(a["date"]) + (f" at {nice_time(a['time'])}" if a.get("time") else " (first free time)")
                return f"Book {p['name'] if p else 'an appointment'}{' (' + p['clinic'] + ')' if p else ''} on {when}?"
            if tool == "file_claim":
                p = next((x for x in env.data["providers"] if x["provider_id"] == a.get("provider_id")), None)
                amt = f"${float(a['amount']):,.2f}".removesuffix(".00")
                svc = env.coverage[env.member["plan"]].get(str(a.get("service")), {}).get("label") or str(a.get("service", ""))
                return (f"File a {amt} claim for a {svc.lower()} with "
                        f"{p['name'] if p else 'the provider'} on {nice_date(a['service_date'])}?")
        except Exception:
            pass
        return f"Allow {LABELS.get(tool, tool).lower()}?"

    # ---- the turn -------------------------------------------------------------------------------
    def run_turn(self, conv: Conversation, text: str) -> Generator[dict[str, Any], Any, None]:
        env, red = self.env(conv.member_id), conv.red
        t0 = time.perf_counter()
        # The student was trained on single requests. Keep the previous turn only when it ended with a
        # question to the member (a clarifying question); otherwise start a fresh context. This also keeps
        # prompts short, which is most of the per-step latency.
        last = next((m for m in reversed(conv.messages) if m["role"] == "assistant"), None)
        if conv.messages and not (last and str(last["action"].get("reply", "")).rstrip().endswith("?")):
            conv.history.extend(conv.messages)
            conv.messages = []
        never_leave = is_never_leave(text)
        user_red = red.redact_text(text)
        start = len(conv.messages)
        conv.messages.append({"role": "user", "kind": "request",
                              "content": request_message(red.redact_obj(env.session_context()), user_red)})
        yield {"type": "start", "never_leave": never_leave, "redacted": user_red}

        controller, reasons, flagged, handoff = "student", [], False, False
        served: list[str] = []
        final_red: str | None = None
        tool_steps = 0
        model_steps = 0
        paused = 0.0  # time spent waiting on confirmation cards (not latency)

        def queue_offline(why: list[str]) -> None:
            row = self._ledger(reason="offline_queue", status="queued", payload=None, conv_id=conv.id, request=user_red,
                               note="waiting for a connection: " + ", ".join(why))
            self.queue.append({"conv_id": conv.id, "messages": [dict(m) for m in conv.messages[start:]], "reasons": why,
                               "row_id": row["id"]})

        for _ in range(MAX_STEPS + 3):
            msgs = conv.messages
            action, native, who = None, None, controller
            if controller == "student":
                if self.student is None:
                    yield {"type": "error", "message": self.load_error or "The on-device model is still loading."}
                    return
                yield {"type": "thinking", "who": "student"}
                with self.lock:
                    out = self.student.step(msgs)
                model_steps += 1
                action = out.action
                real_args = red.refill_obj(action["args"]) if action and "tool" in action else None
                gate = self.router.check(action, real_args, env, out.logprobs,
                                         [m["content"] for m in msgs if m["role"] == "user"])
                if action and "tool" in action and tool_steps >= MAX_STEPS:
                    gate.ok, gate.reasons = False, gate.reasons + ["max_steps"]
                yield {"type": "model_step", "who": "student", "latency_s": round(out.latency_s, 2),
                       "prompt_tokens": out.prompt_tokens, "action": action, "raw": None if action else out.raw[:300],
                       "gate": gate.reasons}
                if not gate.ok:
                    if never_leave:
                        handoff, who = True, "local_handoff"
                        action = {"tool": "escalate_to_human", "args": {"member_id": env.member_id, "topic": "medical",
                                                                        "summary": "Member asked a medical question; needs a human."}}
                        action = red.redact_obj(action)
                        yield {"type": "handoff", "plain": "This stays on your phone. A Bayside team member will follow up."}
                    elif self.settings.offline:
                        flagged = True
                        if action and "reply" in action and action["reply"].strip():
                            conv.messages.append({"role": "assistant", "content": action_json(action), "action": action,
                                                  "served_by": "student"})
                            final_red = action["reply"]
                        else:
                            final_red = OFFLINE_REPLY
                        served.append("student")
                        queue_offline(gate.reasons)
                        break
                    else:
                        controller, reasons = "teacher", gate.reasons
                        yield {"type": "escalate", "reasons": reasons, "plain": plain_reason(reasons)}
                        continue
                elif self.settings.shadow_rate > 0 and not never_leave and not self.settings.offline \
                        and self.rng.random() < self.settings.shadow_rate:
                    self._shadow([dict(m) for m in msgs[start:]], action, conv.id)
            else:
                yield {"type": "thinking", "who": "teacher"}
                try:
                    # Only this turn goes to the cloud: earlier turns may include never-leave requests.
                    out = self.teacher(msgs[start:], "escalation:" + ",".join(reasons or ["teacher"]), conv.id)
                except Blocked:
                    handoff, who = True, "blocked_by_pii_scan"
                    action = red.redact_obj({"tool": "escalate_to_human", "args": {
                        "member_id": env.member_id, "topic": "other",
                        "summary": "Request could not be sent safely; needs a human."}})
                    yield {"type": "handoff", "plain": "This couldn't be sent safely, so a person will follow up."}
                except Offline:
                    flagged = True
                    final_red = OFFLINE_REPLY
                    queue_offline(reasons or ["offline"])
                    break
                else:
                    action, native = out.get("action"), out.get("native")
                    yield {"type": "model_step", "who": "teacher", "latency_s": round((out.get("latency_ms") or 0) / 1000, 2),
                           "action": action, "gate": []}
                    if action is None:
                        final_red = "Sorry, I can't help with that one. A Bayside team member can."
                        served.append("teacher")
                        break
                    if "tool" in action and tool_steps >= MAX_STEPS + 1:
                        final_red = "I've done what I can here. A Bayside team member can help with the rest."
                        served.append("teacher")
                        break

            served.append(who)
            msg = {"role": "assistant", "content": action_json(action), "action": action, "served_by": who}
            if native:
                msg["native"] = native
            if "reply" in action:
                conv.messages.append(msg)
                final_red = action["reply"]
                break

            tool, real_args = action["tool"], red.refill_obj(action["args"])
            if tool in WRITE_TOOLS:
                t_wait = time.perf_counter()
                approved = yield {"type": "confirm", "tool": tool, "summary": self.summary(env, tool, real_args)}
                paused += time.perf_counter() - t_wait
                if not approved:
                    conv.messages.append(msg)
                    conv.messages.append({"role": "user", "kind": "tool_result", "tool": tool, "content":
                                          tool_result_message(tool, {"error": "the member declined this change"})})
                    final_red = DECLINED_REPLY
                    served.append("local_template")
                    break
            yield {"type": "tool", "tool": tool, "label": LABELS.get(tool, tool), "status": "running"}
            result = env.execute(tool, real_args)
            tool_steps += 1
            conv.messages.append(msg)
            conv.messages.append({"role": "user", "kind": "tool_result", "tool": tool,
                                  "content": tool_result_message(tool, red.redact_obj(result))})
            ok = "error" not in result
            yield {"type": "tool", "tool": tool, "label": LABELS.get(tool, tool), "status": "ok" if ok else "error",
                   "error": result.get("error")}
            if ok and tool in STATE_TOOLS:
                yield {"type": "state_changed", "tool": tool}
            if handoff:
                final_red = (f"I've passed this to a Bayside team member (ticket {result.get('ticket_id')}). "
                             f"They'll follow up {result.get('expected_response', 'soon')}.")
                break
        else:
            final_red = "I've done what I can here. A Bayside team member can help with the rest."

        if final_red is not None and not any(m.get("action", {}).get("reply") == final_red for m in conv.messages[start:]):
            conv.messages.append({"role": "assistant", "content": action_json({"reply": final_red}),
                                  "action": {"reply": final_red}, "served_by": "local_template"})
        escalated = "teacher" in served
        kind = ("handoff" if handoff else "cloud" if escalated else "device")
        latency = round(time.perf_counter() - t0 - paused, 2)
        self.counters["requests"] += 1
        self.counters["on_device" if not escalated else "escalated"] += 1
        if flagged:
            self.counters["flagged"] += 1
        yield {"type": "reply", "text": red.refill_text(final_red or ""), "served": kind, "flagged": flagged,
               "escalated": escalated, "reason": plain_reason(reasons) if escalated else None,
               "latency_s": latency, "model_steps": model_steps, "tool_steps": tool_steps, "never_leave": never_leave}
        self.sync_trace(conv, {"start": start, "served_by": "teacher" if escalated else "student",
                               "escalated": escalated, "reasons": reasons, "never_leave": never_leave,
                               "redacted": user_red, "reply_redacted": final_red, "latency_s": latency,
                               "steps": model_steps})

    # ---- member data for the app screens --------------------------------------------------------
    def state(self, member_id: str) -> dict[str, Any]:
        env = self.env(member_id)
        m = env.member
        pol = env._policy()
        cov = env.coverage[m["plan"]]
        claims = sorted((c for c in env.data["claims"] if c["member_id"] == member_id),
                        key=lambda c: c["service_date"], reverse=True)
        appts = sorted((a for a in env.data["appointments"] if a["member_id"] == member_id and a["status"] == "scheduled"),
                       key=lambda a: a["datetime"])
        return {
            "today": env.today.isoformat(),
            "member": {k: m[k] for k in ("member_id", "name", "first_name", "city", "plan", "policy_id", "dob", "phone",
                                         "email", "address")},
            "policy": {k: pol[k] for k in ("policy_id", "plan_name", "status", "effective_date", "renewal_date",
                                           "deductible", "deductible_met", "out_of_pocket_max", "out_of_pocket_spent",
                                           "dependents")},
            "coverage": [{"service": k, "label": v["label"], "covered": v["covered"], "copay": v.get("copay"),
                          "coinsurance": v.get("coinsurance"), "requires_referral": v.get("requires_referral", False),
                          "requires_prior_auth": v.get("requires_prior_auth", False),
                          "visit_limit_per_year": v.get("visit_limit_per_year"),
                          "out_of_network": v.get("out_of_network")} for k, v in cov.items()],
            "claims": [{k: c.get(k) for k in ("claim_id", "provider_name", "service", "service_date", "status",
                                               "amount_billed", "amount_covered", "denial_reason", "expected_decision_by")}
                       for c in claims],
            "appointments": [{k: a.get(k) for k in ("appointment_id", "provider_name", "specialty", "datetime", "reason")}
                             for a in appts],
            "tickets": list(env.data["tickets"]),
        }

    def members(self) -> list[dict[str, str]]:
        return [{"member_id": m["member_id"], "name": m["name"], "plan": m["plan"], "city": m["city"]}
                for m in load_fixtures()["members"]]
