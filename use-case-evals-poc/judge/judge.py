"""LLM-as-judge for the final reply, on a different provider than the teacher (OpenAI), so
Sonnet never grades its own answers. The deterministic metrics (state diff, tool-call match)
stay primary; the judge covers what they can't: is the reply correct, grounded and useful.

Calibrated before each run against hand-labelled cases (pattern from Loopr's interview evals):
a good reply rejected or a bad reply accepted fails calibration.
"""

from __future__ import annotations

import json
import os
from typing import Any

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-6.1-sol")

RUBRIC = """You grade a support assistant's final reply in an insurance member app.
You get the session context the assistant saw (member, today's date, upcoming appointments, recent claims), the
member's request, the tool calls the assistant made with their results (ground truth), hints from the
scenario author, and the final reply. Judge only the reply:
- correct: it answers what the member asked, consistent with the tool results (numbers, dates, names, statuses), and
  confirms any action that was taken. If the request was ambiguous, asking one short clarifying question is correct.
  If the request was a medical question or something tools can't do, handing off to a human is correct.
- grounded: it states no policy facts, amounts, dates or provider details that are absent from the tool results and
  the session context.
- score: 1 (wrong or harmful) to 5 (correct, complete, concise).
Be strict about wrong numbers and invented facts; ignore tone and length unless they hide the answer."""

SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "grounded": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["correct", "grounded", "score", "reason"],
    "additionalProperties": False,
}


class Judge:
    def __init__(self, model: str = JUDGE_MODEL):
        from openai import OpenAI

        self.client = OpenAI(max_retries=3)
        self.model = model

    def grade(self, request: str, calls: list[dict[str, Any]], reply: str | None, hints: dict[str, Any] | None,
              session: dict[str, Any] | None = None) -> dict[str, Any]:
        if not reply:
            return {"correct": False, "grounded": True, "score": 1, "reason": "no reply"}
        payload = {"session_context_shown_to_assistant": session or {}, "member_request": request,
                   "tool_calls": [{"tool": c["tool"], "args": c["args"], "result": c["result"]} for c in calls],
                   "scenario_hints": hints or {}, "final_reply": reply}
        try:
            resp = self.client.responses.create(
                model=self.model,
                instructions=RUBRIC,
                input=json.dumps(payload, ensure_ascii=False, default=str),
                reasoning={"effort": "low"},
                text={"format": {"type": "json_schema", "name": "judgment", "schema": SCHEMA, "strict": True}},
                store=False,
            )
            return json.loads(resp.output_text)
        except Exception as e:  # a judge failure is reported, never silently counted as a pass
            return {"correct": None, "grounded": None, "score": None, "reason": f"judge_error: {type(e).__name__}"}


CALIBRATION = [
    # (request, calls, reply, expected_correct)
    ("Is physical therapy covered?",
     [{"tool": "check_coverage", "args": {"service": "physical_therapy"},
       "result": {"covered": True, "copay": 40, "visit_limit_per_year": 20, "requires_referral": False}}],
     "Yes, physical therapy is covered with a $40 copay per visit, up to 20 visits a year.", True),
    ("Is physical therapy covered?",
     [{"tool": "check_coverage", "args": {"service": "physical_therapy"},
       "result": {"covered": True, "copay": 40, "visit_limit_per_year": 20, "requires_referral": False}}],
     "Yes, physical therapy is covered with a $20 copay and unlimited visits.", False),
    ("Move my eye exam to Friday.",
     [{"tool": "reschedule_appointment", "args": {"appointment_id": "APT-7001", "new_date": "2026-11-13"},
       "result": {"new_datetime": "2026-11-13T13:30", "provider_name": "Dr. Desmond Weiss"}}],
     "Done. Your eye exam with Dr. Desmond Weiss is now Friday, November 13 at 1:30 PM.", True),
    ("Move my eye exam to Friday.",
     [{"tool": "reschedule_appointment", "args": {"appointment_id": "APT-7001", "new_date": "2026-11-13"},
       "result": {"error": "Dr. Desmond Weiss does not work on Fridays", "working_days": ["Monday", "Tuesday"]}}],
     "Your eye exam has been moved to Friday.", False),
    ("Why was my claim denied?",
     [{"tool": "get_claim_status", "args": {"claim_id": "CLM-88013"},
       "result": {"status": "denied", "denial_reason": "Service not covered under your plan"}}],
     "Your claim was denied because the service isn't covered under your plan. You can appeal if you think that's wrong.", True),
    ("Can you move my appointment to Friday?", [],
     "Sure, which appointment would you like to move, your dentist visit on Nov 10 or your PT session on Nov 12?", True,
     {"today": "2026-11-09", "upcoming_appointments": [
         {"id": "APT-7010", "provider": "Dr. Adrienne Hoang", "specialty": "dentist", "datetime": "2026-11-10T09:00"},
         {"id": "APT-7011", "provider": "Dr. Jasper Reyes", "specialty": "physical_therapist", "datetime": "2026-11-12T14:00"}]}),
    ("What's my deductible?",
     [{"tool": "get_policy", "args": {}, "result": {"deductible": 1500, "deductible_met": 750}}],
     "Your deductible is $1,500 and you've met $750 of it, so $750 is left.", True),
    ("What's my deductible?", [], "Your deductible is $1,500.", False),
]


def calibrate(judge: Judge) -> dict[str, Any]:
    misses = []
    for case in CALIBRATION:
        req, calls, reply, want = case[:4]
        got = judge.grade(req, calls, reply, None, case[4] if len(case) > 4 else None)
        if got.get("correct") is not want:
            misses.append({"request": req, "reply": reply, "expected": want, "got": got})
    return {"cases": len(CALIBRATION), "misses": misses, "ok": len(misses) == 0}
