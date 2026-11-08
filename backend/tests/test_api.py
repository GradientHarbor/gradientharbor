import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db

SHARED = Path(__file__).resolve().parents[2] / "shared"


@pytest.fixture()
def client(tmp_path):
    db.reset_for_tests(tmp_path / "t.db")
    from app.main import app

    return TestClient(app)


def test_redaction_vectors():
    from gradientharbor.privacy import Redactor, load_rules

    rules = load_rules(SHARED)
    vec = json.loads((SHARED / "redaction_vectors.json").read_text())
    for g in vec["groups"]:
        r = Redactor(rules, g["known"])
        for c in g["cases"]:
            assert r.redact_text(c["input"]) == c["expected"]
        for o in g["objects"]:
            assert r.redact_obj(o["input"]) == o["expected"]


def test_trace_with_raw_pii_is_rejected(client):
    good = {"id": "t1", "source": "eval", "request_redacted": "Is PT covered for [MEMBER_ID_1]?",
            "messages": [{"role": "user", "content": "Is PT covered for [MEMBER_ID_1]?"}], "outcome": {"task_success": True}}
    bad = {"id": "t2", "source": "eval", "request_redacted": "I'm BM-10200, call (408) 555-0100",
           "messages": [{"role": "user", "content": "I'm BM-10200"}]}
    res = client.post("/v1/traces", json={"device_id": "test", "traces": [good, bad]}).json()
    assert res["accepted"] == 1 and res["rejected"] == 1
    t2 = client.get("/v1/traces/t2").json()
    assert t2["messages"] is None and t2["reject_reason"].startswith("pii_scan")


def test_teacher_proxy_fails_closed(client):
    r = client.post("/v1/teacher", json={"messages": [{"role": "user", "content": "member BM-10200 here"}]})
    assert r.status_code == 422
    led = client.get("/v1/ledger").json()
    assert led["stats"]["blocked"] == 1 and led["stats"]["raw_pii_fields_sent"] == 0


def test_dataset_builder_keeps_teacher_steps_and_filters_failures(client):
    msgs = [{"role": "user", "kind": "request", "content": "[session] {}\n\nIs PT covered?"},
            {"role": "assistant", "content": '{"tool": "check_coverage", "args": {}}', "served_by": "teacher"},
            {"role": "user", "kind": "tool_result", "content": '<tool_result name="check_coverage">{}</tool_result>'},
            {"role": "assistant", "content": '{"reply": "Yes."}', "served_by": "teacher"}]
    ok = {"id": "a", "source": "eval", "run_id": "r", "messages": msgs, "request_redacted": "Is PT covered?",
          "outcome": {"task_success": True}}
    fail = {**ok, "id": "b", "outcome": {"task_success": False}}
    client.post("/v1/traces", json={"traces": [ok, fail]})
    res = client.post("/v1/datasets/build", json={"round": 1, "run_ids": ["r"], "valid_frac": 0.0}).json()
    assert res["stats"]["filtered_outcome"] == 1
    assert res["stats"]["train"] == 2
    first = json.loads((Path(res["path"]) / "train.jsonl").read_text().splitlines()[0])
    assert first["messages"][0]["role"] == "system" and first["messages"][-1]["role"] == "assistant"


def _convo(request: str, student_tool: str, served_by: str = "student"):
    return [{"role": "user", "kind": "request", "content": f"[session] {{}}\n\n{request}"},
            {"role": "assistant", "content": json.dumps({"tool": student_tool, "args": {}}), "served_by": served_by},
            {"role": "user", "kind": "tool_result", "content": f'<tool_result name="{student_tool}">{{}}</tool_result>'},
            {"role": "assistant", "content": '{"reply": "Done."}', "served_by": served_by}]


def test_shadow_disagreements_become_examples(client):
    t = {"id": "s1", "source": "traffic", "run_id": "traffic-1", "messages": _convo("Move my visit", "get_appointments"),
         "request_redacted": "Move my visit", "outcome": {"task_success": True},
         "shadow": [{"step": 0, "student": {"tool": "get_appointments", "args": {}},
                     "teacher": {"tool": "list_appointments", "args": {}}, "agree": False},
                    {"step": 1, "student": {"reply": "Done."}, "teacher": {"reply": "Done."}, "agree": True}]}
    client.post("/v1/traces", json={"traces": [t]})
    off = client.post("/v1/datasets/build", json={"round": 9, "run_ids": ["traffic-1"], "valid_frac": 0.0}).json()
    assert off["stats"]["train"] == 0  # student-served steps alone are never training data
    on = client.post("/v1/datasets/build", json={"round": 9, "run_ids": ["traffic-1"], "valid_frac": 0.0,
                                                 "include_shadow": True}).json()
    assert on["stats"]["by_source"] == {"shadow": 1}
    ex = json.loads((Path(on["path"]) / "train.jsonl").read_text().splitlines()[0])["messages"]
    assert json.loads(ex[-1]["content"])["tool"] == "list_appointments" and len(ex) == 3  # system, request, label


def test_ood_traces_are_relabelled_and_ledgered(client, monkeypatch):
    from app import teacher

    calls = []

    def fake_teacher(messages, effort="low"):
        calls.append(messages)
        return {"action": {"tool": "file_appeal", "args": {}}, "model": "fake", "latency_ms": 1, "cost_usd": 0.001}

    monkeypatch.setattr(teacher, "call_teacher", fake_teacher)
    near = "Is physical therapy covered under my plan?"
    train = [{"id": f"d{i}", "source": "teacher-datagen", "run_id": "dg", "request_redacted": f"[session] {{}}\n\n{near}",
              "messages": _convo(near, "check_coverage", "teacher"), "outcome": {"task_success": True}} for i in range(2)]
    seen = {"id": "n1", "source": "traffic", "run_id": "tr", "request_redacted": f"[session] {{}}\n\n{near}",
            "messages": _convo(near, "check_coverage"), "outcome": {"task_success": True}}
    far = {"id": "f1", "source": "traffic", "run_id": "tr",
           "request_redacted": "[session] {}\n\nwhy was my ambulance bill rejected, appeal it",
           "messages": _convo("why was my ambulance bill rejected, appeal it", "get_claims"),
           "outcome": {"task_success": False}}
    medical = {**far, "id": "m1", "never_leave": True}
    client.post("/v1/traces", json={"traces": train + [seen, far, medical]})

    dry = client.post("/v1/ood/label", json={"run_ids": ["tr"], "dry_run": True}).json()
    assert dry["ood"] == 1 and dry["traces"][0]["id"] == "f1" and not calls  # medical never scored, nothing sent

    res = client.post("/v1/ood/label", json={"run_ids": ["tr"]}).json()
    assert res["steps_labelled"] == 2 and len(calls) == 2
    assert calls[0][-1]["kind"] == "request"  # prefix ends right before the student's step
    again = client.post("/v1/ood/label", json={"run_ids": ["tr"]}).json()
    assert again["steps_already_labelled"] == 2 and len(calls) == 2
    led = client.get("/v1/ledger").json()
    assert sum(c["reason"] == "ood_label" for c in led["calls"]) == 2 and led["stats"]["raw_pii_fields_sent"] == 0

    ds = client.post("/v1/datasets/build", json={"round": 9, "run_ids": ["tr"], "valid_frac": 0.0,
                                                 "include_ood": True}).json()
    assert ds["stats"]["by_source"] == {"ood": 2}  # failed-outcome trace still contributes its teacher labels
