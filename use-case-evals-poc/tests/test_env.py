import json

from harbor_env.agent import RULES, SCANNER, run_episode
from harbor_env.env import HarborEnv
from harbor_env.models import StepOut
from harbor_env.paths import SHARED
from harbor_env.prompts import parse_action
from harbor_env.router import Router, is_never_leave
from harbor_env.scoring import score_episode

M = "BM-10200"  # has APT-7001 (optometrist, 2026-11-10 13:30)


def test_reschedule_state_diff():
    env = HarborEnv(M)
    r = env.execute("reschedule_appointment", {"member_id": M, "appointment_id": "APT-7001", "new_date": "2026-11-13"})
    assert "error" not in r, r
    assert env.state_diff() == [{"type": "appointment_rescheduled", "appointment_id": "APT-7001",
                                 "date": "2026-11-13", "time": "13:30"}]


def test_wrong_member_is_an_error_and_no_write():
    env = HarborEnv(M)
    r = env.execute("get_policy", {"member_id": "BM-10207"})
    assert "error" in r and env.state_diff() == []


def test_router_policy_and_validators():
    env = HarborEnv(M)
    router = Router()
    big = {"tool": "file_claim", "args": {"member_id": M, "provider_id": "PRV-301", "service_date": "2026-11-02",
                                          "service": "dental_cleaning", "amount": 2400}}
    g = router.check(big, big["args"], env, [-0.01])
    assert "policy_claim_over_threshold" in g.reasons
    bad = {"tool": "check_coverage", "args": {"member_id": M, "service": "physical therapy"}}
    assert any(r.startswith("schema") for r in router.check(bad, bad["args"], env, None).reasons)
    assert router.check(None, None, env, None).reasons == ["invalid_json"]


def test_never_leave():
    assert is_never_leave("I was diagnosed with diabetes, is blood work covered?")
    assert not is_never_leave("Is physical therapy covered?")


def test_parse_action():
    assert parse_action('```json\n{"tool": "get_policy", "args": {"member_id": "x"}}\n```')["tool"] == "get_policy"
    assert parse_action('{"reply": "hi"} trailing') == {"reply": "hi"}
    assert parse_action("no json") is None


class Scripted:
    """Fake student that replays fixed actions, to test the loop without a model."""
    name, kind = "scripted", "local"

    def __init__(self, actions):
        self.actions = list(actions)

    def step(self, messages):
        a = self.actions.pop(0)
        return StepOut(action=a, raw=json.dumps(a), latency_s=0.01, logprobs=[-0.01])


def test_episode_is_redacted_and_scored():
    sc = {"id": "t", "member_id": M, "message": "Hi, it's Owen Novak. Move my eye exam to Friday.",
          "expected": {"required_calls": [{"tool": "reschedule_appointment", "args": {"appointment_id": "APT-7001",
                                                                                     "new_date": "2026-11-13"}}],
                       "state": [{"type": "appointment_rescheduled", "appointment_id": "APT-7001", "date": "2026-11-13"}]}}
    st = Scripted([{"tool": "reschedule_appointment", "args": {"member_id": "[MEMBER_ID_1]", "appointment_id": "APT-7001",
                                                               "new_date": "2026-11-13"}},
                   {"reply": "Done, [NAME_1]."}])
    ep = run_episode(sc, student=st, router=Router(), mode="hybrid")
    assert "Owen" not in ep["messages"][0]["content"] and "BM-10200" not in ep["messages"][0]["content"]
    assert not SCANNER.scan("\n".join(m["content"] for m in ep["messages"] if m["role"] == "user"))
    assert ep["final_reply"] == "Done, Owen." and not ep["left_device"]
    assert score_episode(sc, ep)["task_success"]


def test_redaction_vectors_match():
    from gradientharbor.privacy import Redactor

    vec = json.loads((SHARED / "redaction_vectors.json").read_text())
    for g in vec["groups"]:
        r = Redactor(RULES, g["known"])
        for c in g["cases"]:
            assert r.redact_text(c["input"]) == c["expected"]


def test_reply_grounding():
    from harbor_env.grounding import ungrounded

    ctx = ['<tool_result name="get_policy">{"deductible":1500,"deductible_met":750,"out_of_pocket_max":4000}</tool_result>',
           '<tool_result name="reschedule_appointment">{"new_datetime":"2026-11-13T13:30"}</tool_result>']
    assert ungrounded("Your deductible is $1,500 and you've met $750, so $750 is left.", ctx) == []
    assert ungrounded("Moved to Friday, November 13th at 1:30 PM.", ctx) == []
    assert ungrounded("Your deductible is $3,000.", ctx) == ["3000"]
