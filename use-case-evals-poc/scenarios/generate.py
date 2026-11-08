"""Generate member requests with programmatic ground truth, then paraphrase them with Sonnet.

Each scenario = {id, split, category, member_id, message, template, expected}. `expected` is
derived from the template and, for write tools, by executing the gold calls on a fresh env,
so scoring is against real ground truth, not just agreement with the teacher.

    uv run python scenarios/generate.py            # all splits, paraphrased
    uv run python scenarios/generate.py --no-llm   # templates only (offline)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor_env.agent import write_jsonl  # noqa: E402
from harbor_env.env import HarborEnv, load_fixtures  # noqa: E402
from harbor_env.paths import ROOT, load_env  # noqa: E402

OUT = ROOT / "scenarios" / "splits"
FX = load_fixtures()
TODAY = date.fromisoformat(FX["meta"]["sim_today"])

SERVICE_PHRASES = {
    "primary_care": ["a checkup with my regular doctor", "a primary care visit", "an annual physical"],
    "specialist": ["seeing a cardiologist", "a specialist visit", "an orthopedist appointment"],
    "dermatology": ["a dermatologist visit", "a skin check", "seeing a dermatologist about a rash"],
    "dental_cleaning": ["a teeth cleaning", "a dental cleaning"],
    "dental_filling": ["a cavity filling", "getting a filling"],
    "physical_therapy": ["physical therapy", "PT sessions", "physio for my shoulder"],
    "mental_health": ["therapy with a counselor", "talk therapy", "seeing a therapist"],
    "chiropractic": ["a chiropractor", "chiropractic adjustments"],
    "vision_exam": ["an eye exam", "a vision checkup"],
    "urgent_care": ["urgent care", "an urgent care visit"],
    "mri": ["an MRI", "an MRI scan of my knee"],
    "acupuncture": ["acupuncture", "acupuncture sessions"],
    "telehealth": ["a video visit with a doctor", "telehealth appointments"],
    "lab_work": ["blood work", "lab tests"],
}
SPEC_PHRASES = {
    "dentist": ["dentist"], "dermatologist": ["dermatologist", "skin doctor"],
    "physical_therapist": ["physical therapist", "PT"], "primary_care": ["primary care doctor", "family doctor"],
    "cardiologist": ["cardiologist", "heart doctor"], "optometrist": ["optometrist", "eye doctor"],
    "chiropractor": ["chiropractor"], "therapist": ["therapist", "counselor"],
    "orthopedist": ["orthopedist", "orthopedic doctor"], "acupuncturist": ["acupuncturist"],
}
PERSONAS = ["terse, types in lowercase with no punctuation", "friendly and chatty, adds a bit of context",
            "polite older member, formal full sentences", "busy parent, quick and slightly stressed",
            "non-native English speaker, simple phrasing", "frustrated but civil", "uses casual texting style",
            "precise and businesslike"]


def h_date(d: date) -> str:
    return d.strftime("%B ") + str(d.day)


def time12(t: str) -> str:
    h, m = map(int, t.split(":"))
    return f"{(h - 1) % 12 + 1}{':%02d' % m if m else ''} {'am' if h < 12 else 'pm'}"


def date_phrase(d: date, rng: random.Random) -> tuple[str, str]:
    """Returns (phrase, keep-string). Days later this week are named by weekday only."""
    if d - TODAY <= timedelta(days=4) and d.weekday() > TODAY.weekday():
        w = d.strftime("%A")
        return rng.choice([w, f"this {w}"]), w
    s = rng.choice([f"{d.strftime('%A')}, {h_date(d)}", h_date(d), f"{d.strftime('%b')} {d.day}"])
    return s, s


def member_ctx(m):
    env = HarborEnv(m["member_id"])
    return env, env.session_context()


def gold_state(member_id, calls):
    env = HarborEnv(member_id)
    for c in calls:
        r = env.execute(c["tool"], {"member_id": member_id, **c["args"]})
        if "error" in r:
            return None
    return env.state_diff()


def providers_for(m, specialty=None, city=None, accepting=True):
    return [p for p in FX["providers"] if (specialty is None or p["specialty"] == specialty)
            and p["city"] == (city or m["city"]) and m["plan"] in p["in_network_plans"]
            and (p["accepting_new_patients"] or not accepting)]


def working_days_ahead(p, rng, span=14, exclude=None):
    days = [TODAY + timedelta(days=i) for i in range(1, span + 1)]
    days = [d for d in days if d.weekday() in p["working_days"] and d != exclude]
    return rng.choice(days) if days else None


# ---- category builders: return (template_text, keep_strings, expected) or None ------------------

def c_coverage(m, ctx, rng):
    s = rng.choice(list(SERVICE_PHRASES))
    ph = rng.choice(SERVICE_PHRASES[s])
    t = rng.choice(["Is {p} covered on my plan?", "How much will {p} cost me?",
                    "Do I need a referral for {p}, and is it covered?", "Does my insurance pay for {p}?",
                    "What's my copay for {p}?"]).format(p=ph)
    cov = FX["coverage"][m["plan"]][s]
    return t, [], {"required_calls": [{"tool": "check_coverage", "args": {"service": s}}], "state": [],
                   "reply_facts": {k: cov[k] for k in ["covered", "copay", "coinsurance", "visit_limit_per_year",
                                                        "requires_referral"]}}


def c_plan_info(m, ctx, rng):
    t = rng.choice(["How much of my deductible have I met so far?", "How close am I to my out-of-pocket max?",
                    "When does my plan renew?", "What plan am I on, and what's my deductible?",
                    "How many dependents are on my policy?", "Is my policy active?"])
    pol = next(p for p in FX["policies"] if p["member_id"] == m["member_id"])
    return t, [], {"required_calls": [{"tool": "get_policy", "args": {}}], "state": [],
                   "reply_facts": {k: pol[k] for k in ["plan_name", "deductible", "deductible_met", "out_of_pocket_max",
                                                        "out_of_pocket_spent", "renewal_date", "dependents"]}}


def _claims(m, denied):
    rc = [c for c in FX["claims"] if c["member_id"] == m["member_id"]]
    rc.sort(key=lambda c: c["service_date"], reverse=True)
    return [c for c in rc[:5] if (c["status"] == "denied") == denied]


def c_claim_status(m, ctx, rng):
    cs = _claims(m, denied=False)
    if not cs:
        return None
    c = rng.choice(cs)
    d = date.fromisoformat(c["service_date"])
    form = rng.randrange(3)
    if form == 0:
        t, keep = f"Any update on my claim from my visit with {c['provider_name']} on {h_date(d)}?", [c["provider_name"]]
    elif form == 1:
        t, keep = f"What's the status of claim {c['claim_id']}?", [c["claim_id"]]
    else:
        t, keep = f"Has my claim for the {h_date(d)} appointment with {c['provider_name']} been paid yet?", [c["provider_name"]]
    return t, keep, {"required_calls": [{"tool": "get_claim_status", "args": {"claim_id": c["claim_id"]}}], "state": [],
                     "reply_facts": {k: c[k] for k in ["status", "amount_billed", "amount_covered", "expected_decision_by"]}}


def c_claim_denied(m, ctx, rng):
    cs = _claims(m, denied=True)
    if not cs:
        return None
    c = rng.choice(cs)
    t = rng.choice([f"Why was my claim for the visit with {c['provider_name']} denied?",
                    f"My claim with {c['provider_name']} says denied. What happened?"])
    return t, [c["provider_name"]], {"required_calls": [{"tool": "get_claim_status", "args": {"claim_id": c["claim_id"]}}],
                                     "state": [], "reply_facts": {"denial_reason": c["denial_reason"]}}


def c_find_provider(m, ctx, rng):
    sp = rng.choice(list(SPEC_PHRASES))
    ph = rng.choice(SPEC_PHRASES[sp])
    if rng.random() < 0.6:
        city, t = m["city"], rng.choice(["Can you find me an in-network {p} near me?",
                                         "I need a {p} close to home that takes my insurance.",
                                         "Who's a {p} near me that's in my network?"]).format(p=ph)
        keep = []
    else:
        city = rng.choice([c for c in ["San Jose", "Oakland", "Palo Alto", "San Mateo", "Fremont"] if c != m["city"]])
        t = rng.choice(["I need a {p} in {c}. Who's in my network there?", "Any in-network {p} in {c}?"]).format(p=ph, c=city)
        keep = [city]
    found = [p["name"] for p in FX["providers"] if p["specialty"] == sp and p["city"] == city
             and m["plan"] in p["in_network_plans"]]
    return t, keep, {"required_calls": [{"tool": "find_provider", "args": {"specialty": sp, "city": city}}], "state": [],
                     "reply_facts": {"in_network_providers": found}}


def c_book(m, ctx, rng):
    cands = providers_for(m)
    rng.shuffle(cands)
    for p in cands:
        d = working_days_ahead(p, rng)
        if not d:
            continue
        env = HarborEnv(m["member_id"])
        free = env._free_times(p, d)
        if not free:
            continue
        tm = rng.choice(free)
        dp, dkeep = date_phrase(d, rng)
        ph = rng.choice(SPEC_PHRASES[p["specialty"]])
        t = rng.choice([f"Please book me with {p['name']}, the {ph}, on {dp} at {time12(tm)}.",
                        f"Can I get an appointment with {p['name']} ({ph}) {dp} at {time12(tm)}?"])
        calls = [{"tool": "book_appointment", "args": {"provider_id": p["provider_id"], "date": d.isoformat(), "time": tm}}]
        st = gold_state(m["member_id"], calls)
        if st is None:
            continue
        return t, [p["name"], dkeep, time12(tm)], {"required_calls": calls, "state": st}
    return None


def c_reschedule(m, ctx, rng):
    appts = ctx["upcoming_appointments"]
    if not appts:
        return None
    a = rng.choice(appts)
    full = next(x for x in FX["appointments"] if x["appointment_id"] == a["id"])
    p = next(x for x in FX["providers"] if x["provider_id"] == full["provider_id"])
    cur = date.fromisoformat(a["datetime"][:10])
    d = working_days_ahead(p, rng, exclude=cur)
    if not d:
        return None
    dp, dkeep = date_phrase(d, rng)
    ph = rng.choice(SPEC_PHRASES[p["specialty"]])
    who = f"{ph} appointment" if len({x["specialty"] for x in appts}) == len(appts) else f"appointment with {p['name']}"
    keep = [dkeep] + ([p["name"]] if "with" in who else [])
    args = {"appointment_id": a["id"], "new_date": d.isoformat()}
    if rng.random() < 0.35:
        free = HarborEnv(m["member_id"])._free_times(p, d, ignore_appt=a["id"])
        tm = rng.choice(free)
        args["new_time"] = tm
        t = f"Can you move my {who} to {dp} at {time12(tm)}?"
        keep.append(time12(tm))
    else:
        t = rng.choice([f"Can you move my {who} to {dp}?", f"I need to reschedule my {who} to {dp}, same time if possible.",
                        f"Something came up, please push my {who} to {dp}."])
    calls = [{"tool": "reschedule_appointment", "args": args}]
    st = gold_state(m["member_id"], calls)
    if st is None:
        return None
    req = {"tool": "reschedule_appointment", "args": {k: v for k, v in args.items()}}
    return t, keep, {"required_calls": [req], "state": st}


def c_file_claim(m, ctx, rng, big=False):
    cands = providers_for(m, accepting=False)
    if not cands:
        return None
    p = rng.choice(cands)
    d = TODAY - timedelta(days=rng.randint(2, 45))
    amt = rng.choice([1250, 1480, 2200, 2875, 3600]) if big else rng.choice([85, 120, 145, 180, 240, 310, 450, 620, 875])
    ph = rng.choice(SPEC_PHRASES[p["specialty"]])
    amt_s = f"${amt:,}"
    t = rng.choice([f"I'd like to file a claim for my visit with {p['name']}, my {ph}, on {h_date(d)}. The bill was {amt_s}.",
                    f"Please submit a claim: {ph} visit with {p['name']} on {h_date(d)}, {amt_s}."])
    calls = [{"tool": "file_claim", "args": {"provider_id": p["provider_id"], "service_date": d.isoformat(),
                                              "service": p["service"], "amount": amt}}]
    st = gold_state(m["member_id"], calls)
    return t, [p["name"], h_date(d), amt_s], {"required_calls": calls, "state": st, "policy_review": big}


def c_big_claim(m, ctx, rng):
    return c_file_claim(m, ctx, rng, big=True)


def c_human(m, ctx, rng):
    t, topic = rng.choice([("I want to talk to a real person about a charge on my bill.", "billing"),
                           ("I'd like to file a complaint about how my last call was handled.", "complaint"),
                           ("Can I speak to a human agent please?", "*"),
                           ("I was billed twice for the same visit and want someone to fix it.", "billing")])
    return t, [], {"required_calls": [{"tool": "escalate_to_human", "args": {"topic": topic}}],
                   "state": [{"type": "ticket", "topic": topic}]}


def c_clarify(m, ctx, rng):
    if len(ctx["upcoming_appointments"]) >= 2 and rng.random() < 0.6:
        t = "Can you move my appointment to Friday?"
    else:
        t = rng.choice(["I want to file a claim for my recent visit.", "Please book me an appointment."])
    return t, [], {"required_calls": [], "state": [], "first_call": None, "clarify": True}


CONDITIONS = [("depression", "mental_health"), ("diabetes", "lab_work"), ("a torn ACL", "physical_therapy"),
              ("anxiety", "mental_health"), ("a herniated disc", "chiropractic"), ("high blood pressure", "specialist")]
MEDS = ["Ozempic", "metformin", "Lexapro", "insulin", "Zoloft"]


def c_never_leave(m, ctx, rng):
    if rng.random() < 0.6:
        cond, s = rng.choice(CONDITIONS)
        ph = rng.choice(SERVICE_PHRASES[s])
        t = rng.choice([f"I was diagnosed with {cond} last month. Is {ph} covered?",
                        f"My doctor says my {cond} diagnosis means I need {ph}. How much will it cost me?"])
        req = [{"any_of": [{"tool": "check_coverage", "args": {"service": s}},
                           {"tool": "escalate_to_human", "args": {"topic": "medical"}}]}]
        return t, [cond], {"required_calls": req, "state_any_of": [[], [{"type": "ticket", "topic": "medical"}]]}
    med = rng.choice(MEDS)
    t = rng.choice([f"My doctor prescribed {med}. Will my plan pay for it?", f"Is my {med} prescription covered?"])
    return t, [med], {"required_calls": [{"tool": "escalate_to_human", "args": {"topic": "medical"}}],
                      "state": [{"type": "ticket", "topic": "medical"}]}


def n_appeal(m, ctx, rng):
    cs = _claims(m, denied=True) or [c for c in FX["claims"] if c["member_id"] == m["member_id"] and c["status"] == "denied"]
    if not cs:
        return None
    c = rng.choice(cs)
    t = f"My claim with {c['provider_name']} was denied and I think that's wrong. I want to appeal it."
    return t, [c["provider_name"]], {"required_calls": [{"tool": "escalate_to_human", "args": {"topic": "claim_appeal"}}],
                                     "state": [{"type": "ticket", "topic": "claim_appeal"}]}


def n_move_and_check(m, ctx, rng):
    r = c_reschedule(m, ctx, rng)
    if not r:
        return None
    t, keep, exp = r
    a_id = exp["required_calls"][0]["args"]["appointment_id"]
    full = next(x for x in FX["appointments"] if x["appointment_id"] == a_id)
    t = t.rstrip("?.") + f", and make sure {full['provider_name']} is still in my network."
    exp["required_calls"].append({"any_of": [{"tool": "find_provider", "args": {"specialty": full["specialty"]}},
                                             {"tool": "check_coverage", "args": {}}]})
    return t, keep + [full["provider_name"]], exp


def n_find_and_book(m, ctx, rng):
    sps = list(SPEC_PHRASES)
    rng.shuffle(sps)
    for sp in sps:
        cands = providers_for(m, sp)
        if not cands:
            continue
        for _ in range(10):
            d = TODAY + timedelta(days=rng.randint(2, 12))
            if any(d.weekday() in p["working_days"] for p in cands):
                break
        else:
            continue
        dp, dkeep = date_phrase(d, rng)
        ph = rng.choice(SPEC_PHRASES[sp])
        t = f"Find me an in-network {ph} near me and book the first opening on {dp}."
        return t, [dkeep], {"required_calls": [{"tool": "find_provider", "args": {"specialty": sp}},
                                               {"tool": "book_appointment", "args": {"provider_id": "*", "date": d.isoformat()}}],
                            "state": [{"type": "appointment_booked", "provider_id": "*", "date": d.isoformat(), "time": "*"}]}
    return None


def n_compare(m, ctx, rng):
    a, b = rng.sample(["chiropractic", "physical_therapy", "acupuncture", "mental_health", "telehealth"], 2)
    pa, pb = SERVICE_PHRASES[a][0], SERVICE_PHRASES[b][0]
    t = f"Which would cost me less out of pocket, {pa} or {pb}?"
    return t, [], {"required_calls": [{"tool": "check_coverage", "args": {"service": a}},
                                      {"tool": "check_coverage", "args": {"service": b}}], "state": []}


def n_cancel(m, ctx, rng):
    if not ctx["upcoming_appointments"]:
        return None
    a = rng.choice(ctx["upcoming_appointments"])
    t = f"Please cancel my {a['specialty'].replace('_', ' ')} appointment, I don't need it anymore."
    return t, [], {"required_calls": [], "forbid_tools": ["reschedule_appointment", "book_appointment"],
                   "state_any_of": [[], [{"type": "ticket", "topic": "*"}]], "first_call": {"any_of": [
                       {"tool": "escalate_to_human", "args": {}}]}}


STANDARD = [(c_coverage, 22), (c_plan_info, 8), (c_claim_status, 12), (c_claim_denied, 5), (c_find_provider, 12),
            (c_book, 10), (c_reschedule, 12), (c_file_claim, 9), (c_human, 4), (c_clarify, 3), (c_never_leave, 3)]
NOVEL = [n_appeal, n_move_and_check, n_find_and_book, n_compare, n_cancel]


def build(split, members, n, builders, rng, seen):
    out = []
    fns, weights = zip(*builders) if isinstance(builders[0], tuple) else (builders, [1] * len(builders))
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        fn = rng.choices(fns, weights)[0]
        m = rng.choice(members)
        env, ctx = member_ctx(m)
        r = fn(m, ctx, rng)
        if not r:
            continue
        text, keep, exp = r
        if "state" in exp and exp["state"] is None:
            continue
        key = (fn.__name__, m["member_id"], text)
        if key in seen:
            continue
        seen.add(key)
        cat = fn.__name__.split("_", 1)[1] if "_" in fn.__name__ else split
        if cat == "file_claim" and exp.get("policy_review"):
            cat = "policy"
        # Some requests carry PII in the text, to exercise redaction.
        roll = rng.random()
        if roll < 0.12:
            text = f"Hi, this is {m['name']}. " + text
            keep = keep + [m["name"]]
        elif roll < 0.18:
            text = text + f" You can reach me at {m['phone']}."
            keep = keep + [m["phone"]]
        out.append({"id": f"{split}-{len(out):04d}", "split": split, "category": cat, "member_id": m["member_id"],
                    "template": text, "message": text, "keep": keep, "persona": rng.choice(PERSONAS), "expected": exp})
    return out


def paraphrase(rows, rng):
    import anthropic

    client = anthropic.Anthropic(max_retries=4)
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["id", "text"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
    kept = 0
    for i in range(0, len(rows), 25):
        batch = rows[i:i + 25]
        items = [{"id": r["id"], "text": r["template"], "persona": r["persona"], "must_keep_verbatim": r["keep"]}
                 for r in batch]
        prompt = ("Rewrite each insurance-app support chat message as that member would actually type it, in the "
                  "given persona. Keep the meaning and every detail (dates, times, amounts, names, IDs, cities). "
                  "Every string in must_keep_verbatim must appear exactly as written. One message each, no "
                  "greetings unless already present, no extra requests.\n\n" + json.dumps(items, ensure_ascii=False))
        try:
            resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=8000,
                                          messages=[{"role": "user", "content": prompt}],
                                          output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
            text = next(b.text for b in resp.content if b.type == "text")
            got = {x["id"]: x["text"] for x in json.loads(text)["items"]}
        except Exception as e:  # keep templates for this batch
            print(f"paraphrase batch {i} failed: {type(e).__name__}", file=sys.stderr)
            continue
        for r in batch:
            new = got.get(r["id"], "").strip()
            if new and all(k.lower() in new.lower() for k in r["keep"]):
                r["message"] = new
                kept += 1
        print(f"paraphrased {min(i + 25, len(rows))}/{len(rows)}", file=sys.stderr)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    load_env()
    rng = random.Random(args.seed)
    members = FX["members"]
    train_m, held_m = members[:30], members[30:]
    seen: set = set()
    splits = {
        "train": build("train", train_m, 500, STANDARD, rng, seen),
        "traffic": build("traffic", train_m, 150, STANDARD, rng, seen),
        "heldout": build("heldout", held_m, 100, [b for b in STANDARD if b[0] is not c_never_leave], rng, seen),
        "novel": build("novel", held_m, 30, NOVEL, rng, seen),
        "never_leave": build("never_leave", held_m, 15, [c_never_leave], rng, seen),
        "policy": build("policy", held_m, 10, [c_big_claim], rng, seen),
    }
    for r in splits["policy"]:
        r["category"] = "policy"
    for name, rows in splits.items():
        if not args.no_llm:
            k = paraphrase(rows, rng)
            print(f"{name}: {k}/{len(rows)} paraphrased", file=sys.stderr)
        write_jsonl(OUT / f"{name}.jsonl", rows)
        cats = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(name, len(rows), cats)


if __name__ == "__main__":
    main()
