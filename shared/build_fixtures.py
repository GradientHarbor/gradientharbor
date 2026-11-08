"""Generate the mock insurer data in shared/fixtures/.

Deterministic (seeded), so the Python harness, the backend and the iOS app all
see identical data. Everything here is fictional: "Bayside Mutual" is a made-up
insurer, phone numbers use the reserved 555-01xx range, and names are random
combinations.

    python3 shared/build_fixtures.py
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20261108
SIM_TODAY = date(2026, 11, 9)  # a Monday; "Friday" in a request means 2026-11-13
OUT = Path(__file__).parent / "fixtures"

FIRST = ["Maya", "Ethan", "Priya", "Lucas", "Sofia", "Noah", "Amara", "Diego", "Hana", "Owen",
         "Leila", "Mateo", "Grace", "Ravi", "Chloe", "Samir", "Ingrid", "Tomas", "Yuki", "Elena",
         "Marcus", "Zoe", "Arjun", "Nadia", "Felix", "Imani", "Kenji", "Rosa", "Viktor", "Aisha",
         "Julian", "Mei", "Andre", "Talia", "Omar", "Freya", "Caleb", "Lina", "Rafael", "Wren"]
LAST = ["Okafor", "Lindqvist", "Ramirez", "Chen", "Haddad", "Novak", "Patel", "Brennan", "Sato",
        "Moreau", "Kowalski", "Adeyemi", "Fischer", "Castillo", "Nakamura", "Whitfield", "Duarte",
        "Ibrahim", "Larsen", "Quinn", "Varga", "Mbeki", "Rossi", "Tanaka", "Holloway", "Serrano",
        "Kaur", "Delacroix", "Abara", "Lund"]
STREETS = ["Harbor View Dr", "Marlin Way", "Cypress Ln", "Tidewater Ct", "Juniper St", "Pelican Ave",
           "Foghorn Rd", "Seacliff Blvd", "Kelp Forest Way", "Lighthouse Ln", "Driftwood St", "Gull Ct"]
CITIES = ["San Jose", "Oakland", "Palo Alto", "San Mateo", "Fremont"]
ZIP = {"San Jose": "95112", "Oakland": "94607", "Palo Alto": "94301", "San Mateo": "94401", "Fremont": "94536"}

PLANS = {
    "bronze": {"deductible": 3000, "oop_max": 8000},
    "silver": {"deductible": 1500, "oop_max": 6000},
    "gold": {"deductible": 500, "oop_max": 4000},
}

# service -> per-plan coverage. copay in dollars, coinsurance as member share after deductible.
SERVICES = {
    "primary_care":      {"label": "Primary care visit", "bronze": (True, 40, 0.0), "silver": (True, 25, 0.0), "gold": (True, 10, 0.0)},
    "specialist":        {"label": "Specialist visit", "bronze": (True, 80, 0.0), "silver": (True, 50, 0.0), "gold": (True, 30, 0.0), "referral": True},
    "dermatology":       {"label": "Dermatology visit", "bronze": (True, 80, 0.0), "silver": (True, 50, 0.0), "gold": (True, 30, 0.0), "referral": True},
    "dental_cleaning":   {"label": "Dental cleaning", "bronze": (False, 0, 0.0), "silver": (True, 0, 0.0), "gold": (True, 0, 0.0), "limit": 2},
    "dental_filling":    {"label": "Dental filling", "bronze": (False, 0, 0.0), "silver": (True, 0, 0.3), "gold": (True, 0, 0.2)},
    "physical_therapy":  {"label": "Physical therapy", "bronze": (True, 60, 0.0), "silver": (True, 40, 0.0), "gold": (True, 20, 0.0), "limit": 20},
    "mental_health":     {"label": "Mental health therapy", "bronze": (True, 40, 0.0), "silver": (True, 25, 0.0), "gold": (True, 10, 0.0)},
    "chiropractic":      {"label": "Chiropractic care", "bronze": (False, 0, 0.0), "silver": (True, 45, 0.0), "gold": (True, 25, 0.0), "limit": 12},
    "vision_exam":       {"label": "Eye exam", "bronze": (True, 0, 0.0), "silver": (True, 0, 0.0), "gold": (True, 0, 0.0), "limit": 1},
    "urgent_care":       {"label": "Urgent care", "bronze": (True, 100, 0.0), "silver": (True, 75, 0.0), "gold": (True, 50, 0.0)},
    "mri":               {"label": "MRI / advanced imaging", "bronze": (True, 0, 0.4), "silver": (True, 0, 0.3), "gold": (True, 0, 0.1), "prior_auth": True},
    "acupuncture":       {"label": "Acupuncture", "bronze": (False, 0, 0.0), "silver": (False, 0, 0.0), "gold": (True, 30, 0.0), "limit": 10},
    "telehealth":        {"label": "Telehealth visit", "bronze": (True, 0, 0.0), "silver": (True, 0, 0.0), "gold": (True, 0, 0.0)},
    "lab_work":          {"label": "Lab work", "bronze": (True, 0, 0.3), "silver": (True, 0, 0.2), "gold": (True, 0, 0.0)},
}

# provider specialty -> service category it bills under
SPECIALTIES = {
    "dentist": "dental_cleaning", "dermatologist": "dermatology", "physical_therapist": "physical_therapy",
    "primary_care": "primary_care", "cardiologist": "specialist", "optometrist": "vision_exam",
    "chiropractor": "chiropractic", "therapist": "mental_health", "orthopedist": "specialist",
    "acupuncturist": "acupuncture",
}
PROVIDER_LAST = ["Alvarez", "Bergstrom", "Coleman", "Dimitrov", "Esposito", "Farouk", "Gallagher", "Hoang",
                 "Iverson", "Jaramillo", "Kimura", "Lefevre", "Mendez", "Nilsson", "Oduya", "Petrov",
                 "Quintero", "Reyes", "Sorensen", "Thornton", "Ueda", "Valdez", "Weiss", "Xu", "Yilmaz",
                 "Zielinski", "Ashby", "Bhatt", "Cruz", "Dahl"]
CLINIC = {"dentist": "Dental", "dermatologist": "Dermatology", "physical_therapist": "Physical Therapy",
          "primary_care": "Family Medicine", "cardiologist": "Heart Center", "optometrist": "Eye Care",
          "chiropractor": "Chiropractic", "therapist": "Counseling", "orthopedist": "Orthopedics",
          "acupuncturist": "Acupuncture"}
PROVIDER_FIRST = ["Adrienne", "Bernard", "Celeste", "Desmond", "Evangeline", "Fletcher", "Georgina", "Horace",
                  "Isadora", "Jasper", "Katarina", "Leopold", "Marisol", "Nathaniel", "Ophelia", "Percival",
                  "Rosalind", "Sebastian", "Theodora", "Ulysses", "Vivienne", "Winston"]
DENIAL_REASONS = ["Service not covered under your plan", "Out-of-network provider without prior authorization",
                  "Missing referral from primary care provider", "Annual visit limit reached",
                  "Duplicate submission"]


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # Providers: 6 per specialty across cities (60 total).
    providers = []
    pid = 301
    last_pool = PROVIDER_LAST[:]
    rng.shuffle(last_pool)
    for spec in SPECIALTIES:
        for i in range(6):
            city = CITIES[(i + pid) % len(CITIES)]
            last = last_pool[(pid - 301) % len(last_pool)]
            first = rng.choice(PROVIDER_FIRST)
            network = sorted(rng.sample(list(PLANS), k=rng.choice([1, 2, 3, 3, 3])))
            providers.append({
                "provider_id": f"PRV-{pid}",
                "name": f"Dr. {first} {last}",
                "clinic": f"{city} {CLINIC[spec]}",
                "specialty": spec,
                "service": SPECIALTIES[spec],
                "city": city,
                "in_network_plans": network,
                "accepting_new_patients": rng.random() > 0.15,
                "working_days": sorted(rng.sample([0, 1, 2, 3, 4], k=rng.choice([3, 4, 5]))),  # Mon=0
                "hours": {"start": "09:00", "end": "17:00"},
            })
            pid += 1

    # Members + policies (40).
    members, policies = [], []
    used = set()
    for i in range(40):
        while True:
            first, last = rng.choice(FIRST), rng.choice(LAST)
            if (first, last) not in used:
                used.add((first, last))
                break
        plan = ["bronze", "silver", "gold"][i % 3]
        city = CITIES[i % len(CITIES)]
        dob = date(rng.randint(1958, 2002), rng.randint(1, 12), rng.randint(1, 28))
        mid, pol = f"BM-{10200 + i * 7}", f"POL-{55000 + i * 11}"
        members.append({
            "member_id": mid,
            "name": f"{first} {last}",
            "first_name": first,
            "dob": dob.isoformat(),
            "phone": f"(408) 555-01{i:02d}",
            "email": f"{first.lower()}.{last.lower()}@example.com",
            "address": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}, {city}, CA {ZIP[city]}",
            "city": city,
            "plan": plan,
            "policy_id": pol,
            "primary_care_provider_id": None,
        })
        ded = PLANS[plan]["deductible"]
        met = rng.choice([0, 0.25, 0.5, 0.8, 1.0]) * ded
        policies.append({
            "policy_id": pol,
            "member_id": mid,
            "plan": plan,
            "plan_name": f"Bayside {plan.title()} PPO",
            "status": "active",
            "effective_date": "2026-01-01",
            "renewal_date": "2027-01-01",
            "deductible": ded,
            "deductible_met": round(met, 2),
            "out_of_pocket_max": PLANS[plan]["oop_max"],
            "out_of_pocket_spent": round(met + rng.randint(0, 600), 2),
            "dependents": rng.choice([0, 0, 1, 2, 3]),
        })

    pcps = [p for p in providers if p["specialty"] == "primary_care"]
    for m in members:
        m["primary_care_provider_id"] = rng.choice(pcps)["provider_id"]

    coverage = {}
    for plan in PLANS:
        coverage[plan] = {}
        for svc, spec in SERVICES.items():
            covered, copay, coins = spec[plan]
            coverage[plan][svc] = {
                "label": spec["label"],
                "covered": covered,
                "copay": copay,
                "coinsurance": coins,
                "applies_to_deductible": coins > 0,
                "requires_referral": bool(spec.get("referral")),
                "requires_prior_auth": bool(spec.get("prior_auth")),
                "visit_limit_per_year": spec.get("limit"),
                "out_of_network": "not covered" if plan == "bronze" else "covered at 50% after deductible",
            }

    # Claims (~80).
    claims = []
    cid = 88000
    for m in members:
        for _ in range(rng.choice([1, 2, 2, 3])):
            prov = rng.choice(providers)
            svc_date = SIM_TODAY - timedelta(days=rng.randint(5, 120))
            billed = float(rng.choice([95, 140, 180, 220, 260, 320, 450, 780, 1250, 2400]))
            status = rng.choices(["submitted", "in_review", "approved", "paid", "denied"], [2, 3, 2, 4, 2])[0]
            covered_amt = 0.0 if status == "denied" else round(billed * rng.choice([0.6, 0.7, 0.8, 0.9]), 2)
            claims.append({
                "claim_id": f"CLM-{cid}",
                "member_id": m["member_id"],
                "provider_id": prov["provider_id"],
                "provider_name": prov["name"],
                "service": prov["service"],
                "service_date": svc_date.isoformat(),
                "submitted_date": (svc_date + timedelta(days=rng.randint(1, 6))).isoformat(),
                "amount_billed": billed,
                "amount_covered": covered_amt if status in ("approved", "paid") else None,
                "status": status,
                "denial_reason": rng.choice(DENIAL_REASONS) if status == "denied" else None,
                "last_update": (SIM_TODAY - timedelta(days=rng.randint(0, 4))).isoformat(),
                "expected_decision_by": (SIM_TODAY + timedelta(days=rng.randint(3, 14))).isoformat()
                if status in ("submitted", "in_review") else None,
            })
            cid += 13

    # Appointments (~50 upcoming, on provider working days).
    appointments = []
    aid = 7001
    for m in members:
        for _ in range(rng.choice([0, 1, 1, 2])):
            prov = rng.choice(providers)
            for _try in range(30):
                d = SIM_TODAY + timedelta(days=rng.randint(1, 13))
                if d.weekday() in prov["working_days"]:
                    break
            hh = rng.choice([9, 10, 11, 13, 14, 15, 16])
            mm = rng.choice([0, 30])
            appointments.append({
                "appointment_id": f"APT-{aid}",
                "member_id": m["member_id"],
                "provider_id": prov["provider_id"],
                "provider_name": prov["name"],
                "specialty": prov["specialty"],
                "datetime": datetime(d.year, d.month, d.day, hh, mm).isoformat(timespec="minutes"),
                "status": "scheduled",
                "reason": {"dentist": "Cleaning", "dermatologist": "Skin check", "physical_therapist": "PT session",
                           "primary_care": "Annual physical", "cardiologist": "Follow-up", "optometrist": "Eye exam",
                           "chiropractor": "Adjustment", "therapist": "Session", "orthopedist": "Knee follow-up",
                           "acupuncturist": "Treatment"}[prov["specialty"]],
            })
            aid += 1

    meta = {
        "insurer": "Bayside Mutual",
        "sim_today": SIM_TODAY.isoformat(),
        "sim_weekday": SIM_TODAY.strftime("%A"),
        "slot_minutes": 30,
        "claim_review_threshold": 1000,
        "seed": SEED,
        "note": "All data is fictional and generated by shared/build_fixtures.py.",
    }

    for name, data in [("meta", meta), ("members", members), ("policies", policies), ("coverage", coverage),
                       ("providers", providers), ("claims", claims), ("appointments", appointments)]:
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"members={len(members)} providers={len(providers)} claims={len(claims)} appointments={len(appointments)}")


if __name__ == "__main__":
    main()
