# shared: the contract between the evals, the backend and the iOS app

| File | Used by |
| --- | --- |
| `tools.json` | 8 tool schemas (long description for the teacher/base models, `short` for the fine-tuned student) |
| `fixtures/*.json` | Fictional Bayside Mutual data: 40 members, 60 providers, 85 claims, 37 appointments. Regenerate with `python3 build_fixtures.py` (seeded). Sim date: Mon 2026-11-09. |
| `prompts/` | `system_core.txt` (all models), `action_format.txt` (on-device models) |
| `action_format.md` | One JSON action per step; conversation layout |
| `redaction.json` | Placeholder rules, known fields, patterns, fail-closed scan config |
| `redaction_vectors.json` | Golden cases: the Python and Swift redactors must match them exactly |
| `router.json` | Max steps, confidence thresholds (tuned by `use-case-evals-poc/runs/tune_router.py`), validators, policy ($1,000 claims), never-leave regex, shadow rate, OOD |

Changing any file here changes what every component does, so rerun the evals after editing.
