# Bayside — iOS demo app requirements

The demo app for GradientHarbor: a member app for a fictional insurer, **Bayside Mutual**. Its support agent runs on the iPhone (Gemma 4 student, MLX), escalates to the cloud teacher only when the router says so, and shows every cloud call in a privacy ledger. It must work in airplane mode.

The app is the iOS port of the simulated environment in `use-case-evals-poc/`. Both read the same files in `shared/`. The rule throughout: **the phone runs exactly what the evals measured**: same weights, prompt, action format, tools, router thresholds and redaction rules.

Target device: iPhone Air (A19 Pro, 12 GB RAM), iOS 26+. Simulator support for UI work; model inference only on the device.

---

## 1. Screens

| Screen | Content | Updates when the agent… |
| --- | --- | --- |
| **Home** | Greeting, plan card (plan name, deductible progress bar), next appointment, latest claim status, "Ask Bayside" button | books/moves an appointment, files a claim |
| **Policy** | Plan, status, dates, deductible / out-of-pocket progress, dependents, coverage table by service (covered, copay, coinsurance, limits) | — (read only) |
| **Claims** | List (status chip, provider, date, billed/covered); detail with denial reason and expected decision date | files a claim (new row animates in) |
| **Appointments** | Upcoming list grouped by day; detail sheet | books or reschedules (row moves with an animation, toast "Moved by Bayside assistant") |
| **Support chat** | The agent conversation (see §2) | — |
| **Privacy ledger** | Every cloud call: time, gate that fired, exact redacted text sent, response summary; filter "today"; header counters: requests handled on device / sent / blocked | any escalation or shadow call |
| **Settings** | Model round + size + version hash, "Check for model update", sync opt-in toggle (default **off**), shadow rate (1% default, demo 15%), backend URL, **Eval mode** (debug), reset demo data | — |

The demo member is pre-selected (`BM-10200`, from `shared/fixtures/members.json`). A hidden long-press on the avatar switches the member, for rehearsing different data.

## 2. Agent capabilities (Support chat)

### 2.1 Tools
Exactly the 8 tools in `shared/tools.json`, executed locally over the bundled fixtures (a mutable copy in the app's container, resettable):

| Tool | Local behavior | Confirmation |
| --- | --- | --- |
| `get_policy` | read policy | — |
| `check_coverage` | read coverage table for the member's plan | — |
| `get_claim_status` | one claim or recent 5 | — |
| `find_provider` | filter by specialty / city / network; max 5 | — |
| `book_appointment` | slot rules: 30-min slots 09:00–16:30 on provider working days, after today, ≤ 60 days | **confirmation card** |
| `reschedule_appointment` | keep time if free, else earliest free slot | **confirmation card** |
| `file_claim` | new claim, status `submitted`, decision in 10 days | **confirmation card**, shows amount |
| `escalate_to_human` | creates a support ticket (local, the insurer's own system; not a cloud model) | — |

Tool semantics, error payloads (for example `available_times` when a slot is taken) and the session-context JSON must match `use-case-evals-poc/harbor_env/env.py`. The parity test in §7 enforces that.

**Confirmation cards.** Write tools pause the loop and show a card ("Move eye exam with Dr. Desmond Weiss to Fri, Nov 13, 1:30 PM?" [Confirm] [Change]). In eval mode cards auto-confirm.

### 2.2 Loop
- Max **4 tool steps** per request, then the agent must reply (`shared/router.json:max_steps`).
- One action per step in the JSON action format (`shared/action_format.md`): `{"tool", "args"}` or `{"reply"}`.
- Stop generation as soon as the JSON object closes (braces balance), with a 320-token cap.
- Replies stream into the bubble. Tool steps show as compact chips ("Checking coverage…", "Moving appointment…") with a ✓ or an error.
- Each bubble carries a small provenance tag: **On device** (student), **Cloud** (teacher, tap for the ledger entry) or **Handed to a person**.

### 2.3 What the agent should handle on device (fine-tuned student)
Coverage questions, plan/deductible questions, claim status and denial reasons, provider search, booking, rescheduling (including relative dates like "Friday"), filing claims up to $1,000, handing off to a human, and asking one clarifying question when a detail is missing. These are the `heldout` categories in the evals.

### 2.4 What goes to the cloud (router gates)
Escalate the request to the teacher (via the backend proxy) when any gate fires on a student step:
- the output isn't valid action JSON, names an unknown tool, or fails the tool's JSON Schema;
- `member_id` ≠ the signed-in member, or a referenced appointment/claim/provider ID doesn't exist for this member;
- booking/reschedule date not after today; claim service date in the future;
- **low confidence**: min token logprob or mean token logprob below the thresholds in `shared/router.json` (tuned on the calibration split);
- **policy**: `file_claim` with amount > $1,000 always goes to the teacher.

Once escalated, the teacher drives the rest of that request. The chat shows the reason in plain words ("Sent to Bayside Cloud: unusual request (low confidence)").

### 2.5 Never-leave requests
If the member's message matches the never-leave pattern (diagnoses, medications, …, `shared/router.json:never_leave`), nothing about that request leaves the phone, including shadow samples. The student answers locally; if a gate fires, the app calls `escalate_to_human(topic: medical)` locally and says a person will follow up.

### 2.6 Offline
- With no network, the student serves everything. A request whose gate fires gets a best-effort local answer marked **"Flagged for review"** and goes into the offline queue.
- On reconnect, queued requests are sent (redacted) to the teacher for labeling. The ledger shows them with reason `offline_queue`.
- Airplane-mode acceptance: §8, beat 4.

## 3. Privacy

- **The model only sees redacted text.** Before the first model call, the session context and the member's message are redacted with `shared/redaction.json`: known fields of the signed-in member (name, first and last name, DOB, phone, email, address) plus patterns (member, policy and claim IDs, phone, email, `$` amounts). Tool results are redacted before they're appended; tool arguments are re-filled before execution; the reply is re-filled before display. The student and the teacher therefore see the same placeholder-space conversation, and stored traces are train-ready.
- **Scan, fail closed.** Every outbound payload is scanned (`shared/redaction.json:scan`). If anything matches, the request doesn't leave: the app hands off to a human locally and logs a `blocked` ledger row.
- **Golden vectors.** The Swift `Redactor` must reproduce `shared/redaction_vectors.json` byte for byte (unit test).
- **Storage.** Traces and the ledger live in the app container with `FileProtectionType.complete`. Nothing syncs unless the member turns on sync. When on, sync uploads only redacted traces (`POST /v1/traces`); the backend rejects any trace its scanner flags.
- **No keys on the phone.** Escalations go through `POST /v1/teacher` on the backend, which holds the Anthropic key, scans the payload again, and writes its own ledger row.
- **Shadow eval.** At the configured rate, a student-served step is also sent (redacted) to the teacher in the background; agreement is posted to `POST /v1/shadow`. Never-leave requests are skipped.

## 4. HarborKit (Swift package inside the app)

| Module | Responsibility | Source of truth |
| --- | --- | --- |
| `ModelRuntime` | Load the fused MLX 4-bit model with mlx-swift-lm; render with the model's chat template (`enable_thinking=false`); greedy decode with per-token logprobs; stop on closed JSON; **KV prompt cache** for the system prompt (reused across requests) | weights from `models/ft-e4b-rN/` |
| `Prompt` | System prompt = `system_core.txt` + `action_format.txt` + short tool block; session line; tool-result wrapping | `shared/prompts/`, `harbor_env/prompts.py` |
| `ToolRegistry` | Load `tools.json`; JSON-Schema validation; dispatch to local implementations | `shared/tools.json` |
| `MockStore` | Fixtures copy, slot logic, state diff, reset | `shared/fixtures/`, `harbor_env/env.py` |
| `Router` | Validators, confidence gate, policy, never-leave regex | `shared/router.json`, `harbor_env/router.py` |
| `Redactor` / `PIIScanner` | Redact, re-fill (type-aware for amounts), fail-closed scan | `shared/redaction.json`, `gradientharbor/privacy.py` |
| `TeacherClient` | `POST /v1/teacher` with the canonical redacted messages; handles 422 (scan failed) as blocked | `backend/app/main.py` |
| `TraceStore` | Append-only local store (SQLite or JSONL), file protection complete | — |
| `SyncClient` | Opt-in batch upload of redacted traces; offline queue replay | `POST /v1/traces` |
| `Ledger` | Rows for every outbound call (sent / blocked / shadow / offline_queue) | — |
| `ModelUpdater` | `GET /v1/models/latest`, compare sha256, download over Wi-Fi only; **side-load first** (Finder/USB) for the demo | `backend` |

A Swift client for the backend can be generated from `GET /openapi.json`.

## 5. Model and performance

- **Default student:** Gemma 4 E4B fine-tuned (round N), MLX 4-bit. The E2B student is the fallback if E4B misses the latency target on the Air.
- **Memory:** add the `com.apple.developer.kernel.increased-memory-limit` entitlement; check `os_proc_available_memory()` at launch and refuse to load (with a clear message) if under the model's requirement plus 1 GB.
- **Targets on the Air:** time to first token < 1.0 s with the cached system prompt; one tool step ≤ 1.5 s (E2B) or ≤ 3 s (E4B); a 2–3 step request under 6 s end to end. Measure tokens/s, TTFT, peak memory and battery per 100 requests, and show them in Settings → Diagnostics.
- **Thermals:** keep demo requests short; Diagnostics shows the thermal state.
- **Model files:** side-load round N before the demo (it is ~4–6 GB); never download on venue Wi-Fi.

## 6. Backend contract (laptop, same LAN)

| Call | When |
| --- | --- |
| `POST /v1/teacher` | escalation, shadow, offline-queue labeling |
| `POST /v1/traces` | sync (opt-in), batched |
| `POST /v1/shadow` | shadow agreement result |
| `GET /v1/models/latest` | Settings → Check for update |
| `POST /v1/eval-runs` | Eval mode results |

Base URL in Settings (default `http://<laptop>.local:8000`). Add `NSLocalNetworkUsageDescription` and an ATS exception for the local host.

## 7. Eval mode and parity tests

- **Eval mode** (Settings, debug builds): runs the bundled `heldout.jsonl` scenarios on the phone with auto-confirmed writes, scores them with the same rules as `harbor_env/scoring.py` (state diff + required calls), and posts the run as `phone-ft-e4b` to `POST /v1/eval-runs`. This gives the real on-device latency in the dashboard next to the Mac numbers.
- **Unit tests:** redaction golden vectors; router validators against fixtures; tool implementations against recorded `harbor_env` results (`tests/fixtures/tool_cases.json`, exported from the Python env); prompt rendering identical to Python for 5 sample conversations (string equality).
- **Parity check:** on the 98 held-out scenarios, phone task success within ±3 points of the Mac `ft-e4b` run with the same model.

## 8. Demo acceptance criteria (on-stage script)

1. **Routine request on device.** "Move my eye exam to Friday" is handled on the phone in under 6 s; the Appointments screen updates; the bubble is tagged *On device*; the ledger is unchanged.
2. **Escalation with a reason.** "My claim with Dr. … was denied and I want to appeal" is escalated; the chat shows the reason; the ledger row shows `[MEMBER_ID_1]`, `[CLAIM_ID_1]`, never the raw values.
3. **Never-leave.** "I was diagnosed with diabetes, is blood work covered?" is answered or handed to a person with no ledger row.
4. **Airplane mode.** With airplane mode on, two routine requests still work; a novel one gets a best-effort answer marked *Flagged for review*; after reconnecting, it appears in the ledger as `offline_queue`.
5. **Ledger proof.** Running the PII scanner over the ledger's sent payloads finds zero raw fields (also shown in the dashboard).

Each beat passes twice in a row before the demo; keep a screen recording as backup.

## 9. Out of scope for the hackathon
Real authentication, real member data, push notifications, per-user on-device adapters (roadmap), LiteRT-LM/Android/NPU (roadmap), payments.
