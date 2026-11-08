# Bayside: iOS demo app

The member app for the fictional insurer Bayside Mutual, with the GradientHarbor assistant. `REQUIREMENTS.md` is the full spec. This README covers the hackathon build: **iOS Simulator + a device runtime on the Mac**.

## Why a device runtime

MLX doesn't run in the iOS Simulator. So for the Simulator demo, the "phone" stack runs on the Mac in `use-case-evals-poc/device_runtime/` (port 8100). That stack is the round-2 fine-tuned Gemma 4 E4B, the router, the redactor, the eight tools over the member's data, and the privacy ledger. It uses the same Python code and weights the evals measured, so the demo matches the published numbers.

The SwiftUI app is the phone UI on top of it. The cloud side stays separate: escalations go from the runtime to the backend's teacher proxy (port 8000), which holds the Anthropic key. That's the same path a real iPhone would use.

```
iOS Simulator (Bayside app) ──HTTP──▶ device runtime :8100 ("the phone": model, router, redactor, tools, ledger)
                                            └──redacted only──▶ backend :8000 /v1/teacher ──▶ Claude Sonnet 5.5
```

Say this on stage: *"The model runs on this laptop with MLX, the same weights the phone loads."* On a real iPhone, the runtime moves into the app as HarborKit (`REQUIREMENTS.md` §4), and the screens and events stay the same.

## Run it

Three terminals, from the repo root (`~/GitHub/gradientharbor`):

```bash
cd backend && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
cd use-case-evals-poc && uv run uvicorn device_runtime.server:app --host 0.0.0.0 --port 8100
```

The model takes about 30 s to load. The app shows "Loading the on-device model…" until it's ready.

```bash
cd ios-app/Bayside && xcodegen generate && open Bayside.xcodeproj
```

In Xcode, pick an **iPhone Air** simulator and press Run. No signing is needed for the Simulator.

To build and launch without Xcode instead:

```bash
cd ios-app/Bayside && xcodebuild -project Bayside.xcodeproj -scheme Bayside -destination 'platform=iOS Simulator,name=iPhone Air' -derivedDataPath build build && xcrun simctl install booted build/Build/Products/Debug-iphonesimulator/Bayside.app && xcrun simctl launch booted com.gradientharbor.bayside
```

Reset between rehearsals with Settings (gear on Home) → **Reset demo data**. Or run:

```bash
curl -s -XPOST localhost:8100/v1/device/reset -H 'content-type: application/json' -d '{}'
```

## Demo script (member: Owen Novak, Bronze PPO; simulated today is Mon, Nov 9)

| # | Do | What the audience sees |
|---|---|---|
| 1 | Ask Bayside → "Move my eye exam to Friday" → **Confirm** | A confirmation card, then "Moving the appointment ✓", a reply tagged **On device**, a toast, and the Visits tab updated to Fri, Nov 13, 1:30 PM. The Privacy ledger is unchanged. |
| 2 | "File a claim: I saw Dr. Percival Nilsson, my orthopedist in San Mateo, on November 2. The bill was $1,450." → **Confirm** | The student finds the provider and drafts the claim on the device; then "Sent to Bayside Cloud: claims over $1,000 are always reviewed". Sonnet reviews and files it; the reply is tagged **Cloud**. Privacy shows the exact text sent with `[AMOUNT_1]`, and **0 raw personal fields**. |
| 3 | "I was diagnosed with diabetes, is blood work covered?" | "Medical topic: this request stays on your phone." Answered **On device**; no ledger row. |
| 4 | Settings → **Airplane mode (simulated)** on. Ask "How much of my deductible is left?", then "Please file a claim for my eye exam with Dr. Desmond Weiss on November 2. It was $1,200." | The first is answered on device as normal. The second hits the $1,000 claim-review policy and can't reach the cloud, so it's marked **Flagged for review** and the ledger shows "Saved while offline". Turn airplane mode off: the row flips to *replayed* and a new "Saved while offline · sent" row appears. |
| 5 | Privacy tab | The counters: on device / sent / blocked / queued, and **0 raw personal fields sent**. Tap any row to see the exact redacted text and what came back. |

Then switch to the dashboard (http://localhost:8000) and the eval report (http://localhost:8000/report).

Notes from rehearsing:
- **Use the scripted phrasings.** The model is deterministic, so the same request in the same state takes the same path. Reset between runs so the appointment is back on Nov 10.
- **Replies take 2–4 s per model step on the M5**, about 5–8 s for a request with a tool call.
- **Shadow quality checks are off by default** so beat 1 leaves the ledger empty. Turn them on (15%) in Settings to show background quality sampling.
- **Keep the doctor's specialty and city in beat 2.** Without them, the student's provider search can fail a check (also an escalation, tap Details to show it), and Sonnet sometimes asks for the provider ID instead of finding it.
- **An appeal ("my claim was denied, I want to appeal") doesn't go to the cloud.** The student correctly files a `claim_appeal` handoff ticket on the device. That's a good "it knows when a person is needed" moment, but not an escalation demo.

## Deep links (rehearsal, screen recording, scripted demos)

```bash
xcrun simctl openurl booted "bayside://ask?text=Move%20my%20eye%20exam%20to%20Friday"
```

`bayside://confirm?ok=1` (or `ok=0`) answers a confirmation card, `bayside://new` starts a new conversation, and `bayside://tab/ledger` (also `home`, `visits`, `claims`, `chat`) switches tabs.

## Files

| Path | What |
|---|---|
| `Bayside/project.yml` | XcodeGen spec. `Bayside.xcodeproj` is generated from it. |
| `Bayside/Bayside/DeviceClient.swift` | HTTP + streaming (NDJSON) client for the device runtime |
| `Bayside/Bayside/AppStore.swift` | App state, chat event handling, deep links |
| `Bayside/Bayside/ChatView.swift` | Support chat: tool chips, escalation notices, confirmation cards, provenance tags |
| `Bayside/Bayside/HomeView.swift`, `ListsView.swift` | Home, Policy, Visits, Claims |
| `Bayside/Bayside/LedgerSettingsView.swift` | Privacy ledger and Settings |
| `../use-case-evals-poc/device_runtime/` | The on-device stack served for the Simulator (`live.py` is the multi-turn agent loop) |

## On-device mode (real iPhone)

On an iPhone, the whole stack runs in the app; Settings → **Assistant runs on → This iPhone (MLX)** is the default. Only escalations, shadow checks and opt-in trace sync go to the backend (`http://Aydars-MacBook-Pro.local:8000` by default, editable in Settings), and real airplane mode works.

- **`HarborKit/`** is a Swift package. `HarborCore` ports `harbor_env` (tools over the bundled fixtures, redactor, PII scanner, router with reply grounding, prompts, the live agent loop and the ledger). `HarborMLX` runs the fused round-2 student with mlx-swift-lm 3.32 (Gemma 4 is built in). It renders the Gemma chat template by hand, without a Jinja engine, and decodes greedily with per-token logprobs.
- **Parity with the evals:** `harbor-check` replays the recorded eval episodes through the Swift code. As of Nov 8: all redaction vectors, 1,748 model-visible strings and 87 rendered prompts are identical to Python, and all 336 router gates match. `harbor-parity prompts|generate` also checks token ids and greedy outputs against the Mac (needs the model).

  ```bash
  cd HarborKit && xcodebuild -scheme harbor-check -destination 'platform=macOS' -derivedDataPath .build/xc -skipMacroValidation build && .build/xc/Build/Products/Debug/harbor-check parity
  ```

  Regenerate the fixtures with `uv run python runs/export_parity.py` in `use-case-evals-poc`.
- **Deploy:** run `./deploy-device.sh` with the iPhone unlocked and connected. It builds a Release app, installs it, and copies the 5.5 GB model into the app's `Documents/Models/ft-e4b-r2`. The first copy takes a few minutes over USB. `SKIP_MODEL=1` reinstalls the app only. The app has the increased-memory entitlement, which the 5.5 GB model needs.

## Not done yet (the real-device path)

- **Eval mode on the phone:** run the held-out scenarios on the device and post them as run `phone-ft-e4b`.
- **Profiling on the iPhone Air:** memory, thermals and battery. Also a KV prompt cache for the system prompt.
- **Model update checks** (`GET /v1/models/latest`).
