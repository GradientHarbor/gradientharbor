# backend: data collection service (POC)

FastAPI + SQLite collector for GradientHarbor. Clients are the eval harness, the iOS app (HarborKit, over the LAN), the Python SDK `wrap()`, and the dashboard.

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000   # dashboard at / once frontend/dist is built
uv run pytest -q
```

`.env` holds `ANTHROPIC_API` (copied from Loopr, chmod 600). Only the backend holds the key; the phone calls `/v1/teacher`.

## API (`/openapi.json`)

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/traces` | Batch of redacted traces. Each is PII-scanned again; flagged traces are stored as rejected with no content. Cloud calls and shadow results inside a trace go to the ledger and shadow tables. |
| `POST /v1/traces/{id}/feedback` | Outcome signal (`trace.feedback(ok=True)` in the SDK) |
| `GET /v1/traces`, `GET /v1/traces/{id}` | Browse (redacted view) |
| `POST /v1/teacher` | Escalation proxy: scan (fail closed, 422) → Claude Sonnet 5.5 → ledger row |
| `POST /v1/shadow` | Shadow-eval agreement |
| `GET /v1/ledger` | Every cloud call with the exact text sent, re-scanned (`raw_pii_fields_sent`, target 0) |
| `POST /v1/datasets/build` | Outcome filter → teacher-labeled steps → PII reject → dedupe → mlx-lm chat JSONL, split by trace. `include_shadow` adds shadow-sample disagreements and `include_ood` adds OOD relabels; stats show `by_source`. |
| `POST /v1/rounds`, `GET /v1/rounds` | Training rounds, loss logs |
| `POST /v1/models`, `GET /v1/models/latest` | Model registry; manifest the phone polls (sha256, size, router config) |
| `POST /v1/eval-runs`, `GET /v1/eval-runs[/{id}]` | Eval runs and per-scenario results |
| `GET /v1/scenarios/{id}/episodes` | One scenario across runs (dashboard drill-down) |
| `GET /v1/metrics/overview` | Dashboard summary: escalation per round, ledger, traces, shadow |
| `POST /v1/ood/score` | Async OOD distance to the training requests. This POC uses character trigrams; embeddinggemma-300m is the production plan in `shared/router.json`. |
| `POST /v1/ood/label` | Scores synced traces, then has the teacher relabel each student step of the most distant OOD ones (`limit`, `dry_run`). PII-scanned, ledgered as `ood_label`, stored in `labels`; already-labelled steps are skipped. |

## SDK

`sdk/gradientharbor` is a small installable package:
- `wrap(client, task=...)`: one line around an Anthropic or OpenAI client; every `create()` becomes a trace posted in the background.
- `privacy`: the `Redactor` (placeholders and type-aware re-fill) and the fail-closed `PIIScanner`. The eval harness uses them too, and they are tested against `shared/redaction_vectors.json`.

```python
import anthropic, gradientharbor
client = gradientharbor.wrap(anthropic.Anthropic(), task="support-agent")
```

Storage: `data/harbor.db` (WAL) and `data/datasets/<id>/{train,valid}.jsonl`. Both are git-ignored.
