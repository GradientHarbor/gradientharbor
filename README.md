# GradientHarbor

The hybrid inference and continuous learning platform for on-device AI agents: the phone answers first, private by default; the cloud handles what it can't, and every escalation trains the next on-device model. Built at the Lenovo × AGI House Build Day, Nov 8, 2026.

| Folder | What it is |
| --- | --- |
| [`landing-page/`](landing-page/) | The site at https://gradientharbor.com/ (Cloudflare Worker + static assets) |
| [`shared/`](shared/) | The contract: tool specs, mock insurer data, prompts, redaction and router rules |
| [`use-case-evals-poc/`](use-case-evals-poc/) | Simulated customer (Bayside Mutual support agent): scenarios, eval runs, training |
| [`backend/`](backend/) | Collector: traces, privacy ledger, teacher proxy, dataset builder, model registry, SDK |
| [`frontend/`](frontend/) | Flywheel dashboard for the backend |
| [`ios-app/`](ios-app/) | Bayside iOS demo app (SwiftUI, runs in the Simulator against `use-case-evals-poc/device_runtime`), requirements and demo script |

Run order: backend (`:8000`), then the pipeline in `use-case-evals-poc/README.md`; the dashboard is at http://localhost:8000/ after `cd frontend && npm run build`.
