# frontend: flywheel dashboard

React + Vite + TypeScript, no chart library (hand-built SVG, light and dark). Talks only to the collector API.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /v1 to http://localhost:8000
npm run build    # dist/, served by the backend at http://localhost:8000/ (one URL, also from the phone on the LAN)
```

Tabs: **Overview** (escalation per round, task success by model, latency on device vs cloud, privacy), **Eval runs** (side-by-side metrics, per-category, scenario drill-down with each configuration's trajectory and the judge's reason), **Traces**, **Privacy ledger** (exact redacted text that left the device, re-scanned), **Training** (rounds, loss, datasets, models). The layout is responsive down to phone width.
