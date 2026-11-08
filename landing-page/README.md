# landing-page

Landing page for https://gradientharbor.com/: GradientHarbor, the hybrid inference and continuous learning platform for on-device AI agents. It's static HTML, CSS and JS in `public/`, served by a small Cloudflare Worker (`src/worker.js`).

## Routing on gradientharbor.com

| Path | Served by |
| --- | --- |
| `/*` | this Worker, `gradientharbor-landing` |
| `/bi`, `/bi/*` | the old BI site, Worker `gradientharbor` (repo `GradientHarbor/website`, built with base `/bi/`) |
| `/blog/*`, `/docs/*` | 301 redirect to `/bi/blog/*` and `/bi/docs/*`, so old links keep working |
| `www.gradientharbor.com/*` | 301 redirect to the apex domain |

Cloudflare picks the more specific route, so `/bi*` reaches the old Worker even though this one claims `/*`.

## Develop and deploy

```bash
npm install
npm run dev      # http://localhost:8787
npm run deploy   # wrangler deploy, needs `npx wrangler login` once
```

## Screenshots

The files in `public/screenshots/` (from the Bayside app in the iPhone Air Simulator, and the backend dashboard) replace their placeholders automatically. To swap one, replace the file and deploy:

- `support-chat.png`: phone screenshot (9:19.5)
- `privacy-ledger.png`: phone screenshot
- `airplane-mode.png`: phone screenshot
- `dashboard.png`: flywheel dashboard (16:9)

## Results

The metric tiles in the Results section hold the Nov 8 overnight-run numbers from `use-case-evals-poc/RESULTS.md`. To update them, edit the `data-metric` spans in `public/index.html`.
