"""Build a self-contained HTML eval report with every agent run, for the demo.

    uv run python runs/build_report.py            # -> reports/eval-report.html

Reads runs/output/<run>/episodes.jsonl (+ scenario splits for the expected outcome) and
writes one HTML file with the data inlined: no server, no network, works offline. The
backend also serves it at http://localhost:8000/report.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "output"
SPLITS = ["heldout", "novel", "never_leave", "policy"]

RUNS = [  # (run dir, label, short label, kind)
    ("teacher-eval", "Sonnet 5.5 (cloud baseline)", "Sonnet", "cloud"),
    ("base-e4b-eval", "Gemma 4 E4B, no fine-tuning", "Base E4B", "local"),
    ("ft-e4b-r1-eval", "E4B fine-tuned, round 1", "FT r1", "local"),
    ("ft-e4b-r2-eval", "E4B fine-tuned, round 2", "FT r2", "local"),
    ("hybrid-e4b-r1-eval", "E4B fine-tuned + router, round 1", "Hybrid r1", "hybrid"),
    ("hybrid-e4b-r2-eval", "E4B fine-tuned + router, round 2", "Hybrid r2", "hybrid"),
    # Liquid AI LFM3-1.7B (0.97 GB at 4-bit), same round-2 training data, run with mlx-vlm.
    ("base-lfm3-eval", "LFM3-1.7B, no fine-tuning", "LFM3 base", "local"),
    ("ft-lfm3-r2-eval", "LFM3-1.7B LoRA", "LFM3 LoRA", "local"),
    ("ft-lfm3full-r2-eval", "LFM3-1.7B full fine-tune", "LFM3 full", "local"),
    ("hybrid-lfm3-r2-eval", "LFM3-1.7B LoRA + router", "LFM3 hybrid", "hybrid"),
    ("hybrid-lfm3full-r2-eval", "LFM3-1.7B full fine-tune + router", "LFM3 full hybrid", "hybrid"),
]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def trim(v, n=1800):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " …"


def compact_steps(ep: dict) -> list[dict]:
    out = []
    for s in ep.get("steps", []):
        out.append({k: v for k, v in {
            "served_by": s.get("served_by"), "model": s.get("model"), "latency_s": s.get("latency_s"),
            "prompt_tokens": s.get("prompt_tokens"), "reasons": (s.get("gate") or {}).get("reasons") or None,
            "action": s.get("action"), "discarded": s.get("discarded_action"), "raw": s.get("raw") if not s.get("action") else None,
            "result": trim(s["tool_result"], 1400) if s.get("tool_result") is not None else None,
            "cost": s.get("cost_usd"),
        }.items() if v not in (None, [], "")})
    return out


def main() -> None:
    expected = {}
    for sp in SPLITS:
        for sc in read_jsonl(ROOT / "scenarios" / "splits" / f"{sp}.jsonl"):
            expected[sc["id"]] = sc.get("expected", {})

    runs, scen = [], {}
    for rid, label, short, kind in RUNS:
        p = OUT_DIR / rid / "episodes.jsonl"
        if not p.exists():
            print(f"skip {rid}: no episodes", file=sys.stderr)
            continue
        eps = read_jsonl(p)
        lat, toks, cost, n_esc, kept, kept_ok, judged_ok = [], [], 0.0, 0, 0, 0, 0
        by_split: dict[str, list[int]] = {}
        for e in eps:
            ep, ok = e["episode"], bool(e["score"]["task_success"])
            by_split.setdefault(e["split"], []).append(ok)
            for s in ep["steps"]:
                if s.get("latency_s") and s.get("served_by") in ("student", "teacher", "escalated"):
                    if (kind == "cloud") == (s.get("served_by") == "teacher"):
                        lat.append(s["latency_s"])
                if s.get("prompt_tokens") and s.get("served_by") in ("student", "teacher"):
                    toks.append(s["prompt_tokens"])
            cost += sum(c.get("cost_usd") or 0 for c in ep.get("cloud_calls", []) if c.get("sent"))
            # Hybrid: count router escalations (matches RESULTS.md). One r2 escalation fired at the step
            # limit and the loop ended before the teacher was called, so 24 of 25 actually reached the cloud.
            left = bool(ep.get("escalated")) if kind == "hybrid" else False
            if kind == "cloud":
                left = True
            n_esc += bool(left)
            if not left:
                kept += 1
                kept_ok += ok
            judged_ok += bool((e.get("judge") or {}).get("correct"))
            sid = e["scenario_id"]
            row = scen.setdefault(sid, {"id": sid, "split": e["split"], "category": e["category"], "message": e["message"],
                                        "redacted": ep.get("redacted_request"), "expected": expected.get(sid, {}),
                                        "runs": {}})
            j = e.get("judge") or {}
            row["runs"][rid] = {
                "ok": ok, "score": {k: v for k, v in e["score"].items() if k != "task_success"},
                "judge": {"correct": j.get("correct"), "score": j.get("score"), "reason": j.get("reason")} if j else None,
                "steps": compact_steps(ep), "reply": ep.get("final_reply"), "error": ep.get("error"),
                "diff": ep.get("state_diff"), "escalated": ep.get("escalated"),
                "esc_reasons": ep.get("escalation_reasons") or None, "left": bool(left), "never_leave": ep.get("never_leave"),
                "cloud": [{"reason": c.get("reason"), "sent": c.get("sent"), "payload": trim(c.get("payload") or "", 1600),
                           "cost": c.get("cost_usd")} for c in ep.get("cloud_calls", [])],
                "pii": ep.get("pii_fields_sent", 0),
            }
        n = len(eps)
        runs.append({
            "id": rid, "label": label, "short": short, "kind": kind, "n": n,
            "success": sum(bool(e["score"]["task_success"]) for e in eps) / n,
            "by_split": {k: sum(v) / len(v) for k, v in by_split.items()},
            "split_n": {k: len(v) for k, v in by_split.items()},
            "judge": judged_ok / n, "escalated": n_esc / n,
            "local_success": (kept_ok / kept) if kept and kind != "cloud" else None,
            "p50": statistics.median(lat) if lat else None,
            "tokens": round(statistics.mean(toks)) if toks else None,
            "cost": round(cost, 2),
            "pii": sum(r["runs"][rid]["pii"] for r in scen.values() if rid in r["runs"]),
        })
        print(f"{rid:22s} success {runs[-1]['success']:.3f}  escalated {runs[-1]['escalated']:.3f}  "
              f"p50 {runs[-1]['p50']}  cost ${cost:.2f}", file=sys.stderr)

    data = {"runs": runs, "scenarios": sorted(scen.values(), key=lambda r: (SPLITS.index(r["split"]), r["id"]))}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    out = ROOT / "reports" / "eval-report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(scen)} scenarios x {len(runs)} runs)", file=sys.stderr)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GradientHarbor eval report</title>
<style>
:root{--bg:#f6f7fb;--surface:#fff;--surface-2:#f0f2f8;--line:#e2e6ef;--ink:#121826;--ink-2:#4a5468;--muted:#7a8397;
--accent:#4f46e5;--s1:#4f46e5;--s2:#0e9f8f;--s3:#d97706;--good:#13795b;--good-bg:#e3f4ec;--bad:#b42318;--bad-bg:#fde8e6;
--warn:#9a5b00;--warn-bg:#fdf0d5;--ph:#ede9fe;--ph-ink:#4338ca;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b0f19;--surface:#121826;--surface-2:#182033;--line:#263049;
--ink:#e8ecf5;--ink-2:#b4bdd0;--muted:#8590a8;--accent:#8b85ff;--s1:#8b85ff;--s2:#2fc4b2;--s3:#f0a43a;--good:#4ade9a;--good-bg:#12301f;
--bad:#ff8a7a;--bad-bg:#3a1714;--warn:#f5c05a;--warn-bg:#3a2a0c;--ph:#2a2450;--ph-ink:#c4bfff}}
:root[data-theme="dark"]{--bg:#0b0f19;--surface:#121826;--surface-2:#182033;--line:#263049;--ink:#e8ecf5;--ink-2:#b4bdd0;--muted:#8590a8;
--accent:#8b85ff;--s1:#8b85ff;--s2:#2fc4b2;--s3:#f0a43a;--good:#4ade9a;--good-bg:#12301f;--bad:#ff8a7a;--bad-bg:#3a1714;--warn:#f5c05a;
--warn-bg:#3a2a0c;--ph:#2a2450;--ph-ink:#c4bfff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,-apple-system,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:28px 24px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:18px;margin:0 0 4px}p{margin:0}
.muted{color:var(--muted)}.sub{color:var(--ink-2);max-width:900px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px}
.grid{display:grid;gap:14px}.tiles{grid-template-columns:repeat(4,1fr);margin:22px 0}
.tile .v{font-size:30px;font-weight:750;letter-spacing:-.02em}.tile .l{color:var(--ink-2);font-size:13px}
section{margin-top:22px}.head{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;margin-bottom:12px;flex-wrap:wrap}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:var(--surface-2);font-weight:650}.tw{overflow-x:auto}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.good{background:var(--good-bg);color:var(--good)}.bad{background:var(--bad-bg);color:var(--bad)}.warn{background:var(--warn-bg);color:var(--warn)}
.neutral{background:var(--surface-2);color:var(--ink-2)}
.ph{background:var(--ph);color:var(--ph-ink);border-radius:4px;padding:0 3px;font-family:var(--mono);font-size:.92em}
.chart text{fill:var(--ink-2);font-size:12px}.chart .lab{fill:var(--ink);font-weight:600}.grid-l{stroke:var(--line)}
.picks{grid-template-columns:repeat(4,1fr)}.pick{cursor:pointer;transition:border-color .15s}.pick:hover{border-color:var(--accent)}
.pick .k{font-size:12px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.05em}.pick .q{margin-top:6px;font-weight:600}
.pick .w{margin-top:6px;color:var(--ink-2);font-size:13px}
.explorer{display:grid;grid-template-columns:360px 1fr;gap:14px;align-items:start}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
select,input{font:inherit;font-size:13px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink)}
input{flex:1;min-width:120px}
.list{max-height:760px;overflow-y:auto;border-top:1px solid var(--line)}
.item{padding:9px 6px;border-bottom:1px solid var(--line);cursor:pointer}.item:hover{background:var(--surface-2)}.item.sel{background:var(--surface-2);box-shadow:inset 3px 0 var(--accent)}
.item .m{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.item .meta{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:3px}
.dots{display:flex;gap:3px}.dot{width:9px;height:9px;border-radius:50%}.dot.y{background:var(--good)}.dot.n{background:var(--bad)}.dot.e{outline:2px solid var(--s3);outline-offset:1px}
.req{font-size:17px;font-weight:600}.saw{font-family:var(--mono);font-size:12.5px;color:var(--ink-2);background:var(--surface-2);border-radius:8px;padding:10px;margin-top:8px;white-space:pre-wrap;word-break:break-word;max-height:150px;overflow:auto}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.col{border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
.col h3{font-size:14px;margin:0;display:flex;justify-content:space-between;align-items:center;gap:8px}
.step{border-left:3px solid var(--line);padding:6px 0 6px 10px;margin:10px 0}.step.student{border-color:var(--s1)}.step.teacher{border-color:var(--s3)}
.step.escalated{border-color:var(--warn)}.step.local_handoff,.step.local_template{border-color:var(--s2)}
.step .who{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center;flex-wrap:wrap}
pre{font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-word;margin:4px 0 0;background:var(--surface-2);border-radius:6px;padding:6px 8px;max-height:220px;overflow:auto}
details summary{cursor:pointer;color:var(--ink-2);font-size:12px}
.reply{background:var(--surface-2);border-radius:10px;padding:10px;margin-top:8px}
.kv{font-size:12px;color:var(--ink-2);margin-top:8px}
.tabs{display:flex;gap:6px;flex-wrap:wrap}.tab{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
.tt{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:5px 8px;border-radius:6px;display:none;z-index:9}
.theme{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:8px;padding:5px 10px;font:inherit;font-size:13px;cursor:pointer}
@media (max-width:1000px){.tiles{grid-template-columns:repeat(2,1fr)}.picks{grid-template-columns:1fr 1fr}.explorer{grid-template-columns:1fr}.list{max-height:360px}}
@media (max-width:640px){.wrap{padding:18px 16px 48px}.tiles,.picks,.cols{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="head" style="align-items:flex-start">
    <div>
      <h1>GradientHarbor eval report</h1>
      <p class="sub">Every configuration ran the same 152 synthetic member requests for a fictional insurer (Bayside Mutual), including Liquid AI's LFM3-1.7B (0.97 GB) next to Gemma 4 E4B (5.5 GB): 98 held-out members, 29 deliberately novel requests, 15 medical (never-leave) and 10 claims over $1,000. Task success is checked against ground truth: the right tool calls, the right final state, and a reply. Run overnight on Nov 8, 2026; latency on an M5 Mac with MLX.</p>
    </div>
    <button class="theme" id="theme" aria-label="Toggle theme">Theme</button>
  </div>
  <div class="grid tiles" id="tiles"></div>

  <section class="card">
    <div class="head"><div><h2>Runs side by side</h2><p class="muted">Hover a bar for details. "Escalated" counts requests the router sent to the cloud teacher, not background shadow samples.</p></div></div>
    <div id="bars"></div>
    <div class="tw" style="margin-top:12px"><table id="runs"></table></div>
  </section>

  <section class="card">
    <div class="head"><div><h2>By test split</h2><p class="muted">Task success per split. Never-leave requests never go to the cloud; policy claims always do.</p></div></div>
    <div class="tw"><table id="splits"></table></div>
  </section>

  <section>
    <div class="head"><div><h2>Demo picks</h2><p class="muted">Real runs that show each part of the story. Click one to open it below.</p></div></div>
    <div class="grid picks" id="picks"></div>
  </section>

  <section class="card" id="explore">
    <div class="head"><div><h2>Every agent run</h2><p class="muted">Pick a request, then choose two configurations to compare. Dots: one per run, left to right as in the table; ringed = sent to the cloud.</p></div></div>
    <div class="explorer">
      <div>
        <div class="filters">
          <select id="fSplit" aria-label="Split"><option value="">All splits</option><option value="heldout">Held-out</option><option value="novel">Novel</option><option value="never_leave">Never-leave</option><option value="policy">Policy</option></select>
          <select id="fOut" aria-label="Outcome"><option value="">All outcomes</option><option value="anyfail">Any run failed</option><option value="fixed">Base failed, hybrid r2 passed</option><option value="esc">Hybrid r2 escalated</option><option value="hfail">Hybrid r2 failed</option></select>
          <select id="fCat" aria-label="Category"><option value="">All categories</option></select>
          <input id="fQ" placeholder="Search requests" aria-label="Search">
        </div>
        <div class="muted" id="count" style="font-size:12px;margin-bottom:6px"></div>
        <div class="list" id="list"></div>
      </div>
      <div id="detail"><div class="muted" style="padding:40px;text-align:center">Select a request</div></div>
    </div>
  </section>
</div>
<div class="tt" id="tt"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const R = D.runs, S = D.scenarios, byId = Object.fromEntries(S.map(s => [s.id, s]));
const $ = (q) => document.querySelector(q);
const pct = (v) => v == null ? '—' : (v * 100).toFixed(1) + '%';
const esc = (t) => String(t ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ph = (t) => esc(t).replace(/\[[A-Z_]+_\d+\]/g, m => `<span class="ph">${m}</span>`);
const SPL = {heldout:'Held-out', novel:'Novel', never_leave:'Never-leave', policy:'Policy ($1,000+)'};
const WHO = {student:'On device', teacher:'Cloud teacher', escalated:'Router escalated', local_handoff:'Handed to a person (local)', local_template:'Local reply', blocked_by_pii_scan:'Blocked by PII scan'};
const COLOR = {cloud:'var(--s3)', local:'var(--s1)', hybrid:'var(--s2)'};
const tt = $('#tt');
function tip(e, html){ tt.innerHTML = html; tt.style.display = 'block'; tt.style.left = Math.min(e.clientX + 12, innerWidth - 260) + 'px'; tt.style.top = (e.clientY - 34) + 'px'; }
function untip(){ tt.style.display = 'none'; }

// theme
try { const t = localStorage.getItem('gh-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
$('#theme').onclick = () => { const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const nxt = cur === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = nxt; try { localStorage.setItem('gh-theme', nxt); } catch (e) {} };

// tiles
const T = R.find(r => r.id === 'teacher-eval'), H = R.find(r => r.id === 'hybrid-e4b-r2-eval') || R[R.length - 1], B = R.find(r => r.id === 'base-e4b-eval');
$('#tiles').innerHTML = [
  [pct(H.success), `Task success, fine-tuned E4B + router (Sonnet ${pct(T?.success)}, untrained E4B ${pct(B?.success)})`],
  [pct(1 - H.escalated), `Requests handled on the device (${pct(H.escalated)} sent to the cloud)`],
  [String(R.reduce((a, r) => a + r.pii, 0)), 'Raw personal fields sent to the cloud, all runs'],
  [`$${H.cost.toFixed(2)}`, `Cloud cost for 152 requests (Sonnet alone: $${T?.cost.toFixed(2)})`],
].map(([v, l]) => `<div class="card tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');

// bars (one measure: task success)
(function(){
  const W = 760, L = 230, Rr = 64, bh = 18, gap = 12, Hh = R.length * (bh + gap) + 8, sx = v => v * (W - L - Rr);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${Hh}" style="width:100%;max-width:${W}px" role="img" aria-label="Task success by run">`;
  [0, .5, 1].forEach(f => s += `<line class="grid-l" x1="${L + sx(f)}" x2="${L + sx(f)}" y1="0" y2="${Hh - 4}"/>`);
  R.forEach((r, i) => { const y = i * (bh + gap) + 4, w = Math.max(sx(r.success), 2);
    s += `<g data-i="${i}"><text class="lab" x="${L - 10}" y="${y + bh - 4}" text-anchor="end">${esc(r.label)}</text>
      <path d="M${L},${y} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z" fill="${COLOR[r.kind]}"/>
      <rect x="${L}" y="${y - 3}" width="${W - L - Rr}" height="${bh + 6}" fill="transparent"/>
      <text x="${L + w + 6}" y="${y + bh - 4}">${pct(r.success)}</text></g>`; });
  s += '</svg><div class="muted" style="font-size:12px;display:flex;gap:14px;margin-top:4px">' +
    [['cloud','Cloud only'],['local','On device only'],['hybrid','On device + router']].map(([k, l]) => `<span><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${COLOR[k]};margin-right:5px"></i>${l}</span>`).join('') + '</div>';
  $('#bars').innerHTML = s;
  $('#bars').querySelectorAll('g[data-i]').forEach(g => { const r = R[+g.dataset.i];
    g.onmousemove = e => tip(e, `<b>${esc(r.label)}</b><br>Task success ${pct(r.success)} · sent to cloud ${pct(r.escalated)}`); g.onmouseleave = untip; });
})();

// runs table
$('#runs').innerHTML = `<tr><th>Configuration</th><th class="n">Task success</th><th class="n">Held-out</th><th class="n">Novel</th><th class="n">Judge: reply correct</th><th class="n">Escalated</th><th class="n">Success kept on device</th><th class="n">p50 / step</th><th class="n">Prompt tokens / step</th><th class="n">Cloud cost</th></tr>` +
  R.map(r => `<tr class="${r === H ? 'hl' : ''}"><td>${esc(r.label)}</td><td class="n">${pct(r.success)}</td><td class="n">${pct(r.by_split.heldout)}</td><td class="n">${pct(r.by_split.novel)}</td>
    <td class="n">${pct(r.judge)}</td><td class="n">${pct(r.escalated)}</td><td class="n">${pct(r.local_success)}</td>
    <td class="n">${r.p50 == null ? '—' : r.p50.toFixed(2) + ' s' + (r.kind === 'cloud' ? ' (cloud)' : '')}</td><td class="n">${r.tokens?.toLocaleString() ?? '—'}</td><td class="n">$${r.cost.toFixed(2)}</td></tr>`).join('');

// splits table
$('#splits').innerHTML = `<tr><th>Configuration</th>${Object.keys(SPL).map(k => `<th class="n">${SPL[k]} (${R[0].split_n[k] ?? 0})</th>`).join('')}</tr>` +
  R.map(r => `<tr class="${r === H ? 'hl' : ''}"><td>${esc(r.label)}</td>${Object.keys(SPL).map(k => `<td class="n">${pct(r.by_split[k])}</td>`).join('')}</tr>`).join('');

// demo picks
const run = (s, id) => s.runs[id] || {};
const hasReason = (s, id, re) => (run(s, id).steps || []).some(st => (st.reasons || []).some(x => re.test(x)));
const PICKS = [
  ['Fine-tuning fixes it', 'Untrained E4B fails, the round-2 student handles it on the device.', s => s.split === 'heldout' && !run(s,'base-e4b-eval').ok && run(s,'ft-e4b-r2-eval').ok && /reschedule|book/.test(s.category), ['base-e4b-eval','hybrid-e4b-r2-eval']],
  ['Untrained model breaks the format', 'The base model answers in prose instead of an action; the episode fails.', s => run(s,'base-e4b-eval').error === 'invalid_action' && run(s,'hybrid-e4b-r2-eval').ok, ['base-e4b-eval','hybrid-e4b-r2-eval']],
  ['Router catches a bad step', 'A validator fires on the student’s step; Sonnet finishes the request from redacted text.', s => run(s,'hybrid-e4b-r2-eval').escalated && run(s,'hybrid-e4b-r2-eval').ok && hasReason(s,'hybrid-e4b-r2-eval', /^schema|not_grounded|bad_|unknown/), ['ft-e4b-r2-eval','hybrid-e4b-r2-eval']],
  ['Claim over $1,000', 'Policy: large claims always go to the cloud, with every ID and amount redacted.', s => s.split === 'policy' && run(s,'hybrid-e4b-r2-eval').ok, ['teacher-eval','hybrid-e4b-r2-eval']],
  ['Medical request stays on the phone', 'Never-leave: no cloud call at all; a person follows up.', s => s.split === 'never_leave' && run(s,'hybrid-e4b-r2-eval').ok && !(run(s,'hybrid-e4b-r2-eval').cloud || []).length, ['teacher-eval','hybrid-e4b-r2-eval']],
  ['Liquid LFM3-1.7B learns the job', 'Untrained, LFM3 answers in prose instead of an action; after LoRA on the same data, the 1 GB model gets it right on the device.', s => s.split === 'heldout' && run(s,'base-lfm3-eval').error === 'invalid_action' && run(s,'hybrid-lfm3-r2-eval').ok && !run(s,'hybrid-lfm3-r2-eval').escalated && /reschedule|book|file_claim/.test(s.category), ['base-lfm3-eval','hybrid-lfm3-r2-eval']],
  ['Round 2 learned a new request type', 'Novel request: the round-1 hybrid fails it; round 2 learned from round 1’s escalations.', s => s.split === 'novel' && !run(s,'hybrid-e4b-r1-eval').ok && run(s,'hybrid-e4b-r2-eval').ok && !run(s,'hybrid-e4b-r2-eval').escalated, ['hybrid-e4b-r1-eval','hybrid-e4b-r2-eval']],
];
$('#picks').innerHTML = PICKS.map(([k, w, f, cmp], i) => { const s = S.find(f); if (!s) return '';
  return `<div class="card pick" data-id="${s.id}" data-cmp="${cmp.join(',')}"><div class="k">${esc(k)}</div><div class="q">“${esc(s.message)}”</div><div class="w">${esc(w)}</div></div>`; }).join('');
$('#picks').querySelectorAll('.pick').forEach(el => el.onclick = () => { cmp = el.dataset.cmp.split(','); select(el.dataset.id); $('#explore').scrollIntoView({behavior:'smooth'}); });

// explorer
const cats = [...new Set(S.map(s => s.category))].sort();
$('#fCat').innerHTML += cats.map(c => `<option>${esc(c)}</option>`).join('');
let sel = null, cmp = ['base-e4b-eval', 'hybrid-e4b-r2-eval'];
function filtered(){ const sp = $('#fSplit').value, o = $('#fOut').value, c = $('#fCat').value, q = $('#fQ').value.toLowerCase();
  return S.filter(s => (!sp || s.split === sp) && (!c || s.category === c) && (!q || s.message.toLowerCase().includes(q)) && (
    !o || (o === 'anyfail' && R.some(r => s.runs[r.id] && !s.runs[r.id].ok)) || (o === 'fixed' && !run(s,'base-e4b-eval').ok && run(s,'hybrid-e4b-r2-eval').ok) ||
    (o === 'esc' && run(s,'hybrid-e4b-r2-eval').escalated) || (o === 'hfail' && run(s,'hybrid-e4b-r2-eval').ok === false))); }
function renderList(){ const f = filtered(); $('#count').textContent = `${f.length} of ${S.length} requests`;
  $('#list').innerHTML = f.map(s => `<div class="item ${sel === s.id ? 'sel' : ''}" data-id="${s.id}"><div class="m">${esc(s.message)}</div>
    <div class="meta"><span>${SPL[s.split]} · ${esc(s.category)}</span><span class="dots">${R.map(r => { const x = s.runs[r.id]; return x ? `<i class="dot ${x.ok ? 'y' : 'n'} ${x.left && r.kind === 'hybrid' ? 'e' : ''}" title="${esc(r.short)}: ${x.ok ? 'pass' : 'fail'}"></i>` : ''; }).join('')}</span></div></div>`).join('');
  $('#list').querySelectorAll('.item').forEach(el => el.onclick = () => select(el.dataset.id)); }
['#fSplit', '#fOut', '#fCat'].forEach(q => $(q).onchange = renderList); $('#fQ').oninput = renderList;

function stepHtml(st){
  const who = WHO[st.served_by] || st.served_by;
  let h = `<div class="step ${st.served_by}"><div class="who"><b>${esc(who)}</b>${st.model ? `<span>${esc(st.model)}</span>` : ''}${st.latency_s ? `<span>${st.latency_s.toFixed(2)} s</span>` : ''}${st.prompt_tokens ? `<span>${st.prompt_tokens} prompt tokens</span>` : ''}</div>`;
  if (st.reasons) h += `<div style="margin-top:4px">${st.reasons.map(r => `<span class="badge warn">${esc(r)}</span>`).join(' ')}</div>`;
  const a = st.served_by === 'escalated' ? st.discarded : st.action;
  if (a) h += `<pre>${ph(JSON.stringify(a))}</pre>`; else if (st.raw) h += `<pre>${esc(st.raw)}</pre>`;
  if (st.served_by === 'escalated') h += `<div class="kv">Student’s step discarded; the cloud teacher takes over.</div>`;
  if (st.result) h += `<details><summary>Tool result (real values, on the device)</summary><pre>${esc(st.result)}</pre></details>`;
  return h + '</div>';
}
function colHtml(id, s){
  const r = R.find(x => x.id === id), x = s.runs[id]; if (!x) return `<div class="col"><h3>${esc(r?.label ?? id)}</h3><p class="muted">No run</p></div>`;
  const cloudSent = (x.cloud || []).filter(c => c.sent);
  let h = `<div class="col"><h3><span>${esc(r.label)}</span><span class="badge ${x.ok ? 'good' : 'bad'}">${x.ok ? '✓ pass' : '✕ fail'}</span></h3>
    <div class="kv">${x.left && r.kind !== 'cloud' ? `<span class="badge warn">sent to cloud${x.esc_reasons ? ': ' + esc(x.esc_reasons.join(', ')) : ''}</span>` : r.kind === 'cloud' ? '<span class="badge neutral">cloud only</span>' : '<span class="badge good">stayed on device</span>'}
    ${x.never_leave ? ' <span class="badge neutral">never-leave</span>' : ''}${x.error ? ` <span class="badge bad">${esc(x.error)}</span>` : ''}</div>`;
  h += x.steps.map(stepHtml).join('');
  h += `<div class="reply"><div class="muted" style="font-size:12px">Reply the member sees</div>${x.reply ? esc(x.reply) : '<span class="muted">(no reply)</span>'}</div>`;
  if (x.judge) h += `<div class="kv"><b>Judge</b> ${x.judge.correct ? '<span class="badge good">correct</span>' : '<span class="badge bad">not correct</span>'} ${x.judge.score ?? ''}/5 · ${esc(x.judge.reason || '')}</div>`;
  const sc = Object.entries(x.score || {}).filter(([, v]) => v === false).map(([k]) => k);
  if (sc.length) h += `<div class="kv"><b>Missed</b> ${sc.map(k => `<span class="badge bad">${esc(k)}</span>`).join(' ')}</div>`;
  if (x.diff?.length) h += `<div class="kv"><b>State change</b><pre>${esc(JSON.stringify(x.diff))}</pre></div>`;
  if (cloudSent.length && r.kind !== 'cloud') h += `<details class="kv"><summary>What left the device (${cloudSent.length} call${cloudSent.length > 1 ? 's' : ''}, ${x.pii} raw PII fields)</summary>${cloudSent.map(c => `<div style="margin-top:6px"><span class="badge neutral">${esc(c.reason)}</span><pre>${ph(c.payload)}</pre></div>`).join('')}</details>`;
  return h + '</div>';
}
function select(id){ sel = id; const s = byId[id]; renderList();
  const tabs = R.map(r => `<button class="tab ${cmp.includes(r.id) ? 'on' : ''}" data-r="${r.id}">${esc(r.short)}</button>`).join('');
  $('#detail').innerHTML = `<div class="req">“${esc(s.message)}”</div>
    <div class="kv">${SPL[s.split]} · ${esc(s.category)} · <span class="mono">${esc(s.id)}</span></div>
    <details><summary style="margin-top:8px">What the model saw (redacted on the device)</summary><div class="saw">${ph(s.redacted)}</div></details>
    <details><summary>Expected outcome (ground truth)</summary><pre>${esc(JSON.stringify(s.expected, null, 1))}</pre></details>
    <div class="head" style="margin:12px 0 0"><span class="muted" style="font-size:12px">Compare two runs</span><div class="tabs">${tabs}</div></div>
    <div class="cols">${cmp.map(c => colHtml(c, s)).join('')}</div>`;
  $('#detail').querySelectorAll('.tab').forEach(b => b.onclick = () => { const r = b.dataset.r; if (cmp.includes(r)) return; cmp = [cmp[1], r]; select(id); });
}
renderList();
const first = S.find(PICKS[0][2]) || S[0]; cmp = PICKS[0][3]; select(first.id);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
