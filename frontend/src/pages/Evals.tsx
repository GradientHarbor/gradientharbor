import { Fragment, useEffect, useMemo, useState } from 'react'
import { get, pct, runLabel, secs, usd, type Json } from '../api'
import { Badge, Redacted } from '../charts'
import { latestEvalRuns } from './Overview'

const METRICS: [string, string, (s: Json) => string][] = [
  ['Task success', 'Tool calls + final state + reply, against template ground truth', (s) => pct(s.task_success)],
  ['First call', 'First action matches the expected tool and arguments', (s) => pct(s.first_call_match)],
  ['Judge correct', 'GPT judge: reply correct and grounded', (s) => pct(s.judge_correct)],
  ['Escalated', 'Share of requests handed to the cloud teacher', (s) => pct(s.escalation_rate)],
  ['Local success', 'Task success on requests served on device', (s) => pct(s.local_task_success)],
  ['p50 step', 'Median latency per agent step', (s) => secs(s.p50_step_latency_local_s ?? s.p50_step_latency_cloud_s)],
  ['Prompt tok', 'Mean prompt tokens per step (prefill dominates on the phone)', (s) => String(s.mean_prompt_tokens ?? '—')],
  ['PII sent', 'Raw PII fields in anything sent to the cloud', (s) => String(s.pii_fields_sent ?? 0)],
  ['Cost', 'Teacher API spend for the run', (s) => usd(s.cost_usd)],
]

export default function Evals() {
  const [runs, setRuns] = useState<Json[] | null>(null)
  const [all, setAll] = useState(false)
  const [detail, setDetail] = useState<Record<string, Json>>({})
  const [scenario, setScenario] = useState<string | null>(null)
  const [episodes, setEpisodes] = useState<Json[]>([])
  const [filter, setFilter] = useState<'all' | 'disagree'>('disagree')

  useEffect(() => { get('/v1/eval-runs').then(setRuns) }, [])
  const shown = useMemo(() => (all ? (runs ?? []).filter((r) => r.kind !== 'teacher-datagen') : latestEvalRuns(runs ?? [])), [runs, all])
  useEffect(() => {
    shown.forEach((r) => { if (!detail[r.id]) get(`/v1/eval-runs/${r.id}`).then((d) => setDetail((x) => ({ ...x, [r.id]: d }))) })
  }, [shown]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (scenario) get(`/v1/scenarios/${scenario}/episodes?run_ids=${shown.map((r) => r.id).join(',')}`).then(setEpisodes)
  }, [scenario, shown])

  const cats = Array.from(new Set(shown.flatMap((r) => Object.keys(r.summary?.by_category ?? {})))).sort()
  const scen = useMemo(() => {
    const m: Record<string, { id: string; category: string; split: string; res: Record<string, Json> }> = {}
    for (const r of shown) for (const x of detail[r.id]?.results ?? []) {
      m[x.scenario_id] ??= { id: x.scenario_id, category: x.category, split: x.split, res: {} }
      m[x.scenario_id].res[r.id] = x
    }
    let rows = Object.values(m).sort((a, b) => a.id.localeCompare(b.id))
    if (filter === 'disagree') rows = rows.filter((s) => new Set(shown.map((r) => s.res[r.id]?.score?.task_success)).size > 1)
    return rows
  }, [shown, detail, filter])

  if (!runs) return <div className="card empty">Loading…</div>
  if (!shown.length) return <div className="card empty">No eval runs yet. Run <code>runs/run_eval.py</code>.</div>
  return (
    <>
      <div className="card">
        <div className="card-head">
          <div><h2>Eval runs</h2><p>Same scenarios for every configuration: held-out, novel (OOD), never-leave and policy splits</p></div>
          <label className="muted"><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> show every run</label>
        </div>
        <div className="table-wrap"><table>
          <thead><tr><th>Configuration</th><th>Split</th><th className="num">n</th>{METRICS.map(([h, tip]) => <th key={h} className="num" title={tip}>{h}</th>)}</tr></thead>
          <tbody>{shown.map((r) => (
            <tr key={r.id}><td><b>{runLabel(r.kind)}</b><div className="muted mono">{r.id}{r.round != null ? ` · round ${r.round}` : ''}</div></td>
              <td className="muted">{String(r.split).replace(/,/g, ', ')}</td><td className="num">{r.summary?.n}</td>
              {METRICS.map(([h, , f]) => <td key={h} className="num">{f(r.summary ?? {})}</td>)}</tr>))}</tbody>
        </table></div>
      </div>

      <div className="card section">
        <div className="card-head"><div><h2>By category</h2><p>Task success per request type</p></div></div>
        <div className="table-wrap"><table>
          <thead><tr><th>Category</th>{shown.map((r) => <th key={r.id} className="num">{runLabel(r.kind)}</th>)}</tr></thead>
          <tbody>{cats.map((c) => (
            <tr key={c}><td>{c.replace(/_/g, ' ')}</td>{shown.map((r) => {
              const v = r.summary?.by_category?.[c]
              return <td key={r.id} className="num">{v ? <span className="bar-cell">{pct(v.task_success)}<span className="track"><span className="fill" style={{ width: `${(v.task_success ?? 0) * 100}%` }} /></span><span className="muted">{v.n}</span></span> : '—'}</td>
            })}</tr>))}</tbody>
        </table></div>
      </div>

      <div className="split section">
        <div className="card">
          <div className="card-head"><div><h2>Scenarios</h2><p>{scen.length} shown · click one to compare trajectories</p></div>
            <div className="filters" style={{ margin: 0 }}>
              <button className="chip" aria-pressed={filter === 'disagree'} onClick={() => setFilter('disagree')}>Configs disagree</button>
              <button className="chip" aria-pressed={filter === 'all'} onClick={() => setFilter('all')}>All</button>
            </div></div>
          <div className="table-wrap" style={{ maxHeight: 520, overflowY: 'auto' }}><table>
            <thead><tr><th>Scenario</th>{shown.map((r) => <th key={r.id} className="num">{r.kind}</th>)}</tr></thead>
            <tbody>{scen.map((s) => (
              <tr key={s.id} className={`clickable ${scenario === s.id ? 'selected' : ''}`} onClick={() => setScenario(s.id)}>
                <td><span className="mono">{s.id}</span><div className="muted">{s.category.replace(/_/g, ' ')}</div></td>
                {shown.map((r) => <td key={r.id} className="num"><Badge ok={s.res[r.id]?.score?.task_success} /></td>)}</tr>))}</tbody>
          </table></div>
        </div>
        <div className="card">
          {!scenario ? <div className="empty">Select a scenario</div> : (
            <>
              <div className="card-head"><div><h2 className="mono">{scenario}</h2><p>What each configuration did (redacted view, as stored by the collector)</p></div></div>
              {episodes.map((e) => {
                const run = shown.find((r) => r.id === e.run_id)
                return (
                  <div key={e.run_id} className="section">
                    <h3>{runLabel(run?.kind ?? e.run_id)} <Badge ok={e.score?.task_success} /> {e.escalation_reasons?.length > 0 &&
                      <span className="badge warn">escalated: {e.escalation_reasons.join(', ')}</span>}</h3>
                    {e.judge?.reason && <p className="muted">Judge: {e.judge.reason}</p>}
                    <Convo messages={e.messages ?? []} />
                  </div>
                )
              })}
            </>
          )}
        </div>
      </div>
    </>
  )
}

export function Convo({ messages }: { messages: Json[] }) {
  return (
    <div className="convo">{messages.map((m, i) => (
      <Fragment key={i}>
        <div className={`msg ${m.role}`}>
          <div className="who"><span>{m.role === 'assistant' ? (m.model ?? 'assistant') : m.kind === 'tool_result' ? `tool result · ${m.tool}` : 'member'}</span>
            {m.served_by && m.served_by !== 'student' && <span className="badge neutral">{m.served_by}</span>}</div>
          <pre><Redacted text={m.content} /></pre>
        </div>
      </Fragment>))}</div>
  )
}
