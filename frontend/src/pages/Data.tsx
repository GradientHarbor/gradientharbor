import { useEffect, useState } from 'react'
import { get, secs, usd, when, type Json } from '../api'
import { Badge, LineChart, Redacted } from '../charts'
import { Convo } from './Evals'

export function Traces() {
  const [source, setSource] = useState('')
  const [served, setServed] = useState('')
  const [escalated, setEscalated] = useState('')
  const [data, setData] = useState<Json>(null)
  const [sel, setSel] = useState<Json>(null)
  useEffect(() => {
    const q = new URLSearchParams({ limit: '200' })
    if (source) q.set('source', source)
    if (served) q.set('served_by', served)
    if (escalated) q.set('escalated', escalated)
    get(`/v1/traces?${q}`).then(setData)
  }, [source, served, escalated])
  return (
    <div className="split">
      <div className="card">
        <div className="card-head"><div><h2>Traces</h2><p>{data?.total ?? 0} stored · redacted on device, scanned again on arrival</p></div></div>
        <div className="filters">
          <select value={source} onChange={(e) => setSource(e.target.value)} aria-label="Source">
            <option value="">All sources</option><option value="teacher-datagen">Teacher data generation</option>
            <option value="eval">Eval runs</option><option value="traffic">Live traffic (sim)</option><option value="phone">Phone</option><option value="sdk">SDK wrap()</option>
          </select>
          <select value={served} onChange={(e) => setServed(e.target.value)} aria-label="Served by">
            <option value="">Served by anyone</option><option value="student">Student</option><option value="teacher">Teacher</option><option value="mixed">Mixed</option>
          </select>
          <select value={escalated} onChange={(e) => setEscalated(e.target.value)} aria-label="Escalated">
            <option value="">Escalated or not</option><option value="true">Escalated</option><option value="false">Not escalated</option>
          </select>
        </div>
        <div className="table-wrap" style={{ maxHeight: 600, overflowY: 'auto' }}><table>
          <thead><tr><th>Request</th><th>Served</th><th className="num">Outcome</th></tr></thead>
          <tbody>{(data?.traces ?? []).map((t: Json) => (
            <tr key={t.id} className={`clickable ${sel?.id === t.id ? 'selected' : ''}`} onClick={() => get(`/v1/traces/${encodeURIComponent(t.id)}`).then(setSel)}>
              <td><Redacted text={(t.request_redacted ?? '(rejected: ' + t.reject_reason + ')').split('\n\n').slice(-1)[0].slice(0, 110)} />
                <div className="muted mono">{t.category} · {t.run_id}</div></td>
              <td>{t.served_by}{t.escalated ? <div><span className="badge warn">escalated</span></div> : null}</td>
              <td className="num"><Badge ok={t.accepted ? t.outcome?.task_success : false} no={t.accepted ? 'fail' : 'rejected'} /></td></tr>))}</tbody>
        </table></div>
      </div>
      <div className="card">
        {!sel ? <div className="empty">Select a trace</div> : (
          <>
            <div className="card-head"><div><h2 className="mono">{sel.id}</h2><p>{sel.model} · {sel.steps} steps · {secs((sel.latency_ms ?? 0) / 1000)}</p></div></div>
            <Convo messages={sel.messages ?? []} />
          </>
        )}
      </div>
    </div>
  )
}

export function Ledger() {
  const [data, setData] = useState<Json>(null)
  useEffect(() => { get('/v1/ledger?limit=300').then(setData) }, [])
  if (!data) return <div className="card empty">Loading…</div>
  return (
    <div className="card">
      <div className="card-head"><div><h2>Privacy ledger</h2>
        <p>Every cloud call with the exact text that left the device. The phone shows the same list for its own calls.</p></div>
        <div className="filters" style={{ margin: 0 }}>
          <span className="badge neutral">{data.stats.sent ?? 0} sent</span>
          <span className="badge neutral">{data.stats.blocked ?? 0} blocked</span>
          <span className={`badge ${data.stats.raw_pii_fields_sent ? 'bad' : 'good'}`}>{data.stats.raw_pii_fields_sent ? '✕' : '✓'} {data.stats.raw_pii_fields_sent} raw PII fields sent</span>
          <span className="badge neutral">{usd(data.stats.cost_usd)}</span>
        </div></div>
      <div className="table-wrap"><table>
        <thead><tr><th>When</th><th>Why</th><th>Exact text sent (redacted)</th><th className="num">Scan</th></tr></thead>
        <tbody>{data.calls.map((c: Json) => (
          <tr key={c.id}><td className="muted" style={{ whiteSpace: 'nowrap' }}>{when(c.created_at)}<div className="mono">{c.run_id}</div></td>
            <td>{c.reason}</td>
            <td style={{ maxWidth: 640 }}><pre className="mono" style={{ whiteSpace: 'pre-wrap', margin: 0, maxHeight: 120, overflow: 'auto' }}>
              {c.sent ? <Redacted text={(c.payload ?? '').slice(0, 1200)} /> : <span className="muted">not sent: blocked by PII scan ({(c.findings ?? []).map((f: Json) => f.type).join(', ')})</span>}</pre></td>
            <td className="num">{c.sent ? <Badge ok={c.raw_pii_found === 0} yes="clean" no={`${c.raw_pii_found} PII`} /> : <span className="badge warn">blocked</span>}</td></tr>))}</tbody>
      </table></div>
    </div>
  )
}

export function Training() {
  const [rounds, setRounds] = useState<Json[]>([])
  const [ds, setDs] = useState<Json[]>([])
  const [models, setModels] = useState<Json[]>([])
  useEffect(() => {
    get('/v1/rounds').then(setRounds); get('/v1/datasets').then(setDs); get('/v1/models').then(setModels)
  }, [])
  const withLoss = rounds.filter((r) => r.loss_log?.length)
  return (
    <>
      <div className="grid cols-2">
        <div className="card">
          <div className="card-head"><div><h2>Rounds</h2><p>Each round: dataset from accepted traces → LoRA → fused MLX 4-bit model for the phone</p></div></div>
          <div className="table-wrap"><table>
            <thead><tr><th>Round</th><th>Status</th><th>Dataset</th><th>Model</th></tr></thead>
            <tbody>{rounds.map((r) => <tr key={r.round}><td>{r.round}</td><td><span className="badge neutral">{r.status}</span></td>
              <td className="mono">{r.dataset_id ?? '—'}</td><td className="mono">{r.model_id ?? '—'}</td></tr>)}</tbody>
          </table></div>
        </div>
        <div className="card">
          <div className="card-head"><div><h2>Loss</h2><p>mlx-lm LoRA, validation loss per round</p></div></div>
          {withLoss.length ? (
            <LineChart yMax={Math.max(...withLoss.flatMap((r) => r.loss_log.map((x: Json) => x.val_loss ?? x.train_loss ?? 0))) * 1.1}
              yFormat={(v) => v.toFixed(2)}
              series={withLoss.flatMap((r, i) => [{ name: `Round ${r.round} val`, color: ['var(--series-1)', 'var(--series-2)', 'var(--series-3)'][i % 3],
                points: r.loss_log.filter((x: Json) => x.val_loss != null).map((x: Json) => ({ x: x.iter, y: x.val_loss })) }])} />
          ) : <div className="empty">No training logs yet</div>}
        </div>
      </div>
      <div className="grid cols-2 section">
        <div className="card">
          <div className="card-head"><div><h2>Datasets</h2><p>Outcome-filtered, PII-scanned, deduped teacher steps</p></div></div>
          <div className="table-wrap"><table>
            <thead><tr><th>Dataset</th><th className="num">Traces</th><th className="num">Failed outcome</th><th className="num">PII rejected</th><th className="num">Train</th><th className="num">Valid</th></tr></thead>
            <tbody>{ds.map((d) => <tr key={d.id}><td className="mono">{d.id}</td><td className="num">{d.stats.traces}</td><td className="num">{d.stats.filtered_outcome}</td>
              <td className="num">{d.stats.rejected_pii}</td><td className="num">{d.stats.train}</td><td className="num">{d.stats.valid}</td></tr>)}</tbody>
          </table></div>
        </div>
        <div className="card">
          <div className="card-head"><div><h2>Models</h2><p>What <code>GET /v1/models/latest</code> hands the phone</p></div></div>
          <div className="table-wrap"><table>
            <thead><tr><th>Model</th><th>Round</th><th>Format</th><th className="num">Size</th><th className="num">Val loss</th></tr></thead>
            <tbody>{models.map((m) => <tr key={m.id}><td className="mono">{m.id}</td><td>{m.round}</td><td>{m.format}</td>
              <td className="num">{m.size_bytes ? `${(m.size_bytes / 1e9).toFixed(2)} GB` : '—'}</td><td className="num">{m.metrics?.final_val_loss?.toFixed?.(3) ?? '—'}</td></tr>)}</tbody>
          </table></div>
        </div>
      </div>
    </>
  )
}
