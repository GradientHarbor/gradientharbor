import { useEffect, useState } from 'react'
import { get, pct, runLabel, secs, usd, type Json } from '../api'
import { BarChart, LineChart, Stat } from '../charts'

const ORDER = ['teacher', 'base-e4b', 'base-e2b', 'ft-e4b', 'ft-e2b', 'hybrid-e4b', 'hybrid-e2b']

export function latestEvalRuns(runs: Json[]): Json[] {
  // One run per kind: the highest round, then the latest eval over the main split (not datagen, not traffic).
  // Re-running an older round's eval must not replace the newer round.
  const out: Record<string, Json> = {}
  for (const r of [...runs].sort((a, b) => (a.round ?? -1) - (b.round ?? -1) || a.created_at - b.created_at)) {
    if (r.kind === 'teacher-datagen' || String(r.split).startsWith('traffic') || String(r.split).startsWith('train')) continue
    out[r.kind] = r
  }
  return ORDER.filter((k) => out[k]).map((k) => out[k]).concat(Object.values(out).filter((r) => !ORDER.includes(r.kind)))
}

export default function Overview() {
  const [ov, setOv] = useState<Json>(null)
  const [runs, setRuns] = useState<Json[]>([])
  const [err, setErr] = useState('')
  useEffect(() => {
    const load = () => Promise.all([get('/v1/metrics/overview'), get('/v1/eval-runs')])
      .then(([o, r]) => { setOv(o); setRuns(r) }).catch((e) => setErr(String(e)))
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [])
  if (err) return <div className="card empty">Collector not reachable: {err}</div>
  if (!ov) return <div className="card empty">Loading…</div>

  const evals = latestEvalRuns(runs)
  const byKind = Object.fromEntries(evals.map((r) => [r.kind, r]))
  const hybrid = byKind['hybrid-e4b'] ?? byKind['hybrid-e2b']
  const teacher = byKind['teacher']
  const ft = byKind['ft-e4b']
  const heldout = (r: Json) => r?.summary?.by_split?.heldout?.task_success ?? r?.summary?.task_success
  const rounds = (ov.rounds as Json[]).filter((r) => r.escalation_rate !== null && r.escalation_rate !== undefined)
  const localP50 = hybrid?.summary?.p50_step_latency_local_s ?? ft?.summary?.p50_step_latency_local_s
  const cloudP50 = teacher?.summary?.p50_step_latency_cloud_s

  return (
    <>
      <div className="grid cols-5">
        <Stat label="Escalated to the cloud" value={pct(hybrid?.summary?.escalation_rate)}
          sub={hybrid ? `round ${hybrid.round ?? '?'} · was 100% at round 0` : 'run the hybrid eval'} />
        <Stat label="Held-out task success" value={pct(heldout(hybrid ?? ft))}
          sub={`Sonnet alone: ${pct(heldout(teacher))}`} />
        <Stat label="p50 step latency, on device" value={secs(localP50)} sub={`cloud: ${secs(cloudP50)} (Mac proxy for the phone)`} />
        <Stat label="Raw PII fields sent" value={ov.ledger?.raw_pii_fields_sent ?? 0}
          badge={<span className={`badge ${(ov.ledger?.raw_pii_fields_sent ?? 0) === 0 ? 'good' : 'bad'}`}>
            {(ov.ledger?.raw_pii_fields_sent ?? 0) === 0 ? '✓ target 0' : '✕ leak'}</span>}
          sub={`${ov.ledger?.sent ?? 0} cloud calls, ${ov.ledger?.blocked ?? 0} blocked by scan`} />
        <Stat label="Traces collected" value={ov.traces?.accepted ?? 0}
          sub={`${ov.traces?.rejected ?? 0} rejected by PII scan · shadow agree ${pct(ov.shadow?.agree)}`} />
      </div>

      <div className="grid cols-2 section">
        <div className="card">
          <div className="card-head"><div><h2>The flywheel</h2><p>Share of requests escalated to the cloud teacher, by training round</p></div></div>
          {rounds.length > 0 ? (
            <LineChart xLabel="round" series={[{ name: 'Escalation rate', color: 'var(--series-1)',
              points: rounds.map((r) => ({ x: r.round, y: r.escalation_rate, label: r.task_success != null ? `task success ${pct(r.task_success)}` : undefined })) }]} />
          ) : <div className="empty">No hybrid runs yet</div>}
        </div>
        <div className="card">
          <div className="card-head"><div><h2>Task success by model</h2><p>Latest eval per configuration: held-out and novel requests</p></div></div>
          {evals.length ? (
            <BarChart series={[{ name: 'Held-out', color: 'var(--series-1)' }, { name: 'Novel (OOD)', color: 'var(--series-2)' }]}
              rows={evals.map((r) => ({ label: runLabel(r.kind), values: [r.summary?.by_split?.heldout?.task_success ?? null,
                r.summary?.by_split?.novel?.task_success ?? null], note: `n=${r.summary?.n}` }))} />
          ) : <div className="empty">No eval runs yet</div>}
        </div>
      </div>

      <div className="grid cols-2 section">
        <div className="card">
          <div className="card-head"><div><h2>Latency per step</h2><p>p50, seconds. On-device is the student on the Mac (MLX); cloud is Sonnet 5.5</p></div></div>
          {evals.length ? (
            <BarChart max={Math.max(1, ...evals.flatMap((r) => [r.summary?.p50_step_latency_local_s ?? 0, r.summary?.p50_step_latency_cloud_s ?? 0])) * 1.1}
              format={(v) => `${v.toFixed(2)} s`}
              series={[{ name: 'On device', color: 'var(--series-1)' }, { name: 'Cloud', color: 'var(--series-2)' }]}
              rows={evals.map((r) => ({ label: runLabel(r.kind), values: [r.summary?.p50_step_latency_local_s ?? null, r.summary?.p50_step_latency_cloud_s ?? null] }))} />
          ) : <div className="empty">No eval runs yet</div>}
        </div>
        <div className="card">
          <div className="card-head"><div><h2>Privacy</h2><p>What left the device in the latest runs</p></div></div>
          <div className="table-wrap"><table>
            <thead><tr><th>Configuration</th><th className="num">Left device</th><th className="num">PII sent</th><th className="num">Never-leave violations</th><th className="num">Cost</th></tr></thead>
            <tbody>{evals.map((r) => (
              <tr key={r.id}><td>{runLabel(r.kind)}</td><td className="num">{pct(r.summary?.left_device_rate)}</td>
                <td className="num">{r.summary?.pii_fields_sent ?? 0}</td>
                <td className="num">{r.summary?.never_leave_violations ?? 0} / {r.summary?.never_leave_n ?? 0}</td>
                <td className="num">{usd(r.summary?.cost_usd)}</td></tr>))}</tbody>
          </table></div>
        </div>
      </div>
    </>
  )
}
