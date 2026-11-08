// Small hand-built SVG charts: thin marks, recessive grid, hover tooltips, one axis.
import { useRef, useState, type ReactNode } from 'react'

type Tip = { x: number; y: number; body: ReactNode } | null

function useTip() {
  const [tip, setTip] = useState<Tip>(null)
  const box = useRef<HTMLDivElement>(null)
  const show = (e: React.MouseEvent, body: ReactNode) => {
    const r = box.current?.getBoundingClientRect()
    if (!r) return
    setTip({ x: Math.min(e.clientX - r.left + 12, r.width - 180), y: e.clientY - r.top - 36, body })
  }
  const el = tip ? <div className="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.body}</div> : null
  return { box, show, hide: () => setTip(null), el }
}

export function Stat({ label, value, sub, badge }: { label: string; value: ReactNode; sub?: ReactNode; badge?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
      {badge && <div style={{ marginTop: 6 }}>{badge}</div>}
    </div>
  )
}

export type LinePoint = { x: number; y: number | null; label?: string }

/** Lines over a numeric x (rounds, iterations). Each series is its own entity color. */
export function LineChart({ series, yMax = 1, yFormat = (v: number) => `${Math.round(v * 100)}%`, xLabel = '',
  xFormat = (v: number) => String(v), height = 220 }: {
  series: { name: string; color: string; points: LinePoint[] }[]; yMax?: number; yFormat?: (v: number) => string
  xLabel?: string; xFormat?: (v: number) => string; height?: number }) {
  const t = useTip()
  const W = 560, H = height, L = 44, R = 16, T = 12, B = 30
  const xs = series.flatMap((s) => s.points.map((p) => p.x))
  const x0 = Math.min(...xs, 0), x1 = Math.max(...xs, 1)
  const sx = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R)
  const sy = (y: number) => T + (1 - y / yMax) * (H - T - B)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax)
  const xticks = Array.from(new Set(xs)).sort((a, b) => a - b)
  const showX = xticks.length > 12 ? xticks.filter((_, i) => i % Math.ceil(xticks.length / 8) === 0) : xticks
  return (
    <div className="chart-box" ref={t.box}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={series.map((s) => s.name).join(', ')}>
        {ticks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} />
            <text x={L - 8} y={sy(v) + 4} textAnchor="end">{yFormat(v)}</text>
          </g>
        ))}
        {showX.map((x) => <text key={x} x={sx(x)} y={H - 10} textAnchor="middle">{xLabel ? `${xLabel} ${xFormat(x)}` : xFormat(x)}</text>)}
        {series.map((s) => {
          const pts = s.points.filter((p) => p.y !== null) as { x: number; y: number; label?: string }[]
          const d = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join(' ')
          return (
            <g key={s.name}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
              {pts.length <= 20 && pts.map((p) => (
                <g key={p.x}>
                  <circle cx={sx(p.x)} cy={sy(p.y)} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                  <circle cx={sx(p.x)} cy={sy(p.y)} r={14} fill="transparent"
                    onMouseMove={(e) => t.show(e, <><b>{s.name}</b> · {xLabel} {xFormat(p.x)}: {yFormat(p.y)}{p.label ? ` · ${p.label}` : ''}</>)}
                    onMouseLeave={t.hide} />
                </g>
              ))}
              {pts.length > 20 && pts.map((p, i) => i % 1 === 0 && (
                <rect key={p.x} x={sx(p.x) - 4} y={T} width={8} height={H - T - B} fill="transparent"
                  onMouseMove={(e) => t.show(e, <><b>{s.name}</b> · {xFormat(p.x)}: {yFormat(p.y)}</>)} onMouseLeave={t.hide} />
              ))}
            </g>
          )
        })}
      </svg>
      {series.length > 1 && (
        <div className="legend">{series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
      )}
      {t.el}
    </div>
  )
}

/** Horizontal bars, one measure. Optional second measure shown as a separate series (same scale). */
export function BarChart({ rows, max = 1, format = (v: number) => `${Math.round(v * 100)}%`, series }: {
  rows: { label: string; values: (number | null)[]; note?: string }[]; max?: number; format?: (v: number) => string
  series: { name: string; color: string }[] }) {
  const t = useTip()
  const W = 560, L = 190, R = 52, barH = 12, gap = 2, groupGap = 14
  const groupH = series.length * barH + (series.length - 1) * gap
  const H = rows.length * (groupH + groupGap) + 8
  const sx = (v: number) => (v / max) * (W - L - R)
  return (
    <div className="chart-box" ref={t.box}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={series.map((s) => s.name).join(', ')}>
        {[0, 0.5, 1].map((f) => (
          <line key={f} className="gridline" x1={L + sx(f * max)} x2={L + sx(f * max)} y1={0} y2={H - 4} />
        ))}
        {rows.map((r, i) => {
          const y0 = i * (groupH + groupGap) + 4
          return (
            <g key={r.label}>
              <text className="label-strong" x={L - 10} y={y0 + groupH / 2 + 4} textAnchor="end">{r.label}</text>
              {r.values.map((v, j) => {
                const y = y0 + j * (barH + gap)
                if (v === null) return <text key={j} x={L + 4} y={y + barH - 2}>n/a</text>
                const w = Math.max(sx(v), 2)
                return (
                  <g key={j} onMouseMove={(e) => t.show(e, <><b>{r.label}</b> · {series[j].name}: {format(v)}{r.note ? ` · ${r.note}` : ''}</>)}
                    onMouseLeave={t.hide}>
                    <path d={`M${L},${y} h${w - 4} a4,4 0 0 1 4,4 v${barH - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`} fill={series[j].color} />
                    <rect x={L} y={y - 1} width={W - L - R} height={barH + 2} fill="transparent" />
                    <text x={L + w + 6} y={y + barH - 2}>{format(v)}</text>
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
      {series.length > 1 && (
        <div className="legend">{series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
      )}
      {t.el}
    </div>
  )
}

export function Badge({ ok, yes = 'pass', no = 'fail' }: { ok: boolean | null | undefined; yes?: string; no?: string }) {
  if (ok === null || ok === undefined) return <span className="badge neutral">–</span>
  return <span className={`badge ${ok ? 'good' : 'bad'}`}>{ok ? '✓' : '✕'} {ok ? yes : no}</span>
}

export function Redacted({ text }: { text: string }) {
  const parts = text.split(/(\[[A-Z_]+_\d+\])/g)
  return <>{parts.map((p, i) => (/^\[[A-Z_]+_\d+\]$/.test(p) ? <span key={i} className="ph">{p}</span> : p))}</>
}
