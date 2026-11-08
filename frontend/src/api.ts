// Thin client for the collector API (backend/app/main.py). Same origin in prod; Vite proxies /v1 in dev.
export type Json = any // eslint-disable-line @typescript-eslint/no-explicit-any

export async function get<T = Json>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${r.status} ${path}`)
  return r.json()
}

export const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`
export const secs = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`
export const usd = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `$${v.toFixed(2)}`)
export const when = (t: number) => new Date(t * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

export const RUN_LABELS: Record<string, string> = {
  teacher: 'Sonnet 5.5 (cloud baseline)',
  'base-e4b': 'Gemma 4 E4B, no fine-tune',
  'base-e2b': 'Gemma 4 E2B, no fine-tune',
  'ft-e4b': 'E4B fine-tuned',
  'ft-e2b': 'E2B fine-tuned',
  'hybrid-e4b': 'E4B fine-tuned + router',
  'hybrid-e2b': 'E2B fine-tuned + router',
  'teacher-datagen': 'Teacher data generation',
}
export const runLabel = (kind: string) => RUN_LABELS[kind] ?? kind
