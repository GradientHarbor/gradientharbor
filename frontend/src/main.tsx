import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import Overview from './pages/Overview'
import Evals from './pages/Evals'
import { Ledger, Traces, Training } from './pages/Data'

const TABS = { overview: 'Overview', evals: 'Eval runs', traces: 'Traces', ledger: 'Privacy ledger', training: 'Training' } as const
type Tab = keyof typeof TABS

function readTab(): Tab {
  const h = location.hash.slice(1)
  return (h in TABS ? h : 'overview') as Tab
}

function App() {
  const [tab, setTab] = useState<Tab>(readTab)
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem('gh-theme') ?? '' } catch { return '' } })
  useEffect(() => { const f = () => setTab(readTab()); addEventListener('hashchange', f); return () => removeEventListener('hashchange', f) }, [])
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
    try { localStorage.setItem('gh-theme', theme) } catch { /* private mode */ }
  }, [theme])
  const dark = theme === 'dark' || (!theme && matchMedia('(prefers-color-scheme: dark)').matches)
  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><img className="brand-mark" src="/logo-icon.svg" alt="" width="28" height="28" />
          <div><h1>GradientHarbor</h1><small>Bayside Mutual support agent · flywheel</small></div></div>
        <nav className="tabs" role="tablist">
          {(Object.keys(TABS) as Tab[]).map((t) => (
            <button key={t} role="tab" className="tab" aria-selected={tab === t} onClick={() => { location.hash = t }}>{TABS[t]}</button>
          ))}
        </nav>
        <button className="icon-btn" onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label="Toggle theme">{dark ? '☀︎' : '☾'}</button>
      </header>
      <main>
        {tab === 'overview' && <Overview />}
        {tab === 'evals' && <Evals />}
        {tab === 'traces' && <Traces />}
        {tab === 'ledger' && <Ledger />}
        {tab === 'training' && <Training />}
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
