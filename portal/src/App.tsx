import { useState } from 'react'
import { Dashboard } from './components/Dashboard'
import { LatencyPanel } from './components/LatencyPanel'
import { LiveCall } from './components/LiveCall'

type Tab = 'live' | 'dashboard' | 'latency'

const TABS: { id: Tab; label: string }[] = [
  { id: 'live', label: 'Live call' },
  { id: 'dashboard', label: 'Supervisor' },
  { id: 'latency', label: 'Latency' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('live')

  return (
    <div className="app">
      <header className="top">
        <h1>EdgeSense</h1>
        <span className="tagline">
          Raw audio and PII never leave the client. Only redacted, structured data reaches here.
        </span>
        <nav className="tabs">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              onClick={() => setTab(entry.id)}
              aria-current={tab === entry.id ? 'page' : undefined}
            >
              {entry.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === 'live' && <LiveCall />}
      {tab === 'dashboard' && <Dashboard />}
      {tab === 'latency' && <LatencyPanel />}
    </div>
  )
}
