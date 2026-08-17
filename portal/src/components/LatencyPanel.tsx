import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useQuery } from '../api'
import type { LatencyRow } from '../types'
import { Card, Legend, Tile, TooltipCard, axisProps, gridProps } from './chrome'

/** p50/p95/p99 are ordered, so they get a single-hue ordinal ramp, not
 *  categorical hues. Validated for monotone lightness in both modes. */
const RAMP = ['var(--ramp-1)', 'var(--ramp-2)', 'var(--ramp-3)']

const BUDGET_MS = 2000

const STAGE_LABEL: Record<string, string> = {
  asr: 'ASR (edge)',
  redact: 'Redaction (edge)',
  ingest: 'Ingest',
  queue: 'Kafka queue',
  analyze: 'Analysis',
  llm: 'LLM call',
  e2e: 'End to end',
}

const STAGE_NOTE: Record<string, string> = {
  asr: 'Local whisper decode, on the operator machine.',
  redact: 'Detection plus substitution, per segment.',
  ingest: 'Validation, dedupe and Kafka publish.',
  queue: 'Time spent waiting in the broker.',
  analyze: 'Fast path plus agent path for one window.',
  llm: 'Provider round-trip inside the agent loop.',
  e2e: 'Segment emitted at the edge to signal published. This is the SLO.',
}

export function LatencyPanel() {
  const [days, setDays] = useState(7)
  const { data, error, loading } = useQuery<LatencyRow[]>(`/api/latency?days=${days}`, [days])

  const rows = data ?? []
  const e2e = rows.find((row) => row.stage === 'e2e')
  const chartData = rows.map((row) => ({ ...row, label: STAGE_LABEL[row.stage] ?? row.stage }))
  const worst = Math.max(BUDGET_MS, ...rows.map((row) => row.p99_ms), 0)

  return (
    <div>
      <div className="controls">
        <label htmlFor="lrange">Range</label>
        <select id="lrange" value={days} onChange={(event) => setDays(Number(event.target.value))}>
          <option value={1}>Last 24 hours</option>
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
        </select>
        {error && <span className="err">API error: {error}</span>}
      </div>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Tile
          label="End-to-end p95"
          value={e2e ? `${e2e.p95_ms.toFixed(0)} ms` : '—'}
          note={`Budget ${BUDGET_MS} ms`}
        />
        <Tile
          label="End-to-end p99"
          value={e2e ? `${e2e.p99_ms.toFixed(0)} ms` : '—'}
        />
        <Tile
          label="Over budget"
          value={e2e ? `${e2e.pct_over_budget.toFixed(2)}%` : '—'}
          note="Share of segments breaching 2 s"
        />
        <Tile
          label="Samples"
          value={rows.length ? rows[0]!.samples.toLocaleString() : '—'}
          note="Per stage, in the selected range"
        />
      </div>

      <Card
        title="Pipeline latency by stage"
        subtitle="p50 / p95 / p99 per stage. The dashed line is the 2 s end-to-end budget."
        table={
          <table className="data">
            <thead>
              <tr>
                <th>Stage</th>
                <th className="num">Samples</th>
                <th className="num">p50</th>
                <th className="num">p95</th>
                <th className="num">p99</th>
                <th className="num">max</th>
                <th className="num">over budget</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.stage}>
                  <td>
                    {STAGE_LABEL[row.stage] ?? row.stage}
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {STAGE_NOTE[row.stage]}
                    </div>
                  </td>
                  <td className="num">{row.samples.toLocaleString()}</td>
                  <td className="num">{row.p50_ms.toFixed(1)}</td>
                  <td className="num">{row.p95_ms.toFixed(1)}</td>
                  <td className="num">{row.p99_ms.toFixed(1)}</td>
                  <td className="num">{row.max_ms.toFixed(1)}</td>
                  <td className="num">{row.pct_over_budget.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      >
        <Legend
          items={[
            { name: 'p50', color: RAMP[0]! },
            { name: 'p95', color: RAMP[1]! },
            { name: 'p99', color: RAMP[2]! },
          ]}
        />
        {loading && <div className="empty">Loading…</div>}
        {!loading && rows.length === 0 && (
          <div className="empty">
            No latency samples yet. Run a call or the load test:
            <br />
            <code style={{ fontSize: 12 }}>make loadtest</code>
          </div>
        )}
        {rows.length > 0 && (
          <ResponsiveContainer width="100%" height={70 + rows.length * 62}>
            <BarChart
              layout="vertical"
              data={chartData}
              margin={{ top: 4, right: 56, bottom: 0, left: 12 }}
              barGap={2}
            >
              <CartesianGrid {...gridProps} horizontal={false} vertical />
              {/* Log scale, because these stages differ by four orders of
                  magnitude: redaction runs in ~0.3 ms and the budget line
                  sits at 2000 ms. On a linear axis every stage that matters
                  collapses onto the y-axis and the chart shows nothing. */}
              <XAxis
                type="number"
                scale="log"
                unit=" ms"
                domain={[0.1, Math.max(2400, Math.ceil(worst * 1.2))]}
                ticks={[0.1, 1, 10, 100, 1000, 2000]}
                allowDataOverflow
                {...axisProps}
              />
              <YAxis type="category" dataKey="label" width={132} {...axisProps} />
              <Tooltip
                cursor={{ fill: 'color-mix(in srgb, var(--series-1) 8%, transparent)' }}
                content={({ active, payload }) => {
                  const row = payload?.[0]?.payload as LatencyRow | undefined
                  return active && row ? (
                    <TooltipCard
                      title={STAGE_LABEL[row.stage] ?? row.stage}
                      rows={[
                        { name: 'p50', value: `${row.p50_ms.toFixed(1)} ms`, color: RAMP[0] },
                        { name: 'p95', value: `${row.p95_ms.toFixed(1)} ms`, color: RAMP[1] },
                        { name: 'p99', value: `${row.p99_ms.toFixed(1)} ms`, color: RAMP[2] },
                        { name: 'samples', value: row.samples.toLocaleString() },
                      ]}
                    />
                  ) : null
                }}
              />
              <ReferenceLine
                x={BUDGET_MS}
                stroke="var(--status-critical)"
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={{
                  value: '2 s budget',
                  position: 'top',
                  fill: 'var(--status-critical)',
                  fontSize: 11,
                }}
              />
              <Bar dataKey="p50_ms" fill={RAMP[0]} radius={[0, 4, 4, 0]} barSize={13} />
              <Bar dataKey="p95_ms" fill={RAMP[1]} radius={[0, 4, 4, 0]} barSize={13} />
              <Bar dataKey="p99_ms" fill={RAMP[2]} radius={[0, 4, 4, 0]} barSize={13} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Card>
    </div>
  )
}
