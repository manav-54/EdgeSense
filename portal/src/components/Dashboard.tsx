import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useQuery } from '../api'
import type {
  AgentRow,
  IntentRow,
  Overview,
  SentimentPoint,
  ViolationRow,
} from '../types'
import { Card, Legend, Tile, TooltipCard, axisProps, gridProps } from './chrome'

const SERIES_1 = 'var(--series-1)'
const SERIES_2 = 'var(--series-2)'

function hourLabel(iso: string): string {
  const date = new Date(iso)
  return `${date.getMonth() + 1}/${date.getDate()} ${String(date.getHours()).padStart(2, '0')}:00`
}

export function Dashboard() {
  const [days, setDays] = useState(30)

  const overview = useQuery<Overview>(`/api/dashboard/overview?days=${days}`, [days])
  const sentiment = useQuery<SentimentPoint[]>(`/api/dashboard/sentiment?days=${days}`, [days])
  const intents = useQuery<IntentRow[]>(`/api/dashboard/intents?days=${days}`, [days])
  const violations = useQuery<ViolationRow[]>(`/api/dashboard/violations?days=${days}`, [days])
  const agents = useQuery<AgentRow[]>(`/api/dashboard/agents?days=${days}`, [days])

  const o = overview.data

  return (
    <div>
      <div className="controls">
        <label htmlFor="range">Range</label>
        <select
          id="range"
          value={days}
          onChange={(event) => setDays(Number(event.target.value))}
        >
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
        {overview.error && <span className="err">API error: {overview.error}</span>}
      </div>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Tile label="Calls analysed" value={o ? o.calls.toLocaleString() : '—'} />
        <Tile
          label="Escalated"
          value={o ? `${((100 * o.escalations) / Math.max(o.calls, 1)).toFixed(1)}%` : '—'}
          note={o ? `${o.escalations} of ${o.calls} calls` : undefined}
        />
        <Tile
          label="Compliance violations"
          value={o ? o.violations.toLocaleString() : '—'}
          note={o ? `${(o.violations / Math.max(o.calls, 1)).toFixed(2)} per call` : undefined}
        />
        <Tile
          label="PII spans redacted"
          value={o ? o.redactions.toLocaleString() : '—'}
          note="Removed at the edge; never transmitted"
        />
      </div>

      <div className="grid cols-2">
        {/* Change over time -> line. Two series, so a legend is required. */}
        <Card
          title="Customer sentiment over time"
          subtitle="Mean sentiment at the start and end of each call, bucketed hourly."
          table={
            <table className="data">
              <thead>
                <tr>
                  <th>Hour</th>
                  <th className="num">Calls</th>
                  <th className="num">Start</th>
                  <th className="num">End</th>
                  <th className="num">Delta</th>
                </tr>
              </thead>
              <tbody>
                {(sentiment.data ?? []).slice(-40).map((row) => (
                  <tr key={row.hour}>
                    <td>{hourLabel(row.hour)}</td>
                    <td className="num">{row.calls}</td>
                    <td className="num">{row.avg_start.toFixed(2)}</td>
                    <td className="num">{row.avg_end.toFixed(2)}</td>
                    <td className="num">{row.avg_delta.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          <Legend
            items={[
              { name: 'Start of call', color: SERIES_1 },
              { name: 'End of call', color: SERIES_2 },
            ]}
          />
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={sentiment.data ?? []} margin={{ top: 4, right: 12, bottom: 0, left: -18 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="hour" tickFormatter={hourLabel} minTickGap={44} {...axisProps} />
              <YAxis domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 1]} {...axisProps} />
              <Tooltip
                cursor={{ stroke: 'var(--axis)', strokeWidth: 1 }}
                content={({ active, payload, label }) =>
                  active && payload?.length ? (
                    <TooltipCard
                      title={hourLabel(String(label))}
                      rows={[
                        { name: 'Start', value: Number(payload[0]?.value).toFixed(2), color: SERIES_1 },
                        { name: 'End', value: Number(payload[1]?.value).toFixed(2), color: SERIES_2 },
                      ]}
                    />
                  ) : null
                }
              />
              <Line
                type="monotone"
                dataKey="avg_start"
                stroke={SERIES_1}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
              />
              <Line
                type="monotone"
                dataKey="avg_end"
                stroke={SERIES_2}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--surface-1)' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Magnitude by category -> horizontal bar. One series, so no legend. */}
        <Card
          title="Violation rate by policy"
          subtitle="Share of calls in the period with at least one violation of each policy."
          table={
            <table className="data">
              <thead>
                <tr>
                  <th>Policy</th>
                  <th className="num">Calls</th>
                  <th className="num">Rate</th>
                  <th className="num">Severe</th>
                </tr>
              </thead>
              <tbody>
                {(violations.data ?? []).map((row) => (
                  <tr key={row.policy_id}>
                    <td>{row.policy_id}</td>
                    <td className="num">{row.calls_with_violation}</td>
                    <td className="num">{row.violation_rate_pct}%</td>
                    <td className="num">{row.severe_events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          <ResponsiveContainer width="100%" height={230}>
            <BarChart
              layout="vertical"
              data={violations.data ?? []}
              margin={{ top: 4, right: 44, bottom: 0, left: 12 }}
            >
              <CartesianGrid {...gridProps} horizontal={false} vertical />
              <XAxis type="number" unit="%" {...axisProps} />
              <YAxis type="category" dataKey="policy_id" width={86} {...axisProps} />
              <Tooltip
                cursor={{ fill: 'color-mix(in srgb, var(--series-1) 8%, transparent)' }}
                content={({ active, payload }) => {
                  const row = payload?.[0]?.payload as ViolationRow | undefined
                  return active && row ? (
                    <TooltipCard
                      title={row.policy_id}
                      rows={[
                        { name: 'Violation rate', value: `${row.violation_rate_pct}%` },
                        { name: 'Calls affected', value: `${row.calls_with_violation}` },
                        { name: 'Severe events', value: `${row.severe_events}` },
                      ]}
                    />
                  ) : null
                }}
              />
              <Bar
                dataKey="violation_rate_pct"
                fill={SERIES_1}
                radius={[0, 4, 4, 0]}
                barSize={16}
                label={{
                  position: 'right',
                  fill: 'var(--text-secondary)',
                  fontSize: 11,
                  formatter: (value: number) => `${value}%`,
                }}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="Intent distribution"
          subtitle="What customers called about, by share of calls."
          table={
            <table className="data">
              <thead>
                <tr>
                  <th>Intent</th>
                  <th className="num">Calls</th>
                  <th className="num">Share</th>
                  <th className="num">Escalated</th>
                  <th className="num">Resolved</th>
                  <th className="num">Turns</th>
                </tr>
              </thead>
              <tbody>
                {(intents.data ?? []).map((row) => (
                  <tr key={row.primary_intent}>
                    <td>{row.primary_intent.replace(/_/g, ' ')}</td>
                    <td className="num">{row.calls}</td>
                    <td className="num">{row.share_pct}%</td>
                    <td className="num">{row.escalation_rate_pct}%</td>
                    <td className="num">{row.resolved_pct}%</td>
                    <td className="num">{row.avg_turns}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              layout="vertical"
              data={(intents.data ?? []).slice(0, 10)}
              margin={{ top: 4, right: 40, bottom: 0, left: 12 }}
            >
              <CartesianGrid {...gridProps} horizontal={false} vertical />
              <XAxis type="number" {...axisProps} />
              <YAxis
                type="category"
                dataKey="primary_intent"
                width={132}
                tickFormatter={(value: string) => value.replace(/_/g, ' ')}
                {...axisProps}
              />
              <Tooltip
                cursor={{ fill: 'color-mix(in srgb, var(--series-1) 8%, transparent)' }}
                content={({ active, payload }) => {
                  const row = payload?.[0]?.payload as IntentRow | undefined
                  return active && row ? (
                    <TooltipCard
                      title={row.primary_intent.replace(/_/g, ' ')}
                      rows={[
                        { name: 'Calls', value: `${row.calls}` },
                        { name: 'Share', value: `${row.share_pct}%` },
                        { name: 'Escalation rate', value: `${row.escalation_rate_pct}%` },
                        { name: 'Resolved', value: `${row.resolved_pct}%` },
                      ]}
                    />
                  ) : null
                }}
              />
              <Bar
                dataKey="calls"
                fill={SERIES_1}
                radius={[0, 4, 4, 0]}
                barSize={15}
                label={{ position: 'right', fill: 'var(--text-secondary)', fontSize: 11 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="Per-agent rollup"
          subtitle="From the agent_daily aggregate. Agents with at least 3 calls in the period."
          wide
        >
          <div className="scroll-x">
            <table className="data">
              <thead>
                <tr>
                  <th>Agent</th>
                  <th className="num">Calls</th>
                  <th className="num">Escalation rate</th>
                  <th className="num">Violations</th>
                  <th className="num">Resolved</th>
                  <th className="num">Sentiment Δ</th>
                  <th className="num">Turns/call</th>
                </tr>
              </thead>
              <tbody>
                {(agents.data ?? []).map((row) => (
                  <tr key={row.agent_id}>
                    <td>{row.agent_id}</td>
                    <td className="num">{row.call_count}</td>
                    <td className="num">{row.escalation_rate_pct}%</td>
                    <td className="num">{row.violation_count}</td>
                    <td className="num">{row.resolved_pct}%</td>
                    <td
                      className="num"
                      style={{
                        color: row.sentiment_delta >= 0 ? 'var(--success-text)' : 'var(--status-critical)',
                      }}
                    >
                      {row.sentiment_delta >= 0 ? '+' : ''}
                      {row.sentiment_delta.toFixed(2)}
                    </td>
                    <td className="num">{row.turns_per_call}</td>
                  </tr>
                ))}
                {(agents.data ?? []).length === 0 && (
                  <tr>
                    <td colSpan={7} className="empty">
                      No agent rollups in this period.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  )
}
