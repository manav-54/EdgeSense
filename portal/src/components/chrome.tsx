import type { ReactNode } from 'react'

/** Shared chart chrome: recessive axes, one hairline grid, tabular tooltips. */

export const AXIS_STYLE = {
  fontSize: 11,
  fill: 'var(--text-muted)',
} as const

export const axisProps = {
  tick: AXIS_STYLE,
  tickLine: false,
  axisLine: { stroke: 'var(--axis)' },
} as const

export const gridProps = {
  stroke: 'var(--grid)',
  strokeDasharray: '0',
  vertical: false,
} as const

export interface TooltipRow {
  name: string
  value: string
  color?: string
}

export function TooltipCard({ title, rows }: { title: string; rows: TooltipRow[] }) {
  return (
    <div className="tooltip">
      <div className="t-title">{title}</div>
      {rows.map((row) => (
        <div className="t-row" key={row.name}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            {row.color && (
              <span
                aria-hidden="true"
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 2,
                  background: row.color,
                  display: 'inline-block',
                }}
              />
            )}
            {row.name}
          </span>
          <b>{row.value}</b>
        </div>
      ))}
    </div>
  )
}

/** Legend rendered outside the plot, so identity is never colour-alone. */
export function Legend({ items }: { items: { name: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span className="item" key={item.name}>
          <span className="swatch" style={{ background: item.color }} aria-hidden="true" />
          {item.name}
        </span>
      ))}
    </div>
  )
}

export function Card({
  title,
  subtitle,
  children,
  table,
  wide,
}: {
  title: string
  subtitle?: string
  children: ReactNode
  table?: ReactNode
  wide?: boolean
}) {
  return (
    <section className={wide ? 'card span-2' : 'card'}>
      <h2>{title}</h2>
      {subtitle && <p className="sub">{subtitle}</p>}
      {children}
      {table && (
        <details className="table-view">
          <summary>View as table</summary>
          <div className="scroll-x">{table}</div>
        </details>
      )}
    </section>
  )
}

export function Tile({
  label,
  value,
  note,
}: {
  label: string
  value: string
  note?: string
}) {
  return (
    <section className="card tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </section>
  )
}
