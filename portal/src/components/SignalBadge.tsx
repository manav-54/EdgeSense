import { useState } from 'react'
import type { EvidenceSpan, Severity, SignalType } from '../types'

/**
 * Severity glyphs. Colour never carries the meaning on its own -- each badge
 * shows a glyph and the severity word, so the distinction survives colour
 * blindness, greyscale printing and forced-colors mode.
 */
const GLYPH: Record<Severity, string> = {
  critical: '◆', // filled diamond
  high: '▲', // triangle
  medium: '●', // circle
  low: '○', // hollow circle
  info: '•', // dot
}

const TYPE_LABEL: Record<SignalType, string> = {
  compliance_violation: 'Compliance',
  escalation_risk: 'Escalation',
  sentiment_shift: 'Sentiment',
  intent: 'Intent',
}

function formatMs(ms: number): string {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export interface BadgeProps {
  severity: Severity
  type: SignalType
  label: string
  confidence: number
  rationale?: string
  policyId?: string | null
  evidence: EvidenceSpan[]
}

/**
 * A risk badge that reveals its evidence on hover.
 *
 * The evidence card is the point of the component. Every signal in this system
 * carries the transcript span that justifies it, and a supervisor acting on a
 * compliance flag needs to see the words before they act -- an unsourced badge
 * is an accusation.
 */
export function SignalBadge({
  severity,
  type,
  label,
  confidence,
  rationale,
  policyId,
  evidence,
}: BadgeProps) {
  const [open, setOpen] = useState(false)

  return (
    <span
      className="evidence-wrap"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span className={`badge ${severity}`} tabIndex={0} role="button" aria-expanded={open}>
        <span className="glyph" aria-hidden="true">
          {GLYPH[severity]}
        </span>
        {TYPE_LABEL[type]}: {label}
        <span style={{ opacity: 0.72, fontWeight: 400 }}>
          {severity} · {(confidence * 100).toFixed(0)}%
        </span>
      </span>

      {open && (
        <span className="evidence-card" role="tooltip">
          <span className="head">
            {TYPE_LABEL[type]} — {label}
            {policyId ? ` · ${policyId}` : ''}
          </span>
          {evidence.length === 0 ? (
            <span className="meta">No evidence recorded (this signal would not be published).</span>
          ) : (
            evidence.map((span, i) => (
              <span key={i} style={{ display: 'block' }}>
                <blockquote>{span.quote}</blockquote>
                <span className="meta">
                  {span.speaker} · turn {span.seq} · {formatMs(span.start_ms)}–
                  {formatMs(span.end_ms)}
                </span>
              </span>
            ))
          )}
          {rationale && (
            <span className="meta" style={{ display: 'block', marginTop: 8 }}>
              {rationale}
            </span>
          )}
        </span>
      )}
    </span>
  )
}
