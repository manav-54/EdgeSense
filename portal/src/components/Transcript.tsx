import type { TranscriptSegment } from '../types'

const PLACEHOLDER = /(<[A-Z_]+_\d+>)/g

/**
 * Render a redacted transcript line.
 *
 * Placeholders are styled rather than hidden: a supervisor should be able to
 * see that a card number was spoken and removed, which is different from the
 * customer never having said one. The original value is not available to this
 * process -- it never left the edge agent -- so there is nothing to reveal on
 * click, by design.
 */
function renderText(text: string) {
  return text.split(PLACEHOLDER).map((part, i) =>
    PLACEHOLDER.test(part) ? (
      <span key={i} className="placeholder" title="Redacted at the edge; never transmitted">
        {part}
      </span>
    ) : (
      <span key={i}>{part}</span>
    ),
  )
}

function clock(ms: number): string {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export function Transcript({ segments }: { segments: TranscriptSegment[] }) {
  if (segments.length === 0) {
    return (
      <div className="empty">
        No transcript yet. Start a call:
        <br />
        <code style={{ fontSize: 12 }}>make demo-call</code>
      </div>
    )
  }

  return (
    <div className="transcript">
      {segments.map((segment) => (
        <div key={`${segment.call_id}-${segment.seq}`} className={`turn ${segment.speaker}`}>
          <div className="who">
            {segment.speaker}
            <div className="ts">{clock(segment.start_ms)}</div>
          </div>
          <div className="body">{renderText(segment.text)}</div>
        </div>
      ))}
    </div>
  )
}
