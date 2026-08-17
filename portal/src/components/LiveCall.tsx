import { useMemo, useState } from 'react'
import { useLiveFeed, useQuery } from '../api'
import type { CallRow, Severity, Signal, StoredSignal } from '../types'
import { SignalBadge } from './SignalBadge'
import { Transcript } from './Transcript'
import { Card } from './chrome'

const SEVERITY_ORDER: Severity[] = ['critical', 'high', 'medium', 'low', 'info']

/** ClickHouse stores evidence as parallel arrays; rehydrate into spans. */
function fromStored(stored: StoredSignal, callId: string): Signal {
  return {
    signal_id: stored.signal_id,
    call_id: callId,
    type: stored.signal_type,
    label: stored.label,
    severity: stored.severity,
    confidence: stored.confidence,
    rationale: stored.rationale,
    policy_id: stored.policy_id || null,
    window_start_ms: 0,
    window_end_ms: 0,
    emitted_at: stored.emitted_at,
    latency: { segment_to_signal_ms: stored.latency_segment_to_signal_ms },
    model_name: stored.model_name,
    prompt_version: stored.prompt_version,
    evidence: (stored.evidence_seq ?? []).map((seq, i) => ({
      seq,
      start_ms: stored.evidence_start_ms?.[i] ?? 0,
      end_ms: stored.evidence_end_ms?.[i] ?? 0,
      speaker: stored.evidence_speaker?.[i] ?? 'unknown',
      quote: stored.evidence_quote?.[i] ?? '',
    })),
  }
}

export function LiveCall() {
  const live = useLiveFeed()
  const [selected, setSelected] = useState<string | null>(null)

  const recent = useQuery<CallRow[]>('/api/calls?limit=25')
  const detail = useQuery<{ signals: StoredSignal[]; summary: CallRow | null }>(
    selected ? `/api/calls/${encodeURIComponent(selected)}` : '/api/health',
    [selected],
  )

  // Calls currently streaming, newest first.
  const liveCallIds = useMemo(() => {
    const seen: string[] = []
    for (let i = live.segments.length - 1; i >= 0; i--) {
      const id = live.segments[i]!.call_id
      if (!seen.includes(id)) seen.push(id)
    }
    return seen
  }, [live.segments])

  const activeCall = selected ?? liveCallIds[0] ?? null
  const isLive = activeCall !== null && liveCallIds.includes(activeCall)

  const segments = useMemo(
    () => live.segments.filter((segment) => segment.call_id === activeCall),
    [live.segments, activeCall],
  )

  const signals: Signal[] = useMemo(() => {
    if (isLive) {
      return live.signals.filter((signal) => signal.call_id === activeCall)
    }
    if (!activeCall || !detail.data?.signals) return []
    return detail.data.signals.map((stored) => fromStored(stored, activeCall))
  }, [isLive, live.signals, activeCall, detail.data])

  const ranked = useMemo(
    () =>
      [...signals].sort(
        (a, b) =>
          SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity) ||
          b.confidence - a.confidence,
      ),
    [signals],
  )

  return (
    <div>
      <div className="controls">
        <span className="status-line">
          <span className={`dot ${live.connected ? 'on' : 'off'}`} aria-hidden="true" />
          {live.connected ? 'Feed connected' : 'Feed disconnected'}
          {live.connected && !live.kafkaConnected && ' · broker unreachable (dashboard still live)'}
        </span>
        <label htmlFor="call" style={{ marginLeft: 12 }}>
          Call
        </label>
        <select
          id="call"
          value={activeCall ?? ''}
          onChange={(event) => setSelected(event.target.value || null)}
        >
          <option value="">
            {liveCallIds.length > 0 ? `Live: ${liveCallIds[0]}` : 'Select a completed call'}
          </option>
          {liveCallIds.map((id) => (
            <option key={id} value={id}>
              ● live — {id}
            </option>
          ))}
          {(recent.data ?? []).map((row) => (
            <option key={row.call_id} value={row.call_id}>
              {row.call_id} — {row.primary_intent.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
        {activeCall && (
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {isLive ? 'streaming' : 'from history'} · {signals.length} signals
          </span>
        )}
      </div>

      <div className="grid cols-2">
        <Card
          title="Transcript"
          subtitle="Redacted at the edge. Placeholders mark values that never left the client."
        >
          {isLive ? (
            <Transcript segments={segments} />
          ) : detail.data?.summary ? (
            <div>
              <p style={{ fontSize: 13.5, lineHeight: 1.55, margin: '0 0 12px' }}>
                {detail.data.summary.summary}
              </p>
              <div className="status-line">
                {detail.data.summary.turn_count} turns ·{' '}
                {detail.data.summary.redaction_count} PII spans redacted ·{' '}
                {detail.data.summary.resolution.replace(/_/g, ' ')}
              </div>
              <p className="sub" style={{ marginTop: 12 }}>
                Full turn-by-turn text is only retained for calls still streaming; completed
                calls keep the summary and the evidence spans cited by their signals.
              </p>
            </div>
          ) : (
            <Transcript segments={segments} />
          )}
        </Card>

        <Card
          title="Signals"
          subtitle="Every signal cites the transcript span that justifies it. Hover a badge to read it."
        >
          {ranked.length === 0 ? (
            <div className="empty">
              No signals for this call yet.
              <br />
              Signals appear as the sliding window advances.
            </div>
          ) : (
            <div className="signal-feed">
              {ranked.map((signal) => (
                <div className="signal-row" key={signal.signal_id}>
                  <div className="line1">
                    <SignalBadge
                      severity={signal.severity}
                      type={signal.type}
                      label={signal.label}
                      confidence={signal.confidence}
                      rationale={signal.rationale}
                      policyId={signal.policy_id}
                      evidence={signal.evidence}
                    />
                    {signal.latency.segment_to_signal_ms != null && (
                      <span className="type">
                        {signal.latency.segment_to_signal_ms.toFixed(0)} ms
                      </span>
                    )}
                  </div>
                  {signal.rationale && <div className="why">{signal.rationale}</div>}
                  <div className="type">
                    {signal.model_name} · {signal.prompt_version}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
