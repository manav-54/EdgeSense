/** Types mirroring the pydantic contracts in packages/edgesense_core. */

export type Speaker = 'agent' | 'customer' | 'unknown'
export type Severity = 'info' | 'low' | 'medium' | 'high' | 'critical'
export type SignalType =
  | 'sentiment_shift'
  | 'escalation_risk'
  | 'compliance_violation'
  | 'intent'

export interface RedactionRef {
  type: string
  placeholder: string
  start: number
  end: number
  detector: string
  confidence: number
}

export interface TranscriptSegment {
  call_id: string
  seq: number
  speaker: Speaker
  text: string
  is_final: boolean
  start_ms: number
  end_ms: number
  emitted_at: string
  redactions: RedactionRef[]
  asr_confidence: number
  agent_id?: string | null
}

export interface EvidenceSpan {
  seq: number
  start_ms: number
  end_ms: number
  speaker: Speaker
  quote: string
}

export interface Signal {
  signal_id: string
  call_id: string
  type: SignalType
  label: string
  severity: Severity
  confidence: number
  rationale: string
  evidence: EvidenceSpan[]
  policy_id?: string | null
  window_start_ms: number
  window_end_ms: number
  emitted_at: string
  agent_id?: string | null
  latency: { segment_to_signal_ms?: number | null; analyze_ms?: number | null }
  model_name: string
  prompt_version: string
}

/** Signals arrive from ClickHouse with evidence flattened into parallel arrays. */
export interface StoredSignal {
  signal_id: string
  signal_type: SignalType
  label: string
  severity: Severity
  policy_id: string
  confidence: number
  rationale: string
  emitted_at: string
  evidence_seq: number[]
  evidence_start_ms: number[]
  evidence_end_ms: number[]
  evidence_speaker: Speaker[]
  evidence_quote: string[]
  latency_segment_to_signal_ms: number | null
  model_name: string
  prompt_version: string
}

export interface CallRow {
  call_id: string
  agent_id: string
  primary_intent: string
  resolution: string
  escalated: number
  sentiment_start: number
  sentiment_end: number
  turn_count: number
  redaction_count: number
  violation_count: number
  compliance_violations: string[]
  started_at: string
  ended_at: string
  summary: string
}

export interface Overview {
  calls: number
  escalations: number
  violations: number
  resolved: number
  avg_sentiment_delta: number
  avg_turns: number
  redactions: number
}

export interface SentimentPoint {
  hour: string
  calls: number
  avg_start: number
  avg_end: number
  avg_delta: number
  ended_unhappy: number
}

export interface ViolationRow {
  policy_id: string
  calls_with_violation: number
  total_calls: number
  violation_rate_pct: number
  severe_events: number
  avg_confidence: number
}

export interface IntentRow {
  primary_intent: string
  calls: number
  share_pct: number
  escalated_calls: number
  escalation_rate_pct: number
  resolved_pct: number
  avg_turns: number
}

export interface AgentRow {
  agent_id: string
  call_count: number
  escalation_count: number
  escalation_rate_pct: number
  violation_count: number
  resolved_pct: number
  sentiment_delta: number
  turns_per_call: number
}

export interface LatencyRow {
  stage: string
  samples: number
  p50_ms: number
  p95_ms: number
  p99_ms: number
  max_ms: number
  pct_over_budget: number
}

export interface Policy {
  id: string
  title: string
  kind: string
  severity: string
  summary: string
  rationale: string
}

export type LiveEvent =
  | { type: 'hello'; kafka_connected: boolean; replay: LiveEvent[] }
  | { type: 'segment'; data: TranscriptSegment }
  | { type: 'signal'; data: Signal }
  | { type: 'summary'; data: unknown }
