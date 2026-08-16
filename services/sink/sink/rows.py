"""Mapping from insight envelopes to ClickHouse rows.

Kept apart from both the Kafka consumer and the writer so the shape of a row
is testable without a broker or a database, and so a schema change touches one
file. The column lists here are the contract with schema.sql; the sink test
asserts they match the live table definition.
"""

from __future__ import annotations

from typing import Any

from edgesense_core.contracts import CallInsights, Signal
from edgesense_core.timeutil import iso_to_datetime, utc_now

SIGNAL_TABLE = "edgesense.signals"
SIGNAL_COLUMNS = [
    "signal_id", "call_id", "agent_id", "signal_type", "label", "severity",
    "policy_id", "confidence", "rationale",
    "emitted_at", "window_start_ms", "window_end_ms",
    "evidence_seq", "evidence_start_ms", "evidence_end_ms",
    "evidence_speaker", "evidence_quote",
    "latency_asr_ms", "latency_redact_ms", "latency_ingest_ms",
    "latency_queue_ms", "latency_analyze_ms", "latency_llm_ms",
    "latency_segment_to_signal_ms",
    "model_name", "prompt_version", "trace_id",
]

SUMMARY_TABLE = "edgesense.call_summaries"
SUMMARY_COLUMNS = [
    "call_id", "agent_id", "primary_intent", "secondary_intents", "resolution",
    "escalated", "summary", "sentiment_start", "sentiment_end",
    "compliance_violations", "disclosures_given",
    "action_items", "action_item_owners",
    "turn_count", "duration_ms", "redaction_count",
    "started_at", "ended_at", "model_name", "prompt_version", "trace_id",
]

LATENCY_TABLE = "edgesense.segment_latency"
LATENCY_COLUMNS = [
    "call_id", "seq", "agent_id", "stage", "duration_ms", "is_final", "emitted_at",
]

#: Stage name -> attribute on StageLatency. "e2e" is the SLO number.
LATENCY_STAGES = (
    ("asr", "asr_ms"),
    ("redact", "redact_ms"),
    ("ingest", "ingest_ms"),
    ("queue", "queue_ms"),
    ("analyze", "analyze_ms"),
    ("llm", "llm_ms"),
    ("e2e", "segment_to_signal_ms"),
)


def _trace_id(traceparent: str | None) -> str:
    """Extract the 32-hex trace id from a W3C traceparent."""
    if not traceparent:
        return ""
    parts = traceparent.split("-")
    return parts[1] if len(parts) >= 3 else ""


def _ts(value: str):
    try:
        return iso_to_datetime(value)
    except Exception:
        # A malformed timestamp must not lose the row. Stamping arrival time
        # keeps the insight queryable and visibly late rather than absent.
        return utc_now()


def signal_row(signal: Signal, agent_id: str | None) -> list[Any]:
    lat = signal.latency
    return [
        signal.signal_id,
        signal.call_id,
        signal.agent_id or agent_id or "",
        signal.type.value,
        signal.label,
        signal.severity.value,
        signal.policy_id or "",
        float(signal.confidence),
        signal.rationale,
        _ts(signal.emitted_at),
        int(signal.window_start_ms),
        int(signal.window_end_ms),
        [int(e.seq) for e in signal.evidence],
        [int(e.start_ms) for e in signal.evidence],
        [int(e.end_ms) for e in signal.evidence],
        [e.speaker.value for e in signal.evidence],
        [e.quote for e in signal.evidence],
        lat.asr_ms, lat.redact_ms, lat.ingest_ms, lat.queue_ms,
        lat.analyze_ms, lat.llm_ms, lat.segment_to_signal_ms,
        signal.model_name,
        signal.prompt_version,
        _trace_id(signal.traceparent),
    ]


def latency_rows(signal: Signal, agent_id: str | None) -> list[list[Any]]:
    """One row per populated stage.

    Stages that did not run are skipped rather than written as zero: a zero
    would drag every percentile down and make the latency panel report a
    system that is faster than it is.
    """
    out: list[list[Any]] = []
    emitted = _ts(signal.emitted_at)
    seq = signal.evidence[0].seq if signal.evidence else 0
    for stage, attr in LATENCY_STAGES:
        value = getattr(signal.latency, attr, None)
        if value is None:
            continue
        out.append([
            signal.call_id, int(seq), signal.agent_id or agent_id or "",
            stage, float(value), 1, emitted,
        ])
    return out


def summary_row(
    insights: CallInsights,
    *,
    turn_count: int = 0,
    duration_ms: int = 0,
    redaction_count: int = 0,
    started_at: str | None = None,
) -> list[Any] | None:
    summary = insights.summary
    if summary is None:
        return None
    ended = _ts(insights.emitted_at)
    return [
        summary.call_id,
        insights.agent_id or "",
        summary.primary_intent,
        list(summary.secondary_intents),
        summary.resolution.value,
        1 if summary.escalated else 0,
        summary.summary,
        float(summary.customer_sentiment_start),
        float(summary.customer_sentiment_end),
        list(summary.compliance_violations),
        list(summary.disclosures_given),
        [a.description for a in summary.action_items],
        [a.owner for a in summary.action_items],
        int(turn_count),
        int(duration_ms),
        int(redaction_count),
        _ts(started_at) if started_at else ended,
        ended,
        summary.model_name,
        summary.prompt_version,
        _trace_id(insights.traceparent),
    ]
