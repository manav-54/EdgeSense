"""Edge-agent metrics, plus the shared logging and tracing helpers.

Logging, scrubbing, and trace setup live in ``edgesense_core.obs`` so the edge
agent and the worker cannot drift apart on a privacy control. Only the metric
definitions -- which are genuinely per-service -- are declared here.
"""

from __future__ import annotations

import logging
from typing import Any

from edgesense_core.obs import (  # re-exported for call sites in this service
    FieldLogger,
    JSONFormatter,
    current_traceparent,
    get_logger,
    scrub,
    setup_logging as _setup_logging,
    setup_tracing,
    tracer,
)

__all__ = [
    "FieldLogger", "JSONFormatter", "current_traceparent", "get_logger",
    "scrub", "setup_logging", "setup_tracing", "tracer", "start_metrics",
    "METRICS_AVAILABLE", "SEGMENTS_EMITTED", "REDACTIONS", "HELD_FRAGMENTS",
    "ASR_LATENCY", "REDACT_LATENCY", "SEND_LATENCY", "ACTIVE_CALLS",
]


def setup_logging(level: str = "info") -> None:
    _setup_logging("edge-agent", level)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    SEGMENTS_EMITTED = Counter(
        "edgesense_edge_segments_emitted_total",
        "Transcript segments sent to ingest.",
        ["call_id_present", "is_final"],
    )
    REDACTIONS = Counter(
        "edgesense_edge_redactions_total",
        "PII spans redacted, by type and detector.",
        ["pii_type", "detector"],
    )
    HELD_FRAGMENTS = Counter(
        "edgesense_edge_held_fragments_total",
        "Trailing digit runs withheld pending the next segment.",
    )
    ASR_LATENCY = Histogram(
        "edgesense_edge_asr_seconds",
        "Wall time for one ASR decode.",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
    REDACT_LATENCY = Histogram(
        "edgesense_edge_redact_seconds",
        "Wall time for one redaction pass.",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )
    SEND_LATENCY = Histogram(
        "edgesense_edge_send_seconds",
        "Wall time to hand a segment to the transport.",
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 2.0),
    )
    ACTIVE_CALLS = Gauge("edgesense_edge_active_calls", "Calls currently streaming.")
    METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    METRICS_AVAILABLE = False

    def start_http_server(*_a: Any, **_k: Any) -> None:  # type: ignore[misc]
        return None


def start_metrics(port: int) -> None:
    if not METRICS_AVAILABLE:
        return
    try:
        start_http_server(port)
    except OSError:
        # Port already bound (another agent on this host). Metrics are
        # best-effort; the call matters more.
        logging.getLogger(__name__).warning("metrics port %d unavailable", port)
