"""Worker metrics, plus the shared logging and tracing helpers."""

from __future__ import annotations

import logging
from typing import Any

from edgesense_core.obs import (
    FieldLogger,
    context_from_traceparent,
    current_traceparent,
    get_logger,
    scrub,
    setup_logging as _setup_logging,
    setup_tracing,
    tracer,
)

__all__ = [
    "FieldLogger", "context_from_traceparent", "current_traceparent",
    "get_logger", "scrub", "setup_logging", "setup_tracing", "tracer",
    "start_metrics", "METRICS_AVAILABLE",
]


def setup_logging(level: str = "info") -> None:
    _setup_logging("worker", level)


try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    SEGMENTS_CONSUMED = Counter(
        "edgesense_worker_segments_consumed_total",
        "Segments read from the transcript topic.",
        ["is_final"],
    )
    SIGNALS_EMITTED = Counter(
        "edgesense_worker_signals_emitted_total",
        "Signals published, by type and source path.",
        ["signal_type", "path"],
    )
    SIGNALS_REJECTED = Counter(
        "edgesense_worker_signals_rejected_total",
        "Model findings dropped before publication, by reason.",
        ["reason"],
    )
    SUMMARIES_EMITTED = Counter(
        "edgesense_worker_summaries_total",
        "Post-call summaries published, by outcome.",
        ["outcome"],
    )
    SCHEMA_RETRIES = Counter(
        "edgesense_worker_schema_retries_total",
        "Retries triggered by a model response failing schema validation.",
    )
    # The headline SLO metric. Buckets are dense either side of the 2s budget
    # so the p95 can be read off the histogram without interpolation error at
    # exactly the point that matters.
    SEGMENT_TO_SIGNAL = Histogram(
        "edgesense_segment_to_signal_seconds",
        "Wall-clock from segment emission at the edge to signal publication.",
        buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 10.0),
    )
    ANALYSIS_LATENCY = Histogram(
        "edgesense_worker_analysis_seconds",
        "Time for one analysis pass.",
        ["path"],
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    LLM_LATENCY = Histogram(
        "edgesense_worker_llm_seconds",
        "Time for one provider call.",
        ["provider"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    )
    LLM_TOKENS = Counter(
        "edgesense_worker_llm_tokens_total",
        "Tokens consumed, by provider and direction.",
        ["provider", "direction"],
    )
    LLM_ERRORS = Counter(
        "edgesense_worker_llm_errors_total",
        "Provider failures, by kind.",
        ["provider", "kind"],
    )
    AGENT_STEPS = Histogram(
        "edgesense_worker_agent_steps",
        "Tool-loop steps taken per analysis.",
        buckets=(1, 2, 3, 4, 5, 6, 8, 10),
    )
    TOOL_CALLS = Counter(
        "edgesense_worker_tool_calls_total",
        "Tool invocations, by tool and outcome.",
        ["tool", "outcome"],
    )
    ACTIVE_CALLS = Gauge(
        "edgesense_worker_active_calls", "Calls with live state in this worker."
    )
    CONSUMER_LAG = Gauge(
        "edgesense_worker_consumer_lag",
        "Estimated messages behind the log head, by partition.",
        ["topic", "partition"],
    )
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
        logging.getLogger(__name__).warning("metrics port %d unavailable", port)
