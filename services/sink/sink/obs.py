"""Sink metrics, plus the shared logging and tracing helpers."""

from __future__ import annotations

import logging
from typing import Any

from edgesense_core.obs import (
    context_from_traceparent,
    get_logger,
    scrub,
    setup_logging as _setup_logging,
    setup_tracing,
    tracer,
)

__all__ = [
    "context_from_traceparent", "get_logger", "scrub", "setup_logging",
    "setup_tracing", "tracer", "start_metrics", "METRICS_AVAILABLE",
]


def setup_logging(level: str = "info") -> None:
    _setup_logging("sink", level)


try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    INSIGHTS_CONSUMED = Counter(
        "edgesense_sink_insights_consumed_total",
        "Insight envelopes read from Kafka.",
        ["kind"],
    )
    ROWS_WRITTEN = Counter(
        "edgesense_sink_rows_written_total",
        "Rows inserted into ClickHouse.",
        ["table"],
    )
    ROWS_DROPPED = Counter(
        "edgesense_sink_rows_dropped_total",
        "Rows shed after the write buffer overflowed.",
        ["table"],
    )
    WRITE_ERRORS = Counter(
        "edgesense_sink_write_errors_total",
        "Failed ClickHouse inserts.",
        ["table"],
    )
    BUFFER_DEPTH = Gauge(
        "edgesense_sink_buffer_depth",
        "Rows waiting to be flushed.",
        ["table"],
    )
    FLUSH_LATENCY = Histogram(
        "edgesense_sink_flush_seconds",
        "Time for one batch insert.",
        ["table"],
        buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    )
    CONSUMER_LAG = Gauge(
        "edgesense_sink_consumer_lag",
        "Estimated messages behind the log head.",
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
