"""Structured logging, metrics, and tracing for the edge agent.

The logger has one unusual property: it refuses to emit a message that looks
like it contains PII. The edge agent is the only process that ever holds raw
card numbers, so a stray ``log.info("heard %s", text)`` on the wrong line is
the most plausible way this system leaks. Rather than rely on nobody ever
writing that line, the formatter scrubs long digit runs and email-shaped
tokens from every record on the way out.

That is a backstop, not a licence -- code should not log raw transcripts. But
backstops are what make the guarantee survive contact with maintenance.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any

_LONG_DIGITS = re.compile(r"\d[\d\s\-.]{6,}\d")
_EMAILISH = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def scrub(text: str) -> str:
    """Remove anything digit- or email-shaped from a log line."""
    out = _LONG_DIGITS.sub("[redacted:digits]", text)
    return _EMAILISH.sub("[redacted:email]", out)


class JSONFormatter(logging.Formatter):
    """One JSON object per line, with trace correlation when available."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "service": "edge-agent",
            "logger": record.name,
            "msg": scrub(record.getMessage()),
        }
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = scrub(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["error"] = scrub(self.formatException(record.exc_info))

        span = _current_span_ids()
        if span:
            payload["trace_id"], payload["span_id"] = span
        return json.dumps(payload, separators=(",", ":"))


def _current_span_ids() -> tuple[str, str] | None:
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except Exception:
        pass
    return None


class FieldLogger:
    """Thin adapter so call sites read ``log.info("msg", call_id=...)``."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, msg: str, **fields: Any) -> None:
        self._log.log(level, msg, extra={"fields": fields})

    def debug(self, msg: str, **f: Any) -> None: self._emit(logging.DEBUG, msg, **f)
    def info(self, msg: str, **f: Any) -> None: self._emit(logging.INFO, msg, **f)
    def warning(self, msg: str, **f: Any) -> None: self._emit(logging.WARNING, msg, **f)
    def error(self, msg: str, **f: Any) -> None: self._emit(logging.ERROR, msg, **f)

    def exception(self, msg: str, **f: Any) -> None:
        self._log.exception(msg, extra={"fields": f})


def setup_logging(level: str = "info") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # These are chatty and say nothing useful at info level.
    for noisy in (
        "websockets", "faster_whisper", "urllib3", "asyncio",
        "httpx", "httpcore", "huggingface_hub", "filelock", "kafka", "aiokafka",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> FieldLogger:
    return FieldLogger(name)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

_tracer = None


def setup_tracing(service_name: str, endpoint: str = "") -> Any:
    """Configure OTLP tracing. A missing collector must not break the edge.

    The edge agent runs on an operator's machine; if the collector is
    unreachable we degrade to a no-op tracer rather than blocking the call.
    """
    global _tracer
    from opentelemetry import trace

    if not endpoint:
        _tracer = trace.get_tracer(service_name)
        return _tracer

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({
                "service.name": service_name,
                "service.namespace": os.environ.get("OTEL_SERVICE_NAMESPACE", "edgesense"),
            })
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
    except Exception:
        logging.getLogger(__name__).warning("tracing disabled: exporter setup failed")

    _tracer = trace.get_tracer(service_name)
    return _tracer


def tracer() -> Any:
    global _tracer
    if _tracer is None:
        from opentelemetry import trace

        _tracer = trace.get_tracer("edge-agent")
    return _tracer


def current_traceparent() -> str | None:
    """W3C traceparent for the active span, to put on the wire."""
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if not ctx.is_valid:
            return None
        flags = "01" if ctx.trace_flags.sampled else "00"
        return f"00-{format(ctx.trace_id, '032x')}-{format(ctx.span_id, '016x')}-{flags}"
    except Exception:
        return None


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
