"""Structured logging and tracing shared by the Python services.

Metrics stay per-service (the names and labels differ), but logging and trace
setup are identical everywhere and were duplicated once already. The scrubbing
formatter in particular must not drift: it is a privacy control, and a service
that quietly loses it becomes the leak path.

The formatter refuses to emit anything digit- or email-shaped. Code should not
log raw transcripts, but backstops are what keep a guarantee alive through
maintenance by someone who has not read DESIGN.md.
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

NOISY_LOGGERS = (
    "websockets", "faster_whisper", "urllib3", "asyncio", "httpx", "httpcore",
    "huggingface_hub", "filelock", "kafka", "aiokafka", "clickhouse_connect",
)


def scrub(text: str) -> str:
    """Remove anything digit- or email-shaped from a log line."""
    out = _LONG_DIGITS.sub("[redacted:digits]", text)
    return _EMAILISH.sub("[redacted:email]", out)


class JSONFormatter(logging.Formatter):
    """One JSON object per line, with trace correlation when available."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "service": self.service,
            "logger": record.name,
            "msg": scrub(record.getMessage()),
        }
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = scrub(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["error"] = scrub(self.formatException(record.exc_info))

        ids = _current_span_ids()
        if ids:
            payload["trace_id"], payload["span_id"] = ids
        return json.dumps(payload, separators=(",", ":"), default=str)


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
    """Adapter so call sites read ``log.info("msg", call_id=...)``."""

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


def setup_logging(service: str, level: str = "info") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> FieldLogger:
    return FieldLogger(name)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

_tracer: Any = None


def setup_tracing(service_name: str, endpoint: str = "") -> Any:
    """Configure OTLP tracing. A missing collector degrades to a no-op tracer.

    Observability must not be able to take the pipeline down; a worker that
    refuses to start because a collector is unreachable converts a monitoring
    outage into a processing outage.
    """
    global _tracer
    from opentelemetry import trace
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.textmap import default_getter  # noqa: F401

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


def tracer(default_name: str = "edgesense") -> Any:
    global _tracer
    if _tracer is None:
        from opentelemetry import trace

        _tracer = trace.get_tracer(default_name)
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


def context_from_traceparent(traceparent: str | None):
    """Rebuild an OTel context from a wire traceparent, for span linking."""
    if not traceparent:
        return None
    try:
        from opentelemetry.propagate import extract

        return extract({"traceparent": traceparent})
    except Exception:
        return None
