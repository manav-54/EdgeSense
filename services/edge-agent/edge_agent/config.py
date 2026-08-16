"""Edge agent configuration, resolved from environment with working defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EdgeConfig:
    ingest_url: str = field(default_factory=lambda: _env("INGEST_WS_URL", "ws://localhost:8080/v1/stream"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "tiny.en"))
    whisper_compute_type: str = field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "int8"))
    model_dir: Path = field(default_factory=lambda: Path(_env("WHISPER_MODEL_DIR", "data/models")))

    frame_ms: int = field(default_factory=lambda: _env_int("FRAME_MS", 20))
    #: How often a partial hypothesis is produced while the caller is speaking.
    partial_interval_ms: int = field(default_factory=lambda: _env_int("PARTIAL_INTERVAL_MS", 700))
    #: Silence this long closes an utterance and forces a final segment.
    silence_ms: int = field(default_factory=lambda: _env_int("SILENCE_MS", 240))
    #: Frames quieter than this count as silence. Tuned for 16-bit PCM speech.
    vad_rms_threshold: int = field(default_factory=lambda: _env_int("VAD_RMS_THRESHOLD", 220))
    #: Hard cap on utterance length, so a caller who never pauses still gets
    #: segmented rather than growing the decode window without bound.
    max_utterance_ms: int = field(default_factory=lambda: _env_int("MAX_UTTERANCE_MS", 12_000))

    realtime: bool = field(default_factory=lambda: _env_bool("EDGE_REALTIME", True))
    emit_partials: bool = field(default_factory=lambda: _env_bool("EDGE_EMIT_PARTIALS", True))
    ner_backend: str = field(default_factory=lambda: _env("EDGE_NER_BACKEND", "auto"))

    otlp_endpoint: str = field(default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT", ""))
    prometheus_port: int = field(default_factory=lambda: _env_int("EDGE_PROMETHEUS_PORT", 9101))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "info"))

    @property
    def frame_bytes(self) -> int:
        # 16 kHz, mono, 16-bit
        return int(16_000 * 2 * self.frame_ms / 1000)
