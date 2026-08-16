"""Local streaming transcription with faster-whisper.

Everything here runs on the operator's CPU. No audio is uploaded, which is the
entire point: the network boundary sits *after* this module and after
redaction, so the cloud never receives a sample of the customer's voice.

Whisper is not a streaming model -- it decodes a window, not a stream. The
approximation used here is the standard one: re-decode the growing utterance
buffer at intervals for partials, then decode once more at the boundary for
the final. It costs redundant compute on partials, which is why the partial
interval is configurable and why partials use a cheaper decode path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgesense_core.timeutil import monotonic_ms

from edge_agent.obs import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float
    latency_ms: float
    language: str = "en"


class Transcriber:
    """faster-whisper wrapper. Thread-safe: one model, serialised decodes."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        compute_type: str = "int8",
        download_root: Path | None = None,
        device: str = "cpu",
        cpu_threads: int = 0,
    ) -> None:
        from faster_whisper import WhisperModel

        t0 = monotonic_ms()
        self.model_size = model_size
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root else None,
            cpu_threads=cpu_threads,
        )
        # ctranslate2 is not documented as re-entrant for a single model
        # instance; serialise rather than discover the hard way under load.
        self._lock = threading.Lock()
        log.info(
            "asr model loaded",
            model=model_size,
            compute_type=compute_type,
            load_ms=round(monotonic_ms() - t0, 1),
        )

    def transcribe(self, audio: Any, *, partial: bool = False) -> Transcript:
        """Decode a float32 mono 16 kHz array.

        Partials use greedy decoding with no temperature fallback: a partial is
        thrown away in under a second, so spending extra beams on it buys
        nothing but latency.
        """
        t0 = monotonic_ms()
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language="en",
                beam_size=1 if partial else 5,
                temperature=0.0 if partial else (0.0, 0.2, 0.4),
                condition_on_previous_text=False,
                vad_filter=False,  # segmentation already happened upstream
                without_timestamps=True,
            )
            collected = list(segments)

        text = " ".join(s.text.strip() for s in collected).strip()
        if collected:
            # avg_logprob is a log-probability; exp maps it to a rough 0-1
            # confidence. It is a comparable signal, not a calibrated one.
            import math

            avg = sum(s.avg_logprob for s in collected) / len(collected)
            confidence = max(0.0, min(1.0, math.exp(avg)))
        else:
            confidence = 0.0

        return Transcript(
            text=text,
            confidence=confidence,
            latency_ms=monotonic_ms() - t0,
            language=getattr(info, "language", "en"),
        )


class ScriptedTranscriber:
    """Replays known text instead of decoding audio.

    Used *only* by the text-mode eval harness, where the point is to measure
    the redactor against exact ground-truth spans without ASR error in the
    way. It is never wired into the demo or the docker-compose path -- that
    runs real audio through the real model.
    """

    model_size = "scripted"

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._idx = 0

    def transcribe(self, audio: Any, *, partial: bool = False) -> Transcript:
        if self._idx >= len(self._lines):
            return Transcript("", 0.0, 0.0)
        text = self._lines[self._idx]
        if not partial:
            self._idx += 1
        return Transcript(text, 1.0, 0.0)
