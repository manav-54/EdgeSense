"""20 ms frame streaming and utterance segmentation.

The frame loop exists to make the simulation honest: a real agent receives
audio in small chunks at wall-clock pace and must decide what to do with each
one, so the pipeline is exercised under the same arrival pattern a live call
produces. Reading the whole WAV and transcribing it once would produce nicer
latency numbers that mean nothing.

Segmentation is energy-based VAD rather than a neural one. That is a real
limitation on noisy audio and is listed as such in DESIGN.md; on the clean
synthesised corpus it finds turn boundaries reliably, and it costs
microseconds per frame instead of milliseconds.
"""

from __future__ import annotations

import math
import struct
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2


@dataclass(frozen=True)
class Frame:
    """One fixed-size slice of PCM audio."""

    pcm: bytes
    seq: int
    start_ms: int
    end_ms: int

    @property
    def rms(self) -> float:
        n = len(self.pcm) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"<{n}h", self.pcm[: n * 2])
        return math.sqrt(sum(s * s for s in samples) / n)


def stream_wav(path: Path, frame_ms: int = 20, realtime: bool = True) -> Iterator[Frame]:
    """Yield fixed-size frames from a 16 kHz mono WAV, optionally paced to wall clock.

    Pacing uses an absolute schedule rather than ``sleep(frame_ms)`` per frame,
    so processing time inside the loop does not accumulate into drift -- after
    a thousand frames a naive sleep would be seconds behind real time.
    """
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError(
                f"{path}: expected 16 kHz mono 16-bit, got {w.getframerate()} Hz "
                f"{w.getnchannels()}ch {w.getsampwidth() * 8}-bit"
            )
        samples_per_frame = int(SAMPLE_RATE * frame_ms / 1000)
        started = time.perf_counter()
        seq = 0
        while True:
            pcm = w.readframes(samples_per_frame)
            if not pcm:
                return
            start_ms = int(seq * frame_ms)
            end_ms = start_ms + int(len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE * 1000)
            if realtime:
                target = started + (start_ms / 1000.0)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield Frame(pcm=pcm, seq=seq, start_ms=start_ms, end_ms=end_ms)
            seq += 1


@dataclass
class Utterance:
    """Audio accumulated since the last boundary."""

    start_ms: int
    frames: list[Frame] = field(default_factory=list)

    @property
    def pcm(self) -> bytes:
        return b"".join(f.pcm for f in self.frames)

    @property
    def end_ms(self) -> int:
        return self.frames[-1].end_ms if self.frames else self.start_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def __len__(self) -> int:
        return len(self.frames)


class Segmenter:
    """Turn a frame stream into utterance boundaries.

    Emits three kinds of decision: keep buffering, produce a partial, or close
    the utterance and produce a final.
    """

    KEEP = "keep"
    PARTIAL = "partial"
    FINAL = "final"

    def __init__(
        self,
        *,
        rms_threshold: float = 220.0,
        silence_ms: int = 240,
        partial_interval_ms: int = 700,
        max_utterance_ms: int = 12_000,
        frame_ms: int = 20,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.silence_frames_needed = max(1, silence_ms // frame_ms)
        self.partial_interval_ms = partial_interval_ms
        self.max_utterance_ms = max_utterance_ms
        self._silence_run = 0
        self._speech_frames = 0
        self._last_partial_ms = 0
        self.current: Utterance | None = None

    def push(self, frame: Frame) -> str:
        voiced = frame.rms >= self.rms_threshold

        if self.current is None:
            if not voiced:
                return self.KEEP  # leading silence, nothing to open yet
            self.current = Utterance(start_ms=frame.start_ms)
            self._silence_run = 0
            self._speech_frames = 0
            self._last_partial_ms = frame.start_ms

        self.current.frames.append(frame)
        if voiced:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        # Enough trailing silence, and we actually heard something: close it.
        if self._silence_run >= self.silence_frames_needed and self._speech_frames >= 3:
            return self.FINAL

        if self.current.duration_ms >= self.max_utterance_ms:
            return self.FINAL

        if frame.end_ms - self._last_partial_ms >= self.partial_interval_ms and self._speech_frames >= 3:
            self._last_partial_ms = frame.end_ms
            return self.PARTIAL

        return self.KEEP

    def take(self) -> Utterance | None:
        """Close the current utterance and return it."""
        utt = self.current
        self.current = None
        self._silence_run = 0
        self._speech_frames = 0
        return utt


def pcm_to_float32(pcm: bytes):
    """Convert 16-bit PCM to the float32 array faster-whisper expects."""
    import numpy as np

    return np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
