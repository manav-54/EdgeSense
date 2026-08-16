"""Edge pipeline: audio frames in, redacted segments out.

Ordering here is the privacy guarantee, so it is worth stating plainly:

    frames -> segmenter -> ASR -> redactor -> transport

Redaction sits between ASR and transport with nothing in between. There is no
path from ``Transcript.text`` to ``Transport.send`` that does not pass through
``Redactor.push``, and the egress test exists to keep it that way as the code
changes.

Speaker labels come from the audio manifest rather than a diariser. Speaker
diarisation is an explicit non-goal (DESIGN.md); pretending to do it badly
would corrupt every per-speaker metric downstream, so the simulation uses the
known turn boundaries and the field is honestly sourced.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from edgesense_core.contracts import Speaker, TranscriptSegment
from edgesense_core.timeutil import monotonic_ms, utc_now_iso

from edge_agent.asr import Transcriber
from edge_agent.audio import Segmenter, Utterance, pcm_to_float32, stream_wav
from edge_agent.config import EdgeConfig
from edge_agent.obs import (
    ASR_LATENCY,
    HELD_FRAGMENTS,
    METRICS_AVAILABLE,
    REDACTIONS,
    REDACT_LATENCY,
    SEGMENTS_EMITTED,
    SEND_LATENCY,
    current_traceparent,
    get_logger,
    tracer,
)
from edge_agent.redact.redactor import Redactor, RedactorConfig
from edge_agent.transport import Transport

log = get_logger(__name__)


@dataclass
class RunStats:
    call_id: str
    segments_final: int = 0
    segments_partial: int = 0
    redactions: int = 0
    held_events: int = 0
    audio_ms: int = 0
    wall_ms: float = 0.0
    asr_ms_total: float = 0.0
    redact_ms_total: float = 0.0
    per_type: dict[str, int] = field(default_factory=dict)

    @property
    def realtime_factor(self) -> float:
        """Wall time per second of audio. Below 1.0 means it keeps up live."""
        return (self.wall_ms / self.audio_ms) if self.audio_ms else 0.0


class TurnMap:
    """Maps an audio timestamp to the speaker who was talking.

    Ground truth from synthesis, not inference. ``UNKNOWN`` is returned rather
    than guessed when a timestamp falls in a gap, so a downstream per-speaker
    aggregate can exclude it instead of silently attributing it to whoever
    spoke last.
    """

    def __init__(self, turns: list[dict] | None) -> None:
        self._turns = turns or []

    @classmethod
    def from_manifest(cls, path: Path) -> TurnMap:
        if not path.exists():
            return cls(None)
        return cls(json.loads(path.read_text()).get("turns", []))

    def speaker_at(self, start_ms: int, end_ms: int) -> Speaker:
        mid = (start_ms + end_ms) // 2
        best: tuple[int, Speaker] | None = None
        for t in self._turns:
            if t["start_ms"] <= mid <= t["end_ms"]:
                return Speaker(t["speaker"])
            overlap = min(end_ms, t["end_ms"]) - max(start_ms, t["start_ms"])
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, Speaker(t["speaker"]))
        return best[1] if best else Speaker.UNKNOWN


class EdgePipeline:
    def __init__(
        self,
        call_id: str,
        config: EdgeConfig,
        transcriber: Transcriber,
        transport: Transport,
        *,
        turn_map: TurnMap | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.call_id = call_id
        self.config = config
        self.transcriber = transcriber
        self.transport = transport
        self.turn_map = turn_map or TurnMap(None)
        self.agent_id = agent_id
        self.redactor = Redactor(
            call_id,
            RedactorConfig(
                allowlist=(agent_name,) if agent_name else (),
                ner_backend=config.ner_backend,
            ),
        )
        self.stats = RunStats(call_id=call_id)
        self._seq = 0

    # -- emission ----------------------------------------------------------

    def _emit(
        self,
        text: str,
        redactions,
        *,
        is_final: bool,
        start_ms: int,
        end_ms: int,
        confidence: float,
    ) -> None:
        if not text.strip():
            return

        segment = TranscriptSegment(
            call_id=self.call_id,
            seq=self._seq,
            speaker=self.turn_map.speaker_at(start_ms, end_ms),
            text=text,
            is_final=is_final,
            start_ms=start_ms,
            end_ms=end_ms,
            emitted_at=utc_now_iso(),
            redactions=list(redactions),
            asr_confidence=confidence,
            agent_id=self.agent_id,
            traceparent=current_traceparent(),
        )

        t0 = monotonic_ms()
        self.transport.send(segment)
        send_ms = monotonic_ms() - t0

        if METRICS_AVAILABLE:
            SEND_LATENCY.observe(send_ms / 1000.0)
            SEGMENTS_EMITTED.labels(
                call_id_present="1", is_final=str(is_final).lower()
            ).inc()
            for ref in redactions:
                REDACTIONS.labels(pii_type=ref.type.value, detector=ref.detector).inc()

        if is_final:
            self.stats.segments_final += 1
        else:
            self.stats.segments_partial += 1
        self.stats.redactions += len(redactions)
        for ref in redactions:
            key = ref.type.value
            self.stats.per_type[key] = self.stats.per_type.get(key, 0) + 1

    def _transcribe(self, utt: Utterance, *, partial: bool):
        t0 = monotonic_ms()
        result = self.transcriber.transcribe(pcm_to_float32(utt.pcm), partial=partial)
        elapsed = monotonic_ms() - t0
        self.stats.asr_ms_total += elapsed
        if METRICS_AVAILABLE:
            ASR_LATENCY.observe(elapsed / 1000.0)
        return result

    # -- main loop ---------------------------------------------------------

    def run(self, wav_path: Path) -> RunStats:
        tr = tracer()
        wall0 = monotonic_ms()
        segmenter = Segmenter(
            rms_threshold=self.config.vad_rms_threshold,
            silence_ms=self.config.silence_ms,
            partial_interval_ms=self.config.partial_interval_ms,
            max_utterance_ms=self.config.max_utterance_ms,
            frame_ms=self.config.frame_ms,
        )

        with tr.start_as_current_span("edge.call") as call_span:
            call_span.set_attribute("edgesense.call_id", self.call_id)
            call_span.set_attribute("edgesense.audio_path", str(wav_path))

            for frame in stream_wav(wav_path, self.config.frame_ms, self.config.realtime):
                self.stats.audio_ms = frame.end_ms
                decision = segmenter.push(frame)

                if decision == Segmenter.KEEP:
                    continue

                utt = segmenter.current
                if utt is None or not utt.frames:
                    continue

                if decision == Segmenter.PARTIAL:
                    if not self.config.emit_partials:
                        continue
                    with tr.start_as_current_span("edge.segment.partial") as span:
                        span.set_attribute("edgesense.call_id", self.call_id)
                        span.set_attribute("edgesense.seq", self._seq)
                        hyp = self._transcribe(utt, partial=True)
                        if not hyp.text:
                            continue
                        out = self._preview(hyp.text)
                        self._emit(
                            out.text, out.redactions, is_final=False,
                            start_ms=utt.start_ms, end_ms=utt.end_ms,
                            confidence=hyp.confidence,
                        )
                    continue

                # FINAL
                closed = segmenter.take()
                if closed is None:
                    continue
                with tr.start_as_current_span("edge.segment.final") as span:
                    span.set_attribute("edgesense.call_id", self.call_id)
                    span.set_attribute("edgesense.seq", self._seq)
                    hyp = self._transcribe(closed, partial=False)
                    if not hyp.text:
                        continue
                    t0 = monotonic_ms()
                    out = self.redactor.push(hyp.text, is_final=True)
                    redact_ms = monotonic_ms() - t0
                    self.stats.redact_ms_total += redact_ms
                    if METRICS_AVAILABLE:
                        REDACT_LATENCY.observe(redact_ms / 1000.0)
                        if out.held_digits:
                            HELD_FRAGMENTS.inc()
                    if out.held_digits:
                        self.stats.held_events += 1
                        span.set_attribute("edgesense.held_digits", out.held_digits)

                    if out.has_output:
                        self._emit(
                            out.text, out.redactions, is_final=True,
                            start_ms=closed.start_ms, end_ms=closed.end_ms,
                            confidence=hyp.confidence,
                        )
                        self._seq += 1

            # Anything still held must be emitted redacted, never dropped and
            # never released raw.
            tail = self.redactor.flush()
            if tail.has_output:
                self._emit(
                    tail.text, tail.redactions, is_final=True,
                    start_ms=self.stats.audio_ms, end_ms=self.stats.audio_ms,
                    confidence=0.0,
                )
                self._seq += 1

        self.stats.wall_ms = monotonic_ms() - wall0
        log.info(
            "call complete",
            call_id=self.call_id,
            finals=self.stats.segments_final,
            partials=self.stats.segments_partial,
            redactions=self.stats.redactions,
            held_events=self.stats.held_events,
            audio_s=round(self.stats.audio_ms / 1000, 1),
            wall_s=round(self.stats.wall_ms / 1000, 1),
            rtf=round(self.stats.realtime_factor, 3),
            by_type=self.stats.per_type,
        )
        return self.stats

    def _preview(self, text: str):
        """Redact a partial without mutating carry state.

        A partial is a guess the ASR may revise. Committing it to the hold
        buffer would let a revised value slip past, so partials get a
        throwaway redaction pass against a scratch redactor that shares the
        vault -- so placeholder numbering stays stable across the revision.
        """
        scratch = Redactor(
            self.call_id,
            self.redactor.config,
            ner=self.redactor._ner,
            vault=self.redactor.vault,
        )
        return scratch.push(text, is_final=False)


def new_call_id() -> str:
    return f"call-{uuid.uuid4().hex[:16]}"
