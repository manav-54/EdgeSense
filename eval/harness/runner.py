"""Run the full evaluation over the golden corpus.

Two measurement modes, because they answer different questions and mixing them
would make both meaningless:

``text``
    Corpus transcripts go straight into the real redactor and the real
    analyzers. ASR is skipped, so the labelled character offsets still line up
    and span-level precision/recall are exact. This is the mode a prompt or
    detector change is judged by, because it isolates the change from ASR
    variance.

``audio``
    The complete path, faster-whisper included. Exact spans no longer exist --
    the ASR output is not the labelled text -- so span metrics are omitted and
    the leak rate becomes the headline. This is the number that describes the
    shipped system.

Within text mode there are two sub-measurements, also deliberately separate:

* **span coverage** comes from a per-turn redaction pass that yields exact
  original-text coordinates. It is fed the same accumulated context the
  streaming path carries, so it measures the same detector behaviour rather
  than a handicapped version of it -- without that, values the shipped
  pipeline catches (bare digits typed by a "date of birth" two turns earlier)
  would be scored as misses.
* **leak rate** comes from the *streaming* output of the whole call, which is
  the production path and the only one that exercises the cross-segment hold.

A value can pass one and fail the other, and knowing which is the difference
between "the detector missed it" and "the stream released it early".
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for sub in ("services/edge-agent", "services/worker", "eval"):
    path = str(REPO / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

from edgesense_core.contracts import TranscriptSegment  # noqa: E402
from edgesense_core.timeutil import monotonic_ms  # noqa: E402

from edge_agent.redact.redactor import Redactor, RedactorConfig  # noqa: E402
from harness.classification import (  # noqa: E402
    ClassificationReport,
    evaluate_escalation_band,
    evaluate_signals,
    evaluate_summary,
)
from harness.metrics import mean, percentile  # noqa: E402
from harness.redaction import RedactionReport, TurnOutcome, evaluate_call  # noqa: E402
from worker.analysis import rules  # noqa: E402
from worker.analysis.live import LiveAnalyzer, LiveConfig  # noqa: E402
from worker.analysis.postcall import PostCallAnalyzer  # noqa: E402
from worker.llm import build_provider  # noqa: E402
from worker.policies import load_policy_store  # noqa: E402
from worker.state import CallState  # noqa: E402


@dataclass
class RunConfig:
    calls_dir: Path = REPO / "eval/golden/calls"
    audio_dir: Path = REPO / "data/audio"
    mode: str = "text"
    provider: str | None = None
    live_prompt: str = "latest"
    post_prompt: str = "latest"
    ner_backend: str = "auto"
    limit: int = 0
    categories: tuple[str, ...] = ()
    use_agent: bool = True


@dataclass
class RunResult:
    config: dict
    redaction: RedactionReport
    classification: ClassificationReport
    timings: dict = field(default_factory=dict)
    per_call: list[dict] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 2),
            "timings": self.timings,
            "redaction": self.redaction.as_dict(),
            "classification": self.classification.as_dict(),
            "per_call": self.per_call,
        }


def _segments_from_text(call: dict, redactor: Redactor) -> tuple[list[TurnOutcome], list[TranscriptSegment], list[float]]:
    """Stream a call's turns through the redactor, keeping both measurements."""
    outcomes: list[TurnOutcome] = []
    segments: list[TranscriptSegment] = []
    redact_ms: list[float] = []
    seq = 0

    # A second redactor for single-shot span measurement. It shares nothing
    # with the streaming one, so its placeholder numbering is independent --
    # only the spans are used from it.
    span_redactor = Redactor(call["call_id"], redactor.config, ner=redactor._ner)

    context_tail = ""
    for turn in call["turns"]:
        # The single-shot pass gets the same carried context the streaming
        # path uses. Without it the span measurement would report misses on
        # values the shipped pipeline catches -- "date of birth" in turn 3 is
        # what types the bare digits in turn 4.
        single = span_redactor.redact(turn["text"], extra_context=context_tail)
        context_tail = f"{context_tail} {turn['text']}"[-240:]
        t0 = monotonic_ms()
        streamed = redactor.push(turn["text"], is_final=True)
        redact_ms.append(monotonic_ms() - t0)

        outcomes.append(TurnOutcome(
            original=turn["text"],
            emitted=streamed.text,
            covered_spans=[(d.start, d.end) for d in single.detections],
            refs=list(single.redactions),
        ))

        if streamed.has_output:
            start_ms = turn["idx"] * 4200
            segments.append(TranscriptSegment(
                call_id=call["call_id"], seq=seq, speaker=turn["speaker"],
                text=streamed.text, is_final=True,
                start_ms=start_ms, end_ms=start_ms + 3800,
                emitted_at="2026-08-17T09:00:00.000000Z",
                redactions=list(streamed.redactions), asr_confidence=1.0,
                agent_id=call["agent_id"],
            ))
            seq += 1

    tail = redactor.flush()
    if tail.has_output:
        outcomes.append(TurnOutcome(tail.text, tail.text, [], list(tail.redactions)))
        segments.append(TranscriptSegment(
            call_id=call["call_id"], seq=seq, speaker="unknown", text=tail.text,
            is_final=True, start_ms=0, end_ms=0,
            emitted_at="2026-08-17T09:00:00.000000Z",
            redactions=list(tail.redactions), agent_id=call["agent_id"],
        ))
    return outcomes, segments, redact_ms


def _segments_from_audio(call: dict, redactor: Redactor, transcriber, audio_dir: Path):
    """Full path: real audio, real ASR, real redaction."""
    from edge_agent.audio import Segmenter, pcm_to_float32, stream_wav
    from edge_agent.config import EdgeConfig
    from edge_agent.pipeline import TurnMap

    wav = audio_dir / f"{call['call_id']}.wav"
    if not wav.exists():
        return [], [], []

    config = EdgeConfig()
    turn_map = TurnMap.from_manifest(audio_dir / f"{call['call_id']}.turns.json")
    segmenter = Segmenter(
        rms_threshold=config.vad_rms_threshold, silence_ms=config.silence_ms,
        partial_interval_ms=config.partial_interval_ms,
        max_utterance_ms=config.max_utterance_ms, frame_ms=config.frame_ms,
    )

    outcomes: list[TurnOutcome] = []
    segments: list[TranscriptSegment] = []
    redact_ms: list[float] = []
    seq = 0

    for frame in stream_wav(wav, config.frame_ms, realtime=False):
        if segmenter.push(frame) != Segmenter.FINAL:
            continue
        utt = segmenter.take()
        if utt is None:
            continue
        hyp = transcriber.transcribe(pcm_to_float32(utt.pcm), partial=False)
        if not hyp.text:
            continue
        t0 = monotonic_ms()
        out = redactor.push(hyp.text, is_final=True)
        redact_ms.append(monotonic_ms() - t0)
        # ASR output has no labelled offsets, so covered_spans stays empty and
        # only the leak check consumes `emitted`.
        outcomes.append(TurnOutcome(hyp.text, out.text, [], list(out.redactions)))
        if out.has_output:
            segments.append(TranscriptSegment(
                call_id=call["call_id"], seq=seq,
                speaker=turn_map.speaker_at(utt.start_ms, utt.end_ms),
                text=out.text, is_final=True,
                start_ms=utt.start_ms, end_ms=utt.end_ms,
                emitted_at="2026-08-17T09:00:00.000000Z",
                redactions=list(out.redactions), asr_confidence=hyp.confidence,
                agent_id=call["agent_id"],
            ))
            seq += 1

    tail = redactor.flush()
    if tail.has_output:
        outcomes.append(TurnOutcome(tail.text, tail.text, [], list(tail.redactions)))
    return outcomes, segments, redact_ms


def run(config: RunConfig) -> RunResult:
    started = time.time()
    provider = build_provider(config.provider)
    policies = load_policy_store()
    live = LiveAnalyzer(
        provider, policies,
        LiveConfig(prompt_version=config.live_prompt, use_agent=config.use_agent),
    )
    post = PostCallAnalyzer(provider, policies, prompt_version=config.post_prompt)

    transcriber = None
    if config.mode == "audio":
        from edge_agent.asr import Transcriber
        from edge_agent.config import EdgeConfig

        edge_config = EdgeConfig()
        transcriber = Transcriber(edge_config.whisper_model,
                                  edge_config.whisper_compute_type,
                                  edge_config.model_dir)

    redaction = RedactionReport()
    classification = ClassificationReport()
    analysis_ms: list[float] = []
    post_ms: list[float] = []
    redact_ms_all: list[float] = []
    per_call: list[dict] = []

    paths = sorted(config.calls_dir.glob("*.json"))
    calls = [json.loads(p.read_text()) for p in paths]
    if config.categories:
        calls = [c for c in calls if c["category"] in config.categories]
    if config.limit:
        calls = calls[: config.limit]

    for i, call in enumerate(calls, 1):
        redactor = Redactor(
            call["call_id"],
            RedactorConfig(allowlist=(call.get("agent_name", ""),),
                           ner_backend=config.ner_backend),
        )

        if config.mode == "audio":
            outcomes, segments, redact_ms = _segments_from_audio(
                call, redactor, transcriber, config.audio_dir
            )
        else:
            outcomes, segments, redact_ms = _segments_from_text(call, redactor)
        redact_ms_all.extend(redact_ms)

        if config.mode == "text":
            evaluate_call(call, outcomes, redaction)
        else:
            # Audio mode: leak-only scoring, since spans do not survive ASR.
            _leak_only(call, outcomes, redaction)

        state = CallState(call_id=call["call_id"], agent_id=call["agent_id"])
        signals = []
        for segment in segments:
            state.add(segment)
            if live.should_analyse(state):
                t0 = monotonic_ms()
                signals.extend(live.analyse(state))
                analysis_ms.append(monotonic_ms() - t0)

        transcript_turns = [turn.text for turn in state.turns]
        evaluate_signals(signals, transcript_turns, classification)

        band, _, _ = rules.escalation_risk(state.all_turns())
        evaluate_escalation_band(call, band, classification)

        t0 = monotonic_ms()
        result = post.summarise(state)
        post_ms.append(monotonic_ms() - t0)
        evaluate_summary(call, result.summary, result.repairs > 0,
                         transcript_turns, classification)

        per_call.append({
            "call_id": call["call_id"],
            "category": call["category"],
            "turns": len(state.turns),
            "signals": len(signals),
            "summary_ok": result.ok,
            "expected_intent": call["labels"]["primary_intent"],
            "actual_intent": result.summary.primary_intent if result.summary else None,
            "expected_violations": call["labels"]["compliance_violations"],
            "actual_violations": (
                result.summary.compliance_violations if result.summary else []
            ),
        })

        if i % 12 == 0:
            print(f"  ... {i}/{len(calls)} calls", flush=True)

    return RunResult(
        config={
            "mode": config.mode,
            "provider": provider.name,
            "model": provider.model,
            "live_prompt": config.live_prompt,
            "post_prompt": config.post_prompt,
            "ner_backend": config.ner_backend,
            "use_agent": config.use_agent,
            "calls": len(calls),
            "categories": list(config.categories) or "all",
        },
        redaction=redaction,
        classification=classification,
        timings={
            "redact_ms_mean": round(mean(redact_ms_all), 3),
            "redact_ms_p95": round(percentile(redact_ms_all, 0.95), 3),
            "analysis_ms_mean": round(mean(analysis_ms), 3),
            "analysis_ms_p95": round(percentile(analysis_ms, 0.95), 3),
            "post_call_ms_mean": round(mean(post_ms), 3),
            "post_call_ms_p95": round(percentile(post_ms, 0.95), 3),
        },
        per_call=per_call,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        duration_s=time.time() - started,
    )


def _leak_only(call: dict, outcomes: list[TurnOutcome], report: RedactionReport) -> None:
    """Audio-mode scoring: did any labelled value survive anywhere?"""
    from harness.metrics import PRCounts
    from harness.redaction import SpanResult, find_leak

    emitted = "\n".join(o.emitted for o in outcomes)
    for turn in call["turns"]:
        for span in turn["pii"]:
            if span["type"] == "NON_CARD":
                continue
            report.spans_total += 1
            space = find_leak(span["value"], emitted)
            counts = PRCounts(fn=1) if space else PRCounts(tp=1)
            report.by_type[span["type"]] = (
                report.by_type.get(span["type"], PRCounts()) + counts
            )
            report.by_surface[span["surface_form"]] = (
                report.by_surface.get(span["surface_form"], PRCounts()) + counts
            )
            report.by_category[call["category"]] = (
                report.by_category.get(call["category"], PRCounts()) + counts
            )
            if space:
                report.leaks.append(SpanResult(
                    call_id=call["call_id"], turn_idx=turn["idx"],
                    pii_type=span["type"], surface_form=span["surface_form"],
                    is_partial=span["is_partial"], value=span["value"],
                    covered=False, type_correct=False, leaked=True, leak_space=space,
                ))
