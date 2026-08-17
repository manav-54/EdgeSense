"""Concurrency load test: how many simultaneous calls before p95 breaches budget.

Two modes, and the difference matters for reading the result:

``local`` (default, runs anywhere)
    Every simultaneous call runs the real CPU-bound path in its own thread:
    frame streaming, faster-whisper decode, PII redaction, sliding-window
    analysis, post-call summary. The broker, the Go ingest service and
    ClickHouse are not in the loop.

    This is the honest way to find the *compute* ceiling, which is where the
    2 s budget is actually spent -- ASR dominates by two orders of magnitude
    over a Kafka publish. What it does not measure is queueing delay under
    broker backpressure, so the number it produces is an upper bound on
    per-host capacity, not an end-to-end SLO.

``wire``
    Points real edge agents at a running ingest endpoint. Includes every hop.
    Requires the docker-compose stack.

The ramp stops at the first concurrency level where p95 breaches the budget,
and reports the last level that held. Each level runs from a cold segmenter
but a warm model, because a production host does not reload whisper per call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("services/edge-agent", "services/worker"):
    sys.path.insert(0, str(REPO / sub))

from edgesense_core.contracts import TranscriptSegment  # noqa: E402
from edgesense_core.obs import setup_logging  # noqa: E402
from edgesense_core.timeutil import monotonic_ms, utc_now_iso  # noqa: E402

from edge_agent.audio import Segmenter, pcm_to_float32, stream_wav  # noqa: E402
from edge_agent.config import EdgeConfig  # noqa: E402
from edge_agent.pipeline import TurnMap  # noqa: E402
from edge_agent.redact.redactor import Redactor, RedactorConfig  # noqa: E402
from worker.analysis.live import LiveAnalyzer, LiveConfig  # noqa: E402
from worker.analysis.postcall import PostCallAnalyzer  # noqa: E402
from worker.llm import build_provider  # noqa: E402
from worker.policies import load_policy_store  # noqa: E402
from worker.state import CallState  # noqa: E402

BUDGET_MS = 2000.0


@dataclass
class CallStats:
    call_id: str
    audio_ms: int = 0
    wall_ms: float = 0.0
    segments: int = 0
    signals: int = 0
    # Per-segment: emit -> signal published for the window that segment closed.
    e2e_ms: list[float] = field(default_factory=list)
    asr_ms: list[float] = field(default_factory=list)
    redact_ms: list[float] = field(default_factory=list)
    analyze_ms: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class LevelResult:
    concurrency: int
    calls: int
    duration_s: float
    audio_s: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    asr_p95: float
    redact_p95: float
    analyze_p95: float
    realtime_factor: float
    errors: int

    @property
    def within_budget(self) -> bool:
        return self.e2e_p95 <= BUDGET_MS

    def as_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "calls": self.calls,
            "duration_s": round(self.duration_s, 2),
            "audio_s": round(self.audio_s, 1),
            "e2e_p50_ms": round(self.e2e_p50, 1),
            "e2e_p95_ms": round(self.e2e_p95, 1),
            "e2e_p99_ms": round(self.e2e_p99, 1),
            "asr_p95_ms": round(self.asr_p95, 1),
            "redact_p95_ms": round(self.redact_p95, 1),
            "analyze_p95_ms": round(self.analyze_p95, 1),
            "realtime_factor": round(self.realtime_factor, 4),
            "errors": self.errors,
            "within_budget": self.within_budget,
        }


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[index]


def run_one_call(
    wav: Path,
    manifest: Path,
    call_id: str,
    transcriber,
    live: LiveAnalyzer,
    post: PostCallAnalyzer,
    realtime: bool,
) -> CallStats:
    """One simulated call through the real CPU path."""
    stats = CallStats(call_id=call_id)
    config = EdgeConfig()
    config.realtime = realtime

    try:
        redactor = Redactor(call_id, RedactorConfig())
        turn_map = TurnMap.from_manifest(manifest)
        segmenter = Segmenter(
            rms_threshold=config.vad_rms_threshold, silence_ms=config.silence_ms,
            partial_interval_ms=config.partial_interval_ms,
            max_utterance_ms=config.max_utterance_ms, frame_ms=config.frame_ms,
        )
        state = CallState(call_id=call_id)
        seq = 0
        wall0 = monotonic_ms()

        for frame in stream_wav(wav, config.frame_ms, realtime=realtime):
            stats.audio_ms = frame.end_ms
            if segmenter.push(frame) != Segmenter.FINAL:
                continue
            utt = segmenter.take()
            if utt is None:
                continue

            # The clock for the SLO starts when the utterance closes: that is
            # the moment the words exist and the pipeline owes an answer.
            segment_t0 = monotonic_ms()

            t = monotonic_ms()
            hyp = transcriber.transcribe(pcm_to_float32(utt.pcm), partial=False)
            stats.asr_ms.append(monotonic_ms() - t)
            if not hyp.text:
                continue

            t = monotonic_ms()
            out = redactor.push(hyp.text, is_final=True)
            stats.redact_ms.append(monotonic_ms() - t)
            if not out.has_output:
                continue

            state.add(TranscriptSegment(
                call_id=call_id, seq=seq,
                speaker=turn_map.speaker_at(utt.start_ms, utt.end_ms),
                text=out.text, is_final=True,
                start_ms=utt.start_ms, end_ms=utt.end_ms,
                emitted_at=utc_now_iso(), redactions=list(out.redactions),
                asr_confidence=hyp.confidence,
            ))
            seq += 1
            stats.segments += 1

            if live.should_analyse(state):
                t = monotonic_ms()
                signals = live.analyse(state)
                stats.analyze_ms.append(monotonic_ms() - t)
                stats.signals += len(signals)
                stats.e2e_ms.append(monotonic_ms() - segment_t0)

        tail = redactor.flush()
        if tail.has_output:
            stats.segments += 1
        post.summarise(state)
        stats.wall_ms = monotonic_ms() - wall0
    except Exception as exc:  # pragma: no cover - load harness robustness
        stats.error = f"{type(exc).__name__}: {exc}"
    return stats


def run_level(
    concurrency: int,
    wavs: list[tuple[Path, Path]],
    transcriber,
    live: LiveAnalyzer,
    post: PostCallAnalyzer,
    realtime: bool,
) -> LevelResult:
    results: list[CallStats] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        wav, manifest = wavs[index % len(wavs)]
        stats = run_one_call(
            wav, manifest, f"load-{concurrency}-{index}",
            transcriber, live, post, realtime,
        )
        with lock:
            results.append(stats)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    duration = time.perf_counter() - start

    e2e = [v for r in results for v in r.e2e_ms]
    audio_s = sum(r.audio_ms for r in results) / 1000.0

    return LevelResult(
        concurrency=concurrency,
        calls=len(results),
        duration_s=duration,
        audio_s=audio_s,
        e2e_p50=pct(e2e, 0.50),
        e2e_p95=pct(e2e, 0.95),
        e2e_p99=pct(e2e, 0.99),
        asr_p95=pct([v for r in results for v in r.asr_ms], 0.95),
        redact_p95=pct([v for r in results for v in r.redact_ms], 0.95),
        analyze_p95=pct([v for r in results for v in r.analyze_ms], 0.95),
        realtime_factor=(duration * 1000.0) / max(
            statistics.mean([r.audio_ms for r in results]) if results else 1, 1
        ),
        errors=sum(1 for r in results if r.error),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="EdgeSense concurrency load test")
    ap.add_argument("--audio-dir", type=Path, default=REPO / "data/audio")
    ap.add_argument("--levels", type=int, nargs="*",
                    default=[1, 2, 4, 8, 12, 16, 24, 32],
                    help="Concurrency levels to ramp through.")
    ap.add_argument("--distinct-calls", type=int, default=8,
                    help="How many different WAVs to cycle through.")
    ap.add_argument("--realtime", action="store_true",
                    help="Pace frames to wall clock, as a live call would.")
    ap.add_argument("--budget-ms", type=float, default=BUDGET_MS)
    ap.add_argument("--out", type=Path, default=REPO / "eval/reports/loadtest.json")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    setup_logging("loadtest", "warning")

    wavs = []
    for wav in sorted(args.audio_dir.glob("*.wav"))[: args.distinct_calls]:
        manifest = wav.parent / f"{wav.stem}.turns.json"
        if manifest.exists():
            wavs.append((wav, manifest))
    if not wavs:
        print(f"no audio in {args.audio_dir}; run `make audio` first", file=sys.stderr)
        return 2

    from edge_agent.asr import Transcriber

    config = EdgeConfig()
    print(f"loading whisper {args.model or config.whisper_model} ...")
    transcriber = Transcriber(
        args.model or config.whisper_model, config.whisper_compute_type, config.model_dir
    )
    provider = build_provider()
    policies = load_policy_store()
    live = LiveAnalyzer(provider, policies, LiveConfig())
    post = PostCallAnalyzer(provider, policies)

    print(f"\nbudget: p95 segment->signal <= {args.budget_ms:.0f} ms")
    print(f"corpus: {len(wavs)} distinct calls, "
          f"{sum(1 for _ in wavs)} manifests, realtime={args.realtime}\n")
    header = (f"{'conc':>5} {'calls':>6} {'audio_s':>8} {'wall_s':>7} "
              f"{'p50':>8} {'p95':>8} {'p99':>8} {'asr_p95':>8} {'errs':>5}  verdict")
    print(header)
    print("-" * len(header))

    levels: list[LevelResult] = []
    last_ok = 0
    for concurrency in args.levels:
        result = run_level(concurrency, wavs, transcriber, live, post, args.realtime)
        levels.append(result)
        verdict = "OK" if result.e2e_p95 <= args.budget_ms else "BREACH"
        print(f"{result.concurrency:>5} {result.calls:>6} {result.audio_s:>8.1f} "
              f"{result.duration_s:>7.1f} {result.e2e_p50:>8.1f} {result.e2e_p95:>8.1f} "
              f"{result.e2e_p99:>8.1f} {result.asr_p95:>8.1f} {result.errors:>5}  {verdict}",
              flush=True)
        if result.e2e_p95 > args.budget_ms:
            break
        last_ok = concurrency

    payload = {
        "budget_ms": args.budget_ms,
        "mode": "local",
        "realtime": args.realtime,
        "distinct_calls": len(wavs),
        "model": args.model or config.whisper_model,
        "max_sustained_concurrent_calls": last_ok,
        "levels": [level.as_dict() for level in levels],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print()
    print(f"max sustained concurrent calls within budget: {last_ok}")
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
