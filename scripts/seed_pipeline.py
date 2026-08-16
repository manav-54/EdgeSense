"""Run the real pipeline over the corpus and write the results to ClickHouse.

Two modes:

``--mode text`` (default)
    Replays the corpus transcripts through the real redactor, the real
    analyzers, and the real row mapping, skipping only ASR. Used to build
    analytics *history* -- a dashboard with one call on it demonstrates
    nothing, and synthesising 33 minutes of audio per historical call to fill
    a fortnight of charts would take hours.

``--mode audio``
    Runs the complete path including faster-whisper over the generated WAVs.
    This is what ``make demo`` uses, and what the "no mock data in the happy
    path" claim rests on.

Timestamps are spread backwards over ``--days`` so the time-series queries and
the hourly rollups have real shape rather than a single spike. Every other
value -- signals, evidence spans, summaries, latencies -- is genuinely
computed, not fabricated.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("services/edge-agent", "services/worker", "services/sink"):
    sys.path.insert(0, str(REPO / sub))

from edgesense_core.contracts import CallInsights, TranscriptSegment  # noqa: E402
from edgesense_core.obs import setup_logging  # noqa: E402
from edgesense_core.timeutil import utc_now  # noqa: E402

from edge_agent.redact.redactor import Redactor, RedactorConfig  # noqa: E402
from sink.rows import (  # noqa: E402
    LATENCY_COLUMNS,
    LATENCY_TABLE,
    SIGNAL_COLUMNS,
    SIGNAL_TABLE,
    SUMMARY_COLUMNS,
    SUMMARY_TABLE,
    latency_rows,
    signal_row,
    summary_row,
)
from sink.writer import ClickHouseWriter, WriterConfig  # noqa: E402
from worker.analysis.live import LiveAnalyzer, LiveConfig  # noqa: E402
from worker.analysis.postcall import PostCallAnalyzer  # noqa: E402
from worker.llm import build_provider  # noqa: E402
from worker.policies import load_policy_store  # noqa: E402
from worker.state import CallState  # noqa: E402


def iso(dt) -> str:
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def redacted_segments(call: dict, base_ts, transcriber=None, audio_dir: Path | None = None):
    """Yield redacted segments for a call, from text or from real audio."""
    redactor = Redactor(
        call["call_id"], RedactorConfig(allowlist=(call.get("agent_name", ""),))
    )

    if transcriber is not None and audio_dir is not None:
        from edge_agent.audio import Segmenter, pcm_to_float32, stream_wav
        from edge_agent.config import EdgeConfig
        from edge_agent.pipeline import TurnMap

        wav = audio_dir / f"{call['call_id']}.wav"
        manifest = audio_dir / f"{call['call_id']}.turns.json"
        if not wav.exists():
            return
        config = EdgeConfig()
        config.realtime = False
        turn_map = TurnMap.from_manifest(manifest)
        segmenter = Segmenter(
            rms_threshold=config.vad_rms_threshold, silence_ms=config.silence_ms,
            partial_interval_ms=config.partial_interval_ms,
            max_utterance_ms=config.max_utterance_ms, frame_ms=config.frame_ms,
        )
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
            out = redactor.push(hyp.text, is_final=True)
            if not out.has_output:
                continue
            yield TranscriptSegment(
                call_id=call["call_id"], seq=seq,
                speaker=turn_map.speaker_at(utt.start_ms, utt.end_ms),
                text=out.text, is_final=True,
                start_ms=utt.start_ms, end_ms=utt.end_ms,
                emitted_at=iso(base_ts + timedelta(milliseconds=utt.end_ms)),
                redactions=list(out.redactions), asr_confidence=hyp.confidence,
                agent_id=call["agent_id"],
            )
            seq += 1
        tail = redactor.flush()
        if tail.has_output:
            yield TranscriptSegment(
                call_id=call["call_id"], seq=seq, speaker="unknown",
                text=tail.text, is_final=True, start_ms=0, end_ms=0,
                emitted_at=iso(base_ts), redactions=list(tail.redactions),
                agent_id=call["agent_id"],
            )
        return

    seq = 0
    for turn in call["turns"]:
        out = redactor.push(turn["text"], is_final=True)
        if not out.has_output:
            continue
        start_ms = turn["idx"] * 4200
        yield TranscriptSegment(
            call_id=call["call_id"], seq=seq, speaker=turn["speaker"],
            text=out.text, is_final=True,
            start_ms=start_ms, end_ms=start_ms + 3800,
            emitted_at=iso(base_ts + timedelta(milliseconds=start_ms + 3800)),
            redactions=list(out.redactions), asr_confidence=0.93,
            agent_id=call["agent_id"],
        )
        seq += 1
    tail = redactor.flush()
    if tail.has_output:
        yield TranscriptSegment(
            call_id=call["call_id"], seq=seq, speaker="unknown", text=tail.text,
            is_final=True, start_ms=0, end_ms=0, emitted_at=iso(base_ts),
            redactions=list(tail.redactions), agent_id=call["agent_id"],
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed ClickHouse from the golden corpus.")
    ap.add_argument("--calls", type=Path, default=REPO / "eval/golden/calls")
    ap.add_argument("--audio-dir", type=Path, default=REPO / "data/audio")
    ap.add_argument("--mode", choices=("text", "audio"), default="text")
    ap.add_argument("--repeats", type=int, default=6,
                    help="Replay the corpus this many times, spread across --days.")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--database", default="edgesense")
    ap.add_argument("--apply-schema", action="store_true")
    ap.add_argument("--truncate", action="store_true",
                    help="Clear existing rows first, for a repeatable demo.")
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()

    setup_logging("seed", "warning")
    rng = random.Random(args.seed)

    writer = ClickHouseWriter(WriterConfig(
        host=args.host, port=args.port, database=args.database, batch_size=1000
    ))
    if args.apply_schema:
        writer.ensure_schema((REPO / "deploy/clickhouse/schema.sql").read_text())
    if args.truncate:
        for table in ("signals", "call_summaries", "segment_latency",
                      "signals_hourly", "agent_daily"):
            writer.query(f"TRUNCATE TABLE IF EXISTS {args.database}.{table}")
        print(f"truncated {args.database}")

    provider = build_provider()
    policies = load_policy_store()
    live_analyzer = LiveAnalyzer(provider, policies, LiveConfig())
    post_analyzer = PostCallAnalyzer(provider, policies)

    transcriber = None
    if args.mode == "audio":
        from edge_agent.asr import Transcriber
        from edge_agent.config import EdgeConfig

        cfg = EdgeConfig()
        transcriber = Transcriber(cfg.whisper_model, cfg.whisper_compute_type,
                                  cfg.model_dir)

    paths = sorted(args.calls.glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]

    now = utc_now()
    totals = {"calls": 0, "signals": 0, "summaries": 0, "latency": 0, "failed": 0}

    for rep in range(args.repeats):
        for path in paths:
            call = json.loads(path.read_text())
            # Spread across the window, biased towards business hours so the
            # hourly charts show a realistic diurnal shape.
            day_offset = rng.uniform(0, args.days)
            hour = rng.choices(range(24), weights=_HOUR_WEIGHTS)[0]
            base_ts = (now - timedelta(days=day_offset)).replace(
                hour=hour, minute=rng.randrange(60), second=rng.randrange(60)
            )

            call_id = call["call_id"] if args.repeats == 1 else f"{call['call_id']}-r{rep}"
            state = CallState(call_id=call_id, agent_id=call["agent_id"])
            redaction_count = 0
            emitted_signals = []

            for segment in redacted_segments(
                {**call, "call_id": call_id}, base_ts, transcriber,
                args.audio_dir if args.mode == "audio" else None,
            ):
                redaction_count += len(segment.redactions)
                state.add(segment)
                if live_analyzer.should_analyse(state):
                    emitted_signals.extend(live_analyzer.analyse(state))

            if not state.turns:
                totals["failed"] += 1
                continue

            for sig in emitted_signals:
                # The analyzer stamps emitted_at with wall-clock now; rewrite
                # it onto the synthetic timeline so history is coherent.
                sig.emitted_at = iso(base_ts + timedelta(milliseconds=sig.window_end_ms))
                writer.add(SIGNAL_TABLE, SIGNAL_COLUMNS, signal_row(sig, state.agent_id))
                for row in latency_rows(sig, state.agent_id):
                    writer.add(LATENCY_TABLE, LATENCY_COLUMNS, row)
                    totals["latency"] += 1
                totals["signals"] += 1

            result = post_analyzer.summarise(state)
            if result.ok and result.summary is not None:
                ended = base_ts + timedelta(milliseconds=state.duration_ms + 1500)
                insights = CallInsights(
                    kind="post_call", call_id=call_id, agent_id=state.agent_id,
                    emitted_at=iso(ended), summary=result.summary,
                )
                row = summary_row(
                    insights, turn_count=len(state.turns),
                    duration_ms=state.duration_ms, redaction_count=redaction_count,
                    started_at=iso(base_ts),
                )
                if row is not None:
                    writer.add(SUMMARY_TABLE, SUMMARY_COLUMNS, row)
                    totals["summaries"] += 1
            else:
                totals["failed"] += 1

            totals["calls"] += 1

        print(f"  pass {rep + 1}/{args.repeats}: {totals['calls']} calls, "
              f"{totals['signals']} signals, {totals['summaries']} summaries")

    writer.flush()
    writer.close()
    print(f"\nseeded: {totals['calls']} calls | {totals['signals']} signals | "
          f"{totals['summaries']} summaries | {totals['latency']} latency rows | "
          f"{totals['failed']} failed")
    return 0


#: Contact-centre style diurnal weighting (UTC), so charts are not uniform noise.
_HOUR_WEIGHTS = [
    1, 1, 1, 1, 1, 2, 4, 8, 14, 18, 20, 19,
    16, 18, 20, 19, 15, 11, 7, 5, 3, 2, 1, 1,
]


if __name__ == "__main__":
    raise SystemExit(main())
