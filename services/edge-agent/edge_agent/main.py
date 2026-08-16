"""Edge agent CLI.

    python -m edge_agent.main --audio data/audio/<call>.wav
    python -m edge_agent.main --audio ... --out segments.jsonl   # no ingest needed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edge_agent.asr import Transcriber
from edge_agent.config import EdgeConfig
from edge_agent.obs import get_logger, setup_logging, setup_tracing, start_metrics
from edge_agent.pipeline import EdgePipeline, TurnMap, new_call_id
from edge_agent.transport import JSONLTransport, WebSocketTransport

log = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EdgeSense edge agent")
    ap.add_argument("--audio", type=Path, required=True, help="16 kHz mono WAV to stream")
    ap.add_argument("--call-id", default=None)
    ap.add_argument("--agent-id", default=None)
    ap.add_argument("--agent-name", default=None,
                    help="Excluded from PERSON redaction; the client already knows it.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="<call>.turns.json for speaker labels. Defaults to sibling file.")
    ap.add_argument("--url", default=None, help="Ingest WebSocket URL")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write segments to JSONL instead of connecting to ingest")
    ap.add_argument("--model", default=None, help="whisper model size (tiny.en, base.en, ...)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="Stream as fast as the CPU allows instead of at wall-clock pace")
    ap.add_argument("--no-partials", action="store_true")
    ap.add_argument("--metrics-port", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = EdgeConfig()
    if args.url:
        config.ingest_url = args.url
    if args.model:
        config.whisper_model = args.model
    if args.no_realtime:
        config.realtime = False
    if args.no_partials:
        config.emit_partials = False

    setup_logging(config.log_level)
    setup_tracing("edge-agent", config.otlp_endpoint)
    start_metrics(args.metrics_port or config.prometheus_port)

    if not args.audio.exists():
        log.error("audio file not found", path=str(args.audio))
        print(
            f"error: {args.audio} does not exist.\n"
            "Generate the corpus audio first:\n"
            "  python -m tools.corpus.generate && python -m tools.audio.synthesize",
            file=sys.stderr,
        )
        return 2

    call_id = args.call_id or new_call_id()
    manifest_path = args.manifest or args.audio.with_suffix("").with_suffix(".turns.json")
    if not manifest_path.exists():
        manifest_path = args.audio.parent / f"{args.audio.stem}.turns.json"
    turn_map = TurnMap.from_manifest(manifest_path)

    agent_id = args.agent_id
    agent_name = args.agent_name
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        agent_id = agent_id or meta.get("agent_id")

    transcriber = Transcriber(
        model_size=config.whisper_model,
        compute_type=config.whisper_compute_type,
        download_root=config.model_dir,
    )

    if args.out:
        transport = JSONLTransport(args.out)
    else:
        transport = WebSocketTransport(config.ingest_url, call_id=call_id)
        transport.wait_ready()

    pipeline = EdgePipeline(
        call_id, config, transcriber, transport,
        turn_map=turn_map, agent_id=agent_id, agent_name=agent_name,
    )

    try:
        stats = pipeline.run(args.audio)
    finally:
        transport.close()

    print(
        f"call {stats.call_id}: {stats.segments_final} final "
        f"({stats.segments_partial} partial), {stats.redactions} redactions "
        f"{stats.per_type}, held {stats.held_events}x, "
        f"rtf {stats.realtime_factor:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
