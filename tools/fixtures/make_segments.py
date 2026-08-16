"""Emit wire fixtures from the real Python edge path.

The Go ingest service re-declares the segment struct because it cannot import
pydantic. That duplication is only safe if something checks it, so this writes
segments produced by the actual redactor and the actual encoder, and the Go
contract test validates them. If either side drifts, the Go test fails.

Run: ``python -m tools.fixtures.make_segments``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "services" / "edge-agent"))

from edgesense_core.contracts import TranscriptSegment  # noqa: E402
from edge_agent.redact.redactor import Redactor, RedactorConfig  # noqa: E402
from edge_agent.transport import encode  # noqa: E402


def build(calls_dir: Path, limit: int) -> list[str]:
    lines: list[str] = []
    paths = sorted(calls_dir.glob("*.json"))[:limit]
    for path in paths:
        call = json.loads(path.read_text())
        redactor = Redactor(
            call["call_id"], RedactorConfig(allowlist=(call.get("agent_name", ""),))
        )
        seq = 0
        for turn in call["turns"]:
            out = redactor.push(turn["text"], is_final=True)
            if not out.has_output:
                continue
            segment = TranscriptSegment(
                call_id=call["call_id"],
                seq=seq,
                speaker=turn["speaker"],
                text=out.text,
                is_final=True,
                start_ms=turn["idx"] * 1000,
                end_ms=turn["idx"] * 1000 + 900,
                emitted_at="2026-08-16T12:00:00.000000Z",
                redactions=list(out.redactions),
                asr_confidence=0.92,
                agent_id=call["agent_id"],
                traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            )
            lines.append(encode(segment))
            seq += 1
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=Path, default=REPO / "eval" / "golden" / "calls")
    ap.add_argument("--out", type=Path,
                    default=REPO / "services" / "ingest" / "testdata" / "segments.jsonl")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    lines = build(args.calls, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    with_redactions = sum(1 for line in lines if json.loads(line)["redactions"])
    print(f"wrote {len(lines)} segments ({with_redactions} carrying redactions) -> {args.out}")


if __name__ == "__main__":
    main()
