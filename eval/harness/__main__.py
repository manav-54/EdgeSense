"""Evaluation CLI.

    python -m harness                                   # full text-mode eval
    python -m harness --mode audio                      # full path, real ASR
    python -m harness --categories adversarial          # just the hard set
    python -m harness --compare v1 v2                   # prompt regression
    python -m harness --compare-reports a.json b.json   # diff saved runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "eval") not in sys.path:
    sys.path.insert(0, str(REPO / "eval"))

from harness.regression import (  # noqa: E402
    compare,
    format_table,
    has_gate_failure,
    load_report,
    run_pair,
)
from harness.report import format_run  # noqa: E402
from harness.runner import RunConfig, run  # noqa: E402

DEFAULT_REPORTS = REPO / "eval/reports"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="harness", description="EdgeSense evaluation")
    ap.add_argument("--mode", choices=("text", "audio"), default="text")
    ap.add_argument("--provider", default=None, help="azure | offline | auto")
    ap.add_argument("--live-prompt", default="latest")
    ap.add_argument("--post-prompt", default="latest")
    ap.add_argument("--ner", default="auto", help="auto | spacy | heuristic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--categories", nargs="*", default=[],
                    help="clean pii_heavy compliance escalation ambiguous adversarial")
    ap.add_argument("--no-agent", action="store_true",
                    help="Fast path only; skips the LLM agent loop.")
    ap.add_argument("--out", type=Path, default=None, help="Write the JSON report here.")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="Run the corpus against two prompt versions and diff.")
    ap.add_argument("--compare-which", choices=("live", "post"), default="live")
    ap.add_argument("--compare-reports", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                    help="Diff two saved JSON reports without re-running.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    DEFAULT_REPORTS.mkdir(parents=True, exist_ok=True)

    if args.compare_reports:
        before = load_report(Path(args.compare_reports[0]))
        after = load_report(Path(args.compare_reports[1]))
        comparisons = compare(before, after)
        print(format_table(
            comparisons,
            f"{before['config'].get('live_prompt', '?')} ({Path(args.compare_reports[0]).name})",
            f"{after['config'].get('live_prompt', '?')} ({Path(args.compare_reports[1]).name})",
        ))
        return 1 if has_gate_failure(comparisons) else 0

    config = RunConfig(
        mode=args.mode,
        provider=args.provider,
        live_prompt=args.live_prompt,
        post_prompt=args.post_prompt,
        ner_backend=args.ner,
        limit=args.limit,
        categories=tuple(args.categories),
        use_agent=not args.no_agent,
    )

    if args.compare:
        before_version, after_version = args.compare
        before, after = run_pair(config, before_version, after_version, args.compare_which)

        for result, version in ((before, before_version), (after, after_version)):
            path = DEFAULT_REPORTS / f"run-{args.compare_which}-{version}.json"
            path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
            if not args.quiet:
                print(format_run(result))

        comparisons = compare(before.as_dict(), after.as_dict())
        table = format_table(comparisons,
                             f"{args.compare_which}@{before_version}",
                             f"{args.compare_which}@{after_version}")
        print(table)
        (DEFAULT_REPORTS / "regression.md").write_text(
            f"# Prompt regression\n\n```\n{table}\n```\n"
        )
        return 1 if has_gate_failure(comparisons) else 0

    result = run(config)
    if not args.quiet:
        print(format_run(result))

    out = args.out or DEFAULT_REPORTS / f"run-{config.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    print(f"report written to {out}")

    # A leak is a failing build, not a metric to note and move past.
    leaks = result.redaction.as_dict()["leak_count"]
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
