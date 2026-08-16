"""Execute the analytical queries and capture real EXPLAIN output.

Writes docs/clickhouse-explain.md. Nothing in that document is hand-written
output: every plan, granule count and timing comes from running the query
against a live server, so the ORDER BY justification in schema.sql is checked
rather than asserted.

The Q0 comparison needs a table large enough for granule skipping to be
visible. The seeded corpus is ~600 signals, which fits in a single 8192-row
granule, so this builds two 4M-row tables that differ *only* in ORDER BY. The
rows there are synthetic on purpose: what is being measured is index
behaviour, not analytics.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCALE_ROWS = 4_000_000

SCALE_INSERT = """INSERT INTO edgesense.{name}
(signal_id, call_id, agent_id, signal_type, label, severity, policy_id,
 confidence, emitted_at, latency_segment_to_signal_ms, model_name, prompt_version)
SELECT
  concat('sig-', toString(number)),
  concat('call-', toString(intDiv(number, 7))),
  concat('agent_', leftPad(toString(number % 60), 2, '0')),
  ['compliance_violation','escalation_risk','intent','sentiment_shift'][(number % 4) + 1],
  ['negative_shift','supervisor_request','billing_dispute','collections','repeat_contact'][(number % 5) + 1],
  ['info','low','medium','high','critical'][(number % 5) + 1],
  ['REC-001','PCI-002','MINI-003','VERIF-004','PROHIB-005','FDCPA-006','RTC-007'][(number % 7) + 1],
  0.5 + (number % 50) / 100.0,
  now64(3) - toIntervalSecond(number % 5184000),
  toFloat32(200 + (number % 1800)),
  'rules', 'live_analysis@v2'
FROM numbers({rows})"""

Q0 = """SELECT policy_id, count() AS events
FROM edgesense.{table}
WHERE signal_type = 'compliance_violation' AND emitted_date >= today() - 7
GROUP BY policy_id ORDER BY events DESC"""

INTERESTING = ("Granules:", "Parts:", "Condition:", "PrimaryKey", "Partition",
               "ReadFromMergeTree", "MinMax", "Keys:", "Skip")


def split_queries(sql: str) -> list[tuple[str, str]]:
    """Split queries.sql into (title, sql) pairs using the banner comments."""
    out: list[tuple[str, str]] = []
    title = "Query"
    buffer: list[str] = []
    for line in sql.splitlines():
        m = re.match(r"^--\s*(Q\d+\..*)$", line.strip())
        if m:
            title = m.group(1).strip()
        if line.strip().startswith("--") or not line.strip():
            continue
        buffer.append(line)
        if line.rstrip().endswith(";"):
            out.append((title, "\n".join(buffer).strip().rstrip(";")))
            buffer = []
    return out


def plan_lines(client, sql: str) -> list[str]:
    rows = client.query(f"EXPLAIN indexes=1 {sql}").result_rows
    return [r[0] for r in rows if any(k in r[0] for k in INTERESTING)]


def timed(client, sql: str, runs: int = 3) -> tuple[float, int]:
    """Best-of-N wall time and row count, to reduce cache noise."""
    best = float("inf")
    rows = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        result = client.query(sql)
        best = min(best, (time.perf_counter() - t0) * 1000)
        rows = len(result.result_rows)
    return best, rows


def build_scale_tables(client) -> None:
    for name, order in (
        ("signals_scale_chosen", "(signal_type, emitted_date, agent_id, call_id, emitted_at)"),
        ("signals_scale_timefirst", "(emitted_at, call_id)"),
    ):
        client.command(f"DROP TABLE IF EXISTS edgesense.{name}")
        client.command(
            f"CREATE TABLE edgesense.{name} AS edgesense.signals "
            f"ENGINE = MergeTree PARTITION BY toYYYYMM(emitted_at) ORDER BY {order}"
        )
        client.command(SCALE_INSERT.format(name=name, rows=SCALE_ROWS))


def granule_summary(lines: list[str]) -> str:
    """Pull the primary-key narrowing out of a plan, for the summary table."""
    for i, line in enumerate(lines):
        if "PrimaryKey" in line:
            for follow in lines[i : i + 5]:
                m = re.search(r"Granules:\s*(\d+)/(\d+)", follow)
                if m:
                    return f"{m.group(1)}/{m.group(2)}"
    return "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--out", type=Path, default=REPO / "docs/clickhouse-explain.md")
    ap.add_argument("--skip-scale", action="store_true",
                    help="Reuse existing scale tables instead of rebuilding them.")
    args = ap.parse_args()

    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=args.host, port=args.port, send_receive_timeout=900
    )
    version = client.query("SELECT version()").result_rows[0][0]

    if not args.skip_scale:
        print("building 4M-row scale tables (this takes a minute)...")
        build_scale_tables(client)

    doc: list[str] = []
    doc.append("# ClickHouse query plans\n")
    doc.append(
        "Generated by `scripts/explain_report.py` against a live server. "
        "Every plan and timing below is captured output, not written by hand.\n"
    )
    doc.append(f"- ClickHouse: `{version}`")

    counts = {}
    for table in ("signals", "call_summaries", "segment_latency"):
        counts[table] = client.query(
            f"SELECT count() FROM edgesense.{table}"
        ).result_rows[0][0]
    doc.append(
        "- Seeded corpus: "
        + ", ".join(f"`{t}` {n:,} rows" for t, n in counts.items())
    )
    doc.append(f"- Index-behaviour tables: {SCALE_ROWS:,} rows each\n")

    # ---- Q0: the ORDER BY justification ---------------------------------
    doc.append("## Q0 — Why this ORDER BY key\n")
    doc.append(
        "Two tables, identical data, differing only in sort key. The query is "
        "the compliance dashboard's headline: violations by policy over the "
        "last week.\n"
    )

    results = {}
    for label, table in (
        ("chosen — `(signal_type, emitted_date, agent_id, call_id, emitted_at)`",
         "signals_scale_chosen"),
        ("rejected — `(emitted_at, call_id)`", "signals_scale_timefirst"),
    ):
        sql = Q0.format(table=table)
        lines = plan_lines(client, sql)
        ms, _ = timed(client, sql)
        results[table] = (label, lines, ms, granule_summary(lines))

    doc.append("| Sort key | PrimaryKey granules | Best-of-3 wall time |")
    doc.append("|---|---|---|")
    for _, (label, _, ms, granules) in results.items():
        doc.append(f"| {label} | {granules} | {ms:.1f} ms |")
    doc.append("")

    for table, (label, lines, ms, _) in results.items():
        doc.append(f"### {label}\n")
        doc.append("```")
        doc.extend(line.rstrip() for line in lines)
        doc.append("```\n")

    doc.append(
        "The chosen key puts `signal_type` first, so an equality predicate on "
        "it collapses the candidate range before anything is read; "
        "`emitted_date` then narrows the remainder to the requested window. "
        "The rejected key sorts by a near-unique timestamp, so the primary "
        "index reports `Condition: true` and contributes nothing — every "
        "granule in the partition is read and discarded during aggregation.\n"
    )

    # ---- Q1..Q5 on the real seeded data ---------------------------------
    doc.append("## Dashboard queries on seeded data\n")
    doc.append(
        "These run against the corpus written by `scripts/seed_pipeline.py`. "
        "The row counts are small, so timings show fixed overhead rather than "
        "scan cost; the plans are what matter here.\n"
    )

    queries = split_queries((REPO / "deploy/clickhouse/queries.sql").read_text())
    for title, sql in queries:
        if title.startswith("Q0"):
            continue
        doc.append(f"### {title}\n")
        try:
            ms, rows = timed(client, sql)
            lines = plan_lines(client, sql)
        except Exception as exc:
            doc.append(f"> query failed: `{str(exc)[:300]}`\n")
            continue
        doc.append(f"`{rows}` rows, best-of-3 **{ms:.1f} ms**\n")
        doc.append("```sql")
        doc.append(sql)
        doc.append("```\n")
        doc.append("```")
        doc.extend(line.rstrip() for line in lines)
        doc.append("```\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(doc) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
