"""Sink: consume call.insights, write to ClickHouse.

    python -m sink.main
    python -m sink.main --apply-schema       # create tables then run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path

from edgesense_core.contracts import CallInsights, require_compatible

from sink.obs import (
    CONSUMER_LAG,
    INSIGHTS_CONSUMED,
    METRICS_AVAILABLE,
    get_logger,
    setup_logging,
    setup_tracing,
    start_metrics,
)
from sink.rows import (
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
from sink.writer import ClickHouseWriter, WriterConfig

log = get_logger(__name__)

DEFAULT_SCHEMA = Path("deploy/clickhouse/schema.sql")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EdgeSense ClickHouse sink")
    ap.add_argument("--brokers", default=os.environ.get("KAFKA_BROKERS", "localhost:9092"))
    ap.add_argument("--topic", default=os.environ.get("TOPIC_INSIGHTS", "call.insights"))
    ap.add_argument("--group-id", default=os.environ.get("SINK_GROUP_ID", "edgesense-sink"))
    ap.add_argument("--ch-host", default=os.environ.get("CLICKHOUSE_HOST", "localhost"))
    ap.add_argument("--ch-port", type=int, default=int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8123")))
    ap.add_argument("--ch-db", default=os.environ.get("CLICKHOUSE_DB", "edgesense"))
    ap.add_argument("--ch-user", default=os.environ.get("CLICKHOUSE_USER", "default"))
    ap.add_argument("--ch-password", default=os.environ.get("CLICKHOUSE_PASSWORD", ""))
    ap.add_argument("--apply-schema", action="store_true")
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--metrics-port", type=int,
                    default=int(os.environ.get("SINK_PROMETHEUS_PORT", "9104")))
    ap.add_argument("--from-beginning", action="store_true")
    return ap


def handle(writer: ClickHouseWriter, insights: CallInsights) -> None:
    """Fan one envelope out into the tables it belongs in."""
    if METRICS_AVAILABLE:
        INSIGHTS_CONSUMED.labels(kind=insights.kind).inc()

    if insights.kind == "live":
        for sig in insights.signals:
            writer.add(SIGNAL_TABLE, SIGNAL_COLUMNS, signal_row(sig, insights.agent_id))
            for row in latency_rows(sig, insights.agent_id):
                writer.add(LATENCY_TABLE, LATENCY_COLUMNS, row)
        return

    row = summary_row(insights)
    if row is not None:
        writer.add(SUMMARY_TABLE, SUMMARY_COLUMNS, row)


async def amain(args: argparse.Namespace) -> int:
    from aiokafka import AIOKafkaConsumer

    writer = ClickHouseWriter(WriterConfig(
        host=args.ch_host, port=args.ch_port, database=args.ch_db,
        user=args.ch_user, password=args.ch_password,
    ))
    if args.apply_schema:
        if not args.schema.exists():
            log.error("schema file not found", path=str(args.schema))
            return 2
        writer.ensure_schema(args.schema.read_text())

    consumer = AIOKafkaConsumer(
        args.topic,
        bootstrap_servers=args.brokers,
        group_id=args.group_id,
        enable_auto_commit=True,
        auto_offset_reset="earliest" if args.from_beginning else "latest",
    )
    await consumer.start()
    log.info("sink started", topic=args.topic, clickhouse=f"{args.ch_host}:{args.ch_port}")

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)

    async def ticker() -> None:
        """Flush aged buffers even when no new messages arrive."""
        while not stopping.is_set():
            await asyncio.sleep(1.0)
            await asyncio.to_thread(writer.flush_due)

    tick = asyncio.create_task(ticker())
    consumed = 0
    try:
        async for message in consumer:
            if stopping.is_set():
                break
            try:
                payload = json.loads(message.value)
                require_compatible(payload.get("schema_version", "1.0"))
                insights = CallInsights.model_validate(payload)
            except Exception as exc:
                # Skip and advance: one malformed envelope must not wedge the
                # partition behind it.
                log.warning("skipping unparseable insight",
                            offset=message.offset, error=str(exc)[:200])
                continue

            await asyncio.to_thread(handle, writer, insights)
            consumed += 1
            if METRICS_AVAILABLE and message.highwater is not None:
                CONSUMER_LAG.labels(
                    topic=message.topic, partition=str(message.partition)
                ).set(max(0, message.highwater - message.offset - 1))
    finally:
        tick.cancel()
        await consumer.stop()
        await asyncio.to_thread(writer.close)
        log.info("sink stopped", consumed=consumed)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(os.environ.get("LOG_LEVEL", "info"))
    setup_tracing("sink", os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""))
    start_metrics(args.metrics_port)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
