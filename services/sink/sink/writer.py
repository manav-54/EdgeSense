"""Batched ClickHouse writer.

ClickHouse wants few large inserts, not many small ones: every insert creates a
part, and a flood of single-row inserts produces thousands of parts that the
merge scheduler then has to chase, degrading read performance and eventually
tripping ``too many parts``. So rows accumulate and flush on whichever comes
first -- batch size or age -- with the age bound keeping a quiet period from
stranding a row indefinitely.

Failure policy: a failed flush retries with backoff, and rows are kept in the
buffer across attempts so a transient outage does not lose insights. If the
buffer exceeds its hard ceiling the oldest rows are dropped and counted,
because unbounded growth would take the process down and lose everything
rather than the oldest something.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from edgesense_core.timeutil import monotonic_ms

from sink.obs import (
    BUFFER_DEPTH,
    FLUSH_LATENCY,
    METRICS_AVAILABLE,
    ROWS_DROPPED,
    ROWS_WRITTEN,
    WRITE_ERRORS,
    get_logger,
)

log = get_logger(__name__)


@dataclass
class WriterConfig:
    host: str = "localhost"
    port: int = 8123
    database: str = "edgesense"
    user: str = "default"
    password: str = ""
    batch_size: int = 500
    max_age_s: float = 2.0
    #: Hard ceiling per table before shedding. Sized so a few minutes of
    #: outage at demo throughput is absorbed without unbounded memory.
    max_buffer: int = 50_000
    max_retries: int = 5


@dataclass
class TableBuffer:
    columns: list[str]
    rows: list[list[Any]] = field(default_factory=list)
    oldest_at: float = 0.0


class ClickHouseWriter:
    def __init__(self, config: WriterConfig) -> None:
        import clickhouse_connect

        self.config = config
        self._client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            database=config.database,
            username=config.user,
            password=config.password,
            connect_timeout=10,
            send_receive_timeout=30,
        )
        self._buffers: dict[str, TableBuffer] = {}
        self._lock = threading.Lock()
        log.info("clickhouse writer connected",
                 host=config.host, port=config.port, database=config.database)

    def ensure_schema(self, schema_sql: str) -> None:
        """Apply the schema. Statements are idempotent (IF NOT EXISTS)."""
        for statement in _split_statements(schema_sql):
            self._client.command(statement)
        log.info("schema applied")

    def add(self, table: str, columns: list[str], row: list[Any]) -> None:
        with self._lock:
            buf = self._buffers.get(table)
            if buf is None:
                buf = TableBuffer(columns=columns)
                self._buffers[table] = buf
            if not buf.rows:
                buf.oldest_at = time.monotonic()
            buf.rows.append(row)

            if len(buf.rows) > self.config.max_buffer:
                overflow = len(buf.rows) - self.config.max_buffer
                del buf.rows[:overflow]
                if METRICS_AVAILABLE:
                    ROWS_DROPPED.labels(table=table).inc(overflow)
                log.error("write buffer overflowed; dropped oldest rows",
                          table=table, dropped=overflow)
            if METRICS_AVAILABLE:
                BUFFER_DEPTH.labels(table=table).set(len(buf.rows))

        if self._should_flush(table):
            self.flush(table)

    def _should_flush(self, table: str) -> bool:
        with self._lock:
            buf = self._buffers.get(table)
            if buf is None or not buf.rows:
                return False
            if len(buf.rows) >= self.config.batch_size:
                return True
            return (time.monotonic() - buf.oldest_at) >= self.config.max_age_s

    def flush_due(self) -> None:
        """Flush any table whose buffer has aged out. Called by a ticker."""
        for table in list(self._buffers):
            if self._should_flush(table):
                self.flush(table)

    def flush(self, table: str | None = None) -> None:
        tables = [table] if table else list(self._buffers)
        for name in tables:
            with self._lock:
                buf = self._buffers.get(name)
                if buf is None or not buf.rows:
                    continue
                rows, columns = buf.rows, buf.columns
                buf.rows = []
            self._write(name, columns, rows)

    def _write(self, table: str, columns: list[str], rows: list[list[Any]]) -> None:
        t0 = monotonic_ms()
        delay = 0.25
        for attempt in range(self.config.max_retries):
            try:
                self._client.insert(table, rows, column_names=columns)
            except Exception as exc:
                if METRICS_AVAILABLE:
                    WRITE_ERRORS.labels(table=table).inc()
                if attempt == self.config.max_retries - 1:
                    # Out of retries. Push the rows back so the next flush
                    # tries again rather than discarding them here.
                    with self._lock:
                        buf = self._buffers.setdefault(table, TableBuffer(columns=columns))
                        buf.rows = rows + buf.rows
                        buf.oldest_at = time.monotonic()
                    log.error("clickhouse insert failed; rows requeued",
                              table=table, rows=len(rows), error=str(exc)[:300])
                    return
                log.warning("clickhouse insert failed; retrying",
                            table=table, attempt=attempt + 1, error=str(exc)[:200])
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue

            elapsed = monotonic_ms() - t0
            if METRICS_AVAILABLE:
                ROWS_WRITTEN.labels(table=table).inc(len(rows))
                FLUSH_LATENCY.labels(table=table).observe(elapsed / 1000.0)
                BUFFER_DEPTH.labels(table=table).set(len(self._buffers[table].rows))
            log.debug("flushed", table=table, rows=len(rows), ms=round(elapsed, 1))
            return

    def query(self, sql: str) -> list[tuple]:
        return self._client.query(sql).result_rows

    def close(self) -> None:
        self.flush()
        self._client.close()


def _split_statements(sql: str) -> list[str]:
    """Split a schema file on semicolons, ignoring comments and blanks."""
    out: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(current).strip().rstrip(";").strip()
            if statement:
                out.append(statement)
            current = []
    tail = "\n".join(current).strip().rstrip(";").strip()
    if tail:
        out.append(tail)
    return out
