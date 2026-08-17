"""Read API and live feed for the portal.

This lives in the sink rather than in a sixth service because the sink already
owns the ClickHouse connection and already consumes ``call.insights``. Adding
a reader alongside the writer keeps one service responsible for the analytics
store, which is how the architecture is described; splitting them would mean
two deployments sharing one schema and one credential for no benefit at this
size. DESIGN.md notes when that stops being true.

The live feed subscribes to both Kafka topics and fans out to connected
browsers. If Kafka is unreachable the REST endpoints keep working from
ClickHouse and the socket reports a degraded state -- a supervisor losing the
live tail should not also lose the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sink.obs import get_logger, setup_logging

log = get_logger(__name__)

CH_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8123"))
CH_DB = os.environ.get("CLICKHOUSE_DB", "edgesense")
CH_USER = os.environ.get("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "")
BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:9092")
TOPIC_SEGMENTS = os.environ.get("TOPIC_SEGMENTS", "transcript.segments")
TOPIC_INSIGHTS = os.environ.get("TOPIC_INSIGHTS", "call.insights")

#: Recent live events kept in memory so a browser that connects mid-call sees
#: the conversation so far instead of an empty pane.
REPLAY_BUFFER = 400


@dataclass
class Hub:
    """Fan-out of live pipeline events to connected browsers."""

    clients: set[WebSocket] = field(default_factory=set)
    recent: deque = field(default_factory=lambda: deque(maxlen=REPLAY_BUFFER))
    kafka_connected: bool = False

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        await ws.send_text(json.dumps({
            "type": "hello",
            "kafka_connected": self.kafka_connected,
            "replay": list(self.recent),
        }))

    def drop(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, event: dict[str, Any]) -> None:
        self.recent.append(event)
        if not self.clients:
            return
        payload = json.dumps(event, default=str)
        dead: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                # A browser that closed mid-send is normal, not an error.
                dead.append(ws)
        for ws in dead:
            self.drop(ws)


hub = Hub()


class Store:
    """Thin ClickHouse query wrapper returning dicts.

    One client **per thread**, not one per process. FastAPI runs sync endpoints
    in a threadpool, so the dashboard's five concurrent panel requests land on
    five different threads at once. A clickhouse_connect client is not safe to
    query concurrently: sharing one produced intermittent 500s that never
    reproduced under sequential curl, only under a real browser loading the
    whole dashboard at once.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._clients: list = []
        self._lock = threading.Lock()
        self._connect()  # fail fast at startup rather than on first request

    def _connect(self):
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, database=CH_DB,
            username=CH_USER, password=CH_PASSWORD,
            connect_timeout=5, send_receive_timeout=30,
        )
        self._local.client = client
        with self._lock:
            self._clients.append(client)
        return client

    @property
    def client(self):
        client = getattr(self._local, "client", None)
        return client if client is not None else self._connect()

    def rows(self, sql: str, params: dict | None = None) -> list[dict]:
        result = self.client.query(sql, parameters=params or {})
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    def close(self) -> None:
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()


store: Store | None = None


async def consume_kafka() -> None:
    """Bridge Kafka into the WebSocket hub."""
    try:
        from aiokafka import AIOKafkaConsumer
    except Exception:
        log.warning("aiokafka unavailable; live feed disabled")
        return

    while True:
        consumer = AIOKafkaConsumer(
            TOPIC_SEGMENTS, TOPIC_INSIGHTS,
            bootstrap_servers=BROKERS,
            # A fresh group per process: the portal is a tail, not a durable
            # consumer. Sharing a group id would make two open browsers steal
            # each other's partitions.
            group_id=f"edgesense-portal-{os.getpid()}",
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        try:
            await consumer.start()
            hub.kafka_connected = True
            log.info("live feed connected", brokers=BROKERS)
            async for message in consumer:
                try:
                    payload = json.loads(message.value)
                except Exception:
                    continue
                if message.topic == TOPIC_SEGMENTS:
                    await hub.broadcast({"type": "segment", "data": payload})
                else:
                    kind = payload.get("kind")
                    if kind == "live":
                        for signal in payload.get("signals", []):
                            await hub.broadcast({"type": "signal", "data": signal})
                    else:
                        await hub.broadcast({"type": "summary", "data": payload})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            hub.kafka_connected = False
            log.warning("live feed disconnected; retrying", error=str(exc)[:200])
            await asyncio.sleep(5)
        finally:
            try:
                await consumer.stop()
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    setup_logging(os.environ.get("LOG_LEVEL", "info"))
    store = Store()
    task = asyncio.create_task(consume_kafka())
    yield
    task.cancel()
    if store:
        store.close()


app = FastAPI(title="EdgeSense API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("PORTAL_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def db() -> Store:
    if store is None:
        raise RuntimeError("store not initialised")
    return store


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    try:
        db().rows("SELECT 1")
        clickhouse_ok = True
    except Exception:
        clickhouse_ok = False
    return {
        "status": "ok" if clickhouse_ok else "degraded",
        "clickhouse": clickhouse_ok,
        "live_feed": hub.kafka_connected,
        "clients": len(hub.clients),
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/api/dashboard/overview")
def overview(days: int = Query(30, ge=1, le=365)) -> dict:
    rows = db().rows(
        """
        SELECT
            uniq(call_id)                                   AS calls,
            countIf(escalated = 1)                          AS escalations,
            sum(length(compliance_violations))              AS violations,
            countIf(resolution = 'resolved')                AS resolved,
            round(avg(sentiment_end - sentiment_start), 3)  AS avg_sentiment_delta,
            round(avg(turn_count), 1)                       AS avg_turns,
            sum(redaction_count)                            AS redactions
        FROM call_summaries
        WHERE ended_date >= today() - {days:UInt16}
        """,
        {"days": days},
    )
    return rows[0] if rows else {}


@app.get("/api/dashboard/sentiment")
def sentiment(days: int = Query(14, ge=1, le=90)) -> list[dict]:
    return db().rows(
        """
        SELECT
            toStartOfHour(ended_at)                        AS hour,
            count()                                        AS calls,
            round(avg(sentiment_start), 3)                 AS avg_start,
            round(avg(sentiment_end), 3)                   AS avg_end,
            round(avg(sentiment_end - sentiment_start), 3) AS avg_delta,
            countIf(sentiment_end < -0.4)                  AS ended_unhappy
        FROM call_summaries
        WHERE ended_date >= today() - {days:UInt16}
        GROUP BY hour
        ORDER BY hour
        """,
        {"days": days},
    )


@app.get("/api/dashboard/violations")
def violations(days: int = Query(30, ge=1, le=365)) -> list[dict]:
    return db().rows(
        """
        SELECT
            v.policy_id                                            AS policy_id,
            v.calls_with_violation                                 AS calls_with_violation,
            t.total_calls                                          AS total_calls,
            round(100 * v.calls_with_violation / t.total_calls, 2) AS violation_rate_pct,
            v.severe_events                                        AS severe_events,
            round(v.avg_confidence, 3)                             AS avg_confidence
        FROM
        (
            SELECT policy_id,
                   uniq(call_id)                            AS calls_with_violation,
                   countIf(severity IN ('critical','high')) AS severe_events,
                   avg(confidence)                          AS avg_confidence
            FROM signals
            WHERE signal_type = 'compliance_violation'
              AND emitted_date >= today() - {days:UInt16}
              AND policy_id != ''
            GROUP BY policy_id
        ) AS v
        CROSS JOIN
        (
            SELECT greatest(uniq(call_id), 1) AS total_calls
            FROM call_summaries
            WHERE ended_date >= today() - {days:UInt16}
        ) AS t
        ORDER BY violation_rate_pct DESC
        """,
        {"days": days},
    )


@app.get("/api/dashboard/intents")
def intents(days: int = Query(30, ge=1, le=365)) -> list[dict]:
    return db().rows(
        """
        SELECT
            primary_intent,
            count()                                                    AS calls,
            round(100 * count() / sum(count()) OVER (), 2)             AS share_pct,
            countIf(escalated = 1)                                     AS escalated_calls,
            round(100 * countIf(escalated = 1) / count(), 2)           AS escalation_rate_pct,
            round(100 * countIf(resolution = 'resolved') / count(), 2) AS resolved_pct,
            round(avg(turn_count), 1)                                  AS avg_turns
        FROM call_summaries
        WHERE ended_date >= today() - {days:UInt16}
        GROUP BY primary_intent
        ORDER BY calls DESC
        """,
        {"days": days},
    )


@app.get("/api/dashboard/agents")
def agents(days: int = Query(30, ge=1, le=365)) -> list[dict]:
    return db().rows(
        """
        SELECT
            agent_id,
            uniqMerge(calls)                                     AS call_count,
            sum(escalations)                                     AS escalation_count,
            round(100 * sum(escalations) / uniqMerge(calls), 2)  AS escalation_rate_pct,
            sum(violations)                                      AS violation_count,
            round(100 * sum(resolved) / uniqMerge(calls), 2)     AS resolved_pct,
            round(avgMerge(avg_sentiment_delta), 3)              AS sentiment_delta,
            round(avgMerge(avg_turns), 1)                        AS turns_per_call
        FROM agent_daily
        WHERE day >= today() - {days:UInt16}
        GROUP BY agent_id
        HAVING call_count >= 3
        ORDER BY escalation_rate_pct DESC, violation_count DESC
        """,
        {"days": days},
    )


@app.get("/api/latency")
def latency(days: int = Query(7, ge=1, le=90)) -> list[dict]:
    return db().rows(
        """
        SELECT
            stage,
            count()                                               AS samples,
            round(quantile(0.50)(duration_ms), 1)                 AS p50_ms,
            round(quantile(0.95)(duration_ms), 1)                 AS p95_ms,
            round(quantile(0.99)(duration_ms), 1)                 AS p99_ms,
            round(max(duration_ms), 1)                            AS max_ms,
            round(100 * countIf(duration_ms > 2000) / count(), 3) AS pct_over_budget
        FROM segment_latency
        WHERE emitted_date >= today() - {days:UInt16}
        GROUP BY stage
        ORDER BY p95_ms DESC
        """,
        {"days": days},
    )


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


@app.get("/api/calls")
def recent_calls(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return db().rows(
        """
        SELECT
            call_id, agent_id, primary_intent, resolution, escalated,
            sentiment_start, sentiment_end, turn_count, redaction_count,
            length(compliance_violations) AS violation_count,
            compliance_violations, started_at, ended_at, summary
        FROM call_summaries
        ORDER BY ended_at DESC
        LIMIT {limit:UInt16}
        """,
        {"limit": limit},
    )


@app.get("/api/calls/{call_id}")
def call_detail(call_id: str) -> dict:
    summary = db().rows(
        "SELECT * FROM call_summaries WHERE call_id = {call_id:String} "
        "ORDER BY generated_at DESC LIMIT 1",
        {"call_id": call_id},
    )
    signals = db().rows(
        """
        SELECT signal_id, signal_type, label, severity, policy_id, confidence,
               rationale, emitted_at, window_start_ms, window_end_ms,
               evidence_seq, evidence_start_ms, evidence_end_ms,
               evidence_speaker, evidence_quote,
               latency_segment_to_signal_ms, model_name, prompt_version
        FROM signals
        WHERE call_id = {call_id:String}
        ORDER BY emitted_at
        """,
        {"call_id": call_id},
    )
    return {
        "call_id": call_id,
        "summary": summary[0] if summary else None,
        "signals": signals,
    }


@app.get("/api/policies")
def policies() -> list[dict]:
    """Policy catalog, so the portal can render a violation's full text."""
    from worker.policies import load_policy_store

    return [p.as_dict() for p in load_policy_store().all()]


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


@app.websocket("/api/live")
async def live(ws: WebSocket) -> None:
    await hub.register(ws)
    try:
        while True:
            # The portal does not send anything; this read exists to observe
            # the disconnect.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        hub.drop(ws)
