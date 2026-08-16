"""Kafka consumption, call lifecycle, and insight publication.

The worker holds per-call state in memory, which is the right trade for a
sliding window (rebuilding it from the log on every message would dominate the
latency budget) and the wrong trade for durability. The consequence is stated
plainly: if a worker dies mid-call, the in-flight window is lost and the call
resumes from the next segment with an empty history. Because segments are
partitioned by ``call_id``, a rebalance moves the whole call to one worker
rather than splitting it, so the damage is bounded to that call. DESIGN.md
covers what a durable version would look like.

Analysis runs in a thread rather than on the event loop. The rules path is
CPU-bound and the agent path blocks on HTTP; either would stall every other
call on the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field

from edgesense_core.contracts import CallInsights, TranscriptSegment, require_compatible
from edgesense_core.timeutil import monotonic_ms, utc_now_iso

from worker.analysis.live import LiveAnalyzer
from worker.analysis.postcall import PostCallAnalyzer
from worker.obs import (
    ACTIVE_CALLS,
    CONSUMER_LAG,
    METRICS_AVAILABLE,
    SEGMENTS_CONSUMED,
    context_from_traceparent,
    get_logger,
    tracer,
)
from worker.state import CallState

log = get_logger(__name__)


@dataclass
class ConsumerConfig:
    brokers: str = "localhost:9092"
    segments_topic: str = "transcript.segments"
    insights_topic: str = "call.insights"
    group_id: str = "edgesense-worker"
    #: A call with no new segments for this long is treated as ended. Set well
    #: above a natural conversational pause -- ending a call early produces a
    #: summary of half a conversation, which is worse than a late one.
    idle_timeout_s: float = 25.0
    sweep_interval_s: float = 5.0
    #: Bound on concurrently tracked calls, so a stuck sweeper cannot grow
    #: memory without limit.
    max_tracked_calls: int = 2000
    auto_offset_reset: str = "latest"


@dataclass
class WorkerStats:
    consumed: int = 0
    finals: int = 0
    live_published: int = 0
    post_published: int = 0
    post_failed: int = 0
    invalid: int = 0
    calls_completed: int = 0
    started_at: float = field(default_factory=time.time)


class InsightWorker:
    def __init__(
        self,
        config: ConsumerConfig,
        live: LiveAnalyzer,
        post_call: PostCallAnalyzer,
    ) -> None:
        self.config = config
        self.live = live
        self.post_call = post_call
        self.calls: dict[str, CallState] = {}
        self.stats = WorkerStats()
        self._consumer = None
        self._producer = None
        self._stopping = asyncio.Event()
        # One analysis at a time per call; different calls proceed in parallel.
        self._locks: dict[str, asyncio.Lock] = {}

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

        self._consumer = AIOKafkaConsumer(
            self.config.segments_topic,
            bootstrap_servers=self.config.brokers,
            group_id=self.config.group_id,
            enable_auto_commit=True,
            auto_offset_reset=self.config.auto_offset_reset,
            # Bounded fetch so a backlog is worked through in measured batches
            # instead of pulling a huge chunk into memory at once.
            max_partition_fetch_bytes=1024 * 1024,
        )
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.brokers,
            linger_ms=20,
            compression_type="snappy",
            acks=1,
        )
        await self._consumer.start()
        await self._producer.start()
        log.info("worker started",
                 brokers=self.config.brokers,
                 consuming=self.config.segments_topic,
                 producing=self.config.insights_topic,
                 group=self.config.group_id)

    async def stop(self) -> None:
        self._stopping.set()
        # Finish the calls we are holding rather than dropping their summaries.
        await self._finalise_all("shutdown")
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        log.info("worker stopped", **self.stats.__dict__)

    async def run(self) -> None:
        sweeper = asyncio.create_task(self._sweep_idle_calls())
        try:
            async for message in self._consumer:
                if self._stopping.is_set():
                    break
                await self._handle(message)
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper

    # -- message handling --------------------------------------------------

    async def _handle(self, message) -> None:
        self.stats.consumed += 1
        try:
            payload = json.loads(message.value)
            require_compatible(payload.get("schema_version", "1.0"))
            segment = TranscriptSegment.model_validate(payload)
        except Exception as exc:
            # A poison message must not stop the partition. It is counted and
            # skipped; the offset advances so one bad payload cannot wedge
            # every call on this partition behind it.
            self.stats.invalid += 1
            log.warning("skipping unparseable segment",
                        topic=message.topic, partition=message.partition,
                        offset=message.offset, error=str(exc)[:200])
            return

        if METRICS_AVAILABLE:
            SEGMENTS_CONSUMED.labels(is_final=str(segment.is_final).lower()).inc()
            if message.highwater is not None:
                CONSUMER_LAG.labels(
                    topic=message.topic, partition=str(message.partition)
                ).set(max(0, message.highwater - message.offset - 1))

        if not segment.is_final:
            return  # partials are previews; analysing them cites vanishing text
        self.stats.finals += 1

        state = self.calls.get(segment.call_id)
        if state is None:
            if len(self.calls) >= self.config.max_tracked_calls:
                log.warning("tracked-call ceiling reached; finalising the oldest",
                            tracked=len(self.calls))
                await self._finalise_oldest()
            state = CallState(call_id=segment.call_id, agent_id=segment.agent_id)
            self.calls[segment.call_id] = state
            self._locks[segment.call_id] = asyncio.Lock()
            if METRICS_AVAILABLE:
                ACTIVE_CALLS.set(len(self.calls))

        state.add(segment)

        async with self._locks[segment.call_id]:
            if self.live.should_analyse(state):
                await self._analyse_live(state)

    async def _analyse_live(self, state: CallState) -> None:
        parent = context_from_traceparent(state.traceparent)
        with tracer("worker").start_as_current_span("worker.live_analysis", context=parent):
            signals = await asyncio.to_thread(self.live.analyse, state)
        if not signals:
            return
        insights = CallInsights(
            kind="live",
            call_id=state.call_id,
            agent_id=state.agent_id,
            emitted_at=utc_now_iso(),
            signals=signals,
            traceparent=state.traceparent,
        )
        await self._publish(insights)
        self.stats.live_published += len(signals)

    async def _analyse_post_call(self, state: CallState, reason: str) -> None:
        t0 = monotonic_ms()
        parent = context_from_traceparent(state.traceparent)
        with tracer("worker").start_as_current_span("worker.post_call", context=parent):
            result = await asyncio.to_thread(self.post_call.summarise, state)

        if not result.ok or result.summary is None:
            self.stats.post_failed += 1
            log.error("post-call analysis failed; publishing nothing",
                      call_id=state.call_id, reason=reason, error=result.error)
            return

        insights = CallInsights(
            kind="post_call",
            call_id=state.call_id,
            agent_id=state.agent_id,
            emitted_at=utc_now_iso(),
            summary=result.summary,
            traceparent=state.traceparent,
        )
        await self._publish(insights)
        self.stats.post_published += 1
        log.info("call finalised", call_id=state.call_id, reason=reason,
                 turns=len(state.turns), signals=state.signals_emitted,
                 duration_ms=round(monotonic_ms() - t0, 1))

    async def _publish(self, insights: CallInsights) -> None:
        await self._producer.send_and_wait(
            self.config.insights_topic,
            key=insights.call_id.encode("utf-8"),
            value=insights.model_dump_json().encode("utf-8"),
        )

    # -- call lifecycle ----------------------------------------------------

    async def _sweep_idle_calls(self) -> None:
        """Finalise calls that have gone quiet."""
        while not self._stopping.is_set():
            await asyncio.sleep(self.config.sweep_interval_s)
            for call_id in list(self.calls):
                state = self.calls.get(call_id)
                if state is None or state.ended:
                    continue
                if state.idle_seconds() >= self.config.idle_timeout_s:
                    await self._finalise(call_id, "idle_timeout")

    async def _finalise(self, call_id: str, reason: str) -> None:
        state = self.calls.get(call_id)
        if state is None or state.ended:
            return
        lock = self._locks.get(call_id)
        if lock is None:
            return
        async with lock:
            state.ended = True
            await self._analyse_post_call(state, reason)
        self.calls.pop(call_id, None)
        self._locks.pop(call_id, None)
        self.stats.calls_completed += 1
        if METRICS_AVAILABLE:
            ACTIVE_CALLS.set(len(self.calls))

    async def _finalise_oldest(self) -> None:
        oldest = min(self.calls.values(), key=lambda s: s.started_at or "", default=None)
        if oldest is not None:
            await self._finalise(oldest.call_id, "capacity")

    async def _finalise_all(self, reason: str) -> None:
        for call_id in list(self.calls):
            await self._finalise(call_id, reason)
