"""The network boundary.

Everything above this module may hold raw PII. Nothing below it ever does.
That makes ``Transport.send`` the single choke point worth auditing, and it is
where the last-line assertion lives: a segment is re-validated against the
contract immediately before serialisation, so a caller that hand-builds a dict
with an extra field cannot smuggle one past ``extra="forbid"``.

Two implementations share one interface so tests can capture bytes at exactly
the place the real client would have written them -- the egress test asserts
on the same buffer the socket would have received, not on a mock's arguments.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Protocol

from edgesense_core.contracts import TranscriptSegment

from edge_agent.obs import get_logger

log = get_logger(__name__)


class Transport(Protocol):
    def send(self, segment: TranscriptSegment) -> None: ...
    def close(self) -> None: ...


def encode(segment: TranscriptSegment) -> str:
    """Serialise a segment for the wire, re-validating on the way out.

    ``model_validate`` on an already-typed object looks redundant. It is not:
    it re-runs the span checks that guarantee every ``RedactionRef`` still
    points at its own placeholder, catching a caller that mutated ``text``
    after redaction.
    """
    checked = TranscriptSegment.model_validate(segment.model_dump())
    return checked.model_dump_json()


class WebSocketTransport:
    """Sync WebSocket client with a bounded outbound queue and reconnect.

    The queue is bounded on purpose. If ingest stalls, an unbounded queue turns
    a downstream outage into unbounded memory growth on the operator's laptop
    and, eventually, a crash that loses the whole call. Bounded means we make
    an explicit choice under pressure -- see ``on_full``.
    """

    def __init__(
        self,
        url: str,
        *,
        call_id: str,
        max_queue: int = 512,
        connect_timeout: float = 5.0,
        on_full: str = "block",
    ) -> None:
        self.url = url
        self.call_id = call_id
        self.on_full = on_full
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._connect_timeout = connect_timeout
        self._closed = threading.Event()
        self._dropped = 0
        self._sent = 0
        self._thread = threading.Thread(
            target=self._run, name=f"ws-{call_id}", daemon=True
        )
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread.start()

    # -- public ------------------------------------------------------------

    def send(self, segment: TranscriptSegment) -> None:
        payload = encode(segment)
        if self.on_full == "drop_partials" and not segment.is_final:
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                self._dropped += 1
                log.warning("dropped partial under backpressure",
                            call_id=self.call_id, dropped=self._dropped)
            return
        # Finals block: losing a final segment loses transcript content, which
        # is worse than the latency of waiting for the queue to drain.
        self._q.put(payload)

    def wait_ready(self, timeout: float = 10.0) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"transport not ready after {timeout}s")
        if self._error:
            raise self._error

    def close(self, drain_timeout: float = 10.0) -> None:
        self._q.put(None)
        self._thread.join(timeout=drain_timeout)
        self._closed.set()

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": self._sent, "dropped": self._dropped, "queued": self._q.qsize()}

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        from websockets.sync.client import connect

        backoff = 0.25
        ws = None
        while not self._closed.is_set():
            try:
                ws = connect(
                    f"{self.url}?call_id={self.call_id}",
                    open_timeout=self._connect_timeout,
                    max_queue=None,
                )
                self._ready.set()
                backoff = 0.25
                self._pump(ws)
                return
            except Exception as exc:
                self._error = exc
                log.warning("ingest connection failed; retrying",
                            call_id=self.call_id, error=str(exc), backoff_s=backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                    ws = None

    def _pump(self, ws) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            ws.send(item)
            self._sent += 1


class CapturingTransport:
    """Records exactly what would have gone on the wire.

    This is the oracle for the egress test. It stores the encoded *bytes*, not
    the objects, so an assertion about "no raw PII crossed the boundary" is
    made against the same representation a packet capture would show.
    """

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.segments: list[TranscriptSegment] = []

    def send(self, segment: TranscriptSegment) -> None:
        self.frames.append(encode(segment).encode("utf-8"))
        self.segments.append(segment)

    def close(self, drain_timeout: float = 0.0) -> None:
        return None

    def wait_ready(self, timeout: float = 0.0) -> None:
        return None

    @property
    def wire_bytes(self) -> bytes:
        """Every byte that crossed the boundary, concatenated."""
        return b"\n".join(self.frames)

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": len(self.frames), "dropped": 0, "queued": 0}


class JSONLTransport:
    """Writes segments to a file. Used by the eval harness and the load test
    when the goal is to measure the edge in isolation, without ingest."""

    def __init__(self, path) -> None:
        self._fh = open(path, "w", encoding="utf-8")
        self._sent = 0

    def send(self, segment: TranscriptSegment) -> None:
        self._fh.write(encode(segment) + "\n")
        self._sent += 1

    def close(self, drain_timeout: float = 0.0) -> None:
        self._fh.close()

    def wait_ready(self, timeout: float = 0.0) -> None:
        return None

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": self._sent, "dropped": 0, "queued": 0}
