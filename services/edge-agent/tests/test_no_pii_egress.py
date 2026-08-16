"""The hard requirement: prove no raw PII string ever crosses the network boundary.

The tests escalate in how little they trust the implementation:

1. ``test_redacted_output_contains_no_pii`` -- redactor output, per call.
2. ``test_transport_wire_bytes_contain_no_pii`` -- the encoded bytes handed to
   the transport, which is what a packet capture would show.
3. ``test_real_websocket_server_receives_no_pii`` -- bytes actually read off a
   real socket by a real server process boundary. Nothing is mocked; if the
   client sent it, this test sees it.
4. ``test_no_long_digit_run_survives`` -- a shape invariant that holds even for
   values the corpus never told us about.
5. ``test_contract_cannot_carry_raw_pii`` -- the type system refuses to model
   a payload with an original value in it.
6. ``test_logs_scrub_pii`` -- the other egress path everyone forgets.

Together they cover the two ways this guarantee dies in practice: someone adds
a field, or someone adds a log line.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from leakcheck import find_leaks, long_digit_runs

from edgesense_core.contracts import TranscriptSegment
from edge_agent.obs import scrub
from edge_agent.redact.redactor import Redactor, RedactorConfig
from edge_agent.transport import CapturingTransport, encode


def _all_pii(call: dict) -> list[dict]:
    return [span for turn in call["turns"] for span in turn["pii"]]


def _stream_call(call: dict) -> tuple[Redactor, list[str]]:
    """Push every turn of a call through the streaming redactor."""
    redactor = Redactor(
        call["call_id"],
        RedactorConfig(allowlist=(call.get("agent_name", ""),)),
    )
    emitted: list[str] = []
    for turn in call["turns"]:
        out = redactor.push(turn["text"], is_final=True)
        if out.has_output:
            emitted.append(out.text)
    tail = redactor.flush()
    if tail.has_output:
        emitted.append(tail.text)
    return redactor, emitted


# ---------------------------------------------------------------------------
# 1. Redactor output
# ---------------------------------------------------------------------------


def test_redacted_output_contains_no_pii(golden_calls):
    """No corpus PII value survives redaction, in any of its three spaces."""
    all_leaks = []
    for call in golden_calls:
        _, emitted = _stream_call(call)
        all_leaks.extend(find_leaks(call["call_id"], " ".join(emitted), _all_pii(call)))

    assert not all_leaks, "PII leaked past the redactor:\n" + "\n".join(
        str(leak) for leak in all_leaks[:20]
    )


def test_vault_holds_the_originals(golden_calls):
    """Sanity check on the oracle itself.

    If the vault were empty, every leak test above would pass trivially. This
    asserts the redactor really did capture secrets, so a passing suite means
    "redacted", not "nothing was ever detected".
    """
    total = 0
    for call in golden_calls:
        redactor, _ = _stream_call(call)
        total += len(redactor.vault.originals())
    assert total > 100, f"expected the corpus to yield many vault entries, got {total}"


# ---------------------------------------------------------------------------
# 2. Transport wire bytes
# ---------------------------------------------------------------------------


def test_transport_wire_bytes_contain_no_pii(golden_calls):
    """Assert on the encoded bytes, not on the objects."""
    all_leaks = []
    for call in golden_calls:
        redactor = Redactor(call["call_id"], RedactorConfig(allowlist=(call.get("agent_name", ""),)))
        transport = CapturingTransport()
        seq = 0
        for turn in call["turns"]:
            out = redactor.push(turn["text"], is_final=True)
            if not out.has_output:
                continue
            transport.send(
                TranscriptSegment(
                    call_id=call["call_id"], seq=seq, speaker=turn["speaker"],
                    text=out.text, is_final=True,
                    start_ms=turn["idx"] * 1000, end_ms=turn["idx"] * 1000 + 900,
                    emitted_at="2026-08-16T00:00:00.000000Z",
                    redactions=list(out.redactions), asr_confidence=1.0,
                )
            )
            seq += 1
        wire = transport.wire_bytes.decode("utf-8")
        all_leaks.extend(find_leaks(call["call_id"], wire, _all_pii(call)))

    assert not all_leaks, "PII leaked onto the wire:\n" + "\n".join(
        str(leak) for leak in all_leaks[:20]
    )


def test_vault_contents_never_serialised(golden_calls):
    """Nothing the vault holds may appear in any encoded segment."""
    for call in golden_calls[:12]:
        redactor, emitted = _stream_call(call)
        blob = " ".join(emitted)
        for original in redactor.vault.originals():
            digits = "".join(c for c in original if c.isdigit())
            if len(digits) >= 5:
                assert digits not in "".join(c for c in blob if c.isdigit()), (
                    f"{call['call_id']}: vault value leaked into output"
                )
            elif len(original) >= 5:
                assert original not in blob, (
                    f"{call['call_id']}: vault value {original!r} leaked into output"
                )


def test_vault_repr_does_not_render_secrets():
    """A traceback or debug log must not print the vault's contents."""
    redactor = Redactor("c1")
    redactor.redact("my card is 4242 4242 4242 4242")
    assert "4242" not in repr(redactor.vault)
    assert "4242" not in str(redactor.vault)
    assert "4242" not in json.dumps(redactor.vault.summary())


# ---------------------------------------------------------------------------
# 3. A real WebSocket server
# ---------------------------------------------------------------------------


class _CaptureServer:
    """A real WebSocket server that records every byte it receives."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.port: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Future | None = None

    def __enter__(self) -> _CaptureServer:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise TimeoutError("capture server did not start")
        return self

    def __exit__(self, *_exc) -> None:
        if self._loop and self._stop and not self._stop.done():
            self._loop.call_soon_threadsafe(self._stop.set_result, None)
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        import websockets

        async def handler(ws):
            async for message in ws:
                self.received.append(
                    message if isinstance(message, str) else message.decode("utf-8")
                )

        async def serve():
            self._loop = asyncio.get_running_loop()
            self._stop = self._loop.create_future()
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._ready.set()
                await self._stop

        asyncio.run(serve())


@pytest.mark.timeout(120)
def test_real_websocket_server_receives_no_pii(golden_calls):
    """End to end over a real socket: nothing mocked, nothing stubbed."""
    from edge_agent.transport import WebSocketTransport

    calls = [c for c in golden_calls if c["category"] in ("pii_heavy", "adversarial")][:6]
    assert calls, "expected PII-bearing calls in the corpus"

    with _CaptureServer() as server:
        for call in calls:
            redactor = Redactor(
                call["call_id"], RedactorConfig(allowlist=(call.get("agent_name", ""),))
            )
            transport = WebSocketTransport(
                f"ws://127.0.0.1:{server.port}/v1/stream", call_id=call["call_id"]
            )
            transport.wait_ready(timeout=10)
            seq = 0
            for turn in call["turns"]:
                out = redactor.push(turn["text"], is_final=True)
                if not out.has_output:
                    continue
                transport.send(
                    TranscriptSegment(
                        call_id=call["call_id"], seq=seq, speaker=turn["speaker"],
                        text=out.text, is_final=True,
                        start_ms=turn["idx"] * 1000, end_ms=turn["idx"] * 1000 + 900,
                        emitted_at="2026-08-16T00:00:00.000000Z",
                        redactions=list(out.redactions), asr_confidence=1.0,
                    )
                )
                seq += 1
            transport.close()

        # Give the server loop a moment to drain the last frames.
        for _ in range(100):
            if server.received:
                break
            threading.Event().wait(0.05)

    assert server.received, "capture server received nothing; the test proved nothing"
    wire = "\n".join(server.received)

    leaks = []
    for call in calls:
        leaks.extend(find_leaks(call["call_id"], wire, _all_pii(call)))
    assert not leaks, "PII crossed a real socket:\n" + "\n".join(str(x) for x in leaks[:20])


# ---------------------------------------------------------------------------
# 4. Shape invariant
# ---------------------------------------------------------------------------


def test_no_long_digit_run_survives(golden_calls):
    """Independent of the corpus labels: no 7+ digit run may be emitted.

    This is the check that still works when the ASR hears a number the corpus
    never contained.
    """
    offenders = []
    for call in golden_calls:
        _, emitted = _stream_call(call)
        for text in emitted:
            for run in long_digit_runs(text, minimum=7):
                offenders.append(f"{call['call_id']}: {run!r} in {text[:90]!r}")
    assert not offenders, "long digit runs survived redaction:\n" + "\n".join(offenders[:20])


def test_split_across_segments_is_not_leaked():
    """The case that motivates the hold buffer.

    Neither segment contains a whole card. A per-segment redactor emits the
    first half verbatim; this asserts we do not.
    """
    redactor = Redactor("split-test")
    first = redactor.push("Okay, the card number, first part is 4000 0566", is_final=True)
    second = redactor.push("and the rest is 5566 5556.", is_final=True)
    combined = f"{first.text} {second.text}"

    assert "4000" not in combined, f"leaked the first half: {combined!r}"
    assert "0566" not in combined
    assert "<CARD_1>" in combined, f"card was never reassembled: {combined!r}"
    assert redactor.vault.resolve("<CARD_1>") is not None


# ---------------------------------------------------------------------------
# 5. The contract itself
# ---------------------------------------------------------------------------


def test_contract_cannot_carry_raw_pii():
    """You cannot add an ``original`` field to a payload; validation refuses."""
    from pydantic import ValidationError

    base = {
        "call_id": "c1", "seq": 0, "speaker": "customer", "text": "hello",
        "is_final": True, "start_ms": 0, "end_ms": 10,
        "emitted_at": "2026-08-16T00:00:00.000000Z",
    }
    TranscriptSegment.model_validate(base)  # baseline is valid

    with pytest.raises(ValidationError):
        TranscriptSegment.model_validate({**base, "raw_text": "4242424242424242"})

    with pytest.raises(ValidationError):
        TranscriptSegment.model_validate({
            **base,
            "redactions": [{
                "type": "CARD", "placeholder": "<CARD_1>", "start": 0, "end": 8,
                "detector": "regex", "original": "4242424242424242",
            }],
        })


def test_encode_rejects_text_mutated_after_redaction():
    """A caller who edits ``text`` post-redaction breaks the span check."""
    from pydantic import ValidationError

    segment = TranscriptSegment(
        call_id="c1", seq=0, speaker="customer", text="card <CARD_1> ok",
        is_final=True, start_ms=0, end_ms=10,
        emitted_at="2026-08-16T00:00:00.000000Z",
        redactions=[{
            "type": "CARD", "placeholder": "<CARD_1>", "start": 5, "end": 13,
            "detector": "regex+checksum", "confidence": 0.99,
        }],
    )
    encode(segment)  # valid as built

    segment.text = "card 4242424242424242 ok"  # someone "restores" the value
    with pytest.raises(ValidationError):
        encode(segment)


# ---------------------------------------------------------------------------
# 6. Logs
# ---------------------------------------------------------------------------


def test_logs_scrub_pii():
    """Logging is an egress path too."""
    assert "4242424242424242" not in scrub("card was 4242424242424242")
    assert "4242 4242 4242 4242" not in scrub("card was 4242 4242 4242 4242")
    assert "900-45-6789" not in scrub("ssn 900-45-6789")
    assert "j.calloway@example.com" not in scrub("email j.calloway@example.com")
    # Short, non-sensitive numbers stay readable so logs remain useful.
    assert "seq=42" in scrub("seq=42")
