"""The redaction engine: detect, substitute, and -- critically -- decide what
is not safe to emit yet.

Single-shot redaction is the easy half. The hard half is streaming, where the
ASR hands over "the card number, first part is 4000 0566" and the rest arrives
a second later. Redacting each segment independently leaks eight digits of a
live card, and no amount of downstream care undoes that: it is already on the
wire.

So the engine holds back a trailing digit run that could still be growing.
Emission of that fragment is deferred until either the next segment resolves
it (the two halves join, Luhn passes, one ``<CARD_1>`` is emitted) or the hold
expires, at which point the fragment is redacted rather than released. The
cost is bounded: at most one segment of extra latency on the affected segment
only, and the latency budget in EVAL.md is measured with the hold enabled.

The alternative designs, and why they lost:

* *Emit everything, retract later.* Retraction over a network is a fiction --
  the bytes have already left, and the receiver may have logged them.
* *Buffer the whole call, redact at the end.* Correct, and useless: there are
  no live signals if nothing is emitted until hangup.
* *Overlapping windows without a hold.* Catches the join but emits the first
  half in the meantime. This is the design that looks right and is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from edgesense_core.contracts import PIIType, RedactionRef
from edgesense_core.timeutil import monotonic_ms

from edge_agent.redact.detectors import (
    Detection,
    _identifier_nearby,
    build_digit_stream,
    detect_digit_stream,
    detect_emails,
    detect_formatted,
    resolve,
)
from edge_agent.redact.ner import NERBackend, load_ner
from edge_agent.redact.vault import PIIVault

#: Longest PII digit sequence we would ever expect (19-digit PAN).
MAX_PII_DIGITS = 19

#: A trailing run shorter than this is not worth holding -- "press 1" should
#: not add latency to every segment that ends in a digit.
MIN_HOLD_DIGITS = 3

#: How much previously-seen raw text is kept purely to give detectors context.
#: Never emitted, never transmitted.
CONTEXT_TAIL_CHARS = 240

SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of redacting one piece of text."""

    text: str
    redactions: tuple[RedactionRef, ...]
    detections: tuple[Detection, ...]
    latency_ms: float

    @property
    def redacted_count(self) -> int:
        return len(self.redactions)


@dataclass(frozen=True)
class StreamResult:
    """Outcome of pushing one streaming segment.

    ``text`` may be empty when the entire segment was held back; the caller
    should emit nothing rather than an empty segment in that case.
    """

    text: str
    redactions: tuple[RedactionRef, ...]
    held_digits: int
    latency_ms: float

    @property
    def has_output(self) -> bool:
        return bool(self.text.strip())


@dataclass
class RedactorConfig:
    #: Names the edge knows belong to the agent on this call. Redacting the
    #: agent's own first name is pure precision loss -- the client already
    #: knows who the agent is, so it is excluded rather than detected.
    allowlist: tuple[str, ...] = ()
    #: Types to redact. Narrowing this is a policy decision, not a tuning knob.
    enabled_types: frozenset[PIIType] = frozenset(PIIType)
    #: Detections below this confidence are still redacted by default (0.0).
    #: Raising it trades recall for precision and is measured in EVAL.md.
    min_confidence: float = 0.0
    #: Hold trailing digit runs across segments.
    hold_enabled: bool = True
    #: Give up on a hold after this many segments and redact the fragment.
    max_hold_segments: int = 4
    ner_backend: str = "auto"


class Redactor:
    """Stateless single-shot redaction plus a stateful streaming mode."""

    def __init__(
        self,
        call_id: str,
        config: RedactorConfig | None = None,
        ner: NERBackend | None = None,
        vault: PIIVault | None = None,
    ) -> None:
        self.call_id = call_id
        self.config = config or RedactorConfig()
        self.vault = vault if vault is not None else PIIVault(call_id=call_id)
        self._ner = ner if ner is not None else load_ner(self.config.ner_backend)
        self._carry: str = ""
        self._carry_age: int = 0
        self._context_tail: str = ""

    # -- detection ---------------------------------------------------------

    def detect(self, text: str, extra_context: str = "") -> list[Detection]:
        """All PII detections in ``text``, overlaps resolved."""
        found: list[Detection] = []
        found.extend(detect_emails(text))
        found.extend(detect_formatted(text, extra_context))
        found.extend(detect_digit_stream(text, None, extra_context))
        found.extend(self._ner.entities(text))

        cfg = self.config
        kept = [
            d for d in found
            if d.type in cfg.enabled_types
            and d.confidence >= cfg.min_confidence
            and not self._allowlisted(text, d)
        ]
        return resolve(kept)

    def _allowlisted(self, text: str, det: Detection) -> bool:
        if det.type not in (PIIType.PERSON, PIIType.ADDRESS):
            return False
        surface = text[det.start : det.end].strip().casefold()
        return any(
            surface == a.casefold() or surface in a.casefold().split()
            for a in self.config.allowlist
        )

    # -- substitution ------------------------------------------------------

    def redact(
        self,
        text: str,
        extra_context: str = "",
        detections: list[Detection] | None = None,
    ) -> RedactionResult:
        """Replace every detection with a typed placeholder.

        ``detections`` may be supplied by a caller that has already run
        detection over a wider context than ``text`` -- the streaming path does
        this so a cross-segment finding is not recomputed (and lost) against
        the narrower emitted slice.
        """
        t0 = monotonic_ms()
        if detections is None:
            detections = self.detect(text, extra_context)

        parts: list[str] = []
        refs: list[RedactionRef] = []
        cursor = 0
        for det in detections:
            parts.append(text[cursor : det.start])
            original = text[det.start : det.end]
            placeholder = self.vault.placeholder_for(det.type, original)
            start = sum(len(p) for p in parts)
            parts.append(placeholder)
            refs.append(
                RedactionRef(
                    type=det.type,
                    placeholder=placeholder,
                    start=start,
                    end=start + len(placeholder),
                    detector=det.detector,  # type: ignore[arg-type]
                    confidence=det.confidence,
                )
            )
            cursor = det.end
        parts.append(text[cursor:])

        return RedactionResult(
            text="".join(parts),
            redactions=tuple(refs),
            detections=tuple(detections),
            latency_ms=monotonic_ms() - t0,
        )

    # -- streaming ---------------------------------------------------------

    def _hold_point(
        self, text: str, detections: list[Detection], context: str = ""
    ) -> int | None:
        """Index from which the tail of ``text`` must be withheld, if any.

        Returns ``None`` when the whole segment is safe to emit.
        """
        if not self.config.hold_enabled or not text:
            return None

        stream = build_digit_stream(text)
        if not len(stream):
            return None

        last = stream.digits[-1]
        trailing = text[last.end :]

        # Walk back to the start of the trailing contiguous run.
        run_start_idx = len(stream) - 1
        while run_start_idx > 0 and (run_start_idx not in stream.breaks):
            run_start_idx -= 1
        run_len = len(stream) - run_start_idx
        char_start = stream.digits[run_start_idx].start

        # Already fully claimed by a confident detection that ends at the run's
        # end? Then it is a complete value, not a fragment -- emit it redacted.
        claimed = any(
            det.start <= char_start and det.end >= last.end and det.confidence >= 0.9
            for det in detections
        )
        if claimed:
            return None

        if run_len < MIN_HOLD_DIGITS or run_len >= MAX_PII_DIGITS:
            return None

        # Punctuation is NOT proof the number ended.
        #
        # This looked safe and was not. faster-whisper punctuates aggressively:
        # a caller reading "4242 4242 4242 4242" comes back as four separate
        # segments, each rendered "4242." with a full stop. Treating that stop
        # as "the speaker finished" released all four groups in the clear, and
        # a card read aloud in chunks was reconstructible from consecutive
        # segments. The audio-mode eval caught it; the text-mode eval never
        # could, because authored transcripts do not punctuate mid-number.
        #
        # So a terminator only releases the hold when nothing nearby suggests
        # an identifier is being read out. With identifier context in scope, a
        # short unclaimed run keeps waiting -- and if it never resolves, the
        # expiry path redacts it rather than letting it go.
        identifier_context = _identifier_nearby(
            text, char_start, last.end, context
        )
        if SENTENCE_END.search(trailing) and not identifier_context:
            return None

        # Enough words followed that this is no longer an open readback.
        if len(trailing.split()) > 4 and not identifier_context:
            return None
        if len(trailing.split()) > 12:
            return None

        return char_start

    def _protect_orphaned_carry(
        self, combined: str, carry_len: int, detections: list[Detection]
    ) -> list[Detection]:
        """Redact a held fragment that the next segment did not complete.

        The hold assumes the number continues in the following segment. Often
        it does not: the agent says "Go ahead." in between, and the fragment is
        now stranded in the middle of the combined text rather than at its end.
        The hold releases, the fragment is too short to trip any detector, and
        eight digits of a card go out in the clear.

        A fragment was withheld precisely because it looked like part of a
        secret. If nothing has since claimed it, that suspicion stands, so it
        is redacted as an untyped ``ACCOUNT`` rather than released. Choosing
        the wrong label is survivable; releasing the digits is not.
        """
        carry_region = combined[:carry_len]
        if any(d.start < carry_len and d.end > 0 for d in detections):
            return detections  # something claimed it; nothing to do

        stream = build_digit_stream(carry_region)
        if len(stream) < MIN_HOLD_DIGITS:
            return detections

        start, end = stream.span(0, len(stream))
        return detections + [
            Detection(
                type=PIIType.ACCOUNT,
                start=start,
                end=end,
                detector="cross_segment",
                confidence=0.5,
                canonical=stream.text,
            )
        ]

    def push(self, text: str, *, is_final: bool = True) -> StreamResult:
        """Feed one ASR segment. Returns what is safe to emit right now.

        Partial (non-final) segments are redacted and previewed but never
        mutate carry state, because a partial is speculative -- the ASR may
        revise it. Treating a partial as committed would let a revised value
        escape the hold.
        """
        t0 = monotonic_ms()
        combined = f"{self._carry} {text}".strip() if self._carry else text
        extra = self._context_tail

        detections = self.detect(combined, extra)
        carry_len = len(self._carry)
        if carry_len:
            detections = resolve(
                self._protect_orphaned_carry(combined, carry_len, detections)
            )
        hold_at = self._hold_point(combined, detections, extra)

        if hold_at is None:
            emit_src, held_src = combined, ""
        else:
            emit_src, held_src = combined[:hold_at], combined[hold_at:]

        # Redact only the part being emitted, so a held fragment never gets a
        # placeholder minted for a value we have not finished seeing. The
        # detections computed over `combined` are reused rather than recomputed,
        # so a cross-segment finding survives the narrowing.
        emit_dets = [d for d in detections if d.end <= len(emit_src)]
        result = (
            self.redact(emit_src, extra, emit_dets)
            if emit_src.strip()
            else RedactionResult("", (), (), 0.0)
        )
        held_digits = sum(ch.isdigit() for ch in held_src) if held_src else 0

        if is_final:
            self._carry = held_src.strip()
            self._carry_age = self._carry_age + 1 if held_src.strip() else 0
            self._context_tail = f"{self._context_tail} {text}"[-CONTEXT_TAIL_CHARS:]

            # A hold that never resolves must not become a leak. Flush it
            # redacted rather than releasing it.
            if self._carry and self._carry_age >= self.config.max_hold_segments:
                forced = self._redact_carry(self._carry, extra)
                self._carry, self._carry_age = "", 0
                joined = f"{result.text} {forced.text}".strip()
                shift = len(result.text) + 1 if result.text else 0
                refs = list(result.redactions) + [
                    r.model_copy(update={"start": r.start + shift, "end": r.end + shift})
                    for r in forced.redactions
                ]
                return StreamResult(
                    text=joined,
                    redactions=tuple(refs),
                    held_digits=0,
                    latency_ms=monotonic_ms() - t0,
                )

        return StreamResult(
            text=result.text.strip(),
            redactions=result.redactions,
            held_digits=held_digits,
            latency_ms=monotonic_ms() - t0,
        )

    def _redact_carry(self, carry: str, extra: str) -> RedactionResult:
        """Redact a fragment that is being released because the hold expired.

        The fragment was withheld precisely because it looked like part of a
        secret, so releasing it raw when no detector happens to claim it turns
        the hold into a delay rather than a protection. A four-digit tail of a
        chunked card readback lands exactly here: too short for any length
        rule, but the last thing that should go out in the clear.

        So anything the detectors do not claim is redacted wholesale as an
        untyped ACCOUNT before release.
        """
        detections = self.detect(carry, extra)
        detections = resolve(self._protect_orphaned_carry(carry, len(carry), detections))
        return self.redact(carry, extra, detections)

    def flush(self) -> StreamResult:
        """Emit whatever is still held. Called at end of call."""
        t0 = monotonic_ms()
        if not self._carry:
            return StreamResult("", (), 0, monotonic_ms() - t0)
        result = self._redact_carry(self._carry, self._context_tail)
        self._carry, self._carry_age = "", 0
        return StreamResult(
            text=result.text.strip(),
            redactions=result.redactions,
            held_digits=0,
            latency_ms=monotonic_ms() - t0,
        )

    @property
    def pending_chars(self) -> int:
        return len(self._carry)
