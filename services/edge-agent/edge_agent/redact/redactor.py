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
    max_hold_segments: int = 2
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

    def redact(self, text: str, extra_context: str = "") -> RedactionResult:
        """Replace every detection with a typed placeholder."""
        t0 = monotonic_ms()
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

    def _hold_point(self, text: str, detections: list[Detection]) -> int | None:
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
        # A sentence ended after the number: the speaker finished saying it.
        if SENTENCE_END.search(trailing):
            return None
        # Enough words followed that this is no longer an open readback.
        if len(trailing.split()) > 4:
            return None

        # Walk back to the start of the trailing contiguous run.
        run_start_idx = len(stream) - 1
        while run_start_idx > 0 and (run_start_idx not in stream.breaks):
            run_start_idx -= 1
        run_len = len(stream) - run_start_idx

        if run_len < MIN_HOLD_DIGITS or run_len >= MAX_PII_DIGITS:
            return None

        char_start = stream.digits[run_start_idx].start

        # Already fully claimed by a confident detection that ends at the run's
        # end? Then it is a complete value, not a fragment -- emit it redacted.
        for det in detections:
            if det.start <= char_start and det.end >= last.end and det.confidence >= 0.9:
                return None

        return char_start

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
        hold_at = self._hold_point(combined, detections)

        if hold_at is None:
            emit_src, held_src = combined, ""
        else:
            emit_src, held_src = combined[:hold_at], combined[hold_at:]

        # Redact only the part being emitted, so a held fragment never gets a
        # placeholder minted for a value we have not finished seeing.
        result = self.redact(emit_src, extra) if emit_src.strip() else RedactionResult(
            "", (), (), 0.0
        )
        held_digits = sum(ch.isdigit() for ch in held_src) if held_src else 0

        if is_final:
            self._carry = held_src.strip()
            self._carry_age = self._carry_age + 1 if held_src.strip() else 0
            self._context_tail = f"{self._context_tail} {text}"[-CONTEXT_TAIL_CHARS:]

            # A hold that never resolves must not become a leak. Flush it
            # redacted rather than releasing it.
            if self._carry and self._carry_age >= self.config.max_hold_segments:
                forced = self.redact(self._carry, extra)
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

    def flush(self) -> StreamResult:
        """Emit whatever is still held. Called at end of call."""
        t0 = monotonic_ms()
        if not self._carry:
            return StreamResult("", (), 0, monotonic_ms() - t0)
        result = self.redact(self._carry, self._context_tail)
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
