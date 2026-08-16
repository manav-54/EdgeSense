"""Per-call transcript state and the sliding window the live path analyses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from edgesense_core.contracts import EvidenceSpan, Speaker, TranscriptSegment
from edgesense_core.timeutil import iso_to_datetime, utc_now


@dataclass
class Turn:
    """A finalised segment, in the shape the rules engine expects."""

    idx: int
    seq: int
    speaker: str
    text: str
    start_ms: int
    end_ms: int
    emitted_at: str
    redactions: int = 0

    def as_dict(self) -> dict:
        return {
            "idx": self.idx, "seq": self.seq, "speaker": self.speaker,
            "text": self.text, "start_ms": self.start_ms, "end_ms": self.end_ms,
        }


@dataclass
class CallState:
    """Everything the worker knows about one in-flight call.

    Only final segments are retained. Partials are previews that a final
    supersedes, and analysing them would produce signals citing text that no
    longer exists -- evidence a reviewer could not find in the transcript.
    """

    call_id: str
    agent_id: str | None = None
    turns: list[Turn] = field(default_factory=list)
    started_at: str = ""
    last_activity: str = ""
    ended: bool = False
    signals_emitted: int = 0
    #: Windows already analysed, so a re-delivered segment does not re-emit.
    analysed_upto: int = -1
    traceparent: str | None = None
    last_sentiment: float | None = None
    disclosures_given: set[str] = field(default_factory=set)
    violations_flagged: set[str] = field(default_factory=set)
    #: Last published signal per type, used to suppress repeats across
    #: windows. A supervisor needs to know that risk *changed*, not that it
    #: is still true -- an escalation badge re-firing every two turns trains
    #: people to ignore the panel.
    last_signal: dict[str, tuple] = field(default_factory=dict)

    def add(self, segment: TranscriptSegment) -> Turn | None:
        if not segment.is_final:
            return None
        turn = Turn(
            idx=len(self.turns),
            seq=segment.seq,
            speaker=segment.speaker.value if isinstance(segment.speaker, Speaker) else str(segment.speaker),
            text=segment.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            emitted_at=segment.emitted_at,
            redactions=len(segment.redactions),
        )
        self.turns.append(turn)
        self.agent_id = self.agent_id or segment.agent_id
        self.traceparent = segment.traceparent or self.traceparent
        if not self.started_at:
            self.started_at = segment.emitted_at
        self.last_activity = segment.emitted_at
        return turn

    # -- windows -----------------------------------------------------------

    def window(self, size: int = 6) -> list[dict]:
        """The last ``size`` turns, as dicts for the rules engine.

        Six turns is roughly three exchanges: long enough for a sentiment
        trend to be visible, short enough that a signal points at something
        the supervisor can still see on screen.
        """
        return [t.as_dict() for t in self.turns[-size:]]

    def all_turns(self) -> list[dict]:
        return [t.as_dict() for t in self.turns]

    def turn_by_idx(self, idx: int) -> Turn | None:
        return self.turns[idx] if 0 <= idx < len(self.turns) else None

    def evidence_for(self, idx: int, quote: str | None = None) -> EvidenceSpan | None:
        """Build an evidence span for a turn, refusing to invent one."""
        turn = self.turn_by_idx(idx)
        if turn is None:
            return None
        text = quote or turn.text
        if not text.strip():
            return None
        return EvidenceSpan(
            seq=turn.seq,
            start_ms=turn.start_ms,
            end_ms=turn.end_ms,
            speaker=Speaker(turn.speaker) if turn.speaker in
            {s.value for s in Speaker} else Speaker.UNKNOWN,
            quote=text[:500],
        )

    def search(self, query: str, top: int = 5) -> list[dict]:
        """Token-overlap search over the transcript.

        This is what ``search_transcript`` calls. It is intentionally simple:
        the agent uses it to locate the turn that supports a claim it is about
        to make, and exact-ish recall over a few dozen short turns does not
        need embeddings.
        """
        terms = {t for t in re.findall(r"[a-z']+", query.lower()) if len(t) > 2}
        if not terms:
            return []
        scored: list[tuple[float, Turn]] = []
        for turn in self.turns:
            tokens = set(re.findall(r"[a-z']+", turn.text.lower()))
            overlap = len(terms & tokens)
            if not overlap:
                continue
            # Favour turns where the matched terms are a large share of the
            # turn, so a short precise turn beats a long rambling one.
            score = overlap / len(terms) + 0.25 * (overlap / max(len(tokens), 1))
            scored.append((score, turn))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [
            {
                "idx": t.idx, "seq": t.seq, "speaker": t.speaker, "text": t.text,
                "start_ms": t.start_ms, "end_ms": t.end_ms, "score": round(s, 3),
            }
            for s, t in scored[:top]
        ]

    # -- lifecycle ---------------------------------------------------------

    @property
    def duration_ms(self) -> int:
        return self.turns[-1].end_ms if self.turns else 0

    def idle_seconds(self) -> float:
        if not self.last_activity:
            return 0.0
        try:
            return (utc_now() - iso_to_datetime(self.last_activity)).total_seconds()
        except Exception:
            return 0.0

    def transcript_text(self, max_turns: int | None = None) -> str:
        turns = self.turns[-max_turns:] if max_turns else self.turns
        return "\n".join(f"[{t.idx}] {t.speaker}: {t.text}" for t in turns)
