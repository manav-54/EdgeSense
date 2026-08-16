"""Wire contracts for every EdgeSense process boundary.

Design rule that outranks convenience: **no field on any model in this module
may carry raw PII.** ``RedactionRef`` records the *type* and *placeholder* of
something that was removed, plus its offsets in the already-redacted text. It
has no slot for the original value, so a service cannot leak one by filling in
a field -- there is no field to fill. The local-only mapping from placeholder
back to original lives in the edge agent's ``PIIVault`` and is never
serialised into any of these types.

``model_config`` sets ``extra="forbid"`` throughout, so a well-meaning
``segment.model_dump() | {"raw_text": ...}`` fails validation at the receiver
instead of silently propagating.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.0"
"""Bumped on any breaking change to the shapes below.

Consumers reject payloads whose major version they do not understand rather
than best-effort parsing them; see ``require_compatible``.
"""

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Milliseconds = Annotated[int, Field(ge=0)]

_STRICT = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=False)


class Speaker(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"
    UNKNOWN = "unknown"


class PIIType(str, Enum):
    """PII classes the edge redactor recognises.

    Values double as the placeholder stem: ``CARD`` -> ``<CARD_1>``.
    """

    CARD = "CARD"
    SSN = "SSN"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    DOB = "DOB"
    ACCOUNT = "ACCOUNT"
    PERSON = "PERSON"
    ADDRESS = "ADDRESS"


class SignalType(str, Enum):
    SENTIMENT_SHIFT = "sentiment_shift"
    ESCALATION_RISK = "escalation_risk"
    COMPLIANCE_VIOLATION = "compliance_violation"
    INTENT = "intent"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"
    FOLLOW_UP_REQUIRED = "follow_up_required"


# ---------------------------------------------------------------------------
# Edge -> ingest
# ---------------------------------------------------------------------------


class RedactionRef(BaseModel):
    """A pointer to something that *was* removed.

    ``start``/``end`` are character offsets into the **redacted** text and span
    the placeholder token itself, so the portal can highlight it without ever
    seeing the original. There is deliberately no ``original`` field.
    """

    model_config = _STRICT

    type: PIIType
    placeholder: str = Field(
        pattern=r"^<[A-Z_]+_\d+>$",
        description="Typed placeholder substituted into the text, e.g. <CARD_1>.",
    )
    start: int = Field(ge=0, description="Char offset of the placeholder in redacted text.")
    end: int = Field(gt=0, description="Exclusive end offset of the placeholder.")
    detector: Literal["regex", "ner", "regex+checksum", "context", "cross_segment"] = Field(
        description="Which detector fired. Used by the eval harness to attribute misses."
    )
    confidence: Confidence = 1.0

    @model_validator(mode="after")
    def _check_span(self) -> RedactionRef:
        if self.end <= self.start:
            raise ValueError("redaction end must be greater than start")
        if self.end - self.start != len(self.placeholder):
            raise ValueError(
                f"span width {self.end - self.start} does not match placeholder "
                f"{self.placeholder!r} of length {len(self.placeholder)}"
            )
        return self


class TranscriptSegment(BaseModel):
    """One ASR segment, already redacted, on its way to the cloud.

    Partial segments (``is_final=False``) are superseded by a later segment
    with the same ``seq``; the ingest layer dedupes on ``(call_id, seq,
    is_final)`` so a retransmitted final is idempotent.
    """

    model_config = _STRICT

    schema_version: str = CONTRACT_VERSION
    call_id: str = Field(min_length=1, max_length=64)
    seq: int = Field(ge=0, description="Monotonic per call. Gaps mean loss, not reorder.")
    speaker: Speaker
    text: str = Field(description="Redacted transcript text. Never contains raw PII.")
    is_final: bool
    start_ms: Milliseconds
    end_ms: Milliseconds
    emitted_at: str = Field(description="RFC 3339 UTC, set at the moment of send.")
    redactions: list[RedactionRef] = Field(default_factory=list)
    asr_confidence: Confidence = 0.0
    agent_id: str | None = Field(default=None, max_length=64)
    traceparent: str | None = Field(
        default=None,
        description="W3C trace context, so one call_id is traceable end to end.",
    )

    @model_validator(mode="after")
    def _check_times_and_spans(self) -> TranscriptSegment:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        for ref in self.redactions:
            if ref.end > len(self.text):
                raise ValueError(
                    f"redaction {ref.placeholder} ends at {ref.end}, past text length "
                    f"{len(self.text)}"
                )
            found = self.text[ref.start : ref.end]
            if found != ref.placeholder:
                raise ValueError(
                    f"redaction span points at {found!r}, expected {ref.placeholder!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Worker -> insights
# ---------------------------------------------------------------------------


class EvidenceSpan(BaseModel):
    """The transcript span that justifies a claim.

    Every signal carries at least one. A model assertion without a span is
    dropped by the worker rather than published -- see
    ``worker.agent.validate_signal``.
    """

    model_config = _STRICT

    seq: int = Field(ge=0)
    start_ms: Milliseconds
    end_ms: Milliseconds
    speaker: Speaker
    quote: str = Field(
        min_length=1,
        description="Verbatim slice of the redacted transcript. Placeholders stay as-is.",
    )


class StageLatency(BaseModel):
    """Per-stage timings attached to a signal, in milliseconds.

    Populated opportunistically: a stage that did not run leaves its field
    unset rather than reporting zero, so percentiles are not skewed by
    absent stages.
    """

    model_config = _STRICT

    asr_ms: float | None = None
    redact_ms: float | None = None
    ingest_ms: float | None = None
    queue_ms: float | None = None
    analyze_ms: float | None = None
    llm_ms: float | None = None
    segment_to_signal_ms: float | None = Field(
        default=None,
        description="Wall-clock from segment emit to signal publish. The p95 budget.",
    )


class Signal(BaseModel):
    """A real-time observation about a call, always sourced to a transcript span."""

    model_config = _STRICT

    schema_version: str = CONTRACT_VERSION
    signal_id: str = Field(min_length=1, max_length=64)
    call_id: str = Field(min_length=1, max_length=64)
    type: SignalType
    label: str = Field(
        min_length=1,
        max_length=128,
        description="Type-specific value: intent name, policy id, 'negative_shift', ...",
    )
    severity: Severity = Severity.INFO
    confidence: Confidence
    rationale: str = Field(max_length=1000, default="")
    evidence: list[EvidenceSpan] = Field(min_length=1)
    policy_id: str | None = Field(default=None, max_length=64)
    window_start_ms: Milliseconds = 0
    window_end_ms: Milliseconds = 0
    emitted_at: str
    agent_id: str | None = Field(default=None, max_length=64)
    latency: StageLatency = Field(default_factory=StageLatency)
    model_name: str = Field(default="", max_length=128)
    prompt_version: str = Field(default="", max_length=64)
    traceparent: str | None = None

    @model_validator(mode="after")
    def _require_evidence(self) -> Signal:
        if not self.evidence:
            raise ValueError("signals must cite at least one transcript span")
        return self


class ActionItem(BaseModel):
    model_config = _STRICT

    description: str = Field(min_length=1, max_length=500)
    owner: Literal["agent", "customer", "supervisor", "system"]
    due: str | None = Field(default=None, max_length=64)
    evidence: list[EvidenceSpan] = Field(min_length=1)


class CallSummary(BaseModel):
    """Strict post-call structure. The worker retries the model until a payload
    validates against exactly this, then gives up and records a failure rather
    than publishing something malformed."""

    model_config = _STRICT

    schema_version: str = CONTRACT_VERSION
    call_id: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=2000)
    resolution: ResolutionStatus
    primary_intent: str = Field(min_length=1, max_length=64)
    secondary_intents: list[str] = Field(default_factory=list, max_length=5)
    action_items: list[ActionItem] = Field(default_factory=list, max_length=20)
    customer_sentiment_start: float = Field(ge=-1.0, le=1.0)
    customer_sentiment_end: float = Field(ge=-1.0, le=1.0)
    escalated: bool = False
    compliance_violations: list[str] = Field(default_factory=list, max_length=20)
    disclosures_given: list[str] = Field(default_factory=list, max_length=20)
    evidence: list[EvidenceSpan] = Field(default_factory=list, max_length=40)
    model_name: str = Field(default="", max_length=128)
    prompt_version: str = Field(default="", max_length=64)
    generated_at: str = ""


class CallInsights(BaseModel):
    """Envelope published to ``call.insights`` and sunk into ClickHouse.

    A message carries either live signals (``kind='live'``) or the post-call
    bundle (``kind='post_call'``), never both, so the sink can route on one
    field without inspecting the payload.
    """

    model_config = _STRICT

    schema_version: str = CONTRACT_VERSION
    kind: Literal["live", "post_call"]
    call_id: str = Field(min_length=1, max_length=64)
    agent_id: str | None = Field(default=None, max_length=64)
    emitted_at: str
    signals: list[Signal] = Field(default_factory=list)
    summary: CallSummary | None = None
    traceparent: str | None = None

    @model_validator(mode="after")
    def _kind_matches_payload(self) -> CallInsights:
        if self.kind == "post_call" and self.summary is None:
            raise ValueError("post_call insights must carry a summary")
        if self.kind == "live" and not self.signals:
            raise ValueError("live insights must carry at least one signal")
        return self


def require_compatible(payload_version: str, *, expected: str = CONTRACT_VERSION) -> None:
    """Reject payloads from an incompatible major contract version.

    Minor differences are tolerated (additive, non-breaking by convention);
    a major mismatch raises rather than risking a misparse.
    """
    got_major = payload_version.split(".", 1)[0]
    want_major = expected.split(".", 1)[0]
    if got_major != want_major:
        raise ValueError(
            f"incompatible contract version {payload_version!r}; this build speaks {expected!r}"
        )
