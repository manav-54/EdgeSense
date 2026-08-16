"""The tools the agent can call.

The design choice worth defending: ``flag_risk`` is a *tool*, not a parsed
field of a final JSON blob. The agent raises a finding by calling it, and the
tool refuses the call when the evidence does not resolve to a real transcript
turn.

That inverts the usual failure mode. With one-shot prompting the model returns
a summary containing an assertion, and the code downstream has to decide
whether to believe it -- by which point the claim already exists and the
pressure is to publish it. Here an unsourced claim never becomes a Signal at
all: it becomes a tool error the model sees and can correct on the next turn.

``lookup_policy`` and ``search_transcript`` exist for the same reason. The
agent has to go and get the policy text before citing a violation, and has to
find the turn before quoting it, rather than recalling either from the prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from edgesense_core.contracts import EvidenceSpan, Severity, SignalType

from worker.llm.base import ToolSpec
from worker.obs import get_logger
from worker.policies import PolicyStore
from worker.state import CallState

log = get_logger(__name__)


@dataclass
class FlaggedRisk:
    """A finding raised by the agent, already checked against the transcript."""

    type: SignalType
    label: str
    severity: Severity
    confidence: float
    rationale: str
    evidence: list[EvidenceSpan]
    policy_id: str | None = None


@dataclass
class ToolContext:
    """What the tools are allowed to see for one analysis pass."""

    state: CallState
    policies: PolicyStore
    flagged: list[FlaggedRisk] = field(default_factory=list)
    calls_made: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ToolError(Exception):
    """Returned to the model as a tool result, not raised to the caller.

    A tool failure is information the agent can act on -- usually by searching
    for real evidence and trying again. Raising would abandon the turn.
    """


# ---------------------------------------------------------------------------
# Specs advertised to the model
# ---------------------------------------------------------------------------

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="lookup_policy",
        description=(
            "Fetch the full text of a compliance policy by id (e.g. 'REC-001'), "
            "or search for policies by description when the id is unknown. "
            "Call this before asserting any compliance violation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "Policy identifier such as REC-001, PCI-002.",
                },
                "query": {
                    "type": "string",
                    "description": "Free-text description, used when policy_id is unknown.",
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="search_transcript",
        description=(
            "Search this call's transcript for turns matching a query. Returns "
            "turn index, speaker, timestamps and text. Use it to locate the exact "
            "turn that supports a finding before flagging it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words to search for."},
                "top": {"type": "integer", "description": "Max results (default 5).",
                        "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="flag_risk",
        description=(
            "Raise a signal about this call. Every flag must cite at least one "
            "transcript turn index that justifies it; a flag whose evidence does "
            "not match the transcript is rejected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [t.value for t in SignalType],
                    "description": "Kind of signal being raised.",
                },
                "label": {
                    "type": "string",
                    "description": "Intent name, policy id, or shift descriptor.",
                },
                "severity": {
                    "type": "string",
                    "enum": [s.value for s in Severity],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {
                    "type": "string",
                    "description": "One or two sentences on why this fires.",
                },
                "evidence_turns": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Transcript turn indices supporting the finding.",
                    "minItems": 1,
                },
                "policy_id": {
                    "type": "string",
                    "description": "Required when type is compliance_violation.",
                },
            },
            "required": ["type", "label", "confidence", "evidence_turns"],
            "additionalProperties": False,
        },
    ),
]


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def lookup_policy(ctx: ToolContext, args: dict[str, Any]) -> dict:
    policy_id = (args.get("policy_id") or "").strip()
    query = (args.get("query") or "").strip()

    if policy_id:
        policy = ctx.policies.get(policy_id)
        if policy is None:
            known = ", ".join(p.id for p in ctx.policies.all())
            raise ToolError(f"no policy with id {policy_id!r}. Known ids: {known}")
        return {"policy": policy.as_dict()}

    if query:
        hits = ctx.policies.search(query, top=3)
        if not hits:
            return {"policies": [], "note": f"no policy matched {query!r}"}
        return {"policies": [p.as_dict() for p in hits]}

    raise ToolError("lookup_policy requires either policy_id or query")


def search_transcript(ctx: ToolContext, args: dict[str, Any]) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("search_transcript requires a non-empty query")
    top = int(args.get("top") or 5)
    results = ctx.state.search(query, top=max(1, min(top, 20)))
    if not results:
        return {
            "results": [],
            "note": (
                f"nothing in the transcript matched {query!r}. "
                "Do not flag a finding you cannot evidence."
            ),
        }
    return {"results": results}


def flag_risk(ctx: ToolContext, args: dict[str, Any]) -> dict:
    """Record a finding, after checking its evidence resolves."""
    try:
        signal_type = SignalType(args["type"])
    except (KeyError, ValueError):
        raise ToolError(
            f"type must be one of {[t.value for t in SignalType]}, got {args.get('type')!r}"
        ) from None

    label = (args.get("label") or "").strip()
    if not label:
        raise ToolError("label is required")

    turn_indices = args.get("evidence_turns") or []
    if not isinstance(turn_indices, list) or not turn_indices:
        raise ToolError(
            "evidence_turns must be a non-empty list of transcript turn indices. "
            "Use search_transcript to find them."
        )

    spans: list[EvidenceSpan] = []
    bad: list[int] = []
    for raw_idx in turn_indices[:8]:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            bad.append(raw_idx)
            continue
        span = ctx.state.evidence_for(idx)
        if span is None:
            bad.append(idx)
            continue
        spans.append(span)

    if not spans:
        raise ToolError(
            f"none of the evidence turns {turn_indices} exist in this transcript "
            f"(valid range 0..{len(ctx.state.turns) - 1}). "
            "Call search_transcript to find the real turn index."
        )
    if bad:
        # Partial credit: keep the valid spans, tell the model what was wrong.
        log.debug("dropped unresolvable evidence turns", call_id=ctx.state.call_id,
                  bad=str(bad))

    if signal_type is SignalType.COMPLIANCE_VIOLATION:
        policy_id = (args.get("policy_id") or "").strip()
        if not policy_id:
            raise ToolError(
                "compliance_violation requires policy_id. "
                "Call lookup_policy first to confirm which policy applies."
            )
        if ctx.policies.get(policy_id) is None:
            raise ToolError(f"policy_id {policy_id!r} is not in the catalog")
    else:
        policy_id = (args.get("policy_id") or "").strip() or None

    try:
        confidence = float(args.get("confidence", 0.5))
    except (TypeError, ValueError):
        raise ToolError("confidence must be a number in [0,1]") from None
    confidence = max(0.0, min(1.0, confidence))

    severity_raw = (args.get("severity") or "").strip().lower()
    try:
        severity = Severity(severity_raw) if severity_raw else Severity.INFO
    except ValueError:
        severity = Severity.INFO

    risk = FlaggedRisk(
        type=signal_type,
        label=label[:128],
        severity=severity,
        confidence=confidence,
        rationale=(args.get("rationale") or "")[:1000],
        evidence=spans,
        policy_id=policy_id,
    )
    ctx.flagged.append(risk)
    return {
        "flagged": True,
        "type": signal_type.value,
        "label": risk.label,
        "evidence_turns": [s.seq for s in spans],
        "dropped_turns": bad,
    }


HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], dict]] = {
    "lookup_policy": lookup_policy,
    "search_transcript": search_transcript,
    "flag_risk": flag_risk,
}


def invoke(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Run a tool and return the JSON string the model will see."""
    ctx.calls_made.append(name)
    handler = HANDLERS.get(name)
    if handler is None:
        ctx.errors.append(f"unknown tool {name}")
        return json.dumps({"error": f"unknown tool {name!r}",
                           "available": sorted(HANDLERS)})
    try:
        return json.dumps(handler(ctx, args), default=str)
    except ToolError as exc:
        ctx.errors.append(f"{name}: {exc}")
        return json.dumps({"error": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("tool raised unexpectedly", tool=name)
        ctx.errors.append(f"{name}: {exc}")
        return json.dumps({"error": f"tool {name} failed: {exc}"})
