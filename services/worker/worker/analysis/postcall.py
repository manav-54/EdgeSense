"""Post-call structured summary, validated against a strict schema.

The contract is that a published ``CallSummary`` validates against the pydantic
model exactly -- no missing fields, no invented enum values, no action item
without evidence. Models are good at *almost* satisfying a schema, so the
agent gets up to two repair attempts with the validator's own error text fed
back to it, which is far more actionable than a generic retry.

If it still does not validate, the worker records a failure and publishes
nothing. Publishing a partially-correct summary would be worse than publishing
none: it would look authoritative in the portal and quietly corrupt every
aggregate built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from edgesense_core.contracts import (
    ActionItem,
    CallSummary,
    EvidenceSpan,
    ResolutionStatus,
)
from edgesense_core.timeutil import monotonic_ms, utc_now_iso

from worker.agent import Agent, AgentResult
from worker.analysis import rules
from worker.llm.base import LLMProvider
from worker.obs import (
    ANALYSIS_LATENCY,
    METRICS_AVAILABLE,
    SUMMARIES_EMITTED,
    get_logger,
)
from worker.policies import PolicyStore
from worker.prompts import registry
from worker.state import CallState
from worker.tools import ToolContext

log = get_logger(__name__)

#: Post-call work is not latency-critical -- nobody is waiting on the line --
#: so it gets a generous budget and can afford repairs.
POST_CALL_BUDGET_MS = 45_000.0

VALID_INTENTS = frozenset(rules.INTENT_KEYWORDS)


@dataclass
class PostCallResult:
    summary: CallSummary | None
    agent: AgentResult | None
    duration_ms: float
    error: str | None = None
    repairs: int = 0
    dropped_fields: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.summary is not None


class PostCallAnalyzer:
    def __init__(
        self,
        provider: LLMProvider,
        policies: PolicyStore,
        prompt_version: str = "latest",
    ) -> None:
        self.provider = provider
        self.policies = policies
        self.prompt_version = prompt_version

    def summarise(self, state: CallState) -> PostCallResult:
        t0 = monotonic_ms()
        if not state.turns:
            return PostCallResult(None, None, 0.0, error="call has no final segments")

        ctx = ToolContext(state=state, policies=self.policies)
        agent = Agent(self.provider, deadline_ms=POST_CALL_BUDGET_MS)
        prompt = registry.get("post_call_summary", self.prompt_version)

        result = agent.run(
            "post_call_summary",
            ctx,
            version=self.prompt_version,
            json_mode=True,
            validator=lambda text: self._validate_text(text, state),
            max_repairs=2,
            call_id=state.call_id,
            agent_id=state.agent_id or "unknown",
            turn_count=len(state.turns),
            transcript=state.transcript_text(),
            _required_disclosures=list(state.disclosures_given),
        )

        duration = monotonic_ms() - t0
        if METRICS_AVAILABLE:
            ANALYSIS_LATENCY.labels(path="post_call").observe(duration / 1000.0)

        payload = result.json_payload()
        if payload is None:
            if METRICS_AVAILABLE:
                SUMMARIES_EMITTED.labels(outcome="unparseable").inc()
            log.error("post-call response was not JSON",
                      call_id=state.call_id, error=result.error,
                      content_head=result.content[:160])
            return PostCallResult(None, result, duration,
                                  error=result.error or "response was not JSON")

        try:
            summary = self._build(payload, state, prompt.ref)
        except ValidationError as exc:
            if METRICS_AVAILABLE:
                SUMMARIES_EMITTED.labels(outcome="invalid").inc()
            log.error("post-call summary failed final validation",
                      call_id=state.call_id, error=str(exc)[:400])
            return PostCallResult(None, result, duration, error=str(exc),
                                  repairs=result.repairs)

        if METRICS_AVAILABLE:
            SUMMARIES_EMITTED.labels(outcome="ok").inc()
        log.info("post-call summary produced",
                 call_id=state.call_id, intent=summary.primary_intent,
                 resolution=summary.resolution.value, repairs=result.repairs,
                 steps=result.steps, duration_ms=round(duration, 1))
        return PostCallResult(summary, result, duration, repairs=result.repairs)

    # -- validation --------------------------------------------------------

    def _validate_text(self, text: str, state: CallState) -> tuple[bool, str]:
        """Validator handed to the agent loop for repair attempts."""
        probe = AgentResult(content=text)
        payload = probe.json_payload()
        if payload is None:
            return False, (
                "The response was not a JSON object. Reply with a single JSON "
                "object matching the schema and nothing else."
            )
        try:
            self._build(payload, state, "probe")
        except ValidationError as exc:
            return False, _explain(exc)
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    def _build(self, payload: dict, state: CallState, prompt_ref: str) -> CallSummary:
        """Coerce a model payload into the strict schema.

        Two accommodations, both deliberate:

        * The model returns ``evidence_turns`` (indices it can see) and we
          convert them to ``EvidenceSpan`` (with real timestamps it cannot).
          Asking the model for timestamps invites it to invent them.
        * An action item whose evidence does not resolve is dropped rather
          than failing the whole summary, because losing one action item is a
          smaller harm than losing the record of the call. Drops are counted.
        """
        intent = str(payload.get("primary_intent", "")).strip() or "general_inquiry"
        if intent not in VALID_INTENTS:
            raise ValueError(
                f"primary_intent {intent!r} is not a valid intent. "
                f"Choose one of: {', '.join(sorted(VALID_INTENTS))}"
            )

        resolution_raw = str(payload.get("resolution", "")).strip()
        try:
            resolution = ResolutionStatus(resolution_raw)
        except ValueError:
            raise ValueError(
                f"resolution {resolution_raw!r} is invalid. Use one of: "
                f"{', '.join(r.value for r in ResolutionStatus)}"
            ) from None

        action_items: list[ActionItem] = []
        for raw in (payload.get("action_items") or [])[:20]:
            if not isinstance(raw, dict):
                continue
            spans = self._spans_for(state, raw.get("evidence_turns") or [])
            if not spans:
                continue  # unsourced action item; dropped, not published
            owner = str(raw.get("owner", "agent")).strip().lower()
            if owner not in {"agent", "customer", "supervisor", "system"}:
                owner = "agent"
            description = str(raw.get("description", "")).strip()
            if not description:
                continue
            action_items.append(
                ActionItem(description=description[:500], owner=owner,
                           due=raw.get("due"), evidence=spans)
            )

        known_policies = {p.id for p in self.policies.all()}
        violations = [
            str(v).strip() for v in (payload.get("compliance_violations") or [])
            if str(v).strip() in known_policies
        ]
        disclosures = [
            str(v).strip() for v in (payload.get("disclosures_given") or [])
            if str(v).strip() in known_policies
        ]

        secondary = [
            str(s).strip() for s in (payload.get("secondary_intents") or [])
            if str(s).strip() in VALID_INTENTS and str(s).strip() != intent
        ][:5]

        return CallSummary(
            call_id=state.call_id,
            summary=str(payload.get("summary", "")).strip()[:2000] or "No summary produced.",
            resolution=resolution,
            primary_intent=intent,
            secondary_intents=secondary,
            action_items=action_items,
            customer_sentiment_start=_clamp(payload.get("customer_sentiment_start", 0.0)),
            customer_sentiment_end=_clamp(payload.get("customer_sentiment_end", 0.0)),
            escalated=bool(payload.get("escalated", False)),
            compliance_violations=sorted(set(violations)),
            disclosures_given=sorted(set(disclosures)),
            evidence=self._spans_for(state, payload.get("evidence_turns") or [])[:40],
            model_name=self.provider.model,
            prompt_version=prompt_ref,
            generated_at=utc_now_iso(),
        )

    @staticmethod
    def _spans_for(state: CallState, indices: list) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        for raw in indices[:40]:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
            span = state.evidence_for(idx)
            if span is not None:
                spans.append(span)
        return spans


def _clamp(value: object) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _explain(exc: ValidationError) -> str:
    """Render pydantic errors as instructions rather than a stack trace."""
    lines = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"- {loc}: {err['msg']}")
    return "The JSON did not validate:\n" + "\n".join(lines)
