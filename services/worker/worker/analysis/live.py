"""Live sliding-window analysis.

Two paths run over every window, and the split is what makes the p95 budget
achievable:

* The **fast path** is pure rules. It runs in well under a millisecond and
  emits compliance and escalation signals immediately. It never calls a
  network service, so it cannot be slow and cannot fail when a provider is
  throttled. On a call where the agent threatens legal action, the supervisor
  sees it now, not after a model round-trip.

* The **agent path** runs the tool loop for the judgement calls -- intent,
  sentiment movement, anything needing context the rules do not encode. It is
  bounded by a wall-clock deadline derived from the remaining budget, and if
  it blows the deadline the window still produced fast-path signals.

Publishing both would double-report, so signals are deduplicated per call on
(type, label, policy_id). The fast path wins ties because it arrived first.

This mirrors how these systems are actually built: deterministic guardrails
for the obligations you must never miss, model judgement for the things rules
cannot express.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from edgesense_core.contracts import (
    Severity,
    Signal,
    SignalType,
    StageLatency,
)
from edgesense_core.timeutil import iso_to_datetime, monotonic_ms, utc_now, utc_now_iso

from worker.agent import Agent
from worker.analysis import rules
from worker.llm.base import LLMProvider
from worker.obs import (
    ANALYSIS_LATENCY,
    METRICS_AVAILABLE,
    SEGMENT_TO_SIGNAL,
    SIGNALS_EMITTED,
    SIGNALS_REJECTED,
    current_traceparent,
    get_logger,
)
from worker.policies import PolicyStore
from worker.prompts import registry
from worker.state import CallState
from worker.tools import FlaggedRisk, ToolContext

log = get_logger(__name__)

#: Disclosures every call must contain, regardless of intent. Intent-scoped
#: ones (mini-Miranda on collections, right-to-cancel on upgrades) are added
#: from the policy catalog's `applies_when`.
BASE_REQUIRED = ("REC-001",)

#: Analyse every N new final turns. Analysing on every turn triples LLM spend
#: for signals that rarely change between adjacent turns; waiting longer means
#: a supervisor learns about an escalation after it has already happened.
ANALYSIS_STRIDE = 2

#: Budget for the whole segment-to-signal path. The agent gets what is left
#: after the fast path, minus headroom for publishing.
LIVE_BUDGET_MS = 2000.0
PUBLISH_HEADROOM_MS = 250.0


@dataclass
class LiveConfig:
    window_size: int = 6
    stride: int = ANALYSIS_STRIDE
    prompt_version: str = "latest"
    use_agent: bool = True
    budget_ms: float = LIVE_BUDGET_MS
    min_confidence: float = 0.35


class LiveAnalyzer:
    def __init__(
        self,
        provider: LLMProvider,
        policies: PolicyStore,
        config: LiveConfig | None = None,
    ) -> None:
        self.provider = provider
        self.policies = policies
        self.config = config or LiveConfig()

    # -- entry point -------------------------------------------------------

    def should_analyse(self, state: CallState) -> bool:
        if not state.turns:
            return False
        newest = len(state.turns) - 1
        return newest - state.analysed_upto >= self.config.stride

    def analyse(self, state: CallState) -> list[Signal]:
        """Analyse the newest window. Returns signals ready to publish."""
        t0 = monotonic_ms()
        window = state.window(self.config.window_size)
        if not window:
            return []

        required = self._required_disclosures(state)
        emitted: list[Signal] = []
        seen: set[tuple] = set()

        # --- fast path ----------------------------------------------------
        fast_t0 = monotonic_ms()
        for risk in self._fast_path(state, window, required):
            signal = self._to_signal(state, risk, path="fast",
                                     prompt_ref="rules@fast-path")
            if signal and self._accept(signal, seen, state):
                emitted.append(signal)
        fast_ms = monotonic_ms() - fast_t0
        if METRICS_AVAILABLE:
            ANALYSIS_LATENCY.labels(path="fast").observe(fast_ms / 1000.0)

        # --- agent path ---------------------------------------------------
        agent_ms = 0.0
        prompt_ref = "rules@fast-path"
        if self.config.use_agent:
            remaining = self.config.budget_ms - fast_ms - PUBLISH_HEADROOM_MS
            if remaining <= 100:
                log.warning("skipping agent path; fast path consumed the budget",
                            call_id=state.call_id, fast_ms=round(fast_ms, 2))
            else:
                agent_t0 = monotonic_ms()
                risks, prompt_ref = self._agent_path(state, window, required, remaining)
                for risk in risks:
                    signal = self._to_signal(state, risk, path="agent",
                                             prompt_ref=prompt_ref)
                    if signal and self._accept(signal, seen, state):
                        emitted.append(signal)
                agent_ms = monotonic_ms() - agent_t0
                if METRICS_AVAILABLE:
                    ANALYSIS_LATENCY.labels(path="agent").observe(agent_ms / 1000.0)

        state.analysed_upto = len(state.turns) - 1
        state.signals_emitted += len(emitted)

        # Attach latency and record the SLO metric.
        newest_emitted_at = state.turns[-1].emitted_at
        for signal in emitted:
            signal.latency.analyze_ms = round(monotonic_ms() - t0, 3)
            signal.latency.llm_ms = round(agent_ms, 3) if agent_ms else None
            e2e = _elapsed_ms_since(newest_emitted_at)
            if e2e is not None:
                signal.latency.segment_to_signal_ms = round(e2e, 3)
                if METRICS_AVAILABLE:
                    SEGMENT_TO_SIGNAL.observe(e2e / 1000.0)

        log.info("live window analysed",
                 call_id=state.call_id, turns=len(state.turns),
                 signals=len(emitted), fast_ms=round(fast_ms, 2),
                 agent_ms=round(agent_ms, 2), prompt=prompt_ref)
        return emitted

    # -- paths -------------------------------------------------------------

    def _fast_path(
        self, state: CallState, window: list[dict], required: list[str]
    ) -> list[FlaggedRisk]:
        """Deterministic findings that must not wait for a model."""
        out: list[FlaggedRisk] = []
        all_turns = state.all_turns()

        for finding in rules.compliance_findings(window, self.policies.raw()):
            span = state.evidence_for(finding.turn_idx, finding.quote)
            if span is None:
                continue
            out.append(FlaggedRisk(
                type=SignalType.COMPLIANCE_VIOLATION,
                label=finding.label,
                severity=_severity(finding.score),
                confidence=round(min(0.97, 0.65 + finding.score * 0.3), 3),
                rationale=finding.detail,
                evidence=[span],
                policy_id=finding.policy_id,
            ))

        # Disclosure checks run over the whole call so far, not the window: a
        # disclosure given in turn 0 must still count in turn 20.
        _, missing = rules.disclosure_status(all_turns, self.policies.raw(), required)
        given, _ = rules.disclosure_status(all_turns, self.policies.raw(), required)
        state.disclosures_given.update(given)
        for finding in missing:
            # Only report a missing disclosure once its window has closed;
            # flagging at turn 1 that a "first three turns" disclosure is
            # absent would fire on every call before the agent has spoken.
            if not self._disclosure_window_closed(finding.policy_id, all_turns):
                continue
            span = state.evidence_for(finding.turn_idx, finding.quote)
            if span is None:
                continue
            out.append(FlaggedRisk(
                type=SignalType.COMPLIANCE_VIOLATION,
                label=finding.label,
                severity=_severity(finding.score),
                confidence=round(min(0.92, 0.6 + finding.score * 0.3), 3),
                rationale=finding.detail,
                evidence=[span],
                policy_id=finding.policy_id,
            ))

        band, score, findings = rules.escalation_risk(window)
        if band in ("medium", "high") and findings:
            spans = [
                s for s in (state.evidence_for(f.turn_idx, f.quote) for f in findings[:3])
                if s is not None
            ]
            if spans:
                top = max(findings, key=lambda f: f.score)
                out.append(FlaggedRisk(
                    type=SignalType.ESCALATION_RISK,
                    label=top.label,
                    severity=Severity.HIGH if band == "high" else Severity.MEDIUM,
                    confidence=round(score, 3),
                    rationale=f"escalation risk {band}: {top.label}",
                    evidence=spans,
                ))
        return out

    def _agent_path(
        self, state: CallState, window: list[dict], required: list[str],
        budget_ms: float,
    ) -> tuple[list[FlaggedRisk], str]:
        ctx = ToolContext(state=state, policies=self.policies)
        agent = Agent(self.provider, deadline_ms=budget_ms)
        prompt = registry.get("live_analysis", self.config.prompt_version)

        result = agent.run(
            "live_analysis",
            ctx,
            version=self.config.prompt_version,
            call_id=state.call_id,
            window_start=window[0]["idx"],
            window_end=window[-1]["idx"],
            transcript=_render(window),
            disclosures_given=", ".join(sorted(state.disclosures_given)) or "none yet",
            already_flagged=", ".join(sorted(state.violations_flagged)) or "none",
            _required_disclosures=required,
        )

        if result.error:
            log.warning("agent path failed; fast-path signals still stand",
                        call_id=state.call_id, error=result.error)
        return result.flagged, prompt.ref

    # -- helpers -----------------------------------------------------------

    def _required_disclosures(self, state: CallState) -> list[str]:
        required = list(BASE_REQUIRED)
        intent, secondary, _ = rules.classify_intent(state.all_turns())
        intents = {intent, *secondary}
        for policy in self.policies.all():
            clause = policy.applies_when or ""
            if not clause:
                continue
            listed = {
                token.strip().strip("[]'\"")
                for token in clause.split("in", 1)[-1].split(",")
            }
            if intents & listed:
                required.append(policy.id)
        return list(dict.fromkeys(required))

    def _disclosure_window_closed(self, policy_id: str | None, turns: list[dict]) -> bool:
        if not policy_id:
            return False
        policy = self.policies.get(policy_id)
        if policy is None:
            return False
        agent_turns = sum(1 for t in turns if t.get("speaker") == "agent")
        if policy.window == "first_3_agent_turns":
            return agent_turns > 3
        if policy.window == "first_2_agent_turns":
            return agent_turns > 2
        # Whole-call obligations can only be judged at the end.
        return False

    def _to_signal(
        self, state: CallState, risk: FlaggedRisk, *, path: str, prompt_ref: str
    ) -> Signal | None:
        if not risk.evidence:
            if METRICS_AVAILABLE:
                SIGNALS_REJECTED.labels(reason="no_evidence").inc()
            return None
        if risk.confidence < self.config.min_confidence:
            if METRICS_AVAILABLE:
                SIGNALS_REJECTED.labels(reason="low_confidence").inc()
            return None

        window = state.window(self.config.window_size)
        return Signal(
            signal_id=f"sig-{uuid.uuid4().hex[:16]}",
            call_id=state.call_id,
            type=risk.type,
            label=risk.label,
            severity=risk.severity,
            confidence=risk.confidence,
            rationale=risk.rationale[:1000],
            evidence=risk.evidence,
            policy_id=risk.policy_id,
            window_start_ms=window[0]["start_ms"] if window else 0,
            window_end_ms=window[-1]["end_ms"] if window else 0,
            emitted_at=utc_now_iso(),
            agent_id=state.agent_id,
            latency=StageLatency(),
            model_name=self.provider.model if path == "agent" else "rules",
            prompt_version=prompt_ref,
            traceparent=current_traceparent() or state.traceparent,
        )

    def _accept(self, signal: Signal, seen: set[tuple], state: CallState) -> bool:
        """Deduplicate within the window and across the call.

        Within a window, the fast and agent paths often find the same thing;
        the first one wins. Across windows the rule is that a signal must
        represent a *change*, because a sliding window re-derives the same
        conclusion from the same turns every time it moves:

        * compliance violations fire once per policy per call;
        * intent fires only when the classification changes;
        * escalation risk fires when the driver or severity changes, or when
          confidence climbs materially -- "now demanding a supervisor" is
          news, "still annoyed" is not;
        * a sentiment shift is identified by the turn it anchors on, so a new
          shift reports and a re-observed one does not.
        """
        key = (signal.type.value, signal.label, signal.policy_id)
        if key in seen:
            return False

        if signal.type is SignalType.COMPLIANCE_VIOLATION and signal.policy_id:
            if signal.policy_id in state.violations_flagged:
                if METRICS_AVAILABLE:
                    SIGNALS_REJECTED.labels(reason="duplicate_violation").inc()
                return False
            state.violations_flagged.add(signal.policy_id)
        else:
            signature = self._signature(signal)
            previous = state.last_signal.get(signal.type.value)
            if previous is not None and not self._is_change(signal, previous):
                if METRICS_AVAILABLE:
                    SIGNALS_REJECTED.labels(reason="unchanged").inc()
                return False
            state.last_signal[signal.type.value] = signature

        seen.add(key)
        if METRICS_AVAILABLE:
            SIGNALS_EMITTED.labels(
                signal_type=signal.type.value,
                path="fast" if signal.model_name == "rules" else "agent",
            ).inc()
        return True


    @staticmethod
    def _signature(signal: Signal) -> tuple:
        anchor = signal.evidence[0].seq if signal.evidence else -1
        return (signal.label, signal.severity.value, round(signal.confidence, 2), anchor)

    @staticmethod
    def _is_change(signal: Signal, previous: tuple) -> bool:
        prev_label, prev_severity, prev_confidence, prev_anchor = previous
        if signal.type is SignalType.SENTIMENT_SHIFT:
            # A shift is identified by where it happened.
            anchor = signal.evidence[0].seq if signal.evidence else -1
            return anchor != prev_anchor or signal.label != prev_label
        if signal.type is SignalType.INTENT:
            return signal.label != prev_label
        # escalation_risk and anything new: report movement, not persistence.
        if signal.label != prev_label or signal.severity.value != prev_severity:
            return True
        return signal.confidence - prev_confidence >= 0.15


def _render(turns: list[dict]) -> str:
    return "\n".join(f"[{t['idx']}] {t['speaker']}: {t['text']}" for t in turns)


def _severity(score: float) -> Severity:
    if score >= 0.95:
        return Severity.CRITICAL
    if score >= 0.75:
        return Severity.HIGH
    if score >= 0.5:
        return Severity.MEDIUM
    return Severity.LOW


def _elapsed_ms_since(iso_ts: str) -> float | None:
    """Milliseconds since an RFC 3339 timestamp produced on another host.

    Returns None for implausible values. Cross-host wall-clock deltas are only
    as good as NTP; a negative or hour-long delta means clock skew, and
    feeding that into the SLO histogram would corrupt the percentile rather
    than reveal a problem.
    """
    try:
        delta = (utc_now() - iso_to_datetime(iso_ts)).total_seconds() * 1000.0
    except Exception:
        return None
    if delta < 0 or delta > 3_600_000:
        return None
    return delta
