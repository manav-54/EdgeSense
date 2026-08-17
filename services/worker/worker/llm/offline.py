"""Deterministic provider implementing the same tool-calling protocol.

This is not a stub that returns canned text. It walks the same loop the model
walks: it calls ``lookup_policy`` before asserting a violation, calls
``search_transcript`` to locate evidence, raises findings through
``flag_risk``, and emits the post-call summary as JSON validated against the
same schema. Swapping providers changes the quality of the judgement, not the
shape of the pipeline.

Why build it this way rather than calling the rules engine directly and
skipping the loop? Because then the loop would only ever be exercised with
live credentials, and every bug in tool dispatch, evidence resolution, retry,
and schema validation would surface for the first time in production. Here the
loop is on the default path and the eval covers it.

It is bound to the analysis context in-process, so it reads call state
directly instead of parsing it back out of the prompt.
"""

from __future__ import annotations

import json
from typing import Any

from edgesense_core.timeutil import monotonic_ms

from worker.analysis import rules
from worker.llm.base import Completion, Message, Role, ToolCall, ToolSpec, Usage
from worker.tools import ToolContext


class OfflineProvider:
    """Rule-driven provider. Deterministic, ~1 ms, no network."""

    name = "offline"
    model = "edgesense-rules-v1"
    needs_context = True

    def __init__(self) -> None:
        self._ctx: ToolContext | None = None
        self._required: list[str] = []
        self._flagged_yet = False

    def set_context(self, ctx: ToolContext, required_disclosures: list[str] | None = None) -> None:
        self._ctx = ctx
        self._required = list(required_disclosures or [])
        # Each analysis pass is a fresh conversation, so the once-only flag
        # guard resets with the context rather than persisting across calls.
        self._flagged_yet = False

    # -- protocol ----------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion:
        t0 = monotonic_ms()
        if self._ctx is None:
            return Completion(content="DONE", model=self.model,
                              latency_ms=monotonic_ms() - t0)

        # The loop's position is inferred from how many assistant turns have
        # already happened, which is the only state a stateless provider has.
        step = sum(1 for m in messages if m.role is Role.ASSISTANT)
        is_post_call = self._is_post_call(messages)

        if step == 0:
            calls = self._gather_calls()
            if calls:
                return Completion(content="", tool_calls=calls, model=self.model,
                                  latency_ms=monotonic_ms() - t0,
                                  usage=Usage(prompt_tokens=0, completion_tokens=0),
                                  finish_reason="tool_calls")
            step = 1  # nothing to look up; fall through to flagging

        # Flag once. Without this guard the provider re-derives the same
        # findings on every step until max_steps, which the dedupe absorbs but
        # which wastes three quarters of the loop's budget.
        if step >= 1 and not self._flagged_yet:
            calls = self._flag_calls(is_post_call)
            if calls:
                self._flagged_yet = True
                return Completion(content="", tool_calls=calls, model=self.model,
                                  latency_ms=monotonic_ms() - t0,
                                  finish_reason="tool_calls")

        content = self._final_content(is_post_call)
        return Completion(content=content, model=self.model,
                          latency_ms=monotonic_ms() - t0)

    # -- phases ------------------------------------------------------------

    def _gather_calls(self) -> list[ToolCall]:
        """Phase 1: fetch the policies and locate the evidence."""
        assert self._ctx is not None
        ctx = self._ctx
        turns = ctx.state.all_turns()
        calls: list[ToolCall] = []

        suspects: list[str] = []
        prohibited = rules.compliance_findings(turns, ctx.policies.raw())
        suspects.extend(f.policy_id for f in prohibited if f.policy_id)
        if self._required:
            _, missing = rules.disclosure_status(turns, ctx.policies.raw(), self._required)
            suspects.extend(f.policy_id for f in missing if f.policy_id)

        for i, policy_id in enumerate(dict.fromkeys(suspects)):
            calls.append(ToolCall(id=f"call_pol_{i}", name="lookup_policy",
                                  arguments={"policy_id": policy_id}))

        band, _, escalation = rules.escalation_risk(turns)
        if band != "none" and escalation:
            calls.append(ToolCall(
                id="call_search_0", name="search_transcript",
                arguments={"query": escalation[0].quote[:60], "top": 3},
            ))
        return calls[:8]

    def _flag_calls(self, is_post_call: bool) -> list[ToolCall]:
        """Phase 2: raise each finding through flag_risk."""
        assert self._ctx is not None
        ctx = self._ctx
        turns = ctx.state.all_turns()
        calls: list[ToolCall] = []
        n = 0

        def add(args: dict[str, Any]) -> None:
            nonlocal n
            calls.append(ToolCall(id=f"call_flag_{n}", name="flag_risk", arguments=args))
            n += 1

        # Compliance: prohibited phrases.
        for finding in rules.compliance_findings(turns, ctx.policies.raw()):
            if finding.policy_id in ctx.state.violations_flagged:
                continue
            add({
                "type": "compliance_violation",
                "label": finding.label,
                "policy_id": finding.policy_id,
                "severity": _severity_for(finding.score),
                "confidence": round(min(0.95, 0.6 + finding.score * 0.35), 3),
                "rationale": finding.detail,
                "evidence_turns": [finding.turn_idx],
            })

        # Compliance: missing required disclosures.
        if self._required:
            given, missing = rules.disclosure_status(
                turns, ctx.policies.raw(), self._required
            )
            ctx.state.disclosures_given.update(given)
            for finding in missing:
                if finding.policy_id in ctx.state.violations_flagged:
                    continue
                add({
                    "type": "compliance_violation",
                    "label": finding.label,
                    "policy_id": finding.policy_id,
                    "severity": _severity_for(finding.score),
                    "confidence": round(min(0.9, 0.55 + finding.score * 0.35), 3),
                    "rationale": finding.detail,
                    "evidence_turns": [finding.turn_idx],
                })

        # Escalation.
        band, score, escalation = rules.escalation_risk(turns)
        if band in ("medium", "high") and escalation:
            top = max(escalation, key=lambda f: f.score)
            add({
                "type": "escalation_risk",
                "label": top.label,
                "severity": "high" if band == "high" else "medium",
                "confidence": round(score, 3),
                "rationale": f"escalation risk {band} ({top.label})",
                "evidence_turns": [f.turn_idx for f in escalation[:3]],
            })

        # Sentiment shift: only when tone actually moved.
        shift = self._sentiment_shift(turns)
        if shift is not None:
            label, delta, idx = shift
            add({
                "type": "sentiment_shift",
                "label": label,
                "severity": "medium" if abs(delta) > 0.6 else "low",
                "confidence": round(min(0.9, abs(delta)), 3),
                "rationale": f"customer sentiment moved by {delta:+.2f} across the window",
                "evidence_turns": [idx],
            })

        # Intent, once per pass.
        intent, _, _ = rules.classify_intent(turns)
        evidence = rules.intent_evidence(turns, intent)
        if evidence is not None:
            add({
                "type": "intent",
                "label": intent,
                "severity": "info",
                "confidence": 0.7,
                "rationale": "keyword-weighted intent classification",
                "evidence_turns": [evidence[0]],
            })

        return calls[:12]

    def _final_content(self, is_post_call: bool) -> str:
        if not is_post_call:
            return "DONE"
        return json.dumps(self._summary_json())

    def _summary_json(self) -> dict:
        assert self._ctx is not None
        ctx = self._ctx
        turns = ctx.state.all_turns()

        intent, secondary, _ = rules.classify_intent(turns)
        band, _, escalation_hits = rules.escalation_risk(turns)
        escalated = any(
            f.label in ("supervisor_request", "supervisor_mention", "manager_mention")
            for f in escalation_hits
        )
        resolution, res_idx, _ = rules.classify_resolution(turns, escalated)

        customer = [t for t in turns if t.get("speaker") == "customer"]
        head = customer[: max(1, len(customer) // 3)] or customer[:1]
        tail = customer[-max(1, len(customer) // 3):] or customer[-1:]
        s_start = (sum(rules.sentiment(t["text"]) for t in head) / len(head)) if head else 0.0
        s_end = (sum(rules.sentiment(t["text"]) for t in tail) / len(tail)) if tail else 0.0

        violations = sorted({
            f.policy_id for f in rules.compliance_findings(turns, ctx.policies.raw())
            if f.policy_id
        } | set(ctx.state.violations_flagged))

        given: list[str] = sorted(ctx.state.disclosures_given)
        if self._required:
            observed, missing = rules.disclosure_status(
                turns, ctx.policies.raw(), self._required
            )
            given = sorted(set(given) | set(observed))
            violations = sorted(set(violations) | {f.policy_id for f in missing if f.policy_id})

        actions = [
            {
                "description": f.label[:200],
                "owner": f.detail or "agent",
                "evidence_turns": [f.turn_idx],
            }
            for f in rules.action_items(turns)[:8]
        ]

        return {
            "summary": self._compose_summary(intent, resolution, escalated, len(turns)),
            "resolution": resolution,
            "primary_intent": intent,
            "secondary_intents": secondary[:3],
            "action_items": actions,
            "customer_sentiment_start": round(max(-1.0, min(1.0, s_start)), 3),
            "customer_sentiment_end": round(max(-1.0, min(1.0, s_end)), 3),
            "escalated": escalated,
            "compliance_violations": violations,
            "disclosures_given": given,
            "evidence_turns": [res_idx],
        }

    def _compose_summary(self, intent: str, resolution: str, escalated: bool, turns: int) -> str:
        topic = intent.replace("_", " ")
        base = f"The customer called about {topic} over {turns} turns."
        if escalated:
            return base + " The call was escalated to a supervisor before it concluded."
        endings = {
            "resolved": " The agent resolved the issue on the call.",
            "follow_up_required": " The agent committed to follow-up action after the call.",
            "unresolved": " The issue was not settled during the call.",
            "escalated": " The call was escalated.",
        }
        return base + endings.get(resolution, "")

    # -- helpers -----------------------------------------------------------

    def _sentiment_shift(self, turns: list[dict]) -> tuple[str, float, int] | None:
        """Detect a genuine move in customer tone across the window."""
        customer = [t for t in turns if t.get("speaker") == "customer"]
        if len(customer) < 3:
            return None
        half = max(1, len(customer) // 2)
        early = sum(rules.sentiment(t["text"]) for t in customer[:half]) / half
        late_turns = customer[-half:]
        late = sum(rules.sentiment(t["text"]) for t in late_turns) / len(late_turns)
        delta = late - early
        if abs(delta) < 0.35:
            return None
        label = "negative_shift" if delta < 0 else "positive_shift"
        anchor = min(late_turns, key=lambda t: rules.sentiment(t["text"])) if delta < 0 \
            else max(late_turns, key=lambda t: rules.sentiment(t["text"]))
        return label, delta, anchor.get("idx", 0)

    @staticmethod
    def _is_post_call(messages: list[Message]) -> bool:
        for m in messages:
            if m.role is Role.USER and "Full transcript:" in m.content:
                return True
        return False


def _severity_for(score: float) -> str:
    if score >= 0.95:
        return "critical"
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
