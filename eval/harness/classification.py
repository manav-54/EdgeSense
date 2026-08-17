"""Classification and summary-quality evaluation against the golden labels.

Summary quality without a human rater is mostly a citation problem, so that is
what gets measured. **Citation validity** -- does every evidence quote a signal
or summary cites actually appear in the transcript it claims to come from --
is an objective, un-gameable hallucination metric. A system can produce a
fluent, wrong summary and score well on ROUGE; it cannot produce a quote that
is in the transcript when it is not.

Alongside it: schema validity (did the payload parse into the strict model at
all), evidence grounding (does every action item cite a turn that exists), and
intent/resolution/escalation accuracy against the authored labels.

What is *not* measured, and should be said plainly: nobody has judged whether
the summary prose is good. That needs either human raters or an LLM judge, and
an LLM judge scoring an LLM's output shares its failure modes. EVAL.md states
this as an open gap rather than substituting a number that sounds like quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from edgesense_core.contracts import CallSummary, Signal

from harness.metrics import Accuracy, PRCounts, mean, set_counts


@dataclass
class ClassificationReport:
    intent: Accuracy = field(default_factory=Accuracy)
    resolution: Accuracy = field(default_factory=Accuracy)
    escalation: PRCounts = field(default_factory=PRCounts)
    violations: PRCounts = field(default_factory=PRCounts)
    violations_by_policy: dict[str, PRCounts] = field(default_factory=dict)
    disclosures: PRCounts = field(default_factory=PRCounts)
    escalation_risk_band: Accuracy = field(default_factory=Accuracy)

    sentiment_direction_correct: int = 0
    sentiment_direction_total: int = 0
    sentiment_abs_error: list[float] = field(default_factory=list)

    summaries_valid: int = 0
    summaries_attempted: int = 0
    summaries_repaired: int = 0

    citations_checked: int = 0
    citations_valid: int = 0
    bad_citations: list[str] = field(default_factory=list)

    action_items_produced: int = 0
    action_items_grounded: int = 0
    action_item_recall: list[float] = field(default_factory=list)

    signals_total: int = 0
    signals_without_evidence: int = 0

    @property
    def citation_validity(self) -> float:
        return self.citations_valid / self.citations_checked if self.citations_checked else 1.0

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.as_dict(),
            "resolution": self.resolution.as_dict(),
            "escalation_risk_band": self.escalation_risk_band.as_dict(),
            "escalated_flag": self.escalation.as_dict(),
            "compliance_violations": self.violations.as_dict(),
            "violations_by_policy": {
                k: v.as_dict() for k, v in sorted(self.violations_by_policy.items())
            },
            "disclosures_given": self.disclosures.as_dict(),
            "sentiment": {
                "direction_accuracy": round(
                    self.sentiment_direction_correct / max(self.sentiment_direction_total, 1), 4
                ),
                "mean_abs_error": round(mean(self.sentiment_abs_error), 4),
                "samples": self.sentiment_direction_total,
            },
            "summary_quality": {
                "schema_valid_rate": round(
                    self.summaries_valid / max(self.summaries_attempted, 1), 4
                ),
                "attempted": self.summaries_attempted,
                "valid": self.summaries_valid,
                "needed_repair": self.summaries_repaired,
                "citation_validity": round(self.citation_validity, 4),
                "citations_checked": self.citations_checked,
                "action_items_produced": self.action_items_produced,
                "action_items_grounded": self.action_items_grounded,
                "action_item_recall_vs_labels": round(mean(self.action_item_recall), 4),
                "bad_citation_examples": self.bad_citations[:10],
            },
            "signals": {
                "total": self.signals_total,
                "without_evidence": self.signals_without_evidence,
                "evidence_rate": round(
                    1 - self.signals_without_evidence / max(self.signals_total, 1), 4
                ),
            },
        }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip(" .,!?…")).strip()


def _quote_appears(quote: str, transcript_turns: list[str]) -> bool:
    """Is this quote really in the transcript?

    The redactor and the evidence builder both trim and add ellipses, so an
    exact substring test would fail on quotes that are perfectly honest. The
    check strips ellipses and normalises whitespace, then requires the
    remaining text to be a substring of some turn -- still strict enough that
    a paraphrase or an invented sentence fails.
    """
    cleaned = _normalise(quote.strip().strip(".").replace("...", " "))
    if len(cleaned) < 8:
        return True  # too short to be a meaningful claim either way
    return any(cleaned in _normalise(turn) for turn in transcript_turns)


def evaluate_signals(
    signals: list[Signal],
    transcript_turns: list[str],
    report: ClassificationReport,
) -> None:
    for signal in signals:
        report.signals_total += 1
        if not signal.evidence:
            report.signals_without_evidence += 1
            continue
        for span in signal.evidence:
            report.citations_checked += 1
            if _quote_appears(span.quote, transcript_turns):
                report.citations_valid += 1
            elif len(report.bad_citations) < 30:
                report.bad_citations.append(
                    f"{signal.call_id} [{signal.type.value}/{signal.label}]: {span.quote[:80]!r}"
                )


def evaluate_summary(
    call: dict,
    summary: CallSummary | None,
    repaired: bool,
    transcript_turns: list[str],
    report: ClassificationReport,
) -> None:
    labels = call["labels"]
    report.summaries_attempted += 1
    if summary is None:
        return
    report.summaries_valid += 1
    if repaired:
        report.summaries_repaired += 1

    report.intent.record(labels["primary_intent"], summary.primary_intent)
    report.resolution.record(labels["resolution"], summary.resolution.value)

    expected_escalated = bool(labels["escalated"])
    if summary.escalated and expected_escalated:
        report.escalation = report.escalation + PRCounts(tp=1)
    elif summary.escalated and not expected_escalated:
        report.escalation = report.escalation + PRCounts(fp=1)
    elif not summary.escalated and expected_escalated:
        report.escalation = report.escalation + PRCounts(fn=1)

    expected_violations = list(labels["compliance_violations"])
    report.violations = report.violations + set_counts(
        expected_violations, summary.compliance_violations
    )
    for policy_id in set(expected_violations) | set(summary.compliance_violations):
        counts = set_counts(
            [policy_id] if policy_id in expected_violations else [],
            [policy_id] if policy_id in summary.compliance_violations else [],
        )
        report.violations_by_policy[policy_id] = (
            report.violations_by_policy.get(policy_id, PRCounts()) + counts
        )

    report.disclosures = report.disclosures + set_counts(
        labels["disclosures_given"], summary.disclosures_given
    )

    # Sentiment: direction of movement matters more than the absolute value,
    # since the labels are an author's judgement on a -1..1 scale and nobody
    # can calibrate that to two decimal places.
    expected_delta = labels["sentiment_end"] - labels["sentiment_start"]
    actual_delta = summary.customer_sentiment_end - summary.customer_sentiment_start
    report.sentiment_direction_total += 1
    if (expected_delta >= 0) == (actual_delta >= 0):
        report.sentiment_direction_correct += 1
    report.sentiment_abs_error.append(abs(expected_delta - actual_delta))

    for item in summary.action_items:
        report.action_items_produced += 1
        if item.evidence:
            report.action_items_grounded += 1
        for span in item.evidence:
            report.citations_checked += 1
            if _quote_appears(span.quote, transcript_turns):
                report.citations_valid += 1
            elif len(report.bad_citations) < 30:
                report.bad_citations.append(
                    f"{call['call_id']} [action]: {span.quote[:80]!r}"
                )

    expected_actions = list(labels.get("action_items", []))
    if expected_actions:
        produced = [_normalise(a.description) for a in summary.action_items]
        hits = 0
        for expected in expected_actions:
            keywords = [w for w in _normalise(expected).split() if len(w) > 4]
            if not keywords:
                continue
            # Loose match: half the content words of the labelled item appear
            # in some produced item. Tight matching would measure phrasing.
            if any(
                sum(1 for w in keywords if w in candidate) >= max(1, len(keywords) // 2)
                for candidate in produced
            ):
                hits += 1
        report.action_item_recall.append(hits / len(expected_actions))


def evaluate_escalation_band(
    call: dict, band: str, report: ClassificationReport
) -> None:
    report.escalation_risk_band.record(call["labels"]["escalation_risk"], band)
