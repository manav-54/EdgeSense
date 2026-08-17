"""Prompt regression: run the eval twice and diff it.

The point is to make a prompt change reviewable. Editing a prompt is editing
behaviour, and "it looks better" is not a review. This runs the full corpus
against two prompt versions and prints a before/after table with the deltas,
so a change that improves intent accuracy by two points while quietly leaking
a card number is visible as exactly that.

Metrics are tagged with a direction and a severity:

* ``leak_rate`` and ``citation_validity`` are **gates**. A regression in
  either fails the run with a non-zero exit code, whatever else improved.
  There is no threshold at which more leaks are an acceptable trade.
* everything else is reported with a delta and a marker, for a human to weigh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from harness.runner import RunConfig, RunResult, run


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    get: Callable[[dict], float]
    higher_is_better: bool = True
    #: A gate metric fails the whole comparison when it regresses.
    gate: bool = False
    fmt: str = "{:.2%}"
    #: Deltas smaller than this are noise, not signal.
    tolerance: float = 0.0


METRICS: tuple[Metric, ...] = (
    Metric("leak_rate", "PII leak rate",
           lambda d: d["redaction"]["leak_rate"], higher_is_better=False,
           gate=True, fmt="{:.3%}"),
    Metric("redaction_recall", "Redaction recall",
           lambda d: d["redaction"]["overall"]["recall"], gate=True),
    Metric("redaction_precision", "Redaction precision",
           lambda d: d["redaction"]["overall"]["precision"], tolerance=0.005),
    Metric("redaction_f2", "Redaction F2 (recall-weighted)",
           lambda d: d["redaction"]["overall"]["f2"], fmt="{:.3f}"),
    Metric("type_accuracy", "PII type accuracy",
           lambda d: d["redaction"]["type_accuracy"], tolerance=0.005),
    Metric("citation_validity", "Citation validity",
           lambda d: d["classification"]["summary_quality"]["citation_validity"],
           gate=True),
    Metric("evidence_rate", "Signals with evidence",
           lambda d: d["classification"]["signals"]["evidence_rate"], gate=True),
    Metric("schema_valid", "Summary schema valid",
           lambda d: d["classification"]["summary_quality"]["schema_valid_rate"]),
    Metric("intent_accuracy", "Intent accuracy",
           lambda d: d["classification"]["intent"]["accuracy"], tolerance=0.005),
    Metric("resolution_accuracy", "Resolution accuracy",
           lambda d: d["classification"]["resolution"]["accuracy"], tolerance=0.005),
    Metric("escalation_band", "Escalation band accuracy",
           lambda d: d["classification"]["escalation_risk_band"]["accuracy"],
           tolerance=0.005),
    Metric("escalated_f1", "Escalated flag F1",
           lambda d: d["classification"]["escalated_flag"]["f1"], fmt="{:.3f}"),
    Metric("violation_recall", "Violation recall",
           lambda d: d["classification"]["compliance_violations"]["recall"]),
    Metric("violation_precision", "Violation precision",
           lambda d: d["classification"]["compliance_violations"]["precision"],
           tolerance=0.005),
    Metric("disclosure_f1", "Disclosure F1",
           lambda d: d["classification"]["disclosures_given"]["f1"], fmt="{:.3f}"),
    Metric("sentiment_direction", "Sentiment direction",
           lambda d: d["classification"]["sentiment"]["direction_accuracy"],
           tolerance=0.005),
    Metric("action_item_recall", "Action item recall",
           lambda d: d["classification"]["summary_quality"]["action_item_recall_vs_labels"],
           tolerance=0.01),
    Metric("live_p95_ms", "Live analysis p95 (ms)",
           lambda d: d["timings"]["analysis_ms_p95"], higher_is_better=False,
           fmt="{:.1f}", tolerance=1.0),
)


@dataclass
class Comparison:
    metric: Metric
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def significant(self) -> bool:
        return abs(self.delta) > self.metric.tolerance

    @property
    def improved(self) -> bool:
        if not self.significant:
            return False
        return self.delta > 0 if self.metric.higher_is_better else self.delta < 0

    @property
    def regressed(self) -> bool:
        if not self.significant:
            return False
        return self.delta < 0 if self.metric.higher_is_better else self.delta > 0

    @property
    def marker(self) -> str:
        if not self.significant:
            return "  ="
        if self.regressed:
            return "FAIL" if self.metric.gate else "  ▼"
        return "  ▲"


def compare(before: dict, after: dict) -> list[Comparison]:
    out: list[Comparison] = []
    for metric in METRICS:
        try:
            out.append(Comparison(metric, metric.get(before), metric.get(after)))
        except (KeyError, TypeError):
            continue
    return out


def format_table(
    comparisons: list[Comparison], before_label: str, after_label: str
) -> str:
    lines: list[str] = []
    lines.append("=" * 86)
    lines.append(f"PROMPT REGRESSION   {before_label}  →  {after_label}")
    lines.append("=" * 86)
    lines.append(
        f"{'metric':<32} {'before':>12} {'after':>12} {'delta':>12}   {'':<6}"
    )
    lines.append("-" * 86)

    for comparison in comparisons:
        metric = comparison.metric
        before_text = metric.fmt.format(comparison.before)
        after_text = metric.fmt.format(comparison.after)
        if metric.fmt.endswith("%}"):
            delta_text = f"{comparison.delta * 100:+.3f}pp"
        else:
            delta_text = f"{comparison.delta:+.3f}"
        gate = " (gate)" if metric.gate else ""
        lines.append(
            f"{metric.label + gate:<32} {before_text:>12} {after_text:>12} "
            f"{delta_text:>12}   {comparison.marker}"
        )

    lines.append("-" * 86)
    regressions = [c for c in comparisons if c.regressed]
    gate_failures = [c for c in regressions if c.metric.gate]
    improvements = [c for c in comparisons if c.improved]

    lines.append(
        f"{len(improvements)} improved, {len(regressions)} regressed, "
        f"{len(comparisons) - len(improvements) - len(regressions)} unchanged"
    )
    if gate_failures:
        lines.append("")
        lines.append("GATE FAILURES — this change must not ship:")
        for failure in gate_failures:
            lines.append(
                f"  {failure.metric.label}: "
                f"{failure.metric.fmt.format(failure.before)} → "
                f"{failure.metric.fmt.format(failure.after)}"
            )
    elif regressions:
        lines.append("")
        lines.append("Regressions to weigh (no gate breached):")
        for regression in regressions:
            lines.append(
                f"  {regression.metric.label}: "
                f"{regression.metric.fmt.format(regression.before)} → "
                f"{regression.metric.fmt.format(regression.after)}"
            )
    else:
        lines.append("No regressions.")
    lines.append("")
    return "\n".join(lines)


def run_pair(
    base_config: RunConfig,
    before_prompt: str,
    after_prompt: str,
    which: str = "live",
) -> tuple[RunResult, RunResult]:
    """Run the corpus twice, varying one prompt version."""
    from dataclasses import replace

    field_name = "live_prompt" if which == "live" else "post_prompt"
    before_config = replace(base_config, **{field_name: before_prompt})
    after_config = replace(base_config, **{field_name: after_prompt})

    print(f"→ baseline: {which} prompt {before_prompt}")
    before = run(before_config)
    print(f"→ candidate: {which} prompt {after_prompt}")
    after = run(after_config)
    return before, after


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def has_gate_failure(comparisons: list[Comparison]) -> bool:
    return any(c.regressed and c.metric.gate for c in comparisons)
