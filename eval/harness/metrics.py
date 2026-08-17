"""Metric primitives.

Kept separate from the evaluators so the counting rules are stated once and
can be read without wading through corpus handling.

One decision worth spelling out: ``PRCounts`` reports precision and recall
separately and computes F-beta with beta=2 by default rather than F1. F1 says
a missed card and an over-redacted order number are equally bad. They are not
-- one is a disclosure incident, the other is a support ticket -- so the
headline number weights recall four times as heavily. F1 is still reported
alongside, because hiding it would make the numbers look better than they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PRCounts:
    """True/false positive and negative counts for one class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: PRCounts) -> PRCounts:
        return PRCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    def f_beta(self, beta: float = 2.0) -> float:
        """F-beta. beta>1 weights recall, which is the policy here."""
        p, r = self.precision, self.recall
        if p == 0 and r == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    @property
    def f1(self) -> float:
        return self.f_beta(1.0)

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "f2": round(self.f_beta(2.0), 4),
        }


@dataclass
class Accuracy:
    """Correct / total, with the confusions kept for the report."""

    correct: int = 0
    total: int = 0
    confusions: dict[str, int] = field(default_factory=dict)

    def record(self, expected: str, actual: str) -> None:
        self.total += 1
        if expected == actual:
            self.correct += 1
        else:
            key = f"{expected} -> {actual}"
            self.confusions[key] = self.confusions.get(key, 0) + 1

    @property
    def value(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def as_dict(self, top_confusions: int = 6) -> dict:
        ranked = sorted(self.confusions.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "accuracy": round(self.value, 4),
            "correct": self.correct,
            "total": self.total,
            "top_confusions": dict(ranked[:top_confusions]),
        }


def set_counts(expected: Iterable[str], actual: Iterable[str]) -> PRCounts:
    """Compare two label sets (policy ids, disclosures)."""
    exp, act = set(expected), set(actual)
    return PRCounts(
        tp=len(exp & act),
        fp=len(act - exp),
        fn=len(exp - act),
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation: with a few hundred samples
    the interpolated value implies a precision the sample size does not have."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[index]
