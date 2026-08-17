"""Redaction evaluation: precision, recall, and leak rate per PII type.

Three levels of strictness, reported separately because they answer different
questions:

**Leak rate** -- did the secret survive anywhere in the output, in any of its
three spaces (literal, digit projection, spoken words)? This is the safety
metric. It is the only one that maps onto a disclosure incident, and the only
one that can be measured in audio mode where exact spans do not exist.

**Recall (span coverage)** -- was the labelled span actually covered by a
redaction? Slightly stricter than the leak rate: a value can be non-leaking
because ASR mangled it while the redactor still missed it.

**Type accuracy** -- given the value was caught, was it labelled correctly? A
phone number redacted as ``<ACCOUNT_1>`` is a near-miss: the secret is gone
and only the placeholder is wrong. Counting that as a failure would push the
system towards fewer, more confident detections, which is the wrong direction.

Precision is measured against the labelled spans, so a redaction that covers
no labelled PII is a false positive. The corpus deliberately contains
non-PII lookalikes (a Luhn-invalid sixteen-digit order number) so the
precision cost of the recall bias shows up as a number rather than a caveat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from edgesense_core.contracts import RedactionRef

from harness.metrics import PRCounts

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

#: Below this many digits a value is not treated as a secret in the leak check
#: (last-four is routinely disclosed by design).
MIN_LEAK_DIGITS = 5


@dataclass
class SpanResult:
    """Outcome for one labelled PII span."""

    call_id: str
    turn_idx: int
    pii_type: str
    surface_form: str
    is_partial: bool
    value: str
    covered: bool
    type_correct: bool
    leaked: bool
    leak_space: str = ""
    detector: str = ""


@dataclass
class RedactionReport:
    by_type: dict[str, PRCounts] = field(default_factory=dict)
    by_surface: dict[str, PRCounts] = field(default_factory=dict)
    by_category: dict[str, PRCounts] = field(default_factory=dict)
    by_detector: dict[str, int] = field(default_factory=dict)
    type_correct: int = 0
    type_confused: int = 0
    leaks: list[SpanResult] = field(default_factory=list)
    spans_total: int = 0
    false_positives: int = 0
    fp_examples: list[str] = field(default_factory=list)

    @property
    def overall(self) -> PRCounts:
        total = PRCounts()
        for counts in self.by_type.values():
            total = total + counts
        return total

    @property
    def leak_rate(self) -> float:
        return len(self.leaks) / self.spans_total if self.spans_total else 0.0

    def as_dict(self) -> dict:
        return {
            "spans_total": self.spans_total,
            "overall": self.overall.as_dict(),
            "leak_count": len(self.leaks),
            "leak_rate": round(self.leak_rate, 5),
            "type_correct": self.type_correct,
            "type_confused": self.type_confused,
            "type_accuracy": round(
                self.type_correct / max(self.type_correct + self.type_confused, 1), 4
            ),
            "by_type": {k: v.as_dict() for k, v in sorted(self.by_type.items())},
            "by_surface_form": {k: v.as_dict() for k, v in sorted(self.by_surface.items())},
            "by_category": {k: v.as_dict() for k, v in sorted(self.by_category.items())},
            "by_detector": dict(sorted(self.by_detector.items())),
            "false_positives": self.false_positives,
            "fp_examples": self.fp_examples[:12],
            "leaks": [
                {
                    "call_id": leak.call_id,
                    "turn": leak.turn_idx,
                    "type": leak.pii_type,
                    "surface_form": leak.surface_form,
                    "space": leak.leak_space,
                }
                for leak in self.leaks[:40]
            ],
        }


def digits_of(text: str) -> str:
    return re.sub(r"\D", "", text)


def spoken_of(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def as_digit_words(digits: str) -> str:
    return " ".join(DIGIT_WORDS[d] for d in digits if d in DIGIT_WORDS)


def find_leak(value: str, emitted: str) -> str:
    """Return the space a value leaked in, or '' if it did not.

    Checked in three spaces because a card reformatted on the way out is still
    the same card. See services/edge-agent/tests/leakcheck.py -- the same
    oracle, kept in both places on purpose: the test enforces it in CI and the
    eval measures it over the corpus.
    """
    digits = digits_of(value)

    if len(digits) >= MIN_LEAK_DIGITS:
        if digits in digits_of(emitted):
            return "digits"
        words = as_digit_words(digits)
        if words and words in spoken_of(emitted):
            return "spoken"
        return ""

    if "@" in value:
        return "raw" if value in emitted else ""

    if not digits and len(value.strip()) >= 5:
        return "raw" if value.strip() in emitted else ""

    # Too short to assert on -- e.g. a 4-digit fragment. Not counted either way.
    return ""


#: Fraction of a non-numeric span's characters that must be covered to count.
#: Names and addresses have fuzzy boundaries -- a detector that takes
#: "1420 Marigold Lane, Apartment 3" and leaves the trailing "B" has removed
#: the address; failing that would measure boundary agreement, not redaction.
CHAR_COVERAGE_THRESHOLD = 0.9


def _covers(
    ref_spans: list[tuple[int, int]], expected: tuple[int, int], text: str
) -> bool:
    """Did the redactions remove this labelled value?

    The rule differs by content, because "removed" means different things:

    * **If the span contains digits**, every digit position must be covered.
      That is the whole secret: a card number with its digits gone is gone,
      and the label prefix ("ACCT-") staying readable is intentional -- it
      keeps the word "account" available to downstream intent classification.
    * **Otherwise** (names, addresses), at least ``CHAR_COVERAGE_THRESHOLD`` of
      the characters must be covered.

    Coverage is computed across *all* redactions rather than requiring one to
    contain the span, because the redactor legitimately splits or widens spans
    -- swallowing disfluencies between digit groups, or redacting a labelled
    number without its label.
    """
    start, end = expected
    covered = set()
    for ref_start, ref_end in ref_spans:
        lo, hi = max(ref_start, start), min(ref_end, end)
        if lo < hi:
            covered.update(range(lo, hi))

    digit_positions = [i for i in range(start, end) if i < len(text) and text[i].isdigit()]
    if digit_positions:
        return all(i in covered for i in digit_positions)

    width = end - start
    return width > 0 and len(covered) / width >= CHAR_COVERAGE_THRESHOLD


@dataclass
class TurnOutcome:
    """What the redactor did to one turn."""

    original: str
    emitted: str
    #: (start, end) of each redaction in ORIGINAL coordinates.
    covered_spans: list[tuple[int, int]]
    refs: list[RedactionRef]


def evaluate_call(
    call: dict,
    outcomes: list[TurnOutcome],
    report: RedactionReport,
) -> list[SpanResult]:
    """Score one call's redaction against its labels.

    ``outcomes`` is indexed by turn. The whole call's emitted text is used for
    the leak check, because a value withheld from turn 3 and released in turn 4
    has still leaked.
    """
    category = call.get("category", "unknown")
    all_emitted = "\n".join(outcome.emitted for outcome in outcomes)
    results: list[SpanResult] = []

    matched_ref_ids: set[tuple[int, int, int]] = set()

    for turn in call["turns"]:
        idx = turn["idx"]
        if idx >= len(outcomes):
            continue
        outcome = outcomes[idx]

        for span in turn["pii"]:
            expected = (span["start"], span["end"])
            pii_type = span["type"]
            # NON_CARD is a deliberate lookalike: not PII, so it has no
            # recall obligation. Handled in the false-positive pass below.
            if pii_type == "NON_CARD":
                continue

            covered = _covers(outcome.covered_spans, expected, outcome.original)
            type_correct = False
            detector = ""
            if covered:
                # Attribute the catch to the redaction overlapping the span.
                for i, ref_span in enumerate(outcome.covered_spans):
                    if ref_span[0] < expected[1] and expected[0] < ref_span[1]:
                        matched_ref_ids.add((idx, ref_span[0], ref_span[1]))
                        ref = outcome.refs[i] if i < len(outcome.refs) else None
                        if ref is not None:
                            detector = ref.detector
                            type_correct = ref.type.value == pii_type
                        break

            leak_space = find_leak(span["value"], all_emitted)
            result = SpanResult(
                call_id=call["call_id"], turn_idx=idx, pii_type=pii_type,
                surface_form=span["surface_form"], is_partial=span["is_partial"],
                value=span["value"], covered=covered, type_correct=type_correct,
                leaked=bool(leak_space), leak_space=leak_space, detector=detector,
            )
            results.append(result)
            report.spans_total += 1

            counts = PRCounts(tp=1) if covered else PRCounts(fn=1)
            report.by_type[pii_type] = report.by_type.get(pii_type, PRCounts()) + counts
            report.by_surface[span["surface_form"]] = (
                report.by_surface.get(span["surface_form"], PRCounts()) + counts
            )
            report.by_category[category] = (
                report.by_category.get(category, PRCounts()) + counts
            )

            if covered:
                report.by_detector[detector or "unknown"] = (
                    report.by_detector.get(detector or "unknown", 0) + 1
                )
                if type_correct:
                    report.type_correct += 1
                else:
                    report.type_confused += 1

            if leak_space:
                report.leaks.append(result)

    # False positives: redactions that covered no labelled PII span.
    for turn in call["turns"]:
        idx = turn["idx"]
        if idx >= len(outcomes):
            continue
        outcome = outcomes[idx]
        labelled = [(s["start"], s["end"]) for s in turn["pii"] if s["type"] != "NON_CARD"]
        for i, ref_span in enumerate(outcome.covered_spans):
            overlaps = any(
                ref_span[0] < end and start < ref_span[1] for start, end in labelled
            )
            if overlaps:
                continue
            report.false_positives += 1
            ref_type = outcome.refs[i].type.value if i < len(outcome.refs) else "?"
            surface = outcome.original[ref_span[0] : ref_span[1]]
            report.by_type[ref_type] = report.by_type.get(ref_type, PRCounts()) + PRCounts(fp=1)
            if len(report.fp_examples) < 40:
                report.fp_examples.append(
                    f"{call['call_id']} turn {idx}: <{ref_type}> over {surface[:60]!r}"
                )

    return results
