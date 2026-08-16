"""Build the golden corpus from hand-authored scenarios.

Run: ``python -m tools.corpus.generate --out eval/golden``

Placeholders in scenario text are replaced with fabricated PII values, and the
character offsets of every substitution are recorded as ground truth. Because
the spans are computed at substitution time rather than annotated afterwards,
they are exact -- there is no annotator drift in the redaction numbers.

Determinism matters more than variety here: the same ``--seed`` always yields
byte-identical output, so an eval delta is attributable to a code or prompt
change rather than to a reshuffled corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from tools.corpus import fills
from tools.corpus.fills import SurfaceForm
from tools.corpus.templates import SCENARIOS, Scenario

# {TYPE}, {TYPE@slot}, {TYPE@slot/half}
PLACEHOLDER = re.compile(r"\{([A-Z_]+)(?:@(\d+))?(?:/([12]))?\}")

#: Categories that get a second fill variant, to reach ~48 calls from 30
#: scenarios without diluting the corpus with near-duplicate easy calls.
DOUBLED_CATEGORIES = {"pii_heavy", "compliance", "adversarial"}

AGENT_POOL = ("Ray", "Nadia", "Marcus", "Imani", "Devon", "Sofia")


class SlotTable:
    """Per-call registry mapping ``TYPE@slot`` to a concrete fabricated value.

    A scenario that mentions ``{CARD}`` twice gets two different cards; one
    that writes ``{CARD}`` then ``{CARD@1}`` gets the same card twice, which is
    how a genuine readback is expressed.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._slots: dict[str, str] = {}
        self._auto: dict[str, int] = {}

    def resolve(self, kind: str, slot: str | None) -> tuple[str, str]:
        if slot is None:
            self._auto[kind] = self._auto.get(kind, 0) + 1
            slot = str(self._auto[kind])
        key = f"{kind}@{slot}"
        if key not in self._slots:
            self._slots[key] = fills.pick(kind, self._rng)
        return key, self._slots[key]


def _split_halves(value: str) -> tuple[str, str]:
    """Split a value into two halves at a digit boundary.

    Splitting on digit count rather than string length keeps both halves
    meaningful when the value carries separators (``900-45-6789``).
    """
    digits = [i for i, ch in enumerate(value) if ch.isdigit()]
    if len(digits) < 4:
        mid = len(value) // 2
        return value[:mid], value[mid:]
    cut = digits[len(digits) // 2]
    return value[:cut].rstrip(" -"), value[cut:].lstrip(" -")


def _render_turn(
    text: str,
    table: SlotTable,
    forms: tuple[SurfaceForm, ...],
    rng: random.Random,
    agent_name: str,
) -> tuple[str, list[dict]]:
    """Substitute placeholders, returning the final text and its PII spans."""
    out: list[str] = []
    spans: list[dict] = []
    cursor = 0

    for m in PLACEHOLDER.finditer(text):
        out.append(text[cursor : m.start()])
        cursor = m.end()

        kind, slot, half = m.group(1), m.group(2), m.group(3)

        if kind == "AGENT":
            out.append(agent_name)
            continue

        key, value = table.resolve(kind, slot)
        form = rng.choice(forms) if forms else SurfaceForm.PLAIN

        if half is not None:
            piece = _split_halves(value)[int(half) - 1]
            rendered = fills.render(piece, form, rng)
        else:
            piece = value
            rendered = fills.render(value, form, rng)

        start = sum(len(part) for part in out)
        out.append(rendered)
        spans.append(
            {
                "type": kind,
                "start": start,
                "end": start + len(rendered),
                "value": piece,
                "canonical": value,
                "surface_form": form.value,
                "is_partial": half is not None,
                "split_group": key if half is not None else None,
            }
        )

    out.append(text[cursor:])
    final = "".join(out)

    # Offsets were accumulated against the pieces list; assert they survived
    # the join, because a silent off-by-one here would corrupt every recall
    # number downstream.
    for s in spans:
        actual = final[s["start"] : s["end"]]
        expected = fills.render(s["value"], SurfaceForm(s["surface_form"]), random.Random(0))
        if actual != expected and s["surface_form"] != SurfaceForm.NOISY.value:
            raise AssertionError(
                f"span drift: text[{s['start']}:{s['end']}]={actual!r} for {s['type']}"
            )
    return final, spans


def build_call(scenario: Scenario, variant: int, seed: int) -> dict:
    rng = random.Random(f"{scenario.slug}:{variant}:{seed}")
    table = SlotTable(rng)
    agent_name = AGENT_POOL[rng.randrange(len(AGENT_POOL))]
    agent_id = f"agent_{AGENT_POOL.index(agent_name) + 1:02d}"
    forms = tuple(SurfaceForm(f) for f in scenario.surface_forms)

    turns: list[dict] = []
    for idx, (speaker, raw) in enumerate(scenario.turns):
        text, spans = _render_turn(raw, table, forms, rng, agent_name)
        turns.append({"idx": idx, "speaker": speaker, "text": text, "pii": spans})

    call_id = f"gold-{scenario.slug}-v{variant}"
    labels = asdict(scenario.labels)
    labels = {k: list(v) if isinstance(v, tuple) else v for k, v in labels.items()}

    return {
        "call_id": call_id,
        "scenario": scenario.slug,
        "variant": variant,
        "category": scenario.category,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "turns": turns,
        "labels": labels,
        "pii_count": sum(len(t["pii"]) for t in turns),
    }


def generate(out_dir: Path, seed: int = 20260816) -> dict:
    calls_dir = out_dir / "calls"
    if calls_dir.exists():
        shutil.rmtree(calls_dir)
    calls_dir.mkdir(parents=True, exist_ok=True)

    calls: list[dict] = []
    for scenario in SCENARIOS:
        variants = 2 if scenario.category in DOUBLED_CATEGORIES else 1
        for v in range(variants):
            calls.append(build_call(scenario, v, seed))

    for call in calls:
        path = calls_dir / f"{call['call_id']}.json"
        path.write_text(json.dumps(call, indent=2, ensure_ascii=False) + "\n")

    by_category: dict[str, int] = {}
    by_pii_type: dict[str, int] = {}
    by_surface: dict[str, int] = {}
    for call in calls:
        by_category[call["category"]] = by_category.get(call["category"], 0) + 1
        for turn in call["turns"]:
            for span in turn["pii"]:
                by_pii_type[span["type"]] = by_pii_type.get(span["type"], 0) + 1
                by_surface[span["surface_form"]] = by_surface.get(span["surface_form"], 0) + 1

    index = {
        "seed": seed,
        "call_count": len(calls),
        "turn_count": sum(len(c["turns"]) for c in calls),
        "pii_span_count": sum(c["pii_count"] for c in calls),
        "by_category": dict(sorted(by_category.items())),
        "by_pii_type": dict(sorted(by_pii_type.items())),
        "by_surface_form": dict(sorted(by_surface.items())),
        "calls": [
            {
                "call_id": c["call_id"],
                "category": c["category"],
                "scenario": c["scenario"],
                "pii_count": c["pii_count"],
                "violations": c["labels"]["compliance_violations"],
            }
            for c in calls
        ],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the EdgeSense golden corpus.")
    ap.add_argument("--out", type=Path, default=Path("eval/golden"))
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()

    index = generate(args.out, args.seed)
    print(f"wrote {index['call_count']} calls -> {args.out}/calls")
    print(f"  turns:     {index['turn_count']}")
    print(f"  PII spans: {index['pii_span_count']}")
    print(f"  category:  {index['by_category']}")
    print(f"  pii type:  {index['by_pii_type']}")
    print(f"  surface:   {index['by_surface_form']}")


if __name__ == "__main__":
    main()
