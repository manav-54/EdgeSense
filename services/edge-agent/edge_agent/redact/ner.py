"""A small NER pass for the entity types regex cannot express.

Names and street addresses have no reliable surface form, so they get a model.
Everything numeric stays on regex + checksum, which is faster and auditable --
the point of the hybrid is to use the model only where it earns its cost.

Two backends:

* ``SpacyNER`` -- ``en_core_web_sm`` (~12 MB, CPU, ~1-3 ms per segment). Its
  labels are noisy on conversational text: it will call a street a PERSON and
  a house number a DATE. That is tolerable here because we only ask it *where*
  an entity is, then re-type the span ourselves. We never trust its label for
  anything numeric.
* ``HeuristicNER`` -- trigger phrases and a street-suffix pattern. Runs when
  spaCy is not installed, so the edge agent has no hard model dependency.

Both are wrapped so a model failure degrades to the heuristic rather than
taking down the redactor: a crashed NER must never mean unredacted output.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from edgesense_core.contracts import PIIType

from edge_agent.redact.detectors import Detection

log = logging.getLogger(__name__)

STREET_SUFFIX = (
    "lane|road|street|st|avenue|ave|boulevard|blvd|drive|dr|court|ct|way|"
    "place|pl|terrace|circle|parkway|highway"
)
UNIT_SUFFIX = r"(?:,?\s*(?:apartment|apt|unit|suite|ste|flat|#)\.?\s*[\w\-]+)?"

RE_ADDRESS = re.compile(
    rf"\b\d{{1,6}}\s+(?:[A-Z][\w'\-]*\s+){{1,4}}(?:{STREET_SUFFIX})\b{UNIT_SUFFIX}",
    re.IGNORECASE,
)

RE_NAME_TRIGGER = re.compile(
    r"\b(?:my name is|this is|i'm|i am|speaking with|it's|name on the account is|"
    r"account holder is|calling for|on behalf of)\s+"
    r"((?:[A-Z][\w'\-]+)(?:\s+[A-Z][\w'\-]+){0,2})",
)

#: Words that get capitalised in transcripts without being names.
NAME_STOPWORDS = frozenset({
    "i", "okay", "ok", "yes", "no", "thanks", "thank", "hi", "hello", "sorry",
    "sure", "please", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "northwind", "financial", "recovery", "visa", "mastercard", "amex",
    "discover", "american", "express",
})


class NERBackend(Protocol):
    name: str

    def entities(self, text: str) -> list[Detection]: ...


class HeuristicNER:
    """Trigger phrases and street patterns. No model, no startup cost."""

    name = "heuristic"

    def entities(self, text: str) -> list[Detection]:
        out: list[Detection] = []

        for m in RE_ADDRESS.finditer(text):
            out.append(
                Detection(PIIType.ADDRESS, m.start(), m.end(), "ner", 0.8, m.group(0))
            )

        for m in RE_NAME_TRIGGER.finditer(text):
            name = m.group(1)
            if name.split()[0].lower() in NAME_STOPWORDS:
                continue
            s = m.start(1)
            out.append(Detection(PIIType.PERSON, s, s + len(name), "ner", 0.75, name))

        return out


class SpacyNER:
    """spaCy ``en_core_web_sm``, used for span discovery only."""

    name = "spacy"

    #: Labels we accept as *some* kind of personal entity. We re-type them
    #: ourselves; the model's own choice between PERSON / FAC / ORG on
    #: conversational text is not reliable enough to propagate.
    PERSONISH = frozenset({"PERSON"})
    PLACEISH = frozenset({"FAC", "LOC", "GPE"})

    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy  # imported lazily so the package stays optional

        # Only the NER pipe is needed; disabling the rest roughly halves latency.
        self._nlp = spacy.load(model, disable=["lemmatizer", "textcat"])
        self._fallback = HeuristicNER()

    def entities(self, text: str) -> list[Detection]:
        out: list[Detection] = []
        try:
            doc = self._nlp(text)
        except Exception:  # pragma: no cover - defensive
            log.exception("spaCy NER failed; falling back to heuristic")
            return self._fallback.entities(text)

        for ent in doc.ents:
            if ent.label_ in self.PERSONISH:
                if ent.text.split()[0].lower() in NAME_STOPWORDS:
                    continue
                out.append(
                    Detection(PIIType.PERSON, ent.start_char, ent.end_char,
                              "ner", 0.8, ent.text)
                )
            elif ent.label_ in self.PLACEISH:
                # Only accept a place as an address if it looks like a street
                # line; bare city names are not PII on their own.
                if re.search(rf"\b(?:{STREET_SUFFIX})\b", ent.text, re.IGNORECASE):
                    out.append(
                        Detection(PIIType.ADDRESS, ent.start_char, ent.end_char,
                                  "ner", 0.75, ent.text)
                    )

        # The heuristic catches trigger-phrase names and full street lines that
        # the small model splits or mislabels. Union, then let overlap
        # resolution pick -- recall bias means we would rather have both.
        out.extend(self._fallback.entities(text))
        return out


def load_ner(prefer: str = "auto") -> NERBackend:
    """Return the best available NER backend.

    ``prefer='heuristic'`` forces the dependency-free path, which the eval
    harness uses to attribute recall changes to the model rather than to a
    coincidental spaCy version bump.
    """
    if prefer == "heuristic":
        return HeuristicNER()
    if prefer in ("auto", "spacy"):
        try:
            return SpacyNER()
        except Exception as exc:
            if prefer == "spacy":
                raise
            log.info("spaCy unavailable (%s); using heuristic NER", exc)
    return HeuristicNER()
