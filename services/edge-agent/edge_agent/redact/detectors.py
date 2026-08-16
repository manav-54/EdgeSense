"""PII detectors: compiled regex over the surface text, plus checksum- and
context-gated scanning over the normalised digit stream.

**Recall is biased over precision, deliberately.** A missed card number is a
disclosure incident; an over-redacted order number is a support ticket. Where
the two trade off, this module redacts. Three consequences worth naming:

* A 16-digit run that fails Luhn is still redacted (as a lower-confidence
  ``CARD``), because ASR digit errors break checksums on genuine cards --
  a single misheard digit turns a real card into a "not a card".
* Every unclaimed digit run of nine or more digits is redacted as ``ACCOUNT``.
  This is the backstop that makes "no long digit run leaves the process" a
  testable invariant rather than an aspiration.
* Type confusion (calling a phone number an account number) is treated as a
  near-miss, not a failure. The value is gone either way; only the placeholder
  label is wrong.

The precision cost of all three is measured per-type in EVAL.md rather than
hidden behind a single blended F1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from edgesense_core.contracts import PIIType

from edge_agent.redact.normalize import DigitStream, build_digit_stream, luhn_valid

# Priority for overlap resolution: a span claimed by an earlier type wins.
TYPE_PRIORITY: dict[PIIType, int] = {
    PIIType.CARD: 0,
    PIIType.SSN: 1,
    PIIType.DOB: 2,
    PIIType.PHONE: 3,
    PIIType.ACCOUNT: 4,
    PIIType.EMAIL: 5,
    PIIType.PERSON: 6,
    PIIType.ADDRESS: 7,
}

CONTEXT_WINDOW = 70  # characters either side of a candidate

CONTEXT_WORDS: dict[PIIType, tuple[str, ...]] = {
    PIIType.CARD: (
        "card", "visa", "mastercard", "master card", "amex", "american express",
        "discover", "credit", "debit", "cvv", "security code", "expiry",
        "expiration", "pan", "charge", "pay", "payment",
    ),
    PIIType.SSN: (
        "social", "social security", "ssn", "s.s.n", "tax id", "taxpayer", "tin",
    ),
    PIIType.PHONE: (
        "phone", "number to reach", "reach you", "call you", "mobile", "cell",
        "text", "contact number", "best number",
    ),
    PIIType.DOB: (
        "birth", "born", "birthday", "date of birth", "dob", "d.o.b",
    ),
    PIIType.ACCOUNT: (
        "account", "acct", "acc no", "reference", "member", "policy", "order",
        "customer number", "case number",
    ),
}

# --------------------------------------------------------------------------
# Surface-form patterns. Anchored and compiled once at import.
# --------------------------------------------------------------------------

RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Spoken email: "j dot calloway at example dot com"
RE_SPOKEN_EMAIL = re.compile(
    r"\b[A-Za-z0-9]+(?:\s+(?:dot|\.)\s+[A-Za-z0-9]+)*\s+at\s+"
    r"[A-Za-z0-9]+(?:\s+(?:dot|\.)\s+[A-Za-z0-9]+)+\b",
    re.IGNORECASE,
)

RE_SSN_FORMATTED = re.compile(r"\b\d{3}[-\s.]\d{2}[-\s.]\d{4}\b")

RE_PHONE_FORMATTED = re.compile(
    r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
)

RE_DATE_NUMERIC = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b"
    r"|\b(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b"
)

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
RE_DATE_WORDY = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)?\d{{2}}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS}),?\s+(?:19|20)?\d{{2}}\b",
    re.IGNORECASE,
)

RE_ACCOUNT_LABELLED = re.compile(
    r"\b(?:acct|account|ac|a/c)\s*(?:no\.?|number|#)?\s*[-:#]?\s*\d[\d\s-]{4,}\d\b",
    re.IGNORECASE,
)

# Digit-run length rules, longest first so a card is not eaten as a phone.
CARD_LENGTHS = (19, 18, 17, 16, 15, 14, 13)
PHONE_LENGTHS = (11, 10)
SSN_LENGTH = 9
UNCLAIMED_MIN = 9  # backstop: any leftover run this long is redacted


@dataclass(frozen=True)
class Detection:
    """One detected PII span in the *original* text coordinates."""

    type: PIIType
    start: int
    end: int
    detector: str
    confidence: float
    canonical: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Detection) -> bool:
        return self.start < other.end and other.start < self.end


def _has_context(
    text: str, start: int, end: int, kind: PIIType, extra: str = ""
) -> bool:
    """Is there a keyword near this candidate that names its type?

    ``extra`` carries recent text from *previous* segments. It is consulted but
    never redacted: when a caller says "my social is" and then reads the digits
    after a pause, the keyword and the digits land in different segments, and
    without this the number would be redacted as a generic ACCOUNT instead of
    an SSN. The value is protected either way; the carried context is what
    makes the *label* right.
    """
    lo = max(0, start - CONTEXT_WINDOW)
    hi = min(len(text), end + CONTEXT_WINDOW)
    window = text[lo:hi].lower()
    if extra:
        window = f"{extra.lower()} {window}"
    return any(w in window for w in CONTEXT_WORDS.get(kind, ()))


# --------------------------------------------------------------------------
# Surface detectors
# --------------------------------------------------------------------------


def detect_emails(text: str) -> list[Detection]:
    out = [
        Detection(PIIType.EMAIL, m.start(), m.end(), "regex", 0.99, m.group(0))
        for m in RE_EMAIL.finditer(text)
    ]
    for m in RE_SPOKEN_EMAIL.finditer(text):
        # Require a plausible TLD word so "meet me at four dot thirty" is not an email.
        if re.search(r"\b(com|org|net|edu|gov|io|co)\b", m.group(0), re.IGNORECASE):
            out.append(
                Detection(PIIType.EMAIL, m.start(), m.end(), "context", 0.85, m.group(0))
            )
    return out


def detect_formatted(text: str, extra: str = "") -> list[Detection]:
    """Detectors that key off punctuation the speaker actually typed."""
    out: list[Detection] = []

    for m in RE_SSN_FORMATTED.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # A formatted 3-2-4 group is an SSN shape; context raises confidence but
        # is not required, because the shape alone is rare enough.
        conf = 0.97 if _has_context(text, m.start(), m.end(), PIIType.SSN, extra) else 0.8
        out.append(Detection(PIIType.SSN, m.start(), m.end(), "regex", conf, digits))

    for m in RE_PHONE_FORMATTED.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        out.append(Detection(PIIType.PHONE, m.start(), m.end(), "regex", 0.95, digits))

    for m in RE_DATE_NUMERIC.finditer(text):
        conf = 0.95 if _has_context(text, m.start(), m.end(), PIIType.DOB, extra) else 0.75
        out.append(Detection(PIIType.DOB, m.start(), m.end(), "regex", conf, m.group(0)))

    for m in RE_DATE_WORDY.finditer(text):
        conf = 0.93 if _has_context(text, m.start(), m.end(), PIIType.DOB, extra) else 0.7
        out.append(Detection(PIIType.DOB, m.start(), m.end(), "regex", conf, m.group(0)))

    for m in RE_ACCOUNT_LABELLED.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= 5:
            # Redact only the number, keeping the "account" label readable so
            # downstream intent classification still sees the context word.
            num = re.search(r"\d[\d\s-]*\d", m.group(0))
            s = m.start() + num.start()
            e = m.start() + num.end()
            out.append(Detection(PIIType.ACCOUNT, s, e, "regex", 0.9, digits))

    return out


# --------------------------------------------------------------------------
# Digit-stream detectors
# --------------------------------------------------------------------------


def _runs(stream: DigitStream) -> list[tuple[int, int]]:
    """Maximal index ranges of the stream uninterrupted by a hard break."""
    if not len(stream):
        return []
    out: list[tuple[int, int]] = []
    start = 0
    for b in sorted(stream.breaks):
        if b > start:
            out.append((start, b))
        start = b
    if start < len(stream):
        out.append((start, len(stream)))
    return out


def _classify_at(
    text: str, stream: DigitStream, pos: int, run_end: int, extra: str = ""
) -> Detection | None:
    """Try to claim a PII value starting at stream index ``pos``."""
    digits = stream.text
    avail = run_end - pos

    # --- CARD: longest first, checksum preferred, context as fallback -------
    for n in CARD_LENGTHS:
        if n > avail:
            continue
        cand = digits[pos : pos + n]
        s, e = stream.span(pos, pos + n)
        if luhn_valid(cand):
            return Detection(PIIType.CARD, s, e, "regex+checksum", 0.99, cand)

    for n in (16, 15):
        if n > avail:
            continue
        cand = digits[pos : pos + n]
        s, e = stream.span(pos, pos + n)
        if _has_context(text, s, e, PIIType.CARD, extra):
            # Luhn failed but everything else says card. ASR drops digits; a
            # real card with one misheard digit lands exactly here.
            return Detection(PIIType.CARD, s, e, "context", 0.72, cand)

    # A bare 16-digit run with no context and no valid checksum. Redacted
    # anyway under the recall bias; confidence marks it as the weakest tier.
    if avail >= 16:
        s, e = stream.span(pos, pos + 16)
        return Detection(PIIType.CARD, s, e, "context", 0.45, digits[pos : pos + 16])

    # --- SSN ---------------------------------------------------------------
    if avail >= SSN_LENGTH:
        cand = digits[pos : pos + SSN_LENGTH]
        s, e = stream.span(pos, pos + SSN_LENGTH)
        if _has_context(text, s, e, PIIType.SSN, extra):
            return Detection(PIIType.SSN, s, e, "context", 0.9, cand)

    # --- PHONE -------------------------------------------------------------
    for n in PHONE_LENGTHS:
        if n > avail:
            continue
        cand = digits[pos : pos + n]
        if n == 11 and cand[0] != "1":
            continue  # 11 digits only reads as a phone with a US country code
        s, e = stream.span(pos, pos + n)
        conf = 0.9 if _has_context(text, s, e, PIIType.PHONE, extra) else 0.6
        return Detection(PIIType.PHONE, s, e, "context", conf, cand)

    # --- DOB as a bare 8-digit run ----------------------------------------
    if avail >= 8:
        cand = digits[pos : pos + 8]
        s, e = stream.span(pos, pos + 8)
        if _has_context(text, s, e, PIIType.DOB, extra):
            return Detection(PIIType.DOB, s, e, "context", 0.8, cand)

    # --- ACCOUNT / backstop ------------------------------------------------
    if avail >= UNCLAIMED_MIN:
        n = min(avail, 12)
        s, e = stream.span(pos, pos + n)
        conf = 0.85 if _has_context(text, s, e, PIIType.ACCOUNT, extra) else 0.4
        return Detection(PIIType.ACCOUNT, s, e, "context", conf, digits[pos : pos + n])

    return None


def detect_digit_stream(
    text: str, stream: DigitStream | None = None, extra: str = ""
) -> list[Detection]:
    """Scan the normalised digit stream, claiming values left to right."""
    stream = stream if stream is not None else build_digit_stream(text)
    out: list[Detection] = []
    for run_start, run_end in _runs(stream):
        pos = run_start
        while pos < run_end:
            det = _classify_at(text, stream, pos, run_end, extra)
            if det is None:
                pos += 1
                continue
            out.append(det)
            # Advance past the digits this detection consumed.
            consumed = len(det.canonical) or 1
            pos += consumed
    return out


# --------------------------------------------------------------------------
# Overlap resolution
# --------------------------------------------------------------------------


def resolve(detections: list[Detection]) -> list[Detection]:
    """Drop overlapping detections, keeping the strongest claim on each span.

    Ranking: type priority, then longer span, then higher confidence. Longer
    wins before confidence so a full card beats a high-confidence phone number
    detected inside the card's own digits.
    """
    ordered = sorted(
        detections,
        key=lambda d: (TYPE_PRIORITY[d.type], -d.length, -d.confidence, d.start),
    )
    kept: list[Detection] = []
    for det in ordered:
        if any(det.overlaps(k) for k in kept):
            continue
        kept.append(det)
    return sorted(kept, key=lambda d: d.start)
