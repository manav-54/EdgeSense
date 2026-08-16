"""The leak oracle used by the egress tests.

Searching the wire for the literal string is not enough. A card that arrived
as ``4242 4242 4242 4242`` and left as ``4242424242424242`` is still a leak,
and so is one that left as ``4-2-4-2...`` or spread across two JSON fields.
So every probe is checked in three spaces:

1. the raw wire bytes, for the surface form exactly as spoken;
2. the *digit projection* of the wire -- every digit on the wire concatenated
   in order, with separators removed -- for any reformatted variant;
3. the spoken-word projection, for a value re-emitted as digit words.

A probe is a leak if it appears in any of the three. This deliberately errs
towards false alarms: a test that cries wolf costs an engineer ten minutes,
and a test that stays quiet costs a disclosure notification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

#: Digit sequences shorter than this are not treated as secrets on their own:
#: a lone "42" is not a card, and demanding it never appear would make the
#: test meaningless. Four digits is the industry line (last-four is
#: routinely disclosed), so we require five before calling it a leak.
MIN_LEAK_DIGITS = 5


@dataclass(frozen=True)
class Leak:
    call_id: str
    pii_type: str
    probe: str
    space: str  # raw | digits | spoken
    excerpt: str

    def __str__(self) -> str:
        return (
            f"{self.call_id}: {self.pii_type} leaked via {self.space} space "
            f"(probe={self.probe!r}) near ...{self.excerpt}..."
        )


def digit_projection(text: str) -> str:
    """Every digit in ``text``, in order, separators removed."""
    return re.sub(r"\D", "", text)


def spoken_projection(text: str) -> str:
    """Lowercased text with punctuation collapsed, for digit-word matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def as_digit_words(digits: str) -> str:
    return " ".join(DIGIT_WORDS[d] for d in digits if d in DIGIT_WORDS)


def probes_for(value: str, pii_type: str) -> list[tuple[str, str]]:
    """Return ``(space, probe)`` pairs that would each prove a leak of ``value``."""
    out: list[tuple[str, str]] = []
    digits = digit_projection(value)

    if len(digits) >= MIN_LEAK_DIGITS:
        out.append(("digits", digits))
        out.append(("spoken", as_digit_words(digits)))

    if "@" in value:
        out.append(("raw", value))
        local = value.split("@", 1)[0]
        if len(local) >= 4:
            out.append(("raw", local))
    elif not digits and len(value.strip()) >= 5:
        # Names and addresses: match the full surface form only. Matching a
        # single token would fire on ordinary words.
        out.append(("raw", value.strip()))
    elif digits and len(digits) < MIN_LEAK_DIGITS:
        # Too short to assert on; skipped rather than silently passed.
        pass

    return out


def find_leaks(call_id: str, wire: str, expected_pii: list[dict]) -> list[Leak]:
    """Check every known PII value of a call against everything that was sent."""
    raw = wire
    digits = digit_projection(wire)
    spoken = spoken_projection(wire)

    leaks: list[Leak] = []
    for span in expected_pii:
        value = span.get("canonical") or span.get("value", "")
        if not value:
            continue
        for space, probe in probes_for(value, span["type"]):
            if not probe:
                continue
            haystack = {"raw": raw, "digits": digits, "spoken": spoken}[space]
            idx = haystack.find(probe)
            if idx >= 0:
                leaks.append(
                    Leak(
                        call_id=call_id,
                        pii_type=span["type"],
                        probe=probe,
                        space=space,
                        excerpt=haystack[max(0, idx - 40) : idx + len(probe) + 40],
                    )
                )
    return leaks


def long_digit_runs(text: str, minimum: int = 7) -> list[str]:
    """Digit runs long enough to be a secret, as a model-agnostic backstop.

    The corpus-driven check above only knows about values it was told to look
    for. This one asserts the shape invariant instead: after redaction, nothing
    resembling an account, card, or phone number should remain, whatever the
    ASR actually heard.
    """
    # Scanned over the text itself, not its digit projection: projecting would
    # glue unrelated numbers together and invent runs that were never sent.
    return [m.group(0) for m in re.finditer(rf"\d{{{minimum},}}", text)]
