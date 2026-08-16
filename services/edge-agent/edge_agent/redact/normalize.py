"""Spoken-number normalisation that preserves character alignment.

A redactor that only matches ``\\d{16}`` catches a card that was typed and
misses one that was spoken. ASR output is spoken: "four two four two...",
"forty two forty two...", "double four", "oh seven". Those are the same secret,
and a system that leaks them is not privacy-preserving in any useful sense.

So we build a *digit stream*: every digit the utterance contains, whether it
arrived as a numeral or as a word, in order. Detectors run against that stream,
and every digit carries the character range in the original text it came from.
When a detector matches ``stream[4:20]``, we can map straight back to the exact
substring to overwrite -- including any "uh" that was sitting between the digit
groups, which is exactly what should disappear.

The alternative -- normalising to a new string and redacting *that* -- would
mean shipping a rewritten transcript, destroying the readable text and any
chance of showing evidence in the portal. Alignment is what lets us redact in
place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN = re.compile(r"\d+|[A-Za-z']+|[^\w\s]|\s+")

UNITS = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0", "nil": "0",
    "one": "1", "won": "1",
    "two": "2", "to": None, "too": None,   # 'to'/'too' are too common to treat as 2
    "three": "3", "tree": "3",             # common ASR slip for 'three'
    "four": "4", "for": None, "fore": None,
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9",
}

TEENS = {
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19",
}

TENS = {
    "twenty": "2", "thirty": "3", "forty": "4", "fourty": "4", "fifty": "5",
    "sixty": "6", "seventy": "7", "eighty": "8", "ninety": "9",
}

REPEATERS = {"double": 2, "triple": 3, "treble": 3}

#: Words that may sit between digit groups without ending the number. Callers
#: pause, apologise and lose their place mid-readback; a hard break on the
#: first filler word would split every real spoken card in half.
MAX_GAP_WORDS = 5

#: Punctuation that always ends a number.
HARD_BREAK = frozenset({".", "?", "!", ";"})


@dataclass(frozen=True)
class Digit:
    """One digit in the stream and where it came from in the source text."""

    value: str
    start: int
    end: int
    spoken: bool  # True if it arrived as a word rather than a numeral


@dataclass(frozen=True)
class DigitStream:
    """All digits in an utterance, in order, with provenance.

    ``breaks`` holds stream indices at which a hard boundary precedes the
    digit, so a detector can refuse to span two unrelated numbers.
    """

    digits: tuple[Digit, ...]
    breaks: frozenset[int]

    @property
    def text(self) -> str:
        return "".join(d.value for d in self.digits)

    def __len__(self) -> int:
        return len(self.digits)

    def span(self, i: int, j: int) -> tuple[int, int]:
        """Character range in the original text covering stream slice ``[i:j)``."""
        if not 0 <= i < j <= len(self.digits):
            raise IndexError(f"bad stream slice [{i}:{j}) over {len(self.digits)} digits")
        return self.digits[i].start, self.digits[j - 1].end

    def crosses_break(self, i: int, j: int) -> bool:
        return any(i < b < j for b in self.breaks)

    def any_spoken(self, i: int, j: int) -> bool:
        return any(d.spoken for d in self.digits[i:j])


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(text)]


def build_digit_stream(text: str) -> DigitStream:
    """Extract the digit stream from ``text``, mapping each digit to its source."""
    tokens = [t for t in _tokenize(text) if not t[0].isspace()]
    digits: list[Digit] = []
    breaks: set[int] = set()
    gap_words = 0
    pending_repeat = 0

    def mark_break() -> None:
        if digits:
            breaks.add(len(digits))

    i = 0
    while i < len(tokens):
        tok, start, end = tokens[i]
        low = tok.lower()

        if tok.isdigit():
            for off, ch in enumerate(tok):
                digits.append(Digit(ch, start + off, start + off + 1, spoken=False))
            gap_words = 0
            pending_repeat = 0
            i += 1
            continue

        if low in REPEATERS:
            pending_repeat = REPEATERS[low]
            gap_words = 0
            i += 1
            continue

        # "forty two" -> 42, but a bare "forty" -> 40.
        if low in TENS:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            nxt_low = nxt[0].lower() if nxt else ""
            if nxt and UNITS.get(nxt_low) and UNITS[nxt_low] != "0":
                digits.append(Digit(TENS[low], start, end, spoken=True))
                digits.append(Digit(UNITS[nxt_low], nxt[1], nxt[2], spoken=True))
                i += 2
            else:
                digits.append(Digit(TENS[low], start, end, spoken=True))
                digits.append(Digit("0", start, end, spoken=True))
                i += 1
            gap_words = 0
            pending_repeat = 0
            continue

        if low in TEENS:
            for ch in TEENS[low]:
                digits.append(Digit(ch, start, end, spoken=True))
            gap_words = 0
            pending_repeat = 0
            i += 1
            continue

        if UNITS.get(low):
            value = UNITS[low]
            reps = pending_repeat or 1
            for _ in range(reps):
                digits.append(Digit(value, start, end, spoken=True))
            gap_words = 0
            pending_repeat = 0
            i += 1
            continue

        # Not a digit-bearing token.
        pending_repeat = 0
        if tok in HARD_BREAK:
            mark_break()
            gap_words = 0
        elif tok.isalpha():
            gap_words += 1
            if gap_words > MAX_GAP_WORDS:
                mark_break()
                gap_words = 0
        i += 1

    return DigitStream(tuple(digits), frozenset(breaks))


def luhn_valid(digits: str) -> bool:
    """Luhn checksum. Empty and non-numeric inputs are invalid, not exceptions."""
    if not digits or not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    parity = len(digits) % 2
    for idx, ch in enumerate(digits):
        d = ord(ch) - 48
        if idx % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
