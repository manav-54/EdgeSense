"""Synthetic PII value pools and the surface forms they can appear in.

Every value here is fabricated: cards are the public test-card numbers, phone
numbers use the 555 fictional exchange, emails use example.com, and SSNs use
area numbers that the SSA never issued. Nothing in this file corresponds to a
real person.

The point of ``SurfaceForm`` is that a corpus which only ever writes
``4242424242424242`` measures a regex, not a redactor. Real callers read card
numbers aloud in groups, spell them out, and get interrupted halfway through.
Each surface form is a different difficulty tier, and the eval harness reports
recall per (type, surface form) so a regression shows up as "we stopped
catching spelled-out cards" rather than a single blended number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

TEEN_WORDS = {
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen",
}

TENS_WORDS = {
    "2": "twenty", "3": "thirty", "4": "forty", "5": "fifty",
    "6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety",
}


class SurfaceForm(str, Enum):
    """How a PII value is rendered into dialogue text."""

    PLAIN = "plain"          # 4242424242424242
    SPACED = "spaced"        # 4242 4242 4242 4242
    HYPHENATED = "hyphenated"  # 4242-4242-4242-4242
    SPELLED = "spelled"      # four two four two ...
    PAIRED = "paired"        # forty-two forty-two ...
    NOISY = "noisy"          # 4242, uh, 4242 ... sorry, 4242 4242

    @property
    def is_adversarial(self) -> bool:
        return self in {SurfaceForm.SPELLED, SurfaceForm.PAIRED, SurfaceForm.NOISY}


def _group(digits: str, size: int, sep: str) -> str:
    return sep.join(digits[i : i + size] for i in range(0, len(digits), size))


def _spell_digits(digits: str) -> str:
    return " ".join(DIGIT_WORDS[d] for d in digits if d in DIGIT_WORDS)


def _spell_pairs(digits: str) -> str:
    """Render digits as two-digit words: 4242 -> 'forty two forty two'."""
    out: list[str] = []
    i = 0
    while i < len(digits):
        pair = digits[i : i + 2]
        if len(pair) == 2:
            if pair in TEEN_WORDS:
                out.append(TEEN_WORDS[pair])
            elif pair[1] == "0":
                out.append(TENS_WORDS.get(pair[0], DIGIT_WORDS[pair[0]]))
            elif pair[0] == "0":
                out.append("oh " + DIGIT_WORDS[pair[1]])
            else:
                out.append(f"{TENS_WORDS.get(pair[0], DIGIT_WORDS[pair[0]])} {DIGIT_WORDS[pair[1]]}")
            i += 2
        else:
            out.append(DIGIT_WORDS[pair])
            i += 1
    return " ".join(out)


def _noisy(digits: str, rng: random.Random) -> str:
    """Grouped digits with disfluencies wedged between the groups."""
    groups = [digits[i : i + 4] for i in range(0, len(digits), 4)]
    fillers = ["uh", "um", "hold on", "sorry", "let me see", "one sec"]
    out: list[str] = []
    for i, g in enumerate(groups):
        out.append(g)
        if i < len(groups) - 1 and rng.random() < 0.6:
            out.append(f", {rng.choice(fillers)},")
    return " ".join(out).replace(" ,", ",")


def render(value: str, form: SurfaceForm, rng: random.Random) -> str:
    """Render a canonical PII value in the requested surface form.

    Non-numeric values (emails, names) ignore numeric forms and render plain,
    since "spelled-out email" is not a thing callers do.
    """
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits or "@" in value:
        return value

    if form is SurfaceForm.PLAIN:
        return value
    if form is SurfaceForm.SPACED:
        return _group(digits, 4, " ") if len(digits) >= 8 else value.replace("-", " ")
    if form is SurfaceForm.HYPHENATED:
        return _group(digits, 4, "-") if len(digits) >= 8 else value
    if form is SurfaceForm.SPELLED:
        return _spell_digits(digits)
    if form is SurfaceForm.PAIRED:
        return _spell_pairs(digits)
    if form is SurfaceForm.NOISY:
        return _noisy(digits, rng)
    return value


@dataclass(frozen=True)
class Pool:
    """A named pool of fabricated values for one PII type."""

    kind: str
    values: tuple[str, ...]


# Public test card numbers. All Luhn-valid, none issuable.
CARDS = Pool("CARD", (
    "4242424242424242",   # Visa test
    "4000056655665556",   # Visa debit test
    "5555555555554444",   # Mastercard test
    "5200828282828210",   # Mastercard debit test
    "378282246310005",    # Amex test (15 digits)
    "6011111111111117",   # Discover test
))

# Deliberately Luhn-INVALID 16-digit strings. These exist to prove the
# checksum gate works: they look like cards, and a naive regex flags them.
# Recall-biased policy still redacts them, but they are tagged so the eval
# can report the over-redaction cost separately.
NON_CARDS = Pool("NON_CARD", (
    "4242424242424241",
    "1234567890123456",
    "9999888877776666",
))

# Area numbers 900+ and 666 were never issued by the SSA.
SSNS = Pool("SSN", (
    "900-45-6789", "901-23-4567", "666-12-3456", "912-88-7766", "987-65-4329",
))

# 555-01xx is the reserved fictional range.
PHONES = Pool("PHONE", (
    "415-555-0142", "212-555-0177", "650-555-0198", "312-555-0123", "206-555-0164",
))

EMAILS = Pool("EMAIL", (
    "j.calloway@example.com", "m.rivera@example.org", "dana.whitfield@example.net",
    "t.okonkwo@example.com", "priya.nair@example.org",
))

DOBS = Pool("DOB", (
    "03/14/1982", "11/02/1975", "07/29/1990", "12/05/1968", "01/17/1993",
))

ACCOUNTS = Pool("ACCOUNT", (
    "ACCT-4471902", "AC 88213366", "acct no. 55219084", "ACCT-1029384", "AC-77120945",
))

PERSONS = Pool("PERSON", (
    "Jordan Calloway", "Marisol Rivera", "Dana Whitfield",
    "Tobenna Okonkwo", "Priya Nair", "Aleksandr Volkov",
))

ADDRESSES = Pool("ADDRESS", (
    "1420 Marigold Lane, Apartment 3B",
    "88 Fenwick Road",
    "7301 Blackthorn Avenue, Unit 12",
))

POOLS: dict[str, Pool] = {
    "CARD": CARDS,
    "NON_CARD": NON_CARDS,
    "SSN": SSNS,
    "PHONE": PHONES,
    "EMAIL": EMAILS,
    "DOB": DOBS,
    "ACCOUNT": ACCOUNTS,
    "PERSON": PERSONS,
    "ADDRESS": ADDRESSES,
}


def pick(kind: str, rng: random.Random) -> str:
    """Deterministically pick a value for ``kind`` from its pool."""
    pool = POOLS.get(kind)
    if pool is None:
        raise KeyError(f"no fill pool for PII kind {kind!r}")
    return rng.choice(pool.values)
