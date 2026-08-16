"""Turn written transcript text into something a TTS engine will speak like a human.

A written card number is ``4242424242424242``. Handed to any TTS engine that
is what comes out: "four quadrillion, two hundred forty-two trillion...".
Nobody reads a card number that way, and training the pipeline against audio
that says "quadrillion" would measure nothing useful.

So digit runs are re-spaced into individual digits with comma pauses every
four, which is how people actually read account and card numbers aloud. Short
runs (house numbers, version fragments) are left alone.

This transform applies to the *audio* only. The written ground truth in the
corpus is untouched, and the eval harness scores text-mode and audio-mode
separately for exactly this reason -- see EVAL.md, "Two measurement modes".
"""

from __future__ import annotations

import re

DIGIT_RUN = re.compile(r"\d{3,}")


def _spell_run(run: str) -> str:
    """``42424242`` -> ``4 2 4 2, 4 2 4 2`` (commas become prosodic pauses)."""
    groups = [run[i : i + 4] for i in range(0, len(run), 4)]
    return ", ".join(" ".join(g) for g in groups)


def to_speech(text: str) -> str:
    """Rewrite ``text`` so a TTS engine reads numbers the way a caller would."""
    out = DIGIT_RUN.sub(lambda m: _spell_run(m.group(0)), text)
    # Separators inside an identifier are pauses when spoken, not characters.
    out = out.replace("-", " ").replace("/", ", ")
    # "@" and "." in an email are spoken.
    out = re.sub(r"(\S)@(\S)", r"\1 at \2", out)
    out = re.sub(r"\.(com|org|net)\b", r" dot \1", out)
    return re.sub(r"\s{2,}", " ", out).strip()
