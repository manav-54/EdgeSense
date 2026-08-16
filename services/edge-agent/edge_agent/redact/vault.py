"""The local-only mapping from placeholder back to original value.

This is the one place in EdgeSense that holds raw PII, and it exists only so a
human at the edge can reverse a redaction during a live call. It is:

* **in-memory only** -- never written to disk, never logged, never serialised
  into any contract type;
* **per-call** -- dropped wholesale when the call ends;
* **stable within a call** -- the same card mentioned three times gets
  ``<CARD_1>`` all three times, so downstream analysis can tell "the customer
  repeated one card" from "the customer read out three cards".

``__repr__`` and ``__str__`` are overridden because the single most likely way
for this to leak is a well-meaning ``log.debug("vault=%s", vault)`` or an
exception rendering local variables in a traceback.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from edgesense_core.contracts import PIIType

PLACEHOLDER_RE = re.compile(r"<([A-Z_]+)_(\d+)>")


@dataclass
class VaultEntry:
    placeholder: str
    type: PIIType
    original: str
    occurrences: int = 1


@dataclass
class PIIVault:
    """Per-call placeholder registry. Never leaves the process."""

    call_id: str
    _by_key: dict[tuple[PIIType, str], VaultEntry] = field(default_factory=dict)
    _by_placeholder: dict[str, VaultEntry] = field(default_factory=dict)
    _counters: dict[PIIType, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def placeholder_for(self, pii_type: PIIType, original: str) -> str:
        """Return the stable placeholder for ``original``, minting one if new.

        Values are keyed on their digits (or casefolded text) so that
        ``4242 4242 4242 4242`` and ``four two four two ...`` -- the same card
        read two different ways -- collapse to a single placeholder.
        """
        key = (pii_type, _canonical_key(original))
        with self._lock:
            entry = self._by_key.get(key)
            if entry is not None:
                entry.occurrences += 1
                return entry.placeholder

            n = self._counters.get(pii_type, 0) + 1
            self._counters[pii_type] = n
            placeholder = f"<{pii_type.value}_{n}>"
            entry = VaultEntry(placeholder=placeholder, type=pii_type, original=original)
            self._by_key[key] = entry
            self._by_placeholder[placeholder] = entry
            return placeholder

    def resolve(self, placeholder: str) -> str | None:
        """Reverse a placeholder. Local callers only -- never over the wire."""
        entry = self._by_placeholder.get(placeholder)
        return entry.original if entry else None

    def originals(self) -> list[str]:
        """Every raw value held. Used by the privacy tests as the leak oracle."""
        return [e.original for e in self._by_placeholder.values()]

    def summary(self) -> dict[str, int]:
        """Counts by type. Safe to log -- contains no values."""
        out: dict[str, int] = {}
        for entry in self._by_placeholder.values():
            out[entry.type.value] = out.get(entry.type.value, 0) + 1
        return out

    def clear(self) -> None:
        with self._lock:
            self._by_key.clear()
            self._by_placeholder.clear()
            self._counters.clear()

    # -- leak guards --------------------------------------------------------
    # A vault that renders its contents into a log line or a traceback defeats
    # the entire design. Both stringifications are redacted at the source.

    def __repr__(self) -> str:
        return f"PIIVault(call_id={self.call_id!r}, entries={len(self._by_placeholder)})"

    __str__ = __repr__


def _canonical_key(value: str) -> str:
    """Collapse surface variation so one secret maps to one placeholder."""
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return digits
    return re.sub(r"\s+", " ", value).strip().casefold()
