"""Shared contracts and utilities for EdgeSense services.

This package is the single source of truth for everything that crosses a
process boundary. Services import it rather than redefining payload shapes,
so a schema change is one edit and a version bump.
"""

from edgesense_core.contracts import (
    CONTRACT_VERSION,
    ActionItem,
    CallInsights,
    CallSummary,
    EvidenceSpan,
    PIIType,
    RedactionRef,
    ResolutionStatus,
    Signal,
    SignalType,
    Speaker,
    StageLatency,
    TranscriptSegment,
)
from edgesense_core.timeutil import monotonic_ms, utc_now, utc_now_iso

__all__ = [
    "CONTRACT_VERSION",
    "ActionItem",
    "CallInsights",
    "CallSummary",
    "EvidenceSpan",
    "PIIType",
    "RedactionRef",
    "ResolutionStatus",
    "Signal",
    "SignalType",
    "Speaker",
    "StageLatency",
    "TranscriptSegment",
    "monotonic_ms",
    "utc_now",
    "utc_now_iso",
]
