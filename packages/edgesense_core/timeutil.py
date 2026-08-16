"""Time helpers.

Two clocks, deliberately kept apart:

* Wall clock (``utc_now``) is what goes on the wire and into ClickHouse. It is
  comparable across hosts but can jump backwards under NTP correction.
* Monotonic (``monotonic_ms``) is what latency budgets are measured with. It
  never jumps, but is only meaningful within a single process.

Cross-process latency (edge -> signal) is therefore measured from wall-clock
timestamps and is only as good as clock sync between hosts; within a process
we always prefer the monotonic clock. See DESIGN.md, "Measuring latency".
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """RFC 3339 UTC timestamp with microsecond precision and a ``Z`` suffix."""
    return utc_now().isoformat(timespec="microseconds").replace("+00:00", "Z")


def monotonic_ms() -> float:
    """Monotonic milliseconds, for intra-process duration measurement."""
    return time.perf_counter() * 1000.0


def iso_to_datetime(value: str) -> datetime:
    """Parse an RFC 3339 timestamp, tolerating a trailing ``Z``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
