"""In-memory metrics for the deterministic response gates (Sprint D).

Mirrors :mod:`src.services.critic_metrics`: singleton, threading.Lock,
fixed-size ring buffer. Tracks per-gate counters for pass / fail /
gate_error / regenerate_success / regenerate_failed events plus a
rolling latency placeholder (gates are microsecond-fast so latency is
not tracked explicitly).

Use :func:`get_gates_metrics` to obtain the process-wide singleton.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

# The 5 gate ids. Kept in sync with ``src.agent.response_gates.GATES``;
# a module-level assert in that file enforces the agreement.
GATE_IDS: tuple[str, ...] = (
    "temporal_freshness",
    "injury_persistence",
    "stats_grounding",
    "holistic_alert",
    "language_mirror",
)

# Possible outcome values per record. Anything else is silently dropped
# so a typo in caller code cannot corrupt the buffer.
_VALID_OUTCOMES: frozenset[str] = frozenset(
    {
        "pass",
        "fail",
        "gate_error",
        "regenerate_success",
        "regenerate_failed",
    }
)


@dataclass(frozen=True)
class GateRecord:
    """Immutable snapshot of one gate outcome."""

    ts: float
    gate_id: str
    outcome: str


class GatesMetrics:
    """Singleton metrics collector. Use :func:`get_gates_metrics`."""

    def __init__(self, max_records: int = 1000) -> None:
        self._records: deque[GateRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, gate_id: str, outcome: str) -> None:
        """Append one gate outcome to the ring buffer.

        Silently drops anything with an unknown gate_id or outcome so a
        typo in caller code cannot corrupt the buffer. We do NOT raise:
        metrics must never break the agent loop.
        """
        if gate_id not in GATE_IDS:
            return
        if outcome not in _VALID_OUTCOMES:
            return
        rec = GateRecord(ts=time.time(), gate_id=gate_id, outcome=outcome)
        with self._lock:
            self._records.append(rec)

    def summary(self) -> dict:
        """Return a JSON-serialisable summary over the full ring buffer."""
        with self._lock:
            recent = list(self._records)

        n = len(recent)
        per_gate: dict[str, dict[str, int]] = {
            gid: {o: 0 for o in _VALID_OUTCOMES} for gid in GATE_IDS
        }
        total_fails = 0
        total_regen_success = 0
        total_regen_failed = 0
        for r in recent:
            bucket = per_gate.get(r.gate_id)
            if bucket is None:
                continue
            bucket[r.outcome] = bucket.get(r.outcome, 0) + 1
            if r.outcome == "fail":
                total_fails += 1
            elif r.outcome == "regenerate_success":
                total_regen_success += 1
            elif r.outcome == "regenerate_failed":
                total_regen_failed += 1

        # Derived rates: each gate's fail / pass ratio.
        rates: dict[str, dict[str, float]] = {}
        for gid, bucket in per_gate.items():
            passes = bucket.get("pass", 0)
            fails = bucket.get("fail", 0)
            total = passes + fails
            rates[gid] = {
                "fail_rate": round(fails / total, 4) if total else 0.0,
                "runs": total,
            }

        return {
            "window_records": n,
            "per_gate": per_gate,
            "rates": rates,
            "totals": {
                "fails": total_fails,
                "regenerate_success": total_regen_success,
                "regenerate_failed": total_regen_failed,
            },
        }

    def reset(self) -> None:
        """Drop all records. Used by tests."""
        with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# Singleton plumbing
# ---------------------------------------------------------------------------


_singleton: GatesMetrics | None = None
_singleton_lock = threading.Lock()


def get_gates_metrics() -> GatesMetrics:
    """Return the process-wide :class:`GatesMetrics` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = GatesMetrics()
    return _singleton


def reset_gates_metrics() -> None:
    """Reset the singleton. Used by tests."""
    global _singleton
    with _singleton_lock:
        _singleton = None
