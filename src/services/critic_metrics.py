"""In-memory metrics for the constitutional critic (Feature 2).

Tracks per-action counters (accept / regenerate / regenerate_failed /
critic_error), per-rule violation counts, and an exponentially decaying
average of critic latency. Singleton, thread-safe (one lock), bounded by
a fixed-size ring buffer.

This mirrors the shape of `src/services/cache_telemetry.py` so the
admin endpoints stay consistent.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

# The STRICT rule ids the critic checks. Kept in sync with the
# constitution table in DESIGN.md and src/agent/critic.py.
RULE_IDS: tuple[str, ...] = (
    "no_em_dash",
    "no_markdown",
    "umlauts",
    "no_fabricated_stats",
    "no_premature_trends",
    "language_mirror",
    "details_before_metrics",
    "sync_then_status",
    "pace_format_correct",
    "pace_comparison_directional",
)


@dataclass(frozen=True)
class CritiqueRecord:
    """Immutable snapshot of one critic decision."""

    ts: float
    action: str  # "accept" | "regenerate" | "regenerate_failed" | "critic_error"
    violations: tuple[str, ...]  # rule ids
    latency_ms: int


class CriticMetrics:
    """Singleton metrics collector. Use ``get_metrics()``."""

    def __init__(self, max_records: int = 500) -> None:
        self._records: deque[CritiqueRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(
        self,
        action: str,
        violations: tuple[str, ...],
        latency_ms: int,
    ) -> None:
        """Append a new critic decision to the ring buffer.

        Args:
            action: one of "accept", "regenerate", "regenerate_failed",
                "critic_error". Anything else is rejected (typo guard).
            violations: tuple of rule ids that fired (empty for accept
                and critic_error).
            latency_ms: wall-clock duration of the critic LLM call.
        """
        if action not in {"accept", "regenerate", "regenerate_failed", "critic_error"}:
            # Defensive: never let metrics corruption break the agent.
            return
        rec = CritiqueRecord(
            ts=time.time(),
            action=action,
            violations=tuple(violations),
            latency_ms=max(0, int(latency_ms)),
        )
        with self._lock:
            self._records.append(rec)

    def summary(self) -> dict:
        """Return a JSON-serializable summary of the metrics buffer."""
        with self._lock:
            recent = list(self._records)

        n = len(recent)
        if n == 0:
            return {
                "window_calls": 0,
                "accept_rate": 0.0,
                "regenerate_rate": 0.0,
                "regenerate_failed_rate": 0.0,
                "critic_error_rate": 0.0,
                "by_rule": {rid: 0 for rid in RULE_IDS},
                "avg_critic_latency_ms": 0,
            }

        # Per-action counters.
        counts = {"accept": 0, "regenerate": 0, "regenerate_failed": 0, "critic_error": 0}
        by_rule: dict[str, int] = {rid: 0 for rid in RULE_IDS}
        total_latency = 0

        for r in recent:
            counts[r.action] = counts.get(r.action, 0) + 1
            for v in r.violations:
                if v in by_rule:
                    by_rule[v] += 1
            total_latency += r.latency_ms

        return {
            "window_calls": n,
            "accept_rate": round(counts["accept"] / n, 4),
            "regenerate_rate": round(counts["regenerate"] / n, 4),
            "regenerate_failed_rate": round(counts["regenerate_failed"] / n, 4),
            "critic_error_rate": round(counts["critic_error"] / n, 4),
            "by_rule": by_rule,
            "avg_critic_latency_ms": total_latency // n,
        }

    def reset(self) -> None:
        """Drop all records. Used by tests."""
        with self._lock:
            self._records.clear()


_singleton: CriticMetrics | None = None
_singleton_lock = threading.Lock()


def get_metrics() -> CriticMetrics:
    """Return the process-wide CriticMetrics singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CriticMetrics()
    return _singleton
