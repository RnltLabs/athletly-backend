"""Cache hit-rate and ITPM telemetry for LLM calls.

In-memory ring buffer of the last N calls. Computes:
- cache hit rate (cache_read / total_input) over a window
- rolling tokens per minute (input + cache_creation) over last 60s
- per-model breakdown

Threading: protected by a single Lock. Append is O(1).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CallRecord:
    """Immutable snapshot of a single LLM call's token usage."""

    ts: float
    model: str
    input_tokens: int  # uncached input
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def itpm_relevant(self) -> int:
        # Cached reads do not count toward ITPM for current Claude models (no dagger)
        return self.input_tokens + self.cache_creation_tokens


class CacheTelemetry:
    """Singleton telemetry collector. Use get_telemetry()."""

    def __init__(self, max_records: int = 200):
        self._records: deque[CallRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record_call(
        self,
        model: str,
        input_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        output_tokens: int,
    ) -> None:
        rec = CallRecord(
            ts=time.time(),
            model=model,
            input_tokens=input_tokens or 0,
            cache_creation_tokens=cache_creation_tokens or 0,
            cache_read_tokens=cache_read_tokens or 0,
            output_tokens=output_tokens or 0,
        )
        with self._lock:
            self._records.append(rec)

    def cache_hit_rate(self, window: int = 20) -> float:
        with self._lock:
            recent = list(self._records)[-window:]
        if not recent:
            return 0.0
        total_input = sum(r.total_input for r in recent)
        if total_input == 0:
            return 0.0
        cached = sum(r.cache_read_tokens for r in recent)
        return cached / total_input

    def itpm(self, seconds: int = 60) -> int:
        cutoff = time.time() - seconds
        with self._lock:
            recent = [r for r in self._records if r.ts >= cutoff]
        return sum(r.itpm_relevant for r in recent)

    def summary(self) -> dict:
        with self._lock:
            recent = list(self._records)[-50:]
        by_model: dict[str, dict[str, int]] = {}
        for r in recent:
            d = by_model.setdefault(r.model, {
                "calls": 0,
                "input_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "output_tokens": 0,
            })
            d["calls"] += 1
            d["input_tokens"] += r.input_tokens
            d["cache_creation_tokens"] += r.cache_creation_tokens
            d["cache_read_tokens"] += r.cache_read_tokens
            d["output_tokens"] += r.output_tokens
        return {
            "window_calls": len(recent),
            "cache_hit_rate_last_20": round(self.cache_hit_rate(20) * 100, 1),
            "itpm_last_60s": self.itpm(60),
            "itpm_last_300s_per_minute": self.itpm(300) // 5 if recent else 0,
            "by_model": by_model,
        }


_singleton: CacheTelemetry | None = None
_singleton_lock = threading.Lock()


def get_telemetry() -> CacheTelemetry:
    """Return the process-wide CacheTelemetry singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CacheTelemetry()
    return _singleton
