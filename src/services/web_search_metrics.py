"""In-memory metrics for the web_search tool.

Tracks per-window counters for:
- total_calls
- cache hits vs fresh API calls (hit rate)
- API errors

Plus a snapshot of the top 10 most-cached queries by hit_count, which is
read on demand from the Postgres cache table.

Singleton, thread-safe (one Lock), bounded ring buffer. Mirrors the
shape of ``src/services/cache_telemetry.py`` and
``src/services/critic_metrics.py`` so the admin endpoints stay
consistent.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchCallRecord:
    """Immutable snapshot of one web_search call.

    Source is one of: "fresh", "cache", "error".
    """

    ts: float
    source: str
    latency_ms: int


class WebSearchMetrics:
    """Singleton metrics collector. Use ``get_web_search_metrics()``."""

    def __init__(self, max_records: int = 500) -> None:
        self._records: deque[SearchCallRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, source: str, latency_ms: int) -> None:
        """Append a new web_search call to the ring buffer.

        Args:
            source: one of "fresh", "cache", "error". Anything else is
                rejected as a typo guard.
            latency_ms: wall-clock duration of the call.
        """
        if source not in {"fresh", "cache", "error"}:
            return
        rec = SearchCallRecord(
            ts=time.time(),
            source=source,
            latency_ms=max(0, int(latency_ms)),
        )
        with self._lock:
            self._records.append(rec)

    def summary(self, window_seconds: int | None = None) -> dict:
        """Return a JSON-serializable summary of metrics over a time window.

        Args:
            window_seconds: How far back to aggregate. ``None`` means all
                records currently in the ring buffer.

        Returns:
            ``{"window_calls": int, "total_calls": int,
                "cache_hit_rate": float, "api_error_rate": float,
                "fresh_calls": int, "cache_hits": int,
                "api_errors": int, "avg_latency_ms": int}``
        """
        with self._lock:
            if window_seconds is None:
                recent = list(self._records)
            else:
                cutoff = time.time() - window_seconds
                recent = [r for r in self._records if r.ts >= cutoff]

        n = len(recent)
        if n == 0:
            return {
                "window_calls": 0,
                "total_calls": 0,
                "cache_hit_rate": 0.0,
                "api_error_rate": 0.0,
                "fresh_calls": 0,
                "cache_hits": 0,
                "api_errors": 0,
                "avg_latency_ms": 0,
            }

        fresh = sum(1 for r in recent if r.source == "fresh")
        cache = sum(1 for r in recent if r.source == "cache")
        errors = sum(1 for r in recent if r.source == "error")
        total_latency = sum(r.latency_ms for r in recent)

        # Cache hit rate excludes errors from the denominator: it measures
        # "of the successful lookups, how many were served from cache".
        successful = fresh + cache
        cache_hit_rate = (cache / successful) if successful > 0 else 0.0

        return {
            "window_calls": n,
            "total_calls": n,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "api_error_rate": round(errors / n, 4),
            "fresh_calls": fresh,
            "cache_hits": cache,
            "api_errors": errors,
            "avg_latency_ms": total_latency // n,
        }

    def reset(self) -> None:
        """Drop all records. Used by tests."""
        with self._lock:
            self._records.clear()


_singleton: WebSearchMetrics | None = None
_singleton_lock = threading.Lock()


def get_web_search_metrics() -> WebSearchMetrics:
    """Return the process-wide WebSearchMetrics singleton."""
    global _singleton  # noqa: PLW0603 -- intentional singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = WebSearchMetrics()
    return _singleton


def get_top_queries(limit: int = 10) -> list[dict]:
    """Read the top-N most-cached queries by hit_count from Postgres.

    Returns an empty list when Supabase is not configured or the table is
    unavailable. Never raises: a failure here must not break the admin
    endpoint.

    Args:
        limit: How many rows to return. Bounded to [1, 50].

    Returns:
        A list of ``{"query": str, "hit_count": int,
        "num_results": int, "freshness": str, "created_at": str,
        "last_hit_at": str}`` rows ordered by hit_count desc.
    """
    bounded = max(1, min(int(limit), 50))
    try:
        from src.db.client import get_supabase
    except Exception:
        return []

    try:
        client = get_supabase()
    except Exception as exc:
        logger.debug("web_search_metrics: supabase unavailable: %s", exc)
        return []

    try:
        resp = (
            client.table("web_search_cache")
            .select("query, hit_count, num_results, freshness, created_at, last_hit_at")
            .order("hit_count", desc=True)
            .limit(bounded)
            .execute()
        )
    except Exception as exc:
        logger.warning("web_search_metrics: top_queries fetch failed: %s", exc)
        return []

    data = getattr(resp, "data", None) or []
    if not isinstance(data, list):
        return []
    return [
        {
            "query": row.get("query", ""),
            "hit_count": int(row.get("hit_count") or 0),
            "num_results": int(row.get("num_results") or 0),
            "freshness": row.get("freshness") or "any",
            "created_at": row.get("created_at"),
            "last_hit_at": row.get("last_hit_at"),
        }
        for row in data
    ]
