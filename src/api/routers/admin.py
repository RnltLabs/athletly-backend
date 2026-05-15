"""Admin/internal endpoints for observability.

Read-only diagnostics over the in-memory LLM telemetry buffers. No PII,
no auth: a reverse proxy on Hetzner restricts access at the network
layer.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.services.cache_telemetry import get_telemetry
from src.services.critic_metrics import get_metrics as get_critic_metrics
from src.services.gates_metrics import get_gates_metrics
from src.services.prompt_metrics import get_prompt_metrics
from src.services.web_search_metrics import (
    get_top_queries,
    get_web_search_metrics,
)

router = APIRouter(tags=["admin"])


@router.get("/cache-stats")
async def cache_stats() -> dict:
    """Cache hit rate, ITPM, per-model breakdown over the last ~50 LLM calls."""
    return get_telemetry().summary()


@router.get("/prompt-metrics")
async def prompt_metrics(
    window_minutes: int = Query(60, ge=1, le=1440),
) -> dict:
    """STRICT rule violation telemetry over a rolling time window.

    Args:
        window_minutes: How far back to aggregate. 1 to 1440 minutes
            (24h). Defaults to 60.

    Returns counts per rule, per model, per prompt variant, plus the
    rolling violation rate and a small sample of recent violations
    (with snippets) for inspection.
    """
    return get_prompt_metrics().summary(window_seconds=window_minutes * 60)


@router.get("/critic-stats")
async def critic_stats() -> dict:
    """Per-action and per-rule constitutional critic stats.

    Returns:
        Dict with action rates (accept / regenerate / regenerate_failed /
        critic_error), per-rule violation counts over the last ~500
        critic calls, and the average critic latency in milliseconds.
    """
    return get_critic_metrics().summary()


@router.get("/web-search-stats")
async def web_search_stats(
    window_minutes: int = Query(60, ge=1, le=1440),
) -> dict:
    """web_search tool telemetry over a rolling time window.

    Args:
        window_minutes: How far back to aggregate. 1 to 1440 minutes
            (24h). Defaults to 60.

    Returns:
        Dict with total_calls, cache_hit_rate, api_errors, avg_latency
        from the in-memory ring buffer over the requested window,
        plus the top 10 most-cached queries by hit_count read from
        Postgres.
    """
    summary = get_web_search_metrics().summary(window_seconds=window_minutes * 60)
    summary["top_queries"] = get_top_queries(limit=10)
    return summary


@router.get("/gates-stats")
async def gates_stats() -> dict:
    """Per-gate counters for the deterministic response-gate layer (Sprint D).

    Returns:
        Dict with per-gate outcome counts (pass / fail / gate_error /
        regenerate_success / regenerate_failed), derived per-gate fail
        rates, and aggregate regenerate counters over the last ~1000
        gate runs. See ``src/services/gates_metrics.py``.
    """
    return get_gates_metrics().summary()
