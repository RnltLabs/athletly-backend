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
        critic_error / budget_skipped), per-rule violation counts over
        the last ~500 critic calls, and the average critic latency in
        milliseconds. ``budget_skipped`` is the count of chains that
        exceeded ``critic_total_budget_s`` before the second-pass LLM
        call and shipped the rewrite annotated as degraded.
    """
    return get_critic_metrics().summary()


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
