"""Admin/internal endpoints for observability.

Read-only diagnostics over the in-memory LLM telemetry buffers. No PII,
no auth: a reverse proxy on Hetzner restricts access at the network
layer.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.services.cache_telemetry import get_telemetry
from src.services.critic_metrics import get_metrics as get_critic_metrics

router = APIRouter(tags=["admin"])


@router.get("/cache-stats")
async def cache_stats() -> dict:
    """Cache hit rate, ITPM, per-model breakdown over the last ~50 LLM calls."""
    return get_telemetry().summary()


@router.get("/critic-stats")
async def critic_stats() -> dict:
    """Per-action and per-rule constitutional critic stats.

    Returns:
        Dict with action rates (accept / regenerate / regenerate_failed /
        critic_error), per-rule violation counts over the last ~500
        critic calls, and the average critic latency in milliseconds.
    """
    return get_critic_metrics().summary()
