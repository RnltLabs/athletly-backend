"""Admin/internal endpoints for observability.

Read-only diagnostics over the in-memory LLM telemetry buffer. No PII,
no auth: a reverse proxy on Hetzner restricts access at the network layer.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.services.cache_telemetry import get_telemetry

router = APIRouter(tags=["admin"])


@router.get("/cache-stats")
async def cache_stats() -> dict:
    """Cache hit rate, ITPM, per-model breakdown over the last ~50 LLM calls."""
    return get_telemetry().summary()
