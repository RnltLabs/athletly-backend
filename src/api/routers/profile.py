"""Profile router: lifecycle operations on the authenticated user's data.

Currently exposes a single destructive operation:

    POST /profile/reset

This wipes every row owned by the authenticated user across all
user-scoped tables. The auth row in ``auth.users`` is preserved so the
user stays signed in and can simply start a fresh onboarding.

Tables touched are aligned with what ``athctl reset full|nuclear`` removes.
Provider tokens are removed too (Garmin disconnects) - the athlete must
reconnect after a reset.

This endpoint is destructive. Frontend MUST gate it behind a confirmation
dialog.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_user_id
from src.db.client import get_supabase

router = APIRouter()
logger = logging.getLogger(__name__)


# Tables keyed by user_id that get wiped during reset.
# Order is from leaf to root (dependents first) where there are FKs.
USER_TABLES_IN_ORDER: list[str] = [
    "pending_actions",
    "proactive_queue",
    "product_recommendations",
    "episodes",
    "plans",
    "activities",
    "health_daily_metrics",
    "athlete_journal",
    "provider_tokens",
    "profiles",
]


def _wipe_session_messages(user_id: str, results: dict[str, str]) -> None:
    """Delete session_messages by joining through sessions for the user."""
    try:
        client = get_supabase()
        sessions = (
            client.table("sessions").select("id").eq("user_id", user_id).execute()
        )
        session_ids = [row["id"] for row in (sessions.data or []) if row.get("id")]
        if not session_ids:
            results["session_messages"] = "ok (no sessions)"
            return
        client.table("session_messages").delete().in_(
            "session_id", session_ids
        ).execute()
        results["session_messages"] = f"ok ({len(session_ids)} sessions)"
    except Exception as exc:
        logger.warning("session_messages wipe failed for %s: %s", user_id, exc)
        results["session_messages"] = f"error: {exc}"


def _wipe_table(table: str, user_id: str, results: dict[str, str]) -> None:
    """Best-effort delete of all rows for *user_id* in *table*."""
    try:
        get_supabase().table(table).delete().eq("user_id", user_id).execute()
        results[table] = "ok"
    except Exception as exc:
        # A missing table or schema-drift here should not abort the whole
        # reset - record the failure and keep going. The client just sees
        # which tables succeeded vs failed.
        logger.warning("Wipe %s for %s failed: %s", table, user_id, exc)
        results[table] = f"error: {exc}"


@router.post("/reset", status_code=200)
async def reset_profile(
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Wipe every user-owned row except the auth identity itself.

    Returns a per-table summary so the frontend can surface partial
    failures (e.g. a renamed table on a fresh schema) without claiming
    success.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    logger.info("Profile reset requested by user=%s", user_id)
    results: dict[str, str] = {}

    # session_messages first (FK on sessions), then sessions.
    _wipe_session_messages(user_id, results)
    _wipe_table("sessions", user_id, results)

    for table in USER_TABLES_IN_ORDER:
        _wipe_table(table, user_id, results)

    failures = [k for k, v in results.items() if v.startswith("error")]
    status = "ok" if not failures else "partial"
    logger.info(
        "Profile reset complete user=%s status=%s failures=%d",
        user_id, status, len(failures),
    )

    return {"status": status, "user_id": user_id, "tables": results}
