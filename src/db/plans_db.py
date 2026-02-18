"""Supabase-backed training plan storage.

Manages the ``plans`` table which stores versioned training plans with
an ``active`` flag so only one plan is current at any time.

Usage::

    from src.db.plans_db import store_plan, get_active_plan

    plan = store_plan(user_id, {"weeks": [...]}, evaluation_score=85)
    current = get_active_plan(user_id)
"""

from __future__ import annotations

import logging

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


def store_plan(
    user_id: str,
    plan_data: dict,
    evaluation_score: int | None = None,
    evaluation_feedback: str | None = None,
) -> dict:
    """Store a new training plan, deactivating any previous active plan.

    Args:
        user_id: UUID of the owning user.
        plan_data: The full plan payload (weeks, sessions, etc.).
        evaluation_score: Optional 0-100 quality score from the evaluator.
        evaluation_feedback: Optional textual feedback from the evaluator.

    Returns:
        The inserted row as a dict.
    """
    db = get_supabase()

    # Deactivate any currently-active plans for this user.
    db.table("plans").update({"active": False}).eq("user_id", user_id).eq(
        "active", True
    ).execute()

    row: dict = {
        "user_id": user_id,
        "plan_data": plan_data,
        "evaluation_score": evaluation_score,
        "evaluation_feedback": evaluation_feedback,
        "active": True,
    }

    result = db.table("plans").insert(row).execute()
    return result.data[0]


def get_active_plan(user_id: str) -> dict | None:
    """Get the current active plan for a user.

    Args:
        user_id: UUID of the owning user.

    Returns:
        Plan dict or ``None`` if no active plan exists.
    """
    result = (
        get_supabase()
        .table("plans")
        .select("*")
        .eq("user_id", user_id)
        .eq("active", True)
        .maybe_single()
        .execute()
    )
    # maybe_single().execute() returns None when no row is found
    return result.data if result is not None else None


def list_plans(user_id: str, limit: int = 10) -> list[dict]:
    """List all plans for a user (active and historical), newest first.

    Args:
        user_id: UUID of the owning user.
        limit: Maximum number of plans to return.

    Returns:
        List of plan dicts ordered by ``created_at`` descending.
    """
    result = (
        get_supabase()
        .table("plans")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def update_plan_evaluation(
    plan_id: str,
    evaluation_score: int,
    evaluation_feedback: str | None = None,
) -> dict | None:
    """Update the evaluation score and feedback on an existing plan.

    Args:
        plan_id: UUID of the plan to update.
        evaluation_score: 0-100 quality score.
        evaluation_feedback: Optional textual feedback.

    Returns:
        Updated plan dict or ``None`` if not found.
    """
    update_data: dict = {"evaluation_score": evaluation_score}
    if evaluation_feedback is not None:
        update_data["evaluation_feedback"] = evaluation_feedback

    result = (
        get_supabase()
        .table("plans")
        .update(update_data)
        .eq("id", plan_id)
        .execute()
    )
    return result.data[0] if result.data else None


def deactivate_plan(plan_id: str) -> None:
    """Mark a plan as inactive.

    Args:
        plan_id: UUID of the plan to deactivate.
    """
    get_supabase().table("plans").update({"active": False}).eq(
        "id", plan_id
    ).execute()
