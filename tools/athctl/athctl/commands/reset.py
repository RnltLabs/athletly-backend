"""Reset / teardown commands for athctl.

Each subcommand wipes a slice of the active test user's data so a fresh
scenario can be exercised without re-running ``athctl init`` (which
would mint a new user every time).

The cuts:

- ``reset profile``    : profiles + beliefs only
- ``reset plans``      : plans + pending_actions
- ``reset sessions``   : sessions + session_messages
- ``reset activities`` : activities + health_daily_metrics
- ``reset full``       : everything user-owned EXCEPT provider_tokens
- ``reset nuclear``    : everything user-owned INCLUDING provider_tokens

``provider_tokens`` are deliberately preserved by ``full`` because losing
the cloned Garmin session means going through password+MFA again on the
real account, which is painful.

All deletes use the service-role Supabase client (bypasses RLS).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import click

from athctl.common import (
    EXIT_BACKEND_DOWN,
    as_json,
    emit,
    get_supabase,
    require_active_user,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confirm_or_abort(message: str, skip_confirm: bool) -> bool:
    """Prompt for confirmation unless *skip_confirm*; True if we may proceed."""
    if skip_confirm:
        return True
    if not click.confirm(message, default=False):
        emit("Aborted.", data={"status": "aborted"})
        return False
    return True


def _delete_table(table: str, user_id: str, results: dict[str, Any]) -> None:
    """Delete all rows for *user_id* in *table*. Records outcome in *results*.

    Errors are caught and surfaced rather than aborting the whole reset:
    if one table fails (e.g. column missing on a fresh schema), the rest
    still run.
    """
    try:
        get_supabase().table(table).delete().eq("user_id", user_id).execute()
        results[table] = "ok"
    except Exception as exc:
        logger.warning("Failed to delete from %s: %s", table, exc)
        results[table] = f"error: {exc}"


def _delete_session_messages(user_id: str, results: dict[str, Any]) -> None:
    """Delete session_messages for *user_id* by joining through sessions.

    ``session_messages`` is keyed by ``session_id`` (not ``user_id``), so
    we resolve the user's session ids first and then delete by those.
    """
    try:
        client = get_supabase()
        sessions = (
            client.table("sessions")
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        session_ids = [row["id"] for row in (sessions.data or []) if row.get("id")]
        if not session_ids:
            results["session_messages"] = "ok (no sessions)"
            return
        client.table("session_messages").delete().in_("session_id", session_ids).execute()
        results["session_messages"] = f"ok ({len(session_ids)} sessions)"
    except Exception as exc:
        logger.warning("Failed to delete from session_messages: %s", exc)
        results["session_messages"] = f"error: {exc}"


def _emit_results(label: str, user_id: str, results: dict[str, Any]) -> None:
    """Emit a uniform summary for a reset operation."""
    failures = {k: v for k, v in results.items() if isinstance(v, str) and v.startswith("error")}
    payload = {
        "status": "ok" if not failures else "partial",
        "scope": label,
        "user_id": user_id,
        "tables": results,
    }
    if failures:
        emit(
            f"[yellow]Reset {label} completed with {len(failures)} failures:[/yellow]\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in failures.items()),
            data=payload,
        )
    else:
        deleted = ", ".join(sorted(results.keys()))
        emit(
            f"[green]Reset {label} done.[/green] tables=[{deleted}]",
            data=payload,
        )


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group("reset", help="Reset slices of the active user's data.")
def reset() -> None:
    """Reset operations."""


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@reset.command("profile", help="Wipe profile + beliefs for the active user.")
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation.")
def reset_profile(skip_confirm: bool) -> None:
    """Wipe the profile row and all beliefs for the active user."""
    uid = require_active_user()
    if not _confirm_or_abort(f"Wipe profile + beliefs for {uid}?", skip_confirm):
        return
    results: dict[str, Any] = {}
    _delete_table("beliefs", uid, results)
    _delete_table("profiles", uid, results)
    _emit_results("profile", uid, results)


@reset.command("plans", help="Wipe plans + pending_actions for the active user.")
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation.")
def reset_plans(skip_confirm: bool) -> None:
    """Wipe all plans and pending actions for the active user."""
    uid = require_active_user()
    if not _confirm_or_abort(f"Wipe plans + pending_actions for {uid}?", skip_confirm):
        return
    results: dict[str, Any] = {}
    _delete_table("pending_actions", uid, results)
    _delete_table("plans", uid, results)
    _emit_results("plans", uid, results)


@reset.command("sessions", help="Wipe sessions + session_messages for the active user.")
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation.")
def reset_sessions(skip_confirm: bool) -> None:
    """Wipe sessions and the messages within them for the active user."""
    uid = require_active_user()
    if not _confirm_or_abort(f"Wipe sessions + session_messages for {uid}?", skip_confirm):
        return
    results: dict[str, Any] = {}
    _delete_session_messages(uid, results)
    _delete_table("sessions", uid, results)
    _emit_results("sessions", uid, results)


@reset.command("activities", help="Wipe activities + health_daily_metrics for the active user.")
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation.")
def reset_activities(skip_confirm: bool) -> None:
    """Wipe synced activities and daily metrics."""
    uid = require_active_user()
    if not _confirm_or_abort(f"Wipe activities + health_daily_metrics for {uid}?", skip_confirm):
        return
    results: dict[str, Any] = {}
    _delete_table("activities", uid, results)
    _delete_table("health_daily_metrics", uid, results)
    _emit_results("activities", uid, results)


# Tables wiped by `reset full`. Order matters: dependents first.
FULL_TABLES_IN_ORDER: list[str] = [
    # message rows are FK'd to sessions, handled by _delete_session_messages.
    "pending_actions",
    "proactive_queue",
    "goal_trajectory_snapshots",
    "product_recommendations",
    "episodes",
    "plans",
    "activities",
    "health_daily_metrics",
    "beliefs",
    "sessions",
    "profiles",
]


@reset.command(
    "full",
    help="Wipe all user-owned data EXCEPT provider_tokens (keeps Garmin session).",
)
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation.")
def reset_full(skip_confirm: bool) -> None:
    """Wipe everything user-owned except provider_tokens.

    Use this between scenarios when you want a clean slate but still
    benefit from the cloned Garmin session (no fresh MFA dance).
    """
    uid = require_active_user()
    if not _confirm_or_abort(
        f"Wipe ALL user-owned data for {uid} except provider_tokens?",
        skip_confirm,
    ):
        return

    results: dict[str, Any] = {}
    # Messages first (FK on sessions).
    _delete_session_messages(uid, results)
    for table in FULL_TABLES_IN_ORDER:
        _delete_table(table, uid, results)
    _emit_results("full", uid, results)


@reset.command(
    "nuclear",
    help="Wipe EVERYTHING user-owned INCLUDING provider_tokens. Requires re-auth.",
)
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation.")
def reset_nuclear(skip_confirm: bool) -> None:
    """Wipe everything user-owned including provider_tokens."""
    uid = require_active_user()
    warning = (
        f"NUCLEAR reset for {uid}.\n"
        "After this you will need to re-auth Garmin (and re-import Strava "
        "bootstrap tokens). Proceed?"
    )
    if not _confirm_or_abort(warning, skip_confirm):
        return

    results: dict[str, Any] = {}
    _delete_session_messages(uid, results)
    for table in FULL_TABLES_IN_ORDER:
        _delete_table(table, uid, results)
    _delete_table("provider_tokens", uid, results)
    _emit_results("nuclear", uid, results)

    if not isinstance(results.get("provider_tokens"), str) or not results["provider_tokens"].startswith("error"):
        _ = as_json  # keep import live
