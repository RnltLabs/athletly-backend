"""State inspection commands for athctl.

Each subcommand pulls a slice of the active test user's persisted state
out of Supabase and renders it as a Rich table (or JSON via --json).
Nothing here writes; this is read-only diagnostic tooling.
"""

from __future__ import annotations

import logging
import sys

import click
from rich.table import Table

from athctl.common import EXIT_BACKEND_DOWN, console, emit, get_supabase, require_active_user

logger = logging.getLogger(__name__)


@click.group(help="Inspect persisted state for the active test user.")
def state() -> None:
    """State inspection."""


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@state.command("profile", help="Show the active user's profile row.")
def profile() -> None:
    uid = require_active_user()
    client = get_supabase()
    res = client.table("profiles").select("*").eq("user_id", uid).execute()
    if not res.data:
        emit("No profile row.", data={"user_id": uid, "profile": None})
        return
    row = res.data[0]
    emit("Profile:", data={"user_id": uid, "profile": row})

    table = Table(show_header=False, box=None)
    table.add_column("field", style="bold")
    table.add_column("value")
    for key in ("name", "sports", "goal", "constraints", "fitness", "preferences"):
        if key in row:
            table.add_row(key, str(row.get(key)))
    console.print(table)


# ---------------------------------------------------------------------------
# beliefs
# ---------------------------------------------------------------------------


@state.command("beliefs", help="List beliefs sorted by confidence.")
@click.option("--min-confidence", type=float, default=0.0, help="Filter threshold.")
@click.option("--category", default=None, help="Filter by category.")
def beliefs(min_confidence: float, category: str | None) -> None:
    uid = require_active_user()
    client = get_supabase()
    q = client.table("beliefs").select("*").eq("user_id", uid)
    if category:
        q = q.eq("category", category)
    if min_confidence > 0:
        q = q.gte("confidence", min_confidence)
    res = q.order("confidence", desc=True).execute()
    rows = res.data or []
    emit(f"{len(rows)} beliefs", data={"user_id": uid, "beliefs": rows})

    table = Table()
    table.add_column("conf", justify="right")
    table.add_column("category", style="cyan")
    table.add_column("text", overflow="fold")
    table.add_column("source", style="dim")
    for r in rows:
        table.add_row(
            f"{r.get('confidence', 0):.2f}",
            str(r.get("category", "")),
            str(r.get("text", ""))[:120],
            str(r.get("source", "")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@state.command("plan", help="Show the active plan (or a specific week with --week).")
@click.option("--week", type=int, default=None, help="Restrict to a single week.")
def plan(week: int | None) -> None:
    uid = require_active_user()
    try:
        from src.db.plans_db import get_active_plan
    except Exception as exc:
        print(f"plans_db unavailable: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    p = get_active_plan(uid)
    if not p:
        emit("No active plan.", data={"user_id": uid, "plan": None})
        return

    plan_data = p.get("plan_data") or {}
    emit(
        f"Active plan {p.get('id')} ({len(plan_data.get('weeks', []))} weeks)",
        data={"user_id": uid, "plan": p},
    )

    weeks = plan_data.get("weeks") or []
    for w in weeks:
        wn = w.get("week_num")
        if week is not None and wn != week:
            continue
        console.print(f"\n[bold]Week {wn}[/bold]")
        table = Table()
        table.add_column("type", style="cyan")
        table.add_column("duration")
        table.add_column("distance")
        table.add_column("zones")
        for s in w.get("sessions") or []:
            table.add_row(
                str(s.get("session_type", "")),
                f"{s.get('duration_minutes', '?')}min",
                f"{s.get('distance_km', '?')}km",
                str(s.get("intensity_zones", "")),
            )
        console.print(table)


# ---------------------------------------------------------------------------
# activities
# ---------------------------------------------------------------------------


@state.command("activities", help="List recent activities.")
@click.option("--last", "limit", type=int, default=20, help="How many to show.")
@click.option("--sport", default=None, help="Filter by sport.")
def activities(limit: int, sport: str | None) -> None:
    uid = require_active_user()
    client = get_supabase()
    q = client.table("activities").select(
        "id, sport, start_time, duration_seconds, distance_meters, avg_hr, source"
    ).eq("user_id", uid)
    if sport:
        q = q.eq("sport", sport)
    res = q.order("start_time", desc=True).limit(limit).execute()
    rows = res.data or []
    emit(f"{len(rows)} activities", data={"user_id": uid, "activities": rows})

    table = Table()
    table.add_column("date")
    table.add_column("sport", style="cyan")
    table.add_column("duration", justify="right")
    table.add_column("distance", justify="right")
    table.add_column("avg_hr", justify="right")
    table.add_column("source", style="dim")
    for r in rows:
        dur = r.get("duration_seconds") or 0
        dist = r.get("distance_meters") or 0
        table.add_row(
            str(r.get("start_time", ""))[:16],
            str(r.get("sport", "")),
            f"{dur // 60}min",
            f"{dist / 1000:.1f}km" if dist else "-",
            str(r.get("avg_hr") or "-"),
            str(r.get("source", "")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


@state.command("session", help="Show a chat session and its messages.")
@click.option("--id", "session_id", default=None, help="Specific session id.")
@click.option("--latest", is_flag=True, help="Use the most recent session.")
@click.option("--full", is_flag=True, help="Show full message content (no preview).")
def session(session_id: str | None, latest: bool, full: bool) -> None:
    uid = require_active_user()
    client = get_supabase()

    if not session_id:
        if latest:
            res = client.table("sessions").select("*").eq("user_id", uid).order(
                "started_at", desc=True
            ).limit(1).execute()
            if not res.data:
                emit("No sessions.", data={"user_id": uid, "session": None})
                return
            session_id = res.data[0].get("id")
        else:
            from athctl.state import load_state
            session_id = load_state().get("last_session_id")
            if not session_id:
                print("No session id. Use --latest or pass --id.", file=sys.stderr)
                sys.exit(2)

    msgs = client.table("session_messages").select("*").eq(
        "session_id", session_id
    ).order("id").execute()
    rows = msgs.data or []
    emit(
        f"Session {session_id}: {len(rows)} messages",
        data={"session_id": session_id, "messages": rows},
    )

    for r in rows:
        role = r.get("role", "?")
        content = r.get("content", "") or ""
        if not full:
            content = content[:200]
        style = {"user": "bold", "model": "green", "tool_call": "cyan dim"}.get(role, "")
        console.print(f"[{style}]{role}[/{style}]: {content}")
