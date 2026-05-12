"""Snapshot command for athctl.

Writes a single Markdown document with the full active-user state so
the operator (and the LLM watching the loop) can grasp everything in
one read. ``athctl snapshot diff`` compares two snapshots.
"""

from __future__ import annotations

import difflib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from athctl.common import console, emit, get_supabase, require_active_user

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _section(title: str, lines: list[str]) -> str:
    body = "\n".join(lines) if lines else "_(empty)_"
    return f"## {title}\n\n{body}\n"


def _profile_section(client, uid: str) -> str:
    res = client.table("profiles").select("*").eq("user_id", uid).execute()
    if not res.data:
        return _section("Profile", [])
    row = res.data[0]
    lines = ["| Field | Value |", "|---|---|"]
    for k in (
        "name", "sports", "goal", "constraints", "fitness", "preferences",
    ):
        v = row.get(k)
        lines.append(f"| {k} | `{json.dumps(v, default=str)}` |")
    return _section("Profile", lines)


def _beliefs_section(client, uid: str) -> str:
    res = (
        client.table("beliefs")
        .select("*")
        .eq("user_id", uid)
        .order("confidence", desc=True)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return _section("Beliefs", [])
    lines = ["| Conf | Category | Statement | Source |", "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('confidence', 0):.2f} "
            f"| {r.get('category', '')} "
            f"| {(r.get('text') or '')[:140]} "
            f"| {r.get('source', '')} |"
        )
    return _section(f"Beliefs ({len(rows)})", lines)


def _providers_section(uid: str) -> str:
    try:
        from src.services.providers.registry import list_providers
    except Exception as exc:
        return _section("Connected Providers", [f"_(import failed: {exc})_"])

    lines = [
        "| Provider | Status | Last Sync | Activities | Daily | Sleep | Body Battery |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in list_providers():
        status = p.check_connection(uid)
        caps = p.capabilities
        lines.append(
            f"| {p.name} "
            f"| {status.get('status', '?')} "
            f"| {status.get('last_sync_at') or '-'} "
            f"| {'yes' if caps.activities else '-'} "
            f"| {'yes' if caps.daily_metrics else '-'} "
            f"| {'yes' if caps.sleep else '-'} "
            f"| {'yes' if caps.body_battery else '-'} |"
        )
    return _section("Connected Providers", lines)


def _plan_section(uid: str) -> str:
    try:
        from src.db.plans_db import get_active_plan
    except Exception as exc:
        return _section("Active Plan", [f"_(import failed: {exc})_"])
    p = get_active_plan(uid)
    if not p:
        return _section("Active Plan", [])
    data = p.get("plan_data") or {}
    weeks = data.get("weeks") or []
    lines = [
        f"- Plan id: `{p.get('id')}`",
        f"- Created: {p.get('created_at')}",
        f"- Weeks: {len(weeks)}",
        f"- Evaluation: {p.get('evaluation_score')} / {p.get('evaluation_feedback')}",
        "",
    ]
    for w in weeks[:3]:
        wn = w.get("week_num")
        lines.append(f"### Week {wn}")
        lines.append("| Type | Duration | Distance | Zones |")
        lines.append("|---|---|---|---|")
        for s in w.get("sessions") or []:
            lines.append(
                f"| {s.get('session_type', '')} "
                f"| {s.get('duration_minutes', '?')}min "
                f"| {s.get('distance_km', '?')}km "
                f"| {json.dumps(s.get('intensity_zones', {}), default=str)} |"
            )
        lines.append("")
    return _section("Active Plan", lines)


def _activities_section(client, uid: str, limit: int = 20) -> str:
    res = (
        client.table("activities")
        .select("sport, start_time, duration_seconds, distance_meters, avg_hr, source")
        .eq("user_id", uid)
        .order("start_time", desc=True)
        .limit(limit)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return _section("Recent Activities", [])
    lines = [
        "| Date | Sport | Duration | Distance | HR | Source |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        dur = r.get("duration_seconds") or 0
        dist = r.get("distance_meters") or 0
        lines.append(
            f"| {str(r.get('start_time', ''))[:16]} "
            f"| {r.get('sport', '')} "
            f"| {dur // 60}min "
            f"| {dist / 1000:.1f}km "
            f"| {r.get('avg_hr') or '-'} "
            f"| {r.get('source', '')} |"
        )
    return _section(f"Recent Activities (last {limit})", lines)


def _sessions_section(client, uid: str, limit: int = 5) -> str:
    res = (
        client.table("sessions")
        .select("id, context, started_at, turn_count")
        .eq("user_id", uid)
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return _section("Recent Sessions", [])
    lines = ["| Started | Id | Context | Turns |", "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {str(r.get('started_at', ''))[:19]} "
            f"| `{str(r.get('id', ''))[:8]}` "
            f"| {r.get('context', '')} "
            f"| {r.get('turn_count', '?')} |"
        )
    return _section(f"Recent Sessions (last {limit})", lines)


def _pending_actions_section(client, uid: str) -> str:
    try:
        res = (
            client.table("pending_actions")
            .select("*")
            .eq("user_id", uid)
            .is_("resolved_at", "null")
            .execute()
        )
    except Exception:
        return _section("Pending Actions", [])
    rows = res.data or []
    if not rows:
        return _section("Pending Actions", [])
    lines = [
        f"- `{r.get('action_type')}`: {r.get('description', '')[:120]}"
        for r in rows
    ]
    return _section("Pending Actions", lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@click.group(help="Capture or compare full snapshots of the active user.", invoke_without_command=True)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output path (default: logs/snapshots/snap_TS.md).",
)
@click.pass_context
def snapshot(ctx: click.Context, out_path: str | None) -> None:
    """Default behavior (no subcommand) writes a fresh snapshot."""
    if ctx.invoked_subcommand is not None:
        return
    _write_snapshot(out_path)


def _write_snapshot(out_path: str | None) -> None:
    uid = require_active_user()
    client = get_supabase()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if out_path:
        target = Path(out_path)
    else:
        target = Path("logs/snapshots") / f"snap_{ts}.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        f"# Snapshot: {uid}",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        _profile_section(client, uid),
        _beliefs_section(client, uid),
        _providers_section(uid),
        _plan_section(uid),
        _activities_section(client, uid),
        _sessions_section(client, uid),
        _pending_actions_section(client, uid),
    ]
    target.write_text("\n".join(sections), encoding="utf-8")

    emit(
        f"Snapshot written to {target}",
        data={"user_id": uid, "path": str(target)},
    )


@snapshot.command("write", help="Write a fresh snapshot (alias for the default).")
@click.option("--out", "out_path", default=None)
def snapshot_write(out_path: str | None) -> None:
    _write_snapshot(out_path)


@snapshot.command("diff", help="Compare two snapshot files.")
@click.argument("path_a", type=click.Path(exists=True, dir_okay=False))
@click.argument("path_b", type=click.Path(exists=True, dir_okay=False))
def snapshot_diff(path_a: str, path_b: str) -> None:
    a = Path(path_a).read_text(encoding="utf-8").splitlines(keepends=True)
    b = Path(path_b).read_text(encoding="utf-8").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(a, b, fromfile=path_a, tofile=path_b, n=2)
    )
    if not diff:
        emit("No differences.")
        return
    sys.stdout.writelines(diff)
