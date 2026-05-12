"""AthleteProfile.md generator: compact markdown context injected each turn.

Mirrors Claude Code's CLAUDE.md pattern. Every coach turn sees a single
markdown document that summarizes everything stable about the athlete:

- Core profile (name, sports, available days)
- Active goal (event, target date, target time)
- Active macrocycle status (phase, current week)
- Top beliefs (sorted by confidence)
- Recent training summary (last 14 days)
- Free-form notes from the user (athlete_md column)

The document is regenerated each turn from canonical DB state, so it
stays in sync without manual maintenance. The user-editable section
(``athlete_md`` column) is appended verbatim - they can add things the
coach should always remember.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_athlete_md(user_id: str, *, max_beliefs: int = 8) -> str:
    """Build the AthleteProfile.md document for *user_id*.

    Pulls from profiles, beliefs, macrocycle_plans, activities,
    health_daily_metrics. Returns a markdown string suitable for direct
    injection into the system prompt as <athlete_profile> context.

    Never raises - any DB failure is logged and the section is skipped.
    """
    try:
        from src.db.client import get_supabase

        client = get_supabase()

        sections: list[str] = ["# Athlete Profile", ""]

        profile = _load_profile(client, user_id)
        sections.append(_render_core_section(profile))
        sections.append(_render_goal_section(profile))
        sections.append(_render_macrocycle_section(client, user_id))
        sections.append(_render_beliefs_section(client, user_id, max_beliefs))
        sections.append(_render_recent_training_section(client, user_id))
        sections.append(_render_recovery_section(client, user_id))
        sections.append(_render_user_notes_section(profile))

        return "\n".join(s for s in sections if s).strip() + "\n"
    except Exception:
        logger.exception("build_athlete_md failed for user %s", user_id)
        return f"# Athlete Profile\n\n_(profile context unavailable)_\n"


# ---------------------------------------------------------------------------
# Section renderers - each returns a markdown chunk or "" on missing data
# ---------------------------------------------------------------------------


def _load_profile(client, user_id: str) -> dict[str, Any]:
    res = client.table("profiles").select("*").eq("user_id", user_id).execute()
    return res.data[0] if res.data else {}


def _render_core_section(profile: dict) -> str:
    if not profile:
        return ""
    lines = ["## Core", ""]
    name = profile.get("name")
    if name:
        lines.append(f"- Name: {name}")
    sports = profile.get("sports") or []
    if sports:
        lines.append(f"- Sports: {', '.join(sports)}")
    days = profile.get("training_days_per_week")
    if days:
        lines.append(f"- Training days/week: {days}")
    max_min = profile.get("max_session_minutes")
    if max_min:
        lines.append(f"- Max session: {max_min} min")
    if len(lines) == 2:
        return ""
    return "\n".join(lines) + "\n"


def _render_goal_section(profile: dict) -> str:
    if not profile:
        return ""
    event = profile.get("goal_event")
    if not event:
        return ""
    lines = ["## Current Goal", "", f"- Event: {event}"]
    if profile.get("goal_target_date"):
        target = profile["goal_target_date"]
        try:
            target_dt = datetime.fromisoformat(str(target)).date()
            days_to_go = (target_dt - date.today()).days
            lines.append(f"- Target date: {target} ({days_to_go} days away)")
        except Exception:
            lines.append(f"- Target date: {target}")
    if profile.get("goal_target_time"):
        lines.append(f"- Target time: {profile['goal_target_time']}")
    return "\n".join(lines) + "\n"


def _render_macrocycle_section(client, user_id: str) -> str:
    try:
        res = (
            client.table("macrocycle_plans")
            .select("id, name, start_date, total_weeks, status, weeks")
            .eq("user_id", user_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
    except Exception:
        return ""
    if not res.data:
        return ""
    mc = res.data[0]
    name = mc.get("name", "(unnamed)")
    total = mc.get("total_weeks", 0)
    start_str = mc.get("start_date")
    current_week = None
    current_phase = None
    if start_str:
        try:
            start_dt = datetime.fromisoformat(str(start_str)).date()
            days = (date.today() - start_dt).days
            current_week = max(1, min(total, days // 7 + 1))
        except Exception:
            pass
    if current_week and mc.get("weeks"):
        try:
            idx = current_week - 1
            weeks = mc["weeks"]
            if 0 <= idx < len(weeks):
                current_phase = (weeks[idx] or {}).get("phase")
        except Exception:
            pass
    lines = ["## Macrocycle Status", "", f"- Active: {name} ({total} weeks total)"]
    if current_week:
        lines.append(f"- Current week: {current_week} of {total}")
    if current_phase:
        lines.append(f"- Phase: {current_phase}")
    return "\n".join(lines) + "\n"


def _render_beliefs_section(client, user_id: str, max_beliefs: int) -> str:
    try:
        res = (
            client.table("beliefs")
            .select("text, category, confidence")
            .eq("user_id", user_id)
            .eq("active", True)
            .order("confidence", desc=True)
            .limit(max_beliefs)
            .execute()
        )
    except Exception:
        return ""
    rows = res.data or []
    if not rows:
        return ""
    lines = [f"## Top Beliefs ({len(rows)} shown, by confidence)", ""]
    for r in rows:
        conf = r.get("confidence", 0)
        cat = r.get("category", "?")
        text = (r.get("text") or "").strip()
        lines.append(f"- [{conf:.2f} {cat}] {text}")
    return "\n".join(lines) + "\n"


def _render_recent_training_section(client, user_id: str) -> str:
    try:
        since = (date.today() - timedelta(days=14)).isoformat()
        res = (
            client.table("activities")
            .select("sport, duration_seconds, distance_meters, avg_hr, start_time")
            .eq("user_id", user_id)
            .gte("start_time", since)
            .order("start_time", desc=True)
            .execute()
        )
    except Exception:
        return ""
    rows = res.data or []
    if not rows:
        return ""

    from collections import Counter

    sports = Counter(r.get("sport") or "unknown" for r in rows)
    total_min = sum((r.get("duration_seconds") or 0) for r in rows) // 60
    total_km = sum((r.get("distance_meters") or 0) for r in rows) / 1000
    sport_summary = ", ".join(f"{c} {s}" for s, c in sports.most_common(5))
    lines = [
        "## Recent Training (last 14 days)",
        "",
        f"- Sessions: {len(rows)} ({sport_summary})",
        f"- Total duration: {total_min} min",
        f"- Total distance: {total_km:.1f} km",
    ]
    return "\n".join(lines) + "\n"


def _render_recovery_section(client, user_id: str) -> str:
    try:
        since = (date.today() - timedelta(days=7)).isoformat()
        res = (
            client.table("health_daily_metrics")
            .select("date, sleep_score, resting_heart_rate, hrv_avg, body_battery_high")
            .eq("user_id", user_id)
            .gte("date", since)
            .order("date", desc=True)
            .execute()
        )
    except Exception:
        return ""
    rows = res.data or []
    if not rows:
        return ""

    def _avg(key: str) -> float | None:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    sleep = _avg("sleep_score")
    rhr = _avg("resting_heart_rate")
    hrv = _avg("hrv_avg")
    bb = _avg("body_battery_high")
    if all(v is None for v in (sleep, rhr, hrv, bb)):
        return ""
    lines = ["## Recovery (last 7 days, averages)", ""]
    if sleep is not None:
        lines.append(f"- Sleep score: {sleep:.0f}")
    if rhr is not None:
        lines.append(f"- Resting HR: {rhr:.0f} bpm")
    if hrv is not None:
        lines.append(f"- HRV: {hrv:.0f} ms")
    if bb is not None:
        lines.append(f"- Body battery (peak): {bb:.0f}")
    return "\n".join(lines) + "\n"


def _render_user_notes_section(profile: dict) -> str:
    notes = (profile.get("athlete_md") or "").strip()
    if not notes:
        return ""
    return f"## Notes from athlete\n\n{notes}\n"
