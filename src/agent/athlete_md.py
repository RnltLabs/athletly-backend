"""AthleteProfile.md generator: compact markdown context injected each turn.

After the journal refactor this is a thin wrapper:

1. Read the athlete journal (single markdown document, source of truth
   for identity / goal / preferences / open threads).
2. Append small *live-data* sections that change every day and do not
   belong in the journal: recent training summary (last 14 days) and
   recovery averages (last 7 days).

The whole thing is regenerated every turn so the live blocks stay
current. Failures are silently logged - context is never load-bearing.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_athlete_md(user_id: str) -> str:
    """Build the AthleteProfile.md document for *user_id*.

    Returns the journal markdown (created if missing) plus appended
    live-data sections. Never raises - any DB failure is logged and the
    failing section is skipped.
    """
    try:
        from src.agent.athlete_journal import (
            ensure_journal_exists,
            read_journal,
        )

        base = read_journal(user_id) or ensure_journal_exists(user_id)
        chunks: list[str] = [base.strip()]

        state = _build_journal_state_block(user_id)
        if state:
            chunks.append(state.strip())

        live = _build_live_sections(user_id)
        if live:
            chunks.append(live.strip())

        return "\n\n".join(chunks) + "\n"
    except Exception:
        logger.exception("build_athlete_md failed for user %s", user_id)
        return "# Athlete Profile\n\n_(profile context unavailable)_\n"


# ---------------------------------------------------------------------------
# Typed journal state (intents, programs, coach_notes)
# ---------------------------------------------------------------------------


def _build_journal_state_block(user_id: str) -> str:
    """Render the typed coach state (intents/programs/notes) as markdown.

    The dedicated helper lives in :mod:`src.agent.journal_state_md`; we
    isolate the import in a try/except so any DB failure simply omits
    the block instead of breaking the full prompt build.
    """
    try:
        from src.agent.journal_state_md import render_journal_state_md
        return render_journal_state_md(user_id)
    except Exception:
        logger.exception("_build_journal_state_block failed for %s", user_id)
        return ""


# ---------------------------------------------------------------------------
# Live-data sections (recent training + recovery)
# ---------------------------------------------------------------------------


def _build_live_sections(user_id: str) -> str:
    """Render the small live-data block appended after the journal."""
    try:
        from src.db.client import get_supabase

        client = get_supabase()
    except Exception:
        return ""

    chunks: list[str] = []
    training = _render_recent_training_section(client, user_id)
    if training:
        chunks.append(training)
    recovery = _render_recovery_section(client, user_id)
    if recovery:
        chunks.append(recovery)
    return "\n".join(chunks).strip()


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
