"""Dynamic trigger evaluation -- agent-defined rules via CalcEngine.

The agent defines trigger rules (condition + action) in the
proactive_trigger_rules table. This module builds a numeric context from
user data and evaluates each rule's condition via CalcEngine.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.calc.engine import CalcEngine
from src.db.agent_config_db import get_proactive_trigger_rules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_trigger_context(
    activities: list[dict],
    daily_metrics: list[dict],
    profile: dict,
) -> dict[str, float]:
    """Build a flat numeric dict for CalcEngine from user data.

    Available variables:
    - total_sessions_7d, total_minutes_7d, total_trimp_7d
    - avg_hrv_7d, avg_sleep_score_7d, avg_resting_hr_7d
    - body_battery_latest, stress_avg_latest, recovery_score_latest
    - days_since_last_session
    - {sport}_sessions_7d (e.g., running_sessions_7d, cycling_sessions_7d)
    - {sport}_trimp_7d
    """
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # Filter activities to last 7 days
    recent_acts = [
        a for a in activities
        if _parse_time(a) and _parse_time(a) >= week_ago
    ]

    ctx: dict[str, float] = {}

    # Session totals
    ctx["total_sessions_7d"] = float(len(recent_acts))
    ctx["total_minutes_7d"] = sum(
        (a.get("duration_seconds") or 0) / 60 for a in recent_acts
    )
    ctx["total_trimp_7d"] = sum(
        a.get("trimp") or a.get("training_load_trimp") or 0
        for a in recent_acts
    )

    # Per-sport breakdown
    sports: dict[str, list[dict]] = {}
    for a in recent_acts:
        sport = (
            a.get("sport") or a.get("activity_type") or "unknown"
        ).lower().replace(" ", "_")
        sports.setdefault(sport, []).append(a)

    for sport, acts in sports.items():
        ctx[f"{sport}_sessions_7d"] = float(len(acts))
        ctx[f"{sport}_trimp_7d"] = sum(
            a.get("trimp") or a.get("training_load_trimp") or 0
            for a in acts
        )

    # Days since last session
    if activities:
        latest_time = max(
            (_parse_time(a) for a in activities if _parse_time(a)),
            default=None,
        )
        if latest_time:
            ctx["days_since_last_session"] = (
                (now - latest_time).total_seconds() / 86400
            )
        else:
            ctx["days_since_last_session"] = 999.0
    else:
        ctx["days_since_last_session"] = 999.0

    # Daily metrics averages (last 7 days)
    recent_metrics = daily_metrics[:7]  # already sorted newest first
    if recent_metrics:
        hrv_vals = [
            m["hrv_avg"] for m in recent_metrics if m.get("hrv_avg")
        ]
        ctx["avg_hrv_7d"] = (
            sum(hrv_vals) / len(hrv_vals) if hrv_vals else 0.0
        )

        sleep_vals = [
            m["sleep_score"] for m in recent_metrics if m.get("sleep_score")
        ]
        ctx["avg_sleep_score_7d"] = (
            sum(sleep_vals) / len(sleep_vals) if sleep_vals else 0.0
        )

        rhr_vals = [
            m["resting_heart_rate"]
            for m in recent_metrics
            if m.get("resting_heart_rate")
        ]
        ctx["avg_resting_hr_7d"] = (
            sum(rhr_vals) / len(rhr_vals) if rhr_vals else 0.0
        )

        # Latest values (first item = newest)
        latest = recent_metrics[0]
        ctx["body_battery_latest"] = float(
            latest.get("body_battery_high") or 0
        )
        ctx["stress_avg_latest"] = float(latest.get("stress_avg") or 0)
        ctx["recovery_score_latest"] = float(
            latest.get("recovery_score") or 0
        )
    else:
        for key in (
            "avg_hrv_7d",
            "avg_sleep_score_7d",
            "avg_resting_hr_7d",
            "body_battery_latest",
            "stress_avg_latest",
            "recovery_score_latest",
        ):
            ctx[key] = 0.0

    return ctx


# ---------------------------------------------------------------------------
# Trigger evaluator
# ---------------------------------------------------------------------------


def evaluate_dynamic_triggers(
    user_id: str,
    activities: list[dict],
    daily_metrics: list[dict],
    profile: dict,
) -> list[dict]:
    """Load rules from DB, evaluate each condition, return fired triggers.

    Returns list of trigger dicts with type, priority, data, action.
    Skips rules that are in cooldown.
    """
    rules = get_proactive_trigger_rules(user_id)
    if not rules:
        return []

    ctx = build_trigger_context(activities, daily_metrics, profile)
    fired: list[dict] = []

    for rule in rules:
        condition = rule.get("condition", "")
        if not condition:
            continue

        # Skip if in cooldown
        cooldown_hours = rule.get("cooldown_hours", 24)
        rule_name = rule.get("name", "unknown")
        if _check_cooldown(user_id, rule_name, cooldown_hours):
            logger.debug(
                "Rule %s in cooldown for user %s", rule_name, user_id
            )
            continue

        # Evaluate condition via CalcEngine
        result = CalcEngine.calculate(condition, ctx)
        if result is None:
            logger.warning(
                "Dynamic trigger rule '%s' failed evaluation: condition='%s'",
                rule_name,
                condition,
            )
            continue

        # Truthy: any non-zero result
        if result:
            fired.append({
                "type": f"dynamic:{rule_name}",
                "priority": "medium",
                "data": {
                    "rule_name": rule_name,
                    "condition": condition,
                    "action": rule.get("action", ""),
                    "context_snapshot": {
                        k: round(v, 2) for k, v in ctx.items()
                    },
                },
            })

    return fired


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_time(activity: dict) -> datetime | None:
    """Parse start_time from an activity dict. Returns None on failure."""
    raw = activity.get("start_time")
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(raw))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _check_cooldown(
    user_id: str, rule_name: str, cooldown_hours: int
) -> bool:
    """Check proactive_queue for recent delivery within cooldown window."""
    try:
        from src.db.client import get_supabase

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
        ).isoformat()

        result = (
            get_supabase()
            .table("proactive_queue")
            .select("id")
            .eq("user_id", user_id)
            .eq("trigger_type", f"dynamic:{rule_name}")
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception as exc:
        logger.debug("Cooldown check failed for %s: %s", rule_name, exc)
        return False  # On error, allow the trigger
