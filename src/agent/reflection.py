"""Reflection pipeline orchestration: trigger detection, generation, and belief updates.

Coordinates existing building blocks (generate_reflection, store_episode,
extract_meta_beliefs from episodes.py) into a production pipeline triggered
at startup. Fitness metric updates are computed deterministically in Python
via fitness_tracker.py.

P7 enhancement: Reflection triggers are now event-driven (significant
compliance deviation, personal best, pattern detection) with LLM-based
significance evaluation. Falls back to time-based triggers if LLM fails.

Public API:
    check_and_generate_reflections(user_model, plan, activities) -> dict | None
    format_belief_update_summary(reflection_result) -> str
"""

import logging
from datetime import datetime, timedelta, timezone

from src.agent.json_utils import extract_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (fallback thresholds when LLM is unavailable)
# ---------------------------------------------------------------------------

REFLECTION_MIN_DAYS = 7
REFLECTION_MIN_ACTIVITIES = 3

# Event-driven trigger thresholds (P7)
COMPLIANCE_DEVIATION_THRESHOLD = 0.3  # >30% plan deviation triggers reflection
PB_DETECTION_ENABLED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_and_generate_reflections(
    user_model,
    plan: dict | None,
    activities: list[dict],
) -> dict | None:
    """Check if a reflection is due and generate one if so.

    Main entry point for the reflection pipeline. Called at startup.
    Returns reflection result dict or None if no reflection is due.

    At most ONE reflection per startup, even if multiple weeks are overdue.
    Wraps LLM calls in try/except so startup never breaks due to reflection failure.

    Args:
        user_model: UserModel instance.
        plan: Current training plan dict (may be None).
        activities: Full list of activity dicts from activity store.

    Returns:
        Dict with episode, meta_beliefs, belief_updates, fitness_updates,
        or None if no reflection is due.
    """
    # Lazy imports to avoid circular dependencies
    from src.memory.episodes import (
        generate_reflection,
        list_episodes,
        store_episode,
        extract_meta_beliefs,
    )
    from src.tools.fitness_tracker import compute_fitness_updates

    # Get last episode
    episodes = list_episodes(limit=1)
    last_episode = episodes[0] if episodes else None

    if not _is_reflection_due(last_episode, activities, plan=plan):
        return None

    # Determine reflection window
    window_start, window_end = _get_reflection_window(last_episode)
    window_activities = _filter_activities_in_window(
        activities, window_start, window_end
    )

    if len(window_activities) < REFLECTION_MIN_ACTIVITIES:
        return None

    # Compute assessment for the window (deterministic Python)
    assessment = _compute_window_assessment(plan, window_activities)

    # Generate reflection via LLM
    try:
        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)
        episode = generate_reflection(
            plan or {"sessions": [], "week_start": window_start},
            window_activities,
            {"assessment": assessment},
            profile,
            beliefs=beliefs,
        )
    except Exception as exc:
        log.warning("Reflection generation failed (%s), skipping", exc)
        return None

    # Store episode
    try:
        store_episode(episode)
    except Exception as exc:
        log.warning("Episode storage failed (%s), continuing", exc)

    # Extract meta-beliefs via LLM
    meta_beliefs = []
    try:
        meta_beliefs = extract_meta_beliefs(episode)
    except Exception as exc:
        log.warning("Meta-belief extraction failed (%s), continuing", exc)

    # Apply meta-beliefs to user model
    belief_updates = _apply_meta_beliefs(user_model, meta_beliefs, episode)

    # Compute fitness metric updates (deterministic Python)
    fitness_updates = compute_fitness_updates(user_model, window_activities)
    _apply_fitness_updates(user_model, fitness_updates)

    # Persist user model
    try:
        user_model.save()
    except Exception as exc:
        log.warning("User model save failed (%s)", exc)

    return {
        "episode": episode,
        "meta_beliefs": meta_beliefs,
        "belief_updates": belief_updates,
        "fitness_updates": fitness_updates,
    }


def format_belief_update_summary(reflection_result: dict) -> str:
    """Format reflection results as transparent coaching update text.

    Produces a string suitable for injection into the startup greeting prompt.
    Returns empty string if no updates to report.

    Example output:
        COACHING UPDATES (from recent training analysis):
          - Threshold pace improved: 4:15/km -> 4:10/km (3 interval/tempo sessions)
          - Coaching insight: Athlete runs easy sessions too fast

    Limits meta-beliefs to top 3 to keep the message concise.
    """
    if not reflection_result:
        return ""

    fitness_updates = reflection_result.get("fitness_updates", [])
    meta_beliefs = reflection_result.get("meta_beliefs", [])

    if not fitness_updates and not meta_beliefs:
        return ""

    parts = ["COACHING UPDATES (from recent training analysis):"]

    for update in fitness_updates:
        old_fmt = update.get("old_value_formatted", "unknown")
        new_fmt = update.get("new_value_formatted", "unknown")
        evidence = update.get("evidence", "")
        field = update.get("field", "")

        if "threshold_pace" in field:
            direction = update.get("direction", "changed")
            if update.get("old_value") is None:
                parts.append(
                    f"  - Threshold pace estimated: {new_fmt} ({evidence})"
                )
            else:
                parts.append(
                    f"  - Threshold pace {direction}: {old_fmt} -> {new_fmt} ({evidence})"
                )
        elif "weekly_volume" in field:
            if update.get("old_value") is None:
                parts.append(
                    f"  - Weekly volume estimated: {new_fmt} ({evidence})"
                )
            else:
                parts.append(
                    f"  - Weekly volume updated: {old_fmt} -> {new_fmt} ({evidence})"
                )

    # Limit meta-beliefs to top 3
    for mb in meta_beliefs[:3]:
        text = mb.get("text", "")
        if text:
            parts.append(f"  - Coaching insight: {text}")

    # If only the header was added (no actual updates), return empty
    if len(parts) <= 1:
        return ""

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------


def _is_reflection_due(
    last_episode: dict | None,
    activities: list[dict],
    plan: dict | None = None,
) -> bool:
    """Check if a reflection is due using event-driven triggers (P7).

    Triggers reflection when ANY of:
    1. Significant compliance deviation (>30% from plan)
    2. Time-based fallback (7+ days since last reflection AND 3+ new activities)
    3. No episodes exist and enough initial data

    Falls back to time-based trigger if event detection fails.
    """
    if not activities:
        return False

    if last_episode is None:
        # No episodes yet: check if we have enough initial activity data
        if len(activities) < REFLECTION_MIN_ACTIVITIES:
            return False
        first_time = activities[0].get("start_time", "")
        last_time = activities[-1].get("start_time", "")
        if not first_time or not last_time:
            return False
        try:
            first_dt = datetime.fromisoformat(first_time)
            last_dt = datetime.fromisoformat(last_time)
            span_days = (last_dt - first_dt).days
            return span_days >= REFLECTION_MIN_DAYS
        except (ValueError, TypeError):
            return False

    # Event-driven: check for significant compliance deviation (P7)
    if plan and plan.get("sessions"):
        compliance = _estimate_recent_compliance(plan, activities, last_episode)
        if compliance is not None and abs(1.0 - compliance) > COMPLIANCE_DEVIATION_THRESHOLD:
            return True

    # Time-based fallback
    last_generated = last_episode.get("generated_at", "")
    if not last_generated:
        last_generated = last_episode.get("period", "")

    if not last_generated:
        return True

    try:
        last_dt = datetime.fromisoformat(last_generated)
    except (ValueError, TypeError):
        return True

    now = datetime.now()
    if last_dt.tzinfo is not None:
        now = datetime.now(timezone.utc)

    days_since = (now - last_dt).days
    if days_since < REFLECTION_MIN_DAYS:
        return False

    cutoff = last_generated
    new_activities = [
        a for a in activities
        if a.get("start_time", "") > cutoff
    ]
    return len(new_activities) >= REFLECTION_MIN_ACTIVITIES


def _estimate_recent_compliance(
    plan: dict,
    activities: list[dict],
    last_episode: dict,
) -> float | None:
    """Estimate compliance since last reflection for event-driven triggering.

    Returns compliance ratio (0.0-1.0) or None if insufficient data.
    """
    cutoff = last_episode.get("generated_at", "") or last_episode.get("period", "")
    if not cutoff:
        return None

    new_activities = [
        a for a in activities
        if a.get("start_time", "") > cutoff
    ]

    if len(new_activities) < 2:
        return None

    planned_sessions = len(plan.get("sessions", []))
    if planned_sessions == 0:
        return None

    return min(1.0, len(new_activities) / planned_sessions)


def _get_reflection_window(last_episode: dict | None) -> tuple[str, str]:
    """Return (start_iso, end_iso) for the reflection window.

    Start = last episode's generated_at or period.
    End = now.
    If no last episode, start = 14 days ago (first reflection covers 2 weeks).
    """
    now = datetime.now(timezone.utc)
    end_iso = now.isoformat(timespec="seconds")

    if last_episode is None:
        # First reflection: analyze last 2 weeks only
        start = now - timedelta(days=14)
        return start.isoformat(timespec="seconds"), end_iso

    start_str = last_episode.get("generated_at", "")
    if not start_str:
        start_str = last_episode.get("period", "")

    if start_str:
        return start_str, end_iso

    # Fallback: 14 days ago
    start = now - timedelta(days=14)
    return start.isoformat(timespec="seconds"), end_iso


def _filter_activities_in_window(
    activities: list[dict],
    start: str,
    end: str,
) -> list[dict]:
    """Filter activities by start_time within [start, end] window."""
    result = []
    for a in activities:
        st = a.get("start_time", "")
        if not st:
            continue
        if st >= start and st <= end:
            result.append(a)
    return result


def _compute_window_assessment(
    plan: dict | None,
    window_activities: list[dict],
) -> dict:
    """Build a minimal assessment dict for generate_reflection.

    Uses existing compute_weekly_trends for volume/TRIMP trends.
    When plan exists, compute compliance from matched sessions.
    When no plan, pass stub assessment focusing on patterns.
    """
    # Lazy import
    from src.tools.activity_context import compute_weekly_trends, match_plan_sessions

    trends = compute_weekly_trends(window_activities)

    # Volume and TRIMP trend direction
    volume_trend = "stable"
    if len(trends) >= 2:
        first_vol = trends[0].get("duration_min", 0)
        last_vol = trends[-1].get("duration_min", 0)
        if first_vol > 0:
            change_pct = ((last_vol - first_vol) / first_vol) * 100
            if change_pct >= 15:
                volume_trend = "increasing"
            elif change_pct <= -15:
                volume_trend = "decreasing"

    assessment = {
        "compliance": 0,
        "fitness_trend": volume_trend,
        "fatigue_level": "moderate",
        "observations": [],
    }

    # Compliance from plan matching
    if plan and plan.get("sessions"):
        match_result = match_plan_sessions(plan, window_activities)
        assessment["compliance"] = match_result.get("compliance_rate", 0)
        matched_count = match_result.get("matched_count", 0)
        planned_count = match_result.get("planned_count", 0)
        assessment["observations"].append(
            f"Completed {matched_count} of {planned_count} planned sessions"
        )
    else:
        # No plan: focus on patterns
        total_sessions = len(window_activities)
        total_dist_km = sum(
            (a.get("distance_meters") or 0) / 1000 for a in window_activities
        )
        assessment["observations"].append(
            f"Completed {total_sessions} sessions totaling {total_dist_km:.1f} km (no plan active)"
        )

    # Add weekly summaries to observations
    for w in trends:
        assessment["observations"].append(
            f"Week of {w['week_start']}: {w['sessions']} sessions, "
            f"{w['distance_km']}km, TRIMP {w['trimp']}"
        )

    return assessment


def _apply_meta_beliefs(
    user_model,
    meta_beliefs: list[dict],
    episode: dict,
) -> list[dict]:
    """Apply meta-beliefs from extract_meta_beliefs to user model.

    For each meta-belief: call user_model.add_belief() with category="meta".
    Returns list of applied belief dicts.
    """
    applied = []
    ep_id = episode.get("id", "unknown")

    for mb in meta_beliefs:
        text = mb.get("text", "")
        if not text:
            continue
        confidence = mb.get("confidence", 0.7)
        try:
            belief = user_model.add_belief(
                text=text,
                category="meta",
                confidence=confidence,
                source="reflection",
                source_ref=ep_id,
            )
            applied.append(belief)
        except Exception as exc:
            log.warning("Failed to add meta-belief: %s", exc)

    return applied


def _apply_fitness_updates(user_model, fitness_updates: list[dict]) -> None:
    """Apply fitness update dicts to user model structured core.

    For each update: call user_model.update_structured_core(field, new_value).
    """
    for update in fitness_updates:
        field = update.get("field", "")
        new_value = update.get("new_value")
        if field and new_value is not None:
            try:
                user_model.update_structured_core(field, new_value)
            except Exception as exc:
                log.warning("Failed to apply fitness update %s: %s", field, exc)
