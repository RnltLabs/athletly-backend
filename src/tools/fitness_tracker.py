"""Python-side fitness metric computation from activity data.

Deterministic computation of fitness metric changes following the hybrid
architecture principle: Python computes, LLM reasons. Never calls the LLM.

Uses median of qualifying sessions and minimum-evidence thresholds to
avoid noisy updates from single-session fluctuations.

Public API:
    compute_fitness_updates(user_model, recent_activities) -> list[dict]
"""

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACE_CHANGE_THRESHOLD = 0.10  # min/km minimum change to report (~2% at 5:00/km)
VOLUME_CHANGE_THRESHOLD = 0.05  # 5% relative change minimum
MIN_QUALIFYING_SESSIONS = 3  # minimum sessions for threshold pace estimation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fitness_updates(
    user_model,
    recent_activities: list[dict],
) -> list[dict]:
    """Compute all fitness metric updates from recent activity data.

    Returns list of update dicts, each containing:
        field, old_value, old_value_formatted, new_value, new_value_formatted,
        evidence, direction (for pace).

    All computation is deterministic Python. No LLM calls.

    Args:
        user_model: UserModel instance for current fitness values.
        recent_activities: List of activity dicts in the reflection window.

    Returns:
        List of update dicts (may be empty if no significant changes).
    """
    if not recent_activities:
        return []

    updates = []

    threshold_update = _estimate_threshold_pace(user_model, recent_activities)
    if threshold_update:
        updates.append(threshold_update)

    volume_update = _estimate_weekly_volume(user_model, recent_activities)
    if volume_update:
        updates.append(volume_update)

    return updates


# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------


def _estimate_threshold_pace(
    user_model,
    activities: list[dict],
) -> dict | None:
    """Estimate threshold pace from interval/tempo running sessions.

    Filters for running activities classified as 'intervals' or 'tempo',
    requires at least MIN_QUALIFYING_SESSIONS, uses median pace, and
    only reports changes above PACE_CHANGE_THRESHOLD.

    Returns update dict or None if insufficient data or no significant change.
    """
    # Lazy imports to avoid circular dependencies
    from src.tools.activity_context import _classify_intensity, format_pace

    qualifying_paces = []
    for a in activities:
        if a.get("sport") != "running":
            continue
        intensity = _classify_intensity(a)
        if intensity not in ("intervals", "tempo"):
            continue
        pace = a.get("pace", {}).get("avg_min_per_km")
        if pace and pace > 0:
            qualifying_paces.append(pace)

    if len(qualifying_paces) < MIN_QUALIFYING_SESSIONS:
        return None

    # Use median to reduce outlier sensitivity
    qualifying_paces.sort()
    mid = len(qualifying_paces) // 2
    if len(qualifying_paces) % 2 == 0:
        median_pace = (qualifying_paces[mid - 1] + qualifying_paces[mid]) / 2
    else:
        median_pace = qualifying_paces[mid]

    current_threshold = (
        user_model.structured_core
        .get("fitness", {})
        .get("threshold_pace_min_km")
    )

    evidence = (
        f"{len(qualifying_paces)} interval/tempo sessions, "
        f"median {format_pace(median_pace)}"
    )

    if current_threshold is None:
        # First estimate: always report
        return {
            "field": "fitness.threshold_pace_min_km",
            "old_value": None,
            "old_value_formatted": "not set",
            "new_value": round(median_pace, 2),
            "new_value_formatted": format_pace(median_pace),
            "evidence": evidence,
            "direction": "established",
        }

    change = abs(median_pace - current_threshold)
    if change < PACE_CHANGE_THRESHOLD:
        return None  # Not significant enough

    # Lower pace = faster = improved for running
    direction = "improved" if median_pace < current_threshold else "regressed"

    return {
        "field": "fitness.threshold_pace_min_km",
        "old_value": round(current_threshold, 2),
        "old_value_formatted": format_pace(current_threshold),
        "new_value": round(median_pace, 2),
        "new_value_formatted": format_pace(median_pace),
        "evidence": evidence,
        "direction": direction,
    }


def _estimate_weekly_volume(
    user_model,
    activities: list[dict],
) -> dict | None:
    """Estimate weekly training volume from recent activities.

    Uses compute_weekly_trends() to get weekly aggregates, computes
    average km/week from the last 4 weeks. Only reports changes above
    VOLUME_CHANGE_THRESHOLD (5% relative).

    Returns update dict or None if insufficient data or no significant change.
    """
    # Lazy import to avoid circular dependencies
    from src.tools.activity_context import compute_weekly_trends

    weeks = compute_weekly_trends(activities)

    if not weeks:
        return None

    # Average km/week from available weeks (up to last 4)
    recent_weeks = weeks[-4:] if len(weeks) > 4 else weeks
    total_km = sum(w.get("distance_km", 0) for w in recent_weeks)
    avg_km = total_km / len(recent_weeks)

    if avg_km <= 0:
        return None

    avg_km_rounded = round(avg_km, 1)
    evidence = f"{len(recent_weeks)}-week average: {avg_km_rounded} km/week"

    current_volume = (
        user_model.structured_core
        .get("fitness", {})
        .get("weekly_volume_km")
    )

    if current_volume is None:
        # First estimate: always report
        return {
            "field": "fitness.weekly_volume_km",
            "old_value": None,
            "old_value_formatted": "not set",
            "new_value": avg_km_rounded,
            "new_value_formatted": f"{avg_km_rounded} km/week",
            "evidence": evidence,
            "direction": "established",
        }

    # Relative change check
    if current_volume > 0:
        relative_change = abs(avg_km_rounded - current_volume) / current_volume
        if relative_change < VOLUME_CHANGE_THRESHOLD:
            return None  # Not significant enough

    # Volume: higher = increased (not necessarily "improved")
    direction = "increased" if avg_km_rounded > current_volume else "decreased"

    return {
        "field": "fitness.weekly_volume_km",
        "old_value": round(current_volume, 1),
        "old_value_formatted": f"{round(current_volume, 1)} km/week",
        "new_value": avg_km_rounded,
        "new_value_formatted": f"{avg_km_rounded} km/week",
        "evidence": evidence,
        "direction": direction,
    }
