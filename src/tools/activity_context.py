"""Activity context builder: aggregate training data for LLM prompt injection.

Computes activity data at three time horizons (last session, 7-day summary,
28-day trends) and formats as human-readable text. All arithmetic is done here
so the LLM never computes numbers -- it only interprets and coaches.

Public API:
    build_activity_context(plan=None) -> str
    build_planning_context(activities, lookback_days=28) -> str
    format_pace(min_per_unit, unit="km") -> str
    format_zone_distribution(zones, duration_seconds) -> str
    compute_weekly_trends(activities) -> list[dict]
    match_plan_sessions(plan, activities) -> dict
"""

import re
from datetime import datetime, timedelta, timezone

from src.tools.activity_store import list_activities


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def format_pace(min_per_unit: float, unit: str = "km") -> str:
    """Convert decimal pace to MM:SS format.

    Examples:
        5.98  -> '5:59/km'
        1.73  -> '1:44/100m'  (with unit='100m')
        4.999 -> '5:00/km'   (seconds=60 rolls up)
    """
    minutes = int(min_per_unit)
    seconds = int(round((min_per_unit - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/{unit}"


def format_zone_distribution(zones: dict | None, duration_seconds: float) -> str:
    """Convert zone seconds to percentage string.

    Input:  {"zone_1_seconds": 27.886, "zone_2_seconds": 186.5, ...}
    Output: "Z1 1% | Z2 5% | Z3 88% | Z4 6% | Z5 <1%"

    Returns empty string if zones is None or duration is 0.
    """
    if not zones or not duration_seconds:
        return ""

    total = sum(zones.get(f"zone_{i}_seconds", 0) for i in range(1, 6))
    if total == 0:
        return ""

    parts = []
    for i in range(1, 6):
        secs = zones.get(f"zone_{i}_seconds", 0)
        pct = (secs / total) * 100
        if 0 < pct < 1:
            parts.append(f"Z{i} <1%")
        else:
            parts.append(f"Z{i} {pct:.0f}%")
    return " | ".join(parts)


def compute_weekly_trends(activities: list[dict]) -> list[dict]:
    """Group activities by ISO week (Monday start), compute per-week aggregates.

    Returns list of dicts sorted chronologically:
        [{"week_start": "2026-01-13", "sessions": 4,
          "duration_min": 268, "distance_km": 0.0, "trimp": 127}, ...]
    """
    if not activities:
        return []

    weeks: dict[str, list[dict]] = {}
    for a in activities:
        st = a.get("start_time", "")
        if not st:
            continue
        dt = datetime.fromisoformat(st)
        week_start = (dt - timedelta(days=dt.weekday())).date().isoformat()
        weeks.setdefault(week_start, []).append(a)

    result = []
    for week_start in sorted(weeks.keys()):
        acts = weeks[week_start]
        total_dur = sum(a.get("duration_seconds", 0) for a in acts)
        total_dist = sum((a.get("distance_meters") or 0) for a in acts)
        total_trimp = sum((a.get("trimp") or 0) for a in acts)
        result.append({
            "week_start": week_start,
            "sessions": len(acts),
            "duration_min": round(total_dur / 60),
            "distance_km": round(total_dist / 1000, 1),
            "trimp": round(total_trimp),
        })
    return result


# ---------------------------------------------------------------------------
# Planning context: per-sport performance summaries for plan generation
# ---------------------------------------------------------------------------


def build_planning_context(activities: list[dict], lookback_days: int = 28) -> str:
    """Build per-sport performance summary for plan generation prompt injection.

    Aggregates recent activities by sport and workout type, computing
    averages, ranges, and trends. All arithmetic is done here so the LLM
    never computes numbers.

    Returns formatted text for LLM prompt injection, or empty string if
    no activities available.
    """
    if not activities:
        return ""

    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=lookback_days)).isoformat()
    extended_cutoff = (now - timedelta(days=lookback_days * 2)).isoformat()

    recent = [a for a in activities if a.get("start_time", "") >= cutoff]

    if not recent:
        # Try extended window for all activities
        recent = [a for a in activities if a.get("start_time", "") >= extended_cutoff]
        if not recent:
            return ""

    # Group by sport
    by_sport: dict[str, list[dict]] = {}
    for a in recent:
        sport = a.get("sport", "unknown")
        by_sport.setdefault(sport, []).append(a)

    # Check sparse sports: if a sport has < 3 sessions, try extended window
    all_extended = None  # lazy load
    for sport, acts in list(by_sport.items()):
        if len(acts) < 3:
            if all_extended is None:
                all_extended = [
                    a for a in activities
                    if a.get("start_time", "") >= extended_cutoff
                ]
            extended_sport = [
                a for a in all_extended
                if a.get("sport", "unknown") == sport
                and a.get("start_time", "") < cutoff
            ]
            if extended_sport:
                by_sport[sport] = extended_sport + acts

    parts = [f"ATHLETE'S RECENT PERFORMANCE DATA (last {lookback_days} days):\n"]

    for sport in sorted(by_sport.keys()):
        acts = by_sport[sport]
        sport_summary = _summarize_sport(sport, acts)
        if sport_summary:
            parts.append(sport_summary)

    # Volume and load trends using existing compute_weekly_trends
    trends = compute_weekly_trends(recent)
    if trends:
        parts.append(_format_trend_summary(trends))

    result = "\n\n".join(parts)
    return result


def _summarize_sport(sport: str, activities: list[dict]) -> str:
    """Summarize activities for a single sport into compact text.

    Fully generic: discovers available metrics from the data itself,
    never branches on sport name. Display formatting (e.g. pace unit)
    is derived from data magnitude, not sport identity.
    """
    if not activities:
        return ""

    count = len(activities)
    limited = count < 3
    header = f"{sport.title()} ({count} sessions{' - limited data' if limited else ''}):"

    # Classify sessions by intensity (works for any sport)
    by_type: dict[str, list[dict]] = {}
    for a in activities:
        intensity = _classify_intensity(a)
        by_type.setdefault(intensity, []).append(a)

    lines = [header]

    for intensity_type in ["easy", "tempo", "intervals", "long"]:
        acts = by_type.get(intensity_type, [])
        if not acts:
            continue

        metrics = _collect_metric_summaries(acts)
        metric_str = " | ".join(metrics)
        if metric_str:
            lines.append(f"  {intensity_type.title()} ({len(acts)} sessions): {metric_str}")
        else:
            lines.append(f"  {intensity_type.title()} ({len(acts)} sessions)")

    # Best recorded pace across all sessions (if pace data exists)
    all_paces = [
        a["pace"]["avg_min_per_km"]
        for a in activities
        if a.get("pace") and a["pace"].get("avg_min_per_km")
    ]
    if all_paces:
        best_pace = min(all_paces)
        pace_unit, display_pace = _pace_display_heuristic(best_pace)
        lines.append(f"  Best avg pace: {format_pace(display_pace, pace_unit)}")

    # Best/avg power across all sessions (if power data exists)
    avg_powers = [
        a["power"]["avg_watts"]
        for a in activities
        if a.get("power") and a["power"].get("avg_watts")
    ]
    if avg_powers:
        lines.append(f"  Avg power: {round(sum(avg_powers) / len(avg_powers))}W")
    max_powers = [
        a["power"]["max_watts"]
        for a in activities
        if a.get("power") and a["power"].get("max_watts")
    ]
    if max_powers:
        lines.append(f"  Max power: {round(max(max_powers))}W")

    # Avg speed across all sessions (if speed data exists)
    speeds = [
        a["speed"]["avg_km_h"]
        for a in activities
        if a.get("speed") and a["speed"].get("avg_km_h")
    ]
    if speeds:
        lines.append(f"  Avg speed: {sum(speeds) / len(speeds):.1f}km/h")

    # Avg elevation across all sessions (if elevation data exists)
    gains = [
        a["elevation"]["gain_meters"]
        for a in activities
        if a.get("elevation") and a["elevation"].get("gain_meters") is not None
    ]
    if gains:
        lines.append(f"  Avg elevation gain: {round(sum(gains) / len(gains))}m")

    # Aggregate zone distribution across all sessions
    total_zones: dict[str, float] = {}
    total_zone_seconds = 0
    for a in activities:
        zones = a.get("zone_distribution")
        if zones:
            for i in range(1, 6):
                key = f"zone_{i}_seconds"
                secs = zones.get(key, 0)
                total_zones[key] = total_zones.get(key, 0) + secs
                total_zone_seconds += secs

    if total_zone_seconds > 0:
        zone_str = format_zone_distribution(total_zones, total_zone_seconds)
        if zone_str:
            lines.append(f"  Zones: {zone_str}")

    # Duration range (always shown)
    durations = [a.get("duration_seconds", 0) for a in activities]
    if durations:
        min_dur = round(min(durations) / 60)
        max_dur = round(max(durations) / 60)
        if min_dur == max_dur:
            lines.append(f"  Typical duration: ~{min_dur}min")
        else:
            lines.append(f"  Typical duration: {min_dur}-{max_dur}min")

    return "\n".join(lines)


def _pace_display_heuristic(avg_min_per_km: float) -> tuple[str, float]:
    """Determine pace display unit from data magnitude, not sport name.

    If avg_min_per_km > 15 (typical swim pace is ~25-50 min/km), display
    as /100m by dividing by 10. Otherwise display as /km.

    Returns (unit_label, display_value).
    """
    if avg_min_per_km > 15:
        return ("100m", avg_min_per_km / 10)
    return ("km", avg_min_per_km)


def _collect_metric_summaries(acts: list[dict]) -> list[str]:
    """Build summary metric strings from whatever data exists in activities."""
    metrics = []

    # Pace (auto-detect unit by magnitude)
    paces = [
        a["pace"]["avg_min_per_km"]
        for a in acts
        if a.get("pace") and a["pace"].get("avg_min_per_km")
    ]
    if paces:
        avg_pace = sum(paces) / len(paces)
        pace_unit, display_pace = _pace_display_heuristic(avg_pace)
        metrics.append(f"avg {format_pace(display_pace, pace_unit)}")

    # Speed
    speeds = [
        a["speed"]["avg_km_h"]
        for a in acts
        if a.get("speed") and a["speed"].get("avg_km_h")
    ]
    if speeds:
        metrics.append(f"avg {sum(speeds) / len(speeds):.1f}km/h")

    # Power
    powers = [
        a["power"]["avg_watts"]
        for a in acts
        if a.get("power") and a["power"].get("avg_watts")
    ]
    if powers:
        metrics.append(f"avg {round(sum(powers) / len(powers))}W")

    # Heart rate
    hrs = [
        a["heart_rate"]["avg"]
        for a in acts
        if a.get("heart_rate") and a["heart_rate"].get("avg")
    ]
    if hrs:
        metrics.append(f"HR avg {round(sum(hrs) / len(hrs))}")

    # Duration range
    durations = [a.get("duration_seconds", 0) for a in acts]
    if durations:
        min_dur = round(min(durations) / 60)
        max_dur = round(max(durations) / 60)
        if min_dur == max_dur:
            metrics.append(f"~{min_dur}min")
        else:
            metrics.append(f"{min_dur}-{max_dur}min")

    return metrics


def _classify_intensity(activity: dict) -> str:
    """Classify a session by intensity based on zone distribution.

    Returns one of: 'easy', 'tempo', 'intervals', 'long'.
    Works for any sport using zone data and duration.
    """
    duration_sec = activity.get("duration_seconds", 0)
    dominant = _get_dominant_zone(activity)

    if dominant is None:
        # No zone data -- classify by duration
        if duration_sec > 3600:
            return "long"
        return "easy"

    if duration_sec > 3600 and dominant in (1, 2):
        return "long"
    if dominant in (4, 5):
        return "intervals"
    if dominant == 3:
        return "tempo"
    return "easy"


def _format_trend_summary(weeks: list[dict]) -> str:
    """Format volume and load trend words from weekly trend data."""
    if len(weeks) < 2:
        return ""

    half = len(weeks) // 2
    first_half = weeks[:half] if half > 0 else weeks[:1]
    second_half = weeks[half:] if half > 0 else weeks[1:]

    def avg_metric(wks: list[dict], key: str) -> float:
        vals = [w[key] for w in wks]
        return sum(vals) / len(vals) if vals else 0

    vol_first = avg_metric(first_half, "duration_min")
    vol_second = avg_metric(second_half, "duration_min")
    trimp_first = avg_metric(first_half, "trimp")
    trimp_second = avg_metric(second_half, "trimp")

    def trend_word(first: float, second: float) -> str:
        if first == 0:
            return "increasing" if second > 0 else "stable"
        change_pct = ((second - first) / first) * 100
        if change_pct >= 15:
            return f"increasing (+{change_pct:.0f}%)"
        elif change_pct <= -15:
            return f"decreasing ({change_pct:.0f}%)"
        else:
            return "stable"

    vol_trend = trend_word(vol_first, vol_second)
    trimp_trend = trend_word(trimp_first, trimp_second)

    return f"Volume trend: {vol_trend}\nTraining load trend: {trimp_trend}"


# ---------------------------------------------------------------------------
# Session-level plan matching
# ---------------------------------------------------------------------------


def match_plan_sessions(plan: dict, activities: list[dict]) -> dict:
    """Match planned sessions to actual activities by date and sport.

    Primary match: exact (date, sport) key lookup.
    Fallback: adjacent days (+/- 1 day) for same sport.
    Tracks used activities to prevent double-matching.

    Returns:
        {
            "matched": [{"planned": ..., "actual": ..., "duration_delta_min": int,
                         "intensity_match": str}, ...],
            "missed": [{"planned": session, "date": plan_date}, ...],
            "unplanned": [activity_dicts],
            "compliance_rate": float,
            "matched_count": int,
            "planned_count": int,
        }
    """
    # Index actual activities by (date_str, sport)
    actual_by_date_sport: dict[tuple[str, str], list[dict]] = {}
    for a in activities:
        st = a.get("start_time", "")
        if not st:
            continue
        date_str = datetime.fromisoformat(st).date().isoformat()
        sport = a.get("sport", "unknown")
        key = (date_str, sport)
        actual_by_date_sport.setdefault(key, []).append(a)

    matched = []
    missed = []
    used_activity_times: set[str] = set()

    for session in plan.get("sessions", []):
        plan_date = session.get("date", "")
        plan_sport = session.get("sport", "unknown")

        # Primary match: exact date + sport
        candidates = _filter_unused(
            actual_by_date_sport.get((plan_date, plan_sport), []),
            used_activity_times,
        )

        # Fallback: check adjacent days (+/- 1 day) for same sport
        if not candidates and plan_date:
            try:
                plan_dt = datetime.fromisoformat(plan_date).date()
            except ValueError:
                plan_dt = None
            if plan_dt:
                for delta in [timedelta(days=-1), timedelta(days=1)]:
                    adj_date = (plan_dt + delta).isoformat()
                    adj_key = (adj_date, plan_sport)
                    candidates = _filter_unused(
                        actual_by_date_sport.get(adj_key, []),
                        used_activity_times,
                    )
                    if candidates:
                        break

        if candidates:
            # Pick the first available candidate (activities are sorted by time)
            actual = candidates[0]
            used_activity_times.add(actual.get("start_time", ""))
            matched.append(_build_match_result(session, actual))
        else:
            missed.append({"planned": session, "date": plan_date})

    # Unplanned: actual activities not matched to any plan session
    unplanned = [
        a for a in activities
        if a.get("start_time", "") not in used_activity_times
    ]

    planned_count = len(plan.get("sessions", []))
    matched_count = len(matched)

    return {
        "matched": matched,
        "missed": missed,
        "unplanned": unplanned,
        "compliance_rate": matched_count / planned_count if planned_count else 0,
        "matched_count": matched_count,
        "planned_count": planned_count,
    }


def _filter_unused(
    candidates: list[dict],
    used_times: set[str],
) -> list[dict]:
    """Filter out activities whose start_time is already used."""
    return [a for a in candidates if a.get("start_time", "") not in used_times]


def _build_match_result(session: dict, actual: dict) -> dict:
    """Build a matched pair result with duration delta and intensity match."""
    planned_dur = session.get("duration_minutes", 0)
    actual_dur_sec = actual.get("duration_seconds", 0)
    actual_dur_min = round(actual_dur_sec / 60)
    duration_delta = actual_dur_min - planned_dur

    # Intensity match
    target_zone = _extract_zone_number(session.get("target_hr_zone", ""))
    actual_zone = _get_dominant_zone(actual)

    if target_zone is not None and actual_zone is not None:
        if actual_zone == target_zone:
            intensity_match = "on_target"
        elif actual_zone < target_zone:
            intensity_match = "lower_than_planned"
        else:
            intensity_match = "higher_than_planned"
    else:
        intensity_match = "unknown"

    return {
        "planned": session,
        "actual": actual,
        "duration_delta_min": duration_delta,
        "intensity_match": intensity_match,
    }


def _get_dominant_zone(activity: dict) -> int | None:
    """Find the HR zone (1-5) with the most seconds from zone_distribution.

    Returns None if no zone data is available.
    """
    zones = activity.get("zone_distribution")
    if not zones:
        return None

    best_zone = None
    best_seconds = -1
    for i in range(1, 6):
        secs = zones.get(f"zone_{i}_seconds", 0)
        if secs > best_seconds:
            best_seconds = secs
            best_zone = i

    return best_zone if best_seconds > 0 else None


def _extract_zone_number(target_hr_zone: str) -> int | None:
    """Parse zone number from plan strings like 'Zone 2', 'Zone 2-3', 'Zone 4-5 (briefly)'.

    Takes the first zone number found. Returns None if no zone data.
    """
    if not target_hr_zone:
        return None
    match = re.search(r"[Zz]one\s*(\d)", target_hr_zone)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_activity_context(plan: dict | None = None) -> str:
    """Build three-horizon activity context for LLM prompt injection.

    Returns formatted text with:
    - LAST SESSION: most recent activity in detail + per-sport most-recent
    - THIS WEEK: 7-day per-sport breakdown
    - 4-WEEK TRENDS: week-by-week aggregates with trend direction
    - PLAN vs ACTUAL: if plan provided, weekly compliance comparison

    Returns graceful message when no activities exist.
    """
    now = datetime.now(timezone.utc)
    all_activities = list_activities()

    if not all_activities:
        return "No training data available."

    cutoff_7 = (now - timedelta(days=7)).isoformat()
    cutoff_28 = (now - timedelta(days=28)).isoformat()
    last_7 = [a for a in all_activities if a.get("start_time", "") >= cutoff_7]
    last_28 = [a for a in all_activities if a.get("start_time", "") >= cutoff_28]

    parts = []
    last_session = _format_last_session(all_activities)
    if last_session:
        parts.append(last_session)

    week_summary = _format_week_summary(last_7)
    if week_summary:
        parts.append(week_summary)

    trends = _format_trends(last_28)
    if trends:
        parts.append(trends)

    if plan:
        plan_cmp = _format_plan_comparison(plan, last_7)
        if plan_cmp:
            parts.append(plan_cmp)

    return "\n\n".join(parts) if parts else "No training data available."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format duration as Xh Ymin or Ymin."""
    total_min = round(seconds / 60)
    if total_min >= 60:
        hours = total_min // 60
        mins = total_min % 60
        return f"{hours}h {mins:02d}min"
    return f"{total_min}min"


def _format_activity_detail(a: dict) -> str:
    """Format a single activity as a detailed multi-line block."""
    sport = a.get("sport", "unknown")
    start_time = a.get("start_time", "")
    duration_sec = a.get("duration_seconds", 0)
    distance_m = a.get("distance_meters")
    hr_data = a.get("heart_rate")
    pace_data = a.get("pace")
    speed_data = a.get("speed")
    power_data = a.get("power")
    elevation_data = a.get("elevation")
    zones = a.get("zone_distribution")
    trimp = a.get("trimp")
    hr_zone = a.get("hr_zone")

    # Date string
    if start_time:
        try:
            dt = datetime.fromisoformat(start_time)
            date_str = dt.strftime("%b %d")
        except ValueError:
            date_str = ""
    else:
        date_str = ""

    # First line: sport, date, key metrics
    line1_parts = []
    if distance_m:
        dist_km = distance_m / 1000
        line1_parts.append(f"{dist_km:.1f}km")
    line1_parts.append(f"in {_format_duration(duration_sec)}")

    # Pace or speed
    if pace_data and pace_data.get("avg_min_per_km"):
        if sport == "swimming":
            # Swimming uses min/100m -- derive from min/km
            pace_100m = pace_data["avg_min_per_km"] / 10
            line1_parts.append(f"Pace {format_pace(pace_100m, '100m')}")
        else:
            line1_parts.append(f"Pace {format_pace(pace_data['avg_min_per_km'])}")
    elif speed_data and speed_data.get("avg_km_h"):
        line1_parts.append(f"Speed {speed_data['avg_km_h']:.1f}km/h")

    # HR
    if hr_data and hr_data.get("avg"):
        zone_str = f" (Zone {hr_zone})" if hr_zone else ""
        line1_parts.append(f"HR {hr_data['avg']} avg{zone_str}")

    # TRIMP
    if trimp:
        line1_parts.append(f"TRIMP {round(trimp)}")

    header = f"LAST SESSION ({date_str}, {sport.title()}):"
    line1 = "  " + " | ".join(line1_parts)

    lines = [header, line1]

    # Zones line
    if zones and duration_sec:
        zone_str = format_zone_distribution(zones, duration_sec)
        if zone_str:
            lines.append(f"  Zones: {zone_str}")

    # Elevation + Power line
    extras = []
    if elevation_data:
        gain = elevation_data.get("gain_meters")
        loss = elevation_data.get("loss_meters")
        if gain is not None and loss is not None:
            extras.append(f"Elevation: +{round(gain)}m/-{round(loss)}m")
    if power_data and power_data.get("avg_watts"):
        extras.append(f"Power: {round(power_data['avg_watts'])}W avg")
    if extras:
        lines.append("  " + " | ".join(extras))

    return "\n".join(lines)


def _format_sport_oneliner(a: dict) -> str:
    """Format a single activity as a one-line per-sport summary."""
    sport = a.get("sport", "unknown")
    start_time = a.get("start_time", "")
    duration_sec = a.get("duration_seconds", 0)
    distance_m = a.get("distance_meters")
    hr_data = a.get("heart_rate")
    pace_data = a.get("pace")
    speed_data = a.get("speed")

    # Date
    if start_time:
        try:
            dt = datetime.fromisoformat(start_time)
            date_str = dt.strftime("%b %d")
        except ValueError:
            date_str = ""
    else:
        date_str = ""

    header = f"Last {sport.title()} ({date_str}):"
    metrics = []

    if distance_m:
        metrics.append(f"{distance_m / 1000:.1f}km")

    metrics.append(_format_duration(duration_sec))

    if pace_data and pace_data.get("avg_min_per_km"):
        if sport == "swimming":
            pace_100m = pace_data["avg_min_per_km"] / 10
            metrics.append(format_pace(pace_100m, "100m"))
        else:
            metrics.append(format_pace(pace_data["avg_min_per_km"]))
    elif speed_data and speed_data.get("avg_km_h"):
        metrics.append(f"{speed_data['avg_km_h']:.1f}km/h")

    if hr_data and hr_data.get("avg"):
        metrics.append(f"HR {hr_data['avg']}")

    return f"  {header} {' | '.join(metrics)}"


def _format_last_session(all_activities: list[dict]) -> str:
    """Show most recent activity in detail + one-liner per other sport."""
    if not all_activities:
        return ""

    # Most recent overall
    most_recent = all_activities[-1]  # list_activities returns sorted ascending
    result_lines = [_format_activity_detail(most_recent)]

    # Find unique sports in last 28 days and get most recent per sport
    now = datetime.now(timezone.utc)
    cutoff_28 = (now - timedelta(days=28)).isoformat()
    recent_activities = [
        a for a in all_activities if a.get("start_time", "") >= cutoff_28
    ]

    # Collect most recent per sport (other than the overall most recent's sport)
    most_recent_sport = most_recent.get("sport")
    sport_latest: dict[str, dict] = {}
    for a in recent_activities:
        s = a.get("sport", "unknown")
        if s == most_recent_sport:
            continue
        # Since activities are sorted ascending, later entries overwrite earlier
        sport_latest[s] = a

    # Add one-liners for other sports
    for sport in sorted(sport_latest.keys()):
        result_lines.append(_format_sport_oneliner(sport_latest[sport]))

    return "\n".join(result_lines)


def _format_week_summary(activities_7d: list[dict]) -> str:
    """7-day per-sport breakdown."""
    if not activities_7d:
        return "THIS WEEK (7 days, 0 sessions):\n  No activities recorded."

    # Group by sport
    by_sport: dict[str, list[dict]] = {}
    for a in activities_7d:
        s = a.get("sport", "unknown")
        by_sport.setdefault(s, []).append(a)

    total_sessions = len(activities_7d)
    lines = [f"THIS WEEK (7 days, {total_sessions} sessions):"]

    for sport in sorted(by_sport.keys()):
        acts = by_sport[sport]
        count = len(acts)
        total_dist_m = sum((a.get("distance_meters") or 0) for a in acts)
        total_dur_sec = sum(a.get("duration_seconds", 0) for a in acts)

        # Data-driven metrics: display whatever exists, no sport branching
        parts = [f"{sport.title()}: {count} sessions"]

        if total_dist_m > 0:
            parts.append(f"{total_dist_m / 1000:.1f}km")

        # Pace (unit determined by data magnitude, not sport name)
        paces = [
            a["pace"]["avg_min_per_km"]
            for a in acts
            if a.get("pace") and a["pace"].get("avg_min_per_km")
        ]
        if paces:
            avg_pace = sum(paces) / len(paces)
            pace_unit, display_pace = _pace_display_heuristic(avg_pace)
            parts.append(f"avg pace {format_pace(display_pace, pace_unit)}")

        # Speed
        speeds = [
            a["speed"]["avg_km_h"]
            for a in acts
            if a.get("speed") and a["speed"].get("avg_km_h")
        ]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            parts.append(f"avg speed {avg_speed:.1f}km/h")

        # Power
        powers = [
            a["power"]["avg_watts"]
            for a in acts
            if a.get("power") and a["power"].get("avg_watts")
        ]
        if powers:
            parts.append(f"avg power {round(sum(powers) / len(powers))}W")

        # Duration (always useful context)
        parts.append(_format_duration(total_dur_sec))

        # Avg HR
        hrs = [
            a["heart_rate"]["avg"]
            for a in acts
            if a.get("heart_rate") and a["heart_rate"].get("avg")
        ]
        if hrs:
            parts.append(f"avg HR {round(sum(hrs) / len(hrs))}")

        lines.append("  " + " | ".join(parts))

    # Totals line
    total_dur_sec = sum(a.get("duration_seconds", 0) for a in activities_7d)
    total_dist_m = sum((a.get("distance_meters") or 0) for a in activities_7d)
    total_trimp = sum((a.get("trimp") or 0) for a in activities_7d)

    totals_parts = [f"Totals: {_format_duration(total_dur_sec)}"]
    if total_dist_m > 0:
        totals_parts.append(f"{total_dist_m / 1000:.1f}km")
    totals_parts.append(f"TRIMP {round(total_trimp)}")
    lines.append("  " + " | ".join(totals_parts))

    return "\n".join(lines)


def _format_trends(activities_28d: list[dict]) -> str:
    """Call compute_weekly_trends(), format each week, add trend direction."""
    weeks = compute_weekly_trends(activities_28d)
    if not weeks:
        return ""

    lines = ["4-WEEK TRENDS:"]
    for w in weeks:
        # Week number from ISO date
        try:
            dt = datetime.fromisoformat(w["week_start"])
            iso_week = dt.isocalendar()[1]
            date_str = dt.strftime("%b %d")
        except ValueError:
            iso_week = "?"
            date_str = w["week_start"]

        lines.append(
            f"  Week {iso_week} ({date_str}): "
            f"{w['sessions']} sessions | "
            f"{w['duration_min']}min | "
            f"{w['distance_km']}km | "
            f"TRIMP {w['trimp']}"
        )

    # Trend direction: compare last 2 weeks avg vs first 2 weeks avg
    if len(weeks) >= 2:
        half = len(weeks) // 2
        first_half = weeks[:half] if half > 0 else weeks[:1]
        second_half = weeks[half:] if half > 0 else weeks[1:]

        def avg_metric(wks: list[dict], key: str) -> float:
            vals = [w[key] for w in wks]
            return sum(vals) / len(vals) if vals else 0

        vol_first = avg_metric(first_half, "duration_min")
        vol_second = avg_metric(second_half, "duration_min")
        trimp_first = avg_metric(first_half, "trimp")
        trimp_second = avg_metric(second_half, "trimp")

        def trend_word(first: float, second: float) -> str:
            if first == 0:
                return "increasing" if second > 0 else "stable"
            change_pct = ((second - first) / first) * 100
            if change_pct >= 15:
                return f"increasing (+{change_pct:.0f}%)"
            elif change_pct <= -15:
                return f"decreasing ({change_pct:.0f}%)"
            else:
                return "stable"

        vol_trend = trend_word(vol_first, vol_second)
        trimp_trend = trend_word(trimp_first, trimp_second)
        lines.append(f"  Trend: Volume {vol_trend}, TRIMP {trimp_trend}")

    return "\n".join(lines)


def _format_plan_comparison(plan: dict, activities_7d: list[dict]) -> str:
    """Compare plan sessions against actual at weekly aggregate level.

    Returns empty string if plan has no sessions.
    """
    # Extract planned sessions from plan dict
    sessions = plan.get("sessions", [])
    if not sessions:
        return ""

    # Group planned sessions by sport
    planned_by_sport: dict[str, int] = {}
    planned_total_dur_min = 0
    for s in sessions:
        sport = s.get("sport", "unknown")
        planned_by_sport[sport] = planned_by_sport.get(sport, 0) + 1
        planned_total_dur_min += s.get("duration_minutes", 0)

    # Group actual sessions by sport
    actual_by_sport: dict[str, int] = {}
    actual_total_dur_sec = 0
    for a in activities_7d:
        sport = a.get("sport", "unknown")
        actual_by_sport[sport] = actual_by_sport.get(sport, 0) + 1
        actual_total_dur_sec += a.get("duration_seconds", 0)

    actual_total_dur_min = round(actual_total_dur_sec / 60)

    lines = ["PLAN vs ACTUAL (this week):"]

    # Planned summary
    planned_parts = [
        f"{count} {sport}" for sport, count in sorted(planned_by_sport.items())
    ]
    lines.append(f"  Planned: {' + '.join(planned_parts)} sessions | {planned_total_dur_min}min total")

    # Actual summary
    actual_parts = [
        f"{count} {sport}" for sport, count in sorted(actual_by_sport.items())
    ]
    lines.append(
        f"  Actual: {' + '.join(actual_parts)} | {actual_total_dur_min}min total"
    )

    # Per-sport compliance for planned sports
    for sport in sorted(planned_by_sport.keys()):
        planned = planned_by_sport[sport]
        actual = actual_by_sport.get(sport, 0)
        pct = round((actual / planned) * 100) if planned else 0
        lines.append(f"  {sport.title()} compliance: {actual}/{planned} sessions ({pct}%)")

    # Volume comparison
    if planned_total_dur_min > 0:
        vol_diff = actual_total_dur_min - planned_total_dur_min
        vol_pct = round((vol_diff / planned_total_dur_min) * 100)
        sign = "+" if vol_diff >= 0 else ""
        lines.append(
            f"  Volume: {actual_total_dur_min}min actual vs {planned_total_dur_min}min planned ({sign}{vol_pct}%)"
        )

    # Note cross-training not in plan
    extra_sports = set(actual_by_sport.keys()) - set(planned_by_sport.keys())
    if extra_sports:
        extra_count = sum(actual_by_sport[s] for s in extra_sports)
        lines.append(
            f"  Note: {extra_count} additional cross-training sessions "
            f"({', '.join(sorted(extra_sports))}) not in plan"
        )

    return "\n".join(lines)
