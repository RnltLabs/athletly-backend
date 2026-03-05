"""FIT file parser: extracts structured activity data from Garmin/Wahoo FIT files."""

import json
from pathlib import Path

import fitdecode


def parse_fit_file(file_path: str) -> dict:
    """Parse a FIT file and return structured activity data.

    Returns a dict with sport, sub_sport, duration, HR, pace/speed, distance,
    power, elevation, zone_distribution, and source_file.
    Also supports loading pre-parsed JSON fixtures for testing.
    """
    path = Path(file_path)

    # Support loading pre-parsed JSON fixtures directly
    if path.suffix == ".json":
        return json.loads(path.read_text())

    # Parse actual FIT file
    sport = "unknown"
    sub_sport = None
    sport_from_session = False
    start_time = None
    duration_seconds = None
    distance_meters = None
    hr_values = []
    speed_values = []
    power_values = []
    altitude_values = []
    calories = None
    zone_distribution = None
    zone_distribution_source = None
    hr_zone_boundaries = None
    device_max_hr = None
    device_resting_hr = None

    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue

            if frame.name == "session":
                # Session frame is primary source for sport detection
                session_sport = _get_field(frame, "sport")
                if session_sport and not sport_from_session:
                    sport = _normalize_sport(
                        str(session_sport),
                        str(_get_field(frame, "sub_sport") or ""),
                    )
                    sub_sport = str(_get_field(frame, "sub_sport")) if _get_field(frame, "sub_sport") else None
                    sport_from_session = True

                start_time = _get_field(frame, "start_time")
                duration_seconds = _get_field(frame, "total_elapsed_time")
                distance_meters = _get_field(frame, "total_distance")
                calories = _get_field(frame, "total_calories")
                # Session-level HR as fallback
                if not hr_values:
                    avg_hr = _get_field(frame, "avg_heart_rate")
                    max_hr = _get_field(frame, "max_heart_rate")
                    if avg_hr and max_hr:
                        hr_values = [avg_hr, max_hr]  # fallback

            elif frame.name == "sport":
                # Sport frame is fallback source for sport detection
                if not sport_from_session:
                    sport_field = _get_field(frame, "sport")
                    if sport_field:
                        sport_sub = str(_get_field(frame, "sub_sport") or "")
                        sport = _normalize_sport(str(sport_field), sport_sub)
                        if not sub_sport:
                            sub_sport = str(_get_field(frame, "sub_sport")) if _get_field(frame, "sub_sport") else None

            elif frame.name == "time_in_zone":
                # Extract device-computed zone distribution (session-level)
                ref = _get_field(frame, "reference_mesg")
                if ref == "session" and zone_distribution is None:
                    hr_zones_raw = _get_field(frame, "time_in_hr_zone")
                    if hr_zones_raw and isinstance(hr_zones_raw, (list, tuple)):
                        zone_distribution = _build_zone_distribution(hr_zones_raw)
                        zone_distribution_source = "device"

                    boundaries_raw = _get_field(frame, "hr_zone_high_boundary")
                    if boundaries_raw and isinstance(boundaries_raw, (list, tuple)):
                        hr_zone_boundaries = list(boundaries_raw)

                    dmhr = _get_field(frame, "max_heart_rate")
                    if dmhr and isinstance(dmhr, (int, float)):
                        device_max_hr = int(dmhr)

                    drhr = _get_field(frame, "resting_heart_rate")
                    if drhr and isinstance(drhr, (int, float)):
                        device_resting_hr = int(drhr)

            elif frame.name == "record":
                hr = _get_field(frame, "heart_rate")
                if hr and isinstance(hr, (int, float)):
                    hr_values.append(hr)
                speed = _get_field(frame, "enhanced_speed") or _get_field(frame, "speed")
                if speed and isinstance(speed, (int, float)):
                    speed_values.append(speed)
                power = _get_field(frame, "power")
                if power and isinstance(power, (int, float)):
                    power_values.append(power)
                altitude = _get_field(frame, "enhanced_altitude") or _get_field(frame, "altitude")
                if altitude and isinstance(altitude, (int, float)):
                    altitude_values.append(altitude)

    # Build result
    result = {
        "source_file": path.name,
        "sport": sport,
        "sub_sport": sub_sport,
        "start_time": start_time.isoformat() if hasattr(start_time, "isoformat") else str(start_time) if start_time else None,
        "duration_seconds": round(duration_seconds) if duration_seconds else None,
        "distance_meters": round(distance_meters) if distance_meters else None,
        "heart_rate": _build_hr(hr_values),
        "pace": _build_pace(speed_values, sport, distance_meters),
        "speed": _build_speed(speed_values, sport),
        "power": _build_power(power_values),
        "elevation": _build_elevation(altitude_values),
        "calories": calories,
        "zone_distribution": zone_distribution,
        "zone_distribution_source": zone_distribution_source,
    }

    # Include device HR zone metadata if available
    if hr_zone_boundaries:
        result["hr_zone_boundaries"] = hr_zone_boundaries
    if device_max_hr:
        result["device_max_hr"] = device_max_hr
    if device_resting_hr:
        result["device_resting_hr"] = device_resting_hr

    return result


def is_activity_file(file_path: str) -> bool:
    """Fast check (~1ms) whether a FIT file is an activity (has session frame).

    Returns True as soon as a session frame is found.
    Returns False if no session frame found or on any parse error.
    """
    try:
        with fitdecode.FitReader(str(file_path)) as fit:
            for frame in fit:
                if frame.frame_type == fitdecode.FIT_FRAME_DATA:
                    if frame.name == "session":
                        return True
        return False
    except Exception:
        return False


def _get_field(frame, field_name):
    """Safely get a field value from a FIT frame."""
    try:
        return frame.get_value(field_name, fallback=None)
    except (KeyError, AttributeError):
        return None


def _normalize_sport(sport: str, sub_sport: str = "") -> str:
    """Normalize sport names from FIT files.

    Strategy: minimal cleanup, preserve original names.
    The agent handles sport semantics — we just clean up formatting.
    """
    sport = sport.lower().strip().replace(" ", "_")
    sub_sport = sub_sport.lower().strip().replace(" ", "_") if sub_sport else ""

    # Use sub_sport if it's more specific than a generic container
    _GENERIC_CONTAINERS = {"training", "fitness_equipment", "racket", "generic"}
    if sport in _GENERIC_CONTAINERS and sub_sport:
        return sub_sport

    return sport if sport else "unknown"


def _build_zone_distribution(hr_zones_raw: list | tuple) -> dict:
    """Convert time_in_hr_zone tuple to zone distribution dict.

    Index 0 = below zone 1 (add to zone_1).
    Indices 1-5 = zones 1-5.
    Index 6+ = above zone 5 (add to zone_5).
    """
    zones = {f"zone_{i}_seconds": 0.0 for i in range(1, 6)}

    for i, val in enumerate(hr_zones_raw):
        if not isinstance(val, (int, float)):
            continue
        if i == 0:
            # Below zone 1 -> add to zone 1
            zones["zone_1_seconds"] += float(val)
        elif 1 <= i <= 5:
            zones[f"zone_{i}_seconds"] += float(val)
        else:
            # Above zone 5 -> add to zone 5
            zones["zone_5_seconds"] += float(val)

    return zones


def _build_hr(hr_values: list) -> dict | None:
    """Build heart rate summary from recorded values."""
    if not hr_values:
        return None
    return {
        "avg": round(sum(hr_values) / len(hr_values)),
        "max": round(max(hr_values)),
        "min": round(min(hr_values)),
    }


def _build_pace(speed_values: list, sport: str, distance_meters: float | None = None) -> dict | None:
    """Build pace summary from speed values (m/s).

    Returns pace in min/km for running and walking.
    Returns pace in min/100m for swimming (when distance < 10000m).
    """
    if not speed_values:
        return None

    # Pace for running and walking (min/km)
    if sport in ("running", "walking"):
        valid_speeds = [s for s in speed_values if s > 0.1]
        if not valid_speeds:
            return None
        avg_speed = sum(valid_speeds) / len(valid_speeds)  # m/s
        max_speed = max(valid_speeds)
        return {
            "avg_min_per_km": round(1000 / (avg_speed * 60), 2) if avg_speed > 0 else None,
            "best_min_per_km": round(1000 / (max_speed * 60), 2) if max_speed > 0 else None,
        }

    # Pace for swimming (min/100m)
    if sport == "swimming" and distance_meters is not None and distance_meters < 10000:
        valid_speeds = [s for s in speed_values if s > 0.01]
        if not valid_speeds:
            return None
        avg_speed = sum(valid_speeds) / len(valid_speeds)  # m/s
        max_speed = max(valid_speeds)
        return {
            "avg_min_per_100m": round(100 / (avg_speed * 60), 2) if avg_speed > 0 else None,
            "best_min_per_100m": round(100 / (max_speed * 60), 2) if max_speed > 0 else None,
        }

    return None


def _build_speed(speed_values: list, sport: str) -> dict | None:
    """Build speed summary (km/h) for cycling, rowing, sup, and other non-running/walking sports."""
    if not speed_values:
        return None
    # Speed is for sports that don't use pace
    if sport in ("running", "walking", "swimming"):
        return None

    valid_speeds = [s for s in speed_values if s > 0.1]
    if not valid_speeds:
        return None
    avg_speed = sum(valid_speeds) / len(valid_speeds)  # m/s
    max_speed = max(valid_speeds)
    return {
        "avg_km_h": round(avg_speed * 3.6, 2),
        "max_km_h": round(max_speed * 3.6, 2),
    }


def _build_power(power_values: list) -> dict | None:
    """Build power summary from recorded values."""
    if not power_values:
        return None
    return {
        "avg_watts": round(sum(power_values) / len(power_values)),
        "max_watts": round(max(power_values)),
        "normalized_watts": None,  # Would need proper NP calculation
    }


def _build_elevation(altitude_values: list) -> dict | None:
    """Build elevation summary from altitude records."""
    if len(altitude_values) < 2:
        return None
    gain = 0.0
    loss = 0.0
    for i in range(1, len(altitude_values)):
        diff = altitude_values[i] - altitude_values[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss += abs(diff)
    return {
        "gain_meters": round(gain),
        "loss_meters": round(loss),
    }
