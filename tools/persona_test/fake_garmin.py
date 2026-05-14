"""Generate realistic synthetic Garmin activity history for a persona.

The generator is deterministic in the (persona.slug, weeks_of_history)
key so repeated seed runs produce the same activities. Activity ids are
``persona_<slug>_<date>_<sport>_<seq>`` so the upsert constraint on
``(user_id, garmin_activity_id)`` hits and re-runs become no-ops.

Design choices:
- Random seeded by ``hash((slug, week_index, day_index))`` so individual
  sessions vary plausibly but the overall distribution is reproducible.
- Pace and HR are derived from each persona's threshold pace and VO2max
  so the numbers stay consistent with the persona definition.
- Per-sport templates (running, cycling, swimming) keep the data
  schema-correct for the agent's data tools.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from tools.persona_test._persona import Persona


@dataclass(frozen=True)
class SyntheticActivity:
    """A fake Garmin activity ready to insert into the activities table."""

    user_id: str
    garmin_activity_id: str
    sport: str
    start_time: str  # ISO 8601 with tz
    duration_seconds: int
    distance_meters: float
    avg_hr: int
    max_hr: int
    avg_pace_min_km: float
    elevation_gain_m: float
    trimp: float
    calories: int
    training_effect: float
    zone_distribution: dict
    laps: list
    raw_data: dict
    source: str = "garmin"

    def to_row(self) -> dict:
        """Render to a Supabase activities-table row dict."""
        return {
            "user_id": self.user_id,
            "sport": self.sport,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "distance_meters": self.distance_meters,
            "avg_hr": self.avg_hr,
            "max_hr": self.max_hr,
            "avg_pace_min_km": self.avg_pace_min_km,
            "elevation_gain_m": self.elevation_gain_m,
            "trimp": self.trimp,
            "zone_distribution": self.zone_distribution,
            "laps": self.laps,
            "raw_data": self.raw_data,
            "source": self.source,
            "garmin_activity_id": self.garmin_activity_id,
            "calories": self.calories,
            "training_effect": self.training_effect,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(persona: Persona, user_id: str, today: date | None = None) -> list[SyntheticActivity]:
    """Generate the persona's training history.

    Args:
        persona: parsed persona definition.
        user_id: Supabase auth user id to attach activities to.
        today: anchor date; defaults to today UTC. Pass a fixed date in tests.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()

    if "running" in persona.sports and persona.goal_type == "marathon":
        return _generate_running_marathon(persona, user_id, today)
    if "running" in persona.sports and persona.goal_type == "half_marathon":
        return _generate_running_beginner_hm(persona, user_id, today)
    if persona.goal_type == "ironman":
        return _generate_triathlon(persona, user_id, today)

    # Fallback: generic running, low volume.
    return _generate_running_beginner_hm(persona, user_id, today)


# ---------------------------------------------------------------------------
# Marathon (Elena profile)
# ---------------------------------------------------------------------------


def _generate_running_marathon(
    persona: Persona,
    user_id: str,
    today: date,
) -> list[SyntheticActivity]:
    """5 sessions/week, progressive volume, 2 recovery weeks."""
    # Progression in km per week, oldest-first.
    weekly_volume_curve = [50, 53, 56, 58, 60, 62, 64, 66, 68, 70, 65, 60]
    weeks = min(persona.weeks_of_history, len(weekly_volume_curve))
    curve = weekly_volume_curve[-weeks:]

    out: list[SyntheticActivity] = []
    for week_idx, weekly_km in enumerate(curve):
        # Distribute volume across 5 sessions.
        # Long: 35 percent; threshold: 18 percent; easy x2: 28 percent; recovery: 10 percent.
        long_km = weekly_km * 0.35
        threshold_km = weekly_km * 0.18
        easy_km = weekly_km * 0.14
        recovery_km = weekly_km * 0.10

        # Week starts Monday. Most-recent week is the one ending TODAY (or
        # very close to it). We pin the FIRST week's Monday to today minus
        # (weeks-1)*7 days, then walk forward.
        week_start = today - timedelta(days=(weeks - 1 - week_idx) * 7 + today.weekday())

        sessions = [
            ("recovery", 0, recovery_km, "easy"),
            ("threshold", 1, threshold_km, "threshold"),
            ("easy", 2, easy_km, "easy"),
            ("easy", 3, easy_km, "easy"),
            ("long", 5, long_km, "long"),
        ]
        for seq, (label, dow, target_km, kind) in enumerate(sessions):
            d = week_start + timedelta(days=dow)
            if d > today:
                continue  # don't generate future sessions
            rng = _rng(persona.slug, week_idx, seq)
            activity = _make_running_session(
                persona=persona,
                user_id=user_id,
                target_km=target_km,
                kind=kind,
                run_date=d,
                seq=seq,
                rng=rng,
            )
            out.append(activity)
    return out


# ---------------------------------------------------------------------------
# Beginner half-marathon (Marco profile)
# ---------------------------------------------------------------------------


def _generate_running_beginner_hm(
    persona: Persona,
    user_id: str,
    today: date,
) -> list[SyntheticActivity]:
    """3 sessions/week, inconsistent, mostly moderate-intensity rut."""
    weeks = persona.weeks_of_history
    out: list[SyntheticActivity] = []
    for week_idx in range(weeks):
        rng = _rng(persona.slug, week_idx, -1)
        # 1 in 4 weeks: missed week (only 1 session).
        is_missed = (week_idx % 4 == 0)
        sessions_per_week = 1 if is_missed else 3
        week_start = today - timedelta(days=(weeks - 1 - week_idx) * 7 + today.weekday())
        # Mon, Wed, Sat
        dows = [0, 2, 5][:sessions_per_week]
        # Volume: 18 to 26 km, low variance.
        weekly_km = 22 + rng.uniform(-4, 4)
        # Roughly even split between sessions, with occasional "all-in" run.
        for seq, dow in enumerate(dows):
            d = week_start + timedelta(days=dow)
            if d > today:
                continue
            rng_s = _rng(persona.slug, week_idx, seq)
            kind = "moderate"
            target_km = weekly_km / max(1, len(dows))
            # 1 in 10 sessions: an "all-in" overdistance.
            if rng_s.random() < 0.1:
                kind = "overdistance"
                target_km = 11 + rng_s.uniform(-1, 1)
            activity = _make_running_session(
                persona=persona,
                user_id=user_id,
                target_km=target_km,
                kind=kind,
                run_date=d,
                seq=seq,
                rng=rng_s,
            )
            out.append(activity)
    return out


# ---------------------------------------------------------------------------
# Triathlete (Lisa profile)
# ---------------------------------------------------------------------------


def _generate_triathlon(
    persona: Persona,
    user_id: str,
    today: date,
) -> list[SyntheticActivity]:
    """6 days/week, multi-sport, with one brick per week."""
    weeks = persona.weeks_of_history
    out: list[SyntheticActivity] = []
    # Weekly template (day-of-week -> [(sport, kind, target)]). Two-a-days
    # use seq 0/1 within the same date so synthetic ids stay unique.
    template = {
        0: [("swimming", "intervals", 3.0)],  # Mon
        1: [("running", "threshold", 13.0), ("swimming", "endurance", 2.5)],  # Tue
        2: [("cycling", "endurance", 65.0)],  # Wed
        3: [("running", "intervals", 10.0)],  # Thu
        4: [("swimming", "endurance", 3.0)],  # Fri
        5: [("cycling", "long", 110.0)],  # Sat
        6: [("cycling", "brick_bike", 75.0), ("running", "brick_run", 8.0)],  # Sun
    }

    for week_idx in range(weeks):
        week_start = today - timedelta(days=(weeks - 1 - week_idx) * 7 + today.weekday())
        for dow, sessions in template.items():
            d = week_start + timedelta(days=dow)
            if d > today:
                continue
            for seq, (sport, kind, target) in enumerate(sessions):
                rng = _rng(persona.slug, week_idx, dow * 10 + seq)
                if sport == "running":
                    activity = _make_running_session(
                        persona=persona,
                        user_id=user_id,
                        target_km=target,
                        kind=kind,
                        run_date=d,
                        seq=dow * 10 + seq,
                        rng=rng,
                    )
                elif sport == "cycling":
                    activity = _make_cycling_session(
                        persona=persona,
                        user_id=user_id,
                        target_km=target,
                        kind=kind,
                        ride_date=d,
                        seq=dow * 10 + seq,
                        rng=rng,
                    )
                else:  # swimming
                    activity = _make_swim_session(
                        persona=persona,
                        user_id=user_id,
                        target_km=target,
                        kind=kind,
                        swim_date=d,
                        seq=dow * 10 + seq,
                        rng=rng,
                    )
                out.append(activity)
    return out


# ---------------------------------------------------------------------------
# Per-sport session builders
# ---------------------------------------------------------------------------


def _make_running_session(
    *,
    persona: Persona,
    user_id: str,
    target_km: float,
    kind: str,
    run_date: date,
    seq: int,
    rng: random.Random,
) -> SyntheticActivity:
    """Produce a single running activity row for the given persona / kind."""
    # Resolve pace and HR by kind from the persona's reference paces.
    threshold = persona.threshold_pace_min_km
    if kind == "easy":
        pace = threshold + 1.00 + rng.uniform(-0.10, 0.20)
        avg_hr = persona.rhr_baseline_bpm + 90 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(6, 14)
        z_dist = {"z1": 0.10, "z2": 0.80, "z3": 0.08, "z4": 0.02, "z5": 0.0}
        te = 2.5 + rng.uniform(-0.3, 0.5)
    elif kind == "long":
        pace = threshold + 0.85 + rng.uniform(-0.15, 0.25)
        avg_hr = persona.rhr_baseline_bpm + 100 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(10, 18)
        z_dist = {"z1": 0.05, "z2": 0.65, "z3": 0.20, "z4": 0.08, "z5": 0.02}
        te = 3.5 + rng.uniform(-0.3, 0.5)
    elif kind == "threshold":
        pace = threshold + 0.10 + rng.uniform(-0.10, 0.10)
        avg_hr = persona.rhr_baseline_bpm + 120 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(8, 15)
        z_dist = {"z1": 0.10, "z2": 0.40, "z3": 0.15, "z4": 0.30, "z5": 0.05}
        te = 4.0 + rng.uniform(-0.3, 0.5)
    elif kind == "intervals":
        pace = threshold - 0.10 + rng.uniform(-0.10, 0.05)
        avg_hr = persona.rhr_baseline_bpm + 125 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(10, 18)
        z_dist = {"z1": 0.10, "z2": 0.35, "z3": 0.15, "z4": 0.30, "z5": 0.10}
        te = 4.2 + rng.uniform(-0.3, 0.5)
    elif kind == "brick_run":
        pace = threshold + 0.50 + rng.uniform(-0.10, 0.15)
        avg_hr = persona.rhr_baseline_bpm + 110 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(8, 14)
        z_dist = {"z1": 0.05, "z2": 0.60, "z3": 0.25, "z4": 0.08, "z5": 0.02}
        te = 3.5 + rng.uniform(-0.3, 0.5)
    elif kind == "moderate":
        pace = threshold + 0.30 + rng.uniform(-0.10, 0.20)
        avg_hr = persona.rhr_baseline_bpm + 100 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(8, 16)
        z_dist = {"z1": 0.0, "z2": 0.25, "z3": 0.60, "z4": 0.14, "z5": 0.01}
        te = 3.2 + rng.uniform(-0.3, 0.5)
    elif kind == "overdistance":
        pace = threshold + 0.10 + rng.uniform(-0.05, 0.15)
        avg_hr = persona.rhr_baseline_bpm + 115 + rng.randint(-5, 5)
        max_hr = avg_hr + rng.randint(10, 16)
        z_dist = {"z1": 0.0, "z2": 0.10, "z3": 0.35, "z4": 0.45, "z5": 0.10}
        te = 4.5 + rng.uniform(-0.3, 0.3)
    else:
        pace = threshold + 0.50
        avg_hr = persona.rhr_baseline_bpm + 100
        max_hr = avg_hr + 10
        z_dist = {"z2": 1.0}
        te = 3.0

    distance_km = max(2.0, target_km + rng.uniform(-0.5, 0.5))
    distance_m = round(distance_km * 1000, 1)
    duration_seconds = int(pace * 60 * distance_km)

    # Heuristic elevation: 3-12 m per km, slightly higher on long.
    elev = round(distance_km * rng.uniform(3.0, 12.0), 0)
    if kind == "long":
        elev *= 1.4

    trimp = round(duration_seconds / 60.0 * (avg_hr / 180.0) ** 1.92, 1)
    calories = int(distance_km * 60 + duration_seconds * 0.05)

    start_dt = datetime.combine(run_date, time(hour=18, minute=15, tzinfo=timezone.utc))
    # Persona has training time variance.
    start_dt = start_dt - timedelta(hours=rng.randint(0, 10))

    activity_id = _synthetic_id(persona.slug, run_date, "running", seq)

    return SyntheticActivity(
        user_id=user_id,
        garmin_activity_id=activity_id,
        sport="running",
        start_time=start_dt.isoformat(),
        duration_seconds=duration_seconds,
        distance_meters=distance_m,
        avg_hr=int(avg_hr),
        max_hr=int(max_hr),
        avg_pace_min_km=round(pace, 2),
        elevation_gain_m=elev,
        trimp=trimp,
        calories=calories,
        training_effect=round(te, 1),
        zone_distribution={k: _zone_seconds(duration_seconds, v) for k, v in z_dist.items()},
        laps=_build_running_laps(distance_km, pace, kind, rng),
        raw_data={
            "synthetic": True,
            "kind": kind,
            "persona_slug": persona.slug,
        },
    )


def _make_cycling_session(
    *,
    persona: Persona,
    user_id: str,
    target_km: float,
    kind: str,
    ride_date: date,
    seq: int,
    rng: random.Random,
) -> SyntheticActivity:
    """Build a synthetic cycling activity using the persona's FTP."""
    ftp = persona.ftp_watts or 200
    if kind == "long":
        np = ftp * (0.78 + rng.uniform(-0.03, 0.03))
        speed_kmh = 30.0 + rng.uniform(-2, 2)
        avg_hr = persona.rhr_baseline_bpm + 95 + rng.randint(-4, 4)
    elif kind == "threshold":
        np = ftp * (0.92 + rng.uniform(-0.03, 0.04))
        speed_kmh = 33.0 + rng.uniform(-2, 2)
        avg_hr = persona.rhr_baseline_bpm + 115 + rng.randint(-4, 4)
    elif kind == "brick_bike":
        np = ftp * (0.80 + rng.uniform(-0.03, 0.04))
        speed_kmh = 31.0 + rng.uniform(-2, 2)
        avg_hr = persona.rhr_baseline_bpm + 100 + rng.randint(-4, 4)
    else:  # endurance
        np = ftp * (0.72 + rng.uniform(-0.03, 0.03))
        speed_kmh = 29.0 + rng.uniform(-2, 2)
        avg_hr = persona.rhr_baseline_bpm + 85 + rng.randint(-4, 4)

    duration_h = target_km / speed_kmh
    duration_seconds = int(duration_h * 3600)
    distance_m = round(target_km * 1000, 1)
    pace_min_per_km = (duration_seconds / 60) / target_km
    elev = round(target_km * rng.uniform(8, 18), 0)
    trimp = round(duration_seconds / 60.0 * (avg_hr / 180.0) ** 1.92, 1)
    calories = int(np * duration_h * 3.6)

    start_dt = datetime.combine(ride_date, time(hour=8, minute=0, tzinfo=timezone.utc))

    return SyntheticActivity(
        user_id=user_id,
        garmin_activity_id=_synthetic_id(persona.slug, ride_date, "cycling", seq),
        sport="cycling",
        start_time=start_dt.isoformat(),
        duration_seconds=duration_seconds,
        distance_meters=distance_m,
        avg_hr=int(avg_hr),
        max_hr=int(avg_hr + rng.randint(12, 20)),
        avg_pace_min_km=round(pace_min_per_km, 2),
        elevation_gain_m=elev,
        trimp=trimp,
        calories=calories,
        training_effect=round(3.0 + rng.uniform(-0.3, 1.0), 1),
        zone_distribution={"z2": int(duration_seconds * 0.7), "z3": int(duration_seconds * 0.2), "z4": int(duration_seconds * 0.1)},
        laps=[],
        raw_data={
            "synthetic": True,
            "kind": kind,
            "persona_slug": persona.slug,
            "normalized_power_w": round(np, 0),
            "ftp_w": ftp,
            "intensity_factor": round(np / ftp, 2),
        },
    )


def _make_swim_session(
    *,
    persona: Persona,
    user_id: str,
    target_km: float,
    kind: str,
    swim_date: date,
    seq: int,
    rng: random.Random,
) -> SyntheticActivity:
    """Build a synthetic swimming activity using the persona's CSS."""
    css_min_per_100m = persona.swim_css_min_per_100m or 1.80
    if kind == "intervals":
        pace_per_100 = css_min_per_100m - 0.03 + rng.uniform(-0.03, 0.03)
    else:
        pace_per_100 = css_min_per_100m + 0.10 + rng.uniform(-0.03, 0.05)

    distance_m = round(target_km * 1000, 0)
    duration_seconds = int(pace_per_100 * 60 * (distance_m / 100))
    avg_hr = persona.rhr_baseline_bpm + 70 + rng.randint(-4, 4)
    calories = int(distance_m * 0.7)

    start_dt = datetime.combine(swim_date, time(hour=6, minute=45, tzinfo=timezone.utc))

    return SyntheticActivity(
        user_id=user_id,
        garmin_activity_id=_synthetic_id(persona.slug, swim_date, "swimming", seq),
        sport="swimming",
        start_time=start_dt.isoformat(),
        duration_seconds=duration_seconds,
        distance_meters=distance_m,
        avg_hr=int(avg_hr),
        max_hr=int(avg_hr + rng.randint(8, 14)),
        avg_pace_min_km=round(pace_per_100 * 10, 2),  # min/km equivalent
        elevation_gain_m=0,
        trimp=round(duration_seconds / 60.0 * (avg_hr / 180.0) ** 1.92, 1),
        calories=calories,
        training_effect=round(2.5 + rng.uniform(-0.2, 0.6), 1),
        zone_distribution={"z2": int(duration_seconds * 0.7), "z3": int(duration_seconds * 0.3)},
        laps=[],
        raw_data={
            "synthetic": True,
            "kind": kind,
            "persona_slug": persona.slug,
            "pace_min_per_100m": round(pace_per_100, 2),
            "css_min_per_100m": css_min_per_100m,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_running_laps(distance_km: float, pace: float, kind: str, rng: random.Random) -> list:
    """Build a believable per-km laps array."""
    laps = []
    full_km = int(distance_km)
    for km in range(1, full_km + 1):
        jitter = rng.uniform(-0.10, 0.10) if kind != "threshold" else rng.uniform(-0.06, 0.06)
        laps.append({
            "lap_index": km,
            "distance_m": 1000,
            "duration_s": int((pace + jitter) * 60),
            "avg_pace_min_km": round(pace + jitter, 2),
        })
    return laps


def _zone_seconds(duration_seconds: int, ratio: float) -> int:
    return int(round(duration_seconds * ratio))


def _synthetic_id(slug: str, d: date, sport: str, seq: int) -> str:
    """Deterministic synthetic Garmin activity id used for idempotent upsert."""
    return f"persona_{slug}_{d.isoformat()}_{sport}_{seq}"


def _rng(slug: str, week_idx: int, seq: int) -> random.Random:
    """Deterministic per-session RNG."""
    return random.Random(hash((slug, week_idx, seq)) & 0xFFFFFFFF)
