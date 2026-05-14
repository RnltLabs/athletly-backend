"""Generate realistic synthetic health_daily_metrics rows for a persona.

30 days of daily metrics with realistic noise around each persona's
baseline. Deterministic per (slug, date) so reruns are no-ops.

Columns covered:
    resting_heart_rate, hrv_avg, sleep_score, sleep_duration_minutes,
    sleep_deep_minutes, sleep_light_minutes, sleep_rem_minutes,
    sleep_awake_minutes, stress_avg, body_battery_high, body_battery_low,
    recovery_score, steps, active_calories, total_calories, vo2max,
    spo2_avg, respiration_avg, intensity_minutes, floors_climbed
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from tools.persona_test._persona import Persona


@dataclass(frozen=True)
class SyntheticDailyMetric:
    """A daily metric row ready to upsert into health_daily_metrics."""

    user_id: str
    date: str  # ISO YYYY-MM-DD
    resting_heart_rate: int
    hrv_avg: float
    sleep_score: int
    sleep_duration_minutes: float
    sleep_deep_minutes: float
    sleep_light_minutes: float
    sleep_rem_minutes: float
    sleep_awake_minutes: float
    stress_avg: int
    body_battery_high: int
    body_battery_low: int
    recovery_score: int
    steps: int
    active_calories: int
    total_calories: int
    floors_climbed: int
    intensity_minutes: int
    spo2_avg: float
    respiration_avg: float
    vo2max: float
    source: str = "garmin"

    def to_row(self) -> dict:
        return {
            "user_id": self.user_id,
            "date": self.date,
            "source": self.source,
            "resting_heart_rate": self.resting_heart_rate,
            "hrv_avg": self.hrv_avg,
            "sleep_score": self.sleep_score,
            "sleep_duration_minutes": self.sleep_duration_minutes,
            "sleep_deep_minutes": self.sleep_deep_minutes,
            "sleep_light_minutes": self.sleep_light_minutes,
            "sleep_rem_minutes": self.sleep_rem_minutes,
            "sleep_awake_minutes": self.sleep_awake_minutes,
            "stress_avg": self.stress_avg,
            "body_battery_high": self.body_battery_high,
            "body_battery_low": self.body_battery_low,
            "recovery_score": self.recovery_score,
            "steps": self.steps,
            "active_calories": self.active_calories,
            "total_calories": self.total_calories,
            "floors_climbed": self.floors_climbed,
            "intensity_minutes": self.intensity_minutes,
            "spo2_avg": self.spo2_avg,
            "respiration_avg": self.respiration_avg,
            "vo2max": self.vo2max,
            "raw_data": {"synthetic": True},
        }


def generate(
    persona: Persona,
    user_id: str,
    today: date | None = None,
    days: int = 30,
) -> list[SyntheticDailyMetric]:
    """Generate *days* worth of health metrics ending on *today* (inclusive)."""
    if today is None:
        today = datetime.now(timezone.utc).date()

    out: list[SyntheticDailyMetric] = []
    for offset in range(days):
        d = today - timedelta(days=offset)
        rng = random.Random(hash((persona.slug, d.isoformat(), "metric")) & 0xFFFFFFFF)
        out.append(_one_day(persona, user_id, d, rng))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _one_day(
    persona: Persona,
    user_id: str,
    d: date,
    rng: random.Random,
) -> SyntheticDailyMetric:
    """Build one synthetic day of metrics for the persona."""
    # Sleep around persona baseline with realistic noise + occasional bad nights.
    base_sleep_h = persona.sleep_hours_baseline
    bad_night = rng.random() < 0.10
    if bad_night:
        sleep_h = max(4.5, base_sleep_h - rng.uniform(1.0, 2.5))
    else:
        sleep_h = max(4.0, base_sleep_h + rng.uniform(-0.7, 0.7))

    sleep_dur_min = round(sleep_h * 60, 1)
    # Phase split (proportions tuned for adult athletes).
    sleep_deep = round(sleep_dur_min * rng.uniform(0.12, 0.20), 1)
    sleep_rem = round(sleep_dur_min * rng.uniform(0.18, 0.25), 1)
    sleep_awake = round(sleep_dur_min * rng.uniform(0.02, 0.07), 1)
    sleep_light = round(sleep_dur_min - sleep_deep - sleep_rem - sleep_awake, 1)
    sleep_score = max(20, min(100, int(85 - (8 - sleep_h) * 9 + rng.uniform(-6, 6))))

    # HRV: baseline +/- noise, bigger drop on bad nights.
    hrv_base = persona.hrv_baseline_ms
    hrv = round(
        hrv_base + (rng.uniform(-6, 6) if not bad_night else rng.uniform(-15, -6)),
        1,
    )
    hrv = max(15.0, hrv)

    # RHR: inverse correlation with HRV.
    rhr = persona.rhr_baseline_bpm + int(round((hrv_base - hrv) / 6 + rng.uniform(-2, 2)))
    rhr = max(35, min(85, rhr))

    # Stress: higher when sleep poor.
    stress = max(15, min(85, int(30 + (7.5 - sleep_h) * 6 + rng.uniform(-8, 8))))
    body_battery_high = max(20, min(100, int(85 - stress / 3 + rng.uniform(-5, 5))))
    body_battery_low = max(5, min(40, int(20 - (stress - 30) / 4 + rng.uniform(-4, 4))))

    recovery = max(15, min(100, int(80 - stress / 2 + (hrv - hrv_base) + rng.uniform(-6, 6))))

    # Activity totals.
    steps = int(rng.uniform(6500, 14500))
    active_cal = int(rng.uniform(350, 950))
    total_cal = int(active_cal + rng.uniform(1500, 1900))
    floors = int(rng.uniform(4, 22))
    intensity_min = int(rng.uniform(20, 90))

    spo2 = round(96 + rng.uniform(-0.5, 1.0), 1)
    resp = round(14.5 + rng.uniform(-1.0, 1.0), 1)

    # VO2max: drifts slowly around persona baseline.
    vo2 = round(persona.estimated_vo2max + rng.uniform(-0.5, 0.5), 1)

    return SyntheticDailyMetric(
        user_id=user_id,
        date=d.isoformat(),
        resting_heart_rate=rhr,
        hrv_avg=hrv,
        sleep_score=sleep_score,
        sleep_duration_minutes=sleep_dur_min,
        sleep_deep_minutes=sleep_deep,
        sleep_light_minutes=sleep_light,
        sleep_rem_minutes=sleep_rem,
        sleep_awake_minutes=sleep_awake,
        stress_avg=stress,
        body_battery_high=body_battery_high,
        body_battery_low=body_battery_low,
        recovery_score=recovery,
        steps=steps,
        active_calories=active_cal,
        total_calories=total_cal,
        floors_climbed=floors,
        intensity_minutes=intensity_min,
        spo2_avg=spo2,
        respiration_avg=resp,
        vo2max=vo2,
    )
