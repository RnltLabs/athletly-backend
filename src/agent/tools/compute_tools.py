"""Compute tool -- deterministic sport-science math the agent can trust.

The agent calls `compute_sport_math(formula=..., inputs=...)` instead
of doing FTP/NP/VDOT/CSS/HR-zone arithmetic inline. Each formula here
is a PURE function: no DB, no I/O, no LLM. Outputs are deterministic
and unit-tested against published reference tables (Jack Daniels for
VDOT, Coggan for FTP zones and NP, Swim Smooth for CSS, Karvonen for
HRR).

Why this tool exists:
    Lisa-bug (Sprint E): Haiku confabulated NP=352W from FTP=240W, an
    algebraically impossible ratio for a sustainable interval. Math
    discipline collapses under multi-step reasoning load. The fix is
    to remove LLM arithmetic from the equation entirely - the LLM
    decides WHAT to compute, the tool computes IT, the LLM quotes
    the result verbatim. The tool description teaches this discipline;
    the STRICT system-prompt rule enforces it.

Cited sources for the formulas (see RESEARCH.md):
    - Daniels J. Daniels' Running Formula, 3rd ed. (2014).
    - Coggan AR, Allen H. Training and Racing with a Power Meter, 3rd ed. (2019).
    - Karvonen MJ et al. Ann Med Exp Biol Fenn 1957; 35(3):307 to 315.
    - Swim Smooth CSS test (swimsmooth.com).
"""

from __future__ import annotations

import math
from typing import Any

from src.agent.tools.registry import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Pure formula implementations.
# Each returns a dict. On bad input: {"status": "error", "message": "..."}.
# On success: {"status": "success", ...result keys}.
# ---------------------------------------------------------------------------


def _format_mmss(seconds: float) -> str:
    """Format a duration in seconds as mm:ss, never negative, always 2-digit sec."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _percent_vo2max_for_time(time_minutes: float) -> float:
    """Daniels' percent-VO2max-sustained-for-T-minutes curve.

    From Jack Daniels, Daniels' Running Formula 3rd ed., p. 80. Returns
    a fraction (0.7 to ~1.0), not a percentage.
    """
    return (
        0.8
        + 0.1894393 * math.exp(-0.012778 * time_minutes)
        + 0.2989558 * math.exp(-0.1932605 * time_minutes)
    )


def _vo2_demand_for_velocity(velocity_m_per_min: float) -> float:
    """Daniels' VO2 demand for a steady velocity (m/min).

    Returns ml/kg/min. Reference: p. 80 of Daniels 2014.
    """
    v = velocity_m_per_min
    return -4.60 + 0.182258 * v + 0.000104 * v * v


def vdot_from_race_time(distance_km: float, time_seconds: float) -> dict:
    """Compute VDOT (Daniels equivalent VO2max) from a race performance.

    Reference: Jack Daniels, "Daniels' Running Formula", 3rd ed. (2014).
    Validation: 5k 20:00 -> VDOT ~49.2; 10k 40:00 -> VDOT ~52.0.
    """
    try:
        distance_m = float(distance_km) * 1000.0
        time_s = float(time_seconds)
    except (TypeError, ValueError):
        return {"status": "error", "message": "distance_km and time_seconds must be numbers"}

    if distance_m <= 0 or time_s <= 0:
        return {"status": "error", "message": "distance_km and time_seconds must be positive"}

    time_min = time_s / 60.0
    velocity_m_per_min = distance_m / time_min
    vo2_demand = _vo2_demand_for_velocity(velocity_m_per_min)
    pct_vo2max = _percent_vo2max_for_time(time_min)
    if pct_vo2max <= 0:
        return {"status": "error", "message": "internal: invalid pct VO2max"}
    vdot = vo2_demand / pct_vo2max

    return {
        "status": "success",
        "vdot": round(vdot, 1),
        "velocity_m_per_min": round(velocity_m_per_min, 1),
        "interpretation": (
            f"VDOT {vdot:.1f} entspricht einer effektiven VO2max von "
            f"{vdot:.0f} ml/kg/min nach Jack Daniels."
        ),
    }


def _seconds_per_km_at_vo2(vo2_target: float) -> float:
    """Invert Daniels velocity-VO2 curve: solve for velocity at target VO2.

    The quadratic 0.000104 v^2 + 0.182258 v - (4.60 + vo2) = 0 has one
    positive root. Returns seconds-per-kilometer at that velocity.
    """
    a, b, c = 0.000104, 0.182258, -(4.60 + vo2_target)
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    v = (-b + math.sqrt(disc)) / (2 * a)  # m/min
    if v <= 0:
        return float("inf")
    return 60.0 / (v / 1000.0)  # s/km


def paces_from_vdot(vdot: float) -> dict:
    """Daniels training paces from VDOT.

    Returns paces for easy, marathon, threshold, interval, repetition
    sessions. Percent VO2max anchors used here match Daniels 2014:

      Easy: midpoint of 65-79 percent of VO2max
      Marathon: 80 percent
      Threshold: 88 percent
      Interval: 98 percent
      Repetition: 105 percent

    Validation: VDOT 50 should yield threshold ~4:18/km, marathon
    ~4:34/km (Daniels table 5.2 reference values).
    """
    try:
        v = float(vdot)
    except (TypeError, ValueError):
        return {"status": "error", "message": "vdot must be a number"}
    if v < 20 or v > 90:
        return {
            "status": "error",
            "message": f"vdot {v} outside plausible range (20 to 90)",
        }

    intensities = {
        "easy": 0.72,
        "marathon": 0.80,
        "threshold": 0.88,
        "interval": 0.98,
        "repetition": 1.05,
    }
    paces: dict[str, Any] = {}
    for name, pct in intensities.items():
        vo2_target = v * pct
        sec_per_km = _seconds_per_km_at_vo2(vo2_target)
        paces[f"{name}_per_km_seconds"] = round(sec_per_km, 1)
        paces[f"{name}_per_km"] = _format_mmss(sec_per_km)

    return {
        "status": "success",
        "vdot": round(v, 1),
        **paces,
        "interpretation": (
            f"Daniels Trainingstempi fuer VDOT {v:.1f}: Easy "
            f"{paces['easy_per_km']}, Marathon {paces['marathon_per_km']}, "
            f"Schwelle {paces['threshold_per_km']}, Intervall "
            f"{paces['interval_per_km']}, Repetition "
            f"{paces['repetition_per_km']} pro km."
        ),
    }


def ftp_zones(ftp_watts: float) -> dict:
    """Coggan 7-zone power model from FTP.

    Reference: Coggan AR, Allen H. Training and Racing with a Power
    Meter, 3rd ed. (2019). Reproduced at TrainingPeaks.

    Returns each zone as [low, high] Watts. The upper bound of Z7 is
    None (open-ended).

    Validation: FTP 240W -> Z4 218-252, Z5 254-288 (rounded to whole W).
    """
    try:
        ftp = float(ftp_watts)
    except (TypeError, ValueError):
        return {"status": "error", "message": "ftp_watts must be a number"}
    if ftp <= 0:
        return {"status": "error", "message": "ftp_watts must be positive"}

    # Zone percent bands (low_pct, high_pct) per Coggan. None on high
    # for Z7 means open-ended.
    bands = [
        ("z1", "Active Recovery", 0.0, 0.55),
        ("z2", "Endurance", 0.56, 0.75),
        ("z3", "Tempo", 0.76, 0.90),
        ("z4", "Lactate Threshold", 0.91, 1.05),
        ("z5", "VO2max", 1.06, 1.20),
        ("z6", "Anaerobic Capacity", 1.21, 1.50),
        ("z7", "Neuromuscular Power", 1.51, None),
    ]
    zones: dict[str, Any] = {}
    for code, label, lo, hi in bands:
        low_w = round(ftp * lo)
        high_w = round(ftp * hi) if hi is not None else None
        zones[code] = {
            "label": label,
            "low_watts": low_w,
            "high_watts": high_w,
            "pct_ftp_low": round(lo * 100),
            "pct_ftp_high": round(hi * 100) if hi is not None else None,
        }

    sanity_warning = None
    if ftp < 80 or ftp > 500:
        sanity_warning = (
            f"FTP {ftp:.0f} W liegt ausserhalb des typischen "
            f"Erwachsenenbereichs (80 bis 500 W). Bitte Wert pruefen."
        )

    return {
        "status": "success",
        "ftp_watts": round(ftp, 1),
        **zones,
        "sanity_warning": sanity_warning,
        "interpretation": (
            f"Coggan 7-Zonen fuer FTP {ftp:.0f} W: Z2 "
            f"{zones['z2']['low_watts']}-{zones['z2']['high_watts']} W, "
            f"Z4 {zones['z4']['low_watts']}-{zones['z4']['high_watts']} W, "
            f"Z5 {zones['z5']['low_watts']}-{zones['z5']['high_watts']} W."
        ),
    }


def _rolling_average(values: list[float], window: int) -> list[float]:
    """Plain rolling-mean over a 1-second-per-sample list, window in seconds.

    The first ``window-1`` samples use a partial window (mean of
    everything available so far). This matches the Coggan convention
    of "the rolling average ramps from the workout start" rather than
    discarding the leading samples.
    """
    if not values:
        return []
    out: list[float] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        denom = min(i + 1, window)
        out.append(running / denom)
    return out


def np_from_intervals(intervals: list, ftp_watts: float | None = None) -> dict:
    """Coggan Normalized Power from a list of [duration_s, watts] segments.

    Algorithm (Coggan):
        1. 30-second rolling average of power.
        2. Raise each to the 4th power.
        3. Mean of the 4th-power series.
        4. 4th root of that mean.

    Args:
        intervals: list of [duration_seconds, watts] pairs. Sequential
            constant-power blocks. Real-world workouts have variable
            power; we approximate by expanding each block to 1-second
            samples at its constant power. Good enough for the
            "tell me NP from this interval plan" use case.
        ftp_watts: optional FTP for sanity comparison. If provided and
            NP > 1.15 * FTP, the response gets a `sanity_warning`.

    Validation:
        Constant 200W for 3600s -> NP == 200 W exactly.
        Mix [200W * 1800, 300W * 1800] -> NP ~261 W (textbook).
    """
    if not isinstance(intervals, list) or not intervals:
        return {"status": "error", "message": "intervals must be a non-empty list"}

    samples: list[float] = []
    try:
        for item in intervals:
            duration_s, watts = item[0], item[1]
            dur = int(round(float(duration_s)))
            w = float(watts)
            if dur < 0 or w < 0:
                return {
                    "status": "error",
                    "message": "duration and watts must be non-negative",
                }
            samples.extend([w] * dur)
    except (TypeError, ValueError, IndexError):
        return {
            "status": "error",
            "message": "intervals must be a list of [duration_s, watts] pairs",
        }

    if not samples:
        return {"status": "error", "message": "intervals yielded zero samples"}

    # Step 1: 30-second rolling average.
    rolled = _rolling_average(samples, window=30)
    # Step 2 + 3: mean of 4th powers.
    mean_fourth = sum(v ** 4 for v in rolled) / len(rolled)
    # Step 4: fourth root.
    np_watts = mean_fourth ** 0.25

    mean_watts = sum(samples) / len(samples)
    variability_index = np_watts / mean_watts if mean_watts > 0 else 0.0

    sanity_warning: str | None = None
    if ftp_watts is not None:
        try:
            ftp = float(ftp_watts)
        except (TypeError, ValueError):
            ftp = 0
        if ftp > 0 and np_watts > 1.15 * ftp:
            sanity_warning = (
                f"NP {np_watts:.0f} W uebersteigt 1.15 x FTP "
                f"({ftp:.0f} W). Plausibel nur fuer kurze "
                f"supra-Schwellen-Intervalle - bitte Intervall-Daten "
                f"pruefen."
            )

    return {
        "status": "success",
        "np_watts": round(np_watts, 1),
        "mean_watts": round(mean_watts, 1),
        "variability_index": round(variability_index, 3),
        "duration_seconds": len(samples),
        "sanity_warning": sanity_warning,
        "interpretation": (
            f"Normalized Power {np_watts:.0f} W bei mittlerer Leistung "
            f"{mean_watts:.0f} W (VI {variability_index:.2f})."
        ),
    }


def hr_zones_karvonen(rhr: float, max_hr: float) -> dict:
    """Karvonen 5-zone heart-rate model using HR reserve.

    Formula:
        HR_target = (HR_max - HR_rest) * intensity + HR_rest

    Zones (HRR percent bands):
        Z1 Recovery:        50-60 percent HRR
        Z2 Aerobic Base:    60-70 percent HRR
        Z3 Tempo:           70-80 percent HRR
        Z4 Threshold:       80-90 percent HRR
        Z5 VO2max:          90-100 percent HRR

    Reference: Karvonen MJ et al., Ann Med Exp Biol Fenn 1957;
    35(3):307-315. ACSM Guidelines 11th ed.

    Validation: RHR=50, HRmax=190 -> HRR=140;
        Z2 (60-70%) = 50 + 0.6*140=134 to 50 + 0.7*140=148.
    """
    try:
        rhr_v = float(rhr)
        max_v = float(max_hr)
    except (TypeError, ValueError):
        return {"status": "error", "message": "rhr and max_hr must be numbers"}
    if rhr_v <= 0 or max_v <= 0 or max_v <= rhr_v:
        return {
            "status": "error",
            "message": "max_hr must be greater than rhr, both positive",
        }

    hrr = max_v - rhr_v
    bands = [
        ("z1", "Recovery", 0.50, 0.60),
        ("z2", "Aerobic Base", 0.60, 0.70),
        ("z3", "Tempo", 0.70, 0.80),
        ("z4", "Threshold", 0.80, 0.90),
        ("z5", "VO2max", 0.90, 1.00),
    ]
    zones: dict[str, Any] = {}
    for code, label, lo, hi in bands:
        low_bpm = round(rhr_v + lo * hrr)
        high_bpm = round(rhr_v + hi * hrr)
        zones[code] = {
            "label": label,
            "low_bpm": low_bpm,
            "high_bpm": high_bpm,
            "pct_hrr_low": round(lo * 100),
            "pct_hrr_high": round(hi * 100),
        }

    return {
        "status": "success",
        "rhr": round(rhr_v),
        "max_hr": round(max_v),
        "hrr": round(hrr),
        **zones,
        "interpretation": (
            f"Karvonen Zonen (HRR {round(hrr)}): Z2 "
            f"{zones['z2']['low_bpm']}-{zones['z2']['high_bpm']} bpm, "
            f"Z4 {zones['z4']['low_bpm']}-{zones['z4']['high_bpm']} bpm."
        ),
    }


def css_paces(css_pace_per_100m: float) -> dict:
    """Pool swim pace targets relative to Critical Swim Speed.

    Source: Swim Smooth CSS method
    (https://www.swimsmooth.com/improve/intermediate/critical-swim-speed).

    Easy:      CSS + 12 sec/100m
    Threshold: CSS
    Race:      CSS - 3 sec/100m (Olympic-distance race effort)

    Split times for common pool intervals (100m, 200m, 400m, 1000m,
    1500m) at race pace.

    Args:
        css_pace_per_100m: CSS pace in seconds per 100 metres.
    """
    try:
        css = float(css_pace_per_100m)
    except (TypeError, ValueError):
        return {"status": "error", "message": "css_pace_per_100m must be a number"}
    if css <= 0:
        return {"status": "error", "message": "css_pace_per_100m must be positive"}

    paces_s = {
        "easy_per_100m_seconds": css + 12.0,
        "threshold_per_100m_seconds": css,
        "race_per_100m_seconds": css - 3.0,
    }
    paces_fmt = {
        "easy_per_100m": _format_mmss(paces_s["easy_per_100m_seconds"]),
        "threshold_per_100m": _format_mmss(paces_s["threshold_per_100m_seconds"]),
        "race_per_100m": _format_mmss(paces_s["race_per_100m_seconds"]),
    }
    race_per_100 = paces_s["race_per_100m_seconds"]
    splits = {}
    for dist_m, label in [(100, "100m"), (200, "200m"), (400, "400m"),
                          (1000, "1000m"), (1500, "1500m")]:
        total_s = race_per_100 * (dist_m / 100.0)
        splits[label] = {
            "seconds": round(total_s, 1),
            "formatted": _format_mmss(total_s),
        }

    return {
        "status": "success",
        "css_pace_per_100m_seconds": round(css, 1),
        "css_pace_per_100m": _format_mmss(css),
        **paces_s,
        **paces_fmt,
        "splits": splits,
        "interpretation": (
            f"CSS {paces_fmt['threshold_per_100m']}/100m: easy "
            f"{paces_fmt['easy_per_100m']}, race "
            f"{paces_fmt['race_per_100m']}. 1500m bei Race Pace: "
            f"{splits['1500m']['formatted']}."
        ),
    }


def pace_target_from_goal(distance_km: float, target_time_seconds: float) -> dict:
    """Required pace per km for a goal time over a distance.

    Trivial division but explicitly tooled so the agent never gets it
    wrong under load (LLMs swap units, lose minutes, drop digits).
    """
    try:
        dist = float(distance_km)
        t = float(target_time_seconds)
    except (TypeError, ValueError):
        return {"status": "error", "message": "distance_km and target_time_seconds must be numbers"}
    if dist <= 0 or t <= 0:
        return {"status": "error", "message": "distance_km and target_time_seconds must be positive"}

    sec_per_km = t / dist
    speed_kmh = 3600.0 / sec_per_km

    return {
        "status": "success",
        "distance_km": round(dist, 3),
        "target_time_seconds": round(t, 1),
        "pace_per_km_seconds": round(sec_per_km, 1),
        "pace_per_km": _format_mmss(sec_per_km),
        "speed_kmh": round(speed_kmh, 2),
        "interpretation": (
            f"Fuer {dist:.2f} km in "
            f"{_format_mmss(t)} brauchst du "
            f"{_format_mmss(sec_per_km)}/km ({speed_kmh:.2f} km/h)."
        ),
    }


# ---------------------------------------------------------------------------
# Dispatch + registration
# ---------------------------------------------------------------------------


SUPPORTED_FORMULAS: dict[str, Any] = {
    "vdot_from_race_time": vdot_from_race_time,
    "paces_from_vdot": paces_from_vdot,
    "ftp_zones": ftp_zones,
    "np_from_intervals": np_from_intervals,
    "hr_zones_karvonen": hr_zones_karvonen,
    "css_paces": css_paces,
    "pace_target_from_goal": pace_target_from_goal,
}


def compute_sport_math(formula: str, inputs: dict | None = None) -> dict:
    """Dispatch entrypoint: pick a formula and run it with inputs.

    Args:
        formula: One of the seven supported names.
        inputs: Keyword-argument dict for the chosen formula.

    Returns:
        Formula result dict, or an error dict with the supported
        formula list when the name is unknown.
    """
    inputs = inputs or {}
    if not isinstance(formula, str):
        return {
            "status": "error",
            "message": "formula must be a string",
            "supported": list(SUPPORTED_FORMULAS.keys()),
        }
    fn = SUPPORTED_FORMULAS.get(formula)
    if fn is None:
        return {
            "status": "error",
            "message": f"unknown formula '{formula}'",
            "supported": list(SUPPORTED_FORMULAS.keys()),
        }
    if not isinstance(inputs, dict):
        return {"status": "error", "message": "inputs must be a dict"}
    try:
        return fn(**inputs)
    except TypeError as e:
        return {"status": "error", "message": f"invalid inputs for {formula}: {e}"}


def register_compute_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register `compute_sport_math` into the tool registry.

    The tool is read-only and pure - safe to expose in restricted
    sub-agent registries too. ``user_model`` parameter kept for
    consistency with other registrar signatures, even though the
    tool itself takes no user context.
    """
    _ = user_model  # unused; kept for signature parity

    registry.register(Tool(
        name="compute_sport_math",
        description=(
            "Compute sport-science formulas with verified, deterministic "
            "math. For ANY race-pace, training-zone, NP, FTP, VDOT, CSS, "
            "or HR-zone calculation, call this tool instead of computing "
            "inline. The tool's output is the source of truth; quote its "
            "numbers verbatim. Supported formulas: "
            "vdot_from_race_time(distance_km, time_seconds), "
            "paces_from_vdot(vdot), "
            "ftp_zones(ftp_watts), "
            "np_from_intervals(intervals=[[duration_s, watts], ...], "
            "ftp_watts=optional), "
            "hr_zones_karvonen(rhr, max_hr), "
            "css_paces(css_pace_per_100m), "
            "pace_target_from_goal(distance_km, target_time_seconds). "
            "Returns a dict with the computed values and a one-line "
            "interpretation. If the tool returns a sanity_warning, "
            "surface it to the athlete instead of suppressing it."
        ),
        handler=compute_sport_math,
        parameters={
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "enum": list(SUPPORTED_FORMULAS.keys()),
                    "description": "Name of the formula to compute.",
                },
                "inputs": {
                    "type": "object",
                    "description": (
                        "Formula-specific input dict. Keys depend on the "
                        "chosen formula - see the tool description."
                    ),
                },
            },
            "required": ["formula", "inputs"],
        },
        category="analysis",
        display_label="Berechne Sport-Werte",
    ))
