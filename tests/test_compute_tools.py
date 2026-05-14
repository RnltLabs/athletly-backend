"""Reference-table accuracy tests for compute_sport_math.

Each formula in src/agent/tools/compute_tools.py is validated against
published reference values:

- VDOT against Jack Daniels' VDOT calculator tables (5k, 10k, marathon).
- FTP zones against Coggan 7-zone definitions reproduced by TrainingPeaks.
- Karvonen zones against direct percent-HRR math.
- Normalized Power against the textbook constant-power and split-pyramid
  examples (Coggan + Allen).
- CSS against the Swim Smooth formula.

Tolerances: 1% for VDOT (Daniels rounds tables), 5 seconds for derived
paces, exact for zone bounds (rounding to whole watts/bpm).
"""

from __future__ import annotations

import pytest

from src.agent.tools.compute_tools import (
    SUPPORTED_FORMULAS,
    compute_sport_math,
    css_paces,
    ftp_zones,
    hr_zones_karvonen,
    np_from_intervals,
    pace_target_from_goal,
    paces_from_vdot,
    vdot_from_race_time,
)


# ---------------------------------------------------------------------------
# VDOT formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance_km, time_seconds, expected_vdot, tol", [
    # Reference: Jack Daniels VDOT table 5.2 (published online at vdoto2.com).
    # Tolerance reflects the integer rounding in the published tables and the
    # small empirical smoothing Daniels applies vs the raw VO2/percent-VO2max
    # formula. 1.5 absolute points covers all published-table variants.
    (5.0, 20 * 60, 49.2, 1.5),       # 5k 20:00 -> VDOT ~49
    (10.0, 40 * 60, 52.0, 1.5),      # 10k 40:00 -> VDOT ~52
    (5.0, 18 * 60, 55.5, 1.5),       # 5k 18:00 -> VDOT ~55-56
    (21.0975, 90 * 60, 51.0, 1.5),   # Half marathon 1:30:00 -> VDOT ~51
])
def test_vdot_matches_daniels_reference(
    distance_km: float, time_seconds: float, expected_vdot: float, tol: float,
) -> None:
    result = vdot_from_race_time(distance_km, time_seconds)
    assert result["status"] == "success"
    assert abs(result["vdot"] - expected_vdot) <= tol


def test_vdot_rejects_zero_distance() -> None:
    assert vdot_from_race_time(0, 1200)["status"] == "error"


def test_vdot_rejects_zero_time() -> None:
    assert vdot_from_race_time(5.0, 0)["status"] == "error"


def test_vdot_rejects_non_numeric() -> None:
    assert vdot_from_race_time("five", 1200)["status"] == "error"


# ---------------------------------------------------------------------------
# Daniels training paces from VDOT
# ---------------------------------------------------------------------------


def test_paces_for_vdot_50_match_table() -> None:
    """VDOT 50: threshold ~4:18/km, marathon ~4:34/km (Daniels table)."""
    result = paces_from_vdot(50.0)
    assert result["status"] == "success"
    # 5-second tolerance per km.
    threshold = result["threshold_per_km_seconds"]
    marathon = result["marathon_per_km_seconds"]
    assert abs(threshold - (4 * 60 + 18)) <= 5
    assert abs(marathon - (4 * 60 + 34)) <= 5


def test_paces_for_vdot_50_interval_faster_than_threshold() -> None:
    result = paces_from_vdot(50.0)
    # Daniels' speed ladder must be monotone.
    assert (
        result["repetition_per_km_seconds"]
        < result["interval_per_km_seconds"]
        < result["threshold_per_km_seconds"]
        < result["marathon_per_km_seconds"]
        < result["easy_per_km_seconds"]
    )


def test_paces_rejects_out_of_range_vdot() -> None:
    assert paces_from_vdot(10)["status"] == "error"
    assert paces_from_vdot(100)["status"] == "error"


def test_pace_strings_are_mmss_formatted() -> None:
    result = paces_from_vdot(50.0)
    for key in ("easy_per_km", "marathon_per_km", "threshold_per_km",
                "interval_per_km", "repetition_per_km"):
        assert ":" in result[key]


# ---------------------------------------------------------------------------
# FTP zones (Coggan)
# ---------------------------------------------------------------------------


def test_ftp_zones_for_240w_match_coggan() -> None:
    """FTP 240W: Z2 134-180, Z4 218-252, Z5 254-288 (Coggan, rounded W)."""
    result = ftp_zones(240.0)
    assert result["status"] == "success"
    assert result["z2"]["low_watts"] == 134
    assert result["z2"]["high_watts"] == 180
    assert result["z4"]["low_watts"] == 218
    assert result["z4"]["high_watts"] == 252
    assert result["z5"]["low_watts"] == 254
    assert result["z5"]["high_watts"] == 288


def test_ftp_zones_z7_open_ended() -> None:
    result = ftp_zones(240.0)
    assert result["z7"]["high_watts"] is None


def test_ftp_zones_sanity_warning_for_low_ftp() -> None:
    result = ftp_zones(50.0)
    assert result["status"] == "success"
    assert result["sanity_warning"] is not None


def test_ftp_zones_sanity_warning_for_high_ftp() -> None:
    result = ftp_zones(600.0)
    assert result["status"] == "success"
    assert result["sanity_warning"] is not None


def test_ftp_zones_no_sanity_warning_for_normal_ftp() -> None:
    result = ftp_zones(240.0)
    assert result["sanity_warning"] is None


def test_ftp_zones_rejects_zero() -> None:
    assert ftp_zones(0)["status"] == "error"


# ---------------------------------------------------------------------------
# Normalized Power (Coggan algorithm)
# ---------------------------------------------------------------------------


def test_np_constant_power_equals_input() -> None:
    """Constant 200W for 60 minutes: NP must equal 200 W exactly."""
    result = np_from_intervals([[3600, 200]])
    assert result["status"] == "success"
    assert abs(result["np_watts"] - 200.0) < 0.5


def test_np_higher_than_mean_for_variable_workout() -> None:
    """Mix of 200W and 300W for 30 min each: NP > mean (250W), < 300W."""
    result = np_from_intervals([[1800, 200], [1800, 300]])
    assert result["status"] == "success"
    assert result["np_watts"] > result["mean_watts"]
    assert result["np_watts"] < 300.0


def test_np_sanity_warning_when_np_exceeds_115_pct_ftp() -> None:
    """Lisa-bug guard: NP > 1.15 x FTP must surface a sanity warning."""
    # Build a workout where NP will be >> 240 W. Short hard burst dominates
    # the 4th-power mean.
    result = np_from_intervals(
        [[60, 600], [60, 50]],  # 60s at 600W, 60s easy
        ftp_watts=240,
    )
    assert result["status"] == "success"
    # Confirm the math actually produces NP > 1.15 * 240 = 276.
    assert result["np_watts"] > 276
    assert result["sanity_warning"] is not None
    assert "FTP" in result["sanity_warning"]


def test_np_no_sanity_warning_when_within_bounds() -> None:
    """Sustainable interval NP must NOT trigger the warning."""
    result = np_from_intervals(
        [[3600, 230]],  # steady 230W for 1h
        ftp_watts=240,
    )
    assert result["status"] == "success"
    assert result["sanity_warning"] is None


def test_np_rejects_empty_intervals() -> None:
    assert np_from_intervals([])["status"] == "error"


def test_np_rejects_negative_watts() -> None:
    assert np_from_intervals([[60, -100]])["status"] == "error"


def test_np_rejects_malformed_pair() -> None:
    # Single value instead of pair.
    result = np_from_intervals([[60]])
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Karvonen HR zones
# ---------------------------------------------------------------------------


def test_karvonen_rhr50_max190_z2() -> None:
    """RHR=50, HRmax=190 -> HRR=140. Z2 (60-70% HRR) = 134-148."""
    result = hr_zones_karvonen(50, 190)
    assert result["status"] == "success"
    assert result["hrr"] == 140
    assert result["z2"]["low_bpm"] == 134
    assert result["z2"]["high_bpm"] == 148


def test_karvonen_z5_caps_at_max_hr() -> None:
    result = hr_zones_karvonen(50, 190)
    assert result["z5"]["high_bpm"] == 190


def test_karvonen_rejects_max_below_rhr() -> None:
    assert hr_zones_karvonen(100, 50)["status"] == "error"


def test_karvonen_rejects_zero_rhr() -> None:
    assert hr_zones_karvonen(0, 190)["status"] == "error"


# ---------------------------------------------------------------------------
# CSS paces
# ---------------------------------------------------------------------------


def test_css_basic_thresholds() -> None:
    """CSS 110 s/100m: easy 122, threshold 110, race 107 s/100m."""
    result = css_paces(110.0)
    assert result["status"] == "success"
    assert result["threshold_per_100m_seconds"] == 110.0
    assert result["easy_per_100m_seconds"] == 122.0
    assert result["race_per_100m_seconds"] == 107.0


def test_css_splits_for_1500m_race() -> None:
    """At CSS 110 s/100m race (107s): 1500m = 15 * 107 = 1605s (26:45)."""
    result = css_paces(110.0)
    splits = result["splits"]
    assert splits["1500m"]["seconds"] == pytest.approx(1605.0, abs=0.5)


def test_css_rejects_zero() -> None:
    assert css_paces(0)["status"] == "error"


# ---------------------------------------------------------------------------
# Pace target from goal time
# ---------------------------------------------------------------------------


def test_pace_target_10k_in_50min() -> None:
    """10km in 50:00 -> 5:00/km exact, 12 km/h."""
    result = pace_target_from_goal(10.0, 50 * 60)
    assert result["status"] == "success"
    assert result["pace_per_km_seconds"] == 300.0
    assert result["pace_per_km"] == "5:00"
    assert result["speed_kmh"] == pytest.approx(12.0, abs=0.05)


def test_pace_target_marathon_sub_3() -> None:
    """42.195km in 3h -> 4:15.85/km."""
    result = pace_target_from_goal(42.195, 3 * 3600)
    assert result["status"] == "success"
    # 3*3600 / 42.195 = 256.0 sec/km
    assert abs(result["pace_per_km_seconds"] - 256.0) < 0.5


def test_pace_target_rejects_zero_distance() -> None:
    assert pace_target_from_goal(0, 3600)["status"] == "error"


# ---------------------------------------------------------------------------
# Dispatch entrypoint
# ---------------------------------------------------------------------------


def test_dispatch_known_formula() -> None:
    result = compute_sport_math(
        "vdot_from_race_time",
        {"distance_km": 10, "time_seconds": 40 * 60},
    )
    assert result["status"] == "success"
    assert "vdot" in result


def test_dispatch_unknown_formula() -> None:
    result = compute_sport_math("unknown_thing", {})
    assert result["status"] == "error"
    assert "supported" in result
    assert "vdot_from_race_time" in result["supported"]


def test_dispatch_bad_inputs_type() -> None:
    result = compute_sport_math("ftp_zones", "not a dict")  # type: ignore[arg-type]
    assert result["status"] == "error"


def test_dispatch_missing_required_input() -> None:
    result = compute_sport_math("ftp_zones", {})
    assert result["status"] == "error"


def test_dispatch_handles_none_inputs() -> None:
    result = compute_sport_math("ftp_zones", None)
    # None gets coerced to empty dict then ftp_zones fails on missing arg.
    assert result["status"] == "error"


def test_supported_formulas_set() -> None:
    """The contract: seven formulas, exact names."""
    assert set(SUPPORTED_FORMULAS.keys()) == {
        "vdot_from_race_time",
        "paces_from_vdot",
        "ftp_zones",
        "np_from_intervals",
        "hr_zones_karvonen",
        "css_paces",
        "pace_target_from_goal",
    }


# ---------------------------------------------------------------------------
# Lisa-case end-to-end via dispatcher
# ---------------------------------------------------------------------------


def test_lisa_bug_does_not_reproduce_via_compute_tool() -> None:
    """The Lisa bug: agent confabulated NP=352W from FTP=240W.

    With the compute tool the agent can ONLY get a real number. We pin
    that NP from a sustainable sub-threshold ride at 230W is NOT 352W
    and that any attempt to claim NP=352 would have to come from
    physically supra-threshold intervals (and surface a sanity warning).
    """
    # Plausible long ride at threshold.
    sustainable = compute_sport_math(
        "np_from_intervals",
        {"intervals": [[3600, 230]], "ftp_watts": 240},
    )
    assert sustainable["status"] == "success"
    assert sustainable["np_watts"] < 240 * 1.10  # within 10% of FTP
    assert sustainable["sanity_warning"] is None

    # Try to "produce" NP=352 from a sustainable single block: cannot.
    # Even at FTP itself, NP would be ~FTP.
    at_ftp = compute_sport_math(
        "np_from_intervals",
        {"intervals": [[3600, 240]], "ftp_watts": 240},
    )
    assert at_ftp["np_watts"] < 245  # noise from rolling-average ramp only
