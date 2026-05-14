"""Tests for deterministic whole-athlete recovery alerts.

Covers:
- One positive + one negative case per pattern check.
- Edge cases: empty data, sparse data, baseline window too short,
  all-green path.
- format_alerts_block: empty list returns empty string, severity
  ordering critical > warn > info.
- system_prompt integration: build_runtime_context emits a
  `# Coach Alerts` section when patterns are active, omits it when
  alerts list is empty, swallows raises gracefully.
- get_recovery_alerts tool: registered, executes, returns the expected
  shape.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.services.recovery_alerts import (
    BB_CHRONIC_DAYS,
    BB_CHRONIC_THRESHOLD,
    HRV_DROP_PCT,
    LOAD_SPIKE_RATIO,
    RECOVERY_CRITICAL_THRESHOLD,
    RECOVERY_LOW_THRESHOLD,
    RHR_ELEVATION_BPM,
    RHR_ELEVATION_DAYS,
    SLEEP_CRITICAL_MINUTES,
    SLEEP_LOW_CONSECUTIVE_DAYS,
    SLEEP_LOW_MINUTES,
    STRESS_CHRONIC_DAYS,
    STRESS_CHRONIC_THRESHOLD,
    RecoveryAlert,
    detect_alerts_from_metrics,
    format_alerts_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _day(offset: int) -> str:
    """ISO date string, ``offset`` days ago from a fixed reference."""
    base = date(2026, 5, 14)  # arbitrary but fixed: today per env
    return (base - timedelta(days=offset)).isoformat()


def _metric_row(
    offset: int,
    *,
    sleep_minutes: int | None = 480,
    hrv: float | None = 60.0,
    resting_hr: int | None = 50,
    stress: int | None = 30,
    body_battery_high: int | None = 80,
    recovery_score: int | None = 70,
) -> dict:
    """Build a single merged daily-metrics row matching what
    `get_merged_daily_metrics` returns.
    """
    return {
        "date": _day(offset),
        "sleep_minutes": sleep_minutes,
        "hrv": hrv,
        "resting_hr": resting_hr,
        "stress": stress,
        "body_battery_high": body_battery_high,
        "recovery_score": recovery_score,
        "source": "garmin",
    }


def _all_green_window(days: int = 30) -> list[dict]:
    """Build `days` rows of all-healthy metrics, newest first."""
    return [_metric_row(i) for i in range(days)]


def _patterns(alerts: list[RecoveryAlert]) -> set[str]:
    return {a.pattern for a in alerts}


# ---------------------------------------------------------------------------
# All-green baseline
# ---------------------------------------------------------------------------


class TestAllGreen:
    def test_no_alerts_on_healthy_data(self) -> None:
        metrics = _all_green_window(30)
        alerts = detect_alerts_from_metrics(metrics)
        assert alerts == []

    def test_no_alerts_on_empty_data(self) -> None:
        alerts = detect_alerts_from_metrics([])
        assert alerts == []


# ---------------------------------------------------------------------------
# sleep_low_3d
# ---------------------------------------------------------------------------


class TestSleepLow3d:
    def test_three_consecutive_short_nights_triggers(self) -> None:
        metrics = _all_green_window(30)
        for i in range(3):
            metrics[i] = _metric_row(i, sleep_minutes=320)
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_low_3d" in _patterns(alerts)
        alert = next(a for a in alerts if a.pattern == "sleep_low_3d")
        assert alert.severity == "warn"
        # German umlauts must be real, not ASCII transliteration.
        assert "Nächte" in alert.message_de
        assert "ae" not in alert.message_de  # no Naechte
        # No em-dash or en-dash.
        assert "—" not in alert.message_de
        assert "–" not in alert.message_de
        assert alert.evidence["days"] == 3

    def test_two_short_nights_does_not_trigger(self) -> None:
        metrics = _all_green_window(30)
        metrics[0] = _metric_row(0, sleep_minutes=320)
        metrics[1] = _metric_row(1, sleep_minutes=300)
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_low_3d" not in _patterns(alerts)

    def test_three_nights_just_above_threshold_does_not_trigger(self) -> None:
        metrics = _all_green_window(30)
        for i in range(3):
            metrics[i] = _metric_row(i, sleep_minutes=SLEEP_LOW_MINUTES)
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_low_3d" not in _patterns(alerts)

    def test_missing_sleep_data_skips_check(self) -> None:
        metrics = _all_green_window(30)
        metrics[0] = _metric_row(0, sleep_minutes=None)
        metrics[1] = _metric_row(1, sleep_minutes=300)
        metrics[2] = _metric_row(2, sleep_minutes=300)
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_low_3d" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# sleep_critical
# ---------------------------------------------------------------------------


class TestSleepCritical:
    def test_single_night_under_4h_triggers_critical(self) -> None:
        metrics = _all_green_window(30)
        metrics[0] = _metric_row(0, sleep_minutes=200)  # ~3h20
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_critical" in _patterns(alerts)
        alert = next(a for a in alerts if a.pattern == "sleep_critical")
        assert alert.severity == "critical"

    def test_five_short_nights_in_7_triggers_critical(self) -> None:
        metrics = _all_green_window(30)
        for i in range(5):
            metrics[i] = _metric_row(i, sleep_minutes=330)
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_critical" in _patterns(alerts)

    def test_four_short_nights_in_7_does_not_trigger_critical(self) -> None:
        """4 short nights triggers sleep_low_3d only, not sleep_critical."""
        metrics = _all_green_window(30)
        for i in range(4):
            metrics[i] = _metric_row(i, sleep_minutes=330)
        alerts = detect_alerts_from_metrics(metrics)
        patterns = _patterns(alerts)
        assert "sleep_critical" not in patterns
        assert "sleep_low_3d" in patterns

    def test_single_short_night_above_critical_threshold_no_trigger(self) -> None:
        metrics = _all_green_window(30)
        metrics[0] = _metric_row(0, sleep_minutes=SLEEP_CRITICAL_MINUTES + 1)
        alerts = detect_alerts_from_metrics(metrics)
        assert "sleep_critical" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# hrv_drop
# ---------------------------------------------------------------------------


class TestHrvDrop:
    def test_15pct_drop_triggers(self) -> None:
        metrics: list[dict] = []
        # Recent 7 days: HRV around 50
        for i in range(7):
            metrics.append(_metric_row(i, hrv=50.0))
        # Older 23 days: HRV around 65 (baseline_avg ~ 61.5, drop ~ 19%)
        for i in range(7, 30):
            metrics.append(_metric_row(i, hrv=65.0))

        alerts = detect_alerts_from_metrics(metrics)
        assert "hrv_drop" in _patterns(alerts)
        alert = next(a for a in alerts if a.pattern == "hrv_drop")
        assert alert.severity == "warn"
        assert alert.evidence["drop_pct"] >= HRV_DROP_PCT

    def test_stable_hrv_no_trigger(self) -> None:
        metrics = [_metric_row(i, hrv=60.0) for i in range(30)]
        alerts = detect_alerts_from_metrics(metrics)
        assert "hrv_drop" not in _patterns(alerts)

    def test_short_history_skips_check(self) -> None:
        metrics = [_metric_row(i, hrv=50.0) for i in range(5)]
        alerts = detect_alerts_from_metrics(metrics)
        assert "hrv_drop" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# rhr_elevation
# ---------------------------------------------------------------------------


class TestRhrElevation:
    def test_3_consecutive_elevated_days_triggers(self) -> None:
        metrics: list[dict] = []
        # Most recent 3 days: RHR elevated to 58 (baseline 50 + 8)
        for i in range(RHR_ELEVATION_DAYS):
            metrics.append(_metric_row(i, resting_hr=58))
        # Older days: baseline ~ 50
        for i in range(RHR_ELEVATION_DAYS, 20):
            metrics.append(_metric_row(i, resting_hr=50))

        alerts = detect_alerts_from_metrics(metrics)
        assert "rhr_elevation" in _patterns(alerts)
        alert = next(a for a in alerts if a.pattern == "rhr_elevation")
        assert alert.severity == "warn"
        assert "über" in alert.message_de  # real umlaut

    def test_2_consecutive_elevated_days_no_trigger(self) -> None:
        metrics: list[dict] = []
        for i in range(2):
            metrics.append(_metric_row(i, resting_hr=60))
        for i in range(2, 20):
            metrics.append(_metric_row(i, resting_hr=50))
        alerts = detect_alerts_from_metrics(metrics)
        assert "rhr_elevation" not in _patterns(alerts)

    def test_baseline_too_short_skips(self) -> None:
        metrics = [_metric_row(i, resting_hr=60) for i in range(5)]
        alerts = detect_alerts_from_metrics(metrics)
        assert "rhr_elevation" not in _patterns(alerts)

    def test_streak_broken_by_normal_day(self) -> None:
        metrics: list[dict] = []
        metrics.append(_metric_row(0, resting_hr=60))
        metrics.append(_metric_row(1, resting_hr=50))  # streak broken
        metrics.append(_metric_row(2, resting_hr=60))
        for i in range(3, 20):
            metrics.append(_metric_row(i, resting_hr=50))
        alerts = detect_alerts_from_metrics(metrics)
        assert "rhr_elevation" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# body_battery_chronic
# ---------------------------------------------------------------------------


class TestBodyBatteryChronic:
    def test_3_consecutive_low_days_triggers(self) -> None:
        metrics = _all_green_window(10)
        for i in range(BB_CHRONIC_DAYS):
            metrics[i] = _metric_row(i, body_battery_high=25)
        alerts = detect_alerts_from_metrics(metrics)
        assert "body_battery_chronic" in _patterns(alerts)
        alert = next(
            a for a in alerts if a.pattern == "body_battery_chronic"
        )
        assert alert.severity == "warn"
        assert "lädst" in alert.message_de  # umlaut check

    def test_two_low_days_no_trigger(self) -> None:
        metrics = _all_green_window(10)
        for i in range(2):
            metrics[i] = _metric_row(i, body_battery_high=20)
        alerts = detect_alerts_from_metrics(metrics)
        assert "body_battery_chronic" not in _patterns(alerts)

    def test_at_threshold_no_trigger(self) -> None:
        metrics = _all_green_window(10)
        for i in range(BB_CHRONIC_DAYS):
            metrics[i] = _metric_row(
                i, body_battery_high=BB_CHRONIC_THRESHOLD,
            )
        alerts = detect_alerts_from_metrics(metrics)
        assert "body_battery_chronic" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# stress_chronic
# ---------------------------------------------------------------------------


class TestStressChronic:
    def test_5_elevated_days_triggers_info(self) -> None:
        metrics = _all_green_window(10)
        for i in range(STRESS_CHRONIC_DAYS):
            metrics[i] = _metric_row(i, stress=75)
        alerts = detect_alerts_from_metrics(metrics)
        assert "stress_chronic" in _patterns(alerts)
        alert = next(a for a in alerts if a.pattern == "stress_chronic")
        assert alert.severity == "info"
        assert "erhöht" in alert.message_de  # umlaut

    def test_4_elevated_days_no_trigger(self) -> None:
        metrics = _all_green_window(10)
        for i in range(4):
            metrics[i] = _metric_row(i, stress=75)
        alerts = detect_alerts_from_metrics(metrics)
        assert "stress_chronic" not in _patterns(alerts)

    def test_5_days_at_threshold_no_trigger(self) -> None:
        metrics = _all_green_window(10)
        for i in range(STRESS_CHRONIC_DAYS):
            metrics[i] = _metric_row(i, stress=STRESS_CHRONIC_THRESHOLD)
        alerts = detect_alerts_from_metrics(metrics)
        assert "stress_chronic" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# recovery_score_low / recovery_critical
# ---------------------------------------------------------------------------


class TestRecoveryScore:
    def test_two_low_days_triggers_warn(self) -> None:
        metrics = _all_green_window(10)
        metrics[0] = _metric_row(0, recovery_score=25)
        metrics[1] = _metric_row(1, recovery_score=22)
        alerts = detect_alerts_from_metrics(metrics)
        assert "recovery_score_low" in _patterns(alerts)
        alert = next(
            a for a in alerts if a.pattern == "recovery_score_low"
        )
        assert alert.severity == "warn"

    def test_critical_score_today_triggers_critical(self) -> None:
        metrics = _all_green_window(10)
        metrics[0] = _metric_row(0, recovery_score=15)
        alerts = detect_alerts_from_metrics(metrics)
        assert "recovery_critical" in _patterns(alerts)
        alert = next(a for a in alerts if a.pattern == "recovery_critical")
        assert alert.severity == "critical"

    def test_critical_score_today_suppresses_low_for_same_root_cause(
        self,
    ) -> None:
        """When today is critical (<20), we do not want both alerts firing."""
        metrics = _all_green_window(10)
        metrics[0] = _metric_row(0, recovery_score=10)
        metrics[1] = _metric_row(1, recovery_score=25)
        alerts = detect_alerts_from_metrics(metrics)
        patterns = _patterns(alerts)
        assert "recovery_critical" in patterns
        assert "recovery_score_low" not in patterns

    def test_score_at_threshold_no_trigger(self) -> None:
        metrics = _all_green_window(10)
        metrics[0] = _metric_row(0, recovery_score=RECOVERY_LOW_THRESHOLD)
        metrics[1] = _metric_row(1, recovery_score=RECOVERY_LOW_THRESHOLD)
        alerts = detect_alerts_from_metrics(metrics)
        assert "recovery_score_low" not in _patterns(alerts)

    def test_score_at_critical_threshold_no_trigger(self) -> None:
        metrics = _all_green_window(10)
        metrics[0] = _metric_row(
            0, recovery_score=RECOVERY_CRITICAL_THRESHOLD,
        )
        alerts = detect_alerts_from_metrics(metrics)
        assert "recovery_critical" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# training_load_spike
# ---------------------------------------------------------------------------


class TestTrainingLoadSpike:
    def test_spike_triggers_warn(self) -> None:
        metrics = _all_green_window(30)
        load_pair = (
            {"total_trimp": 200.0, "total_minutes": 300.0},
            {"total_trimp": 100.0, "total_minutes": 150.0},
        )
        alerts = detect_alerts_from_metrics(metrics, load_pair)
        assert "training_load_spike" in _patterns(alerts)
        alert = next(
            a for a in alerts if a.pattern == "training_load_spike"
        )
        assert alert.severity == "warn"
        assert "höher" in alert.message_de
        assert "erhöht" in alert.message_de

    def test_ratio_under_threshold_no_trigger(self) -> None:
        metrics = _all_green_window(30)
        load_pair = (
            {"total_trimp": 110.0, "total_minutes": 150.0},
            {"total_trimp": 100.0, "total_minutes": 150.0},
        )
        alerts = detect_alerts_from_metrics(metrics, load_pair)
        assert "training_load_spike" not in _patterns(alerts)

    def test_zero_prior_no_trigger(self) -> None:
        """Coming back from a rest week: do not flag, no meaningful ratio."""
        metrics = _all_green_window(30)
        load_pair = (
            {"total_trimp": 200.0, "total_minutes": 300.0},
            {"total_trimp": 0.0, "total_minutes": 0.0},
        )
        alerts = detect_alerts_from_metrics(metrics, load_pair)
        assert "training_load_spike" not in _patterns(alerts)

    def test_no_load_data_no_trigger(self) -> None:
        metrics = _all_green_window(30)
        alerts = detect_alerts_from_metrics(metrics, (None, None))
        assert "training_load_spike" not in _patterns(alerts)


# ---------------------------------------------------------------------------
# format_alerts_block
# ---------------------------------------------------------------------------


class TestFormatBlock:
    def test_empty_alerts_returns_empty_string(self) -> None:
        assert format_alerts_block([]) == ""

    def test_block_contains_header_and_alert_line(self) -> None:
        alert = RecoveryAlert(
            severity="warn",
            pattern="sleep_low_3d",
            message_de="Drei Nächte unter 6h.",
            evidence={"days": 3},
        )
        block = format_alerts_block([alert])
        assert "# Coach Alerts" in block
        assert "[WARN] sleep_low_3d" in block
        assert "Drei Nächte unter 6h." in block

    def test_severity_ordering_critical_first(self) -> None:
        info = RecoveryAlert("info", "p_info", "info msg", {})
        warn = RecoveryAlert("warn", "p_warn", "warn msg", {})
        critical = RecoveryAlert("critical", "p_crit", "critical msg", {})

        block = format_alerts_block([info, warn, critical])

        # critical must appear before warn, warn before info
        crit_idx = block.find("p_crit")
        warn_idx = block.find("p_warn")
        info_idx = block.find("p_info")
        assert crit_idx < warn_idx < info_idx

    def test_block_has_no_dashes_or_emdashes(self) -> None:
        alert = RecoveryAlert(
            severity="warn",
            pattern="x",
            message_de="kein Strichproblem.",
            evidence={},
        )
        block = format_alerts_block([alert])
        assert "—" not in block
        assert "–" not in block


# ---------------------------------------------------------------------------
# RecoveryAlert dataclass
# ---------------------------------------------------------------------------


class TestRecoveryAlert:
    def test_to_dict_serialises(self) -> None:
        alert = RecoveryAlert(
            severity="warn",
            pattern="sleep_low_3d",
            message_de="msg",
            evidence={"days": 3, "sleep_avg_h": 5.4},
        )
        out = alert.to_dict()
        assert out == {
            "severity": "warn",
            "pattern": "sleep_low_3d",
            "message_de": "msg",
            "evidence": {"days": 3, "sleep_avg_h": 5.4},
        }


# ---------------------------------------------------------------------------
# Public detect_alerts wraps the DB fetch
# ---------------------------------------------------------------------------


class TestDetectAlertsEntrypoint:
    def test_db_error_returns_empty(self) -> None:
        """A DB exception must NOT propagate; we return empty alerts."""
        from src.services import recovery_alerts as mod

        with (
            patch.object(
                mod,
                "_load_metrics",
                side_effect=Exception("db down"),
            ),
        ):
            # _load_metrics is wrapped in try inside detect_alerts; but
            # the helper itself catches and returns []. Test the helper
            # path directly.
            pass

        # Verify the helper catches
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            side_effect=Exception("db down"),
        ):
            result = mod._load_metrics("user-x")
            assert result == []

    def test_detect_alerts_with_no_data_returns_empty(self) -> None:
        from src.services import recovery_alerts as mod

        with (
            patch.object(mod, "_load_metrics", return_value=[]),
            patch.object(
                mod, "_load_load_summary_pair", return_value=(None, None),
            ),
        ):
            assert mod.detect_alerts("user-x") == []

    def test_detect_alerts_propagates_alerts(self) -> None:
        from src.services import recovery_alerts as mod

        metrics = _all_green_window(30)
        for i in range(3):
            metrics[i] = _metric_row(i, sleep_minutes=300)

        with (
            patch.object(mod, "_load_metrics", return_value=metrics),
            patch.object(
                mod, "_load_load_summary_pair", return_value=(None, None),
            ),
        ):
            alerts = mod.detect_alerts("user-x")
            assert "sleep_low_3d" in {a.pattern for a in alerts}


# ---------------------------------------------------------------------------
# system_prompt integration
# ---------------------------------------------------------------------------


class TestSystemPromptIntegration:
    """Verify build_runtime_context emits the `# Coach Alerts` section.

    The runtime context builder calls `detect_alerts(user_id)` if
    Supabase is enabled. We patch it directly to control which alerts
    fire, then assert the rendered section appears in the output.
    """

    def _user_model(self) -> MagicMock:
        m = MagicMock()
        m.user_id = "test-user-alerts"
        m.project_profile.return_value = {
            "name": "Test", "sports": ["running"],
        }
        m.get_active_plan_summary.return_value = None
        return m

    def _settings(self) -> MagicMock:
        s = MagicMock()
        s.agenticsports_user_id = "test-user-alerts"
        s.use_supabase = True
        return s

    def test_alerts_appended_when_active(self) -> None:
        from src.agent import system_prompt as sp

        warn_alert = RecoveryAlert(
            severity="warn",
            pattern="sleep_low_3d",
            message_de="Drei Nächte unter 6h Schlaf.",
            evidence={"days": 3},
        )

        # Patch the heavy dependencies so we focus on the alerts wiring.
        with (
            patch(
                "src.config.get_settings", return_value=self._settings(),
            ),
            patch(
                "src.services.recovery_alerts.detect_alerts",
                return_value=[warn_alert],
            ),
            patch(
                "src.services.health_context.build_health_summary",
                return_value=None,
            ),
            patch("src.agent.athlete_md.build_athlete_md", return_value=""),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value={
                    "total_sessions": 0,
                    "total_minutes": 0,
                    "total_trimp": 0,
                    "sports_seen": [],
                    "sessions_by_sport": {},
                    "sessions_by_source": {},
                },
            ),
        ):
            ctx = sp.build_runtime_context(self._user_model())

        assert "# Coach Alerts" in ctx
        assert "[WARN] sleep_low_3d" in ctx
        assert "Drei Nächte unter 6h Schlaf." in ctx

    def test_no_section_when_alerts_empty(self) -> None:
        from src.agent import system_prompt as sp

        with (
            patch(
                "src.config.get_settings", return_value=self._settings(),
            ),
            patch(
                "src.services.recovery_alerts.detect_alerts",
                return_value=[],
            ),
            patch(
                "src.services.health_context.build_health_summary",
                return_value=None,
            ),
            patch("src.agent.athlete_md.build_athlete_md", return_value=""),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value={
                    "total_sessions": 0,
                    "total_minutes": 0,
                    "total_trimp": 0,
                    "sports_seen": [],
                    "sessions_by_sport": {},
                    "sessions_by_source": {},
                },
            ),
        ):
            ctx = sp.build_runtime_context(self._user_model())

        assert "# Coach Alerts" not in ctx

    def test_detect_alerts_failure_does_not_break_context(self) -> None:
        from src.agent import system_prompt as sp

        with (
            patch(
                "src.config.get_settings", return_value=self._settings(),
            ),
            patch(
                "src.services.recovery_alerts.detect_alerts",
                side_effect=Exception("boom"),
            ),
            patch(
                "src.services.health_context.build_health_summary",
                return_value=None,
            ),
            patch("src.agent.athlete_md.build_athlete_md", return_value=""),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value={
                    "total_sessions": 0,
                    "total_minutes": 0,
                    "total_trimp": 0,
                    "sports_seen": [],
                    "sessions_by_sport": {},
                    "sessions_by_source": {},
                },
            ),
        ):
            ctx = sp.build_runtime_context(self._user_model())

        # Context still builds; no alerts section.
        assert "# Coach Alerts" not in ctx
        # The rest of the context is still there.
        assert "# Current Date" in ctx


# ---------------------------------------------------------------------------
# STATIC_SYSTEM_PROMPT must teach the WHOLE-ATHLETE COACHING rule
# ---------------------------------------------------------------------------


class TestStaticPromptRule:
    def test_strict_block_present(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "WHOLE-ATHLETE COACHING" in STATIC_SYSTEM_PROMPT
        assert "# Coach Alerts" in STATIC_SYSTEM_PROMPT
        assert "get_recovery_alerts" in STATIC_SYSTEM_PROMPT

    def test_no_emdash_or_endash_in_block(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "—" not in STATIC_SYSTEM_PROMPT
        assert "–" not in STATIC_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# get_recovery_alerts tool registration + execution
# ---------------------------------------------------------------------------


class TestGetRecoveryAlertsTool:
    def test_tool_is_registered(self) -> None:
        from src.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        mock_settings = MagicMock()
        mock_settings.agenticsports_user_id = "test-user"
        mock_settings.use_supabase = True

        with patch(
            "src.agent.tools.health_tools.get_settings",
            return_value=mock_settings,
        ):
            from src.agent.tools.health_tools import register_health_tools

            register_health_tools(registry)

        names = [t["function"]["name"] for t in registry.get_openai_tools()]
        assert "get_recovery_alerts" in names

    def test_tool_is_in_core_tool_names(self) -> None:
        from src.agent.tools.registry import CORE_TOOL_NAMES

        assert "get_recovery_alerts" in CORE_TOOL_NAMES

    def test_tool_executes_and_returns_expected_shape(self) -> None:
        from src.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        mock_settings = MagicMock()
        mock_settings.agenticsports_user_id = "test-user"
        mock_settings.use_supabase = True

        warn_alert = RecoveryAlert(
            severity="warn",
            pattern="sleep_low_3d",
            message_de="msg",
            evidence={"days": 3},
        )

        with (
            patch(
                "src.agent.tools.health_tools.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "src.services.recovery_alerts.detect_alerts",
                return_value=[warn_alert],
            ),
        ):
            from src.agent.tools.health_tools import register_health_tools

            register_health_tools(registry)
            result = registry.execute("get_recovery_alerts", {})

        assert result["count"] == 1
        assert result["alerts"][0]["pattern"] == "sleep_low_3d"
        assert result["alerts"][0]["severity"] == "warn"
        assert result["alerts"][0]["message_de"] == "msg"
        assert result["alerts"][0]["evidence"] == {"days": 3}

    def test_tool_returns_empty_when_user_id_missing(self) -> None:
        from src.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        mock_settings = MagicMock()
        mock_settings.agenticsports_user_id = None
        mock_settings.use_supabase = True

        with patch(
            "src.agent.tools.health_tools.get_settings",
            return_value=mock_settings,
        ):
            from src.agent.tools.health_tools import register_health_tools

            register_health_tools(registry)
            result = registry.execute("get_recovery_alerts", {})

        assert result == {"count": 0, "alerts": []}


# ---------------------------------------------------------------------------
# No em-dashes / en-dashes anywhere in the rendered messages
# ---------------------------------------------------------------------------


class TestNoDashesAnywhere:
    """Regression guard: every alert message must use ASCII hyphens only."""

    @pytest.mark.parametrize(
        "alerts_factory",
        [
            lambda: _all_green_window(30),
        ],
    )
    def test_no_dashes_in_green_state(self, alerts_factory) -> None:
        metrics = alerts_factory()
        alerts = detect_alerts_from_metrics(metrics)
        for alert in alerts:
            assert "—" not in alert.message_de
            assert "–" not in alert.message_de

    def test_no_dashes_in_all_patterns_when_all_active(self) -> None:
        """Force every pattern to fire, then check every message_de."""
        metrics: list[dict] = []
        # Build 30 days where everything is bad.
        for i in range(30):
            metrics.append(
                _metric_row(
                    i,
                    sleep_minutes=180 if i == 0 else 320,
                    hrv=40.0 if i < 7 else 65.0,
                    resting_hr=70 if i < 4 else 50,
                    stress=80 if i < 6 else 30,
                    body_battery_high=20 if i < 4 else 80,
                    recovery_score=15 if i == 0 else (25 if i == 1 else 70),
                )
            )

        load_pair = (
            {"total_trimp": 200.0, "total_minutes": 300.0},
            {"total_trimp": 100.0, "total_minutes": 150.0},
        )
        alerts = detect_alerts_from_metrics(metrics, load_pair)

        # We expect at least 5 different patterns to fire.
        assert len(alerts) >= 5

        for alert in alerts:
            assert "—" not in alert.message_de, alert
            assert "–" not in alert.message_de, alert

        # Spot-check: real umlauts present where expected
        joined = "\n".join(a.message_de for a in alerts)
        assert "ä" in joined or "ö" in joined or "ü" in joined
