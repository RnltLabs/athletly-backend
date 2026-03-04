"""Tests for self-improvement tools, proactive message formatting, and heartbeat integration.

Covers:
- evaluate_formula_accuracy: provider data, no definition, no data, invalid formula, no provider
- review_all_formulas: valid, mixed, empty
- format_proactive_message: self_improvement_check trigger
- HeartbeatService: tick_count increment, self-improvement every 12th tick
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-self-improvement-test"

# Patch targets — these are imported locally inside closures, so we patch
# the source modules where the names are defined.
_PATCH_SETTINGS = "src.config.get_settings"
_PATCH_METRIC_DEF = "src.db.agent_config_db.get_metric_definition"
_PATCH_METRIC_DEFS = "src.db.agent_config_db.get_metric_definitions"
_PATCH_HEALTH_ACTS = "src.db.health_data_db.list_health_activities"
_PATCH_GARMIN_ACTS = "src.db.health_data_db.list_garmin_activities"


def _make_settings(user_id: str = USER_ID) -> MagicMock:
    """Create a mock settings object."""
    settings = MagicMock()
    settings.agenticsports_user_id = user_id
    return settings


def _make_registry():
    """Create a ToolRegistry with self-improvement tools registered."""
    from src.agent.tools.registry import ToolRegistry
    from src.agent.tools.self_improvement_tools import register_self_improvement_tools

    registry = ToolRegistry()
    register_self_improvement_tools(registry, MagicMock())
    return registry


# ---------------------------------------------------------------------------
# evaluate_formula_accuracy
# ---------------------------------------------------------------------------


class TestEvaluateFormulaAccuracyWithProviderData(unittest.TestCase):
    """Formula evaluated against activities that have provider values."""

    @patch(_PATCH_GARMIN_ACTS)
    @patch(_PATCH_HEALTH_ACTS)
    @patch(_PATCH_METRIC_DEF)
    @patch(_PATCH_SETTINGS)
    def test_returns_accuracy_stats(
        self, mock_settings, mock_metric_def, mock_health, mock_garmin,
    ):
        mock_settings.return_value = _make_settings()
        mock_metric_def.return_value = {
            "name": "trimp_estimate",
            "formula": "duration_minutes * avg_heart_rate / 100",
        }
        mock_health.return_value = [
            {
                "start_time": "2026-02-01T10:00:00Z",
                "duration_seconds": 3600,
                "distance_meters": 10000,
                "avg_heart_rate": 140,
                "max_heart_rate": 170,
                "calories": 500,
                "training_load_trimp": 80,
            },
            {
                "start_time": "2026-02-03T10:00:00Z",
                "duration_seconds": 1800,
                "distance_meters": 5000,
                "avg_heart_rate": 130,
                "max_heart_rate": 155,
                "calories": 250,
                "training_load_trimp": 40,
            },
        ]
        mock_garmin.return_value = []

        registry = _make_registry()
        result = registry.execute("evaluate_formula_accuracy", {"metric_name": "trimp_estimate"})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metric_name"], "trimp_estimate")
        self.assertEqual(result["total_evaluated"], 2)
        self.assertEqual(result["with_provider_comparison"], 2)
        self.assertIn("avg_absolute_error", result)
        self.assertIn("max_absolute_error", result)
        self.assertIn("recommendation", result)
        self.assertIn("sample_comparisons", result)


class TestEvaluateFormulaAccuracyNoDefinition(unittest.TestCase):
    """No metric definition found for the given name."""

    @patch(_PATCH_METRIC_DEF)
    @patch(_PATCH_SETTINGS)
    def test_returns_no_definition(self, mock_settings, mock_metric_def):
        mock_settings.return_value = _make_settings()
        mock_metric_def.return_value = None

        registry = _make_registry()
        result = registry.execute("evaluate_formula_accuracy", {"metric_name": "nonexistent"})

        self.assertEqual(result["status"], "no_definition")
        self.assertIn("nonexistent", result["message"])


class TestEvaluateFormulaAccuracyNoData(unittest.TestCase):
    """Metric definition exists but no activities to evaluate."""

    @patch(_PATCH_GARMIN_ACTS)
    @patch(_PATCH_HEALTH_ACTS)
    @patch(_PATCH_METRIC_DEF)
    @patch(_PATCH_SETTINGS)
    def test_returns_no_data(
        self, mock_settings, mock_metric_def, mock_health, mock_garmin,
    ):
        mock_settings.return_value = _make_settings()
        mock_metric_def.return_value = {"name": "test", "formula": "avg_heart_rate * 2"}
        mock_health.return_value = []
        mock_garmin.return_value = []

        registry = _make_registry()
        result = registry.execute("evaluate_formula_accuracy", {"metric_name": "test"})

        self.assertEqual(result["status"], "no_data")


class TestEvaluateFormulaAccuracyInvalidFormula(unittest.TestCase):
    """Metric definition has an invalid formula."""

    @patch(_PATCH_METRIC_DEF)
    @patch(_PATCH_SETTINGS)
    def test_returns_invalid_formula(self, mock_settings, mock_metric_def):
        mock_settings.return_value = _make_settings()
        mock_metric_def.return_value = {
            "name": "bad_formula",
            "formula": "__import__('os').system('rm -rf /')",
        }

        registry = _make_registry()
        result = registry.execute("evaluate_formula_accuracy", {"metric_name": "bad_formula"})

        self.assertEqual(result["status"], "invalid_formula")
        self.assertIn("formula", result)


class TestEvaluateFormulaAccuracyNoProvider(unittest.TestCase):
    """Activities exist but have no provider values for comparison."""

    @patch(_PATCH_GARMIN_ACTS)
    @patch(_PATCH_HEALTH_ACTS)
    @patch(_PATCH_METRIC_DEF)
    @patch(_PATCH_SETTINGS)
    def test_returns_no_provider_data(
        self, mock_settings, mock_metric_def, mock_health, mock_garmin,
    ):
        mock_settings.return_value = _make_settings()
        mock_metric_def.return_value = {"name": "hr_metric", "formula": "avg_heart_rate * 2"}
        mock_health.return_value = [
            {
                "start_time": "2026-02-01T10:00:00Z",
                "duration_seconds": 3600,
                "distance_meters": 10000,
                "avg_heart_rate": 140,
                "max_heart_rate": 170,
                "calories": 500,
                "training_load_trimp": None,  # No provider value
            },
        ]
        mock_garmin.return_value = []

        registry = _make_registry()
        result = registry.execute("evaluate_formula_accuracy", {"metric_name": "hr_metric"})

        self.assertEqual(result["status"], "no_provider_data")
        self.assertEqual(result["total_evaluated"], 1)
        self.assertIn("sample_computations", result)


# ---------------------------------------------------------------------------
# review_all_formulas
# ---------------------------------------------------------------------------


class TestReviewAllFormulasValid(unittest.TestCase):
    """All metric definitions have valid formulas."""

    @patch(_PATCH_METRIC_DEFS)
    @patch(_PATCH_SETTINGS)
    def test_all_valid(self, mock_settings, mock_definitions):
        mock_settings.return_value = _make_settings()
        mock_definitions.return_value = [
            {"name": "trimp", "formula": "duration_minutes * avg_heart_rate / 100", "description": "TRIMP", "unit": "au"},
            {"name": "pace", "formula": "duration_minutes / (distance_meters / 1000)", "description": "Pace", "unit": "min/km"},
        ]

        registry = _make_registry()
        result = registry.execute("review_all_formulas", {})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_formulas"], 2)
        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["invalid"], 0)
        self.assertEqual(result["recommendation"], "All formulas are valid")


class TestReviewAllFormulasMixed(unittest.TestCase):
    """Some valid, some invalid formulas."""

    @patch(_PATCH_METRIC_DEFS)
    @patch(_PATCH_SETTINGS)
    def test_mixed_validity(self, mock_settings, mock_definitions):
        mock_settings.return_value = _make_settings()
        mock_definitions.return_value = [
            {"name": "good", "formula": "avg_heart_rate * 2", "description": "", "unit": ""},
            {"name": "bad", "formula": "__import__('os')", "description": "", "unit": ""},
            {"name": "empty", "formula": "", "description": "", "unit": ""},
        ]

        registry = _make_registry()
        result = registry.execute("review_all_formulas", {})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_formulas"], 3)
        self.assertEqual(result["valid"], 1)
        self.assertEqual(result["invalid"], 2)
        self.assertIn("2 formula(s) need attention", result["recommendation"])


class TestReviewAllFormulasEmpty(unittest.TestCase):
    """No metric definitions found."""

    @patch(_PATCH_METRIC_DEFS)
    @patch(_PATCH_SETTINGS)
    def test_no_definitions(self, mock_settings, mock_definitions):
        mock_settings.return_value = _make_settings()
        mock_definitions.return_value = []

        registry = _make_registry()
        result = registry.execute("review_all_formulas", {})

        self.assertEqual(result["status"], "no_definitions")


# ---------------------------------------------------------------------------
# format_proactive_message — self_improvement_check
# ---------------------------------------------------------------------------


class TestFormatSelfImprovementMessage(unittest.TestCase):
    """Verify the self_improvement_check trigger produces correct message."""

    def test_format_self_improvement_message(self):
        from src.agent.proactive import format_proactive_message

        trigger = {
            "type": "self_improvement_check",
            "priority": "low",
            "data": {
                "metric_count": 3,
                "metric_names": ["trimp", "pace", "vo2max_est"],
            },
        }

        msg = format_proactive_message(trigger, {})

        self.assertIn("3 metric definitions", msg)
        self.assertIn("trimp", msg)
        self.assertIn("pace", msg)
        self.assertIn("vo2max_est", msg)
        self.assertIn("review_all_formulas()", msg)
        self.assertIn("evaluate_formula_accuracy()", msg)


# ---------------------------------------------------------------------------
# HeartbeatService — tick count and self-improvement scheduling
# ---------------------------------------------------------------------------


class TestHeartbeatTickCountIncrement(unittest.TestCase):
    """Verify _tick_count increments on each tick."""

    @patch("src.services.heartbeat._fetch_active_user_ids", new_callable=AsyncMock)
    def test_tick_count_increments(self, mock_fetch):
        from src.services.heartbeat import HeartbeatService

        mock_fetch.return_value = []

        svc = HeartbeatService(interval_seconds=60)
        self.assertEqual(svc._tick_count, 0)

        asyncio.run(svc._tick())
        self.assertEqual(svc._tick_count, 1)

        asyncio.run(svc._tick())
        self.assertEqual(svc._tick_count, 2)


class TestHeartbeatSelfImprovementEvery12thTick(unittest.TestCase):
    """Verify _run_self_improvement is called every 12th tick."""

    @patch("src.services.heartbeat._fetch_active_user_ids", new_callable=AsyncMock)
    @patch("src.services.heartbeat._process_user", new_callable=AsyncMock)
    def test_self_improvement_on_12th_tick(self, mock_process, mock_fetch):
        from src.services.heartbeat import HeartbeatService

        mock_fetch.return_value = ["user-1"]
        mock_process.return_value = None

        svc = HeartbeatService(interval_seconds=60)
        svc._run_self_improvement = AsyncMock()

        # Run 11 ticks — should not trigger
        for _ in range(11):
            asyncio.run(svc._tick())

        svc._run_self_improvement.assert_not_called()

        # 12th tick — should trigger
        asyncio.run(svc._tick())
        self.assertEqual(svc._tick_count, 12)
        svc._run_self_improvement.assert_called_once_with(["user-1"])

    @patch("src.services.heartbeat._fetch_active_user_ids", new_callable=AsyncMock)
    @patch("src.services.heartbeat._process_user", new_callable=AsyncMock)
    def test_self_improvement_not_on_non_12th(self, mock_process, mock_fetch):
        from src.services.heartbeat import HeartbeatService

        mock_fetch.return_value = ["user-1"]
        mock_process.return_value = None

        svc = HeartbeatService(interval_seconds=60)
        svc._run_self_improvement = AsyncMock()

        # Run 5 ticks — none should trigger
        for _ in range(5):
            asyncio.run(svc._tick())

        svc._run_self_improvement.assert_not_called()


if __name__ == "__main__":
    unittest.main()
