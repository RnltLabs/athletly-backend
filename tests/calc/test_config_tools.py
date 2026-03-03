"""Tests for config tools — define_metric, define_eval_criteria, get_config.

Supabase is never called: all DB interactions are mocked at the import level.
Tests verify:
- Formula validation gates (invalid formula → error, valid → proceed)
- get_config rejects unknown config_type
- get_config returns structured success dict when Supabase is configured
- define_metric stores metric on valid formula
- define_eval_criteria with optional formula field
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry_with_supabase() -> tuple[ToolRegistry, MagicMock]:
    """Return a ToolRegistry with config tools registered, settings patched to
    enable Supabase mode (use_supabase=True)."""
    mock_settings = MagicMock()
    mock_settings.use_supabase = True
    mock_settings.agenticsports_user_id = "test-user-uuid"

    registry = ToolRegistry()

    with patch("src.agent.tools.config_tools.get_settings", return_value=mock_settings):
        from src.agent.tools.config_tools import register_config_tools
        register_config_tools(registry)

    return registry, mock_settings


def _make_registry_no_supabase() -> ToolRegistry:
    """Return a ToolRegistry where use_supabase=False."""
    mock_settings = MagicMock()
    mock_settings.use_supabase = False
    mock_settings.agenticsports_user_id = ""

    registry = ToolRegistry()

    with patch("src.agent.tools.config_tools.get_settings", return_value=mock_settings):
        from src.agent.tools.config_tools import register_config_tools
        register_config_tools(registry)

    return registry


# ---------------------------------------------------------------------------
# define_metric
# ---------------------------------------------------------------------------


class TestDefineMetric:
    def test_define_metric_validates_formula_bad(self):
        """An invalid formula must return status=error before touching DB."""
        registry = _make_registry_no_supabase()
        result = registry.execute("define_metric", {
            "name": "bad_metric",
            "formula": "__import__('os')",
        })
        assert result["status"] == "error"
        assert "formula" in result["error"].lower() or "invalid" in result["error"].lower()

    def test_define_metric_no_supabase(self):
        """Valid formula but Supabase not configured → error."""
        registry = _make_registry_no_supabase()
        result = registry.execute("define_metric", {
            "name": "speed",
            "formula": "distance / time",
        })
        assert result["status"] == "error"
        assert "supabase" in result["error"].lower()

    def test_define_metric_success(self):
        """Valid formula + Supabase configured → calls upsert and returns success."""
        registry, _ = _make_registry_with_supabase()

        fake_row = {
            "name": "trimp",
            "formula": "duration * hr_ratio * 0.64",
            "unit": "au",
        }

        with patch(
            "src.db.agent_config_db.upsert_metric_definition",
            return_value=fake_row,
        ):
            result = registry.execute("define_metric", {
                "name": "trimp",
                "formula": "duration * hr_ratio * 0.64",
                "unit": "au",
            })

        assert result["status"] == "success"
        assert result["metric"]["name"] == "trimp"

    def test_define_metric_empty_formula_rejected(self):
        registry = _make_registry_no_supabase()
        result = registry.execute("define_metric", {
            "name": "bad",
            "formula": "",
        })
        assert result["status"] == "error"

    def test_define_metric_with_description_and_unit(self):
        """describe and unit fields are passed through without error."""
        registry, _ = _make_registry_with_supabase()

        fake_row = {"name": "pace", "formula": "time / distance"}

        with patch(
            "src.db.agent_config_db.upsert_metric_definition",
            return_value=fake_row,
        ):
            result = registry.execute("define_metric", {
                "name": "pace",
                "formula": "time / distance",
                "description": "Minutes per km",
                "unit": "min/km",
            })

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# define_eval_criteria
# ---------------------------------------------------------------------------


class TestDefineEvalCriteria:
    def test_define_eval_criteria_invalid_formula(self):
        """Optional formula that is invalid → error."""
        registry = _make_registry_no_supabase()
        result = registry.execute("define_eval_criteria", {
            "name": "volume",
            "formula": "open('/etc/passwd')",
        })
        assert result["status"] == "error"
        assert "formula" in result["error"].lower() or "invalid" in result["error"].lower()

    def test_define_eval_criteria_no_formula(self):
        """No formula provided (empty string) → skips validation, hits Supabase check."""
        registry = _make_registry_no_supabase()
        # No formula → should reach the Supabase check
        result = registry.execute("define_eval_criteria", {
            "name": "consistency",
        })
        # use_supabase=False, so fails there
        assert result["status"] == "error"
        assert "supabase" in result["error"].lower()

    def test_define_eval_criteria_success(self):
        registry, _ = _make_registry_with_supabase()

        fake_row = {"name": "volume", "weight": 2.0}

        with patch(
            "src.db.agent_config_db.upsert_eval_criteria",
            return_value=fake_row,
        ):
            result = registry.execute("define_eval_criteria", {
                "name": "volume",
                "weight": 2.0,
                "description": "Total training load",
            })

        assert result["status"] == "success"
        assert result["criteria"]["name"] == "volume"


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_get_config_invalid_type(self):
        """Unknown config_type → status=error, no DB call."""
        registry = _make_registry_no_supabase()
        result = registry.execute("get_config", {"config_type": "not_a_real_type"})
        assert result["status"] == "error"
        assert "unknown" in result["error"].lower() or "not_a_real_type" in result["error"]

    def test_get_config_no_supabase(self):
        """Valid config_type but Supabase not configured → error."""
        registry = _make_registry_no_supabase()
        result = registry.execute("get_config", {"config_type": "metric_definitions"})
        assert result["status"] == "error"
        assert "supabase" in result["error"].lower()

    def test_get_config_metric_definitions_success(self):
        registry, _ = _make_registry_with_supabase()

        fake_items = [{"name": "trimp", "formula": "a * b"}]

        with patch(
            "src.db.agent_config_db.get_metric_definitions",
            return_value=fake_items,
        ):
            result = registry.execute("get_config", {"config_type": "metric_definitions"})

        assert result["status"] == "success"
        assert result["config_type"] == "metric_definitions"
        assert result["count"] == 1
        assert result["items"] == fake_items

    def test_get_config_all_valid_types_accepted(self):
        """All five valid config_type values must pass the type check."""
        valid_types = [
            "metric_definitions",
            "eval_criteria",
            "session_schemas",
            "periodization_models",
            "proactive_trigger_rules",
        ]
        registry = _make_registry_no_supabase()

        for config_type in valid_types:
            result = registry.execute("get_config", {"config_type": config_type})
            # Reaches the Supabase check (not the type-check error)
            assert "unknown" not in result.get("error", "").lower(), (
                f"config_type '{config_type}' should be valid but was rejected"
            )
