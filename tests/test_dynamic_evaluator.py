"""Tests for plan evaluator (evaluate_plan with DB-driven criteria).

Covers:
- Dynamic criteria from DB drive evaluation
- No criteria → plan accepted by default (score=100)
- Weight normalization: weights → percentages in prompt
- LLM call mock: response parsing and PlanEvaluation construction
- System prompt generation from DB criteria
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.plan_evaluator import (
    PLAN_ACCEPTANCE_THRESHOLD,
    PlanEvaluation,
    _build_dynamic_system_prompt,
    evaluate_plan,
)


USER_ID = "test-user-evaluator"

SAMPLE_PLAN = {
    "sessions": [
        {"day": "Monday", "sport": "running", "type": "easy",
         "duration_minutes": 45, "targets": {"hr_zone": 2}},
        {"day": "Wednesday", "sport": "running", "type": "intervals",
         "duration_minutes": 60, "targets": {"pace": "4:30/km"}},
        {"day": "Friday", "sport": "cycling", "type": "endurance",
         "duration_minutes": 90, "targets": {"power_zone": 2}},
    ],
}

SAMPLE_PROFILE = {
    "name": "Test Athlete",
    "sports": ["running", "cycling"],
    "goal": {"event": "Half Marathon", "target_date": "2026-06-01"},
    "constraints": {"training_days_per_week": 4, "max_session_minutes": 120},
}

DB_CRITERIA = [
    {"name": "endurance_base", "description": "Sufficient aerobic base volume",
     "weight": 3.0, "formula": ""},
    {"name": "intensity_balance", "description": "Proper hard/easy day distribution",
     "weight": 2.0, "formula": ""},
    {"name": "recovery_quality", "description": "Adequate recovery between sessions",
     "weight": 1.5, "formula": ""},
]


def _mock_llm_response(scores: dict, overall: int = 75) -> MagicMock:
    """Create a mock LLM response with evaluation JSON."""
    result = {
        "overall_score": overall,
        "criteria": scores,
        "issues": ["Test issue 1"],
        "suggestions": ["Test suggestion 1"],
    }
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(result)
    return response


# ---------------------------------------------------------------------------
# _build_dynamic_system_prompt tests
# ---------------------------------------------------------------------------


class TestBuildDynamicSystemPrompt:
    def test_includes_all_criteria(self) -> None:
        prompt = _build_dynamic_system_prompt(DB_CRITERIA)
        assert "endurance_base" in prompt
        assert "intensity_balance" in prompt
        assert "recovery_quality" in prompt

    def test_weight_normalization(self) -> None:
        """Weights 3.0, 2.0, 1.5 = total 6.5 → 46%, 31%, 23%."""
        prompt = _build_dynamic_system_prompt(DB_CRITERIA)
        assert "46%" in prompt  # 3.0 / 6.5 ≈ 46%
        assert "31%" in prompt  # 2.0 / 6.5 ≈ 31%
        assert "23%" in prompt  # 1.5 / 6.5 ≈ 23%

    def test_includes_descriptions(self) -> None:
        prompt = _build_dynamic_system_prompt(DB_CRITERIA)
        assert "Sufficient aerobic base volume" in prompt
        assert "Proper hard/easy day distribution" in prompt

    def test_single_criterion(self) -> None:
        """Single criterion gets 100%."""
        criteria = [{"name": "only_one", "description": "The only criterion", "weight": 1.0}]
        prompt = _build_dynamic_system_prompt(criteria)
        assert "100%" in prompt

    def test_equal_weights(self) -> None:
        """Equal weights → equal percentages."""
        criteria = [
            {"name": "a", "description": "A", "weight": 1.0},
            {"name": "b", "description": "B", "weight": 1.0},
        ]
        prompt = _build_dynamic_system_prompt(criteria)
        assert "50%" in prompt

    def test_zero_total_weight_fallback(self) -> None:
        """If all weights are 0, fallback to equal distribution."""
        criteria = [
            {"name": "a", "description": "A", "weight": 0},
            {"name": "b", "description": "B", "weight": 0},
        ]
        prompt = _build_dynamic_system_prompt(criteria)
        # Should not crash — zero total handled gracefully
        assert "a" in prompt
        assert "b" in prompt

    def test_json_example_in_prompt(self) -> None:
        prompt = _build_dynamic_system_prompt(DB_CRITERIA)
        assert '"endurance_base": 75' in prompt
        assert "overall_score" in prompt


# ---------------------------------------------------------------------------
# evaluate_plan_dynamic tests
# ---------------------------------------------------------------------------


class TestEvaluatePlan:
    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_uses_db_criteria(self, mock_get_criteria, mock_chat) -> None:
        """When DB has criteria, should use dynamic evaluation."""
        mock_get_criteria.return_value = DB_CRITERIA
        mock_chat.return_value = _mock_llm_response(
            {"endurance_base": 80, "intensity_balance": 70, "recovery_quality": 75},
            overall=75,
        )
        result = evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID,
        )
        assert isinstance(result, PlanEvaluation)
        assert result.score == 75
        assert result.acceptable is True
        assert "endurance_base" in result.criteria_scores
        # Verify chat_completion was called (dynamic path)
        mock_chat.assert_called_once()
        # Verify the system prompt used DB criteria
        call_kwargs = mock_chat.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt", "")
        assert "endurance_base" in system_prompt

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_default_accept_when_no_criteria(self, mock_get_criteria, mock_chat) -> None:
        """When DB has no criteria, plan is accepted by default (score=100)."""
        mock_get_criteria.return_value = []
        result = evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID,
        )
        assert isinstance(result, PlanEvaluation)
        assert result.score == 100
        assert result.acceptable is True
        assert result.criteria_scores == {}
        assert result.issues == []
        assert result.suggestions == []
        # No LLM call should be made
        mock_chat.assert_not_called()

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_default_accept_on_db_error(self, mock_get_criteria, mock_chat) -> None:
        """DB error should result in default acceptance (score=100)."""
        mock_get_criteria.side_effect = Exception("DB connection failed")
        result = evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID,
        )
        assert isinstance(result, PlanEvaluation)
        assert result.score == 100
        assert result.acceptable is True
        assert result.criteria_scores == {}
        assert result.issues == []
        assert result.suggestions == []
        # No LLM call should be made
        mock_chat.assert_not_called()

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_low_score_not_acceptable(self, mock_get_criteria, mock_chat) -> None:
        mock_get_criteria.return_value = DB_CRITERIA
        mock_chat.return_value = _mock_llm_response(
            {"endurance_base": 40, "intensity_balance": 50, "recovery_quality": 30},
            overall=40,
        )
        result = evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID,
        )
        assert result.score == 40
        assert result.acceptable is False

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_passes_beliefs(self, mock_get_criteria, mock_chat) -> None:
        mock_get_criteria.return_value = DB_CRITERIA
        mock_chat.return_value = _mock_llm_response(
            {"endurance_base": 80}, overall=80,
        )
        beliefs = [{"text": "Prefers morning runs", "category": "scheduling"}]
        result = evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID, beliefs=beliefs,
        )
        assert isinstance(result, PlanEvaluation)
        # Verify user prompt includes beliefs
        call_args = mock_chat.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
        user_content = messages[0]["content"]
        assert "Prefers morning runs" in user_content

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_issues_and_suggestions(self, mock_get_criteria, mock_chat) -> None:
        mock_get_criteria.return_value = DB_CRITERIA
        mock_chat.return_value = _mock_llm_response(
            {"endurance_base": 80}, overall=80,
        )
        result = evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID,
        )
        assert result.issues == ["Test issue 1"]
        assert result.suggestions == ["Test suggestion 1"]

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_temperature_is_low(self, mock_get_criteria, mock_chat) -> None:
        """Evaluation should use low temperature for consistency."""
        mock_get_criteria.return_value = DB_CRITERIA
        mock_chat.return_value = _mock_llm_response(
            {"endurance_base": 80}, overall=80,
        )
        evaluate_plan(
            SAMPLE_PLAN, SAMPLE_PROFILE, user_id=USER_ID,
        )
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.2


# ---------------------------------------------------------------------------
# PlanEvaluation dataclass tests
# ---------------------------------------------------------------------------


class TestPlanEvaluation:
    def test_acceptable_above_threshold(self) -> None:
        e = PlanEvaluation(score=80, criteria_scores={}, issues=[], suggestions=[])
        assert e.acceptable is True

    def test_not_acceptable_below_threshold(self) -> None:
        e = PlanEvaluation(score=50, criteria_scores={}, issues=[], suggestions=[])
        assert e.acceptable is False

    def test_acceptable_at_threshold(self) -> None:
        e = PlanEvaluation(
            score=PLAN_ACCEPTANCE_THRESHOLD,
            criteria_scores={}, issues=[], suggestions=[],
        )
        assert e.acceptable is True


# ---------------------------------------------------------------------------
# planning_tools integration (evaluate_plan wiring)
# ---------------------------------------------------------------------------


class TestPlanningToolsIntegration:
    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_evaluate_plan_tool_uses_db_criteria(self, mock_get_criteria, mock_chat) -> None:
        """The evaluate_plan tool in planning_tools.py should use DB eval criteria."""
        mock_get_criteria.return_value = DB_CRITERIA
        mock_chat.return_value = _mock_llm_response(
            {"endurance_base": 80}, overall=80,
        )
        user_model = MagicMock()
        user_model.user_id = USER_ID
        user_model.project_profile.return_value = SAMPLE_PROFILE
        user_model.get_active_beliefs.return_value = []

        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.planning_tools import register_planning_tools
        registry = ToolRegistry()

        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID

        with patch("src.agent.tools.planning_tools.get_settings", return_value=settings):
            register_planning_tools(registry, user_model)

        result = registry.execute("evaluate_plan", {"plan": SAMPLE_PLAN})
        assert result["score"] == 80
        assert result["acceptable"] is True

    @patch("src.agent.plan_evaluator.chat_completion")
    @patch("src.db.agent_config_db.get_eval_criteria")
    def test_evaluate_plan_tool_default_accept_no_criteria(self, mock_get_criteria, mock_chat) -> None:
        """When no DB criteria exist, the tool should return default acceptance."""
        mock_get_criteria.return_value = []
        user_model = MagicMock()
        user_model.user_id = USER_ID
        user_model.project_profile.return_value = SAMPLE_PROFILE
        user_model.get_active_beliefs.return_value = []

        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.planning_tools import register_planning_tools
        registry = ToolRegistry()

        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID

        with patch("src.agent.tools.planning_tools.get_settings", return_value=settings):
            register_planning_tools(registry, user_model)

        result = registry.execute("evaluate_plan", {"plan": SAMPLE_PLAN})
        assert result["score"] == 100
        assert result["acceptable"] is True
        assert result["criteria"] == {}
        assert result["issues"] == []
        mock_chat.assert_not_called()
