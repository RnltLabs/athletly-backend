"""Tests for onboarding config gate — complete_onboarding validation.

Verifies that complete_onboarding rejects completion when:
- session_schemas are missing
- metric_definitions are missing
- weekly_plans are missing
And succeeds when all configs + plan exist.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent.tools.registry import ToolRegistry


def _make_user_model(
    sports: list | None = None,
    goal_event: str = "Marathon",
) -> MagicMock:
    """Create a mock user_model with valid profile for onboarding."""
    profile = {
        "sports": ["running"] if sports is None else sports,
        "goal": {"event": goal_event, "target_date": "2026-10-01"},
    }
    mock = MagicMock()
    mock.project_profile.return_value = profile
    mock.user_id = "test-user-123"
    mock.meta = {}
    return mock


class FakeQuery:
    """Chainable Supabase query mock that returns configured data on execute()."""

    def __init__(self, data: list):
        self._data = data

    def select(self, *_a, **_kw) -> "FakeQuery":
        return self

    def eq(self, *_a, **_kw) -> "FakeQuery":
        return self

    def limit(self, *_a, **_kw) -> "FakeQuery":
        return self

    def update(self, *_a, **_kw) -> "FakeQuery":
        return self

    def execute(self) -> MagicMock:
        resp = MagicMock()
        resp.data = self._data
        return resp


def _build_supabase_mock(
    has_schemas: bool = True,
    has_metrics: bool = True,
    has_plans: bool = True,
) -> MagicMock:
    """Build a mock Supabase client with configurable gate responses."""
    sb = MagicMock()
    dummy = [{"id": "fake"}]

    # Track eq calls to dispatch correct response per config_type / table
    class DispatchQuery(FakeQuery):
        def __init__(self) -> None:
            super().__init__([])
            self._eq_values: list[tuple[str, str]] = []

        def eq(self, field: str, value: str) -> "DispatchQuery":
            self._eq_values.append((field, value))
            return self

        def execute(self) -> MagicMock:
            resp = MagicMock()
            eq_map = dict(self._eq_values)

            config_type = eq_map.get("config_type")
            status = eq_map.get("status")

            if config_type == "session_schema":
                resp.data = dummy if has_schemas else []
            elif config_type == "metric":
                resp.data = dummy if has_metrics else []
            elif status == "active":
                # weekly_plans query
                resp.data = dummy if has_plans else []
            else:
                resp.data = dummy
            return resp

        def select(self, *_a, **_kw) -> "DispatchQuery":
            return self

        def limit(self, *_a, **_kw) -> "DispatchQuery":
            return self

        def update(self, *_a, **_kw) -> "DispatchQuery":
            return self

    def table_side_effect(table_name: str):
        return DispatchQuery()

    sb.table = table_side_effect
    return sb


def _register_and_call(user_model: MagicMock, supabase_mock: MagicMock) -> dict:
    """Register onboarding tools and call complete_onboarding."""
    registry = ToolRegistry()
    with (
        patch("src.agent.tools.onboarding_tools.get_settings") as mock_settings,
        patch("src.db.client.get_supabase", return_value=supabase_mock),
    ):
        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = "fallback-user"
        mock_settings.return_value = settings

        from src.agent.tools.onboarding_tools import register_onboarding_tools

        register_onboarding_tools(registry, user_model)

        return registry.execute("complete_onboarding", {})


class TestOnboardingConfigGate:
    """Test the config gate in complete_onboarding."""

    def test_fails_without_session_schemas(self) -> None:
        user_model = _make_user_model()
        sb = _build_supabase_mock(has_schemas=False)
        result = _register_and_call(user_model, sb)

        assert result["status"] == "error"
        assert "session_schemas" in result["error"]

    def test_fails_without_metrics(self) -> None:
        user_model = _make_user_model()
        sb = _build_supabase_mock(has_metrics=False)
        result = _register_and_call(user_model, sb)

        assert result["status"] == "error"
        assert "metrics" in result["error"]

    def test_fails_without_training_plan(self) -> None:
        user_model = _make_user_model()
        sb = _build_supabase_mock(has_plans=False)
        result = _register_and_call(user_model, sb)

        assert result["status"] == "error"
        assert "training_plan" in result["error"]

    def test_fails_with_multiple_missing(self) -> None:
        user_model = _make_user_model()
        sb = _build_supabase_mock(has_schemas=False, has_metrics=False, has_plans=False)
        result = _register_and_call(user_model, sb)

        assert result["status"] == "error"
        assert "session_schemas" in result["error"]
        assert "metrics" in result["error"]
        assert "training_plan" in result["error"]

    def test_succeeds_when_all_present(self) -> None:
        user_model = _make_user_model()
        sb = _build_supabase_mock(has_schemas=True, has_metrics=True, has_plans=True)
        result = _register_and_call(user_model, sb)

        assert result["status"] == "success"
        assert result["onboarding_complete"] is True

    def test_still_fails_on_missing_sports(self) -> None:
        """Gate 1 (sports/goal) still works before Gate 2."""
        user_model = _make_user_model(sports=[])
        sb = _build_supabase_mock()
        result = _register_and_call(user_model, sb)

        assert result["status"] == "error"
        assert "sports" in result["error"]
