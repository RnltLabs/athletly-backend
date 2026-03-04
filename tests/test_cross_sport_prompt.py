"""Tests for Cross-Sport Reasoning in system prompt.

Verifies that:
- STATIC_SYSTEM_PROMPT contains the Cross-Sport Reasoning section
- Runtime context includes per-sport breakdown for multi-sport athletes
- Runtime context handles single-sport athletes gracefully
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent.system_prompt import STATIC_SYSTEM_PROMPT, build_runtime_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_user_model(
    name: str = "Test",
    sports: list[str] | None = None,
) -> MagicMock:
    """Create a mock user_model with a configurable profile."""
    profile = {
        "name": name,
        "sports": sports or ["running", "cycling"],
        "goal": {"event": "10K race", "target_date": None},
        "constraints": {
            "training_days_per_week": 4,
            "max_session_minutes": 60,
        },
        "fitness": {},
    }
    mock = MagicMock()
    mock.project_profile.return_value = profile
    mock.get_active_beliefs.return_value = []
    mock.get_active_plan_summary.return_value = None
    return mock


def _make_settings(user_id: str = "test-user") -> MagicMock:
    s = MagicMock()
    s.use_supabase = True
    s.agenticsports_user_id = user_id
    return s


def _make_load_summary(
    sessions_by_sport: dict[str, int] | None = None,
    sports_seen: list[str] | None = None,
    total_sessions: int = 8,
    total_minutes: float = 360.0,
    total_trimp: float = 520.0,
) -> dict:
    by_sport = sessions_by_sport or {"running": 5, "cycling": 3}
    return {
        "total_sessions": total_sessions,
        "total_minutes": total_minutes,
        "total_trimp": total_trimp,
        "sports_seen": sports_seen or sorted(by_sport.keys()),
        "sessions_by_sport": by_sport,
        "sessions_by_source": {"agent": 4, "health": 4},
    }


# ---------------------------------------------------------------------------
# Static prompt tests
# ---------------------------------------------------------------------------


class TestStaticPromptCrossSport:
    """Verify the Cross-Sport Reasoning section exists in the static prompt."""

    def test_static_prompt_contains_cross_sport_section(self) -> None:
        """STATIC_SYSTEM_PROMPT includes the Cross-Sport Reasoning header."""
        assert "## Cross-Sport Reasoning" in STATIC_SYSTEM_PROMPT

    def test_cross_sport_before_self_correction(self) -> None:
        """Cross-Sport Reasoning appears BEFORE Self-Correction."""
        cross_idx = STATIC_SYSTEM_PROMPT.index("## Cross-Sport Reasoning")
        self_idx = STATIC_SYSTEM_PROMPT.index("## Self-Correction")
        assert cross_idx < self_idx

    def test_cross_sport_mentions_muscle_overlap(self) -> None:
        """Cross-Sport Reasoning covers muscle group overlap."""
        assert "Muscle Group Overlap" in STATIC_SYSTEM_PROMPT

    def test_cross_sport_mentions_energy_system(self) -> None:
        """Cross-Sport Reasoning covers energy system overlap."""
        assert "Energy System Overlap" in STATIC_SYSTEM_PROMPT

    def test_cross_sport_mentions_recovery_awareness(self) -> None:
        """Cross-Sport Reasoning covers recovery window awareness."""
        assert "Recovery Window Awareness" in STATIC_SYSTEM_PROMPT

    def test_cross_sport_mentions_multi_sport_planning(self) -> None:
        """Cross-Sport Reasoning covers multi-sport week planning."""
        assert "Multi-Sport Week Planning" in STATIC_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Runtime context tests
# ---------------------------------------------------------------------------


class TestRuntimeContextSportBreakdown:
    """Verify runtime context includes per-sport breakdown."""

    def test_runtime_context_includes_sport_breakdown(self) -> None:
        """Multi-sport load summary includes per-sport session counts."""
        user_model = _make_mock_user_model(sports=["running", "cycling"])
        summary = _make_load_summary(
            sessions_by_sport={"running": 5, "cycling": 3},
        )

        with (
            patch(
                "src.config.get_settings",
                return_value=_make_settings(),
            ),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value=summary,
            ),
        ):
            result = build_runtime_context(user_model, date="2026-03-04")

        assert "Per-Sport Breakdown" in result
        assert "running: 5 sessions" in result
        assert "cycling: 3 sessions" in result

    def test_runtime_context_handles_single_sport(self) -> None:
        """Single-sport athletes get load summary without per-sport section."""
        user_model = _make_mock_user_model(sports=["running"])
        summary = _make_load_summary(
            sessions_by_sport={"running": 6},
            sports_seen=["running"],
            total_sessions=6,
        )

        with (
            patch(
                "src.config.get_settings",
                return_value=_make_settings(),
            ),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value=summary,
            ),
        ):
            result = build_runtime_context(user_model, date="2026-03-04")

        assert "This Week's Training Load" in result
        assert "Sessions: 6" in result
        # Single sport: no redundant breakdown
        assert "Per-Sport Breakdown" not in result

    def test_runtime_context_no_sessions(self) -> None:
        """When there are zero sessions, no load section appears."""
        user_model = _make_mock_user_model()
        summary = _make_load_summary(
            sessions_by_sport={},
            sports_seen=[],
            total_sessions=0,
            total_minutes=0.0,
            total_trimp=0.0,
        )

        with (
            patch(
                "src.config.get_settings",
                return_value=_make_settings(),
            ),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value=summary,
            ),
        ):
            result = build_runtime_context(user_model, date="2026-03-04")

        assert "This Week's Training Load" not in result
