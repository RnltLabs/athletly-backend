"""Step 5 tests: Trajectory assessment, confidence scoring, proactive communication.

Updated to use the Supabase-backed proactive queue (file-based I/O removed).
All queue operations are mocked via src.db.proactive_queue_db.
"""

from unittest.mock import patch

from src.agent.trajectory import calculate_confidence
from src.agent.proactive import (
    check_proactive_triggers,
    format_proactive_message,
    queue_proactive_message,
    get_pending_messages,
    refresh_proactive_triggers,
)

USER_ID = "test-user-id"


# ── Helpers ───────────────────────────────────────────────────────────

def _test_profile():
    return {
        "name": "Test Athlete",
        "sports": ["running"],
        "goal": {
            "event": "Half Marathon",
            "target_date": "2026-08-15",
            "target_time": "1:45:00",
        },
        "fitness": {
            "estimated_vo2max": None,
            "threshold_pace_min_km": None,
            "weekly_volume_km": None,
            "trend": "unknown",
        },
        "constraints": {
            "training_days_per_week": 5,
            "max_session_minutes": 90,
            "available_sports": ["running"],
        },
    }


def _mock_episodes():
    """Create mock episodes for testing."""
    return [
        {
            "id": "ep_2026-02-02",
            "block": "2026-W05",
            "compliance_rate": 0.4,
            "key_observations": ["Only 2/5 sessions completed"],
            "lessons": ["Need better schedule management for Thursday sessions"],
            "patterns_detected": ["Thursday sessions frequently skipped"],
            "fitness_delta": {"estimated_vo2max_change": "stable", "weekly_volume_trend": "stable"},
            "confidence": 0.5,
        },
        {
            "id": "ep_2026-02-09",
            "block": "2026-W06",
            "compliance_rate": 0.8,
            "key_observations": ["HR improving on easy runs", "Missed Thursday"],
            "lessons": ["Thursday is unreliable for hard sessions", "Easy pace can be updated"],
            "patterns_detected": ["Consistent Thursday skipping", "HR trend improving"],
            "fitness_delta": {"estimated_vo2max_change": "+0.5", "weekly_volume_trend": "increasing"},
            "confidence": 0.6,
        },
        {
            "id": "ep_2026-02-16",
            "block": "2026-W07",
            "compliance_rate": 1.0,
            "key_observations": ["All sessions completed", "Long run improved", "Sunday showed fatigue"],
            "lessons": ["Long run distance can increase", "Watch for fatigue after volume increases"],
            "patterns_detected": ["Steady aerobic improvement"],
            "fitness_delta": {"estimated_vo2max_change": "+0.8", "weekly_volume_trend": "increasing"},
            "confidence": 0.7,
        },
    ]


# ── Confidence Scoring (unit tests, no API) ──────────────────────────

class TestConfidenceScoring:
    def test_less_than_4_weeks_capped_at_05(self):
        conf = calculate_confidence(data_points=10, consistency=0.9, weeks_of_data=3)
        assert conf <= 0.5

    def test_less_than_8_weeks_capped_at_075(self):
        conf = calculate_confidence(data_points=25, consistency=0.9, weeks_of_data=6)
        assert conf <= 0.75

    def test_12_plus_weeks_high_confidence(self):
        conf = calculate_confidence(data_points=60, consistency=0.9, weeks_of_data=12)
        assert conf >= 0.75

    def test_inconsistent_training_reduces_confidence(self):
        high = calculate_confidence(data_points=20, consistency=0.9, weeks_of_data=6)
        low = calculate_confidence(data_points=20, consistency=0.5, weeks_of_data=6)
        assert low < high

    def test_more_data_points_increase_confidence(self):
        few = calculate_confidence(data_points=5, consistency=0.9, weeks_of_data=8)
        many = calculate_confidence(data_points=40, consistency=0.9, weeks_of_data=8)
        assert many >= few

    def test_zero_weeks_very_low(self):
        conf = calculate_confidence(data_points=0, consistency=0.0, weeks_of_data=0)
        assert conf <= 0.15


# ── Proactive Triggers (unit tests, no API) ──────────────────────────

class TestProactiveTriggers:
    def test_on_track_trigger(self):
        trajectory = {
            "trajectory": {"on_track": True, "predicted_race_time": "1:43-1:48"},
            "confidence": 0.65,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], _mock_episodes(), trajectory
        )
        types = [t["type"] for t in triggers]
        assert "on_track" in types

    def test_goal_at_risk_trigger(self):
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "1:55-2:05"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], _mock_episodes(), trajectory
        )
        types = [t["type"] for t in triggers]
        assert "goal_at_risk" in types

    def test_missed_session_pattern_trigger(self):
        episodes = _mock_episodes()  # contain "Thursday" skip patterns
        trajectory = {
            "trajectory": {"on_track": True},
            "confidence": 0.5,
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], episodes, trajectory
        )
        types = [t["type"] for t in triggers]
        assert "missed_session_pattern" in types

    def test_fitness_improving_trigger(self):
        episodes = _mock_episodes()  # contain "increasing" volume trend
        trajectory = {
            "trajectory": {"on_track": True},
            "confidence": 0.5,
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], episodes, trajectory
        )
        types = [t["type"] for t in triggers]
        assert "fitness_improving" in types


class TestProactiveMessages:
    def test_on_track_message(self):
        trigger = {"type": "on_track", "data": {"predicted_time": "1:43-1:48", "confidence": 0.65}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "1:43-1:48" in msg
        assert "65%" in msg

    def test_goal_at_risk_message(self):
        trigger = {"type": "goal_at_risk", "data": {"predicted_time": "1:55-2:05", "target_time": "1:45:00"}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "1:55-2:05" in msg
        assert "1:45:00" in msg

    def test_missed_session_message(self):
        trigger = {"type": "missed_session_pattern", "data": {"day": "Thursday", "missed_count": 3}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "Thursday" in msg

    def test_fatigue_warning_message(self):
        trigger = {"type": "fatigue_warning", "data": {"message": "fatigue detected"}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "fatigue" in msg.lower()


# ── Queue round-trip (DB-mocked) ─────────────────────────────────────

class TestQueueRoundTrip:
    """Replaces the old file-based queue tests with DB-mock equivalents.

    The core behaviour under test is unchanged: queue_proactive_message stores
    a message and get_pending_messages retrieves it.  Only the I/O layer
    (file system → Supabase) has changed.
    """

    def test_queue_message_returns_db_row(self) -> None:
        trigger = {
            "type": "goal_at_risk",
            "priority": "high",
            "data": {"predicted_time": "2:00:00", "target_time": "1:45:00"},
        }
        expected_row = {"id": "row-1", "trigger_type": "goal_at_risk", "priority": 0.9}

        with patch("src.db.proactive_queue_db.queue_message", return_value=expected_row):
            result = queue_proactive_message(USER_ID, trigger, priority=0.9)

        assert result["id"] == "row-1"
        assert result["trigger_type"] == "goal_at_risk"

    def test_get_pending_messages_returns_db_rows(self) -> None:
        stored = [
            {"id": "1", "trigger_type": "goal_at_risk", "priority": 0.9, "status": "pending"},
            {"id": "2", "trigger_type": "fitness_improving", "priority": 0.2, "status": "pending"},
        ]

        with patch("src.db.proactive_queue_db.get_pending_messages", return_value=stored):
            result = get_pending_messages(USER_ID)

        assert len(result) == 2
        assert result[0]["trigger_type"] == "goal_at_risk"

    def test_queue_then_get_reflects_stored_message(self) -> None:
        """Simulate queue → get round-trip via mocked DB."""
        trigger = {
            "type": "fatigue_warning",
            "priority": "high",
            "data": {"message": "Tired"},
        }
        db_row = {"id": "99", "trigger_type": "fatigue_warning", "priority": 0.9, "status": "pending"}

        with patch("src.db.proactive_queue_db.queue_message", return_value=db_row):
            queued = queue_proactive_message(USER_ID, trigger, priority=0.9)

        # After queuing, the same row appears in pending list
        with patch("src.db.proactive_queue_db.get_pending_messages", return_value=[queued]):
            pending = get_pending_messages(USER_ID)

        assert len(pending) == 1
        assert pending[0]["trigger_type"] == "fatigue_warning"


# ── Full cycle: trajectory → triggers → queue (DB-mocked) ────────────

class TestFullCycleTrajectoryToQueue:
    """Trajectory → check_proactive_triggers → refresh_proactive_triggers.

    This mirrors the old test_full_cycle_trajectory_proactive integration test
    but replaces API calls and file I/O with mocks.
    """

    def test_trajectory_assessment_produces_triggers(self) -> None:
        """Meaningful trajectory data yields at least one trigger."""
        profile = _test_profile()
        episodes = _mock_episodes()
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:00:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }

        triggers = check_proactive_triggers(profile, [], episodes, trajectory)
        assert len(triggers) > 0

    def test_triggers_have_meaningful_messages(self) -> None:
        profile = _test_profile()
        episodes = _mock_episodes()
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:00:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }

        triggers = check_proactive_triggers(profile, [], episodes, trajectory)
        for trigger in triggers:
            msg = format_proactive_message(trigger, profile)
            assert isinstance(msg, str)
            assert len(msg) > 10

    def test_refresh_queues_all_new_triggers(self) -> None:
        """refresh_proactive_triggers queues every trigger not already pending."""
        profile = _test_profile()
        episodes = _mock_episodes()
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:00:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }

        queued_count = 0

        def fake_queue(**kwargs):
            nonlocal queued_count
            queued_count += 1
            return {"id": str(queued_count), "trigger_type": kwargs["trigger_type"]}

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=[]),
            patch("src.db.proactive_queue_db.queue_message", side_effect=fake_queue),
        ):
            result = refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=episodes,
                trajectory=trajectory,
                athlete_profile=profile,
            )

        assert len(result) == queued_count
        assert queued_count > 0

    def test_refresh_skips_already_pending_triggers(self) -> None:
        """If goal_at_risk is already pending, it must not be re-queued."""
        profile = _test_profile()
        episodes = _mock_episodes()
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:00:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        already_pending = [{"id": "existing", "trigger_type": "goal_at_risk"}]

        queued_types: list[str] = []

        def fake_queue(**kwargs):
            queued_types.append(kwargs["trigger_type"])
            return {"id": "new", "trigger_type": kwargs["trigger_type"]}

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=already_pending),
            patch("src.db.proactive_queue_db.queue_message", side_effect=fake_queue),
        ):
            refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=episodes,
                trajectory=trajectory,
                athlete_profile=profile,
            )

        assert "goal_at_risk" not in queued_types


