"""Tests for the Supabase-backed proactive communication layer.

Covers:
- queue_proactive_message: delegates to proactive_queue_db.queue_message
- get_pending_messages: delegates to proactive_queue_db.get_pending_messages
- deliver_message: delegates to proactive_queue_db.deliver_message
- record_engagement: delegates to proactive_queue_db.record_engagement
- expire_stale_messages: delegates to proactive_queue_db.expire_stale_messages
- refresh_proactive_triggers: checks triggers, deduplicates against pending, queues new
- check_proactive_triggers: pure logic — correct trigger types returned
- format_proactive_message: pure logic — human-readable output
- calculate_silence_decay: correct urgency boost per silence duration
- check_conversation_triggers: silence detection
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pytest

from src.agent.proactive import (
    calculate_silence_decay,
    check_conversation_triggers,
    check_proactive_triggers,
    deliver_message,
    expire_stale_messages,
    format_proactive_message,
    get_pending_messages,
    queue_proactive_message,
    record_engagement,
    refresh_proactive_triggers,
)

USER_ID = "test-user-id"


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------


def _make_profile() -> dict:
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
    }


def _make_episodes_with_thursday_skip() -> list[dict]:
    return [
        {
            "id": "ep_2026-02-02",
            "block": "2026-W05",
            "compliance_rate": 0.4,
            "lessons": ["Thursday sessions are hard to keep"],
            "patterns_detected": ["Thursday sessions frequently skipped"],
            "fitness_delta": {
                "estimated_vo2max_change": "stable",
                "weekly_volume_trend": "stable",
            },
            "confidence": 0.5,
        },
        {
            "id": "ep_2026-02-09",
            "block": "2026-W06",
            "compliance_rate": 0.8,
            "lessons": ["Thursday is unreliable for hard sessions"],
            "patterns_detected": ["Consistent Thursday skipping", "HR trend improving"],
            "fitness_delta": {
                "estimated_vo2max_change": "+0.5",
                "weekly_volume_trend": "increasing",
            },
            "confidence": 0.6,
        },
    ]


def _make_improving_episodes() -> list[dict]:
    return [
        {
            "id": "ep_improving",
            "block": "2026-W07",
            "compliance_rate": 1.0,
            "lessons": [],
            "patterns_detected": [],
            "fitness_delta": {
                "estimated_vo2max_change": "+0.8",
                "weekly_volume_trend": "increasing",
            },
            "confidence": 0.7,
        },
    ]


# ---------------------------------------------------------------------------
# queue_proactive_message — delegation tests
# ---------------------------------------------------------------------------


class TestQueueProactiveMessage:
    """queue_proactive_message must delegate to proactive_queue_db.queue_message."""

    def test_calls_db_queue_message_with_correct_args(self) -> None:
        trigger = {
            "type": "fatigue_warning",
            "priority": "high",
            "data": {"message": "Fatigue detected"},
        }
        expected_row = {"id": "row-123", "trigger_type": "fatigue_warning"}

        with patch("src.db.proactive_queue_db.queue_message", return_value=expected_row) as mock_queue:
            result = queue_proactive_message(USER_ID, trigger, priority=0.9)

        mock_queue.assert_called_once()
        kwargs = mock_queue.call_args.kwargs
        assert kwargs["user_id"] == USER_ID
        assert kwargs["trigger_type"] == "fatigue_warning"
        assert kwargs["priority"] == 0.9
        assert kwargs["data"] == {"message": "Fatigue detected"}
        assert isinstance(kwargs["message_text"], str)
        assert len(kwargs["message_text"]) > 0
        assert result == expected_row

    def test_message_text_is_formatted_from_trigger(self) -> None:
        trigger = {
            "type": "on_track",
            "priority": "low",
            "data": {"predicted_time": "1:43:00", "confidence": 0.7},
        }
        captured_text: list[str] = []

        def capture_call(**kwargs: object) -> dict:
            captured_text.append(kwargs["message_text"])  # type: ignore[arg-type]
            return {"id": "abc"}

        with patch("src.db.proactive_queue_db.queue_message", side_effect=capture_call):
            queue_proactive_message(USER_ID, trigger, priority=0.2)

        assert len(captured_text) == 1
        # format_proactive_message for on_track includes the predicted time
        assert "1:43:00" in captured_text[0]

    def test_default_priority_is_half(self) -> None:
        trigger = {"type": "goal_at_risk", "priority": "high", "data": {}}

        with patch("src.db.proactive_queue_db.queue_message", return_value={"id": "x"}) as mock_queue:
            queue_proactive_message(USER_ID, trigger)  # no explicit priority

        assert mock_queue.call_args.kwargs["priority"] == 0.5

    def test_unknown_trigger_type_uses_fallback(self) -> None:
        trigger = {"type": "unknown_type", "priority": "low", "data": {"key": "val"}}

        with patch("src.db.proactive_queue_db.queue_message", return_value={"id": "y"}) as mock_queue:
            queue_proactive_message(USER_ID, trigger, priority=0.1)

        assert mock_queue.call_args.kwargs["trigger_type"] == "unknown_type"

    def test_context_is_forwarded_to_formatter(self) -> None:
        trigger = {
            "type": "on_track",
            "priority": "low",
            "data": {"predicted_time": "1:43:00", "confidence": 0.7},
        }
        context = {"goal": {"event": "Marathon"}}
        captured: list[str] = []

        def capture(**kwargs: object) -> dict:
            captured.append(kwargs["message_text"])  # type: ignore[arg-type]
            return {"id": "z"}

        with patch("src.db.proactive_queue_db.queue_message", side_effect=capture):
            queue_proactive_message(USER_ID, trigger, priority=0.2, context=context)

        assert "Marathon" in captured[0]


# ---------------------------------------------------------------------------
# get_pending_messages — delegation test
# ---------------------------------------------------------------------------


class TestGetPendingMessages:
    def test_delegates_to_db(self) -> None:
        expected = [{"id": "1", "trigger_type": "on_track", "priority": 0.2}]

        with patch("src.db.proactive_queue_db.get_pending_messages", return_value=expected) as mock_get:
            result = get_pending_messages(USER_ID)

        mock_get.assert_called_once_with(USER_ID)
        assert result == expected

    def test_returns_empty_list_when_none_pending(self) -> None:
        with patch("src.db.proactive_queue_db.get_pending_messages", return_value=[]):
            result = get_pending_messages(USER_ID)

        assert result == []


# ---------------------------------------------------------------------------
# deliver_message — delegation test
# ---------------------------------------------------------------------------


class TestDeliverMessage:
    def test_delegates_to_db(self) -> None:
        expected = {"id": "msg-42", "status": "delivered"}

        with patch("src.db.proactive_queue_db.deliver_message", return_value=expected) as mock_deliver:
            result = deliver_message(USER_ID, "msg-42")

        mock_deliver.assert_called_once_with(USER_ID, "msg-42")
        assert result == expected

    def test_returns_none_when_not_found(self) -> None:
        with patch("src.db.proactive_queue_db.deliver_message", return_value=None):
            result = deliver_message(USER_ID, "nonexistent-id")

        assert result is None


# ---------------------------------------------------------------------------
# record_engagement — delegation test
# ---------------------------------------------------------------------------


class TestRecordEngagement:
    def test_delegates_all_params(self) -> None:
        expected = {"id": "msg-7", "engagement_tracking": {"user_continued_session": True}}

        with patch("src.db.proactive_queue_db.record_engagement", return_value=expected) as mock_record:
            result = record_engagement(
                USER_ID,
                "msg-7",
                responded=True,
                continued_session=True,
                turns_after=4,
            )

        mock_record.assert_called_once_with(USER_ID, "msg-7", True, True, 4)
        assert result == expected

    def test_default_params_are_false_and_zero(self) -> None:
        with patch("src.db.proactive_queue_db.record_engagement", return_value={"id": "x"}) as mock_record:
            record_engagement(USER_ID, "msg-8")

        mock_record.assert_called_once_with(USER_ID, "msg-8", False, False, 0)

    def test_returns_none_when_not_found(self) -> None:
        with patch("src.db.proactive_queue_db.record_engagement", return_value=None):
            result = record_engagement(USER_ID, "gone")

        assert result is None


# ---------------------------------------------------------------------------
# expire_stale_messages — delegation test
# ---------------------------------------------------------------------------


class TestExpireStaleMessages:
    def test_delegates_to_db_with_defaults(self) -> None:
        expired_rows = [{"id": "old-1", "status": "expired"}]

        with patch("src.db.proactive_queue_db.expire_stale_messages", return_value=expired_rows) as mock_expire:
            result = expire_stale_messages(USER_ID)

        mock_expire.assert_called_once_with(USER_ID, 7)
        assert result == expired_rows

    def test_delegates_custom_max_age_days(self) -> None:
        with patch("src.db.proactive_queue_db.expire_stale_messages", return_value=[]) as mock_expire:
            expire_stale_messages(USER_ID, max_age_days=14)

        mock_expire.assert_called_once_with(USER_ID, 14)

    def test_returns_empty_list_when_nothing_expired(self) -> None:
        with patch("src.db.proactive_queue_db.expire_stale_messages", return_value=[]):
            result = expire_stale_messages(USER_ID)

        assert result == []


# ---------------------------------------------------------------------------
# refresh_proactive_triggers — orchestration tests
# ---------------------------------------------------------------------------


class TestRefreshProactiveTriggers:
    """refresh_proactive_triggers runs check_proactive_triggers and queues new ones."""

    def _trajectory_on_track(self) -> dict:
        return {
            "trajectory": {"on_track": True, "predicted_race_time": "1:43:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }

    def test_queues_new_triggers(self) -> None:
        trajectory = self._trajectory_on_track()
        episodes = _make_improving_episodes()

        queued_rows: list[dict] = []

        def fake_queue_msg(**kwargs: object) -> dict:
            row = {"id": f"row-{len(queued_rows)}", "trigger_type": kwargs["trigger_type"]}
            queued_rows.append(row)
            return row

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=[]),
            patch("src.db.proactive_queue_db.queue_message", side_effect=fake_queue_msg),
        ):
            result = refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=episodes,
                trajectory=trajectory,
                athlete_profile=_make_profile(),
            )

        assert len(result) > 0
        returned_types = {r["trigger_type"] for r in result}
        assert "on_track" in returned_types or "fitness_improving" in returned_types

    def test_skips_already_pending_trigger_types(self) -> None:
        trajectory = self._trajectory_on_track()
        # on_track is already pending
        pending_msgs = [{"id": "existing", "trigger_type": "on_track"}]

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=pending_msgs),
            patch("src.db.proactive_queue_db.queue_message") as mock_queue,
        ):
            refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=[],
                trajectory=trajectory,
                athlete_profile=_make_profile(),
            )

        # on_track must not be re-queued
        for call_kwargs in (c.kwargs for c in mock_queue.call_args_list):
            assert call_kwargs.get("trigger_type") != "on_track"

    def test_returns_empty_list_when_no_triggers(self) -> None:
        # Neutral trajectory — no meaningful triggers
        trajectory = {
            "trajectory": {},
            "confidence": 0.3,
            "goal": {},
        }

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=[]),
            patch("src.db.proactive_queue_db.queue_message") as mock_queue,
        ):
            result = refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=[],
                trajectory=trajectory,
                athlete_profile=_make_profile(),
            )

        assert result == []
        mock_queue.assert_not_called()

    def test_no_duplicate_queues_within_single_call(self) -> None:
        """If the same trigger type appears twice in check_proactive_triggers output,
        only the first occurrence must be queued."""
        trajectory = self._trajectory_on_track()
        episodes = _make_improving_episodes()

        queued_types: list[str] = []

        def fake_queue(**kwargs: object) -> dict:
            trigger_type = kwargs["trigger_type"]  # type: ignore[arg-type]
            queued_types.append(trigger_type)
            return {"id": "x", "trigger_type": trigger_type}

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=[]),
            patch("src.db.proactive_queue_db.queue_message", side_effect=fake_queue),
        ):
            refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=episodes,
                trajectory=trajectory,
                athlete_profile=_make_profile(),
            )

        # No trigger type should appear more than once
        assert len(queued_types) == len(set(queued_types))

    def test_priority_string_mapped_to_float(self) -> None:
        """'high' → 0.9, 'medium' → 0.5, 'low' → 0.2."""
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:10:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        captured_priorities: list[float] = []

        def fake_queue(**kwargs: object) -> dict:
            captured_priorities.append(kwargs["priority"])  # type: ignore[arg-type]
            return {"id": "p", "trigger_type": kwargs["trigger_type"]}

        with (
            patch("src.db.proactive_queue_db.get_pending_messages", return_value=[]),
            patch("src.db.proactive_queue_db.queue_message", side_effect=fake_queue),
        ):
            refresh_proactive_triggers(
                USER_ID,
                activities=[],
                episodes=[],
                trajectory=trajectory,
                athlete_profile=_make_profile(),
            )

        # goal_at_risk has priority "high" → 0.9
        assert 0.9 in captured_priorities


# ---------------------------------------------------------------------------
# check_proactive_triggers — pure logic tests (no I/O)
# ---------------------------------------------------------------------------


class TestCheckProactiveTriggers:
    def test_on_track_trigger_when_on_track_and_confidence_sufficient(self) -> None:
        trajectory = {
            "trajectory": {"on_track": True, "predicted_race_time": "1:43:00"},
            "confidence": 0.65,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        types = [t["type"] for t in triggers]
        assert "on_track" in types

    def test_on_track_skipped_when_confidence_below_threshold(self) -> None:
        trajectory = {
            "trajectory": {"on_track": True, "predicted_race_time": "1:43:00"},
            "confidence": 0.4,  # below 0.5
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        types = [t["type"] for t in triggers]
        assert "on_track" not in types

    def test_goal_at_risk_trigger_when_not_on_track(self) -> None:
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "1:55:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        types = [t["type"] for t in triggers]
        assert "goal_at_risk" in types

    def test_goal_at_risk_data_contains_times(self) -> None:
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:05:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        at_risk = next(t for t in triggers if t["type"] == "goal_at_risk")
        assert at_risk["data"]["predicted_time"] == "2:05:00"
        assert at_risk["data"]["target_time"] == "1:45:00"

    def test_missed_session_pattern_from_episode_text(self) -> None:
        episodes = _make_episodes_with_thursday_skip()
        trajectory = {"trajectory": {}, "confidence": 0.3, "goal": {}}
        triggers = check_proactive_triggers(_make_profile(), [], episodes, trajectory)
        types = [t["type"] for t in triggers]
        assert "missed_session_pattern" in types

    def test_missed_session_pattern_data_contains_day(self) -> None:
        episodes = _make_episodes_with_thursday_skip()
        trajectory = {"trajectory": {}, "confidence": 0.3, "goal": {}}
        triggers = check_proactive_triggers(_make_profile(), [], episodes, trajectory)
        pattern = next(t for t in triggers if t["type"] == "missed_session_pattern")
        assert pattern["data"]["day"] == "Thursday"

    def test_fitness_improving_trigger_on_increasing_trend(self) -> None:
        episodes = _make_improving_episodes()
        trajectory = {"trajectory": {}, "confidence": 0.3, "goal": {}}
        triggers = check_proactive_triggers(_make_profile(), [], episodes, trajectory)
        types = [t["type"] for t in triggers]
        assert "fitness_improving" in types

    def test_milestone_approaching_trigger(self) -> None:
        trajectory = {
            "trajectory": {
                "on_track": True,
                "key_milestones": [
                    {"milestone": "10K time trial", "date": "2026-04-01", "status": "on_track"},
                ],
            },
            "confidence": 0.6,
        }
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        types = [t["type"] for t in triggers]
        assert "milestone_approaching" in types

    def test_no_triggers_with_empty_neutral_data(self) -> None:
        trajectory = {"trajectory": {}, "confidence": 0.2, "goal": {}}
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        # No strong signal → minimal or empty trigger list
        types = [t["type"] for t in triggers]
        assert "goal_at_risk" not in types
        assert "on_track" not in types

    def test_trigger_structure_has_required_fields(self) -> None:
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "2:00:00"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(_make_profile(), [], [], trajectory)
        for trigger in triggers:
            assert "type" in trigger
            assert "priority" in trigger
            assert "data" in trigger


# ---------------------------------------------------------------------------
# format_proactive_message — pure formatting tests
# ---------------------------------------------------------------------------


class TestFormatProactiveMessage:
    def test_on_track_contains_predicted_time_and_confidence(self) -> None:
        trigger = {
            "type": "on_track",
            "data": {"predicted_time": "1:43-1:48", "confidence": 0.65},
        }
        msg = format_proactive_message(trigger, _make_profile())
        assert "1:43-1:48" in msg
        assert "65%" in msg

    def test_goal_at_risk_contains_both_times(self) -> None:
        trigger = {
            "type": "goal_at_risk",
            "data": {"predicted_time": "1:55-2:05", "target_time": "1:45:00"},
        }
        msg = format_proactive_message(trigger, _make_profile())
        assert "1:55-2:05" in msg
        assert "1:45:00" in msg

    def test_missed_session_contains_day(self) -> None:
        trigger = {
            "type": "missed_session_pattern",
            "data": {"day": "Thursday", "missed_count": 3},
        }
        msg = format_proactive_message(trigger, _make_profile())
        assert "Thursday" in msg

    def test_fatigue_warning_mentions_fatigue(self) -> None:
        trigger = {
            "type": "fatigue_warning",
            "data": {"message": "Fatigue detected"},
        }
        msg = format_proactive_message(trigger, _make_profile())
        assert "fatigue" in msg.lower()

    def test_fitness_improving_mentions_trend(self) -> None:
        trigger = {"type": "fitness_improving", "data": {"trend": "improving"}}
        msg = format_proactive_message(trigger, _make_profile())
        assert len(msg) > 20  # not empty/trivial

    def test_milestone_approaching_contains_milestone_name(self) -> None:
        trigger = {
            "type": "milestone_approaching",
            "data": {
                "milestone": {
                    "milestone": "First 10K",
                    "date": "2026-04-01",
                    "status": "on_track",
                }
            },
        }
        msg = format_proactive_message(trigger, _make_profile())
        assert "First 10K" in msg

    def test_unknown_type_returns_fallback_string(self) -> None:
        trigger = {"type": "alien_trigger", "data": {"foo": "bar"}}
        msg = format_proactive_message(trigger, _make_profile())
        assert "alien_trigger" in msg

    def test_messages_are_non_empty_strings(self) -> None:
        trigger_types = [
            {"type": "on_track", "data": {"predicted_time": "1:43:00", "confidence": 0.7}},
            {"type": "goal_at_risk", "data": {"predicted_time": "2:00:00", "target_time": "1:45:00"}},
            {"type": "missed_session_pattern", "data": {"day": "Monday", "missed_count": 2}},
            {"type": "fitness_improving", "data": {}},
            {"type": "fatigue_warning", "data": {}},
        ]
        for trigger in trigger_types:
            msg = format_proactive_message(trigger, {})
            assert isinstance(msg, str)
            assert len(msg) > 10, f"Message too short for trigger {trigger['type']!r}: {msg!r}"


# ---------------------------------------------------------------------------
# calculate_silence_decay — pure computation tests
# ---------------------------------------------------------------------------


class TestCalculateSilenceDecay:
    def _ts(self, days_ago: float) -> str:
        return (datetime.now() - timedelta(days=days_ago)).isoformat()

    def test_no_last_interaction_returns_moderate_boost(self) -> None:
        result = calculate_silence_decay(None)
        assert result == 0.5

    def test_active_within_one_day_returns_zero(self) -> None:
        result = calculate_silence_decay(self._ts(0.5))
        assert result == 0.0

    def test_one_to_three_days_returns_small_boost(self) -> None:
        result = calculate_silence_decay(self._ts(2.0))
        assert result == 0.1

    def test_three_to_five_days_returns_medium_boost(self) -> None:
        result = calculate_silence_decay(self._ts(4.0))
        assert result == 0.3

    def test_five_to_ten_days_returns_large_boost(self) -> None:
        result = calculate_silence_decay(self._ts(7.0))
        assert result == 0.5

    def test_more_than_ten_days_returns_maximum_boost(self) -> None:
        result = calculate_silence_decay(self._ts(15.0))
        assert result == 0.7

    def test_urgency_is_monotonically_nondecreasing_with_silence(self) -> None:
        values = [
            calculate_silence_decay(self._ts(d))
            for d in [0.2, 1.5, 4.0, 7.0, 12.0]
        ]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Urgency decreased from day-gap index {i} to {i+1}: {values}"
            )


# ---------------------------------------------------------------------------
# check_conversation_triggers — trigger detection tests
# ---------------------------------------------------------------------------


class TestCheckConversationTriggers:
    def _ts(self, days_ago: float) -> str:
        return (datetime.now() - timedelta(days=days_ago)).isoformat()

    def test_silence_below_five_days_produces_no_triggers(self) -> None:
        triggers = check_conversation_triggers({}, last_interaction=self._ts(3.0))
        types = [t["type"] for t in triggers]
        assert "silence_checkin" not in types

    def test_silence_five_plus_days_produces_checkin_trigger(self) -> None:
        triggers = check_conversation_triggers({}, last_interaction=self._ts(6.0))
        types = [t["type"] for t in triggers]
        assert "silence_checkin" in types

    def test_checkin_trigger_contains_days_since_last_chat(self) -> None:
        triggers = check_conversation_triggers({}, last_interaction=self._ts(8.0))
        checkin = next(t for t in triggers if t["type"] == "silence_checkin")
        assert checkin["data"]["days_since_last_chat"] >= 7.9

    def test_no_last_interaction_produces_no_triggers(self) -> None:
        # When last_interaction is None the function skips the silence check
        triggers = check_conversation_triggers({}, last_interaction=None)
        assert triggers == []

    def test_checkin_urgency_field_is_present(self) -> None:
        triggers = check_conversation_triggers({}, last_interaction=self._ts(10.0))
        checkin = next(t for t in triggers if t["type"] == "silence_checkin")
        assert "urgency" in checkin
        assert isinstance(checkin["urgency"], float)
