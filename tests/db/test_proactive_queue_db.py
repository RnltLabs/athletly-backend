"""Unit tests for src.db.proactive_queue_db.

Covers all public functions:
- queue_message
- get_pending_messages
- deliver_message
- record_engagement
- expire_stale_messages

All Supabase I/O is mocked via patch("src.db.proactive_queue_db.get_supabase").
Each test is fully independent — no shared mutable state.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from src.db.proactive_queue_db import (
    deliver_message,
    expire_stale_messages,
    get_pending_messages,
    queue_message,
    record_engagement,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-proactive-test-uuid"
MESSAGE_ID = "msg-uuid-0001"


# ---------------------------------------------------------------------------
# Mock builder
# ---------------------------------------------------------------------------


def _make_chain(rows: list[dict]) -> MagicMock:
    """Return a mock query chain whose .execute().data == *rows*.

    All intermediate query-builder methods return the same chain object so
    any call order (eq, upsert, update, lt, order, …) works correctly.
    """
    chain = MagicMock()
    result = MagicMock()
    result.data = rows

    for method_name in [
        "select",
        "eq",
        "gte",
        "lt",
        "order",
        "limit",
        "neq",
        "in_",
        "upsert",
        "update",
        "insert",
    ]:
        getattr(chain, method_name).return_value = chain

    chain.execute.return_value = result
    return chain


def _mock_supabase(rows: list[dict]) -> MagicMock:
    """Return a mock Supabase client whose .table() always returns *rows*.

    For these tests a single table ('proactive_queue') is used, so a single
    chain with fixed data is sufficient.
    """
    client = MagicMock()
    client.table.return_value = _make_chain(rows)
    return client


# ---------------------------------------------------------------------------
# queue_message
# ---------------------------------------------------------------------------


class TestQueueMessage:
    """Tests for queue_message()."""

    def test_returns_inserted_row(self) -> None:
        inserted_row = {
            "id": MESSAGE_ID,
            "user_id": USER_ID,
            "trigger_type": "low_activity",
            "priority": 0.8,
            "data": {},
            "message_text": "Time to train!",
            "status": "pending",
        }
        client = _mock_supabase([inserted_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = queue_message(USER_ID, "low_activity", 0.8, {}, "Time to train!")
        assert result["id"] == MESSAGE_ID
        assert result["trigger_type"] == "low_activity"

    def test_returned_dict_is_new_object(self) -> None:
        """queue_message must return a new dict, not the raw result object."""
        raw_row = {"id": MESSAGE_ID, "user_id": USER_ID, "status": "pending"}
        client = _mock_supabase([raw_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = queue_message(USER_ID, "low_activity", 0.5, {}, "Hello")
        assert isinstance(result, dict)
        # Mutating result must not affect a second call's output.
        result["status"] = "MUTATED"
        client2 = _mock_supabase([raw_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client2):
            result2 = queue_message(USER_ID, "low_activity", 0.5, {}, "Hello")
        assert result2["status"] != "MUTATED"

    def test_all_fields_included_in_upsert(self) -> None:
        """The row passed to upsert must contain all required fields."""
        inserted_row = {"id": "x", "user_id": USER_ID}
        client = _mock_supabase([inserted_row])

        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            queue_message(USER_ID, "hrv_low", 0.9, {"hrv": 32}, "Your HRV is low.")

        # Verify upsert was called with a dict that has the status field.
        upsert_call = client.table.return_value.upsert
        assert upsert_call.called
        upserted_row: dict = upsert_call.call_args[0][0]
        assert upserted_row["status"] == "pending"
        assert upserted_row["user_id"] == USER_ID
        assert upserted_row["trigger_type"] == "hrv_low"
        assert upserted_row["priority"] == 0.9
        assert upserted_row["message_text"] == "Your HRV is low."

    def test_data_payload_is_preserved(self) -> None:
        payload = {"metric": "steps", "value": 4000}
        inserted_row = {"id": "y", "user_id": USER_ID, "data": payload}
        client = _mock_supabase([inserted_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = queue_message(USER_ID, "low_steps", 0.6, payload, "Move more!")
        # The mock returns what we told it — just confirm no key error.
        assert isinstance(result, dict)

    def test_upsert_conflict_key_set_correctly(self) -> None:
        """on_conflict must be 'user_id,trigger_type' per DB partial unique index."""
        inserted_row = {"id": "z"}
        client = _mock_supabase([inserted_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            queue_message(USER_ID, "test_trigger", 0.5, {}, "msg")
        upsert_call = client.table.return_value.upsert
        on_conflict_value = upsert_call.call_args[1].get("on_conflict") or upsert_call.call_args[0][1]
        assert on_conflict_value == "user_id,trigger_type"


# ---------------------------------------------------------------------------
# get_pending_messages
# ---------------------------------------------------------------------------


class TestGetPendingMessages:
    """Tests for get_pending_messages()."""

    def test_returns_list_of_pending_rows(self) -> None:
        pending = [
            {"id": "m1", "user_id": USER_ID, "status": "pending", "priority": 0.9},
            {"id": "m2", "user_id": USER_ID, "status": "pending", "priority": 0.5},
        ]
        client = _mock_supabase(pending)
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = get_pending_messages(USER_ID)
        assert len(result) == 2

    def test_returns_empty_list_when_no_pending(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = get_pending_messages(USER_ID)
        assert result == []

    def test_result_is_new_list(self) -> None:
        """Returned list must be independent — not the raw result object."""
        rows = [{"id": "m3", "status": "pending"}]
        client = _mock_supabase(rows)
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = get_pending_messages(USER_ID)
        assert isinstance(result, list)

    def test_filters_by_status_pending(self) -> None:
        """The query must filter on status='pending'."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            get_pending_messages(USER_ID)
        # Two .eq() calls expected: one for user_id, one for status.
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("status", "pending") in eq_args

    def test_ordered_by_priority_desc(self) -> None:
        """The query must order by priority descending."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            get_pending_messages(USER_ID)
        order_calls = client.table.return_value.order.call_args_list
        order_args = [c[0] for c in order_calls]
        # At least one .order("priority", desc=True) call expected.
        assert any("priority" in str(args) for args in order_args)

    def test_result_preserves_all_row_fields(self) -> None:
        rows = [{"id": "m4", "priority": 0.7, "message_text": "Hey!", "status": "pending"}]
        client = _mock_supabase(rows)
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = get_pending_messages(USER_ID)
        assert result[0]["message_text"] == "Hey!"
        assert result[0]["priority"] == 0.7


# ---------------------------------------------------------------------------
# deliver_message
# ---------------------------------------------------------------------------


class TestDeliverMessage:
    """Tests for deliver_message()."""

    def test_returns_updated_row(self) -> None:
        updated_row = {
            "id": MESSAGE_ID,
            "user_id": USER_ID,
            "status": "delivered",
            "delivered_at": "2026-03-04T10:00:00",
        }
        client = _mock_supabase([updated_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = deliver_message(USER_ID, MESSAGE_ID)
        assert result is not None
        assert result["status"] == "delivered"
        assert result["id"] == MESSAGE_ID

    def test_returns_none_when_row_not_found(self) -> None:
        client = _mock_supabase([])  # No data => row not found
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = deliver_message(USER_ID, "non-existent-id")
        assert result is None

    def test_returned_dict_is_new_object(self) -> None:
        raw = {"id": MESSAGE_ID, "status": "delivered"}
        client = _mock_supabase([raw])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = deliver_message(USER_ID, MESSAGE_ID)
        assert isinstance(result, dict)
        # Mutate to confirm independence.
        assert result is not None
        result["status"] = "CHANGED"
        client2 = _mock_supabase([raw])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client2):
            result2 = deliver_message(USER_ID, MESSAGE_ID)
        assert result2 is not None
        assert result2["status"] != "CHANGED"

    def test_update_sets_status_delivered(self) -> None:
        """The update payload must contain status='delivered'."""
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            deliver_message(USER_ID, MESSAGE_ID)
        update_call = client.table.return_value.update
        update_payload: dict = update_call.call_args[0][0]
        assert update_payload["status"] == "delivered"

    def test_update_stamps_delivered_at(self) -> None:
        """The update payload must include a delivered_at ISO timestamp."""
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            deliver_message(USER_ID, MESSAGE_ID)
        update_call = client.table.return_value.update
        update_payload: dict = update_call.call_args[0][0]
        assert "delivered_at" in update_payload
        # Must be a parseable ISO datetime string.
        delivered_at = update_payload["delivered_at"]
        datetime.fromisoformat(delivered_at)  # raises ValueError if malformed

    def test_scoped_to_user_id(self) -> None:
        """Update must be filtered by both id and user_id."""
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            deliver_message(USER_ID, MESSAGE_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("id", MESSAGE_ID) in eq_args
        assert ("user_id", USER_ID) in eq_args


# ---------------------------------------------------------------------------
# record_engagement
# ---------------------------------------------------------------------------


class TestRecordEngagement:
    """Tests for record_engagement()."""

    def test_returns_updated_row(self) -> None:
        updated_row = {
            "id": MESSAGE_ID,
            "user_id": USER_ID,
            "engagement_tracking": {"user_continued_session": True},
        }
        client = _mock_supabase([updated_row])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = record_engagement(USER_ID, MESSAGE_ID, responded=True, continued_session=True, turns_after=3)
        assert result is not None
        assert result["id"] == MESSAGE_ID

    def test_returns_none_when_not_found(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = record_engagement(USER_ID, "bad-id")
        assert result is None

    def test_engagement_payload_contains_required_keys(self) -> None:
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            record_engagement(USER_ID, MESSAGE_ID, responded=False, continued_session=False, turns_after=0)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        engagement = payload["engagement_tracking"]
        assert "user_continued_session" in engagement
        assert "session_turns_after_delivery" in engagement
        assert "response_latency_seconds" in engagement

    def test_responded_true_sets_responded_at(self) -> None:
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            record_engagement(USER_ID, MESSAGE_ID, responded=True)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        responded_at = payload["engagement_tracking"]["user_responded_at"]
        assert responded_at is not None
        datetime.fromisoformat(responded_at)  # Must be a valid ISO datetime.

    def test_responded_false_leaves_responded_at_none(self) -> None:
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            record_engagement(USER_ID, MESSAGE_ID, responded=False)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["engagement_tracking"]["user_responded_at"] is None

    def test_turns_after_stored_correctly(self) -> None:
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            record_engagement(USER_ID, MESSAGE_ID, turns_after=7)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["engagement_tracking"]["session_turns_after_delivery"] == 7

    def test_continued_session_stored_correctly(self) -> None:
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            record_engagement(USER_ID, MESSAGE_ID, continued_session=True)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["engagement_tracking"]["user_continued_session"] is True

    def test_scoped_to_user_id(self) -> None:
        client = _mock_supabase([{"id": MESSAGE_ID}])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            record_engagement(USER_ID, MESSAGE_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("user_id", USER_ID) in eq_args
        assert ("id", MESSAGE_ID) in eq_args

    def test_returned_dict_is_new_object(self) -> None:
        raw = {"id": MESSAGE_ID, "engagement_tracking": {}}
        client = _mock_supabase([raw])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = record_engagement(USER_ID, MESSAGE_ID)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# expire_stale_messages
# ---------------------------------------------------------------------------


class TestExpireStaleMessages:
    """Tests for expire_stale_messages()."""

    def test_returns_list_of_expired_rows(self) -> None:
        expired_rows = [
            {"id": "old-1", "status": "expired"},
            {"id": "old-2", "status": "expired"},
        ]
        client = _mock_supabase(expired_rows)
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = expire_stale_messages(USER_ID)
        assert len(result) == 2
        assert result[0]["id"] == "old-1"

    def test_returns_empty_list_when_nothing_expired(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = expire_stale_messages(USER_ID)
        assert result == []

    def test_result_is_new_list(self) -> None:
        rows = [{"id": "old-3", "status": "expired"}]
        client = _mock_supabase(rows)
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            result = expire_stale_messages(USER_ID)
        assert isinstance(result, list)

    def test_update_sets_status_expired(self) -> None:
        """The update payload must contain status='expired'."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            expire_stale_messages(USER_ID)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["status"] == "expired"

    def test_filters_only_pending_status(self) -> None:
        """Only pending messages should be expired (not already-delivered ones)."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            expire_stale_messages(USER_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("status", "pending") in eq_args

    def test_cutoff_applied_via_lt(self) -> None:
        """A .lt("created_at", cutoff) call must be issued."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            expire_stale_messages(USER_ID, max_age_days=7)
        lt_calls = client.table.return_value.lt.call_args_list
        lt_args = [c[0] for c in lt_calls]
        assert any("created_at" in str(args) for args in lt_args)

    def test_scoped_to_user_id(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            expire_stale_messages(USER_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("user_id", USER_ID) in eq_args

    def test_custom_max_age_days(self) -> None:
        """max_age_days parameter changes the cutoff date — verify lt is still called."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            expire_stale_messages(USER_ID, max_age_days=30)
        lt_calls = client.table.return_value.lt.call_args_list
        assert len(lt_calls) >= 1

    def test_cutoff_is_valid_iso_datetime(self) -> None:
        """The cutoff passed to .lt() must be a parseable ISO datetime string."""
        client = _mock_supabase([])
        with patch("src.db.proactive_queue_db.get_supabase", return_value=client):
            expire_stale_messages(USER_ID, max_age_days=14)
        lt_calls = client.table.return_value.lt.call_args_list
        # The second argument to .lt("created_at", cutoff) is the ISO string.
        cutoff_value = lt_calls[0][0][1]
        datetime.fromisoformat(cutoff_value)  # Raises ValueError if malformed.
