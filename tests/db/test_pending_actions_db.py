"""Unit tests for src.db.pending_actions_db.

Covers all public functions:
- create_pending_action
- get_pending_for_user
- get_recently_resolved
- resolve_pending_action
- expire_stale_actions

All Supabase I/O is mocked via patch("src.db.pending_actions_db.get_supabase").
Each test is fully independent -- no shared mutable state.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.db.pending_actions_db import (
    create_pending_action,
    expire_stale_actions,
    get_pending_for_user,
    get_recently_resolved,
    resolve_pending_action,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-checkpoint-test-uuid"
ACTION_ID = "action-uuid-0001"


# ---------------------------------------------------------------------------
# Mock builder (same pattern as test_proactive_queue_db)
# ---------------------------------------------------------------------------


def _make_chain(rows: list[dict]) -> MagicMock:
    """Return a mock query chain whose .execute().data == *rows*.

    All intermediate query-builder methods return the same chain object so
    any call order (eq, upsert, update, lt, gte, order, in_, ...) works.
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
    """Return a mock Supabase client whose .table() always returns *rows*."""
    client = MagicMock()
    client.table.return_value = _make_chain(rows)
    return client


# ---------------------------------------------------------------------------
# create_pending_action
# ---------------------------------------------------------------------------


class TestCreatePendingAction:
    """Tests for create_pending_action()."""

    def test_returns_inserted_row(self) -> None:
        inserted_row = {
            "id": ACTION_ID,
            "user_id": USER_ID,
            "action_type": "plan_restructure",
            "description": "Restructure the weekly plan",
            "preview": {},
            "checkpoint_type": "HARD",
            "status": "pending",
        }
        client = _mock_supabase([inserted_row])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = create_pending_action(
                USER_ID, "plan_restructure", "Restructure the weekly plan", {},
            )
        assert result["id"] == ACTION_ID
        assert result["action_type"] == "plan_restructure"

    def test_returned_dict_is_new_object(self) -> None:
        """create_pending_action must return a new dict, not the raw result."""
        raw_row = {"id": ACTION_ID, "user_id": USER_ID, "status": "pending"}
        client = _mock_supabase([raw_row])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = create_pending_action(USER_ID, "test", "desc", {})
        assert isinstance(result, dict)
        result["status"] = "MUTATED"
        client2 = _mock_supabase([raw_row])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client2):
            result2 = create_pending_action(USER_ID, "test", "desc", {})
        assert result2["status"] != "MUTATED"

    def test_upsert_conflict_key(self) -> None:
        """on_conflict must be 'user_id,action_type' per DB partial unique index."""
        inserted_row = {"id": "x"}
        client = _mock_supabase([inserted_row])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            create_pending_action(USER_ID, "schedule_swap", "Swap Mon/Wed", {})
        upsert_call = client.table.return_value.upsert
        on_conflict = (
            upsert_call.call_args[1].get("on_conflict")
            or upsert_call.call_args[0][1]
        )
        assert on_conflict == "user_id,action_type"

    def test_all_fields_in_upsert_payload(self) -> None:
        """The row passed to upsert must contain all required fields."""
        client = _mock_supabase([{"id": "x"}])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            create_pending_action(
                USER_ID, "intensity_change", "Reduce volume 15%",
                {"before": "50km", "after": "42km"}, "SOFT",
            )
        upsert_call = client.table.return_value.upsert
        row: dict = upsert_call.call_args[0][0]
        assert row["status"] == "pending"
        assert row["user_id"] == USER_ID
        assert row["action_type"] == "intensity_change"
        assert row["description"] == "Reduce volume 15%"
        assert row["checkpoint_type"] == "SOFT"

    def test_session_id_included_when_provided(self) -> None:
        client = _mock_supabase([{"id": "x"}])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            create_pending_action(
                USER_ID, "test", "desc", {}, session_id="sess-123",
            )
        upsert_call = client.table.return_value.upsert
        row: dict = upsert_call.call_args[0][0]
        assert row["session_id"] == "sess-123"

    def test_session_id_omitted_when_none(self) -> None:
        client = _mock_supabase([{"id": "x"}])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            create_pending_action(USER_ID, "test", "desc", {})
        upsert_call = client.table.return_value.upsert
        row: dict = upsert_call.call_args[0][0]
        assert "session_id" not in row


# ---------------------------------------------------------------------------
# get_pending_for_user
# ---------------------------------------------------------------------------


class TestGetPendingForUser:
    """Tests for get_pending_for_user()."""

    def test_returns_list_of_pending_rows(self) -> None:
        pending = [
            {"id": "a1", "user_id": USER_ID, "status": "pending"},
            {"id": "a2", "user_id": USER_ID, "status": "pending"},
        ]
        client = _mock_supabase(pending)
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = get_pending_for_user(USER_ID)
        assert len(result) == 2

    def test_returns_empty_list_when_no_pending(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = get_pending_for_user(USER_ID)
        assert result == []

    def test_filters_by_status_pending(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            get_pending_for_user(USER_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("status", "pending") in eq_args

    def test_ordered_by_created_at_desc(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            get_pending_for_user(USER_ID)
        order_calls = client.table.return_value.order.call_args_list
        order_args = [c[0] for c in order_calls]
        assert any("created_at" in str(args) for args in order_args)


# ---------------------------------------------------------------------------
# resolve_pending_action
# ---------------------------------------------------------------------------


class TestResolvePendingAction:
    """Tests for resolve_pending_action()."""

    def test_confirm_sets_status_confirmed(self) -> None:
        updated = {"id": ACTION_ID, "status": "confirmed"}
        client = _mock_supabase([updated])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = resolve_pending_action(USER_ID, ACTION_ID, confirmed=True)
        assert result is not None
        assert result["status"] == "confirmed"
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["status"] == "confirmed"

    def test_reject_sets_status_rejected(self) -> None:
        updated = {"id": ACTION_ID, "status": "rejected"}
        client = _mock_supabase([updated])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = resolve_pending_action(USER_ID, ACTION_ID, confirmed=False)
        assert result is not None
        assert result["status"] == "rejected"
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["status"] == "rejected"

    def test_returns_none_when_not_found(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = resolve_pending_action(USER_ID, "nonexistent", confirmed=True)
        assert result is None

    def test_stamps_resolved_at(self) -> None:
        client = _mock_supabase([{"id": ACTION_ID}])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            resolve_pending_action(USER_ID, ACTION_ID, confirmed=True)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert "resolved_at" in payload
        datetime.fromisoformat(payload["resolved_at"])

    def test_scoped_to_user_and_pending_status(self) -> None:
        client = _mock_supabase([{"id": ACTION_ID}])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            resolve_pending_action(USER_ID, ACTION_ID, confirmed=True)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("id", ACTION_ID) in eq_args
        assert ("user_id", USER_ID) in eq_args
        assert ("status", "pending") in eq_args


# ---------------------------------------------------------------------------
# expire_stale_actions
# ---------------------------------------------------------------------------


class TestExpireStaleActions:
    """Tests for expire_stale_actions()."""

    def test_returns_list_of_expired_rows(self) -> None:
        expired_rows = [
            {"id": "old-1", "status": "expired"},
            {"id": "old-2", "status": "expired"},
        ]
        client = _mock_supabase(expired_rows)
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = expire_stale_actions(USER_ID)
        assert len(result) == 2

    def test_returns_empty_list_when_nothing_expired(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = expire_stale_actions(USER_ID)
        assert result == []

    def test_update_sets_status_expired(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            expire_stale_actions(USER_ID)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert payload["status"] == "expired"

    def test_stamps_resolved_at(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            expire_stale_actions(USER_ID)
        update_call = client.table.return_value.update
        payload: dict = update_call.call_args[0][0]
        assert "resolved_at" in payload
        datetime.fromisoformat(payload["resolved_at"])

    def test_filters_only_pending_status(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            expire_stale_actions(USER_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("status", "pending") in eq_args

    def test_cutoff_applied_via_lt(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            expire_stale_actions(USER_ID, max_age_hours=24)
        lt_calls = client.table.return_value.lt.call_args_list
        lt_args = [c[0] for c in lt_calls]
        assert any("created_at" in str(args) for args in lt_args)

    def test_cutoff_is_valid_iso_datetime(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            expire_stale_actions(USER_ID, max_age_hours=12)
        lt_calls = client.table.return_value.lt.call_args_list
        cutoff_value = lt_calls[0][0][1]
        datetime.fromisoformat(cutoff_value)

    def test_scoped_to_user_id(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            expire_stale_actions(USER_ID)
        eq_calls = client.table.return_value.eq.call_args_list
        eq_args = [c[0] for c in eq_calls]
        assert ("user_id", USER_ID) in eq_args


# ---------------------------------------------------------------------------
# get_recently_resolved
# ---------------------------------------------------------------------------


class TestGetRecentlyResolved:
    """Tests for get_recently_resolved()."""

    def test_returns_resolved_rows(self) -> None:
        resolved = [
            {"id": "r1", "status": "confirmed", "resolved_at": "2026-03-04T10:00:00"},
            {"id": "r2", "status": "rejected", "resolved_at": "2026-03-04T09:30:00"},
        ]
        client = _mock_supabase(resolved)
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = get_recently_resolved(USER_ID, hours=1)
        assert len(result) == 2

    def test_returns_empty_list_when_none_resolved(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = get_recently_resolved(USER_ID)
        assert result == []

    def test_filters_by_confirmed_and_rejected_status(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            get_recently_resolved(USER_ID)
        in_calls = client.table.return_value.in_.call_args_list
        assert len(in_calls) >= 1
        statuses = in_calls[0][0][1]
        assert "confirmed" in statuses
        assert "rejected" in statuses

    def test_cutoff_applied_via_gte(self) -> None:
        client = _mock_supabase([])
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            get_recently_resolved(USER_ID, hours=2)
        gte_calls = client.table.return_value.gte.call_args_list
        assert len(gte_calls) >= 1
        gte_args = [c[0] for c in gte_calls]
        assert any("resolved_at" in str(args) for args in gte_args)

    def test_result_is_new_list(self) -> None:
        rows = [{"id": "r3", "status": "confirmed"}]
        client = _mock_supabase(rows)
        with patch("src.db.pending_actions_db.get_supabase", return_value=client):
            result = get_recently_resolved(USER_ID)
        assert isinstance(result, list)
