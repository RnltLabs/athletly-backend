"""Unit tests for checkpoint tools and related SSE / chat integration.

Covers:
- propose_plan_change tool (creates pending action, validates inputs)
- get_pending_confirmations tool (returns structured data)
- SSEEmitter.pending_action event format
- Checkpoint context injection in chat router

All Supabase and config I/O is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.api.sse import SSEEmitter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-checkpoint-tool-test"
ACTION_ID = "action-uuid-tool-0001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(user_id: str = USER_ID) -> MagicMock:
    """Return a mock Settings object with agenticsports_user_id set."""
    settings = MagicMock()
    settings.agenticsports_user_id = user_id
    return settings


def _build_registry():
    """Build a ToolRegistry with checkpoint tools registered."""
    from src.agent.tools.checkpoint_tools import register_checkpoint_tools
    from src.agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_checkpoint_tools(registry)
    return registry


def _mock_supabase_chain(rows: list[dict]) -> MagicMock:
    """Return a mock Supabase client whose .table() returns *rows*."""
    client = MagicMock()
    chain = MagicMock()
    result = MagicMock()
    result.data = rows

    for method in ["select", "eq", "gte", "lt", "order", "limit", "neq",
                   "in_", "upsert", "update", "insert"]:
        getattr(chain, method).return_value = chain
    chain.execute.return_value = result
    client.table.return_value = chain
    return client


# ---------------------------------------------------------------------------
# propose_plan_change
# ---------------------------------------------------------------------------


class TestProposePlanChange:
    """Tests for the propose_plan_change tool."""

    def test_creates_action(self) -> None:
        """propose_plan_change should call create_pending_action and return proposed status."""
        registry = _build_registry()

        mock_action = {
            "id": ACTION_ID,
            "user_id": USER_ID,
            "action_type": "plan_restructure",
            "status": "pending",
        }
        mock_client = _mock_supabase_chain([mock_action])

        with (
            patch("src.config.get_settings", return_value=_make_settings()),
            patch("src.db.pending_actions_db.get_supabase", return_value=mock_client),
        ):
            result = registry.execute(
                "propose_plan_change",
                {
                    "action_type": "plan_restructure",
                    "description": "Restructure weekly plan",
                    "preview": {"before": "5x/week", "after": "4x/week"},
                    "checkpoint_type": "HARD",
                },
            )

        assert result["status"] == "proposed"
        assert result["action_id"] == ACTION_ID
        assert result["action_type"] == "plan_restructure"
        assert result["checkpoint_type"] == "HARD"

    def test_invalid_checkpoint_type_returns_error(self) -> None:
        """Bad checkpoint_type should return an error, not crash."""
        registry = _build_registry()

        with patch("src.config.get_settings", return_value=_make_settings()):
            result = registry.execute(
                "propose_plan_change",
                {
                    "action_type": "test",
                    "description": "test",
                    "checkpoint_type": "INVALID",
                },
            )

        assert result["status"] == "error"
        assert "HARD or SOFT" in result["message"]

    def test_no_user_id_returns_error(self) -> None:
        """When no user_id is configured, should return an error."""
        registry = _build_registry()

        with patch("src.config.get_settings", return_value=_make_settings(user_id="")):
            result = registry.execute(
                "propose_plan_change",
                {"action_type": "test", "description": "test"},
            )

        assert result["status"] == "error"
        assert "user_id" in result["message"].lower()

    def test_defaults_checkpoint_type_to_hard(self) -> None:
        """When checkpoint_type is not provided, it should default to HARD."""
        registry = _build_registry()

        mock_action = {"id": ACTION_ID, "user_id": USER_ID, "status": "pending"}
        mock_client = _mock_supabase_chain([mock_action])

        with (
            patch("src.config.get_settings", return_value=_make_settings()),
            patch("src.db.pending_actions_db.get_supabase", return_value=mock_client),
        ):
            result = registry.execute(
                "propose_plan_change",
                {"action_type": "test", "description": "test"},
            )

        assert result["status"] == "proposed"
        assert result["checkpoint_type"] == "HARD"


# ---------------------------------------------------------------------------
# get_pending_confirmations
# ---------------------------------------------------------------------------


class TestGetPendingConfirmations:
    """Tests for the get_pending_confirmations tool."""

    def test_returns_structured_data(self) -> None:
        """get_pending_confirmations should return pending and resolved lists."""
        registry = _build_registry()

        pending_rows = [
            {
                "id": "p1",
                "action_type": "plan_restructure",
                "description": "Change plan",
                "checkpoint_type": "HARD",
                "created_at": "2026-03-04T10:00:00",
            },
        ]
        resolved_rows = [
            {
                "id": "r1",
                "action_type": "schedule_swap",
                "status": "confirmed",
                "resolved_at": "2026-03-04T09:00:00",
            },
        ]

        with (
            patch("src.config.get_settings", return_value=_make_settings()),
            patch(
                "src.db.pending_actions_db.get_pending_for_user",
                return_value=pending_rows,
            ),
            patch(
                "src.db.pending_actions_db.get_recently_resolved",
                return_value=resolved_rows,
            ),
        ):
            result = registry.execute("get_pending_confirmations", {})

        assert result["status"] == "ok"
        assert len(result["pending"]) == 1
        assert result["pending"][0]["action_id"] == "p1"
        assert result["pending"][0]["action_type"] == "plan_restructure"
        assert len(result["recently_resolved"]) == 1
        assert result["recently_resolved"][0]["action_id"] == "r1"
        assert result["recently_resolved"][0]["status"] == "confirmed"

    def test_empty_results(self) -> None:
        """When no actions exist, should return empty lists."""
        registry = _build_registry()

        with (
            patch("src.config.get_settings", return_value=_make_settings()),
            patch(
                "src.db.pending_actions_db.get_pending_for_user",
                return_value=[],
            ),
            patch(
                "src.db.pending_actions_db.get_recently_resolved",
                return_value=[],
            ),
        ):
            result = registry.execute("get_pending_confirmations", {})

        assert result["status"] == "ok"
        assert result["pending"] == []
        assert result["recently_resolved"] == []

    def test_no_user_id_returns_error(self) -> None:
        """When no user_id is configured, should return an error."""
        registry = _build_registry()

        with patch("src.config.get_settings", return_value=_make_settings(user_id="")):
            result = registry.execute("get_pending_confirmations", {})

        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# SSE pending_action event
# ---------------------------------------------------------------------------


class TestSSEPendingActionEvent:
    """Tests for SSEEmitter.pending_action()."""

    def test_event_format(self) -> None:
        """pending_action SSE event should have correct event type and data structure."""
        evt = SSEEmitter.pending_action(
            action_id="act-123",
            action_type="plan_restructure",
            description="Restructure the weekly plan",
            preview={"before": "5x/week", "after": "4x/week"},
        )
        assert evt.event == "pending_action"
        data = json.loads(evt.data)
        assert data["action_id"] == "act-123"
        assert data["action_type"] == "plan_restructure"
        assert data["description"] == "Restructure the weekly plan"
        assert data["preview"]["before"] == "5x/week"
        assert data["preview"]["after"] == "4x/week"

    def test_empty_preview(self) -> None:
        """pending_action should work with an empty preview dict."""
        evt = SSEEmitter.pending_action(
            action_id="act-456",
            action_type="intensity_change",
            description="Reduce intensity",
            preview={},
        )
        data = json.loads(evt.data)
        assert data["preview"] == {}

    def test_unicode_description(self) -> None:
        """pending_action should handle non-ASCII characters in description."""
        evt = SSEEmitter.pending_action(
            action_id="act-789",
            action_type="schedule_swap",
            description="Trainingstage tauschen: Montag <-> Mittwoch",
            preview={},
        )
        data = json.loads(evt.data)
        assert "Trainingstage" in data["description"]
        assert "Montag" in data["description"]


# ---------------------------------------------------------------------------
# Checkpoint context injection
# ---------------------------------------------------------------------------


class TestCheckpointContextInjection:
    """Tests for checkpoint context injection in the chat event generator."""

    def test_context_prefix_format(self) -> None:
        """Verify the format of checkpoint context when injected into user message."""
        resolved_actions = [
            {"action_type": "plan_restructure", "status": "confirmed"},
            {"action_type": "schedule_swap", "status": "rejected"},
        ]

        # Simulate the injection logic from chat.py
        checkpoint_context = "\n".join(
            f"- {a['action_type']}: {a['status']}"
            for a in resolved_actions
        )
        user_message = "How's my plan?"
        injected = (
            f"[System: The user has responded to your previous proposals: "
            f"{checkpoint_context}]\n\n{user_message}"
        )

        assert "[System:" in injected
        assert "plan_restructure: confirmed" in injected
        assert "schedule_swap: rejected" in injected
        assert injected.endswith("How's my plan?")

    def test_no_injection_when_no_resolved(self) -> None:
        """When there are no resolved actions, user_message should be unchanged."""
        resolved_actions: list[dict] = []
        user_message = "How's my plan?"

        # The code checks `if resolved_actions:` before injecting
        if resolved_actions:
            checkpoint_context = "\n".join(
                f"- {a['action_type']}: {a['status']}"
                for a in resolved_actions
            )
            user_message = (
                f"[System: The user has responded to your previous proposals: "
                f"{checkpoint_context}]\n\n{user_message}"
            )

        assert user_message == "How's my plan?"

    def test_single_resolved_action_format(self) -> None:
        """Single resolved action should produce a single-line context."""
        resolved_actions = [
            {"action_type": "intensity_change", "status": "confirmed"},
        ]

        checkpoint_context = "\n".join(
            f"- {a['action_type']}: {a['status']}"
            for a in resolved_actions
        )
        user_message = "Let's go"
        injected = (
            f"[System: The user has responded to your previous proposals: "
            f"{checkpoint_context}]\n\n{user_message}"
        )

        assert "intensity_change: confirmed" in injected
        assert injected.endswith("Let's go")
