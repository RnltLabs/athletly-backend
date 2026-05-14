"""Tests for src/agent/tools/journal_tools.py.

Focus of this file: the annotate_activity defensive-validation contract
added in Iter 2 Sprint I. The agent had been observed synthesising
slug-like ids ('today-easy-run-0515') when forced to annotate without a
prior get_activities lookup. The handler now rejects non-UUID ids
BEFORE touching Supabase and returns a recovery hint.

Other journal tools (update_journal_section, append_to_journal,
remove_from_journal) are covered by integration tests; this file pins
only the validation contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent.tools.journal_tools import _is_uuid, register_journal_tools
from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# _is_uuid: the underlying format check
# ---------------------------------------------------------------------------


def test_is_uuid_accepts_canonical_uuid() -> None:
    assert _is_uuid("4f3c1b2a-90de-4abc-9123-fedcba987654") is True


def test_is_uuid_accepts_uppercase_uuid() -> None:
    assert _is_uuid("4F3C1B2A-90DE-4ABC-9123-FEDCBA987654") is True


def test_is_uuid_strips_whitespace() -> None:
    assert _is_uuid("  4f3c1b2a-90de-4abc-9123-fedcba987654  ") is True


def test_is_uuid_rejects_slug() -> None:
    # The Iter 1 confabulation pattern.
    assert _is_uuid("today-easy-run-0515") is False
    assert _is_uuid("last-run") is False
    assert _is_uuid("morning-ride-2026-05-14") is False


def test_is_uuid_rejects_partial_uuid() -> None:
    # Missing trailing block.
    assert _is_uuid("4f3c1b2a-90de-4abc-9123") is False


def test_is_uuid_rejects_non_string() -> None:
    assert _is_uuid(None) is False  # type: ignore[arg-type]
    assert _is_uuid(123) is False  # type: ignore[arg-type]
    assert _is_uuid(["uuid"]) is False  # type: ignore[arg-type]


def test_is_uuid_rejects_empty_string() -> None:
    assert _is_uuid("") is False


# ---------------------------------------------------------------------------
# annotate_activity handler validation
# ---------------------------------------------------------------------------


def _build_registry_with_mock_user() -> tuple[ToolRegistry, MagicMock]:
    registry = ToolRegistry()
    user_model = MagicMock()
    user_model.user_id = "9d77f53a-0001-4000-8000-000000000001"
    register_journal_tools(registry, user_model)
    return registry, user_model


def test_annotate_activity_rejects_synthetic_slug_id() -> None:
    """The Elena/Lisa Iter 1 confabulation: slug id must be rejected
    with a recovery hint that points the agent at get_activities."""
    registry, _ = _build_registry_with_mock_user()
    tool = registry._tools.get("annotate_activity")
    assert tool is not None

    result = tool.handler(
        activity_id="today-easy-run-0515",
        note="Knie zwickt nach 5km",
    )
    assert "error" in result
    assert "UUID" in result["error"]
    assert "synthetic" in result["error"].lower()
    assert "hint" in result
    # The hint must steer the agent to the lookup path.
    assert "get_activities" in result["hint"]


def test_annotate_activity_rejects_empty_note() -> None:
    """Empty note is a no-op write; reject with a hint."""
    registry, _ = _build_registry_with_mock_user()
    tool = registry._tools.get("annotate_activity")
    assert tool is not None

    result = tool.handler(
        activity_id="4f3c1b2a-90de-4abc-9123-fedcba987654",
        note="   ",
    )
    assert "error" in result
    assert "note" in result["error"].lower()


def test_annotate_activity_accepts_real_uuid() -> None:
    """With a real UUID, the handler proceeds to Supabase update."""
    registry, _ = _build_registry_with_mock_user()
    tool = registry._tools.get("annotate_activity")
    assert tool is not None

    fake_client = MagicMock()
    fake_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

    with patch("src.db.client.get_supabase", return_value=fake_client):
        result = tool.handler(
            activity_id="4f3c1b2a-90de-4abc-9123-fedcba987654",
            note="Knie zwickt nach 5km",
        )

    assert result.get("status") == "ok"
    assert result.get("activity_id") == "4f3c1b2a-90de-4abc-9123-fedcba987654"
    # Defensive: the underlying client must have been called once.
    fake_client.table.assert_called_with("activities")


def test_annotate_activity_description_requires_lookup_first() -> None:
    """The tool description must contain the directive 'REQUIRED FIRST
    STEP' so the agent sees the precondition every turn."""
    registry, _ = _build_registry_with_mock_user()
    tool = registry._tools.get("annotate_activity")
    assert tool is not None
    desc = tool.description
    assert "REQUIRED FIRST STEP" in desc
    assert "get_activities" in desc
    # And the negative example so the agent recognises its own failure.
    assert "synthetic" in desc.lower() or "today-easy-run" in desc
