"""Tests for src/agent/tools/journal_state_tools.py.

Covers the atomic CRUD contract for intents, programs, and coach_notes:

* add / update / retire flows on intents
* add / update / deactivate flows on programs
* append-only insertion + list on coach notes

All Supabase I/O is mocked at the DB-module boundary (``src.db.intents_db``,
``src.db.programs_db``, ``src.db.coach_notes_db``) so the tests stay
hermetic. Each test asserts both the return shape AND the underlying
DB call so behaviour is pinned end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.journal_state_tools import (
    _classify_retirement,
    _is_uuid,
    register_journal_state_tools,
)
from src.agent.tools.registry import ToolRegistry


USER_ID = "9d77f53a-0001-4000-8000-000000000001"
INTENT_ID = "11111111-1111-4111-8111-111111111111"
PROGRAM_ID = "22222222-2222-4222-8222-222222222222"
NOTE_ID = "33333333-3333-4333-8333-333333333333"
SESSION_ID = "44444444-4444-4444-8444-444444444444"


def _build_registry() -> tuple[ToolRegistry, MagicMock]:
    """Return a fresh registry with all journal-state tools registered."""
    registry = ToolRegistry()
    user_model = MagicMock()
    user_model.user_id = USER_ID
    register_journal_state_tools(registry, user_model)
    return registry, user_model


# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------


class TestClassifyRetirement:
    """Heuristic that picks 'achieved' vs 'abandoned'."""

    def test_achieved_keyword_routes_to_achieved(self) -> None:
        assert _classify_retirement("done with this") == "achieved"

    def test_german_keyword_routes_to_achieved(self) -> None:
        assert _classify_retirement("geschafft") == "achieved"
        assert _classify_retirement("erreicht") == "achieved"

    def test_empty_routes_to_abandoned(self) -> None:
        assert _classify_retirement("") == "abandoned"

    def test_unknown_routes_to_abandoned(self) -> None:
        assert _classify_retirement("pivoting to a different race") == "abandoned"


class TestIsUuid:
    """Reuses the same UUID guard as journal_tools."""

    def test_canonical_uuid_accepted(self) -> None:
        assert _is_uuid(INTENT_ID) is True

    def test_slug_rejected(self) -> None:
        assert _is_uuid("intent-event-karlsruhe") is False

    def test_none_rejected(self) -> None:
        assert _is_uuid(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# add_intent
# ---------------------------------------------------------------------------


class TestAddIntent:
    def test_inserts_and_returns_row(self) -> None:
        registry, _ = _build_registry()
        inserted = {
            "id": INTENT_ID,
            "user_id": USER_ID,
            "type": "event",
            "title": "Karlsruhe HM sub 1:45",
            "priority": 1,
            "status": "active",
        }
        with patch(
            "src.db.intents_db.insert_intent", return_value=inserted
        ) as mock_insert:
            result = registry.execute(
                "add_intent",
                {
                    "type": "event",
                    "title": "Karlsruhe HM sub 1:45",
                    "priority": 1,
                    "payload": {"race_date": "2026-04-12"},
                },
            )
        assert result["status"] == "ok"
        assert result["intent"]["id"] == INTENT_ID
        mock_insert.assert_called_once()
        kwargs = mock_insert.call_args.kwargs
        assert kwargs["title"] == "Karlsruhe HM sub 1:45"
        assert kwargs["type"] == "event"
        assert kwargs["priority"] == 1
        assert kwargs["payload"] == {"race_date": "2026-04-12"}

    def test_empty_title_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "add_intent",
            {"type": "event", "title": "   "},
        )
        assert "error" in result
        assert "hint" in result

    def test_db_exception_returns_error(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.intents_db.insert_intent",
            side_effect=RuntimeError("db down"),
        ):
            result = registry.execute(
                "add_intent",
                {"type": "habit", "title": "4x/Woche bewegen"},
            )
        assert "error" in result
        assert "db down" in result["error"]


# ---------------------------------------------------------------------------
# retire_intent
# ---------------------------------------------------------------------------


class TestRetireIntent:
    def test_routes_to_achieved_when_reason_contains_keyword(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.intents_db.update_intent_status",
            return_value={"id": INTENT_ID, "status": "achieved"},
        ) as mock_update:
            result = registry.execute(
                "retire_intent",
                {"intent_id": INTENT_ID, "reason": "Finished the race"},
            )
        assert result["status"] == "ok"
        assert result["new_status"] == "achieved"
        mock_update.assert_called_once_with(INTENT_ID, "achieved")

    def test_default_reason_routes_to_abandoned(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.intents_db.update_intent_status",
            return_value={"id": INTENT_ID, "status": "abandoned"},
        ) as mock_update:
            result = registry.execute(
                "retire_intent",
                {"intent_id": INTENT_ID},
            )
        assert result["new_status"] == "abandoned"
        mock_update.assert_called_once_with(INTENT_ID, "abandoned")

    def test_synthetic_id_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "retire_intent",
            {"intent_id": "intent-karlsruhe"},
        )
        assert "error" in result
        assert "UUID" in result["error"]

    def test_not_found_returns_error(self) -> None:
        registry, _ = _build_registry()
        with patch("src.db.intents_db.update_intent_status", return_value=None):
            result = registry.execute(
                "retire_intent",
                {"intent_id": INTENT_ID, "reason": "done"},
            )
        assert "error" in result
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# update_intent
# ---------------------------------------------------------------------------


class TestUpdateIntent:
    def test_priority_change_calls_priority_writer(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.intents_db.update_intent_priority",
            return_value={"id": INTENT_ID, "priority": 5},
        ) as mock_prio:
            result = registry.execute(
                "update_intent",
                {"intent_id": INTENT_ID, "priority": 5},
            )
        assert result["status"] == "ok"
        assert result["intent"]["priority"] == 5
        mock_prio.assert_called_once_with(INTENT_ID, 5)

    def test_payload_change_calls_payload_writer(self) -> None:
        registry, _ = _build_registry()
        new_payload = {"target_time": "1:42:00"}
        with patch(
            "src.db.intents_db.update_intent_payload",
            return_value={"id": INTENT_ID, "payload": new_payload},
        ) as mock_payload:
            result = registry.execute(
                "update_intent",
                {"intent_id": INTENT_ID, "payload": new_payload},
            )
        assert result["status"] == "ok"
        mock_payload.assert_called_once_with(INTENT_ID, new_payload)

    def test_no_change_returns_error(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "update_intent",
            {"intent_id": INTENT_ID},
        )
        assert "error" in result

    def test_synthetic_id_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "update_intent",
            {"intent_id": "intent-x", "priority": 4},
        )
        assert "error" in result
        assert "UUID" in result["error"]


# ---------------------------------------------------------------------------
# list_intents
# ---------------------------------------------------------------------------


class TestListIntents:
    def test_default_filter_is_active(self) -> None:
        registry, _ = _build_registry()
        rows = [{"id": INTENT_ID, "status": "active"}]
        with patch(
            "src.db.intents_db.list_intents", return_value=rows
        ) as mock_list:
            result = registry.execute("list_intents", {})
        assert result["count"] == 1
        assert result["intents"] == rows
        mock_list.assert_called_once_with(USER_ID, status_filter="active")

    def test_all_status_clears_filter(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.intents_db.list_intents", return_value=[]
        ) as mock_list:
            registry.execute("list_intents", {"status": "all"})
        mock_list.assert_called_once_with(USER_ID, status_filter=None)


# ---------------------------------------------------------------------------
# Program tools
# ---------------------------------------------------------------------------


class TestAddProgram:
    def test_inserts_and_returns_row(self) -> None:
        registry, _ = _build_registry()
        inserted = {
            "id": PROGRAM_ID,
            "user_id": USER_ID,
            "kind": "gym_split",
            "label": "Push/Pull/Legs",
            "active": True,
        }
        with patch(
            "src.db.programs_db.insert_program", return_value=inserted
        ) as mock_insert:
            result = registry.execute(
                "add_program",
                {
                    "kind": "gym_split",
                    "label": "Push/Pull/Legs",
                    "payload": {"days_per_week": 3},
                },
            )
        assert result["status"] == "ok"
        assert result["program"]["id"] == PROGRAM_ID
        kwargs = mock_insert.call_args.kwargs
        assert kwargs["kind"] == "gym_split"
        assert kwargs["label"] == "Push/Pull/Legs"
        assert kwargs["payload"] == {"days_per_week": 3}

    def test_empty_kind_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute("add_program", {"kind": "  "})
        assert "error" in result


class TestUpdateProgram:
    def test_label_only_change(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.programs_db.update_program",
            return_value={"id": PROGRAM_ID, "label": "PPL v2"},
        ) as mock_update:
            result = registry.execute(
                "update_program",
                {"program_id": PROGRAM_ID, "label": "PPL v2"},
            )
        assert result["status"] == "ok"
        mock_update.assert_called_once_with(
            PROGRAM_ID, label="PPL v2", payload=None,
        )

    def test_no_change_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "update_program",
            {"program_id": PROGRAM_ID},
        )
        assert "error" in result

    def test_synthetic_id_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "update_program",
            {"program_id": "ppl-split", "label": "x"},
        )
        assert "error" in result
        assert "UUID" in result["error"]


class TestDeactivateProgram:
    def test_flips_active_to_false(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.programs_db.update_program",
            return_value={"id": PROGRAM_ID, "active": False},
        ) as mock_update:
            result = registry.execute(
                "deactivate_program",
                {"program_id": PROGRAM_ID},
            )
        assert result["status"] == "ok"
        mock_update.assert_called_once_with(PROGRAM_ID, active=False)

    def test_synthetic_id_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "deactivate_program",
            {"program_id": "ppl-split"},
        )
        assert "error" in result


class TestListPrograms:
    def test_default_active_only(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.programs_db.list_programs", return_value=[]
        ) as mock_list:
            registry.execute("list_programs", {})
        mock_list.assert_called_once_with(USER_ID, active_only=True)

    def test_include_inactive(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.programs_db.list_programs", return_value=[]
        ) as mock_list:
            registry.execute("list_programs", {"active_only": False})
        mock_list.assert_called_once_with(USER_ID, active_only=False)


# ---------------------------------------------------------------------------
# Coach note tools (append-only)
# ---------------------------------------------------------------------------


class TestAddCoachNote:
    def test_inserts_and_returns_row(self) -> None:
        registry, _ = _build_registry()
        inserted = {
            "id": NOTE_ID,
            "user_id": USER_ID,
            "content": "HRV drops after 3 quality sessions",
            "scope": "training_response",
        }
        with patch(
            "src.db.coach_notes_db.insert_coach_note", return_value=inserted
        ) as mock_insert:
            result = registry.execute(
                "add_coach_note",
                {
                    "content": "HRV drops after 3 quality sessions",
                    "scope": "training_response",
                },
            )
        assert result["status"] == "ok"
        assert result["coach_note"]["id"] == NOTE_ID
        kwargs = mock_insert.call_args.kwargs
        assert kwargs["content"] == "HRV drops after 3 quality sessions"
        assert kwargs["scope"] == "training_response"

    def test_passes_session_id_when_valid(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.coach_notes_db.insert_coach_note",
            return_value={"id": NOTE_ID},
        ) as mock_insert:
            registry.execute(
                "add_coach_note",
                {
                    "content": "good morning long run",
                    "source_session_id": SESSION_ID,
                },
            )
        kwargs = mock_insert.call_args.kwargs
        assert kwargs["source_session_id"] == SESSION_ID

    def test_synthetic_session_id_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "add_coach_note",
            {
                "content": "x",
                "source_session_id": "today-easy-run",
            },
        )
        assert "error" in result
        assert "UUID" in result["error"]

    def test_empty_content_rejected(self) -> None:
        registry, _ = _build_registry()
        result = registry.execute(
            "add_coach_note",
            {"content": "   "},
        )
        assert "error" in result

    def test_no_update_tool_registered(self) -> None:
        """Append-only contract: there must be no update / edit tool."""
        registry, _ = _build_registry()
        names = {t.name for t in registry._tools.values()}
        forbidden = {
            "update_coach_note",
            "edit_coach_note",
            "delete_coach_note",
        }
        assert not (forbidden & names), (
            f"Append-only contract broken: {forbidden & names}"
        )


class TestListCoachNotes:
    def test_default_limit_20(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.coach_notes_db.list_coach_notes", return_value=[]
        ) as mock_list:
            registry.execute("list_coach_notes", {})
        mock_list.assert_called_once_with(
            USER_ID, limit=20, scope_filter=None,
        )

    def test_scope_filter_passed_through(self) -> None:
        registry, _ = _build_registry()
        with patch(
            "src.db.coach_notes_db.list_coach_notes", return_value=[]
        ) as mock_list:
            registry.execute(
                "list_coach_notes",
                {"limit": 50, "scope": "training_response"},
            )
        mock_list.assert_called_once_with(
            USER_ID, limit=50, scope_filter="training_response",
        )


# ---------------------------------------------------------------------------
# Tool registration contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name",
    [
        "add_intent", "retire_intent", "update_intent", "list_intents",
        "add_program", "update_program", "deactivate_program", "list_programs",
        "add_coach_note", "list_coach_notes",
    ],
)
def test_all_tools_registered(tool_name: str) -> None:
    registry, _ = _build_registry()
    assert tool_name in registry._tools, f"{tool_name} missing"


@pytest.mark.parametrize(
    "tool_name",
    [
        "add_intent", "retire_intent", "update_intent", "list_intents",
        "add_program", "update_program", "deactivate_program", "list_programs",
        "add_coach_note", "list_coach_notes",
    ],
)
def test_tool_descriptions_have_no_em_dash(tool_name: str) -> None:
    """Hard constraint: no em-dash or en-dash anywhere in tool descriptions."""
    registry, _ = _build_registry()
    tool = registry._tools[tool_name]
    assert "—" not in tool.description, f"{tool_name} has em-dash"
    assert "–" not in tool.description, f"{tool_name} has en-dash"
