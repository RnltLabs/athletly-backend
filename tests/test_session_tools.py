"""Tests for the 14-day rolling-window session tools.

The tools persist into the new ``training_sessions`` table. To stay
hermetic the tests run with ``use_supabase=False`` so the tools fall
back to the file-backed JSON store. The semantics of both branches
must match: idempotent re-propose, status transitions, athlete_note
persistence, audit trail in ``planned_payload``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from src.agent.tools.registry import ToolRegistry
from src.agent.tools.session_tools import register_session_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Register session_tools against a hermetic file-backed store.

    ``tmp_path`` becomes CWD so the file fallback writes to
    ``tmp_path/data/training_sessions/rows.json`` and does not bleed
    between tests. ``use_supabase`` is forced False so no Supabase
    client is constructed.
    """
    monkeypatch.chdir(tmp_path)

    fake_settings = type("S", (), {
        "use_supabase": False,
        "agenticsports_user_id": "test-user-1",
    })()
    monkeypatch.setattr(
        "src.agent.tools.session_tools.get_settings",
        lambda: fake_settings,
    )

    registry = ToolRegistry()
    register_session_tools(registry, user_id="test-user-1")

    return {
        "propose": registry._tools["propose_sessions"].handler,
        "modify": registry._tools["modify_session"].handler,
        "complete": registry._tools["complete_session"].handler,
        "window": registry._tools["get_session_window"].handler,
        "extend": registry._tools["extend_window"].handler,
        "registry": registry,
    }


def _today_plus(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# propose_sessions
# ---------------------------------------------------------------------------


def test_propose_sessions_persists_rows_and_emits_ui_component(tools) -> None:
    result = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(6),
        sessions=[
            {"date": _today_plus(0), "modality": "run", "payload": {"duration_minutes": 60}},
            {"date": _today_plus(2), "modality": "bike", "notes": "Z2 ride"},
        ],
    )

    assert result["count"] == 2
    assert result["replaced_existing"] == 0
    ui = result["_ui_component"]
    assert ui["type"] == "plan_preview"
    assert len(ui["props"]["sessions"]) == 2
    # Sport is the canonical legacy key for the UI; modality maps to it.
    assert ui["props"]["sessions"][0]["sport"] == "run"


def test_propose_sessions_replaces_existing_planned_rows(tools) -> None:
    """Re-propose semantics: a second call for the same window replaces planned rows."""
    first = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(6),
        sessions=[
            {"date": _today_plus(0), "modality": "run"},
            {"date": _today_plus(2), "modality": "run"},
        ],
    )
    assert first["count"] == 2

    second = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(6),
        sessions=[
            {"date": _today_plus(1), "modality": "swim"},
        ],
    )
    assert second["count"] == 1
    assert second["replaced_existing"] == 2

    window = tools["window"](days_ahead=14, days_back=0)
    assert window["count"] == 1
    assert window["sessions"][0]["modality"] == "swim"


def test_propose_sessions_rejects_session_out_of_window(tools) -> None:
    result = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(3),
        sessions=[{"date": _today_plus(10), "modality": "run"}],
    )
    assert "error" in result
    assert "outside" in result["error"]


def test_propose_sessions_rejects_empty_modality(tools) -> None:
    result = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(3),
        sessions=[{"date": _today_plus(0), "modality": "   "}],
    )
    assert "error" in result
    assert "modality" in result["error"]


def test_propose_sessions_rejects_bad_date_format(tools) -> None:
    result = tools["propose"](
        start_date="not-a-date",
        end_date=_today_plus(3),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    assert "error" in result


def test_propose_sessions_accepts_any_modality_string(tools) -> None:
    """Modality is free text. No enum."""
    result = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(2),
        sessions=[
            {"date": _today_plus(0), "modality": "hyrox"},
            {"date": _today_plus(1), "modality": "skitour"},
            {"date": _today_plus(2), "modality": "kettlebell-flow"},
        ],
    )
    assert result["count"] == 3
    modalities = {s["modality"] for s in result["sessions"]}
    assert modalities == {"hyrox", "skitour", "kettlebell-flow"}


def test_propose_sessions_records_planned_payload_for_audit(tools) -> None:
    """``planned_payload`` is frozen at propose-time for the audit trail."""
    payload = {"duration_minutes": 90, "target_pace": "5:00/km"}
    result = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(1),
        sessions=[{"date": _today_plus(0), "modality": "run", "payload": payload}],
    )
    assert result["count"] == 1
    row = result["sessions"][0]
    assert row["payload"] == payload
    assert row["planned_payload"] == payload


# ---------------------------------------------------------------------------
# modify_session
# ---------------------------------------------------------------------------


def test_modify_session_flips_status_to_modulated_on_substantive_change(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(1),
        sessions=[{"date": _today_plus(0), "modality": "run", "payload": {"duration_minutes": 60}}],
    )
    session_id = proposed["sessions"][0]["id"]

    result = tools["modify"](
        session_id=session_id,
        change={"payload": {"duration_minutes": 45}},
    )
    assert result["modified"] is True
    assert result["substantive"] is True
    assert result["session"]["status"] == "modulated"
    # Audit trail: planned_payload stays as the original prescription.
    assert result["session"]["planned_payload"] == {"duration_minutes": 60}
    assert result["session"]["payload"] == {"duration_minutes": 45}


def test_modify_session_no_op_when_change_matches_existing(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(1),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    session_id = proposed["sessions"][0]["id"]

    result = tools["modify"](
        session_id=session_id,
        change={"modality": "run"},
    )
    assert result["modified"] is False
    # Status remains planned because nothing changed.
    assert result["session"]["status"] == "planned"


def test_modify_session_can_shift_date(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(5),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    session_id = proposed["sessions"][0]["id"]

    result = tools["modify"](
        session_id=session_id,
        change={"date": _today_plus(2)},
    )
    assert result["modified"] is True
    assert result["session"]["date"] == _today_plus(2)
    assert result["session"]["status"] == "modulated"


def test_modify_session_rejects_invalid_date(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(1),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    session_id = proposed["sessions"][0]["id"]
    result = tools["modify"](session_id=session_id, change={"date": "tomorrow"})
    assert "error" in result


def test_modify_session_missing_id_returns_error(tools) -> None:
    result = tools["modify"](
        session_id="00000000-0000-0000-0000-000000000000",
        change={"notes": "x"},
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# complete_session
# ---------------------------------------------------------------------------


def test_complete_session_marks_done_with_athlete_note(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(0),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    session_id = proposed["sessions"][0]["id"]

    result = tools["complete"](
        session_id=session_id,
        athlete_note="Beine schwer, aber lief durch.",
    )
    assert result["completed"] is True
    assert result["session"]["status"] == "completed"
    assert result["session"]["athlete_note"] == "Beine schwer, aber lief durch."


def test_complete_session_supports_skipped_status(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(0),
        sessions=[{"date": _today_plus(0), "modality": "bike"}],
    )
    session_id = proposed["sessions"][0]["id"]

    result = tools["complete"](
        session_id=session_id,
        athlete_note="Wetter zu mies.",
        status="skipped",
    )
    assert result["completed"] is True
    assert result["session"]["status"] == "skipped"


def test_complete_session_rejects_unknown_status(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(0),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    session_id = proposed["sessions"][0]["id"]
    result = tools["complete"](session_id=session_id, status="kind-of-done")
    assert "error" in result


def test_complete_session_default_status_is_completed(tools) -> None:
    proposed = tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(0),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    session_id = proposed["sessions"][0]["id"]
    result = tools["complete"](session_id=session_id)
    assert result["session"]["status"] == "completed"
    assert result["session"]["athlete_note"] == ""


# ---------------------------------------------------------------------------
# get_session_window
# ---------------------------------------------------------------------------


def test_get_session_window_returns_planned_and_completed(tools) -> None:
    tools["propose"](
        start_date=_today_plus(-3),
        end_date=_today_plus(3),
        sessions=[
            {"date": _today_plus(-3), "modality": "run"},
            {"date": _today_plus(0), "modality": "bike"},
            {"date": _today_plus(2), "modality": "swim"},
        ],
    )
    window = tools["window"](days_ahead=7, days_back=7)
    assert window["count"] == 3
    assert window["planned_count"] == 3
    assert window["completed_count"] == 0
    # Sorted chronologically.
    dates = [s["date"] for s in window["sessions"]]
    assert dates == sorted(dates)


def test_get_session_window_clamps_bad_input(tools) -> None:
    """Negative or huge values must not crash the read."""
    window = tools["window"](days_ahead=-5, days_back="oops")
    assert "error" in window


def test_get_session_window_excludes_other_user_rows(tools, monkeypatch) -> None:
    """Hermetic file-backed store: rows from another user_id are invisible."""
    tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(0),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    # Register a second user's tools and confirm the rows are isolated.
    fake_settings = type("S", (), {
        "use_supabase": False,
        "agenticsports_user_id": "test-user-2",
    })()
    monkeypatch.setattr(
        "src.agent.tools.session_tools.get_settings",
        lambda: fake_settings,
    )
    other = ToolRegistry()
    register_session_tools(other, user_id="test-user-2")
    window = other._tools["get_session_window"].handler()
    assert window["count"] == 0


# ---------------------------------------------------------------------------
# extend_window
# ---------------------------------------------------------------------------


def test_extend_window_without_sessions_returns_suggested_range(tools) -> None:
    """Empty call: tool returns the date range it would use, writes nothing.

    The suggested start is the day after the latest existing PLANNED
    row, so to exercise that we plant a row at today+4.
    """
    tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(4),
        sessions=[
            {"date": _today_plus(0), "modality": "run"},
            {"date": _today_plus(4), "modality": "bike"},
        ],
    )
    result = tools["extend"](days=7)
    assert result["extended"] is False
    assert result["suggested_start_date"] == _today_plus(5)
    assert result["suggested_end_date"] == _today_plus(11)


def test_extend_window_with_sessions_appends_chunk(tools) -> None:
    tools["propose"](
        start_date=_today_plus(0),
        end_date=_today_plus(4),
        sessions=[
            {"date": _today_plus(0), "modality": "run"},
            {"date": _today_plus(4), "modality": "bike"},
        ],
    )
    result = tools["extend"](
        days=3,
        sessions=[
            {"date": _today_plus(5), "modality": "bike"},
            {"date": _today_plus(7), "modality": "run"},
        ],
    )
    assert result["extended"] is True
    assert result["count"] == 2

    window = tools["window"](days_ahead=14, days_back=0)
    assert window["count"] == 4


def test_extend_window_starts_at_today_when_no_planned_rows(tools) -> None:
    result = tools["extend"](days=5)
    assert result["extended"] is False
    assert result["suggested_start_date"] == _today_plus(0)
    assert result["suggested_end_date"] == _today_plus(4)
