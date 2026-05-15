"""Snapshot test for the ``plan_preview`` UI component shape.

The mobile app's renderUIComponent registry expects this component to
keep the same ``type`` and the same ``props`` keys regardless of which
backend tool emits it. If this test starts failing, the React Native
plan card will silently render an empty state. Update the snapshot
ONLY when you have also shipped a frontend change.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.agent.tools.registry import ToolRegistry
from src.agent.tools.session_tools import register_session_tools


@pytest.fixture
def session_tools(tmp_path, monkeypatch):
    """Register session_tools against a hermetic file store."""
    monkeypatch.chdir(tmp_path)
    fake_settings = type("S", (), {
        "use_supabase": False,
        "agenticsports_user_id": "snapshot-user",
    })()
    monkeypatch.setattr(
        "src.agent.tools.session_tools.get_settings",
        lambda: fake_settings,
    )
    registry = ToolRegistry()
    register_session_tools(registry, user_id="snapshot-user")
    return registry._tools


def _today_plus(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


# The fields the frontend renderUIComponent registry depends on. If any
# of these go missing, the plan card breaks for every user. New keys
# can be ADDED without breaking compat; removal is the failure mode.
_REQUIRED_TOP_LEVEL_KEYS = {"type", "id", "props"}
_REQUIRED_PROP_KEYS = {"start_date", "end_date", "sessions", "truncated_in_ui"}
_REQUIRED_SESSION_KEYS = {"date"}
# Optional but coach-typically-populated session keys we keep visible
# in the snapshot so a silent removal still fails this test.
_EXPECTED_SESSION_KEYS_WHEN_PRESENT = {
    "sport",
    "name",
    "description",
    "duration_minutes",
    "intensity",
    "status",
    "day",
}


def test_plan_preview_top_level_shape_is_stable(session_tools) -> None:
    """The component envelope (type / id / props) is the public contract."""
    result = session_tools["propose_sessions"].handler(
        start_date=_today_plus(0),
        end_date=_today_plus(6),
        sessions=[
            {
                "date": _today_plus(0),
                "modality": "run",
                "payload": {
                    "name": "Easy",
                    "duration_minutes": 60,
                    "intensity": "low",
                    "description": "Z2 60min",
                },
                "notes": "First long Z2 of the block",
            },
        ],
    )
    ui = result["_ui_component"]
    assert set(ui.keys()) >= _REQUIRED_TOP_LEVEL_KEYS
    assert ui["type"] == "plan_preview", (
        "Component type changed - this breaks renderUIComponent registry."
    )


def test_plan_preview_props_shape_is_stable(session_tools) -> None:
    result = session_tools["propose_sessions"].handler(
        start_date=_today_plus(0),
        end_date=_today_plus(6),
        sessions=[{"date": _today_plus(0), "modality": "run"}],
    )
    props = result["_ui_component"]["props"]
    missing = _REQUIRED_PROP_KEYS - set(props.keys())
    assert not missing, f"plan_preview props missing required keys: {missing}"


def test_plan_preview_session_shape_keeps_legacy_keys(session_tools) -> None:
    """``sport`` (legacy) maps from ``modality``; ``day`` is derived from date."""
    today_plus_0 = _today_plus(0)
    result = session_tools["propose_sessions"].handler(
        start_date=today_plus_0,
        end_date=_today_plus(0),
        sessions=[{
            "date": today_plus_0,
            "modality": "run",
            "payload": {
                "name": "Easy",
                "duration_minutes": 45,
                "intensity": "low",
                "description": "Z2",
            },
        }],
    )
    sessions_props = result["_ui_component"]["props"]["sessions"]
    assert len(sessions_props) == 1
    sess = sessions_props[0]

    # Required keys
    missing = _REQUIRED_SESSION_KEYS - set(sess.keys())
    assert not missing, f"session-shape required keys missing: {missing}"

    # ``sport`` is the historical key the React Native app reads.
    assert sess.get("sport") == "run"
    # Date is verbatim.
    assert sess["date"] == today_plus_0
    # Derived day-of-week is present.
    assert sess["day"] in {
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    }
    # Optional keys, when present, must use these names.
    optional_present = set(sess.keys()) & _EXPECTED_SESSION_KEYS_WHEN_PRESENT
    assert optional_present, "expected at least one optional key on a populated session"


def test_plan_preview_truncates_long_window(session_tools, monkeypatch) -> None:
    """Long windows get the truncated flag so the UI knows to render a hint."""
    # Force a low max so we don't need 31 fixtures to exercise the path.
    monkeypatch.setattr(
        "src.agent.tools.session_tools._MAX_INLINE_PREVIEW_SESSIONS", 3
    )
    sessions = [
        {"date": _today_plus(i), "modality": "run"} for i in range(5)
    ]
    result = session_tools["propose_sessions"].handler(
        start_date=_today_plus(0),
        end_date=_today_plus(4),
        sessions=sessions,
    )
    props = result["_ui_component"]["props"]
    assert props["truncated_in_ui"] is True
    assert len(props["sessions"]) == 3
