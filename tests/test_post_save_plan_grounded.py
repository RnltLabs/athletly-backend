"""Tests for the ``post_save_plan_grounded`` response gate (Sprint Q).

Failure case being prevented: in Iter 3 the coach called ``save_plan``
correctly, then the user-facing reply narrated session details that
did not exist in the saved plan ("morgen 8km bei 4:30/km" when the
saved week-1 day-1 was "easy 60min" with no pace).

The gate is a pure function over ``GateContext`` so these tests do not
need the agent loop: we just construct contexts directly.
"""

from __future__ import annotations

import pytest

from src.agent.response_gates import (
    GateContext,
    _gate_post_save_plan_grounded,
    _extract_session_pairs,
    _normalize_distance_token,
    _normalize_pace_token,
)
from src.services.gates_metrics import reset_gates_metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    reset_gates_metrics()
    yield
    reset_gates_metrics()


def _saved_plan_with_session(session: dict) -> dict:
    """Build the shape ``_extract_saved_plan`` would have produced.

    The agent loop parses ``save_plan``'s JSON tool result into a dict
    that carries ``plan_snapshot.sessions``. Tests pass that shape
    directly into the gate's ``GateContext.saved_plan``.
    """
    return {
        "saved": True,
        "id": "plan_test",
        "plan_snapshot": {
            "start_date": "2026-05-18",
            "focus": "Base block",
            "sessions": [session],
            "total_sessions": 1,
        },
    }


def _ctx(
    response_text: str,
    tools_this_turn: tuple[str, ...] = ("save_plan",),
    saved_plan: dict | None = None,
) -> GateContext:
    return GateContext(
        user_message="kannst du mir einen plan machen?",
        response_text=response_text,
        tools_called_this_turn=tools_this_turn,
        tools_called_recent=tools_this_turn,
        runtime_context_alerts=(),
        saved_plan=saved_plan or {},
    )


# ---------------------------------------------------------------------------
# Acceptance scenarios
# ---------------------------------------------------------------------------


def test_response_with_grounded_session_passes():
    """Phase-level summary referencing a real session passes."""
    plan = _saved_plan_with_session({
        "day": "monday",
        "date": "2026-05-18",
        "sport": "running",
        "name": "Easy run",
        "description": "60 min locker",
        "duration_minutes": 60,
        "intensity": "low",
    })
    ctx = _ctx(
        response_text=(
            "Dein Plan ist gespeichert. Woche 1 hat vier easy runs, "
            "darunter ein 60-min Lauf am Montag. Schau dir die Karte "
            "fuer Details an."
        ),
        saved_plan=plan,
    )
    r = _gate_post_save_plan_grounded(ctx)
    assert r.passed, r.reason


def test_response_with_fabricated_pace_fails():
    """The Iter 3 failure case: response invents 8km @ 4:30/km."""
    plan = _saved_plan_with_session({
        "day": "monday",
        "date": "2026-05-18",
        "sport": "running",
        "name": "Easy run",
        "description": "60 min locker",
        "duration_minutes": 60,
        "intensity": "low",
    })
    ctx = _ctx(
        response_text=(
            "Plan gespeichert. Morgen stehen 8km bei 4:30/km an, "
            "fokussiert auf Tempo."
        ),
        saved_plan=plan,
    )
    r = _gate_post_save_plan_grounded(ctx)
    assert not r.passed
    assert r.gate_id == "post_save_plan_grounded"
    assert "8km" in (r.reason or "") or "8 km" in (r.reason or "")
    assert "4:30" in (r.reason or "")


def test_gate_skips_when_save_plan_not_called():
    """If save_plan did not run this turn the gate must not fire."""
    plan = _saved_plan_with_session({
        "day": "monday",
        "name": "Easy run",
        "duration_minutes": 60,
    })
    ctx = _ctx(
        response_text="Morgen stehen 8km bei 4:30/km an.",
        tools_this_turn=("get_activities",),  # no save_plan
        saved_plan=plan,
    )
    r = _gate_post_save_plan_grounded(ctx)
    assert r.passed


def test_german_decimal_handled():
    """``5,30/km`` in the response matches ``5:30/km`` in the plan."""
    plan = _saved_plan_with_session({
        "day": "monday",
        "date": "2026-05-18",
        "sport": "running",
        "name": "Tempo",
        "description": "8km bei 5:30/km, locker einlaufen",
        "duration_minutes": 50,
        "intensity": "moderate",
    })
    # Coach reply uses the German comma form. The gate must accept it
    # because the canonical mm:ss pace is the same value.
    ctx = _ctx(
        response_text=(
            "Plan gespeichert. Diese Woche: 8km bei 5,30/km am Montag."
        ),
        saved_plan=plan,
    )
    r = _gate_post_save_plan_grounded(ctx)
    assert r.passed, r.reason


def test_response_with_no_specifics_passes():
    """A general plan summary without distance+pace pairs passes."""
    plan = _saved_plan_with_session({
        "day": "monday",
        "name": "Easy run",
        "duration_minutes": 60,
    })
    ctx = _ctx(
        response_text=(
            "Dein Plan ist gespeichert. Woche 1 ist Base-Phase mit "
            "lockeren easy runs und einem laengeren Lauf am Wochenende."
        ),
        saved_plan=plan,
    )
    r = _gate_post_save_plan_grounded(ctx)
    assert r.passed, r.reason


# ---------------------------------------------------------------------------
# Edge cases for the helpers
# ---------------------------------------------------------------------------


def test_normalize_pace_token_pads_seconds():
    assert _normalize_pace_token("5", "30") == "5:30"
    assert _normalize_pace_token("5", "5") == "5:05"


def test_normalize_distance_token_handles_german_comma():
    assert _normalize_distance_token("8,5") == "8.5"
    assert _normalize_distance_token("8.5") == "8.5"
    assert _normalize_distance_token("8") == "8"
    assert _normalize_distance_token("8.0") == "8"


def test_extract_session_pairs_finds_distance_pace():
    pairs = _extract_session_pairs("morgen 8km bei 4:30/km, easy einlaufen")
    assert len(pairs) >= 1
    assert pairs[0][0] == "8"
    assert pairs[0][1] == "4:30"


def test_extract_session_pairs_skips_far_apart_numbers():
    # A 200-char gap between "8km" and "4:30/km" must not pair them.
    text = "8km schon erledigt. " + ("filler text " * 30) + "spaeter 4:30/km auf der Bahn"
    pairs = _extract_session_pairs(text)
    assert pairs == []


def test_extract_session_pairs_empty_on_no_pace():
    pairs = _extract_session_pairs("Heute 10km easy gelaufen.")
    assert pairs == []


def test_empty_response_passes():
    ctx = _ctx(response_text="")
    r = _gate_post_save_plan_grounded(ctx)
    assert r.passed


def test_empty_saved_plan_fails_when_specifics_present():
    """If save_plan ran but the snapshot is empty AND the response has
    a (distance, pace) pair, the pair cannot be grounded -> fail.

    This protects against a planner_failed path where save_plan
    returned an error dict but the agent still narrated specifics.
    """
    ctx = _ctx(
        response_text="Plan gespeichert. Morgen 8km bei 4:30/km.",
        saved_plan={},
    )
    r = _gate_post_save_plan_grounded(ctx)
    assert not r.passed
