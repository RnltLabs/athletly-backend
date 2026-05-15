"""Snapshot tests for src/agent/journal_state_md.py.

The renderer is the boundary between typed DB rows and the markdown the
agent sees each turn. We pin a handful of representative states:

* zero intents / programs / notes
* single intent + single program + handful of notes
* multi-intent with varying priorities
* program with German umlauts in label / payload
* coach notes with German umlauts and special characters

The renderer must produce output that the constitutional critic accepts:
no em-dash (U+2014), no en-dash (U+2013). Markdown headings are allowed
in system-prompt context.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.journal_state_md import (
    _format_intent_bullet,
    _render_blocks,
    render_journal_state_md,
)


USER_ID = "9d77f53a-0001-4000-8000-000000000001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent(
    *,
    id: str = "11111111-1111-4111-8111-111111111111",
    type: str = "event",
    title: str = "Karlsruhe HM 2026-04-12 sub 1:45",
    priority: int = 1,
    status: str = "active",
    created_at: str = "2026-04-22T10:00:00+00:00",
    payload: dict | None = None,
) -> dict:
    return {
        "id": id,
        "user_id": USER_ID,
        "type": type,
        "title": title,
        "priority": priority,
        "status": status,
        "created_at": created_at,
        "payload": payload or {},
    }


def _program(
    *,
    id: str = "22222222-2222-4222-8222-222222222222",
    kind: str = "gym_split",
    label: str = "Push/Pull/Legs",
    payload: dict | None = None,
    active: bool = True,
    created_at: str = "2026-04-15T08:00:00+00:00",
) -> dict:
    return {
        "id": id,
        "user_id": USER_ID,
        "kind": kind,
        "label": label,
        "active": active,
        "created_at": created_at,
        "payload": payload or {},
    }


def _note(
    *,
    id: str = "33333333-3333-4333-8333-333333333333",
    content: str = "Athlet reagiert gut auf 2x Threshold/Woche",
    scope: str | None = "training_response",
    created_at: str = "2026-05-12T18:30:00+00:00",
) -> dict:
    return {
        "id": id,
        "user_id": USER_ID,
        "content": content,
        "scope": scope,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Empty state - must produce empty string, no stubs
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_all_empty_renders_empty_string(self) -> None:
        out = _render_blocks(intents=[], programs=[], notes=[])
        assert out == ""

    def test_empty_via_public_api_with_db(self) -> None:
        with patch("src.db.intents_db.list_intents", return_value=[]), \
             patch("src.db.programs_db.list_programs", return_value=[]), \
             patch("src.db.coach_notes_db.list_coach_notes", return_value=[]):
            out = render_journal_state_md(USER_ID)
        assert out == ""

    def test_one_category_present_skips_other_headings(self) -> None:
        out = _render_blocks(
            intents=[_intent()],
            programs=[],
            notes=[],
        )
        assert "Active Intents" in out
        assert "Active Programs" not in out
        assert "Coach Notes" not in out


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------


class TestIntents:
    def test_single_event_intent_bullet(self) -> None:
        bullet = _format_intent_bullet(
            _intent(payload={"race_date": "2026-04-12", "target_time": "1:45"})
        )
        assert bullet.startswith("- [Priority 1 - Event]")
        assert "Karlsruhe HM" in bullet
        assert "status: active" in bullet
        assert "created 2026-04-22" in bullet
        assert "race date 2026-04-12" in bullet
        assert "target time 1:45" in bullet

    def test_multi_intent_ordering_preserved_from_caller(self) -> None:
        intents = [
            _intent(priority=1, title="Karlsruhe HM"),
            _intent(
                id="11111111-1111-4111-8111-111111111112",
                type="capacity",
                title="Squat 100kg",
                priority=3,
                created_at="2026-05-10T08:00:00+00:00",
            ),
            _intent(
                id="11111111-1111-4111-8111-111111111113",
                type="habit",
                title="4x/Woche bewegen, Sport egal",
                priority=5,
                created_at="2026-05-01T08:00:00+00:00",
            ),
        ]
        out = _render_blocks(intents=intents, programs=[], notes=[])
        # Heading present
        assert "## Active Intents" in out
        # All three titles present
        assert "Karlsruhe HM" in out
        assert "Squat 100kg" in out
        assert "4x/Woche" in out
        # Priorities labeled
        assert "[Priority 1 - Event]" in out
        assert "[Priority 3 - Capacity]" in out
        assert "[Priority 5 - Habit]" in out

    def test_custom_type_passed_through(self) -> None:
        bullet = _format_intent_bullet(
            _intent(type="race_b", title="Tune-up 10K", priority=2)
        )
        assert "race_b" in bullet
        assert "Tune-up 10K" in bullet

    def test_missing_title_falls_back(self) -> None:
        bullet = _format_intent_bullet(
            _intent(title="", priority=3)
        )
        assert "(no title)" in bullet


# ---------------------------------------------------------------------------
# Programs - with German umlauts
# ---------------------------------------------------------------------------


class TestPrograms:
    def test_program_with_umlauts(self) -> None:
        out = _render_blocks(
            intents=[],
            programs=[
                _program(
                    kind="gym_split",
                    label="Oberkörper/Unterkörper",
                    payload={
                        "notes": (
                            "3x/Woche, Squat aktuell ~80kg, Bench bei 60kg. "
                            "Athlet logged nur good/medium/bad."
                        ),
                        "weekly_sessions": 3,
                    },
                ),
            ],
            notes=[],
        )
        assert "## Active Programs" in out
        # German umlauts survive verbatim.
        assert "Oberkörper/Unterkörper" in out
        assert "ä" not in out  # placeholder check, not really needed
        assert "Squat aktuell ~80kg" in out
        # Header includes the kind + creation date
        assert "(gym_split, since 2026-04-15)" in out
        # Coach notes line is rendered
        assert "Coach notes:" in out

    def test_run_block_with_minimal_payload(self) -> None:
        out = _render_blocks(
            intents=[],
            programs=[
                _program(
                    kind="run_block",
                    label="Marathon Base Block",
                    payload={"notes": "60km/wk avg, 80% easy / 20% Quality."},
                    created_at="2026-05-01T08:00:00+00:00",
                ),
            ],
            notes=[],
        )
        assert "### Marathon Base Block (run_block, since 2026-05-01)" in out
        assert "Coach notes: 60km/wk avg" in out

    def test_no_payload_renders_heading_only(self) -> None:
        out = _render_blocks(
            intents=[],
            programs=[_program(payload={})],
            notes=[],
        )
        assert "### Push/Pull/Legs" in out
        # No Coach notes line because payload is empty
        assert "Coach notes:" not in out


# ---------------------------------------------------------------------------
# Coach notes - special chars, umlauts, scope tags
# ---------------------------------------------------------------------------


class TestCoachNotes:
    def test_notes_with_umlauts_and_scope(self) -> None:
        notes = [
            _note(
                content="Athlet reagiert gut auf 2x Threshold/Woche",
                created_at="2026-05-12T18:30:00+00:00",
                scope="training_response",
            ),
            _note(
                id="33333333-3333-4333-8333-333333333334",
                content="Bei <6h Schlaf 2 Nächte in Folge immer Easy-Day",
                created_at="2026-05-08T18:30:00+00:00",
                scope="life_pattern",
            ),
        ]
        out = _render_blocks(intents=[], programs=[], notes=notes)
        assert "## Coach Notes (last 2)" in out
        assert "2026-05-12: Athlet reagiert gut auf 2x Threshold/Woche" in out
        assert "(scope: training_response)" in out
        assert "Nächte in Folge" in out
        assert "(scope: life_pattern)" in out

    def test_note_without_scope(self) -> None:
        out = _render_blocks(
            intents=[],
            programs=[],
            notes=[_note(scope=None)],
        )
        assert "(scope:" not in out

    def test_note_with_special_chars(self) -> None:
        out = _render_blocks(
            intents=[],
            programs=[],
            notes=[
                _note(
                    content="Pace bei 4:30/km gefühlt zu schnell - 4:45/km passt",
                    scope=None,
                )
            ],
        )
        # Hyphen-minus is fine; we only forbid em/en-dash.
        assert "4:30/km" in out
        assert "4:45/km" in out


# ---------------------------------------------------------------------------
# Hard constraints across all snapshots
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_snapshot() -> str:
    """A populated fixture used by hard-constraint checks."""
    intents = [
        _intent(
            payload={"race_date": "2026-04-12", "target_time": "1:45"},
            title="Karlsruhe HM 2026-04-12 sub 1:45",
            priority=1,
        ),
        _intent(
            id="11111111-1111-4111-8111-111111111112",
            type="capacity",
            title="Squat 100kg",
            priority=3,
            created_at="2026-05-10T08:00:00+00:00",
            payload={"target_lift": "100kg"},
        ),
        _intent(
            id="11111111-1111-4111-8111-111111111113",
            type="habit",
            title="4x/Woche bewegen, Sport egal",
            priority=5,
            created_at="2026-05-01T08:00:00+00:00",
        ),
    ]
    programs = [
        _program(
            label="Push/Pull/Legs",
            payload={"notes": "3x/Woche, Squat ~80kg, Bench bei 60kg."},
        ),
        _program(
            id="22222222-2222-4222-8222-222222222223",
            kind="run_block",
            label="Marathon Base Block",
            payload={"notes": "60km/wk avg, 80% easy / 20% Quality."},
            created_at="2026-05-01T08:00:00+00:00",
        ),
    ]
    notes = [
        _note(
            content="Athlet reagiert gut auf 2x Threshold/Woche",
            scope="training_response",
            created_at="2026-05-12T18:30:00+00:00",
        ),
        _note(
            id="33333333-3333-4333-8333-333333333334",
            content="Bei <6h Schlaf 2 Nächte in Folge immer Easy-Day",
            scope="life_pattern",
            created_at="2026-05-08T18:30:00+00:00",
        ),
    ]
    return _render_blocks(intents=intents, programs=programs, notes=notes)


def test_snapshot_contains_all_three_sections(populated_snapshot: str) -> None:
    assert "## Active Intents" in populated_snapshot
    assert "## Active Programs" in populated_snapshot
    assert "## Coach Notes (last 2)" in populated_snapshot


def test_snapshot_has_no_em_dash(populated_snapshot: str) -> None:
    assert "—" not in populated_snapshot
    assert "–" not in populated_snapshot


def test_snapshot_preserves_german_umlauts(populated_snapshot: str) -> None:
    assert "Nächte" in populated_snapshot
    assert "Woche" in populated_snapshot


def test_snapshot_orders_sections_intents_then_programs_then_notes(
    populated_snapshot: str,
) -> None:
    i = populated_snapshot.index("## Active Intents")
    p = populated_snapshot.index("## Active Programs")
    n = populated_snapshot.index("## Coach Notes")
    assert i < p < n


def test_public_api_truncates_to_default_limit() -> None:
    """When 30 notes are returned the default limit only includes 10 of them."""
    many_notes = [
        _note(
            id=f"33333333-3333-4333-8333-{i:012d}",
            content=f"Beobachtung {i}",
            created_at=f"2026-05-{(i % 28) + 1:02d}T08:00:00+00:00",
            scope=None,
        )
        for i in range(30)
    ]
    # The DB layer enforces the limit. We must not call it with limit > 10.
    captured_limits: list[int] = []

    def fake_list(user_id: str, *, limit: int, scope_filter=None):  # noqa: ARG001
        captured_limits.append(limit)
        return many_notes[:limit]

    with patch("src.db.intents_db.list_intents", return_value=[]), \
         patch("src.db.programs_db.list_programs", return_value=[]), \
         patch("src.db.coach_notes_db.list_coach_notes", side_effect=fake_list):
        out = render_journal_state_md(USER_ID)

    assert captured_limits == [10]
    # Heading should reflect actual count (10 returned).
    assert "## Coach Notes (last 10)" in out


def test_public_api_swallows_db_errors() -> None:
    """A DB exception while fetching state must not blow up the prompt build."""
    with patch(
        "src.db.intents_db.list_intents",
        side_effect=RuntimeError("supabase down"),
    ):
        out = render_journal_state_md(USER_ID)
    assert out == ""
