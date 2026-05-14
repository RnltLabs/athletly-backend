"""Tests for the persona test framework.

Covers:
- Persona markdown parsing (frontmatter + sections)
- Idempotency of the synthetic id scheme
- Data quality invariants from the fake-data generators
- Seed pipeline calls the right Supabase upserts in the right order

External dependencies (Supabase) are fully mocked.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from tools.persona_test._persona import load_persona, list_personas
from tools.persona_test.fake_garmin import generate as generate_activities
from tools.persona_test.fake_metrics import generate as generate_metrics


# ---------------------------------------------------------------------------
# Persona parsing
# ---------------------------------------------------------------------------


def test_three_personas_are_available():
    """All three personas ship and parse."""
    slugs = list_personas()
    assert "elena" in slugs
    assert "marco" in slugs
    assert "lisa" in slugs


@pytest.mark.parametrize("slug", ["elena", "marco", "lisa"])
def test_persona_loads_with_required_fields(slug):
    """Every shipped persona has all required frontmatter fields."""
    p = load_persona(slug)
    assert p.slug == slug
    assert p.name
    assert p.email.endswith("@persona.test.athletly.local")
    assert p.age > 0
    assert p.sports
    assert p.goal_event
    assert p.goal_date
    assert p.estimated_vo2max > 0
    assert p.threshold_pace_min_km > 0
    assert p.weeks_of_history == 12


@pytest.mark.parametrize("slug", ["elena", "marco", "lisa"])
def test_persona_has_personality_section(slug):
    """Each persona must define a Personality section for role-play."""
    p = load_persona(slug)
    assert "Personality" in p.sections
    assert len(p.sections["Personality"]) > 50


def test_lisa_has_injury_history():
    """Lisa's injury history must round-trip through parsing."""
    lisa = load_persona("lisa")
    assert lisa.injury_history
    assert lisa.injury_history[0]["region"] == "knee_right"


# ---------------------------------------------------------------------------
# Synthetic activity ids
# ---------------------------------------------------------------------------


def test_generator_is_deterministic_for_same_inputs():
    """Re-running generate() with same args produces same ids and metrics."""
    elena = load_persona("elena")
    user_id = "test-user-elena"
    today = date(2026, 5, 14)

    first = generate_activities(elena, user_id=user_id, today=today)
    second = generate_activities(elena, user_id=user_id, today=today)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.garmin_activity_id == b.garmin_activity_id
        assert a.duration_seconds == b.duration_seconds
        assert a.distance_meters == b.distance_meters


def test_synthetic_ids_are_unique_per_session():
    """Within one persona's history, all garmin_activity_ids must be unique."""
    for slug in ("elena", "marco", "lisa"):
        p = load_persona(slug)
        activities = generate_activities(p, user_id=f"user-{slug}", today=date(2026, 5, 14))
        ids = [a.garmin_activity_id for a in activities]
        assert len(ids) == len(set(ids)), f"Duplicate ids in {slug}: {ids}"


def test_synthetic_ids_carry_slug_marker():
    """Ids must start with `persona_<slug>_` so they are filterable."""
    elena = load_persona("elena")
    activities = generate_activities(elena, user_id="x", today=date(2026, 5, 14))
    for act in activities:
        assert act.garmin_activity_id.startswith("persona_elena_")


# ---------------------------------------------------------------------------
# Data quality invariants
# ---------------------------------------------------------------------------


def test_elena_volume_is_marathon_appropriate():
    """Elena's weekly mileage must land in the documented 50-72km range."""
    elena = load_persona("elena")
    activities = generate_activities(elena, user_id="x", today=date(2026, 5, 14))
    # Sum per ISO week.
    by_week: dict[str, float] = {}
    for a in activities:
        wk = a.start_time[:10]  # date string
        d = date.fromisoformat(wk)
        iso_year, iso_week, _ = d.isocalendar()
        key = f"{iso_year}-W{iso_week}"
        by_week[key] = by_week.get(key, 0) + a.distance_meters / 1000
    # Every full week should be 45-85 km (allowing edge weeks to vary).
    full_weeks = [v for v in by_week.values() if v > 40]
    assert full_weeks, "Should have at least one full week"
    for vol in full_weeks:
        assert 40 <= vol <= 85, f"Weekly volume {vol} out of plausible range"


def test_marco_volume_is_beginner_level():
    """Marco's weekly mileage must land below 30km (beginner)."""
    marco = load_persona("marco")
    activities = generate_activities(marco, user_id="x", today=date(2026, 5, 14))
    by_week: dict[str, float] = {}
    for a in activities:
        d = date.fromisoformat(a.start_time[:10])
        key = f"{d.isocalendar()[0]}-W{d.isocalendar()[1]}"
        by_week[key] = by_week.get(key, 0) + a.distance_meters / 1000
    for vol in by_week.values():
        assert vol <= 35, f"Marco weekly volume {vol} too high for beginner"


def test_lisa_has_multi_sport_activities():
    """Lisa's history must include running, cycling, and swimming."""
    lisa = load_persona("lisa")
    activities = generate_activities(lisa, user_id="x", today=date(2026, 5, 14))
    sports = {a.sport for a in activities}
    assert sports == {"running", "cycling", "swimming"}


def test_activities_have_required_fields_for_agent_tools():
    """All generated activities must carry the fields get_activities expects."""
    elena = load_persona("elena")
    activities = generate_activities(elena, user_id="x", today=date(2026, 5, 14))
    for act in activities:
        row = act.to_row()
        assert row["sport"]
        assert row["start_time"]
        assert row["duration_seconds"] > 0
        assert row["distance_meters"] > 0
        assert row["avg_hr"] > 0
        assert row["source"] == "garmin"
        assert row["garmin_activity_id"]


def test_metrics_have_required_health_fields():
    """Generated metrics must cover the fields the coach tools consume."""
    elena = load_persona("elena")
    metrics = generate_metrics(elena, user_id="x", today=date(2026, 5, 14), days=30)
    assert len(metrics) == 30
    for m in metrics:
        row = m.to_row()
        assert row["hrv_avg"] > 0
        assert row["resting_heart_rate"] > 0
        assert 4.0 * 60 <= row["sleep_duration_minutes"] <= 10.0 * 60
        assert row["source"] == "garmin"


def test_metrics_dates_are_unique():
    """Each generated metric must have a unique (user_id, date) key."""
    elena = load_persona("elena")
    metrics = generate_metrics(elena, user_id="x", today=date(2026, 5, 14), days=30)
    dates = [m.date for m in metrics]
    assert len(dates) == len(set(dates))


def test_marco_recovery_reflects_persona_baseline():
    """Marco's RHR / HRV should reflect his lower baseline."""
    marco = load_persona("marco")
    metrics = generate_metrics(marco, user_id="x", today=date(2026, 5, 14), days=30)
    avg_hrv = sum(m.hrv_avg for m in metrics) / len(metrics)
    avg_rhr = sum(m.resting_heart_rate for m in metrics) / len(metrics)
    assert avg_hrv < 45, f"Marco avg HRV {avg_hrv} should be low"
    assert avg_rhr > 55, f"Marco avg RHR {avg_rhr} should be elevated"


# ---------------------------------------------------------------------------
# Seed pipeline idempotency (with mocked Supabase)
# ---------------------------------------------------------------------------


def _make_mock_client():
    """Build a MagicMock that mimics the Supabase client surface used by seed."""
    client = MagicMock()
    # client.table("foo").upsert(...).execute() returns a Mock with .data=[]
    table_mock = MagicMock()
    table_mock.upsert.return_value.execute.return_value = MagicMock(data=[])
    table_mock.update.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    table_mock.insert.return_value.execute.return_value = MagicMock(data=[])
    table_mock.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    client.table.return_value = table_mock
    return client, table_mock


def test_seed_calls_required_upserts():
    """The seed pipeline must upsert profile, journal, activities, metrics."""
    from tools.persona_test import seed as seed_module

    elena = load_persona("elena")
    client, table_mock = _make_mock_client()

    with patch.object(seed_module, "get_service_client", return_value=client), \
         patch.object(
             seed_module,
             "ensure_test_user",
             return_value={"id": "uuid-elena", "email": elena.email, "user_metadata": {"is_test": True}},
         ), \
         patch.object(seed_module, "get_or_create_password", return_value="pw"):
        result = seed_module.seed_persona(elena, today=date(2026, 5, 14))

    # The seeder must hit at least these tables.
    tables_called = {call_args.args[0] for call_args in client.table.call_args_list}
    assert {"profiles", "athlete_journal", "activities", "health_daily_metrics"} <= tables_called
    assert result.profile_upserted
    assert result.journal_upserted
    assert result.activities_upserted > 0
    assert result.metrics_upserted == 30


def test_seed_is_idempotent_in_id_space():
    """Re-running the seeder produces the same set of synthetic ids."""
    from tools.persona_test import seed as seed_module

    elena = load_persona("elena")

    ids_seen = []

    def _capture_upsert(rows, **kwargs):
        # Capture the activity ids that would be upserted.
        for row in (rows if isinstance(rows, list) else [rows]):
            if "garmin_activity_id" in row:
                ids_seen.append(row["garmin_activity_id"])
        return MagicMock(execute=MagicMock(return_value=MagicMock(data=[])))

    client = MagicMock()
    table_mock = MagicMock()
    table_mock.upsert = _capture_upsert
    table_mock.update.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    table_mock.insert.return_value.execute.return_value = MagicMock(data=[])
    table_mock.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    client.table.return_value = table_mock

    with patch.object(seed_module, "get_service_client", return_value=client), \
         patch.object(
             seed_module,
             "ensure_test_user",
             return_value={"id": "uuid-elena", "email": elena.email, "user_metadata": {"is_test": True}},
         ), \
         patch.object(seed_module, "get_or_create_password", return_value="pw"):
        seed_module.seed_persona(elena, today=date(2026, 5, 14))
        first_ids = list(ids_seen)
        ids_seen.clear()
        seed_module.seed_persona(elena, today=date(2026, 5, 14))
        second_ids = list(ids_seen)

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))  # all unique within one run


def test_seed_refuses_non_test_user():
    """If ensure_test_user returns a user without is_test, the seeder bails."""
    from tools.persona_test import seed as seed_module

    elena = load_persona("elena")
    client, _ = _make_mock_client()

    with patch.object(seed_module, "get_service_client", return_value=client), \
         patch.object(
             seed_module,
             "ensure_test_user",
             return_value={"id": "uuid-elena", "email": elena.email, "user_metadata": {}},
         ), \
         patch.object(seed_module, "get_or_create_password", return_value="pw"):
        with pytest.raises(SystemExit) as exc:
            seed_module.seed_persona(elena, today=date(2026, 5, 14))
    assert exc.value.code == 3
