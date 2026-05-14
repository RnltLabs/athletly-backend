"""Idempotent end-to-end seeder for a persona test user.

Usage::

    python tools/persona_test/seed.py elena
    python tools/persona_test/seed.py marco
    python tools/persona_test/seed.py lisa

Steps performed (all idempotent):
1. Resolve persona markdown.
2. Ensure auth user exists (service-role admin API), with is_test=True metadata.
3. Upsert profiles row.
4. Build athlete_journal markdown and upsert.
5. Generate fake Garmin activities and upsert (deterministic synthetic ids).
6. Generate 30 days of health_daily_metrics and upsert.
7. If persona declares an active plan, upsert a plan row.

Exit codes:
    0 = success
    2 = configuration / Supabase auth error
    3 = refused (user lacks is_test marker)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Path bootstrap so the script is runnable both as a module and as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.persona_test._persona import Persona, list_personas, load_persona
from tools.persona_test._supabase import (
    assert_is_test_user,
    ensure_test_user,
    get_or_create_password,
    get_service_client,
)
from tools.persona_test.fake_garmin import generate as generate_activities
from tools.persona_test.fake_metrics import generate as generate_metrics

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Summary of what the seed run produced (used by tests + CLI report)."""

    persona_slug: str
    user_id: str
    email: str
    profile_upserted: bool = False
    journal_upserted: bool = False
    activities_upserted: int = 0
    metrics_upserted: int = 0
    plan_upserted: bool = False
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Persona:    {self.persona_slug}",
            f"User id:    {self.user_id}",
            f"Email:      {self.email}",
            f"Profile:    {'upserted' if self.profile_upserted else 'skipped'}",
            f"Journal:    {'upserted' if self.journal_upserted else 'skipped'}",
            f"Activities: {self.activities_upserted}",
            f"Metrics:    {self.metrics_upserted}",
            f"Plan:       {'upserted' if self.plan_upserted else 'none'}",
        ]
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Seed pipeline
# ---------------------------------------------------------------------------


def seed_persona(persona: Persona, *, today: date | None = None) -> SeedResult:
    """Run the full seed pipeline for *persona*. Idempotent."""
    client = get_service_client()
    password = get_or_create_password(persona.slug)
    user = ensure_test_user(
        client,
        slug=persona.slug,
        email=persona.email,
        password=password,
    )
    assert_is_test_user(user)

    user_id = user["id"]
    result = SeedResult(
        persona_slug=persona.slug,
        user_id=user_id,
        email=persona.email,
    )

    _upsert_profile(client, persona, user_id)
    result.profile_upserted = True

    _upsert_journal(client, persona, user_id)
    result.journal_upserted = True

    activities = generate_activities(persona, user_id=user_id, today=today)
    if activities:
        rows = [a.to_row() for a in activities]
        _upsert_activities(client, rows)
        result.activities_upserted = len(rows)

    metrics = generate_metrics(persona, user_id=user_id, today=today)
    if metrics:
        rows = [m.to_row() for m in metrics]
        _upsert_metrics(client, rows)
        result.metrics_upserted = len(rows)

    if persona.has_active_plan:
        _upsert_plan(client, persona, user_id)
        result.plan_upserted = True

    return result


# ---------------------------------------------------------------------------
# Per-table writers
# ---------------------------------------------------------------------------


def _upsert_profile(client, persona: Persona, user_id: str) -> None:
    """Upsert the profiles row from persona frontmatter."""
    row = {
        "user_id": user_id,
        "name": persona.name,
        "sports": persona.sports,
        "goal_event": persona.goal_event,
        "goal_target_date": persona.goal_date,
        "goal_target_time": persona.goal_target_time,
        "goal_type": persona.goal_type,
        "estimated_vo2max": persona.estimated_vo2max,
        "threshold_pace_min_km": persona.threshold_pace_min_km,
        "weekly_volume_km": persona.weekly_volume_km,
        "training_days_per_week": persona.training_days_per_week,
        "max_session_minutes": persona.max_session_minutes,
        "available_sports": persona.sports,
        "onboarding_complete": True,
        "meta": {
            "is_test_persona": True,
            "persona_slug": persona.slug,
            "age": persona.age,
            "location": persona.location,
        },
    }
    client.table("profiles").upsert(row, on_conflict="user_id").execute()


def _upsert_journal(client, persona: Persona, user_id: str) -> None:
    """Build the athlete_journal markdown and upsert."""
    content = _render_journal(persona)
    client.table("athlete_journal").upsert(
        {"user_id": user_id, "content": content},
        on_conflict="user_id",
    ).execute()


def _render_journal(persona: Persona) -> str:
    """Compose the markdown journal from persona sections.

    Maps persona-file sections to canonical athlete_journal sections so the
    coach sees the same structure it would see for a real user.
    """
    lines = [f"# {persona.name}", ""]

    identity = persona.sections.get("Identity", "").strip()
    if identity:
        lines += ["## Identity", "", identity, ""]
        # Add structured identity facts so the coach has hard data points.
        lines += [
            f"- Alter: {persona.age}",
            f"- Wohnort: {persona.location}",
            f"- Sportarten: {', '.join(persona.sports)}",
            f"- Trainingstage/Woche: {persona.training_days_per_week}",
            f"- Max Session-Dauer: {persona.max_session_minutes} min",
            "",
        ]

    goal = persona.sections.get("Goal", "").strip()
    lines += [
        "## Current Goal",
        "",
        f"- Event: {persona.goal_event}",
        f"- Datum: {persona.goal_date}",
        f"- Zielzeit: {persona.goal_target_time}",
        f"- Goal-Type: {persona.goal_type}",
    ]
    if persona.goal_pace_min_km:
        lines.append(f"- Goal-Pace: {persona.goal_pace_min_km:.2f} min/km")
    lines.append("")
    if goal:
        lines += [goal, ""]

    lines += [
        "## What I know about my training",
        "",
        f"- Threshold-Pace: {persona.threshold_pace_min_km:.2f} min/km",
        f"- VO2max (Garmin): {persona.estimated_vo2max:.0f}",
        f"- Wochenvolumen Ziel: {persona.weekly_volume_km:.0f} km",
    ]
    if persona.ftp_watts:
        lines.append(f"- FTP Rad: {persona.ftp_watts} W")
    if persona.swim_css_min_per_100m:
        lines.append(f"- Schwimm-CSS: {persona.swim_css_min_per_100m:.2f} min/100m")
    if persona.injury_history:
        lines.append("- Verletzungshistorie:")
        for inj in persona.injury_history:
            line = f"  - {inj.get('region', '?')}: {inj.get('type', '?')}"
            if inj.get("date"):
                line += f" ({inj['date']})"
            if inj.get("status"):
                line += f" - Status: {inj['status']}"
            lines.append(line)
    lines.append("")

    open_threads = persona.sections.get("Open Threads", "").strip()
    if open_threads:
        lines += ["## Open Threads", "", open_threads, ""]

    recovery = persona.sections.get("Recovery patterns", "").strip()
    if recovery:
        lines += ["## Preferences", "", recovery, ""]

    return "\n".join(lines).strip() + "\n"


def _upsert_activities(client, rows: list[dict]) -> None:
    """Upsert activities in batches keyed on (user_id, garmin_activity_id)."""
    BATCH = 50
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        client.table("activities").upsert(
            chunk,
            on_conflict="user_id,garmin_activity_id",
        ).execute()


def _upsert_metrics(client, rows: list[dict]) -> None:
    """Upsert health_daily_metrics keyed on (user_id, date, source)."""
    BATCH = 50
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        client.table("health_daily_metrics").upsert(
            chunk,
            on_conflict="user_id,date,source",
        ).execute()


def _upsert_plan(client, persona: Persona, user_id: str) -> None:
    """For personas with has_active_plan, mint a minimal plan.

    Idempotency: we first check whether the user has an active plan that
    was generated for this persona slug (meta marker). If yes, skip. If
    no, deactivate existing plans and insert.
    """
    existing = (
        client.table("plans")
        .select("id, plan_data, active")
        .eq("user_id", user_id)
        .eq("active", True)
        .execute()
    )
    for row in existing.data or []:
        meta = (row.get("plan_data") or {}).get("meta") or {}
        if meta.get("persona_slug") == persona.slug:
            return  # already in place

    # Build a minimal but realistic 24-week plan skeleton.
    plan_data = {
        "weeks": [
            {
                "week_index": i,
                "phase": (
                    "base" if i < 8 else "build" if i < 18 else "peak" if i < 22 else "taper"
                ),
                "target_volume_km": persona.weekly_volume_km + i * 0.5,
                "key_sessions": (
                    ["long", "threshold"] if persona.goal_type == "ironman"
                    else ["long", "threshold", "easy"]
                ),
            }
            for i in range(24)
        ],
        "goal_event": persona.goal_event,
        "goal_date": persona.goal_date,
        "goal_target_time": persona.goal_target_time,
        "meta": {
            "is_test_persona_plan": True,
            "persona_slug": persona.slug,
        },
    }
    # Deactivate any prior active plan.
    client.table("plans").update({"active": False}).eq("user_id", user_id).eq(
        "active", True,
    ).execute()
    client.table("plans").insert({
        "user_id": user_id,
        "plan_data": plan_data,
        "active": True,
    }).execute()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a persona test user with profile, journal, activities, metrics.",
    )
    parser.add_argument(
        "persona",
        help=f"Persona slug. Available: {list_personas()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        persona = load_persona(args.persona)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    result = seed_persona(persona)
    sys.stdout.write(result.render() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
