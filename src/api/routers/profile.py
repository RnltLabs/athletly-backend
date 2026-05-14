"""Profile router: lifecycle operations on the authenticated user's data.

Currently exposes a single destructive operation:

    POST /profile/reset

This wipes every row owned by the authenticated user across all
user-scoped tables EXCEPT provider_tokens. The auth row in
``auth.users`` is also preserved so the user stays signed in.

Provider tokens (Garmin / Strava OAuth) are deliberately kept so the
athlete does NOT need to redo the SSO flow after a reset. Re-logging in
to Garmin is the heaviest step (rate-limited, sometimes requires MFA),
so the UX optimises for: chat-history + plans + activities go away, but
the wearable stays paired. If the athlete really wants to drop the
provider too, they use the explicit disconnect button in the profile.

This endpoint is destructive. Frontend MUST gate it behind a confirmation
dialog.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import get_user_id
from src.db.client import get_supabase

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GET /profile/identity - "Wie Athletly mich sieht" settings screen
# ---------------------------------------------------------------------------
#
# Single source of truth: the canonical sections live in the athlete journal
# (markdown). This endpoint projects them into a structured JSON payload the
# frontend can render in a settings screen. Structured profile fields come
# from the same profiles table the agent reads via UserModelDB.
#
# Resilience: every data source is wrapped in try/except. A failure in one
# source MUST NOT break the whole response, the frontend renders missing
# pieces as empty-state placeholders.


# Section mapping: journal section name -> (key, German title).
# Iteration order follows CANONICAL_SECTIONS for stable output.
_SECTION_META: dict[str, tuple[str, str]] = {
    "Identity": ("identity", "Identität"),
    "Current Goal": ("current_goal", "Aktuelles Ziel"),
    "Goal Timeline": ("goal_timeline", "Ziel-Historie"),
    "What I know about my training": ("training", "Mein Training"),
    "Preferences": ("preferences", "Präferenzen"),
    "Open Threads": ("open_threads", "Offene Themen"),
    "What we tried": ("what_we_tried", "Was wir probiert haben"),
}

# Placeholder bodies that the journal seed file inserts for empty sections.
# Treat these as "empty" so the frontend shows the empty-state UI rather
# than the seed text.
_EMPTY_BODY_MARKERS: frozenset[str] = frozenset({
    "",
    "_(not yet filled in)_",
    "_(no active goal)_",
})


class IdentitySection(BaseModel):
    """One section of the athlete journal, projected for the UI."""

    key: str = Field(..., description="Stable machine key, e.g. 'identity'.")
    title: str = Field(..., description="German display title for the UI.")
    body: str = Field(..., description="Section body text (markdown).")
    is_empty: bool = Field(
        ...,
        description="True when the section has no real content yet.",
    )


class IdentityGoal(BaseModel):
    event: str | None = None
    target_date: str | None = None
    target_time: str | None = None


class IdentityFitness(BaseModel):
    vo2max_estimate: int | float | None = None
    threshold_pace_min_km: float | None = None
    weekly_volume_km: float | None = None
    ftp_watts: int | float | None = None


class IdentityStructured(BaseModel):
    sports: list[str] = Field(default_factory=list)
    training_days_per_week: int | None = None
    max_session_minutes: int | None = None
    fitness: IdentityFitness = Field(default_factory=IdentityFitness)
    goal: IdentityGoal = Field(default_factory=IdentityGoal)


class IdentityResponse(BaseModel):
    """Canonical contract consumed by the settings screen."""

    athlete_name: str | None = None
    sections: list[IdentitySection] = Field(default_factory=list)
    structured: IdentityStructured = Field(default_factory=IdentityStructured)
    last_updated_at: str | None = None


def _is_empty_body(body: str) -> bool:
    """Decide whether a section body should render as an empty placeholder."""
    return body.strip() in _EMPTY_BODY_MARKERS


def _load_journal_sections(user_id: str) -> tuple[str | None, dict[str, str]]:
    """Read the journal and parse sections. Returns (athlete_name, sections).

    Any failure is swallowed: callers get (None, {}) and individual sections
    will render as empty.
    """
    try:
        from src.agent.athlete_journal import parse_sections, read_journal

        raw = read_journal(user_id)
        if not raw:
            return None, {}
        parsed = parse_sections(raw)
        name = parsed.get("_title")
        # Strip the title pseudo-key so callers only iterate real sections.
        sections = {k: v for k, v in parsed.items() if k != "_title"}
        return name, sections
    except Exception:
        logger.warning(
            "identity: journal read failed for user=%s", user_id, exc_info=True,
        )
        return None, {}


def _build_section_list(journal: dict[str, str]) -> list[IdentitySection]:
    """Project journal sections into the ordered list the API returns.

    Iterates CANONICAL_SECTIONS to guarantee stable order; missing or
    placeholder sections are emitted with is_empty=True and an empty body.
    Does not mutate the *journal* dict.
    """
    try:
        from src.agent.athlete_journal import CANONICAL_SECTIONS
    except Exception:
        logger.warning("identity: CANONICAL_SECTIONS import failed", exc_info=True)
        CANONICAL_SECTIONS = tuple(_SECTION_META.keys())  # noqa: N806

    result: list[IdentitySection] = []
    for journal_name in CANONICAL_SECTIONS:
        meta = _SECTION_META.get(journal_name)
        if meta is None:
            # Defensive: a canonical section without a mapping should still
            # be surfaced rather than dropped silently.
            key = journal_name.lower().replace(" ", "_")
            title = journal_name
        else:
            key, title = meta

        body = journal.get(journal_name, "")
        empty = _is_empty_body(body)
        result.append(
            IdentitySection(
                key=key,
                title=title,
                body="" if empty else body,
                is_empty=empty,
            )
        )
    return result


def _load_structured(user_id: str) -> IdentityStructured:
    """Load structured profile fields. Returns a default-filled model on error."""
    try:
        from src.db.user_model_db import UserModelDB

        model = UserModelDB.load_or_create(user_id)
        profile = model.project_profile()
    except Exception:
        logger.warning(
            "identity: structured profile load failed for user=%s",
            user_id,
            exc_info=True,
        )
        return IdentityStructured()

    fitness_src = profile.get("fitness") or {}
    constraints_src = profile.get("constraints") or {}
    goal_src = profile.get("goal") or {}

    return IdentityStructured(
        sports=list(profile.get("sports") or []),
        training_days_per_week=constraints_src.get("training_days_per_week"),
        max_session_minutes=constraints_src.get("max_session_minutes"),
        fitness=IdentityFitness(
            vo2max_estimate=fitness_src.get("estimated_vo2max"),
            threshold_pace_min_km=fitness_src.get("threshold_pace_min_km"),
            weekly_volume_km=fitness_src.get("weekly_volume_km"),
            ftp_watts=fitness_src.get("ftp_watts"),
        ),
        goal=IdentityGoal(
            event=goal_src.get("event"),
            target_date=goal_src.get("target_date"),
            target_time=goal_src.get("target_time"),
        ),
    )


def _load_last_updated_at(user_id: str) -> str | None:
    """Fetch updated_at from the athlete_journal table. None on any failure."""
    try:
        client = get_supabase()
        res = (
            client.table("athlete_journal")
            .select("updated_at")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows: list[dict[str, Any]] = res.data or []
        if not rows:
            return None
        value = rows[0].get("updated_at")
        return value if isinstance(value, str) else None
    except Exception:
        logger.warning(
            "identity: updated_at lookup failed for user=%s",
            user_id,
            exc_info=True,
        )
        return None


@router.get("/identity", response_model=IdentityResponse)
async def get_identity(
    user_id: Annotated[str, Depends(get_user_id)],
) -> IdentityResponse:
    """Return everything the agent knows about the athlete.

    Powers the "Wie Athletly mich sieht" settings screen. Each data source
    is loaded independently so a single failure cannot break the response.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    athlete_name, journal_sections = _load_journal_sections(user_id)
    sections = _build_section_list(journal_sections)
    structured = _load_structured(user_id)
    last_updated_at = _load_last_updated_at(user_id)

    return IdentityResponse(
        athlete_name=athlete_name,
        sections=sections,
        structured=structured,
        last_updated_at=last_updated_at,
    )


# Tables keyed by user_id that get wiped during reset.
# Order is from leaf to root (dependents first) where there are FKs.
# Deliberately excludes ``provider_tokens`` - see module docstring.
USER_TABLES_IN_ORDER: list[str] = [
    "pending_actions",
    "proactive_queue",
    "product_recommendations",
    "episodes",
    "plans",
    "activities",
    "health_daily_metrics",
    "athlete_journal",
    "profiles",
]


def _wipe_session_messages(user_id: str, results: dict[str, str]) -> None:
    """Delete session_messages by joining through sessions for the user."""
    try:
        client = get_supabase()
        sessions = (
            client.table("sessions").select("id").eq("user_id", user_id).execute()
        )
        session_ids = [row["id"] for row in (sessions.data or []) if row.get("id")]
        if not session_ids:
            results["session_messages"] = "ok (no sessions)"
            return
        client.table("session_messages").delete().in_(
            "session_id", session_ids
        ).execute()
        results["session_messages"] = f"ok ({len(session_ids)} sessions)"
    except Exception as exc:
        logger.warning("session_messages wipe failed for %s: %s", user_id, exc)
        results["session_messages"] = f"error: {exc}"


def _wipe_table(table: str, user_id: str, results: dict[str, str]) -> None:
    """Best-effort delete of all rows for *user_id* in *table*."""
    try:
        get_supabase().table(table).delete().eq("user_id", user_id).execute()
        results[table] = "ok"
    except Exception as exc:
        # A missing table or schema-drift here should not abort the whole
        # reset - record the failure and keep going. The client just sees
        # which tables succeeded vs failed.
        logger.warning("Wipe %s for %s failed: %s", table, user_id, exc)
        results[table] = f"error: {exc}"


@router.post("/reset", status_code=200)
async def reset_profile(
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict:
    """Wipe every user-owned row except the auth identity itself.

    Returns a per-table summary so the frontend can surface partial
    failures (e.g. a renamed table on a fresh schema) without claiming
    success.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    logger.info("Profile reset requested by user=%s", user_id)
    results: dict[str, str] = {}

    # session_messages first (FK on sessions), then sessions.
    _wipe_session_messages(user_id, results)
    _wipe_table("sessions", user_id, results)

    for table in USER_TABLES_IN_ORDER:
        _wipe_table(table, user_id, results)

    failures = [k for k, v in results.items() if v.startswith("error")]
    status = "ok" if not failures else "partial"
    logger.info(
        "Profile reset complete user=%s status=%s failures=%d",
        user_id, status, len(failures),
    )

    return {"status": status, "user_id": user_id, "tables": results}
