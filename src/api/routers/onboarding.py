"""Onboarding router for the Athletly FastAPI backend.

Exposes:
    POST /api/onboarding/parse-voice  — public, rate-limited; AI parses voice transcript
    POST /api/onboarding/setup        — auth required; saves profile data after signup

Architecture:
    - parse-voice is intentionally unauthenticated (runs before account creation).
      Abuse is mitigated with a per-IP slowapi rate limit (10/hour).
    - setup updates the Supabase ``profiles`` table and fires a background agent
      session with context="onboarding" so the coach can prepare the initial plan.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Literal

import litellm
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.api.auth import get_user_id
from src.api.rate_limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["onboarding"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARSE_MODEL = "gemini/gemini-2.0-flash-lite"

_SPORT_SYSTEM_PROMPT = (
    "Extract sport/activity names from the user's German text.\n"
    "Return JSON: {\"items\": [\"Sport1\", \"Sport2\"]}\n"
    "Only include recognized sports/activities. Keep names in German."
)

_GOALS_SYSTEM_PROMPT = (
    "Extract fitness goals from the user's German text.\n"
    "Return JSON: {\n"
    "  \"items\": [\"Goal tag 1\", \"Goal tag 2\"],\n"
    "  \"structured\": {\n"
    "    \"event\": \"Halbmarathon\",\n"
    "    \"location\": \"Karlsruhe\",\n"
    "    \"date\": \"September\",\n"
    "    \"target_time\": \"unter 1:30h\"\n"
    "  }\n"
    "}\n"
    "Keep all text in German. structured is optional, only for race goals."
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "sport": _SPORT_SYSTEM_PROMPT,
    "goals": _GOALS_SYSTEM_PROMPT,
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ParseVoiceRequest(BaseModel):
    """Body for POST /api/onboarding/parse-voice."""

    text: str
    step: Literal["sport", "goals"]


class ParseVoiceResponse(BaseModel):
    """Response for POST /api/onboarding/parse-voice."""

    items: list[str]
    structured: dict | None = None


class OnboardingSetupRequest(BaseModel):
    """Body for POST /api/onboarding/setup."""

    sports: list[str]
    custom_sport: str | None = None
    goals: list[str]
    custom_goal: str | None = None
    available_days: list[str]  # ["mon", "tue", "wed", ...]
    wearable: str | None = None  # "garmin" | "apple_health" | "health_connect"


class OnboardingSetupResponse(BaseModel):
    """Response for POST /api/onboarding/setup."""

    status: str   # "ok"
    message: str  # "Setup gespeichert"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_parse_llm(text: str, step: Literal["sport", "goals"]) -> dict:
    """Call the LiteLLM completion and return the parsed JSON dict.

    On any LLM or JSON error, returns a safe fallback with empty ``items``.
    """
    system_prompt = _SYSTEM_PROMPTS[step]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    try:
        response = await litellm.acompletion(
            model=_PARSE_MODEL,
            messages=messages,
            temperature=0.1,
            drop_params=True,
        )
        raw_content: str = response.choices[0].message.content or ""
        logger.debug("parse-voice LLM raw response (step=%s): %s", step, raw_content)
    except Exception:
        logger.exception("LLM call failed for parse-voice (step=%s)", step)
        return {"items": []}

    # Strip markdown code fences that some models wrap JSON in
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop first line (```json or ```) and last line (```)
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        parsed: dict = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "parse-voice: could not decode JSON from LLM output (step=%s): %r",
            step,
            raw_content[:200],
        )
        return {"items": []}

    # Normalise: items must be a list of strings
    items = parsed.get("items", [])
    if not isinstance(items, list):
        items = []
    items = [str(i) for i in items if i]

    result: dict = {"items": items}

    if step == "goals":
        structured = parsed.get("structured")
        if isinstance(structured, dict) and structured:
            result["structured"] = structured

    return result


async def _update_profile(user_id: str, body: OnboardingSetupRequest) -> None:
    """Persist onboarding data to the Supabase ``profiles`` table.

    Updates:
        sports     — JSONB list of sport strings (+ custom_sport appended)
        goal       — JSONB with goals, custom_goal, and wearable
        meta       — JSONB merged with available_days
        wearable   — plain text field

    Raises:
        HTTPException(503): If the Supabase update fails.
    """
    from src.db.client import get_async_supabase

    supabase = await get_async_supabase()

    # Build sports list (deduplicated, immutable)
    sports: list[str] = list(body.sports)
    if body.custom_sport and body.custom_sport not in sports:
        sports = [*sports, body.custom_sport]

    # Build goal JSONB payload (immutable construction)
    goal_payload: dict = {
        "goals": list(body.goals),
        "custom_goal": body.custom_goal,
        "wearable": body.wearable,
    }

    # Fetch current meta to merge available_days without clobbering other fields
    try:
        meta_result = (
            await supabase.table("profiles")
            .select("meta")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        existing_meta: dict = (meta_result.data or {}).get("meta") or {}
    except Exception:
        logger.warning("Could not fetch existing meta for user %s — using empty dict", user_id)
        existing_meta = {}

    updated_meta = {**existing_meta, "available_days": list(body.available_days)}

    update_payload: dict = {
        "sports": sports,
        "goal": goal_payload,
        "meta": updated_meta,
        "wearable": body.wearable,
    }

    try:
        await (
            supabase.table("profiles")
            .update(update_payload)
            .eq("id", user_id)
            .execute()
        )
        logger.info("Profile updated for user %s (sports=%s)", user_id, sports)
    except Exception:
        logger.exception("Failed to update profile for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profil konnte nicht gespeichert werden. Bitte versuche es erneut.",
        )


async def _trigger_onboarding_agent(user_id: str) -> None:
    """Fire-and-forget: start a background agent session with context='onboarding'.

    The agent will:
      1. Read the freshly-saved profile data
      2. Create session schemas, metrics, and evaluation criteria
      3. Build and persist the initial training plan
      4. Call complete_onboarding() tool (sets onboarding_complete = true)

    This mirrors the pattern in chat.py where AsyncAgentLoop is instantiated
    per-request. The session is started with a synthetic kick-off message so
    the agent has context to work from without waiting for a user message.

    TODO: When the agent fully implements context="onboarding" handling, wire in:
        loop = AsyncAgentLoop(user_model=user_model, context="onboarding")
        loop.start_session()
        await loop.process_message_sse(
            "Erstelle den initialen Trainingsplan für den neuen Nutzer.", emit_fn
        )
    """
    try:
        from src.db.user_model_db import UserModelDB
        from src.agent.agent_loop import AsyncAgentLoop

        user_model = await asyncio.to_thread(UserModelDB.load_or_create, user_id)
        loop = AsyncAgentLoop(user_model=user_model, context="onboarding")
        loop.start_session()

        kick_off_message = (
            "Erstelle den initialen Trainingsplan für den neuen Nutzer. "
            "Lies das Profil, lege Metriken und Evaluationskriterien an und "
            "speichere den fertigen Plan. Rufe abschließend complete_onboarding() auf."
        )

        async def _noop_emit(event_type: str, data: dict) -> None:  # noqa: ARG001
            pass

        await loop.process_message_sse(kick_off_message, _noop_emit)
        logger.info("Onboarding agent session completed for user %s", user_id)
    except Exception:
        # Background task — never propagate to the HTTP response.
        logger.exception("Onboarding agent session failed for user %s", user_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/parse-voice",
    response_model=ParseVoiceResponse,
    summary="Parse voice transcript into structured tags",
)
@limiter.limit("10/hour")
async def parse_voice(
    request: Request,  # Required by slowapi for IP extraction
    body: ParseVoiceRequest,
) -> ParseVoiceResponse:
    """Publicly accessible endpoint — no auth required.

    Accepts a German voice transcript and the current onboarding step, calls
    Gemini Flash Lite to extract structured tags, and returns them as a JSON
    list. Rate-limited to 10 requests per IP per hour to prevent abuse.
    """
    logger.info(
        "parse-voice request: step=%s text_len=%d ip=%s",
        body.step,
        len(body.text),
        request.client.host if request.client else "unknown",
    )

    if not body.text.strip():
        return ParseVoiceResponse(items=[])

    result = await _call_parse_llm(body.text, body.step)

    return ParseVoiceResponse(
        items=result.get("items", []),
        structured=result.get("structured"),
    )


@router.post(
    "/setup",
    response_model=OnboardingSetupResponse,
    summary="Save onboarding profile data after account creation",
)
async def setup_onboarding(
    body: OnboardingSetupRequest,
    user_id: Annotated[str, Depends(get_user_id)],
) -> OnboardingSetupResponse:
    """Auth-required endpoint — called immediately after Supabase signUp.

    Persists the collected onboarding data to the ``profiles`` table, then
    triggers a background agent session (context='onboarding') that builds
    the user's initial training plan. Returns immediately — plan creation
    happens asynchronously.
    """
    logger.info(
        "onboarding/setup: user=%s sports=%s goals=%s days=%s wearable=%s",
        user_id,
        body.sports,
        body.goals,
        body.available_days,
        body.wearable,
    )

    await _update_profile(user_id, body)

    # Fire background agent session — does not block the HTTP response.
    asyncio.create_task(
        _trigger_onboarding_agent(user_id),
        name=f"onboarding_agent_{user_id}",
    )

    return OnboardingSetupResponse(status="ok", message="Setup gespeichert")
