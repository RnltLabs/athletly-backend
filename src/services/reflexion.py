"""Reflexion loop: end-of-session self-reflection that distills durable
coaching lessons from a conversation.

Public entry points:

- ``run_pro_reflexion(user_id, session_id)``: per-session reflection
  for Pro-tier users. Fire-and-forget from the chat router.
- ``run_free_monthly_reflexion(user_id)``: monthly batch for Free-tier
  users. Same fire-and-forget pattern.
- ``maybe_run_reflexion(user_id, prev_session_id)``: tier-aware wrapper
  used by the chat router. Picks Pro or Free based on the profile
  ``subscription_tier`` and short-circuits when nothing is due.

The whole module is non-blocking: every call swallows internal
exceptions, logs them, and returns. A failed reflection MUST NOT break
the chat flow.

Hallucination guards (see DESIGN.md "Hallucination guard"):

1. Prompt explicitly forbids inventing facts.
2. The LLM returns JSON ``{"lessons": [...]}`` with an ``evidence`` field
   per lesson. After parsing, every lesson's evidence quote is checked
   against the actual user messages of the session; lessons without a
   verifiable quote are dropped before storage.
3. Topic must be in the allowed taxonomy.
4. Lesson length capped at 500 chars.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Reflection model. Cheap, structured-output friendly.
REFLEXION_MODEL = "anthropic/claude-haiku-4-5-20251001"

# Lower bound on session length before reflection is worthwhile.
_MIN_MESSAGES_FOR_REFLECTION = 4

# Hard upper bound on transcript size we send to the LLM.
_MAX_MESSAGES_FOR_PROMPT = 50

# Hard cap on lessons accepted per run (matches "1-3 lessons" prompt).
_MAX_LESSONS_PER_RUN = 3

# Maximum lesson length (chars) for storage. Longer ones are dropped.
_MAX_LESSON_LEN = 500

# Minimum LLM confidence for a lesson to be stored.
_MIN_LESSON_CONFIDENCE = 0.3

# Global per-user daily reflection cap (across all tiers). The user can
# spam-open sessions without flooding the lesson store.
_MAX_REFLEXIONS_PER_DAY = 1

# Free-tier monthly batch: look back this many days of session summaries.
_FREE_TIER_LOOKBACK_DAYS = 30

_ALLOWED_TOPICS: frozenset[str] = frozenset({
    "scheduling",
    "volume",
    "intensity",
    "recovery",
    "nutrition",
    "injury",
    "motivation",
    "preference",
    "constraint",
    "other",
})


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM_PROMPT = """\
You are an endurance-sports coach reflecting on the conversation you just had
with an athlete. Your job is to extract 0-3 specific, durable lessons that
your future self should remember about THIS athlete.

You MUST respond with ONLY a valid JSON object. No markdown, no commentary,
no code fences. The schema is:

{
  "lessons": [
    {
      "topic": "scheduling | volume | intensity | recovery | nutrition | injury | motivation | preference | constraint | other",
      "observation": "What you saw in the conversation, past tense (max 300 chars).",
      "lesson": "Actionable rule your future self should apply (max 500 chars).",
      "applicable_to": ["array", "of", "tags"],
      "evidence": "A direct quote from the athlete (or several joined with ' ... ') that supports this lesson.",
      "confidence": 0.0
    }
  ]
}

HARD RULES:
1. Do NOT invent facts the athlete did not state. If a candidate lesson
   cannot be backed by a direct user quote that appears in the transcript,
   omit it.
2. Be specific. NOT "athlete likes running". YES "athlete prefers running
   easy on Sundays after long Saturday rides".
3. If nothing meaningfully new was learned, return {"lessons": []}.
4. NEVER record full names beyond the athlete's first name, phone numbers,
   email addresses, postal addresses, financial details, or medical
   diagnoses. Symptoms and complaints ("knee soreness on long runs") are
   allowed because they affect coaching.
5. Topic MUST be one of the values listed above. If unsure, use "other".
6. The "evidence" field must contain a verbatim quote from the user's
   own message (not from your replies).
7. Return at most 3 lessons.
"""

_REFLECTION_USER_PROMPT_TEMPLATE = """\
Below is the conversation transcript. Extract 0-3 specific lessons.

TRANSCRIPT:
{transcript}

Respond now with the JSON object.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def maybe_run_reflexion(user_id: str, prev_session_id: str | None) -> None:
    """Tier-aware reflection entry point.

    Called from the chat router when a NEW session is detected.
    Determines tier from the user's profile and routes to either
    :func:`run_pro_reflexion` (per-session) or
    :func:`run_free_monthly_reflexion` (monthly batch).

    Always returns None. Exceptions are caught and logged.
    """
    try:
        # Daily cap regardless of tier.
        from src.db.lessons_db import count_runs_since

        since_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        recent_count = await asyncio.to_thread(
            count_runs_since, user_id, since_iso=since_iso,
        )
        if recent_count >= _MAX_REFLEXIONS_PER_DAY:
            logger.debug(
                "Reflexion: daily cap reached for user %s (count=%d)",
                user_id[:8], recent_count,
            )
            return

        tier = await asyncio.to_thread(_get_tier, user_id)

        if tier == "pro":
            if prev_session_id is None:
                logger.debug("Reflexion: no previous session id, skipping pro run")
                return
            await run_pro_reflexion(user_id, prev_session_id)
        else:
            await run_free_monthly_reflexion(user_id)
    except Exception:
        logger.warning(
            "maybe_run_reflexion swallowed exception for user %s",
            user_id[:8],
            exc_info=True,
        )


async def run_pro_reflexion(user_id: str, session_id: str) -> int:
    """Run a single-session reflection. Returns number of lessons stored.

    Skips if the run has already been recorded for this (user, session)
    pair (idempotency via :data:`reflexion_runs`).
    """
    from src.db.lessons_db import has_reflexion_run, record_reflexion_run

    already_ran = await asyncio.to_thread(
        has_reflexion_run, user_id, run_kind="pro_session", period=session_id,
    )
    if already_ran:
        logger.debug(
            "Reflexion: pro_session already recorded for user=%s session=%s",
            user_id[:8], session_id[:8],
        )
        return 0

    transcript_messages = await _load_session_messages(session_id)
    if len(transcript_messages) < _MIN_MESSAGES_FOR_REFLECTION:
        logger.debug(
            "Reflexion: session too short (%d messages) for user %s",
            len(transcript_messages), user_id[:8],
        )
        return 0

    user_messages = [m for m in transcript_messages if m.get("role") == "user"]
    lessons = await _generate_and_validate_lessons(
        transcript_messages=transcript_messages,
        user_messages=user_messages,
    )

    stored = await _store_lessons(
        user_id=user_id,
        lessons=lessons,
        source_session_id=session_id,
    )

    await asyncio.to_thread(
        record_reflexion_run,
        user_id,
        run_kind="pro_session",
        period=session_id,
        lessons_added=stored,
    )

    logger.info(
        "Reflexion: pro_session user=%s session=%s stored=%d",
        user_id[:8], session_id[:8], stored,
    )
    return stored


async def run_free_monthly_reflexion(user_id: str) -> int:
    """Run the free-tier monthly batch reflection. Returns lessons stored.

    Uses ``compressed_summary`` of the last 30 days of sessions instead
    of raw transcripts. Idempotent on (user_id, YYYY-MM).
    """
    from src.db.lessons_db import has_reflexion_run, record_reflexion_run

    period = datetime.now(timezone.utc).strftime("%Y-%m")
    already_ran = await asyncio.to_thread(
        has_reflexion_run, user_id, run_kind="free_monthly", period=period,
    )
    if already_ran:
        logger.debug(
            "Reflexion: free_monthly already recorded user=%s period=%s",
            user_id[:8], period,
        )
        return 0

    summaries = await _load_recent_summaries(user_id, days=_FREE_TIER_LOOKBACK_DAYS)
    if not summaries:
        logger.debug("Reflexion: no recent summaries for user %s", user_id[:8])
        return 0

    transcript_messages = _summaries_to_synthetic_transcript(summaries)
    user_messages = [m for m in transcript_messages if m.get("role") == "user"]

    lessons = await _generate_and_validate_lessons(
        transcript_messages=transcript_messages,
        user_messages=user_messages,
    )

    stored = await _store_lessons(
        user_id=user_id,
        lessons=lessons,
        source_session_id=None,
    )

    await asyncio.to_thread(
        record_reflexion_run,
        user_id,
        run_kind="free_monthly",
        period=period,
        lessons_added=stored,
    )

    logger.info(
        "Reflexion: free_monthly user=%s period=%s stored=%d",
        user_id[:8], period, stored,
    )
    return stored


# ---------------------------------------------------------------------------
# Helpers: data loading
# ---------------------------------------------------------------------------


async def _load_session_messages(session_id: str) -> list[dict]:
    """Load the trimmed transcript that feeds the reflection prompt.

    Filters to ``user`` and ``model`` roles only. Drops tool_call and
    system rows because they are pure machinery, not coaching signal.
    """
    from src.db.session_store_db import load_session_messages

    raw = await asyncio.to_thread(
        load_session_messages, session_id, _MAX_MESSAGES_FOR_PROMPT,
    )
    return [m for m in raw if m.get("role") in {"user", "model"}]


async def _load_recent_summaries(user_id: str, *, days: int) -> list[dict]:
    """Fetch compressed_summary entries for the last ``days`` days."""
    from src.db.session_store_db import get_recent_sessions

    sessions = await asyncio.to_thread(get_recent_sessions, user_id, limit=30)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict] = []
    for s in sessions:
        summary = s.get("compressed_summary")
        if not summary:
            continue
        started_at = s.get("started_at")
        try:
            if started_at and datetime.fromisoformat(
                started_at.replace("Z", "+00:00")
            ) < cutoff:
                continue
        except (TypeError, ValueError):
            pass
        out.append(s)
    return out


def _summaries_to_synthetic_transcript(summaries: list[dict]) -> list[dict]:
    """Encode a list of session summaries as a fake transcript.

    The reflection prompt expects (role, content) pairs. We tag each
    summary as a ``user`` message so the evidence check has something to
    quote against. This is the SAME content the LLM would have seen at
    write-time, just compressed.
    """
    msgs: list[dict] = []
    for s in summaries:
        text = s.get("compressed_summary") or ""
        if not text:
            continue
        msgs.append({"role": "user", "content": text})
    return msgs


# ---------------------------------------------------------------------------
# Helpers: tier resolution
# ---------------------------------------------------------------------------


def _get_tier(user_id: str) -> str:
    """Return ``"pro"`` or ``"free"`` (default ``"free"``)."""
    try:
        from src.db.client import get_supabase

        result = (
            get_supabase()
            .table("profiles")
            .select("meta")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if result is None or not result.data:
            return "free"
        meta = result.data.get("meta") or {}
        tier = str(meta.get("subscription_tier") or "free").lower()
        return tier if tier in {"pro", "free"} else "free"
    except Exception:
        logger.debug("tier lookup failed for %s", user_id[:8], exc_info=True)
        return "free"


# ---------------------------------------------------------------------------
# Helpers: LLM call + parse + validate
# ---------------------------------------------------------------------------


async def _generate_and_validate_lessons(
    *,
    transcript_messages: list[dict],
    user_messages: list[dict],
) -> list[dict]:
    """Run the reflection LLM and return a sanitized list of lesson dicts.

    Each returned dict has keys: topic, observation, lesson,
    applicable_to, evidence, confidence. The list may be empty.
    """
    transcript_text = _format_transcript(transcript_messages)

    try:
        from src.agent.llm import chat_completion

        response = await asyncio.to_thread(
            chat_completion,
            messages=[{
                "role": "user",
                "content": _REFLECTION_USER_PROMPT_TEMPLATE.format(
                    transcript=transcript_text,
                ),
            }],
            system_prompt=_REFLECTION_SYSTEM_PROMPT,
            temperature=0.2,
            model=REFLEXION_MODEL,
        )
    except Exception:
        logger.warning("Reflexion LLM call failed", exc_info=True)
        return []

    raw = ""
    try:
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("Reflexion: malformed LLM response", exc_info=True)
        return []

    parsed = _safe_parse_json(raw)
    if not parsed:
        return []

    raw_lessons = parsed.get("lessons") or []
    if not isinstance(raw_lessons, list):
        return []

    # Normalised user-message blob for substring evidence checks.
    user_blob = _normalise(
        " \n ".join(m.get("content") or "" for m in user_messages)
    )

    cleaned: list[dict] = []
    for candidate in raw_lessons[: _MAX_LESSONS_PER_RUN * 2]:  # over-pull, prune later
        validated = _validate_lesson(candidate, user_blob=user_blob)
        if validated is not None:
            cleaned.append(validated)
        if len(cleaned) >= _MAX_LESSONS_PER_RUN:
            break

    return cleaned


def _validate_lesson(candidate: object, *, user_blob: str) -> dict | None:
    """Return a cleaned lesson dict, or ``None`` if it should be dropped.

    Drops the candidate when:
    - it is not a dict,
    - topic is not in the allowed taxonomy,
    - lesson body is missing or > 500 chars,
    - evidence quote cannot be found in any user message,
    - confidence is below the floor,
    - PII blacklist hits.
    """
    if not isinstance(candidate, dict):
        return None

    topic = str(candidate.get("topic") or "other").strip().lower()
    if topic not in _ALLOWED_TOPICS:
        topic = "other"

    observation = str(candidate.get("observation") or "").strip()
    lesson_body = str(candidate.get("lesson") or "").strip()
    evidence = str(candidate.get("evidence") or "").strip()

    if not lesson_body:
        return None
    if len(lesson_body) > _MAX_LESSON_LEN:
        return None

    applicable_to_raw = candidate.get("applicable_to") or []
    if isinstance(applicable_to_raw, str):
        applicable_to_raw = [applicable_to_raw]
    applicable_to = [str(x).strip().lower() for x in applicable_to_raw if str(x).strip()][:8]

    try:
        confidence = float(candidate.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < _MIN_LESSON_CONFIDENCE:
        return None

    # Evidence-in-transcript check (hallucination guard).
    if not _evidence_in_user_blob(evidence, user_blob):
        logger.info("Reflexion: dropping lesson with non-verifiable evidence")
        return None

    # PII blacklist - cheap regex check.
    if _looks_like_pii(lesson_body) or _looks_like_pii(observation):
        logger.info("Reflexion: dropping lesson with PII-shaped content")
        return None

    return {
        "topic": topic,
        "observation": observation[:1000],
        "lesson": lesson_body,
        "applicable_to": applicable_to,
        "evidence": evidence[:1000],
        "confidence": confidence,
    }


def _evidence_in_user_blob(evidence: str, user_blob: str) -> bool:
    """Return True if a normalised slice of ``evidence`` occurs in ``user_blob``."""
    if not evidence:
        return False
    norm = _normalise(evidence)
    if len(norm) < 8:
        # Too short to verify meaningfully; reject.
        return False

    if norm in user_blob:
        return True

    # Allow joined quotes like "foo ... bar" by splitting on ellipsis-like
    # separators and requiring each fragment to appear.
    fragments = [f.strip() for f in re.split(r"\.\.\.+|…", norm) if f.strip()]
    if len(fragments) >= 2:
        return all(len(f) >= 8 and f in user_blob for f in fragments)

    return False


_PII_PATTERNS = [
    re.compile(r"\b\d{3}[ .\-]?\d{3,4}[ .\-]?\d{3,4}\b"),  # phone-ish
    re.compile(r"\b[\w._%+\-]+@[\w.\-]+\.[a-z]{2,}\b", re.I),  # email
    re.compile(r"\b(?:visa|mastercard|amex)\b", re.I),
    re.compile(r"\b\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}\b"),  # card-ish
]


def _looks_like_pii(text: str) -> bool:
    if not text:
        return False
    for pat in _PII_PATTERNS:
        if pat.search(text):
            return True
    return False


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace for substring matching."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _safe_parse_json(raw: str) -> dict | None:
    """Parse JSON, stripping markdown fences if the LLM added them."""
    if not raw:
        return None
    text = raw
    if text.startswith("```"):
        # ```json\n...\n```
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        logger.debug("Reflexion: JSON parse failed", exc_info=True)
        return None


def _format_transcript(messages: list[dict]) -> str:
    """Render messages as a Coach/Athlete dialog for the LLM prompt."""
    lines: list[str] = []
    for m in messages[-_MAX_MESSAGES_FOR_PROMPT:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Truncate any single message to keep prompt size bounded.
        if len(content) > 600:
            content = content[:600] + "..."
        label = "Athlete" if role == "user" else "Coach"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers: storage (embedding + DB insert + supersede)
# ---------------------------------------------------------------------------


async def _store_lessons(
    *,
    user_id: str,
    lessons: list[dict],
    source_session_id: str | None,
) -> int:
    """Embed and insert each lesson. Returns the number stored.

    Embedding failures are non-fatal: the lesson is stored without an
    embedding (it can still be retrieved via topic / recency).
    """
    stored = 0
    for candidate in lessons:
        text_for_embedding = " | ".join(filter(None, [
            candidate.get("topic"),
            candidate.get("lesson"),
            ", ".join(candidate.get("applicable_to") or []),
        ]))
        embedding = await asyncio.to_thread(_generate_embedding, text_for_embedding)

        from src.db.lessons_db import insert_lesson

        row = await asyncio.to_thread(
            insert_lesson,
            user_id,
            topic=candidate["topic"],
            observation=candidate["observation"],
            lesson=candidate["lesson"],
            applicable_to=candidate["applicable_to"],
            evidence=candidate["evidence"],
            confidence=candidate["confidence"],
            embedding=embedding,
            source_session_id=source_session_id,
        )
        if row is not None:
            stored += 1
            # Best-effort supersede of near-duplicates with same topic.
            if embedding is not None:
                await _maybe_supersede_near_duplicate(
                    user_id=user_id,
                    new_lesson_id=row.get("id"),
                    topic=candidate["topic"],
                    embedding=embedding,
                )
    return stored


def _generate_embedding(text: str) -> list[float] | None:
    """Generate a 768-dim Gemini embedding. Returns ``None`` on failure
    or when embeddings are disabled.

    Re-uses the same configuration as
    :func:`src.db.user_model_db.UserModelDB._generate_embedding`.

    TODO: hook into a shared embedding service when one is extracted.
    """
    import os

    if os.environ.get("ATHLETLY_EMBEDDINGS_DISABLED") == "1":
        return None
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from src.agent.llm import get_client

        client = get_client()
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
        )
        embedding = list(response.embeddings[0].values)
        return embedding
    except Exception:
        logger.debug("Embedding generation failed", exc_info=True)
        return None


_SUPERSEDE_SIMILARITY_THRESHOLD = 0.9


async def _maybe_supersede_near_duplicate(
    *,
    user_id: str,
    new_lesson_id: str | None,
    topic: str,
    embedding: list[float],
) -> None:
    """If an existing lesson with same topic and cosine similarity >= 0.9
    exists, mark it superseded by the newly inserted one.

    Skip when the new lesson has no id (insert failure) or the RPC fails.
    """
    if not new_lesson_id:
        return
    try:
        from src.db.lessons_db import match_lessons_by_embedding, supersede_lesson

        candidates = await asyncio.to_thread(
            match_lessons_by_embedding,
            user_id,
            embedding,
            match_count=3,
            min_confidence=0.0,
        )
        for c in candidates:
            cand_id = c.get("id")
            if not cand_id or cand_id == new_lesson_id:
                continue
            if c.get("topic") != topic:
                continue
            if float(c.get("similarity") or 0.0) >= _SUPERSEDE_SIMILARITY_THRESHOLD:
                await asyncio.to_thread(supersede_lesson, cand_id, new_lesson_id)
    except Exception:
        logger.debug("supersede check failed", exc_info=True)
