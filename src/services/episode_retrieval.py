"""Episode Replay - semantic retrieval of past episodes for agent context.

The agent's per-turn runtime context is augmented with a short list of past
episodes that are semantically similar to the athlete's current message.
Implements the design from ``DESIGN.md``:

- Embed the query text once per turn (Gemini or OpenAI, configurable).
- Cosine-similarity lookup via the ``match_episodes`` pgvector RPC.
- Rerank with ``0.7 * similarity + 0.2 * recency + 0.1 * utility``.
- Format as a labelled, date-prefixed bullet list with anti-confabulation
  footer.
- Hard cap at ``EPISODE_REPLAY_TOKEN_BUDGET`` tokens (default 800).
- Graceful degradation: any failure returns ``""`` so the runtime context
  simply omits the section.

Usage::

    from src.services.episode_retrieval import build_replay_block

    block = build_replay_block(user_id, user_message="how should I taper?")
    if block:
        sections.append(block)
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants - overridable via env / settings
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 3
MAX_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_TOKEN_BUDGET = 800

# Over-fetch multiplier - we ask the DB for more rows than we keep so the
# reranker has room to reorder.
_OVERFETCH_FACTOR = 4

# Episode types considered for replay. Hand-tuned: skip purely structural
# rollups like ``monthly_review`` only if the user has many of them. For now
# keep all coaching-relevant types.
_DEFAULT_EPISODE_TYPES = (
    "weekly_reflection",
    "monthly_review",
    "coaching_insight",
)

# Minimum length of a user message to bother retrieving. Greetings like
# "hi" / "ok" rarely yield useful matches and would waste embedding spend.
_MIN_QUERY_CHARS = 8

# Recency decay half-life proxy. 90-day characteristic time produces
# recency=1.0 today, ~0.45 at 90 days, ~0.20 at 180 days.
_RECENCY_DECAY_DAYS = 90

# Scoring weights. Sum must equal 1.0.
_W_SIMILARITY = 0.7
_W_RECENCY = 0.2
_W_UTILITY = 0.1

# Conservative token estimation - ~4 chars per token works for English /
# German bullet text. Used only to enforce the budget cap.
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Trigger gating
# ---------------------------------------------------------------------------


def should_retrieve(user_message: str | None) -> bool:
    """Decide whether to run episode replay for *user_message*."""
    if os.environ.get("ATHLETLY_EPISODE_REPLAY_DISABLED") == "1":
        return False
    if not user_message or not user_message.strip():
        return False
    if len(user_message.strip()) < _MIN_QUERY_CHARS:
        return False
    return True


# ---------------------------------------------------------------------------
# Scoring + reranking
# ---------------------------------------------------------------------------


def _parse_date(value) -> date | None:
    """Coerce ISO date / datetime strings (or date / datetime objects) into ``date``."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            # ``fromisoformat`` handles both date and full ISO timestamps.
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
    return None


def _recency_score(episode: dict, today: date) -> float:
    """Compute exponential-decay recency score in [0, 1]."""
    ref = (
        _parse_date(episode.get("period_end"))
        or _parse_date(episode.get("period_start"))
        or _parse_date(episode.get("created_at"))
    )
    if ref is None:
        return 0.0
    days = max(0, (today - ref).days)
    return math.exp(-days / _RECENCY_DECAY_DAYS)


def _rerank(
    candidates: list[dict],
    today: date,
) -> list[dict]:
    """Sort *candidates* by combined relevance score, descending.

    Mutates each candidate dict to add ``final_score`` and ``recency_score``
    for observability / tests.
    """
    for c in candidates:
        sim = float(c.get("similarity", 0.0) or 0.0)
        recency = _recency_score(c, today)
        utility = float(c.get("utility", 0.0) or 0.0)
        # Clamp inputs to [0, 1] to defend against bad data.
        sim = max(0.0, min(1.0, sim))
        utility = max(0.0, min(1.0, utility))
        c["recency_score"] = recency
        c["final_score"] = (
            _W_SIMILARITY * sim
            + _W_RECENCY * recency
            + _W_UTILITY * utility
        )

    candidates.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _episode_date_label(episode: dict) -> str:
    """Best-effort YYYY-MM-DD prefix for the bullet."""
    d = (
        _parse_date(episode.get("period_end"))
        or _parse_date(episode.get("period_start"))
        or _parse_date(episode.get("created_at"))
    )
    return d.isoformat() if d else "unknown date"


def _first_sentence(text: str, max_chars: int = 220) -> str:
    """Return the first sentence-ish chunk of *text*, capped to *max_chars*."""
    if not text:
        return ""
    text = text.strip()
    # Try to cut at the first ``. `` boundary.
    cut = text.find(". ")
    if 20 <= cut <= max_chars:
        return text[: cut + 1].strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _compose_bullet(episode: dict) -> str:
    """Render one episode as a markdown bullet line."""
    date_label = _episode_date_label(episode)
    summary = _first_sentence(episode.get("summary") or "")

    insights = episode.get("insights") or []
    insight_text = ""
    if insights:
        first = str(insights[0]).strip()
        if first:
            insight_text = f" Insight: {first}"

    body = summary + insight_text
    if not body:
        body = "(no detail)"
    return f"- {date_label}: {body}"


def _format_block(
    episodes: list[dict],
    token_budget: int,
) -> str:
    """Format the final markdown block, enforcing the token budget."""
    if not episodes:
        return ""

    header = f"# Past Insights ({len(episodes)} relevant episodes from your history)"
    footer = (
        "Use these to inform your reasoning but do NOT claim you just "
        "observed them. Reference them only if the athlete's current "
        "question relates."
    )

    char_budget = token_budget * _CHARS_PER_TOKEN
    fixed_chars = len(header) + len(footer) + 4  # blank lines

    bullet_chars_remaining = char_budget - fixed_chars
    bullets: list[str] = []
    for ep in episodes:
        line = _compose_bullet(ep)
        if len(line) + 1 > bullet_chars_remaining:
            # Try truncated form if at least one bullet already fits.
            if bullets and bullet_chars_remaining > 40:
                truncated = line[: bullet_chars_remaining - 4].rstrip() + "..."
                bullets.append(truncated)
            break
        bullets.append(line)
        bullet_chars_remaining -= len(line) + 1

    if not bullets:
        return ""

    return "\n".join([header, *bullets, "", footer])


# ---------------------------------------------------------------------------
# RPC call - retrieves candidate episodes from pgvector
# ---------------------------------------------------------------------------


def _fetch_candidates(
    user_id: str,
    embedding: list[float],
    match_count: int,
    min_similarity: float,
    episode_types: tuple[str, ...] | None,
) -> list[dict]:
    """Invoke the ``match_episodes`` RPC. Returns ``[]`` on any error."""
    try:
        from src.db.client import get_supabase

        params = {
            "p_user_id": user_id,
            "p_embedding": embedding,
            "p_match_count": match_count,
            "p_min_similarity": min_similarity,
            "p_episode_types": list(episode_types) if episode_types else None,
        }
        result = get_supabase().rpc("match_episodes", params).execute()
        return list(result.data or [])
    except Exception:
        logger.warning(
            "match_episodes RPC failed for user=%s", user_id, exc_info=True
        )
        return []


def _touch_last_replayed(episode_ids: list[str]) -> None:
    """Best-effort telemetry: stamp ``last_replayed_at`` on returned episodes.

    Failures are swallowed - this is purely observational data.
    """
    if not episode_ids:
        return
    try:
        from src.db.client import get_supabase

        now_iso = datetime.now(timezone.utc).isoformat()
        get_supabase().table("episodes").update(
            {"last_replayed_at": now_iso}
        ).in_("id", episode_ids).execute()
    except Exception:
        logger.debug("Failed to update last_replayed_at", exc_info=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_relevant_episodes(
    user_id: str,
    query_text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    episode_types: tuple[str, ...] | None = _DEFAULT_EPISODE_TYPES,
    today: date | None = None,
) -> list[dict]:
    """Return the top-K most relevant episodes for *query_text*.

    Args:
        user_id: UUID of the athlete - hard-required for RLS scoping.
        query_text: The live user message to retrieve against.
        top_k: How many episodes to keep after reranking. Capped at ``MAX_TOP_K``.
        similarity_threshold: Minimum cosine similarity to consider.
        episode_types: Restrict to these episode types. ``None`` means all.
        today: Optional reference date for recency. Defaults to ``date.today()``.

    Returns:
        A list of episode dicts ordered by combined relevance score.
        Returns ``[]`` when no candidate clears the threshold, when the
        embedding step fails, or when retrieval is disabled.
    """
    if not user_id or not query_text:
        return []

    top_k = max(1, min(top_k, MAX_TOP_K))
    ref_today = today or date.today()

    from src.services.embeddings import embed_text

    embedding = embed_text(query_text)
    if embedding is None:
        return []

    overfetch = max(top_k * _OVERFETCH_FACTOR, top_k)
    candidates = _fetch_candidates(
        user_id=user_id,
        embedding=embedding,
        match_count=overfetch,
        min_similarity=similarity_threshold,
        episode_types=episode_types,
    )

    if not candidates:
        return []

    ranked = _rerank(candidates, ref_today)
    selected = ranked[:top_k]

    # Best-effort telemetry update.
    try:
        _touch_last_replayed([str(e["id"]) for e in selected if e.get("id")])
    except Exception:
        pass

    return selected


def build_replay_block(
    user_id: str | None,
    user_message: str | None,
    *,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    token_budget: int | None = None,
    today: date | None = None,
) -> str:
    """End-to-end helper for ``build_runtime_context``.

    Returns the formatted markdown block, or ``""`` when no replay should
    fire (disabled, empty message, no matches, embedding failure, etc.).

    Settings are read from ``src.config.get_settings`` when not passed
    explicitly. Tests can override every knob.
    """
    if not user_id:
        return ""
    if not should_retrieve(user_message):
        return ""

    # Resolve effective parameters with settings fallback.
    resolved_top_k = top_k if top_k is not None else _setting("episode_replay_top_k", DEFAULT_TOP_K)
    resolved_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else _setting("episode_replay_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    )
    resolved_budget = (
        token_budget
        if token_budget is not None
        else _setting("episode_replay_token_budget", DEFAULT_TOKEN_BUDGET)
    )

    episodes = retrieve_relevant_episodes(
        user_id=user_id,
        query_text=user_message or "",
        top_k=int(resolved_top_k),
        similarity_threshold=float(resolved_threshold),
        today=today,
    )

    if not episodes:
        return ""

    return _format_block(episodes, token_budget=int(resolved_budget))


def _setting(name: str, default):
    """Look up a Settings attribute, returning *default* on any failure."""
    try:
        from src.config import get_settings

        return getattr(get_settings(), name, default)
    except Exception:
        return default
