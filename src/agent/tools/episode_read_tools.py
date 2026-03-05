"""Episode read tools -- read-only access to episodic memory.

Provides the agent with the ability to browse weekly reflections and
monthly reviews without loading them through the broader memory tools.

Tool registered:
    - get_episodes: Read episodes filtered by time range, type, or keyword.
"""

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)

_MAX_KEYWORD_LENGTH = 300


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_episode_read_tools(registry: ToolRegistry, user_model) -> None:
    """Register episode read tools bound to the given user_model."""
    _settings = get_settings()

    def _resolve_user_id() -> str:
        if hasattr(user_model, "user_id"):
            return user_model.user_id
        return _settings.agenticsports_user_id

    def get_episodes(
        episode_type: str = "",
        period_start: str = "",
        period_end: str = "",
        keyword: str = "",
        limit: int = 20,
    ) -> dict:
        """Read episodic memory entries with optional filters.

        Returns a list of episodes (weekly reflections, monthly reviews, etc.)
        filtered by the provided criteria. At least one filter should be used
        to avoid returning too many results.
        """
        if not _settings.use_supabase:
            return {"error": "Supabase not configured", "episodes": []}

        user_id = _resolve_user_id()
        clamped_limit = max(1, min(limit, 100))

        if keyword and len(keyword) > _MAX_KEYWORD_LENGTH:
            return {
                "error": f"keyword too long (max {_MAX_KEYWORD_LENGTH} characters)",
                "episodes": [],
            }

        try:
            episodes = _fetch_episodes(
                user_id=user_id,
                episode_type=episode_type.strip() if episode_type else "",
                period_start=period_start.strip() if period_start else "",
                period_end=period_end.strip() if period_end else "",
                keyword=keyword.strip() if keyword else "",
                limit=clamped_limit,
            )
            return {"episodes": episodes, "count": len(episodes)}
        except Exception as exc:
            logger.error("get_episodes failed: %s", exc)
            return {"error": str(exc), "episodes": []}

    registry.register(Tool(
        name="get_episodes",
        description=(
            "Read episodic memory — weekly reflections, monthly reviews, and "
            "coaching insights stored from past training blocks. "
            "Use this to recall patterns, recurring issues, and long-term trends "
            "that inform planning decisions. "
            "Filter by episode_type (e.g. 'weekly_reflection', 'monthly_review'), "
            "date range (period_start / period_end in YYYY-MM-DD), or keyword search."
        ),
        handler=get_episodes,
        parameters={
            "type": "object",
            "properties": {
                "episode_type": {
                    "type": "string",
                    "description": (
                        "Filter by episode type (e.g. 'weekly_reflection', "
                        "'monthly_review', 'coaching_insight'). Empty = all types."
                    ),
                },
                "period_start": {
                    "type": "string",
                    "description": "Start date filter (inclusive, YYYY-MM-DD). Empty = no lower bound.",
                },
                "period_end": {
                    "type": "string",
                    "description": "End date filter (inclusive, YYYY-MM-DD). Empty = no upper bound.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Search keyword applied to episode summary text (ILIKE match).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum episodes to return (1-100, default 20).",
                },
            },
        },
        category="memory",
    ))


# ---------------------------------------------------------------------------
# Query implementation
# ---------------------------------------------------------------------------


def _fetch_episodes(
    user_id: str,
    episode_type: str,
    period_start: str,
    period_end: str,
    keyword: str,
    limit: int,
) -> list[dict]:
    """Query the episodes table with optional filters.

    Builds the Supabase query incrementally based on which filters
    are provided. Results are ordered newest-first.

    Args:
        user_id: Owning user UUID.
        episode_type: Filter by type (empty string = skip).
        period_start: Lower bound for period_start (inclusive).
        period_end: Upper bound for period_end (inclusive).
        keyword: ILIKE search on summary column.
        limit: Maximum rows.

    Returns:
        List of formatted episode dicts.
    """
    from src.db.client import get_supabase

    db = get_supabase()

    query = (
        db.table("episodes")
        .select("id, episode_type, period_start, period_end, summary, insights, created_at")
        .eq("user_id", user_id)
    )

    if episode_type:
        query = query.eq("episode_type", episode_type)

    if period_start:
        query = query.gte("period_start", period_start)

    if period_end:
        query = query.lte("period_end", period_end)

    if keyword:
        query = query.ilike("summary", f"%{keyword}%")

    query = query.order("period_end", desc=True).limit(limit)

    result = query.execute()
    rows: list[dict] = result.data or []
    return [_format_episode(row) for row in rows]


def _format_episode(row: dict) -> dict:
    """Convert a raw episode row to a clean result dict.

    Always returns a new dict -- does not mutate the input.
    """
    return {
        "id": row.get("id", ""),
        "type": row.get("episode_type", ""),
        "period_start": row.get("period_start") or "",
        "period_end": row.get("period_end") or "",
        "summary": (row.get("summary") or "")[:1000],
        "insights": row.get("insights") or [],
        "created_at": _extract_date(row.get("created_at", "")),
    }


def _extract_date(iso_timestamp: str) -> str:
    """Return just the date portion of an ISO timestamp string."""
    if not iso_timestamp:
        return ""
    return iso_timestamp[:10]
