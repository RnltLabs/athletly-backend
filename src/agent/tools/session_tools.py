"""Session tools — search over historical session summaries.

Provides the agent with the ability to recall context from past sessions
without loading full message transcripts into the context window.

Tool registered:
    - search_session_history: ILIKE text search over sessions.summary / tags.
"""

import logging

from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)

_MAX_QUERY_LENGTH = 500  # Maximum characters allowed in a session search query


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_session_tools(registry: ToolRegistry, user_id: str) -> None:
    """Register session search tools bound to the given user_id."""

    def search_session_history(query: str, limit: int = 10) -> dict:
        """Search past session summaries for relevant context."""
        if not query.strip():
            return {"error": "query must not be empty", "results": []}
        if len(query) > _MAX_QUERY_LENGTH:
            return {
                "error": f"query too long (max {_MAX_QUERY_LENGTH} characters)",
                "results": [],
            }

        clamped_limit = max(1, min(limit, 50))

        try:
            results = _search_sessions(user_id, query.strip(), clamped_limit)
            return {"results": results, "count": len(results)}
        except Exception as exc:
            logger.error("search_session_history failed: %s", exc)
            return {"error": str(exc), "results": []}

    registry.register(Tool(
        name="search_session_history",
        description=(
            "Search the history of past coaching sessions by keyword. "
            "Use this to recall what was discussed previously — e.g. past goals, "
            "injuries, plans, or athlete preferences mentioned in earlier sessions. "
            "Returns a list of matching session summaries with dates and tags."
        ),
        handler=search_session_history,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for in session summaries",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to return (1-50, default 10)",
                },
            },
            "required": ["query"],
        },
        category="memory",
    ))


# ---------------------------------------------------------------------------
# Search implementation
# ---------------------------------------------------------------------------


def _search_sessions(user_id: str, query: str, limit: int) -> list[dict]:
    """Query Supabase ``sessions`` table with ILIKE search.

    Searches the ``compressed_summary`` column (and ``tags`` when present)
    for the given query string.  Results are ordered newest-first.

    Args:
        user_id: Owning user UUID.
        query: Search term (will be wrapped in ``%…%``).
        limit: Maximum rows to return.

    Returns:
        List of result dicts with keys: session_id, date, summary, tags.
    """
    from src.db.client import get_supabase

    db = get_supabase()

    pattern = f"%{query}%"

    # Search compressed_summary via ILIKE (PostgREST ilike filter).
    result = (
        db.table("sessions")
        .select("id, started_at, compressed_summary, tags")
        .eq("user_id", user_id)
        .ilike("compressed_summary", pattern)
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )

    rows: list[dict] = result.data or []
    return [_format_session_row(row) for row in rows]


def _format_session_row(row: dict) -> dict:
    """Convert a raw Supabase session row into the tool result shape.

    Always returns a new dict — does not mutate the input.
    """
    return {
        "session_id": row.get("id", ""),
        "date": _extract_date(row.get("started_at", "")),
        "summary": (row.get("compressed_summary") or "")[:500],
        "tags": row.get("tags") or [],
    }


def _extract_date(iso_timestamp: str) -> str:
    """Return just the date portion of an ISO timestamp string."""
    if not iso_timestamp:
        return ""
    return iso_timestamp[:10]  # "YYYY-MM-DD"
