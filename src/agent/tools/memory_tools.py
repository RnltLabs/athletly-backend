"""Memory tools: structured profile + episode storage.

The belief layer has been retired - all free-form athlete facts live in
the athlete journal (see ``journal_tools``). What stays here:

* ``update_profile`` - canonical columns the system queries directly
  (name, sports, goal cache, fitness numbers, training constraints).
* ``store_episode`` - opaque coaching-insight episodes used by the
  reflection pipeline.
* ``consolidate_episodes`` - monthly rollup.
"""

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings


def register_memory_tools(registry: ToolRegistry, user_model):
    """Register the structured profile + episode tools."""
    _settings = get_settings()

    def update_profile(field: str, value) -> dict:
        """Update a field in the athlete's structured profile."""
        import json as _json

        valid_fields = {
            "name", "sports", "goal.event", "goal.target_date", "goal.target_time",
            "fitness.estimated_vo2max", "fitness.threshold_pace_min_km",
            "fitness.weekly_volume_km", "fitness.ftp_watts",
            "constraints.training_days_per_week", "constraints.max_session_minutes",
            "constraints.available_sports",
        }

        if field not in valid_fields:
            return {"error": f"Invalid field: {field}. Valid fields: {sorted(valid_fields)}"}

        # Gemini often sends JSON values as strings -- parse them
        if isinstance(value, str):
            # Try to parse JSON arrays/numbers from string values
            stripped = value.strip()
            if (stripped.startswith("[") and stripped.endswith("]")) or \
               (stripped.startswith("{") and stripped.endswith("}")):
                try:
                    value = _json.loads(stripped)
                except _json.JSONDecodeError:
                    pass
            # Parse numeric strings for numeric fields
            elif field in ("constraints.training_days_per_week", "constraints.max_session_minutes",
                           "fitness.estimated_vo2max", "fitness.weekly_volume_km", "fitness.ftp_watts"):
                try:
                    value = int(value) if "." not in value else float(value)
                except (ValueError, TypeError):
                    pass

        user_model.update_structured_core(field, value)
        user_model.save()

        return {"updated": True, "field": field, "value": value}

    registry.register(Tool(
        name="update_profile",
        description=(
            "Update one canonical profile field (cache for queryability). "
            "Valid fields: name, sports, goal.event, goal.target_date, "
            "goal.target_time, fitness.estimated_vo2max, "
            "fitness.threshold_pace_min_km, fitness.weekly_volume_km, "
            "fitness.ftp_watts, constraints.training_days_per_week, "
            "constraints.max_session_minutes, constraints.available_sports. "
            "Use for hard numeric facts and active sports. Avoid for free-form "
            "info (use update_journal_section) or goal switches (use "
            "update_goal). Invalid fields rejected; writes through immediately."
        ),
        handler=update_profile,
        parameters={
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "The profile field to update (dot notation for nested fields)",
                },
                "value": {
                    "type": "string",
                    "description": "The new value (strings, numbers, or JSON arrays like [\"running\", \"cycling\"])",
                },
            },
            "required": ["field", "value"],
        },
        category="memory",
        display_label="Notiere ein Profil-Detail",
    ))

    def store_episode(summary: str, context: str, learnings: list = None) -> dict:
        """Store a coaching episode for future reference."""
        from datetime import datetime

        if _settings.use_supabase:
            from src.db import store_episode as db_store_episode
            episode = {
                "summary": f"{summary}\n\n{context}",
                "insights": learnings or [],
                "episode_type": "coaching_insight",
            }
            row = db_store_episode(_settings.agenticsports_user_id, episode)
            return {"stored": True, "id": row["id"]}
        else:
            from src.memory.episodes import store_episode as _store
            episode = {
                "summary": summary,
                "context": context,
                "learnings": learnings or [],
                "timestamp": datetime.now().isoformat(),
                "source": "agent_v3",
            }
            path = _store(episode)
            return {"stored": True, "path": str(path)}

    registry.register(Tool(
        name="store_episode",
        description=(
            "Store a coaching insight or episode for future reference. Use this "
            "when you learn something important that should persist across sessions "
            "(e.g., 'Athlete responds well to detailed explanations', "
            "'Knee pain flares up after intervals > 10km')."
        ),
        handler=store_episode,
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief summary of the episode"},
                "context": {"type": "string", "description": "Full context"},
                "learnings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key learnings from this episode",
                },
            },
            "required": ["summary", "context"],
        },
        category="memory",
        display_label="Speichere eine Episode",
    ))

    def consolidate_episodes(month: str = "") -> dict:
        """Trigger monthly episode consolidation.

        Consolidates weekly reflections into a monthly review summary.
        If no month is specified, checks for any unconsolidated months.
        """
        import asyncio

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        uid = user_model.user_id if hasattr(user_model, "user_id") else _settings.agenticsports_user_id

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            from src.services.episode_consolidation import (
                consolidate_month,
                get_unconsolidated_months,
            )

            async def _run():
                if month:
                    result = await consolidate_month(uid, month)
                    if result:
                        return {"status": "success", "month": month, **result}
                    return {"status": "skipped", "reason": "Not enough weekly reflections"}

                months = await get_unconsolidated_months(uid)
                if not months:
                    return {"status": "ok", "message": "No months need consolidation"}

                results = []
                for m in months[:3]:
                    r = await consolidate_month(uid, m)
                    if r:
                        results.append({"month": m, **r})
                return {
                    "status": "success",
                    "consolidated": results,
                    "count": len(results),
                }

            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_run(), loop)
                return future.result(timeout=60)
            else:
                return asyncio.run(_run())
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # DEPRECATED: consolidate_episodes - background-service tool. NOT registered.
    # registry.register(Tool(
    #     name="consolidate_episodes",
    #     description=(
    #         "Consolidate weekly training reflections into monthly review summaries."
    #     ),
    #     handler=consolidate_episodes,
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "month": {"type": "string"},
    #         },
    #     },
    #     category="memory",
    # ))
