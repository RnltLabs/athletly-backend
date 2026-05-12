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
            "Update a single field in the athlete's structured profile - the "
            "small set of canonical columns the system queries directly. The "
            "structured profile is a cache for queryability; the athlete's "
            "identity and free-form facts live in the journal.\n\n"
            "WHEN TO USE:\n"
            "- Athlete states a sport they actively train: update sports.\n"
            "- Athlete commits to a specific event: prefer update_goal "
            "(atomic journal+profile update + macrocycle archive). Only "
            "use update_profile(goal.*) for cache repair.\n"
            "- Athlete shares a hard fitness number (race result, FTP test, "
            "VO2max test, current weekly volume): update the matching "
            "fitness.* field.\n"
            "- Athlete states a scheduling/equipment constraint that maps "
            "directly to a profile slot: update constraints.* fields.\n\n"
            "WHEN NOT TO USE:\n"
            "- Free-form facts, preferences, injuries, life context: use "
            "the journal tools (update_journal_section / append_to_journal).\n"
            "- One-off observations from a single session: use "
            "annotate_activity (per-activity note) or store_episode (durable "
            "coach learning).\n"
            "- Goal switch: use update_goal so the macrocycle is archived.\n"
            "- Anything not in the fixed valid_fields list: the call will "
            "be rejected. Use update_journal_section('Identity'/...) for "
            "free-form facts.\n\n"
            "Valid fields (exact match, dot notation): name, sports, "
            "goal.event, goal.target_date, goal.target_time, "
            "fitness.estimated_vo2max, fitness.threshold_pace_min_km, "
            "fitness.weekly_volume_km, fitness.ftp_watts, "
            "constraints.training_days_per_week, "
            "constraints.max_session_minutes, constraints.available_sports.\n\n"
            "EXAMPLES:\n"
            "- 'I do triathlon, mainly bike and swim' -> "
            "update_profile(field='sports', value=['cycling', 'swimming', "
            "'running']).\n"
            "- 'My FTP test came in at 285 watts' -> "
            "update_profile(field='fitness.ftp_watts', value=285).\n\n"
            "VALUE FORMAT: arrays can be passed as a JSON string (e.g. "
            "'[\"running\",\"cycling\"]') and numeric fields will auto-parse "
            "numeric strings. Invalid field names return an error dict "
            "listing valid fields - do not retry blindly, switch to "
            "update_journal_section for non-canonical info.\n\n"
            "SIDE EFFECT: writes through to user_model.save() immediately. "
            "There is no draft/commit step - the change is durable as soon "
            "as the tool returns."
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

    registry.register(Tool(
        name="consolidate_episodes",
        description=(
            "Consolidate weekly training reflections into monthly review summaries. "
            "Identifies recurring patterns and promotes them to journal entries. "
            "Call without arguments to auto-detect months needing consolidation, "
            "or specify a month (YYYY-MM) to consolidate a specific month."
        ),
        handler=consolidate_episodes,
        parameters={
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "Month to consolidate (YYYY-MM format). Leave empty for auto-detect.",
                },
            },
        },
        category="memory",
    ))
