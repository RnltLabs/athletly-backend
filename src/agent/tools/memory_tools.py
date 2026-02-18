"""Memory tools -- update beliefs, profile, and episodes.

These are the equivalent of Claude Code's Edit/Write tools.
Instead of the LLM returning structured JSON with updates,
it explicitly calls tools to make changes. This is MORE transparent --
every memory change is a deliberate tool call, visible in the log.

FIX vs Blueprint: store_episode() in the actual codebase takes a single
dict parameter, not separate (summary, context, learnings) params.
We build the dict inside the wrapper.
"""

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings


def register_memory_tools(registry: ToolRegistry, user_model):
    """Register all memory management tools."""
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
            "Update a specific field in the athlete's profile. Use this when the "
            "athlete shares new information about themselves. Fields: name, sports, "
            "goal.event, goal.target_date, goal.target_time, "
            "fitness.estimated_vo2max, fitness.threshold_pace_min_km, "
            "fitness.weekly_volume_km, fitness.ftp_watts, "
            "constraints.training_days_per_week, constraints.max_session_minutes, "
            "constraints.available_sports"
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

    def add_belief(text: str, category: str, confidence: float = 0.8) -> dict:
        """Add a new belief about the athlete."""
        valid_categories = [
            "scheduling", "fitness", "constraint", "physical",
            "motivation", "history", "preference", "personality",
        ]
        if category not in valid_categories:
            return {"error": f"Invalid category: {category}. Use one of: {valid_categories}"}

        # Check for existing similar belief (avoid duplicates)
        existing = user_model.get_active_beliefs()
        for b in existing:
            if b.get("text", "").lower().strip() == text.lower().strip():
                return {"skipped": True, "reason": "Identical belief already exists", "existing_id": b["id"]}

        belief = user_model.add_belief(
            text=text,
            category=category,
            confidence=min(0.95, max(0.5, confidence)),
            source="conversation",
        )
        user_model.save()

        return {
            "added": True,
            "id": belief.get("id"),
            "text": text,
            "category": category,
            "confidence": confidence,
        }

    registry.register(Tool(
        name="add_belief",
        description=(
            "Record a new belief about the athlete based on what they've shared. "
            "Beliefs are facts, constraints, preferences, or observations the coach "
            "has learned. Use the MOST SPECIFIC category:\n"
            "- scheduling: availability, training days ('Can train 3x/week')\n"
            "- fitness: performance data, test results ('5K in 24 minutes')\n"
            "- constraint: limitations ('No gym access', 'Max 60 min per session')\n"
            "- physical: body, injuries, health ('Knee pain after running', 'Age 16')\n"
            "- motivation: goals, aspirations ('Wants sub-3 marathon')\n"
            "- history: past experience ('Former competitive swimmer')\n"
            "- preference: subjective choices ('Prefers outdoor running')\n"
            "- personality: communication style ('Wants detailed explanations')\n"
            "CRITICAL: 'Knee pain' is PHYSICAL not preference. 'Age 16' is PHYSICAL not preference."
        ),
        handler=add_belief,
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The belief text (concise, factual)",
                },
                "category": {
                    "type": "string",
                    "description": "Belief category",
                    "enum": ["scheduling", "fitness", "constraint", "physical",
                             "motivation", "history", "preference", "personality"],
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0.5-0.95 (default 0.8)",
                },
            },
            "required": ["text", "category"],
        },
        category="memory",
    ))

    def update_belief(
        belief_id: str,
        new_text: str = None,
        new_confidence: float = None,
        new_category: str = None,
    ) -> dict:
        """Update an existing belief."""
        updated = user_model.update_belief(
            belief_id,
            new_text=new_text,
            new_confidence=new_confidence,
        )
        if not updated:
            return {"error": f"Belief {belief_id} not found"}

        if new_category:
            updated["category"] = new_category

        user_model.save()
        return {"updated": True, "belief": updated}

    registry.register(Tool(
        name="update_belief",
        description=(
            "Update an existing belief when new information changes what you know. "
            "Use get_beliefs first to find the belief ID. You can change the text, "
            "confidence, or category."
        ),
        handler=update_belief,
        parameters={
            "type": "object",
            "properties": {
                "belief_id": {"type": "string", "description": "ID of the belief to update"},
                "new_text": {"type": "string", "description": "Updated belief text", "nullable": True},
                "new_confidence": {"type": "number", "description": "Updated confidence", "nullable": True},
                "new_category": {"type": "string", "description": "Updated category", "nullable": True},
            },
            "required": ["belief_id"],
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
