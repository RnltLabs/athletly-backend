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
            "Update a single field in the athlete's structured profile (the canonical "
            "record of who the athlete is and what they want). Use this whenever the "
            "athlete shares stable, factual information about themselves that should "
            "persist across sessions.\n\n"
            "WHEN TO USE:\n"
            "- Athlete states a sport they actively train: update sports.\n"
            "- Athlete commits to a goal event: update goal.event, goal.target_date, goal.target_time.\n"
            "- Athlete shares a hard fitness number (race result, FTP test, VO2max test, "
            "current weekly volume): update the matching fitness.* field.\n"
            "- Athlete states a scheduling/equipment constraint: update constraints.* fields.\n\n"
            "WHEN NOT TO USE:\n"
            "- Softer subjective info ('I like long runs', 'I prefer mornings'): use add_belief "
            "with category=preference instead.\n"
            "- Injuries, pain, age, body specifics: use add_belief with category=physical.\n"
            "- One-off observations from a single session: store_episode is more appropriate.\n"
            "- Anything not in the fixed valid_fields list: the call will be rejected. Use "
            "add_belief for free-form facts.\n\n"
            "Valid fields (exact match, dot notation): name, sports, goal.event, "
            "goal.target_date, goal.target_time, fitness.estimated_vo2max, "
            "fitness.threshold_pace_min_km, fitness.weekly_volume_km, fitness.ftp_watts, "
            "constraints.training_days_per_week, constraints.max_session_minutes, "
            "constraints.available_sports.\n\n"
            "EXAMPLES:\n"
            "- Athlete: 'I want to run Berlin Marathon next October' -> "
            "update_profile(field='goal.event', value='Berlin Marathon') and "
            "update_profile(field='goal.target_date', value='2026-10-04').\n"
            "- Athlete: 'I do triathlon, mainly bike and swim' -> "
            "update_profile(field='sports', value=['cycling', 'swimming', 'running']).\n"
            "- Athlete: 'My FTP test came in at 285 watts' -> "
            "update_profile(field='fitness.ftp_watts', value=285).\n\n"
            "VALUE FORMAT: arrays can be passed as a JSON string (e.g. '[\"running\",\"cycling\"]') "
            "and numeric fields will auto-parse numeric strings. Invalid field names return "
            "an error dict listing valid fields - do not retry blindly, switch to add_belief "
            "for non-canonical info.\n\n"
            "SIDE EFFECT: writes through to user_model.save() immediately. There is no draft/commit "
            "step - the change is durable as soon as the tool returns."
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
            "Record a free-form belief about the athlete: a fact, constraint, preference, "
            "or observation the coach has learned through conversation that does not fit "
            "the structured profile fields. Beliefs are the agent's long-term memory of "
            "everything that influences coaching decisions but cannot be expressed as a "
            "single profile column.\n\n"
            "WHEN TO USE:\n"
            "- Athlete reveals something about their body, schedule, or context that the "
            "coach should remember for future plans (e.g. injury history, no pool access, "
            "early-morning person).\n"
            "- After a back-and-forth where you inferred a stable trait or pattern "
            "('responds poorly to back-to-back hard days', 'gets demotivated by missed sessions').\n"
            "- Whenever the answer to 'should the coach still know this in 3 weeks?' is yes "
            "and the info is not a structured profile field.\n\n"
            "WHEN NOT TO USE:\n"
            "- Information fits a profile slot (sport, goal event, FTP, weekly volume, "
            "training-days-per-week): use update_profile instead, do not duplicate.\n"
            "- A single workout observation that does not generalize: use store_episode.\n"
            "- A near-duplicate of an existing belief: the tool auto-skips identical text, "
            "but you should call get_beliefs first if uncertain.\n"
            "- Speculation or unverified guesses: confirm with the athlete first.\n\n"
            "CATEGORY (pick the MOST SPECIFIC, this matters for retrieval):\n"
            "- scheduling: when they can train (days, time of day, blackout periods).\n"
            "- fitness: measured performance data, test results, current capability levels.\n"
            "- constraint: hard limits on training (equipment, venue, session duration cap, budget).\n"
            "- physical: body, age, injuries, chronic issues, anthropometrics, medical context.\n"
            "- motivation: goals, aspirations, the 'why' behind training.\n"
            "- history: past sports experience, prior training background, previous coaches.\n"
            "- preference: subjective choices about how they want to train (indoor vs outdoor, music, solo vs group).\n"
            "- personality: communication style, tone preferences, level of detail wanted.\n\n"
            "CATEGORY DISAMBIGUATION (these get miscategorized often):\n"
            "- 'Knee pain after long runs' -> physical (NOT preference, NOT constraint).\n"
            "- 'Age 16, still growing' -> physical (NOT history).\n"
            "- 'Can only train Tue/Thu/Sat' -> scheduling (NOT constraint).\n"
            "- 'Wants sub-3 marathon' -> motivation (NOT fitness).\n"
            "- 'Used to compete in swimming nationally' -> history (NOT fitness).\n"
            "- 'Wants short answers, no lectures' -> personality (NOT preference).\n\n"
            "EXAMPLES:\n"
            "- add_belief(text='Knee pain after runs longer than 15km', category='physical', confidence=0.85)\n"
            "- add_belief(text='Trains best in the early morning before work', category='scheduling', confidence=0.7)\n"
            "- add_belief(text='Prefers structured intervals over fartleks', category='preference', confidence=0.6)\n\n"
            "CONFIDENCE: 0.5 to 0.95 (clamped). Start lower (0.6-0.7) for inferences, higher "
            "(0.85+) for things the athlete stated explicitly. Use update_belief later to "
            "raise or lower confidence as evidence accumulates.\n\n"
            "SIDE EFFECT: persists immediately via user_model.save(). Duplicate detection is "
            "exact-text match only - paraphrases will be added as separate beliefs, so phrase "
            "consistently if you want dedup to work."
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
            "Modify an existing belief when new information changes what the coach knows. "
            "This is how you invalidate a stale belief (lower confidence to <=0.5), revise "
            "wording when a fact gets sharper, or recategorize when you realize the original "
            "category was wrong.\n\n"
            "WHEN TO USE:\n"
            "- Athlete contradicts something previously stored ('Actually my knee is fine now, "
            "haven't had pain in 6 months') -> raise/lower confidence or rewrite the text.\n"
            "- You see the same belief proven correct multiple times -> raise confidence "
            "toward 0.95.\n"
            "- You realize a belief is in the wrong category and want retrieval to surface "
            "it correctly -> change category.\n"
            "- Belief is now wrong but you want an audit trail (do NOT call add_belief with "
            "a negation, that creates contradictory entries).\n\n"
            "WHEN NOT TO USE:\n"
            "- The information is genuinely new and unrelated to any existing belief: "
            "use add_belief.\n"
            "- You want a totally different fact recorded: use add_belief, not a rewrite "
            "that erases the audit trail.\n"
            "- You do not know the belief_id yet: call get_beliefs first.\n\n"
            "INVALIDATION PATTERN: lower confidence to <= 0.5 to effectively retire a belief "
            "without deleting it (queries use min_confidence filters). Beliefs are immutable "
            "by intent - we lower confidence instead of hard-deleting.\n\n"
            "EXAMPLES:\n"
            "- update_belief(belief_id='b_abc123', new_confidence=0.4) to retire a stale "
            "'No gym access' belief after athlete joined a gym.\n"
            "- update_belief(belief_id='b_xyz789', new_text='Knee pain only on downhill runs >10km', "
            "new_confidence=0.9) to refine a vague pain belief.\n\n"
            "Returns {'error': ...} if belief_id is not found - always pass an id obtained "
            "from a fresh get_beliefs call."
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
            "Identifies recurring patterns and promotes them to beliefs. "
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
