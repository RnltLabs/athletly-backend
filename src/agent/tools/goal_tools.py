"""Tools for managing the athlete's goal in one atomic operation.

``update_goal`` is the right way to change the target event/date/time.
It writes the new values to the profile, logs the old->new transition
with reasoning into ``goal_history``, and lets downstream hooks
cascade (archive the active macrocycle so the coach knows to rebuild).

Using ``update_profile(field="goal.event", ...)`` field-by-field works
but loses two things: the structured history and a single point of
audit. Use this tool whenever the goal actually changes.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_goal_tools(registry: ToolRegistry, user_model) -> None:
    """Register goal-change tools."""
    _settings = get_settings()
    _user_id = (
        getattr(user_model, "user_id", None)
        or _settings.agenticsports_user_id
    )

    def update_goal(
        event: str | None = None,
        target_date: str | None = None,
        target_time: str | None = None,
        reasoning: str = "",
    ) -> dict:
        """Update the athlete's goal atomically and log the change.

        Args:
            event: New target event name (e.g. "Karlsruher Halbmarathon").
            target_date: New target date (YYYY-MM-DD).
            target_time: New target time (HH:MM:SS).
            reasoning: One-paragraph rationale from the conversation - WHY
                the athlete made this change. Stored for later self-reflection.

        Returns:
            ``{"status": "ok", "old_goal": {...}, "new_goal": {...}}`` or error.
        """
        from src.db.client import get_supabase

        client = get_supabase()

        # Load current goal from profile
        try:
            res = (
                client.table("profiles")
                .select("goal_event, goal_target_date, goal_target_time")
                .eq("user_id", _user_id)
                .execute()
            )
        except Exception as exc:
            return {"error": f"Could not load current goal: {exc}"}
        old_row = res.data[0] if res.data else {}
        old_goal = {
            "event": old_row.get("goal_event"),
            "target_date": old_row.get("goal_target_date"),
            "target_time": old_row.get("goal_target_time"),
        }

        new_goal = {
            "event": event if event is not None else old_goal["event"],
            "target_date": target_date if target_date is not None else old_goal["target_date"],
            "target_time": target_time if target_time is not None else old_goal["target_time"],
        }

        if new_goal == old_goal:
            return {
                "status": "noop",
                "message": "New goal identical to current goal - nothing to do.",
                "current_goal": old_goal,
            }

        # Update profile + persist via user_model to keep in-memory + DB in sync
        try:
            if event is not None:
                user_model.update_structured_core("goal.event", event)
            if target_date is not None:
                user_model.update_structured_core("goal.target_date", target_date)
            if target_time is not None:
                user_model.update_structured_core("goal.target_time", target_time)
        except Exception as exc:
            return {"error": f"Failed to persist new goal: {exc}"}

        # Audit log row
        try:
            client.table("goal_history").insert({
                "user_id": _user_id,
                "old_goal": old_goal,
                "new_goal": new_goal,
                "reasoning": reasoning.strip() or None,
            }).execute()
        except Exception as exc:
            logger.warning("Failed to write goal_history row: %s", exc)

        # Archive active macrocycle - the goal moved, the old plan is stale
        archived_macrocycle_id: str | None = None
        try:
            arch = (
                client.table("macrocycle_plans")
                .update({"status": "archived"})
                .eq("user_id", _user_id)
                .eq("status", "active")
                .execute()
            )
            if arch.data:
                archived_macrocycle_id = arch.data[0].get("id")
        except Exception as exc:
            logger.warning("Failed to archive active macrocycle: %s", exc)

        return {
            "status": "ok",
            "old_goal": old_goal,
            "new_goal": new_goal,
            "archived_macrocycle_id": archived_macrocycle_id,
            "next_step": (
                "Build a fresh macrocycle for the new goal using "
                "create_macrocycle_plan + save_macrocycle, then derive "
                "the next training week with create_training_plan."
            ),
        }

    registry.register(Tool(
        name="update_goal",
        description=(
            "Atomically change the athlete's target goal (event, target_date, "
            "and/or target_time) and log the change for future reflection. "
            "Use this whenever the athlete commits to a NEW target instead of "
            "amending the existing one.\n\n"
            "WHAT IT DOES:\n"
            "1. Reads the current goal from profile.\n"
            "2. Writes new goal_event / goal_target_date / goal_target_time.\n"
            "3. Logs old -> new + your captured reasoning to goal_history.\n"
            "4. Archives any active macrocycle (the old plan no longer matches).\n\n"
            "WHEN TO USE:\n"
            "- Athlete picks a different target race ('I want to do Karlsruhe "
            "instead of Heidelberg').\n"
            "- Target time changes ('I want sub-1:30 now, not sub-1:35').\n"
            "- Date shifts ('postponed to next year').\n\n"
            "WHEN NOT TO USE:\n"
            "- Just adding a soft fitness goal to an existing target: use "
            "add_belief for context, keep the primary goal stable.\n"
            "- Athlete mentions a far-future dream goal but no commitment "
            "yet: ask first, then update only after a clear yes.\n\n"
            "ALWAYS pass *reasoning*: a one-sentence summary of WHY the "
            "athlete changed the goal. The agent will read this later when "
            "wondering 'why did we abandon the old plan?'.\n\n"
            "AFTER calling update_goal you typically:\n"
            "1. create_macrocycle_plan(...) with the new target.\n"
            "2. save_macrocycle() to persist.\n"
            "3. create_training_plan(macrocycle_week=1).\n"
            "4. save_plan() and report the new structure to the athlete.\n\n"
            "EXAMPLE:\n"
            "  update_goal(\n"
            "    event='Karlsruher Halbmarathon',\n"
            "    target_date='2027-05-17',\n"
            "    target_time='1:35:00',\n"
            "    reasoning='Athlete chose Karlsruhe over Heidelberg for "
            "a flatter course and later date allowing more base building.'\n"
            "  )"
        ),
        handler=update_goal,
        parameters={
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "description": "New target event name. Omit to keep current.",
                    "nullable": True,
                },
                "target_date": {
                    "type": "string",
                    "description": "New target date YYYY-MM-DD. Omit to keep current.",
                    "nullable": True,
                },
                "target_time": {
                    "type": "string",
                    "description": "New target time HH:MM:SS. Omit to keep current.",
                    "nullable": True,
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why the athlete is making this change. Captured "
                        "verbatim into goal_history for future reflection."
                    ),
                },
            },
        },
        category="memory",
    ))
