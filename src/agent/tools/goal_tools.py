"""Tools for managing the athlete's goal in one atomic operation.

``update_goal`` is the right way to change the target event/date/time.
It writes the new goal as a markdown block into the athlete journal's
``Current Goal`` section, appends a timeline entry to ``Goal Timeline``,
archives the active macrocycle, and also updates the structured profile
columns as a query cache.

Using ``update_journal_section`` directly works but loses the timeline
log and the macrocycle archive cascade. Use this tool whenever the goal
actually changes.
"""

from __future__ import annotations

import logging
from datetime import date as _date_cls

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

    def _read_current_goal_from_journal() -> dict:
        """Parse the 'Current Goal' section of the journal back to fields."""
        from src.agent.athlete_journal import parse_sections, read_journal

        doc = read_journal(_user_id)
        sections = parse_sections(doc)
        body = sections.get("Current Goal", "") or ""

        result: dict[str, str | None] = {
            "event": None,
            "target_date": None,
            "target_time": None,
        }
        for raw in body.splitlines():
            line = raw.strip()
            if not line.lower().startswith("**"):
                continue
            # Patterns: **Event:** ..., **Date:** ..., **Target time:** ...
            try:
                label, _, value = line.partition(":")
                label = label.strip("* ").lower()
                value = value.strip().rstrip("*").strip()
            except Exception:
                continue
            if label == "event":
                result["event"] = value or None
            elif label == "date":
                # Body shape: "2027-05-17 (in 245 days)"
                result["target_date"] = value.split(" ", 1)[0] or None
            elif label.startswith("target time"):
                result["target_time"] = value or None
        return result

    def _render_current_goal_md(
        event: str | None,
        target_date: str | None,
        target_time: str | None,
        event_facts: str | None,
        source: str | None,
    ) -> str:
        """Render a clean markdown block for the Current Goal section."""
        lines: list[str] = []
        if event:
            lines.append(f"**Event:** {event}")
        if target_date:
            try:
                days = (
                    _date_cls.fromisoformat(target_date) - _date_cls.today()
                ).days
                lines.append(f"**Date:** {target_date} (in {days} days)")
            except Exception:
                lines.append(f"**Date:** {target_date}")
        if target_time:
            lines.append(f"**Target time:** {target_time}")
        if event_facts and event_facts.strip():
            lines.append(f"**Course facts:** {event_facts.strip()}")
        if source and source.strip():
            lines.append(f"**Source:** {source.strip()}")
        return "\n".join(lines)

    def update_goal(
        event: str | None = None,
        target_date: str | None = None,
        target_time: str | None = None,
        event_facts: str | None = None,
        source: str | None = None,
        reasoning: str = "",
    ) -> dict:
        """Update the athlete's goal atomically and log the change.

        Args:
            event: Target event name (e.g. "Karlsruher Halbmarathon").
            target_date: Target date (YYYY-MM-DD).
            target_time: Target finish time (HH:MM:SS) when applicable.
            event_facts: Course details, elevation, distance, etc.
            source: Where the facts came from (e.g. "official site",
                "spawn_subagent research 2026-05-12").
            reasoning: One-sentence rationale - WHY the athlete changed.

        Returns:
            ``{"status": "ok", "old_goal": {...}, "new_goal": {...}}`` or error.
        """
        from src.agent.athlete_journal import (
            append_to_section,
            update_section,
        )
        from src.db.client import get_supabase

        client = get_supabase()

        old_goal = _read_current_goal_from_journal()
        new_goal = {
            "event": event if event is not None else old_goal["event"],
            "target_date": (
                target_date if target_date is not None else old_goal["target_date"]
            ),
            "target_time": (
                target_time if target_time is not None else old_goal["target_time"]
            ),
        }

        if new_goal == old_goal and not event_facts and not source:
            return {
                "status": "noop",
                "message": "New goal identical to current goal - nothing to do.",
                "current_goal": old_goal,
            }

        # 1. Rewrite the Current Goal section.
        new_md = _render_current_goal_md(
            event=new_goal["event"],
            target_date=new_goal["target_date"],
            target_time=new_goal["target_time"],
            event_facts=event_facts,
            source=source,
        )
        try:
            update_section(_user_id, "Current Goal", new_md)
        except Exception as exc:
            return {"error": f"Could not update journal Current Goal: {exc}"}

        # 2. Append to Goal Timeline.
        today = _date_cls.today().isoformat()
        timeline_parts: list[str] = [today + ":"]
        if new_goal["event"]:
            timeline_parts.append(new_goal["event"])
        if new_goal["target_date"]:
            timeline_parts.append(f"on {new_goal['target_date']}")
        if new_goal["target_time"]:
            timeline_parts.append(f"target {new_goal['target_time']}")
        timeline_entry = " ".join(timeline_parts).strip()
        if reasoning.strip():
            timeline_entry = f"{timeline_entry}. Reasoning: {reasoning.strip()}"
        try:
            append_to_section(_user_id, "Goal Timeline", timeline_entry)
        except Exception as exc:
            logger.warning("Failed to append Goal Timeline entry: %s", exc)

        # 3. Profile cache - keep query-friendly columns in sync.
        try:
            if event is not None:
                user_model.update_structured_core("goal.event", event)
            if target_date is not None:
                user_model.update_structured_core("goal.target_date", target_date)
            if target_time is not None:
                user_model.update_structured_core("goal.target_time", target_time)
        except Exception as exc:
            logger.warning("Failed to update profile goal cache: %s", exc)

        # 4. Archive any active macrocycle - the goal moved.
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
            "Atomically change target goal: rewrites journal 'Current Goal', "
            "appends to 'Goal Timeline', archives active macrocycle, updates "
            "profile cache (goal_event/target_date/target_time). Use when "
            "athlete commits to a new event/date/time. Avoid for musings "
            "(confirm first) or pure journal-text edits (use "
            "update_journal_section). ALWAYS pass reasoning (one sentence "
            "captured in timeline) and event_facts+source when known. After: "
            "create_macrocycle_plan -> save_macrocycle -> weekly plan."
        ),
        handler=update_goal,
        parameters={
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "description": "Target event name. Omit to keep current.",
                    "nullable": True,
                },
                "target_date": {
                    "type": "string",
                    "description": "Target date YYYY-MM-DD. Omit to keep current.",
                    "nullable": True,
                },
                "target_time": {
                    "type": "string",
                    "description": "Target finish time HH:MM:SS. Omit to keep current.",
                    "nullable": True,
                },
                "event_facts": {
                    "type": "string",
                    "description": (
                        "Course details, distance, elevation, surface, "
                        "weather window, etc. Goes into 'Current Goal'."
                    ),
                    "nullable": True,
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Where the facts came from: 'official site', "
                        "'spawn_subagent research YYYY-MM-DD', 'athlete "
                        "told me directly'. Goes into 'Current Goal'."
                    ),
                    "nullable": True,
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why the athlete is making this change. Captured "
                        "into 'Goal Timeline' for future reflection."
                    ),
                },
            },
        },
        category="memory",
    ))
