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
            "Atomically change the athlete's target goal and log the change. "
            "The journal's 'Current Goal' section is rewritten, the 'Goal "
            "Timeline' section gets a new bullet, any active macrocycle is "
            "archived, and profile columns are updated as a query cache.\n\n"
            "WHAT IT DOES:\n"
            "1. Rewrites journal 'Current Goal' with event + date + target "
            "time + course facts + source.\n"
            "2. Appends a 'YYYY-MM-DD: <event> on <date> target <time>. "
            "Reasoning: ...' bullet to 'Goal Timeline'.\n"
            "3. Updates profile.goal_event/goal_target_date/goal_target_time "
            "as a queryability cache.\n"
            "4. Archives any active macrocycle (the old plan no longer "
            "matches the new target).\n\n"
            "WHEN TO USE:\n"
            "- Athlete commits to a NEW target ('I want Karlsruhe instead "
            "of Heidelberg').\n"
            "- Target time changes ('sub 1:30 now, not sub 1:35').\n"
            "- Date shifts ('postponed to next year').\n\n"
            "WHEN NOT TO USE:\n"
            "- Just musing about a future goal - confirm commitment first.\n"
            "- Tweaking the journal 'Current Goal' wording: use "
            "update_journal_section directly instead (it does not archive "
            "the macrocycle).\n\n"
            "ALWAYS pass *event_facts* and *source* when they are known. "
            "Course details and the verification source live in the "
            "journal and the coach will rely on them later for plan "
            "design. ALWAYS pass *reasoning*: one sentence on WHY the "
            "athlete changed - captured into Goal Timeline for future "
            "reflection.\n\n"
            "TYPICAL FLOW:\n"
            "1. spawn_subagent(task='Find date, distance, elevation, "
            "course for <event>') to verify facts.\n"
            "2. Confirm with the user: 'Du meinst den XYZ am DD.MM., ja?'\n"
            "3. update_goal(event=..., target_date=..., target_time=..., "
            "event_facts='21.1km, 180m elevation, flat closing 5km', "
            "source='official site, fetched 2026-05-12 via spawn_subagent', "
            "reasoning='Athlete chose Karlsruhe for flatter course and "
            "later date allowing more base building.')\n"
            "4. create_macrocycle_plan(...) -> save_macrocycle() -> "
            "create_training_plan(macrocycle_week=1) -> save_plan().\n\n"
            "EXAMPLE:\n"
            "  update_goal(\n"
            "    event='Karlsruher Halbmarathon',\n"
            "    target_date='2027-05-17',\n"
            "    target_time='1:35:00',\n"
            "    event_facts='21.1km road, 180m elevation, urban, late May',\n"
            "    source='Athlete confirmed after spawn_subagent research',\n"
            "    reasoning='Flatter course than Heidelberg, more base time.'\n"
            "  )"
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
