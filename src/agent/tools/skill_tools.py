"""Tools to invoke skills - the agent's bridge to Tier 3 of the architecture.

Skills are declarative workflows in markdown. ``invoke_skill`` loads a
skill's body and returns it for the agent to follow. The agent then
executes the playbook step by step using its own tool set.

The list of available skills lives in the system prompt (built by
``build_runtime_context``) so the agent can decide which one matches
the current situation. ``invoke_skill`` is the action that actually
opens the playbook.
"""

from __future__ import annotations

import logging

from src.agent.skills import get_skill, list_skills
from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


def register_skill_tools(registry: ToolRegistry, user_model) -> None:
    """Register invoke_skill + list_skills_detail."""

    def invoke_skill(name: str) -> dict:
        """Load a skill's playbook content for the agent to follow.

        Args:
            name: Exact skill name (matching frontmatter ``name`` field).

        Returns:
            ``{"name": ..., "body": "<playbook>", "required_tools": [...]}``
            or ``{"error": "..."}`` if the skill is not found.

        The body is plain markdown - the agent reads it as fresh
        instructions and starts executing. Skills DO NOT mutate state
        on their own; they orchestrate calls to other tools.
        """
        skill = get_skill(name)
        if not skill:
            available = [s.name for s in list_skills()]
            return {
                "error": f"Unknown skill {name!r}",
                "available": available,
            }
        return {
            "name": skill.name,
            "description": skill.description,
            "required_tools": list(skill.required_tools),
            "body": skill.body,
        }

    registry.register(Tool(
        name="invoke_skill",
        description=(
            "Open a skill playbook - a declarative multi-step workflow "
            "for situations that need orchestration across multiple "
            "tool calls. Returns the playbook body as markdown; you then "
            "follow the steps using your normal tools.\n\n"
            "WHEN TO USE:\n"
            "- The current situation matches a skill's description (e.g. "
            "athlete has incomplete profile -> 'onboarding'; athlete "
            "switches goals -> 'goal_change'; athlete reports injury -> "
            "'injury_response').\n"
            "- You want a structured workflow instead of improvising.\n\n"
            "WHEN NOT TO USE:\n"
            "- The action is atomic (single tool call). Just call the "
            "tool directly.\n"
            "- No skill matches the situation. Use your judgement.\n\n"
            "Available skills are listed in your runtime context under "
            "'# Available Skills'. Pick by name. After opening, follow "
            "the playbook steps - it tells you which tools to call and "
            "in what order. You can deviate when the situation requires."
        ),
        handler=invoke_skill,
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Skill name from the available list (e.g. "
                        "'onboarding', 'goal_change', 'injury_response')."
                    ),
                },
            },
            "required": ["name"],
        },
        category="meta",
    ))

    def list_skills_detail() -> dict:
        """List all loaded skills with name + description + required tools."""
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "when_to_use": s.when_to_use,
                    "required_tools": list(s.required_tools),
                }
                for s in list_skills()
            ],
        }

    registry.register(Tool(
        name="list_skills_detail",
        description=(
            "List all currently loaded skills with full descriptions. "
            "Usually unnecessary - the system prompt already shows the "
            "skill list under '# Available Skills'. Use this only if "
            "you forgot what's available or want the required_tools list."
        ),
        handler=list_skills_detail,
        parameters={"type": "object", "properties": {}},
        category="meta",
    ))
