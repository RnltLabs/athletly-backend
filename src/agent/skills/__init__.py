"""Skill system - declarative workflows the agent can invoke.

A skill is a markdown playbook with YAML frontmatter. The agent reads
the description (frontmatter) to decide when to invoke; the body is
the actual workflow content that tells the agent what to do next.

Tier 3 of the Athletly architecture:

    State (DB) -> Tools (atomic actions) -> Skills (workflows) -> Agents

Skills are loaded once at startup, listed in the system prompt, and
executed via the ``invoke_skill`` tool. They are plug-and-play across
agents: any agent with the required tools can run any skill.
"""

from src.agent.skills.loader import (
    Skill,
    get_skill,
    list_skills,
    load_skills,
)

__all__ = ["Skill", "get_skill", "list_skills", "load_skills"]
