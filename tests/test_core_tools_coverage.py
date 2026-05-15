"""Contract test: every tool name mentioned in STATIC_SYSTEM_PROMPT must
either live in ``CORE_TOOL_NAMES`` or be a server-side / deprecated tool
explicitly listed in the allowlist below.

Why this matters: deferred tools only expose their NAME to the model, not
their description or parameter schema. When the prompt says "call X" but
X is deferred, the model either skips the call (rule violated) or invents
the params (validation error). Both are real bugs we have hit in prod.

Run: ``uv run python -m pytest tests/test_core_tools_coverage.py -q``
"""

from __future__ import annotations

import re

import pytest

from src.agent.system_prompt import STATIC_SYSTEM_PROMPT, ONBOARDING_MODE_INSTRUCTIONS
from src.agent.tools.registry import CORE_TOOL_NAMES


# Tools that may be referenced by name but are intentionally NOT in CORE:
# - web_search / web_fetch: Anthropic server-side tools. The agent does
#   not need a client-side schema; Anthropic handles the dispatch.
# - get_health_summary: legacy alias for get_health_activity_summary;
#   referenced once in a plan-workflow hint. Low frequency, fine deferred.
_ALLOWED_OUT_OF_CORE: frozenset[str] = frozenset({
    "web_search",
    "web_fetch",
    "get_health_summary",
})


# Tool names follow the convention <snake_case>(... or are bare identifiers.
# Match identifiers that look like tool names (snake_case with underscore).
# Restrict to identifiers immediately followed by ``(`` or back-tick close
# so we do not match every common English word.
_TOOL_NAME_RE = re.compile(
    r"`(?P<name>[a-z][a-z0-9_]+)`|"
    r"\b(?P<name2>[a-z][a-z0-9_]*_[a-z0-9_]+)\("
)


def _extract_tool_candidates(prompt: str) -> set[str]:
    r"""Return identifiers in ``prompt`` that look like tool names.

    A candidate is either:
      * Wrapped in back-ticks, e.g. \`save_plan\` -> save_plan
      * Followed by an open paren, e.g. save_plan(plan=...)

    Both forms are how the prompt names tools. We require at least one
    underscore to skip common English words like ``can`` or ``run``.
    """
    candidates: set[str] = set()
    for match in _TOOL_NAME_RE.finditer(prompt):
        name = match.group("name") or match.group("name2")
        if not name:
            continue
        if "_" not in name:
            continue
        candidates.add(name)
    return candidates


def test_every_tool_in_static_prompt_is_core() -> None:
    """Tools named in STATIC_SYSTEM_PROMPT must be in CORE_TOOL_NAMES."""
    mentioned = _extract_tool_candidates(STATIC_SYSTEM_PROMPT)
    # Only consider candidates that are actually registered tools. We
    # treat membership in CORE_TOOL_NAMES OR the allowlist as the
    # universe of "tools we know exist". Unknown identifiers (e.g.
    # "tool_call", "tool_result") are filtered out.
    known_tools = CORE_TOOL_NAMES | _ALLOWED_OUT_OF_CORE
    tool_like = {name for name in mentioned if name in known_tools}

    missing = tool_like - CORE_TOOL_NAMES - _ALLOWED_OUT_OF_CORE
    assert not missing, (
        f"Tools mentioned in STATIC_SYSTEM_PROMPT but neither in "
        f"CORE_TOOL_NAMES nor in _ALLOWED_OUT_OF_CORE: {sorted(missing)}. "
        f"Add them to CORE_TOOL_NAMES so the agent gets the full schema, "
        f"or add to _ALLOWED_OUT_OF_CORE if defer is intentional."
    )


def test_every_tool_in_onboarding_prompt_is_core() -> None:
    """Tools named in ONBOARDING_MODE_INSTRUCTIONS must be in CORE_TOOL_NAMES.

    Onboarding is the high-stakes first-impression flow. Any tool the
    onboarding playbook references must have its schema visible so the
    agent never falls back to ``tool_search`` mid-onboarding.
    """
    mentioned = _extract_tool_candidates(ONBOARDING_MODE_INSTRUCTIONS)
    known_tools = CORE_TOOL_NAMES | _ALLOWED_OUT_OF_CORE
    tool_like = {name for name in mentioned if name in known_tools}

    missing = tool_like - CORE_TOOL_NAMES - _ALLOWED_OUT_OF_CORE
    assert not missing, (
        f"Tools mentioned in ONBOARDING_MODE_INSTRUCTIONS but not in "
        f"CORE_TOOL_NAMES: {sorted(missing)}. The onboarding flow cannot "
        f"afford a deferred-tool round trip on every step."
    )


def test_core_tool_names_is_frozenset() -> None:
    """Guard against future refactors turning CORE_TOOL_NAMES into a list."""
    assert isinstance(CORE_TOOL_NAMES, frozenset)
    assert len(CORE_TOOL_NAMES) > 0


@pytest.mark.parametrize(
    "tool_name",
    [
        # Tools referenced in STRICT rules. Any of these going non-core
        # is a regression that breaks the corresponding STRICT rule.
        "propose_sessions",
        "modify_session",
        "complete_session",
        "get_session_window",
        "update_profile",
        "update_goal",
        "get_activities",
        "get_activity_details",
        "sync_garmin_data",
        "get_provider_status",
        "spawn_subagent",
        "invoke_skill",
    ],
)
def test_strict_rule_tools_are_core(tool_name: str) -> None:
    """Explicit allowlist of tools that STRICT rules depend on."""
    assert tool_name in CORE_TOOL_NAMES, (
        f"{tool_name} is referenced in a STRICT rule but is not in "
        f"CORE_TOOL_NAMES. The agent will see only its name without the "
        f"schema and will violate the rule or invent parameters."
    )
