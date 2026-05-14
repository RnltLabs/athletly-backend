"""Tool Registry -- manages all tools available to the agent.

Equivalent to Claude Code's tool system. Each tool is:
1. An OpenAI-compatible function definition (schema for the model via LiteLLM)
2. A Python callable (implementation)

Tools can be:
- Native (Python functions in this codebase)
- MCP (loaded from external MCP servers)
"""

from dataclasses import dataclass, field
from typing import Any, Callable

_RETRY_HINT = " [Analyze the error and try a different approach.]"


# Set of tools that are ALWAYS loaded (description visible to the agent
# every turn). Everything else is deferred and fetched via tool_search
# on demand. Keep this list tight - each entry adds tokens to every call.
CORE_TOOL_NAMES: frozenset[str] = frozenset({
    # ------------------------------------------------------------------
    # Rule: every tool that is named verbatim in the STATIC system prompt
    # MUST live in this set. If the prompt says "call save_plan" but the
    # schema is deferred, the agent sees only the name and either skips
    # the call or invents the params. The test in
    # tests/test_core_tools_coverage.py enforces this contract.
    # ------------------------------------------------------------------

    # Identity / memory (most-used)
    "update_journal_section",
    "append_to_journal",
    "remove_from_journal",
    "annotate_activity",
    # Goal change (used whenever the athlete commits)
    "update_goal",
    # Profile write: agent hallucinates value types (string "null" -> int col)
    # without seeing the full param schema, so keep description always loaded.
    "update_profile",
    # Plan write: shape is non-trivial ("sessions": [...] required, NOT a
    # weekly_structure dict). Agent without the description guesses wrong
    # and emits an empty plan_preview card.
    "save_plan",
    "get_active_plan",
    # Read context (cheap, often needed)
    "get_activities",
    "get_athlete_profile",
    # Activity detail: STRICT rule "MUST first call get_activity_details"
    # before discussing VO2max / threshold pace / FTP / per-session metrics.
    # Without the schema the model cannot construct the call and either
    # skips the rule or falls back to estimation.
    "get_activity_details",
    # Provider freshness: the STRICT proactive-sync rule in system_prompt
    # tells the agent to call these when the athlete reports a
    # just-completed activity. Without the schema visible, the model
    # either skips the call or constructs it with wrong params
    # (e.g. missing mode="auto"). Always-loaded fixes both.
    "sync_garmin_data",
    "get_provider_status",
    # Subagent spawn: referenced multiple times in the prompt for race
    # research and deeper context fetches. Cannot afford the extra
    # tool_search round-trip every time the model considers research.
    "spawn_subagent",
    # Skill invocation - opens any other workflow
    "invoke_skill",
    # Recovery alerts: STRICT WHOLE-ATHLETE COACHING rule tells the
    # agent to call this when the athlete reports feeling off and no
    # `# Coach Alerts` section was in the runtime context. Deferring
    # would force a tool_search detour exactly when the agent needs
    # the schema most.
    "get_recovery_alerts",
    # Sport math: STRICT SPORT MATH DISCIPLINE rule tells the agent
    # to call this for ANY FTP / NP / VDOT / CSS / Karvonen / pace
    # calculation. Deferring would force a tool_search round-trip
    # exactly when the math discipline matters most (Lisa-bug fix).
    "compute_sport_math",
    # Tool discovery - lets the agent find deferred tools
    # (the tool_search helper itself is added at request build time)

    # ------------------------------------------------------------------
    # Onboarding-only tools. These are only registered when
    # get_default_tools(context="onboarding") runs, so listing them here
    # is a no-op outside onboarding. Listed for completeness so the
    # coverage test passes and so they get full schema visibility when
    # the onboarding playbook references them by name.
    # ------------------------------------------------------------------
    "ask_choice",
    "ask_date",
    "ask_number",
    "complete_onboarding",
    "request_garmin_connect",
    "recommend_products",
    "define_config",
})


@dataclass
class Tool:
    """A single tool available to the agent."""
    name: str
    description: str
    handler: Callable[..., dict]
    parameters: dict = field(default_factory=dict)  # JSON Schema
    category: str = "general"     # data, analysis, research, planning, memory, meta
    source: str = "native"        # native | mcp
    display_label: str = ""       # German action phrase shown to users in UI


class ToolRegistry:
    """Registry of all tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_mcp_tools(self, mcp_tools: list[Tool]) -> None:
        """Register tools loaded from an MCP server."""
        from dataclasses import replace
        for tool in mcp_tools:
            self._tools[tool.name] = replace(tool, source="mcp")

    def get_openai_tools(self, *, defer_non_core: bool = False) -> list[dict]:
        """Get tool declarations in OpenAI/LiteLLM format.

        When ``defer_non_core`` is True, every tool NOT in
        ``CORE_TOOL_NAMES`` gets a top-level ``defer_loading: True`` flag.
        LiteLLM forwards this to Anthropic's API: deferred tools only have
        their NAME in the prompt, descriptions are fetched on demand via
        the ``tool_search`` helper. This cuts the tool-layer token budget
        by 80%+ for our 61-tool registry.

        Returns a list of dicts suitable for the ``tools`` parameter of
        ``litellm.completion()`` / ``openai.chat.completions.create()``.
        """
        result = []
        for tool in self._tools.values():
            entry: dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                },
            }
            if tool.parameters:
                entry["function"]["parameters"] = _clean_parameters(tool.parameters)
            if defer_non_core and tool.name not in CORE_TOOL_NAMES:
                entry["defer_loading"] = True
            result.append(entry)
        return result

    def execute(self, name: str, args: dict) -> dict:
        """Execute a tool by name with given arguments."""
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}.{_RETRY_HINT}"}
        tool = self._tools[name]
        try:
            return tool.handler(**args)
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}.{_RETRY_HINT}"}
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}.{_RETRY_HINT}"}

    def list_tools(self) -> list[dict]:
        """List all registered tools (for debugging)."""
        return [
            {"name": t.name, "category": t.category, "source": t.source}
            for t in self._tools.values()
        ]


def _clean_parameters(schema: dict) -> dict:
    """Clean a JSON Schema dict for OpenAI tool format.

    Removes non-standard keys like ``nullable`` that some tool definitions
    carry (Gemini extension) and recursively cleans nested schemas.
    """
    cleaned: dict = {}
    for key, value in schema.items():
        if key == "nullable":
            continue  # Not standard JSON Schema; skip
        if key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                k: _clean_parameters(v) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = _clean_parameters(value)
        else:
            cleaned[key] = value
    return cleaned



def get_restricted_tools(user_model) -> ToolRegistry:
    """Create a restricted tool registry for background sub-agents.

    Only includes safe, read-oriented tools — no notifications, spawning,
    onboarding, or config-mutation tools.

    Allowed categories: data, analysis, calc, session/memory (read-only).
    Blocked: send_notification, spawn_background_task, define_metric,
             complete_onboarding, consolidate_episodes.
    """
    registry = ToolRegistry()

    from src.agent.tools.data_tools import register_data_tools
    from src.agent.tools.analysis_tools import register_analysis_tools
    from src.agent.tools.calc_tools import register_calc_tools
    from src.agent.tools.compute_tools import register_compute_tools
    from src.agent.tools.health_tools import register_health_tools
    from src.agent.tools.health_trend_tools import register_health_trend_tools
    from src.agent.tools.health_inventory_tools import register_health_inventory_tools

    from src.config import get_settings as _gs
    _uid = (
        user_model.user_id
        if hasattr(user_model, "user_id")
        else _gs().agenticsports_user_id
    )

    register_data_tools(registry, user_model)
    register_health_tools(registry, user_id=_uid)
    register_health_trend_tools(registry, user_id=_uid)
    register_health_inventory_tools(registry)
    register_analysis_tools(registry)
    register_calc_tools(registry, user_model)
    register_compute_tools(registry, user_model)

    return registry


def get_default_tools(user_model, context: str = "coach") -> ToolRegistry:
    """Create the default tool registry with all native tools.

    This is called once at agent startup. MCP tools are added separately.

    Args:
        user_model: The user model instance.
        context: Session context — "coach" or "onboarding". Onboarding tools
                 (e.g. complete_onboarding) are only registered when
                 context == "onboarding".
    """
    registry = ToolRegistry()

    # Import and register all tool modules
    from src.agent.tools.data_tools import register_data_tools
    from src.agent.tools.analysis_tools import register_analysis_tools
    from src.agent.tools.planning_tools import register_planning_tools
    from src.agent.tools.memory_tools import register_memory_tools
    from src.agent.tools.research_tools import register_research_tools
    from src.agent.tools.meta_tools import register_meta_tools
    from src.agent.tools.config_tools import register_config_tools
    from src.agent.tools.calc_tools import register_calc_tools
    from src.agent.tools.compute_tools import register_compute_tools

    from src.agent.tools.health_tools import register_health_tools
    from src.agent.tools.health_trend_tools import register_health_trend_tools
    from src.agent.tools.health_inventory_tools import register_health_inventory_tools

    from src.agent.tools.checkpoint_tools import register_checkpoint_tools
    from src.agent.tools.product_tools import register_product_tools
    from src.agent.tools.garmin_tools import register_garmin_tools
    from src.agent.tools.notification_tools import register_notification_tools
    from src.agent.tools.action_tools import register_action_tools
    from src.agent.tools.ui_tools import register_ui_tools
    from src.agent.tools.session_tools import register_session_tools
    from src.agent.tools.episode_read_tools import register_episode_read_tools
    from src.agent.tools.goal_tools import register_goal_tools
    from src.agent.tools.skill_tools import register_skill_tools
    from src.agent.tools.journal_tools import register_journal_tools

    # Resolve user_id from user_model for tools that need it
    from src.config import get_settings as _get_settings
    _user_id = (
        user_model.user_id
        if hasattr(user_model, "user_id")
        else _get_settings().agenticsports_user_id
    )

    register_data_tools(registry, user_model)
    register_health_tools(registry, user_id=_user_id)
    register_health_trend_tools(registry, user_id=_user_id)
    register_health_inventory_tools(registry)
    register_analysis_tools(registry)
    register_planning_tools(registry, user_model)
    register_memory_tools(registry, user_model)
    register_research_tools(registry)
    register_meta_tools(registry, user_model)
    register_config_tools(registry, user_model)
    register_calc_tools(registry, user_model)
    register_compute_tools(registry, user_model)
    register_checkpoint_tools(registry, user_model)
    register_product_tools(registry, user_model)
    register_garmin_tools(registry, user_model)
    register_notification_tools(registry, _user_id)
    register_action_tools(registry, user_model)
    register_ui_tools(registry, user_model)
    register_session_tools(registry, _user_id)
    register_episode_read_tools(registry, user_model)
    register_goal_tools(registry, user_model)
    register_skill_tools(registry, user_model)
    register_journal_tools(registry, user_model)

    if context == "onboarding":
        from src.agent.tools.onboarding_tools import register_onboarding_tools
        register_onboarding_tools(registry, user_model)

    # Register MCP tools (overrides native fallbacks if available)
    from src.agent.mcp.client import load_mcp_tools
    mcp_tools = load_mcp_tools()
    if mcp_tools:
        registry.register_mcp_tools(mcp_tools)

    return registry
