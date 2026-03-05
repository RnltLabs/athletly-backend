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


@dataclass
class Tool:
    """A single tool available to the agent."""
    name: str
    description: str
    handler: Callable[..., dict]
    parameters: dict = field(default_factory=dict)  # JSON Schema
    category: str = "general"     # data, analysis, research, planning, memory, meta
    source: str = "native"        # native | mcp


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

    def get_openai_tools(self) -> list[dict]:
        """Get tool declarations in OpenAI/LiteLLM format.

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
                # Strip nullable (not part of JSON Schema proper) before sending
                entry["function"]["parameters"] = _clean_parameters(tool.parameters)
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
    from src.agent.tools.health_tools import register_health_tools
    from src.agent.tools.health_trend_tools import register_health_trend_tools
    from src.agent.tools.health_inventory_tools import register_health_inventory_tools
    from src.agent.tools.goal_trajectory_tools import register_goal_trajectory_tools

    register_data_tools(registry, user_model)
    register_health_tools(registry)
    register_health_trend_tools(registry)
    register_health_inventory_tools(registry)
    register_analysis_tools(registry)
    register_calc_tools(registry, user_model)
    register_goal_trajectory_tools(registry)

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

    from src.agent.tools.health_tools import register_health_tools
    from src.agent.tools.health_trend_tools import register_health_trend_tools
    from src.agent.tools.health_inventory_tools import register_health_inventory_tools

    from src.agent.tools.checkpoint_tools import register_checkpoint_tools
    from src.agent.tools.self_improvement_tools import register_self_improvement_tools
    from src.agent.tools.product_tools import register_product_tools
    from src.agent.tools.goal_trajectory_tools import register_goal_trajectory_tools
    from src.agent.tools.macrocycle_tools import register_macrocycle_tools
    from src.agent.tools.garmin_tools import register_garmin_tools
    from src.agent.tools.notification_tools import register_notification_tools
    from src.agent.tools.session_tools import register_session_tools
    from src.agent.tools.episode_read_tools import register_episode_read_tools
    from src.agent.tools.plan_adjust_tools import register_plan_adjust_tools

    # Resolve user_id from user_model for tools that need it
    from src.config import get_settings as _get_settings
    _user_id = (
        user_model.user_id
        if hasattr(user_model, "user_id")
        else _get_settings().agenticsports_user_id
    )

    register_data_tools(registry, user_model)
    register_health_tools(registry)
    register_health_trend_tools(registry)
    register_health_inventory_tools(registry)
    register_analysis_tools(registry)
    register_planning_tools(registry, user_model)
    register_memory_tools(registry, user_model)
    register_research_tools(registry)
    register_meta_tools(registry, user_model)
    register_config_tools(registry, user_model)
    register_calc_tools(registry, user_model)
    register_checkpoint_tools(registry, user_model)
    register_self_improvement_tools(registry, user_model)
    register_product_tools(registry, user_model)
    register_goal_trajectory_tools(registry, user_model)
    register_macrocycle_tools(registry, user_model)
    register_garmin_tools(registry, user_model)
    register_notification_tools(registry, _user_id)
    register_session_tools(registry, _user_id)
    register_episode_read_tools(registry, user_model)
    register_plan_adjust_tools(registry, user_model)

    if context == "onboarding":
        from src.agent.tools.onboarding_tools import register_onboarding_tools
        register_onboarding_tools(registry, user_model)

    # Register MCP tools (overrides native fallbacks if available)
    from src.agent.mcp.client import load_mcp_tools
    mcp_tools = load_mcp_tools()
    if mcp_tools:
        registry.register_mcp_tools(mcp_tools)

    return registry
