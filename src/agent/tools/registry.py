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
            return {"error": f"Unknown tool: {name}"}
        tool = self._tools[name]
        try:
            return tool.handler(**args)
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}

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


def execute_with_budget(
    tool_registry: "ToolRegistry",
    name: str,
    args: dict,
    budget_tokens: int = 2000,
) -> dict:
    """Execute a tool and truncate its result if it exceeds the token budget.

    Token estimation: 4 characters ≈ 1 token (rough GPT/Gemini average).

    Args:
        tool_registry: The registry to dispatch the call through.
        name: Tool name.
        args: Tool arguments.
        budget_tokens: Maximum allowed tokens in the result text. Uses the
            per-tool budget map when available, falling back to budget_tokens.

    Returns:
        The tool result dict, possibly with string fields truncated.
    """
    import json as _json

    per_tool_budget: dict[str, int] = {
        "get_activities": 1500,
        "analyze_training_load": 800,
        "web_search": 1200,
        "create_training_plan": 2000,
    }
    effective_budget = per_tool_budget.get(name, budget_tokens)

    result = tool_registry.execute(name, args)

    # Estimate size via JSON serialisation
    try:
        text = _json.dumps(result)
    except (TypeError, ValueError):
        text = str(result)

    char_budget = effective_budget * 4
    if len(text) > char_budget:
        n_tokens = len(text) // 4
        truncated = text[:char_budget]
        return {
            "result": truncated,
            "_truncated": True,
            "_note": f"... [truncated, {n_tokens} tokens]",
        }

    return result


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

    register_data_tools(registry, user_model)
    register_analysis_tools(registry)
    register_planning_tools(registry, user_model)
    register_memory_tools(registry, user_model)
    register_research_tools(registry)
    register_meta_tools(registry, user_model)
    register_config_tools(registry, user_model)
    register_calc_tools(registry, user_model)

    if context == "onboarding":
        from src.agent.tools.onboarding_tools import register_onboarding_tools
        register_onboarding_tools(registry, user_model)

    # Register MCP tools (overrides native fallbacks if available)
    from src.agent.mcp.client import load_mcp_tools
    mcp_tools = load_mcp_tools()
    if mcp_tools:
        registry.register_mcp_tools(mcp_tools)

    return registry
