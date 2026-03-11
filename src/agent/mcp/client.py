"""MCP Client -- connects to external MCP servers for additional tools.

Phase 1: Manual tool bridging (load MCP tools as native Tool objects)
Phase 2: Full MCP client protocol when more servers are needed

Current MCP servers we can connect to:
- Brave Search MCP (web search)
- Fetch MCP (web page fetching)
- Future: Garmin MCP, Fitbit MCP, Strava MCP
"""

import logging
import os

from src.agent.tools.registry import Tool

logger = logging.getLogger(__name__)


def load_mcp_tools() -> list[Tool]:
    """Load tools from configured MCP servers.

    Reads MCP server configuration from environment or config file.
    Returns a list of Tool objects that can be registered in the ToolRegistry.
    """
    tools = []

    # Check for Brave Search MCP
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if brave_key:
        tools.append(_create_brave_search_tool(brave_key))
        logger.info("Brave Search API configured — web_search tool active")
    else:
        logger.warning("BRAVE_SEARCH_API_KEY not set — web_search will use fallback")

    return tools


def _create_brave_search_tool(api_key: str) -> Tool:
    """Create a web search tool using Brave Search API."""

    def brave_search(query: str) -> dict:
        import requests

        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 8},
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:8]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                })

            if not results:
                return {
                    "results": [],
                    "source": "brave_search",
                    "message": (
                        f"No results found for '{query}'. "
                        "Try rephrasing with different keywords or a more specific query."
                    ),
                }

            return {"results": results, "source": "brave_search"}

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            logger.error("Brave Search HTTP error %s for query '%s'", status, query)
            return {
                "error": f"Search API returned status {status}",
                "fallback": "Use your built-in knowledge to answer.",
            }
        except requests.Timeout:
            logger.error("Brave Search timeout for query '%s'", query)
            return {
                "error": "Search timed out",
                "fallback": "Use your built-in knowledge to answer.",
            }
        except Exception as e:
            logger.error("Brave Search error for query '%s': %s", query, e)
            return {
                "error": str(e),
                "fallback": "Use your built-in knowledge to answer.",
            }

    return Tool(
        name="web_search",
        description=(
            "Search the web for real-time information using Brave Search. "
            "Returns up to 8 results with titles, snippets, and URLs. "
            "USE THIS TOOL when the athlete asks about:\n"
            "- Specific race dates, locations, or registration info\n"
            "- Current events, news, or recent happenings in sports\n"
            "- Training methodologies or sports science you're unsure about\n"
            "- Weather, course profiles, or venue details\n"
            "- Any factual question where your built-in knowledge may be outdated\n"
            "Tips: Use specific, descriptive queries. Include year for time-sensitive info. "
            "Use the athlete's language for local events (e.g., German for German races)."
        ),
        handler=brave_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query — be specific. Examples: "
                        "'Heidelberger Halbmarathon 2026 Datum', "
                        "'marathon tapering protocol 3 weeks', "
                        "'Berlin Marathon 2026 Anmeldung'"
                    ),
                },
            },
            "required": ["query"],
        },
        category="research",
        source="mcp",
    )
