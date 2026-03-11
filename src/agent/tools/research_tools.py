"""Research tools -- external knowledge via web search and MCP.

These are the equivalent of Claude Code's WebSearch and WebFetch tools.
The agent uses these to research training methodologies, race information,
sports science, and other external knowledge.

Implementation Strategy:
- Phase 1: Native web search via SerpAPI/Google Custom Search
- Phase 2: MCP-based web tools (plug in any MCP search server)
- Phase 3: Garmin/Fitbit MCP servers (when API access is available)
"""

import os
from src.agent.tools.registry import Tool, ToolRegistry


def register_research_tools(registry: ToolRegistry):
    """Register research tools (web search, web fetch)."""

    def web_search(query: str) -> dict:
        """Search the web for information."""
        # Try SerpAPI first
        api_key = os.environ.get("SERP_API_KEY")
        if api_key:
            try:
                import requests
                resp = requests.get(
                    "https://serpapi.com/search",
                    params={"q": query, "api_key": api_key, "num": 5},
                    timeout=10,
                )
                data = resp.json()
                results = []
                for item in data.get("organic_results", [])[:5]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", ""),
                    })
                return {"results": results, "source": "serpapi"}
            except ImportError:
                return {"error": "requests library not installed", "fallback": "Use your built-in knowledge."}
            except Exception as e:
                return {"error": f"Search failed: {e}", "fallback": "Use your built-in knowledge."}

        # No search API configured
        return {
            "results": [],
            "source": "none",
            "message": (
                "Web search is not configured (no SERP_API_KEY). "
                "Use your built-in sports science knowledge instead. "
                "Your training knowledge is comprehensive and current."
            ),
        }

    registry.register(Tool(
        name="web_search",
        description=(
            "Search the web for real-time information: race dates, events, training "
            "methodologies, sports science, or any external knowledge. Returns search "
            "results with titles, snippets, and URLs. Use this when the athlete asks "
            "about specific events, dates, or facts you might not know. "
            "Note: Requires SERP_API_KEY or BRAVE_SEARCH_API_KEY to be configured. "
            "If unavailable, use your built-in knowledge and be transparent about it."
        ),
        handler=web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query — be specific and include year for events. "
                        "Examples: 'Heidelberger Halbmarathon 2026 Datum', "
                        "'marathon base phase training 16 weeks'"
                    ),
                },
            },
            "required": ["query"],
        },
        category="research",
    ))

    def web_fetch(url: str, extract_prompt: str = "Extract the key information.") -> dict:
        """Fetch and extract content from a URL."""
        try:
            import requests
        except ImportError:
            return {"error": "requests library not installed. Run: uv add requests"}

        try:
            from html import unescape
            import re

            resp = requests.get(url, timeout=15, headers={"User-Agent": "AgenticSports/1.0"})
            resp.raise_for_status()

            # Basic HTML to text
            text = resp.text
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Truncate to reasonable length
            text = text[:8000]

            return {
                "url": url,
                "content": text,
                "length": len(text),
            }
        except Exception as e:
            return {"error": f"Failed to fetch {url}: {e}"}

    registry.register(Tool(
        name="web_fetch",
        description=(
            "Fetch and extract content from a specific URL. Use this after web_search "
            "to read full articles or pages. Returns plain text content."
        ),
        handler=web_fetch,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "extract_prompt": {
                    "type": "string",
                    "description": "What to extract from the page",
                    "nullable": True,
                },
            },
            "required": ["url"],
        },
        category="research",
    ))
