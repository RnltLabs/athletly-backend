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

    def web_search(query: str, max_uses: int = 5) -> dict:
        """Search the web via Anthropic's native server-side web search.

        Uses Anthropic's ``web_search_20250305`` server tool: a single
        Anthropic API call where the model is given the web_search tool,
        does the search server-side, and returns titles + URLs in
        structured content blocks.

        Cost: ~$0.01 per search. Requires only ANTHROPIC_API_KEY.
        Same mechanism Claude Code uses internally.

        Returns:
            ``{"results": [{title, url}], "source": "anthropic_web_search"}``
            or an error dict on failure.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {
                "results": [],
                "source": "none",
                "message": (
                    "ANTHROPIC_API_KEY missing - web search unavailable. "
                    "Use your built-in knowledge instead."
                ),
            }

        try:
            import anthropic
        except ImportError:
            return {
                "error": "anthropic SDK not installed",
                "fallback": "Use your built-in knowledge.",
            }

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max(1, min(int(max_uses), 8)),
                }],
                messages=[{
                    "role": "user",
                    "content": f"Search the web for: {query}",
                }],
            )
        except Exception as exc:
            return {
                "error": f"Anthropic web_search failed: {exc}",
                "fallback": "Use your built-in knowledge.",
            }

        results: list[dict] = []
        commentary: list[str] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "web_search_tool_result":
                content = getattr(block, "content", None)
                if isinstance(content, list):
                    for hit in content:
                        results.append({
                            "title": getattr(hit, "title", ""),
                            "url": getattr(hit, "url", ""),
                        })
                else:
                    err = getattr(content, "error_code", "unknown")
                    commentary.append(f"web_search_error:{err}")
            elif btype == "text":
                txt = getattr(block, "text", "")
                if txt:
                    commentary.append(txt.strip())

        return {
            "results": results,
            "source": "anthropic_web_search",
            "commentary": " ".join(commentary)[:1000] if commentary else None,
        }

    registry.register(Tool(
        name="web_search",
        description=(
            "Search the public web via Anthropic's server-side web_search; "
            "returns {title,url} hits plus optional commentary (~$0.01/call). "
            "Use for events/products you cannot identify confidently, current "
            "research, time-sensitive facts. Avoid when you know the answer, "
            "for deep multi-source research (use spawn_subagent), or when you "
            "already have the URL (use web_fetch). Query: be specific, include "
            "year ('Berlin Marathon 2026'), use domain terms. CITE URLs."
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
        display_label="Suche online",
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
            "Plain HTTP GET on a URL, returns HTML-stripped text (truncated "
            "~8000 chars, 15s timeout). Use after web_search to read a hit, "
            "for athlete-supplied links, or to verify facts at the source. "
            "Avoid for broad multi-page research (use spawn_subagent) or for "
            "JS-rendered SPAs/auth-walled pages (returns empty). Cite the URL."
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
        display_label="Lese eine Webseite",
    ))
