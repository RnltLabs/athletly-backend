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
            "Search the public web for up-to-date or factual information beyond "
            "the model's built-in knowledge. Backed by Anthropic's server-side "
            "web_search tool - one API call returns a list of {title, url} hits "
            "plus optional commentary text. Cost is roughly $0.01 per call.\n\n"
            "WHEN TO USE:\n"
            "- A specific race, event, product, or venue the athlete named that "
            "you cannot identify confidently from memory (date, location, "
            "course): search with year included.\n"
            "- Current training-methodology research, recent sports-science "
            "publications, updated nutrition or recovery guidelines.\n"
            "- Verifying facts that may have changed since the model was trained "
            "(qualifying times, event dates, equipment specs).\n"
            "- The user explicitly asks for sources or for current info.\n\n"
            "WHEN NOT TO USE:\n"
            "- For things you already know with high confidence: just answer.\n"
            "- For deep multi-source research where intermediate noise would "
            "pollute context: use spawn_subagent (it runs its own web_search "
            "loop and returns only the synthesis).\n"
            "- For reading a specific URL you already have: use web_fetch.\n"
            "- For querying the athlete's own data: use get_activities, "
            "get_athlete_profile, etc.\n\n"
            "QUERY WRITING:\n"
            "- Be specific. Include year for events ('Berlin Marathon 2026', "
            "not 'Berlin Marathon').\n"
            "- Include language hint for non-English events ('Heidelberger "
            "Halbmarathon 2026 Datum').\n"
            "- Use sport-domain terms ('VO2max intervals norwegian protocol', "
            "not 'how to get faster').\n"
            "- One concept per query - run multiple searches for multi-faceted "
            "questions.\n\n"
            "EXAMPLES:\n"
            "- web_search(query='Berlin Marathon 2026 date registration deadline')\n"
            "- web_search(query='Heidelberger Halbmarathon 2026 Datum Strecke')\n"
            "- web_search(query='polarized vs pyramidal training 80/20 marathon "
            "evidence')\n\n"
            "FAILURE MODES:\n"
            "- Missing ANTHROPIC_API_KEY: returns {'results': [], 'source': "
            "'none', 'message': '...'}. Fall back to built-in knowledge and "
            "say so transparently.\n"
            "- API error: returns {'error': '...', 'fallback': 'Use your "
            "built-in knowledge.'}.\n\n"
            "Returns {'results': [{title, url}, ...], 'source': "
            "'anthropic_web_search', 'commentary': '...'}. CITE URLs in your "
            "response when you use search results - the athlete should see where "
            "facts came from."
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
            "Download a specific URL and return its content as plain text "
            "(HTML stripped, scripts/styles removed, truncated to ~8000 chars). "
            "Use this to read a full page after web_search returned a "
            "promising URL, or when the athlete shares a URL you should look "
            "at.\n\n"
            "WHEN TO USE:\n"
            "- web_search returned a URL with the answer in the snippet but "
            "you need details from the full page (course profile, registration "
            "rules, qualification standards).\n"
            "- The user pasted a link and asked about it.\n"
            "- You need to verify a fact directly from an authoritative source "
            "(race website, federation rules page).\n\n"
            "WHEN NOT TO USE:\n"
            "- You do not have a URL: web_search first, then web_fetch.\n"
            "- For broad research across many pages: use spawn_subagent so the "
            "intermediate fetches do not pollute context.\n"
            "- For pages behind authentication or JavaScript-only rendering: "
            "the fetcher does plain HTTP GET, no JS execution, no cookies. "
            "Single-page apps may return empty content.\n\n"
            "ARGS:\n"
            "- url: full URL including scheme (https://...).\n"
            "- extract_prompt: hint for what to look for. Currently informational "
            "(the fetcher returns the whole page text), but you can use it as a "
            "note for yourself about what to look for when reading the result.\n\n"
            "EXAMPLES:\n"
            "- web_fetch(url='https://www.bmw-berlin-marathon.com/en/')\n"
            "- web_fetch(url='https://results.virginmoneylondonmarathon.com/...', "
            "extract_prompt='find finish time and split times for bib 12345').\n\n"
            "LIMITS: 15s timeout, content truncated to 8000 chars. For long "
            "pages, the truncation may cut off relevant info - if so, do a "
            "targeted re-search with more specific terms instead of fetching "
            "again.\n\n"
            "Returns {'url': ..., 'content': '<plain text>', 'length': int} on "
            "success, or {'error': '...'} on network/parsing failure. Always "
            "cite the URL when you use the content."
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
