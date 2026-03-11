"""Tool result truncation with LLM-based compression.

When a tool's output exceeds its token budget, the result is compressed
via a fast LLM call (Haiku/Flash) that preserves numeric values and key
facts while drastically reducing token count.

Usage::

    from src.agent.tools.truncation import execute_with_budget

    result = execute_with_budget(registry, "get_activities", {"days": 90})
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Per-tool token budgets.  Tools not listed use the default budget.
PER_TOOL_BUDGET: dict[str, int] = {
    "get_activities": 1500,
    "analyze_training_load": 800,
    "web_search": 2000,
    "create_training_plan": 2000,
    "get_health_activity_summary": 1200,
    "get_cross_source_load_summary": 1000,
    "list_health_activities": 1500,
    "list_garmin_activities": 1500,
}

# The compression model — fast and cheap.
_COMPRESSION_MODEL = "gemini/gemini-2.5-flash"

_COMPRESSION_PROMPT = (
    "Compress the following JSON tool output into a shorter JSON object. "
    "RULES:\n"
    "1. Preserve ALL numeric values exactly (times, distances, heart rates, scores).\n"
    "2. Preserve ALL dates and timestamps.\n"
    "3. Remove redundant fields, verbose descriptions, and duplicate entries.\n"
    "4. Merge repeated patterns into counts/summaries.\n"
    "5. Output valid JSON only — no markdown, no explanation.\n"
    "6. Target ≤{budget} tokens.\n\n"
    "INPUT:\n{text}"
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count: ~4 chars per token."""
    return len(text) // 4


def _compress_with_llm(text: str, budget_tokens: int) -> str | None:
    """Compress text via a fast LLM call.

    Returns the compressed text, or None on any failure (caller should
    fall back to the original output).
    """
    try:
        from src.agent.llm import chat_completion

        prompt = _COMPRESSION_PROMPT.format(budget=budget_tokens, text=text[:32000])
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=_COMPRESSION_MODEL,
        )
        compressed = (response.choices[0].message.content or "").strip()
        if not compressed:
            return None
        return compressed
    except Exception:
        logger.warning("LLM compression failed — using fallback", exc_info=True)
        return None


def execute_with_budget(
    tool_registry: "ToolRegistry",  # noqa: F821
    name: str,
    args: dict,
    budget_tokens: int = 2000,
) -> dict:
    """Execute a tool and compress its result if it exceeds the token budget.

    Compression pipeline:
    1. Execute tool normally.
    2. Serialize result to JSON and estimate token count.
    3. If under budget → return as-is (fast path).
    4. If over budget → call LLM to compress, preserving numbers.
    5. If LLM fails → fall back to naive truncation.

    Args:
        tool_registry: The registry to dispatch the call through.
        name: Tool name.
        args: Tool arguments.
        budget_tokens: Default token budget (overridden by PER_TOOL_BUDGET).

    Returns:
        The tool result dict, possibly compressed.
    """
    effective_budget = PER_TOOL_BUDGET.get(name, budget_tokens)
    result = tool_registry.execute(name, args)

    # Serialize for size estimation
    try:
        text = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(result)

    estimated_tokens = _estimate_tokens(text)

    # Fast path: under budget
    if estimated_tokens <= effective_budget:
        return result

    logger.info(
        "Tool %s output %d tokens (budget %d) — compressing",
        name, estimated_tokens, effective_budget,
    )

    # Try LLM compression
    compressed = _compress_with_llm(text, effective_budget)
    if compressed is not None:
        compressed_tokens = _estimate_tokens(compressed)
        # Parse back to dict if valid JSON
        try:
            compressed_dict = json.loads(compressed)
            return {
                **compressed_dict,
                "_compressed": True,
                "_original_tokens": estimated_tokens,
                "_compressed_tokens": compressed_tokens,
            }
        except json.JSONDecodeError:
            # LLM returned non-JSON — use as raw string
            return {
                "result": compressed,
                "_compressed": True,
                "_original_tokens": estimated_tokens,
                "_compressed_tokens": compressed_tokens,
            }

    # Fallback: naive truncation
    char_budget = effective_budget * 4
    truncated = text[:char_budget]
    return {
        "result": truncated,
        "_truncated": True,
        "_note": f"... [truncated from {estimated_tokens} to ~{effective_budget} tokens]",
    }
