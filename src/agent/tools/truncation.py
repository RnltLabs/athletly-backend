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

# Substrings that indicate an auth-related failure on the compression provider.
# When any of these match the str(exception), we treat it as a credential
# problem and retry the call against the primary MODEL.
_AUTH_ERROR_HINTS: tuple[str, ...] = (
    "API_KEY_INVALID",
    "invalid key",
    "AuthenticationError",
)


def _get_compression_model() -> str:
    """Return the configured compression model.

    Resolved lazily so that tests / runtime overrides of the env var (or
    of get_settings) take effect.  Falls back to the historic default when
    settings cannot be loaded for any reason.
    """
    try:
        from src.config import get_settings

        return get_settings().compression_model
    except Exception:  # pragma: no cover - defensive
        logger.warning("could not load settings for compression model", exc_info=True)
        return "gemini/gemini-2.5-flash"


def _is_auth_error(exc: BaseException) -> bool:
    """Detect whether an exception signals an auth issue with the provider."""
    try:
        from litellm.exceptions import AuthenticationError, BadRequestError
    except Exception:  # pragma: no cover - litellm is a hard dep, but stay safe
        AuthenticationError = ()  # type: ignore[assignment]
        BadRequestError = ()  # type: ignore[assignment]

    if isinstance(exc, (AuthenticationError, BadRequestError)):
        return True
    message = str(exc)
    return any(hint in message for hint in _AUTH_ERROR_HINTS)


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


def _call_compression_llm(text: str, budget_tokens: int, model: str) -> str | None:
    """Invoke the LLM once with the compression prompt.

    Returns the (stripped) response text, or None when the model gave back
    an empty body.  Exceptions are not caught here so the caller can decide
    whether to retry with a different model.
    """
    from src.agent.llm import chat_completion

    prompt = _COMPRESSION_PROMPT.format(budget=budget_tokens, text=text[:32000])
    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        model=model,
    )
    compressed = (response.choices[0].message.content or "").strip()
    return compressed or None


def _compress_with_llm(text: str, budget_tokens: int) -> str | None:
    """Compress text via a fast LLM call, with auth-failure fallback.

    Pipeline:
    1. Try the configured compression model.
    2. If that call fails with an auth-like error (e.g. invalid Gemini key),
       retry ONCE against the primary ``MODEL`` from ``src.agent.llm`` using
       the same prompt and budget.
    3. If the fallback also fails (or the first failure is non-auth), return
       None so the caller can apply naive truncation.
    """
    primary_model = _get_compression_model()
    try:
        return _call_compression_llm(text, budget_tokens, primary_model)
    except Exception as primary_exc:
        logger.warning(
            "LLM compression failed model=%s error=%s",
            primary_model, primary_exc, exc_info=True,
        )
        if not _is_auth_error(primary_exc):
            return None

        from src.agent.llm import MODEL as fallback_model

        if fallback_model == primary_model:
            logger.warning(
                "compression fallback unavailable model=%s (primary == fallback)",
                primary_model,
            )
            return None

        try:
            result = _call_compression_llm(text, budget_tokens, fallback_model)
        except Exception as fallback_exc:
            logger.warning(
                "compression fallback failed model_primary=%s model_fallback=%s "
                "primary_error=%s fallback_error=%s",
                primary_model, fallback_model, primary_exc, fallback_exc,
                exc_info=True,
            )
            return None

        if result is None:
            logger.warning(
                "compression fallback returned empty model_primary=%s "
                "model_fallback=%s primary_error=%s",
                primary_model, fallback_model, primary_exc,
            )
            return None

        logger.info(
            "compression fallback ok model_primary=%s model_fallback=%s",
            primary_model, fallback_model,
        )
        return result


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
