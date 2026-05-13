"""LLM backend for AgenticSports using LiteLLM for provider-agnostic access.

Provides:
    - chat_completion(): Clean interface for all LLM calls (uses litellm.completion)
    - get_client(): Backward-compatible Gemini client for embeddings only
    - MODEL: Default model identifier (overridable via AGENTICSPORTS_MODEL env var)
    - test_connection(): Quick connectivity check
"""

import os
import logging
import time

import litellm
from litellm.exceptions import RateLimitError
from google import genai

logger = logging.getLogger(__name__)

# Anthropic prompt caching notes (https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
# - 5-minute TTL default. Reads cost 10% of base input tokens. Writes cost 125%.
# - Cache invalidates on ANY change to system or tools, so the cached prefix
#   must be stable across turns.
# - Min cacheable size: 1024 tokens for Haiku, 2048 for Sonnet. Our static
#   system prompt is ~2k+ tokens so we are safe; below threshold we skip
#   caching to avoid an "invalid_request_error".

# Min characters as a cheap proxy for tokens (1 token ~= 4 chars). 1024
# tokens ~= 4096 chars; use a conservative 4000 to skip caching for tiny
# system prompts like test fixtures.
_MIN_CACHE_CHARS = 4000

# Rate-limit retry policy (Anthropic 429s and other provider 429s):
# - max 3 attempts total
# - base delay derived from Retry-After header if present, else 5s
# - exponential multiplier of 2 between attempts
_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_BASE_DELAY_S = 5.0
_RATE_LIMIT_BACKOFF_MULTIPLIER = 2.0

# Default model -- override with AGENTICSPORTS_MODEL env var
# LiteLLM format: "provider/model" (e.g. "gemini/gemini-2.5-flash", "openai/gpt-4o")
MODEL = os.environ.get("AGENTICSPORTS_MODEL", "gemini/gemini-2.5-flash")

# Suppress litellm's noisy info logging unless the user turns it on
litellm.suppress_debug_info = True

# Anthropic requires a tools= param whenever the conversation history
# contains tool_use messages (e.g. the agent retries without tools).
# Enabling modify_params lets LiteLLM inject a dummy tool so the call
# succeeds instead of erroring with UnsupportedParamsError.
litellm.modify_params = True


def chat_completion(
    messages: list[dict],
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    model: str | None = None,
) -> litellm.ModelResponse:
    """Perform a synchronous chat completion via LiteLLM.

    This is the primary LLM interface for the entire codebase. All callers
    should use OpenAI-compatible message format:
        [{"role": "system"/"user"/"assistant"/"tool", "content": "..."}]

    Args:
        messages: Conversation messages in OpenAI format.
        system_prompt: If provided, prepended as a system message.
        tools: OpenAI-format tool definitions (list of dicts).
        temperature: Sampling temperature.
        model: Model to use (defaults to MODULE-level MODEL).

    Returns:
        litellm.ModelResponse (OpenAI-compatible response object).
    """
    resolved_model = model or MODEL
    is_anthropic = "anthropic" in resolved_model or "claude" in resolved_model

    # Build final message list. For Anthropic, wrap the system prompt in a
    # content-blocks structure carrying cache_control so the (large) static
    # coaching prompt is cached for 5 minutes. This cuts token cost ~90%
    # and excludes cached input tokens from the per-minute rate limit.
    final_messages = list(messages)
    if system_prompt:
        # Only cache when the prompt is large enough to clear Anthropic's
        # minimum cacheable size (Haiku >= 1024 tokens, Sonnet >= 2048).
        cache_system = is_anthropic and len(system_prompt) >= _MIN_CACHE_CHARS
        if cache_system:
            final_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ] + final_messages
        else:
            final_messages = [
                {"role": "system", "content": system_prompt}
            ] + final_messages

    kwargs: dict = {
        "model": resolved_model,
        "messages": final_messages,
        "temperature": temperature,
        "drop_params": True,
    }

    if tools:
        if is_anthropic and tools:
            # Anthropic native tool_search: deferred tools only expose their
            # NAME in the prompt, descriptions are fetched on demand via the
            # tool_search helper. Saves ~80% of tool-layer tokens. Tool
            # entries that opt into deferral already carry the
            # `defer_loading: True` flag from get_openai_tools(defer_non_core=True).
            cached_tools = [dict(t) for t in tools]
            has_deferred = any(t.get("defer_loading") for t in cached_tools)
            if has_deferred:
                cached_tools.append({
                    "type": "tool_search_tool_bm25_20251119",
                    "name": "tool_search_tool_bm25",
                })
            # Cache the whole tools block (stable structure across turns).
            cached_tools[-1] = {
                **cached_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }
            kwargs["tools"] = cached_tools
        else:
            kwargs["tools"] = tools

    if "gemini-2.5" in resolved_model:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}

    response = _completion_with_rate_limit_retry(kwargs)
    _log_cache_usage(response, resolved_model)
    return response


def _completion_with_rate_limit_retry(kwargs: dict) -> litellm.ModelResponse:
    """Call litellm.completion with exponential backoff on RateLimitError.

    Only ``RateLimitError`` is retried. Any other exception propagates
    immediately so the agent loop can surface real failures. The base
    delay is read from the ``Retry-After`` response header when present
    (LiteLLM exposes ``exc.response.headers`` on most providers).
    """
    delay = _RATE_LIMIT_BASE_DELAY_S
    for attempt in range(1, _RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return litellm.completion(**kwargs)
        except RateLimitError as exc:
            if attempt >= _RATE_LIMIT_MAX_ATTEMPTS:
                logger.error(
                    "Rate limit exhausted after %d attempts: %s",
                    attempt, exc,
                )
                raise

            retry_after = _extract_retry_after(exc)
            sleep_s = retry_after if retry_after is not None else delay
            logger.warning(
                "Retrying after rate limit, attempt %d/%d, sleeping %s s",
                attempt, _RATE_LIMIT_MAX_ATTEMPTS, sleep_s,
            )
            time.sleep(sleep_s)
            delay *= _RATE_LIMIT_BACKOFF_MULTIPLIER

    # Unreachable - the loop either returns or raises.
    raise RuntimeError("rate-limit retry loop fell through")


def _extract_retry_after(exc: Exception) -> float | None:
    """Return Retry-After header value in seconds, or None if absent."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _log_cache_usage(response: litellm.ModelResponse, model: str) -> None:
    """Emit cache hit/write counters so we can measure cache effectiveness."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    cache_writes = getattr(usage, "cache_creation_input_tokens", None)
    cache_reads = getattr(usage, "cache_read_input_tokens", None)
    if cache_writes is None and cache_reads is None:
        return
    logger.info(
        "LLM cache usage model=%s writes=%s reads=%s",
        model, cache_writes or 0, cache_reads or 0,
    )

    # Record into the in-memory telemetry buffer for /admin/cache-stats.
    # Wrapped so observability failures never bubble into the agent loop.
    try:
        from src.services.cache_telemetry import get_telemetry
        get_telemetry().record_call(
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            cache_creation_tokens=cache_writes or 0,
            cache_read_tokens=cache_reads or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        # ITPM warning at 80% of Tier 1 Haiku limit (50_000 tokens / minute).
        itpm = get_telemetry().itpm(60)
        if itpm > 40000:
            logger.warning("ITPM=%d approaching Anthropic Tier 1 limit 50000", itpm)
    except Exception:
        pass  # observability must never break the agent loop


def get_client() -> genai.Client:
    """Create a Gemini client for embedding operations.

    Retained for backward compatibility -- used by user_model.py for
    embed_content() calls. All chat/generation should use chat_completion().
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


def chat_completion_with_fallback(
    messages: list[dict],
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    temperature: float | None = None,
) -> litellm.ModelResponse:
    """Try models in sequence from settings.fallback_models until one succeeds.

    Iterates through ``settings.fallback_models`` (comma-separated list from
    ``LITELLM_FALLBACK_MODELS`` env var) and returns the first successful
    response.  If every model fails, re-raises the last exception.

    Args:
        messages: Conversation messages in OpenAI format.
        system_prompt: Optional system prompt prepended to messages.
        tools: Optional OpenAI-format tool definitions.
        temperature: Sampling temperature (defaults to settings.agent_temperature).

    Returns:
        litellm.ModelResponse from the first model that succeeds.

    Raises:
        Exception: The last exception raised when all models fail.
    """
    from src.config import get_settings

    settings = get_settings()
    resolved_temperature = temperature if temperature is not None else settings.agent_temperature
    models = settings.fallback_models

    last_error: Exception | None = None
    for model in models:
        try:
            return chat_completion(
                messages,
                system_prompt=system_prompt,
                tools=tools,
                temperature=resolved_temperature,
                model=model,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Model %s failed: %s — trying next...", model, exc)

    raise last_error or RuntimeError("All fallback models failed")


def test_connection() -> str:
    """Send a test prompt via LiteLLM and return the response text."""
    response = chat_completion(
        messages=[{"role": "user", "content": "Say 'AgenticSports connected successfully' and nothing else."}],
        temperature=0.0,
    )
    return response.choices[0].message.content
