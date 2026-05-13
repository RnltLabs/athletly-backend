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

# Anthropic prompt caching notes (Q2 2026, source:
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
# - 5-minute TTL default. Reads cost 10% of base input tokens. Writes cost 125%.
# - Cache invalidates on ANY change to a cached block (system, tools, or a
#   message preceded by a cache_control breakpoint), so the cached prefix
#   must be byte-stable across turns.
# - Min cacheable size for claude-haiku-4-5 / claude-sonnet-4-x is 4096 tokens.
#   Below the threshold the cache_control flag is silently ignored by the API
#   (no error, no caching), which is exactly the bug we are fixing.
# - On claude-haiku-4-5 cache reads do NOT count toward ITPM (Input Tokens
#   Per Minute). This is different from haiku 3.5 which still counts reads
#   against ITPM (see the dagger marker in the rate-limit docs). For us this
#   means a healthy cache hit rate directly buys headroom inside our 50k
#   ITPM Tier 1 budget.
# - Up to 4 cache_control breakpoints per request. We use them for:
#     1. The static system prompt block (stable across all turns)
#     2. An optional per-turn runtime_context system block (stable within
#        the multi-round tool loop of a single user turn)
#     3. The last tool definition (stable across turns)
#     4. The last message in the conversation history (rotating breakpoint
#        so the growing tool-use history is reused on subsequent rounds)

# Min characters as a cheap proxy for tokens (1 token ~= 4 chars). The
# 4096-token Anthropic minimum maps to ~16384 chars; we use 16400 as a
# conservative round number so test fixtures with shorter prompts simply
# skip caching instead of failing.
_MIN_CACHE_CHARS = 16400

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


# Min characters required for a runtime_context block to get its own
# cache breakpoint. ~800 chars (~200 tokens) is the sweet spot: large
# enough that a 1-round write recovers itself on round 2, small enough
# that we do not split tiny per-turn payloads.
_MIN_RUNTIME_CTX_CACHE_CHARS = 800


def _add_message_cache_breakpoints(
    messages: list[dict],
    is_anthropic: bool,
) -> list[dict]:
    """Return a new message list with a rotating cache breakpoint on the
    LAST message.

    Anthropic only caches a prefix UP TO a cache_control marker. To reuse
    the conversation history across multiple tool rounds in the same user
    turn, we attach an ephemeral cache_control to the last block of the
    last message on every call. The breakpoint rotates forward with each
    round; everything before it stays byte-stable, so cache reads grow
    monotonically through the tool loop.

    The input list is NOT mutated (project immutability rule). Messages
    whose content was a plain string get rewrapped into a single-block
    list of content blocks, so LiteLLM can forward cache_control to
    Anthropic in the expected shape (see
    https://docs.litellm.ai/docs/completion/prompt_caching).

    Args:
        messages: OpenAI-format messages. Untouched on return.
        is_anthropic: Whether the target model is an Anthropic model.

    Returns:
        A new list of message dicts. Identical to input when not anthropic
        or when there are fewer than 2 messages.
    """
    if not is_anthropic or len(messages) < 2:
        return messages

    # Deep-copy via dict reconstruction; we only mutate the local copy of
    # the last message. Earlier messages stay shared (their dicts are not
    # rewritten).
    new_messages = [dict(m) for m in messages]
    last = new_messages[-1]
    role = last.get("role")

    content = last.get("content")
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        # Re-copy the list and the last block so we do not mutate caller state.
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
    else:
        # Unsupported content shape - bail out without caching.
        return messages

    if not blocks:
        return messages

    last_block = blocks[-1]
    if not isinstance(last_block, dict):
        return messages

    blocks[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}

    # role="tool" with string content gets rewrapped the same way; LiteLLM
    # forwards the cache_control through to the Anthropic tool_result block.
    new_last = {**last, "content": blocks}
    # Preserve any role-specific fields (e.g. tool_call_id) - dict spread above
    # already does that.
    _ = role  # role retained for future-proofing; logic is uniform per role.
    new_messages[-1] = new_last
    return new_messages


def chat_completion(
    messages: list[dict],
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    model: str | None = None,
    runtime_context: str | None = None,
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
        runtime_context: Optional per-turn context (date, athlete profile,
            plan summary, recovery status). For Anthropic models this is
            placed in a SECOND system block AFTER the static system prompt
            and gets its own ephemeral cache_control, so a single user turn
            can hit the cache on every follow-up tool round. For non-
            Anthropic models it is prepended to the first user message.

    Returns:
        litellm.ModelResponse (OpenAI-compatible response object).
    """
    resolved_model = model or MODEL
    is_anthropic = "anthropic" in resolved_model or "claude" in resolved_model

    # Build final message list. For Anthropic, wrap the system prompt in a
    # content-blocks structure carrying cache_control so the (large) static
    # coaching prompt is cached for 5 minutes. This cuts token cost ~90%
    # and excludes cached input tokens from the per-minute rate limit
    # (claude-haiku-4-5 specifically; haiku 3.5 still counts reads vs ITPM).
    final_messages = list(messages)

    # 1. Optionally inject runtime_context for non-anthropic providers as a
    #    prefix to the first user message. Anthropic gets a dedicated
    #    system block further below.
    if (
        runtime_context
        and not is_anthropic
        and final_messages
    ):
        first_user_idx = next(
            (i for i, m in enumerate(final_messages) if m.get("role") == "user"),
            None,
        )
        if first_user_idx is not None:
            existing = final_messages[first_user_idx]
            existing_content = existing.get("content") or ""
            merged = (
                f"[CONTEXT]\n{runtime_context}\n\n[USER MESSAGE]\n{existing_content}"
            )
            final_messages = (
                final_messages[:first_user_idx]
                + [{**existing, "content": merged}]
                + final_messages[first_user_idx + 1:]
            )

    # 2. Prepend the system prompt (and, for Anthropic, a runtime_context
    #    block) at the very front.
    if system_prompt:
        cache_system = is_anthropic and len(system_prompt) >= _MIN_CACHE_CHARS
        if cache_system:
            system_msg: dict = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
            prefix: list[dict] = [system_msg]

            if (
                runtime_context
                and len(runtime_context) >= _MIN_RUNTIME_CTX_CACHE_CHARS
            ):
                # Second system block: per-turn but stable across tool
                # rounds inside that turn. Own cache_control breakpoint
                # so the runtime context is a cache write on round 1 and
                # a cache read on every subsequent round.
                prefix.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": runtime_context,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                })
            elif runtime_context:
                # Too short to bother caching - inline as plain text after
                # the static block.
                prefix.append({
                    "role": "system",
                    "content": runtime_context,
                })

            final_messages = prefix + final_messages
        else:
            prefix = [{"role": "system", "content": system_prompt}]
            if runtime_context and is_anthropic:
                # System prompt too small to cache, but caller still passed
                # runtime_context. Emit it as an uncached second system block
                # so callers see consistent behaviour.
                prefix.append({"role": "system", "content": runtime_context})
            final_messages = prefix + final_messages

    # 3. Add a rotating cache breakpoint on the LAST message. This makes
    #    accumulating tool_use / tool_result history cacheable across the
    #    rounds of a single user turn.
    final_messages = _add_message_cache_breakpoints(final_messages, is_anthropic)

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
    """Emit per-call token counters so cache effectiveness is observable.

    Always logs, even when the provider returns zero/None for the cache
    fields. That way absent cache reads show up explicitly as
    cache_hit_pct=0 instead of silently disappearing, which is exactly
    the failure mode that hid the original caching bug.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    input_tokens = getattr(usage, "prompt_tokens", None) or 0
    output_tokens = getattr(usage, "completion_tokens", None) or 0
    cache_writes = getattr(usage, "cache_creation_input_tokens", None) or 0
    cache_reads = getattr(usage, "cache_read_input_tokens", None) or 0

    # total_input represents what the provider actually billed as input
    # for this call: fresh input + cache writes + cache reads. Different
    # providers expose this differently; we reconstruct it locally so the
    # log line is consistent across providers.
    total_input = input_tokens + cache_writes + cache_reads
    denom = max(1, total_input)
    cache_hit_pct = round(100 * cache_reads / denom)

    logger.info(
        "LLM call model=%s input=%s cache_create=%s cache_read=%s "
        "output=%s total_input=%s cache_hit_pct=%s",
        model,
        input_tokens,
        cache_writes,
        cache_reads,
        output_tokens,
        total_input,
        cache_hit_pct,
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
    runtime_context: str | None = None,
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
        runtime_context: Optional per-turn context block (see chat_completion).

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
                runtime_context=runtime_context,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Model %s failed: %s - trying next...", model, exc)

    raise last_error or RuntimeError("All fallback models failed")


def test_connection() -> str:
    """Send a test prompt via LiteLLM and return the response text."""
    response = chat_completion(
        messages=[{"role": "user", "content": "Say 'AgenticSports connected successfully' and nothing else."}],
        temperature=0.0,
    )
    return response.choices[0].message.content
