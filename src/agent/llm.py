"""LLM backend for AgenticSports using LiteLLM for provider-agnostic access.

Provides:
    - chat_completion(): Clean interface for all LLM calls (uses litellm.completion)
    - get_client(): Backward-compatible Gemini client for embeddings only
    - MODEL: Default model identifier (overridable via AGENTICSPORTS_MODEL env var)
    - test_connection(): Quick connectivity check
"""

import os
import logging

import litellm
from google import genai

logger = logging.getLogger(__name__)

# Default model -- override with AGENTICSPORTS_MODEL env var
# LiteLLM format: "provider/model" (e.g. "gemini/gemini-2.5-flash", "openai/gpt-4o")
MODEL = os.environ.get("AGENTICSPORTS_MODEL", "gemini/gemini-2.5-flash")

# Suppress litellm's noisy info logging unless the user turns it on
litellm.suppress_debug_info = True


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

    # Build final message list
    final_messages = list(messages)
    if system_prompt:
        final_messages = [{"role": "system", "content": system_prompt}] + final_messages

    kwargs: dict = {
        "model": resolved_model,
        "messages": final_messages,
        "temperature": temperature,
        "drop_params": True,  # Provider compatibility -- drop unsupported params
    }

    if tools:
        kwargs["tools"] = tools

        # Gemini 2.5 "thinking" models need an explicit thinking budget
        # when tools + large system prompts are combined, otherwise they
        # return empty responses with zero completion tokens.
        if "gemini-2.5" in resolved_model:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    response = litellm.completion(**kwargs)
    return response


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
