"""Embedding service - generates 768-dim vectors for pgvector storage.

Routes to Gemini ``text-embedding-004`` (default) or OpenAI
``text-embedding-3-small`` (fallback / explicit override). Both providers
target the same 768-dimensional vector space - OpenAI's native 1536 is
truncated server-side via the ``dimensions`` parameter (MRL-trained, so
truncation is geometrically meaningful).

Selection precedence:

1. ``EMBEDDING_MODEL`` env var if set explicitly (``gemini/...`` or
   ``openai/...``)
2. Gemini when ``GEMINI_API_KEY`` is present
3. OpenAI when ``OPENAI_API_KEY`` is present
4. Disabled - ``embed_text`` returns ``None``

All failures are non-fatal: callers fall back to no-embedding storage and
no-retrieval. Disable explicitly via ``ATHLETLY_EMBEDDINGS_DISABLED=1``.

Usage::

    from src.services.embeddings import embed_text

    vec = embed_text("summary text")
    if vec is not None:
        # store in pgvector
        ...
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol

logger = logging.getLogger(__name__)


# Dimension contract - every provider must produce vectors of this length.
EMBEDDING_DIM = 768

# Maximum characters sent to an embedding provider. Coach-generated summaries
# rarely exceed this; truncation keeps cost predictable.
MAX_EMBEDDING_INPUT_CHARS = 2000

# Provider identifiers.
PROVIDER_GEMINI = "gemini/text-embedding-004"
PROVIDER_OPENAI = "openai/text-embedding-3-small"

# Retry budget. Embeddings are best-effort - we do not block the agent.
_MAX_RETRIES = 2
_RETRY_BASE_DELAY_S = 0.5


class EmbeddingProvider(Protocol):
    """Minimal interface every provider must implement."""

    name: str

    def embed(self, text: str) -> list[float] | None:
        """Return a 768-dim embedding for *text* or ``None`` on failure."""
        ...


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


class GeminiEmbedding:
    """Gemini ``text-embedding-004`` - 768-dim native."""

    name = PROVIDER_GEMINI
    _model_id = "models/text-embedding-004"

    def embed(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            return None
        try:
            from src.agent.llm import get_client

            client = get_client()
            response = client.models.embed_content(
                model=self._model_id,
                contents=text[:MAX_EMBEDDING_INPUT_CHARS],
            )
            values = list(response.embeddings[0].values)
            if len(values) != EMBEDDING_DIM:
                logger.warning(
                    "Gemini returned %d-dim vector, expected %d",
                    len(values),
                    EMBEDDING_DIM,
                )
                return None
            return values
        except Exception as exc:
            logger.debug("Gemini embed_content failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# OpenAI implementation - HTTP via httpx so we do not need the SDK
# ---------------------------------------------------------------------------


class OpenAIEmbedding:
    """OpenAI ``text-embedding-3-small`` truncated to 768 dims."""

    name = PROVIDER_OPENAI
    _model_id = "text-embedding-3-small"
    _endpoint = "https://api.openai.com/v1/embeddings"

    def embed(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            return None

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY missing - cannot embed via OpenAI")
            return None

        try:
            import httpx

            payload = {
                "model": self._model_id,
                "input": text[:MAX_EMBEDDING_INPUT_CHARS],
                "dimensions": EMBEDDING_DIM,
                "encoding_format": "float",
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=15.0) as client:
                response = client.post(self._endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            values = list(data["data"][0]["embedding"])
            if len(values) != EMBEDDING_DIM:
                logger.warning(
                    "OpenAI returned %d-dim vector, expected %d",
                    len(values),
                    EMBEDDING_DIM,
                )
                return None
            return values
        except Exception as exc:
            logger.debug("OpenAI embeddings call failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Provider selection (cached lazy singleton)
# ---------------------------------------------------------------------------


_provider_cache: EmbeddingProvider | None = None
_provider_cache_initialized: bool = False


def _resolve_provider() -> EmbeddingProvider | None:
    """Pick the appropriate provider based on env. Returns ``None`` if disabled."""
    if os.environ.get("ATHLETLY_EMBEDDINGS_DISABLED") == "1":
        return None

    override = (os.environ.get("EMBEDDING_MODEL") or "").strip().lower()

    if override.startswith("openai/"):
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbedding()
        logger.warning(
            "EMBEDDING_MODEL=%s but OPENAI_API_KEY missing; embeddings disabled",
            override,
        )
        return None

    if override.startswith("gemini/") or override == "":
        if os.environ.get("GEMINI_API_KEY"):
            return GeminiEmbedding()
        # Fall through to OpenAI if Gemini key absent and OpenAI key present.
        if os.environ.get("OPENAI_API_KEY"):
            logger.info(
                "GEMINI_API_KEY missing, falling back to OpenAI embeddings"
            )
            return OpenAIEmbedding()
        logger.debug(
            "No embedding provider keys configured - embeddings disabled"
        )
        return None

    logger.warning(
        "Unknown EMBEDDING_MODEL=%s - embeddings disabled", override
    )
    return None


def get_embedding_provider() -> EmbeddingProvider | None:
    """Return the active provider singleton, or ``None`` if disabled."""
    global _provider_cache, _provider_cache_initialized
    if not _provider_cache_initialized:
        _provider_cache = _resolve_provider()
        _provider_cache_initialized = True
    return _provider_cache


def reset_provider_cache() -> None:
    """Reset the cached provider - used by tests after mutating env."""
    global _provider_cache, _provider_cache_initialized
    _provider_cache = None
    _provider_cache_initialized = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float] | None:
    """Generate a 768-dim embedding for *text*.

    Returns ``None`` on any failure - callers must handle that case
    gracefully (store without embedding, skip retrieval).

    Implements bounded exponential-backoff retry to absorb transient
    provider hiccups without blocking the agent loop.
    """
    if not text or not text.strip():
        return None

    provider = get_embedding_provider()
    if provider is None:
        return None

    for attempt in range(_MAX_RETRIES + 1):
        result = provider.embed(text)
        if result is not None:
            return result
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BASE_DELAY_S * (2 ** attempt))

    logger.warning(
        "Embedding failed after %d retries (provider=%s, text=%.40s...)",
        _MAX_RETRIES,
        provider.name,
        text,
    )
    return None


def active_provider_name() -> str | None:
    """Return the active provider identifier, or ``None`` when disabled."""
    provider = get_embedding_provider()
    return provider.name if provider else None
