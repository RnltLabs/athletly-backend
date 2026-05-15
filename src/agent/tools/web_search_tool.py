"""web_search tool with a query-hash-based Postgres cache.

The coach researches sport-specific knowledge via this tool: periodisation
patterns, unknown sports, training methodologies, equipment requirements,
race information, etc. Hardcoded sport-knowledge in the system prompt is
being removed in favour of just-in-time research through this tool.

Cache design:
- Key = sha256(query.strip().lower() + ":" + num_results + ":" + freshness)[:16].
- NOT user-specific: 'Hyrox periodisation 12 weeks' returns the same advice
  for every athlete. Sharing the cache across users is the whole point.
- TTL: 30 days (enforced at read time).
- Backend: Postgres table public.web_search_cache (Supabase). We are
  already on Supabase, so no new infrastructure.

Search backend: Brave Search API. Chosen because BRAVE_SEARCH_API_KEY is
already in our .env scaffold (see .env.example) and pre-paid via the
existing search-infra contract, so no new vendor onboarding is needed.

Error handling is intentionally graceful: any failure - missing API key,
rate limit, timeout, network error - returns an empty results dict with
an ``error`` field. The coach can continue without search; it just
loses the research arm for that turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings
from src.services.web_search_metrics import get_web_search_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_TTL_DAYS: int = 30
DEFAULT_NUM_RESULTS: int = 5
MIN_NUM_RESULTS: int = 1
MAX_NUM_RESULTS: int = 10
ALLOWED_FRESHNESS: frozenset[str] = frozenset({"any", "week", "month", "year"})
DEFAULT_FRESHNESS: str = "any"
SEARCH_TIMEOUT_S: float = 8.0
BRAVE_ENDPOINT: str = "https://api.search.brave.com/res/v1/web/search"
CACHE_TABLE: str = "web_search_cache"

# Brave's `freshness` parameter accepts: pd (past day), pw (past week),
# pm (past month), py (past year). We do not expose "any" downstream;
# we just omit the parameter.
_BRAVE_FRESHNESS_MAP: dict[str, str] = {
    "week": "pw",
    "month": "pm",
    "year": "py",
}

CITATION_HINT: str = (
    "When using these results, cite the source explicitly. "
    "Format: 'laut <title> (<url>)' or 'siehe <url>'. "
    "Do not paraphrase a claim from search without naming the URL."
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def web_search(
    query: str,
    num_results: int = DEFAULT_NUM_RESULTS,
    freshness: str = DEFAULT_FRESHNESS,
) -> dict:
    """Search the public web with a 30-day query-hash cache.

    Args:
        query: Free-text query. Stripped and lowercased for cache key.
        num_results: Hits to return. Bounded to [1, 10]; default 5.
        freshness: One of "any", "week", "month", "year". Anything else
            is coerced to "any".

    Returns:
        A dict with the shape documented in the module docstring. Never
        raises; failures appear in the ``error`` field with ``source``
        set to "error" and ``results`` empty.
    """
    started = time.monotonic()
    metrics = get_web_search_metrics()

    # 1. Normalize input.
    normalized_query = (query or "").strip()
    if not normalized_query:
        latency = int((time.monotonic() - started) * 1000)
        metrics.record("error", latency)
        return _error_response(query or "", "empty query", started)

    bounded_results = _bound_num_results(num_results)
    normalized_freshness = freshness if freshness in ALLOWED_FRESHNESS else DEFAULT_FRESHNESS

    cache_key = _compute_cache_key(normalized_query, bounded_results, normalized_freshness)

    # 2. Try cache lookup (with TTL filter).
    cache_hit = _lookup_cache(cache_key)
    if cache_hit is not None:
        latency = int((time.monotonic() - started) * 1000)
        metrics.record("cache", latency)
        return _shape_cache_response(cache_hit, normalized_query)

    # 3. Cache miss: call the search API.
    settings = get_settings()
    api_key = settings.brave_search_api_key
    if not api_key:
        logger.warning("web_search: BRAVE_SEARCH_API_KEY missing; returning empty")
        latency = int((time.monotonic() - started) * 1000)
        metrics.record("error", latency)
        return _error_response(
            normalized_query,
            "search api key not configured",
            started,
        )

    try:
        api_results = _call_brave_api(
            api_key=api_key,
            query=normalized_query,
            num_results=bounded_results,
            freshness=normalized_freshness,
        )
    except _RateLimitError:
        latency = int((time.monotonic() - started) * 1000)
        metrics.record("error", latency)
        return _error_response(normalized_query, "search rate limit", started)
    except _TimeoutError:
        latency = int((time.monotonic() - started) * 1000)
        metrics.record("error", latency)
        return _error_response(normalized_query, "search timeout", started)
    except Exception as exc:
        logger.warning("web_search: brave api error: %s", exc)
        latency = int((time.monotonic() - started) * 1000)
        metrics.record("error", latency)
        return _error_response(normalized_query, f"search error: {exc}", started)

    response = {
        "query": normalized_query,
        "results": api_results,
        "source": "fresh",
        "cached_at": None,
        "citation_hint": CITATION_HINT,
    }

    # 4. Store in cache. Failure here is non-fatal: we still return the
    #    fresh results to the caller.
    _store_cache(
        cache_key=cache_key,
        query=normalized_query,
        num_results=bounded_results,
        freshness=normalized_freshness,
        response=response,
    )

    latency = int((time.monotonic() - started) * 1000)
    metrics.record("fresh", latency)
    return response


# ---------------------------------------------------------------------------
# Helpers - normalization, cache key, response shaping
# ---------------------------------------------------------------------------


def _bound_num_results(num_results: int) -> int:
    """Clamp num_results into [MIN_NUM_RESULTS, MAX_NUM_RESULTS]."""
    try:
        value = int(num_results)
    except (TypeError, ValueError):
        return DEFAULT_NUM_RESULTS
    if value < MIN_NUM_RESULTS:
        return MIN_NUM_RESULTS
    if value > MAX_NUM_RESULTS:
        return MAX_NUM_RESULTS
    return value


def _compute_cache_key(query: str, num_results: int, freshness: str) -> str:
    """Compute the 16-char hex cache key documented in the spec."""
    payload = f"{query.lower()}:{num_results}:{freshness}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _error_response(query: str, error: str, started: float) -> dict:
    """Build the standard graceful-failure response shape."""
    return {
        "query": query,
        "results": [],
        "source": "error",
        "error": error,
        "cached_at": None,
        "citation_hint": CITATION_HINT,
    }


def _shape_cache_response(cached: dict, query: str) -> dict:
    """Return the cached response with source overridden to 'cache'.

    The originally stored response has ``source='fresh'``. We rewrite it
    to ``source='cache'`` and stamp ``cached_at`` so the coach knows the
    answer is not live.
    """
    results = cached.get("results") or []
    cached_at = cached.get("cached_at_iso") or _now_iso()
    return {
        "query": cached.get("query") or query,
        "results": results,
        "source": "cache",
        "cached_at": cached_at,
        "citation_hint": CITATION_HINT,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helpers - cache I/O
# ---------------------------------------------------------------------------


def _lookup_cache(cache_key: str) -> dict | None:
    """Return the cached row payload, or None on miss / expiry / error.

    Filters out rows older than CACHE_TTL_DAYS. On hit, increments
    hit_count and updates last_hit_at (best-effort, errors logged but
    not raised).
    """
    try:
        from src.db.client import get_supabase
    except Exception:
        return None

    try:
        client = get_supabase()
    except Exception as exc:
        logger.debug("web_search: supabase unavailable: %s", exc)
        return None

    # TTL filter: created_at >= now() - 30 days. We compute the cutoff
    # client-side rather than relying on a server-side interval so the
    # query is easy to read in PostgREST and easy to mock in tests.
    cutoff = _ttl_cutoff_iso()

    try:
        resp = (
            client.table(CACHE_TABLE)
            .select("query, response, hit_count, created_at")
            .eq("cache_key", cache_key)
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("web_search: cache lookup failed: %s", exc)
        return None

    rows = getattr(resp, "data", None) or []
    if not rows:
        return None
    row = rows[0]

    stored_response = row.get("response") or {}
    # Defensive: response column is JSONB. Some clients may return it as
    # a string; tolerate that.
    if isinstance(stored_response, str):
        try:
            stored_response = json.loads(stored_response)
        except json.JSONDecodeError:
            stored_response = {}

    # Stamp cached_at from the row so callers can see how old the entry is.
    stored_response = dict(stored_response)
    stored_response["cached_at_iso"] = row.get("created_at") or _now_iso()

    # Best-effort: increment hit_count and update last_hit_at. We do not
    # block the response on this; failure is logged and ignored.
    _bump_hit_count(client, cache_key, int(row.get("hit_count") or 0))

    return stored_response


def _store_cache(
    *,
    cache_key: str,
    query: str,
    num_results: int,
    freshness: str,
    response: dict,
) -> None:
    """Upsert a fresh response into the cache. Errors are swallowed."""
    try:
        from src.db.client import get_supabase
    except Exception:
        return

    try:
        client = get_supabase()
    except Exception as exc:
        logger.debug("web_search: supabase unavailable for store: %s", exc)
        return

    payload = {
        "cache_key": cache_key,
        "query": query,
        "num_results": num_results,
        "freshness": freshness,
        # Strip transient fields before persisting; we'll re-add cached_at
        # and source on read.
        "response": _strip_transient(response),
        "hit_count": 0,
        "created_at": _now_iso(),
        "last_hit_at": _now_iso(),
    }

    try:
        client.table(CACHE_TABLE).upsert(payload).execute()
    except Exception as exc:
        logger.warning("web_search: cache store failed: %s", exc)


def _bump_hit_count(client: Any, cache_key: str, current: int) -> None:
    """Increment hit_count and refresh last_hit_at for a cache row."""
    try:
        client.table(CACHE_TABLE).update(
            {
                "hit_count": current + 1,
                "last_hit_at": _now_iso(),
            }
        ).eq("cache_key", cache_key).execute()
    except Exception as exc:
        logger.debug("web_search: hit_count bump failed: %s", exc)


def _strip_transient(response: dict) -> dict:
    """Drop fields that should not be persisted into the cache.

    ``source``, ``cached_at`` and ``citation_hint`` are recomputed on
    every read.
    """
    return {
        k: v
        for k, v in response.items()
        if k not in {"source", "cached_at", "citation_hint"}
    }


def _ttl_cutoff_iso() -> str:
    """Return the ISO cutoff for cache TTL filtering."""
    cutoff_ts = time.time() - CACHE_TTL_DAYS * 86400
    return datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helpers - Brave API call
# ---------------------------------------------------------------------------


class _RateLimitError(Exception):
    """Brave returned a 429."""


class _TimeoutError(Exception):
    """HTTP timeout (deadline 8s)."""


def _call_brave_api(
    *,
    api_key: str,
    query: str,
    num_results: int,
    freshness: str,
) -> list[dict]:
    """Call Brave Search API and return a normalized result list.

    Raises:
        _RateLimitError: When Brave returns HTTP 429.
        _TimeoutError: When the request exceeds SEARCH_TIMEOUT_S.
        Exception: For any other unexpected failure.
    """
    params: dict[str, str | int] = {
        "q": query,
        "count": num_results,
    }
    brave_freshness = _BRAVE_FRESHNESS_MAP.get(freshness)
    if brave_freshness:
        params["freshness"] = brave_freshness

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    try:
        with httpx.Client(timeout=SEARCH_TIMEOUT_S) as client:
            resp = client.get(BRAVE_ENDPOINT, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        raise _TimeoutError(str(exc)) from exc

    if resp.status_code == 429:
        raise _RateLimitError(f"brave rate limit: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise Exception(f"brave http {resp.status_code}: {resp.text[:200]}")

    try:
        body = resp.json()
    except Exception as exc:
        raise Exception(f"brave invalid json: {exc}") from exc

    web = body.get("web") or {}
    raw_results = web.get("results") or []
    return [_shape_brave_result(item) for item in raw_results[:num_results]]


def _shape_brave_result(item: dict) -> dict:
    """Reshape one Brave result into our flat response schema."""
    snippet = item.get("description") or item.get("snippet") or ""
    # Brave embeds <strong> tags in descriptions; strip them for a clean
    # plaintext excerpt.
    snippet = snippet.replace("<strong>", "").replace("</strong>", "")
    return {
        "title": item.get("title") or "",
        "url": item.get("url") or "",
        "snippet": snippet,
        "published_date": item.get("page_age") or item.get("age") or None,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_web_search_tool(registry: ToolRegistry) -> None:
    """Register the web_search tool on the given registry.

    Note: this REPLACES the legacy Anthropic-server-side web_search tool
    that lives in ``research_tools.py``. Order of registration matters:
    whichever ``register_*`` runs LAST wins. Callers are expected to run
    this after ``register_research_tools`` so the cached implementation
    wins. The legacy tool will be removed once all callers are migrated.
    """
    registry.register(
        Tool(
            name="web_search",
            description=(
                "Recherchiere sport-spezifisches Wissen via Web-Suche und "
                "cache das Ergebnis 30 Tage lang. Use this tool whenever "
                "you need: periodisation patterns for an unfamiliar sport, "
                "unknown training methodologies, equipment requirements, "
                "race-day logistics, or any external fact you are not sure "
                "of. Returns {query, results:[{title,url,snippet,"
                "published_date}], source:'fresh'|'cache', cached_at, "
                "citation_hint}. The athlete MUST see URLs cited in any "
                "claim that uses these results (format: 'laut <title> "
                "(<url>)' or 'siehe <url>'). Do NOT use this tool when "
                "you already know the answer with confidence. Do NOT use "
                "for athlete-specific data (use get_athlete_profile etc)."
            ),
            handler=web_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific search query. Include year for "
                            "events, sport name for sport-specific "
                            "patterns. Examples: 'Hyrox periodisation "
                            "12 weeks', 'Berlin Marathon 2026 Datum', "
                            "'cycling FTP improvement base phase'."
                        ),
                    },
                    "num_results": {
                        "type": "integer",
                        "description": (
                            "How many hits to return. Bounded to "
                            "[1, 10]; default 5."
                        ),
                        "minimum": MIN_NUM_RESULTS,
                        "maximum": MAX_NUM_RESULTS,
                    },
                    "freshness": {
                        "type": "string",
                        "description": (
                            "Recency filter. 'any' (default), 'week', "
                            "'month', 'year'. Use 'year' for "
                            "methodology research, 'month' for current "
                            "events."
                        ),
                        "enum": ["any", "week", "month", "year"],
                    },
                },
                "required": ["query"],
            },
            category="research",
            display_label="Recherchiere im Web",
        )
    )
