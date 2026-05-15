"""Tests for src/agent/tools/web_search_tool.py.

Covers:
- Cache miss: fresh API call, response cached, returned with source='fresh'.
- Cache hit: second identical call returns source='cache' and hit_count++.
- Cache expiry: stale row (>30 days) is bypassed.
- API error path: rate limit / timeout / generic exception all return
  source='error' with results=[].
- Key normalization: case + whitespace collapses to the same cache key.
- Bounded num_results: 0 -> 1, 20 -> 10, default 5.

All real HTTP and all Supabase calls are mocked. No network access.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import httpx

import src.agent.tools.web_search_tool as ws_tool
from src.agent.tools.web_search_tool import (
    CACHE_TABLE,
    CACHE_TTL_DAYS,
    _bound_num_results,
    _compute_cache_key,
    web_search,
)
from src.services.web_search_metrics import get_web_search_metrics


# ---------------------------------------------------------------------------
# Fake Supabase client
# ---------------------------------------------------------------------------


class _FakeQuery:
    """Chainable fake for the Supabase query builder.

    Stores all operations + a callback that returns the final response.
    """

    def __init__(self, store: "_FakeSupabase", op: str = "select"):
        self._store = store
        self._op = op
        self._filters: dict[str, str] = {}
        self._gte: dict[str, str] = {}
        self._upsert_payload: dict | None = None
        self._update_payload: dict | None = None
        self._limit_n: int | None = None

    def select(self, *_, **__):
        self._op = "select"
        return self

    def eq(self, col, value):
        self._filters[col] = value
        return self

    def gte(self, col, value):
        self._gte[col] = value
        return self

    def limit(self, n):
        self._limit_n = n
        return self

    def upsert(self, payload):
        self._op = "upsert"
        self._upsert_payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._update_payload = payload
        return self

    def order(self, *_, **__):
        return self

    def execute(self):
        if self._op == "select":
            return self._store.handle_select(self)
        if self._op == "upsert":
            return self._store.handle_upsert(self)
        if self._op == "update":
            return self._store.handle_update(self)
        return MagicMock(data=[])


class _FakeSupabase:
    """A minimal in-memory fake for the bits of the Supabase client we use."""

    def __init__(self) -> None:
        # keyed by cache_key
        self.rows: dict[str, dict] = {}
        # raise_on_lookup / raise_on_store flags for error-path coverage
        self.raise_on_lookup = False
        self.raise_on_store = False
        # Track upsert / update counts for assertions
        self.upserts: list[dict] = []
        self.updates: list[dict] = []

    def table(self, name: str) -> _FakeQuery:
        assert name == CACHE_TABLE
        return _FakeQuery(self)

    # -- handlers ------------------------------------------------------

    def handle_select(self, query: _FakeQuery):
        if self.raise_on_lookup:
            raise RuntimeError("simulated supabase select failure")
        cache_key = query._filters.get("cache_key")
        if cache_key is None:
            return MagicMock(data=list(self.rows.values()))
        row = self.rows.get(cache_key)
        if row is None:
            return MagicMock(data=[])
        # Apply TTL filter (gte created_at). We compare ISO strings; fine
        # for our test data since both are UTC iso-format.
        cutoff = query._gte.get("created_at")
        if cutoff and row["created_at"] < cutoff:
            return MagicMock(data=[])
        return MagicMock(data=[row])

    def handle_upsert(self, query: _FakeQuery):
        if self.raise_on_store:
            raise RuntimeError("simulated supabase upsert failure")
        payload = query._upsert_payload or {}
        self.upserts.append(payload)
        self.rows[payload["cache_key"]] = dict(payload)
        return MagicMock(data=[payload])

    def handle_update(self, query: _FakeQuery):
        payload = query._update_payload or {}
        self.updates.append(payload)
        cache_key = query._filters.get("cache_key")
        if cache_key and cache_key in self.rows:
            self.rows[cache_key].update(payload)
        return MagicMock(data=[payload])


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_db():
    """Patch src.db.client.get_supabase to return a fresh fake."""
    db = _FakeSupabase()
    with patch("src.db.client.get_supabase", return_value=db):
        yield db


@pytest.fixture
def with_api_key():
    """Force the settings cache to return a populated Brave API key."""
    from src.config import get_settings

    get_settings.cache_clear()
    with patch.object(
        ws_tool, "get_settings",
        return_value=MagicMock(brave_search_api_key="test-key"),
    ):
        yield


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Each test starts with empty web_search metrics."""
    get_web_search_metrics().reset()
    yield
    get_web_search_metrics().reset()


def _brave_fake_response(num_results: int = 2) -> dict:
    """Return a Brave-shaped JSON body."""
    return {
        "web": {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "description": f"Snippet <strong>{i}</strong>.",
                    "page_age": "2026-04-01T00:00:00Z" if i == 0 else None,
                }
                for i in range(num_results)
            ],
        }
    }


class _MockHttpResponse:
    def __init__(self, status_code: int, body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text or str(body)

    def json(self) -> dict:
        return self._body


class _MockHttpClient:
    """Drop-in for httpx.Client used as a context manager."""

    def __init__(self, response: _MockHttpResponse | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.calls: list[tuple[str, dict, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get(self, url, params=None, headers=None):
        self.calls.append((url, params or {}, headers or {}))
        if self._exc is not None:
            raise self._exc
        return self._response


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


def test_bound_num_results_clamps():
    assert _bound_num_results(20) == 10
    assert _bound_num_results(0) == 1
    assert _bound_num_results(5) == 5
    assert _bound_num_results(-3) == 1
    assert _bound_num_results("not an int") == 5  # default fallback


def test_compute_cache_key_is_stable_and_normalized():
    a = _compute_cache_key("marathon training", 5, "any")
    b = _compute_cache_key("marathon training", 5, "any")
    assert a == b
    # Different num_results / freshness -> different key
    assert _compute_cache_key("x", 5, "any") != _compute_cache_key("x", 6, "any")
    assert _compute_cache_key("x", 5, "any") != _compute_cache_key("x", 5, "year")
    # 16 hex chars
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


# ---------------------------------------------------------------------------
# Cache miss -> fresh API call -> cached
# ---------------------------------------------------------------------------


def test_cache_miss_calls_api_and_stores_response(fake_db, with_api_key):
    """First call: API hit, response shaped, persisted to cache."""
    mock_http = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=2))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        result = web_search("Hyrox periodisation 12 weeks", num_results=5, freshness="any")

    assert result["source"] == "fresh"
    assert result["query"] == "Hyrox periodisation 12 weeks"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result 0"
    assert result["results"][0]["url"] == "https://example.com/0"
    # <strong> stripped from snippet
    assert "<strong>" not in result["results"][0]["snippet"]
    assert result["citation_hint"]  # non-empty
    # Cache row persisted under the canonical key
    expected_key = _compute_cache_key("Hyrox periodisation 12 weeks".lower(), 5, "any")
    assert expected_key in fake_db.rows
    stored = fake_db.rows[expected_key]
    assert stored["query"] == "Hyrox periodisation 12 weeks"
    assert stored["num_results"] == 5
    assert stored["freshness"] == "any"
    # source / cached_at / citation_hint stripped from persisted payload
    assert "source" not in stored["response"]
    assert "cached_at" not in stored["response"]
    # API was called exactly once
    assert len(mock_http.calls) == 1


def test_cache_hit_returns_source_cache_and_increments_hit_count(fake_db, with_api_key):
    """Second identical call hits cache, returns source='cache', bumps hit_count."""
    mock_http = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=1))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        first = web_search("marathon base phase", num_results=5, freshness="any")
        assert first["source"] == "fresh"

        second = web_search("marathon base phase", num_results=5, freshness="any")

    assert second["source"] == "cache"
    assert second["cached_at"] is not None
    assert len(second["results"]) == 1
    assert second["results"][0]["url"] == "https://example.com/0"
    # API called exactly once across both invocations
    assert len(mock_http.calls) == 1
    # hit_count update fired exactly once (from the second call)
    assert len(fake_db.updates) == 1
    assert fake_db.updates[0]["hit_count"] == 1


def test_cache_expiry_bypasses_stale_row(fake_db, with_api_key):
    """Rows older than 30 days are ignored; a fresh API call is made."""
    cache_key = _compute_cache_key("vo2max intervals", 5, "any")
    # Seed a row created 31 days ago.
    stale_iso = datetime.fromtimestamp(
        time.time() - (CACHE_TTL_DAYS + 1) * 86400, tz=timezone.utc
    ).isoformat()
    fake_db.rows[cache_key] = {
        "cache_key": cache_key,
        "query": "vo2max intervals",
        "num_results": 5,
        "freshness": "any",
        "response": {"query": "vo2max intervals", "results": [{"title": "Stale"}]},
        "hit_count": 99,
        "created_at": stale_iso,
        "last_hit_at": stale_iso,
    }

    mock_http = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=2))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        result = web_search("vo2max intervals", num_results=5, freshness="any")

    assert result["source"] == "fresh"
    assert len(mock_http.calls) == 1
    # Old stale row was overwritten by the upsert with a fresh created_at.
    stored = fake_db.rows[cache_key]
    assert stored["created_at"] > stale_iso
    # Stale "Stale" title is gone; new payload has the API results.
    titles = [r["title"] for r in stored["response"]["results"]]
    assert "Stale" not in titles


# ---------------------------------------------------------------------------
# API error paths
# ---------------------------------------------------------------------------


def test_api_rate_limit_returns_graceful_error(fake_db, with_api_key):
    mock_http = _MockHttpClient(response=_MockHttpResponse(429, {}, text="rate limited"))
    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        result = web_search("anything", num_results=5, freshness="any")
    assert result["source"] == "error"
    assert "rate limit" in result["error"].lower()
    assert result["results"] == []
    # Nothing cached on error.
    assert fake_db.rows == {}


def test_api_timeout_returns_graceful_error(fake_db, with_api_key):
    mock_http = _MockHttpClient(exc=httpx.TimeoutException("boom"))
    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        result = web_search("anything", num_results=5, freshness="any")
    assert result["source"] == "error"
    assert "timeout" in result["error"].lower()
    assert result["results"] == []
    assert fake_db.rows == {}


def test_api_generic_exception_returns_graceful_error(fake_db, with_api_key):
    mock_http = _MockHttpClient(exc=RuntimeError("kaboom"))
    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        result = web_search("anything", num_results=5, freshness="any")
    assert result["source"] == "error"
    assert "error" in result["error"].lower()
    assert result["results"] == []


def test_missing_api_key_returns_graceful_error(fake_db):
    """No BRAVE_SEARCH_API_KEY -> graceful error, no HTTP call attempted."""
    from src.config import get_settings

    get_settings.cache_clear()
    with patch.object(
        ws_tool, "get_settings",
        return_value=MagicMock(brave_search_api_key=""),
    ):
        result = web_search("anything", num_results=5, freshness="any")
    assert result["source"] == "error"
    assert "api key" in result["error"].lower()


def test_empty_query_returns_graceful_error(fake_db, with_api_key):
    result = web_search("   ", num_results=5, freshness="any")
    assert result["source"] == "error"
    assert "empty" in result["error"].lower()
    assert result["results"] == []


# ---------------------------------------------------------------------------
# Key normalization (case + whitespace)
# ---------------------------------------------------------------------------


def test_key_normalization_unifies_case_and_whitespace(fake_db, with_api_key):
    """Three different surface forms hit the same cache row."""
    mock_http = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=1))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        r1 = web_search("Marathon Training")
        r2 = web_search("marathon training")
        r3 = web_search("  marathon training  ")

    assert r1["source"] == "fresh"
    assert r2["source"] == "cache"
    assert r3["source"] == "cache"
    # Exactly one HTTP call, exactly one cache row.
    assert len(mock_http.calls) == 1
    assert len(fake_db.rows) == 1


# ---------------------------------------------------------------------------
# Bounded num_results in the integration path
# ---------------------------------------------------------------------------


def test_num_results_capped_at_ten_in_api_call(fake_db, with_api_key):
    """num_results=20 is clamped to 10 in the outgoing API params."""
    mock_http = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=10))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        result = web_search("running form drills", num_results=20, freshness="any")

    assert result["source"] == "fresh"
    # Brave was called with count=10, not 20.
    _, params, _ = mock_http.calls[0]
    assert params["count"] == 10
    # Result list is also bounded.
    assert len(result["results"]) <= 10


def test_freshness_maps_to_brave_filter(fake_db, with_api_key):
    """freshness='year' becomes Brave's 'py'. 'any' omits the filter."""
    mock_http = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=1))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http):
        web_search("foo year", num_results=5, freshness="year")
        web_search("foo any", num_results=5, freshness="any")

    _, params_year, _ = mock_http.calls[0]
    _, params_any, _ = mock_http.calls[1]
    assert params_year.get("freshness") == "py"
    assert "freshness" not in params_any


# ---------------------------------------------------------------------------
# Metrics integration
# ---------------------------------------------------------------------------


def test_metrics_track_fresh_cache_and_error(fake_db, with_api_key):
    """Metrics ring buffer records one entry per call with the right source."""
    metrics = get_web_search_metrics()
    mock_http_ok = _MockHttpClient(
        response=_MockHttpResponse(200, _brave_fake_response(num_results=1))
    )

    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http_ok):
        web_search("metric query", num_results=5, freshness="any")
        web_search("metric query", num_results=5, freshness="any")

    mock_http_err = _MockHttpClient(response=_MockHttpResponse(429, {}, text="rl"))
    with patch("src.agent.tools.web_search_tool.httpx.Client", return_value=mock_http_err):
        web_search("rate limited query", num_results=5, freshness="any")

    summary = metrics.summary()
    assert summary["total_calls"] == 3
    assert summary["fresh_calls"] == 1
    assert summary["cache_hits"] == 1
    assert summary["api_errors"] == 1
    assert 0.0 < summary["cache_hit_rate"] < 1.0
