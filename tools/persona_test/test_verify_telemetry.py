"""Tests for ``tools/persona_test/verify_telemetry.py``.

Downstream persona-test agents paste the fact pack verbatim into reports, so
both the line-by-line shape and the order of keys must stay stable.

Strategy: mock ``httpx.Client.get`` with a tiny fake that returns canned
JSON responses for the four admin endpoints, then assert against the
rendered fact pack string.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.persona_test.verify_telemetry import (  # noqa: E402
    _FACT_LINES,
    fetch_all,
    render_fact_pack,
)


# ---------------------------------------------------------------------------
# Canned fixtures matching the real admin endpoint shapes.
# ---------------------------------------------------------------------------


_CRITIC_STATS_OK = {
    "window_calls": 12,
    "accept_rate": 0.92,
    "regenerate_rate": 0.08,
    "regenerate_failed_rate": 0.0,
    "critic_error_rate": 0.0,
    "avg_critic_latency_ms": 410,
    "by_rule": {
        "no_em_dash": 0,
        "no_markdown": 1,
        "umlauts": 0,
        "no_fabricated_stats": 2,
        "no_premature_trends": 0,
        "language_mirror": 0,
        "details_before_metrics": 0,
        "sync_then_status": 0,
        "pace_format_correct": 0,
        "pace_comparison_directional": 0,
    },
}


_GATES_STATS_OK = {
    "window_records": 14,
    "totals": {
        "fails": 2,
        "regenerate_success": 1,
        "regenerate_failed": 0,
        "tool_forced_regenerate_success": 1,
        "tool_forced_regenerate_failed": 0,
    },
    "rates": {
        "temporal_freshness": {"fail_rate": 0.05, "runs": 12},
        "injury_persistence": {"fail_rate": 0.0, "runs": 10},
        "stats_grounding": {"fail_rate": 0.1, "runs": 11},
        "holistic_alert": {"fail_rate": 0.0, "runs": 9},
        "language_mirror": {"fail_rate": 0.0, "runs": 10},
    },
}


_PROMPT_METRICS_OK = {
    "window_minutes": 60.0,
    "total_responses_scanned": 12,
    "total_violations": 1,
    "violation_rate_pct": 8.33,
    "by_rule": {},
    "by_model": {},
    "by_variant": {},
    "recent_samples": [],
}


_CACHE_STATS_OK = {
    "window_calls": 30,
    "cache_hit_rate_last_20": 75.0,
    "itpm_last_60s": 4200,
    "itpm_last_300s_per_minute": 3900,
    "by_model": {},
}


_RESPONSES: dict[str, tuple[int, dict | str]] = {
    "/admin/critic-stats": (200, _CRITIC_STATS_OK),
    "/admin/gates-stats": (200, _GATES_STATS_OK),
    "/admin/prompt-metrics": (200, _PROMPT_METRICS_OK),
    "/admin/cache-stats": (200, _CACHE_STATS_OK),
}


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | str) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        if isinstance(self._body, str):
            raise json.JSONDecodeError("not json", self._body, 0)
        return self._body


def _make_fake_client(overrides: dict[str, tuple[int, dict | str]] | None = None):
    """Return a fake httpx.Client class that serves canned responses by path."""
    table: dict[str, tuple[int, dict | str]] = dict(_RESPONSES)
    if overrides:
        table.update(overrides)

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def get(self, url: str) -> _FakeResponse:
            for path, (status, body) in table.items():
                if path in url:
                    return _FakeResponse(status, body)
            return _FakeResponse(404, "")

    return _FakeClient


# ---------------------------------------------------------------------------
# Fact-pack rendering
# ---------------------------------------------------------------------------


def test_fact_pack_renders_all_documented_lines(monkeypatch) -> None:
    """All documented keys appear in the documented order with the canned values."""
    monkeypatch.setattr(httpx, "Client", _make_fake_client())

    merged = fetch_all("https://example.invalid", window_minutes=60)
    out = render_fact_pack(
        merged,
        pulled_at=dt.datetime(2026, 5, 14, 14, 35, 30, tzinfo=dt.timezone.utc),
    )

    # Frame markers exactly as documented.
    assert out.startswith("=== TELEMETRY FACT PACK (pulled 2026-05-14T14:35:30Z) ===\n")
    assert out.rstrip("\n").endswith("=== END TELEMETRY FACT PACK ===")

    # Documented line order: assert each documented label appears in turn.
    # Strip the two frame lines, then compare label:value pairs to _FACT_LINES.
    body_lines = out.splitlines()[1:-1]
    assert len(body_lines) == len(_FACT_LINES), (
        f"expected {len(_FACT_LINES)} body lines, got {len(body_lines)}"
    )
    for body_line, (label, _path) in zip(body_lines, _FACT_LINES):
        assert body_line.startswith(label + ": "), (
            f"line {body_line!r} did not start with documented label {label!r}"
        )


def test_fact_pack_emits_canned_values(monkeypatch) -> None:
    """Spot-check the leaf values against the canned fixtures."""
    monkeypatch.setattr(httpx, "Client", _make_fake_client())

    merged = fetch_all("https://example.invalid", window_minutes=60)
    out = render_fact_pack(
        merged, pulled_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    )

    assert "critic_stats.window_calls: 12" in out
    assert "critic_stats.accept_rate: 0.92" in out
    assert "critic_stats.by_rule.no_em_dash: 0" in out
    assert "critic_stats.by_rule.no_fabricated_stats: 2" in out
    assert "gates_stats.window_records: 14" in out
    assert "gates_stats.totals.fails: 2" in out
    assert "gates_stats.totals.tool_forced_regenerate_success: 1" in out
    assert "gates_stats.rates.temporal_freshness.fail_rate: 0.05" in out
    assert "prompt_metrics.window_minutes: 60.0" in out
    assert "prompt_metrics.total_responses_scanned: 12" in out
    assert "prompt_metrics.total_violations: 1" in out
    assert "prompt_metrics.violation_rate_pct: 8.33" in out
    assert "cache_stats.window_calls: 30" in out
    assert "cache_stats.cache_hit_rate_last_20: 75.0" in out
    assert "cache_stats.itpm_last_60s: 4200" in out
    assert "cache_stats.itpm_last_300s_per_minute: 3900" in out


def test_fact_pack_handles_missing_key(monkeypatch) -> None:
    """A key that's absent from the response renders as ``<missing>``, not crash."""
    truncated_critic = {
        "window_calls": 5,
        # 'accept_rate' missing on purpose.
        "by_rule": {"no_em_dash": 0},  # other rules absent.
    }
    monkeypatch.setattr(
        httpx,
        "Client",
        _make_fake_client({"/admin/critic-stats": (200, truncated_critic)}),
    )

    merged = fetch_all("https://example.invalid", window_minutes=60)
    out = render_fact_pack(merged, pulled_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))

    assert "critic_stats.window_calls: 5" in out
    assert "critic_stats.accept_rate: <missing>" in out
    assert "critic_stats.by_rule.no_em_dash: 0" in out
    assert "critic_stats.by_rule.no_markdown: <missing>" in out


def test_fact_pack_handles_failed_endpoint(monkeypatch) -> None:
    """A non-200 response marks every line in that section as failed, others survive."""
    monkeypatch.setattr(
        httpx,
        "Client",
        _make_fake_client({"/admin/gates-stats": (503, "")}),
    )

    merged = fetch_all("https://example.invalid", window_minutes=60)
    out = render_fact_pack(merged, pulled_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))

    # gates_stats lines all carry the failure marker.
    for line in out.splitlines():
        if line.startswith("gates_stats."):
            assert "<endpoint failed: status 503>" in line, line

    # Other sections are still healthy.
    assert "critic_stats.window_calls: 12" in out
    assert "prompt_metrics.window_minutes: 60.0" in out
    assert "cache_stats.window_calls: 30" in out


def test_fact_pack_window_minutes_propagates(monkeypatch) -> None:
    """The --window-minutes value is passed through into the prompt-metrics URL."""
    captured_urls: list[str] = []

    class _CapturingClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def get(self, url: str) -> _FakeResponse:
            captured_urls.append(url)
            for path, (status, body) in _RESPONSES.items():
                if path in url:
                    return _FakeResponse(status, body)
            return _FakeResponse(404, "")

    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    fetch_all("https://example.invalid", window_minutes=30)

    prompt_calls = [u for u in captured_urls if "prompt-metrics" in u]
    assert len(prompt_calls) == 1
    assert "window_minutes=30" in prompt_calls[0]


def test_fact_pack_no_em_or_en_dash() -> None:
    """The fact pack skeleton must never contain em-dash or en-dash chars."""
    # Render with empty merged dict: every line marks <missing>, no real data needed.
    out = render_fact_pack({}, pulled_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    assert "—" not in out  # em-dash
    assert "–" not in out  # en-dash
