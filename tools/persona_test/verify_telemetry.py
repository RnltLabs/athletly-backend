"""Pull production telemetry and print a fact pack persona-test agents paste verbatim.

This is the ground-truth oracle every persona-test agent must consult before
making a claim about coach behaviour. It calls four admin endpoints and
renders a flat, machine-parsable fact pack. See
``tools/persona_test/FAILURE_MODE.md`` for why this exists.

Usage::

    uv run python tools/persona_test/verify_telemetry.py
    uv run python tools/persona_test/verify_telemetry.py --window-minutes 30
    uv run python tools/persona_test/verify_telemetry.py --json
    uv run python tools/persona_test/verify_telemetry.py --api-url http://localhost:8000

Exit codes:
    0 = success (some endpoints may have failed; failures are inlined as
        `<endpoint failed: status N>` markers)
    2 = transport-level error before any endpoint could be reached
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any

import httpx

DEFAULT_API_URL = os.environ.get("ATHLETLY_API_URL", "https://athletly.rnltlabs.de")
_HTTP_TIMEOUT_S = 30.0


# Lines emitted by the fact pack, in order. Each entry is
# (label, [path components into the merged JSON]). A missing path renders as
# ``<missing>`` rather than crashing. Section markers are interleaved
# implicitly by the leading prefix (``critic_stats.``, ``gates_stats.`` etc).
_FACT_LINES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("critic_stats.window_calls", ("critic_stats", "window_calls")),
    ("critic_stats.accept_rate", ("critic_stats", "accept_rate")),
    ("critic_stats.regenerate_rate", ("critic_stats", "regenerate_rate")),
    ("critic_stats.regenerate_failed_rate", ("critic_stats", "regenerate_failed_rate")),
    ("critic_stats.critic_error_rate", ("critic_stats", "critic_error_rate")),
    ("critic_stats.avg_critic_latency_ms", ("critic_stats", "avg_critic_latency_ms")),
    ("critic_stats.by_rule.no_em_dash", ("critic_stats", "by_rule", "no_em_dash")),
    ("critic_stats.by_rule.no_markdown", ("critic_stats", "by_rule", "no_markdown")),
    ("critic_stats.by_rule.umlauts", ("critic_stats", "by_rule", "umlauts")),
    ("critic_stats.by_rule.no_fabricated_stats", ("critic_stats", "by_rule", "no_fabricated_stats")),
    ("critic_stats.by_rule.no_premature_trends", ("critic_stats", "by_rule", "no_premature_trends")),
    ("critic_stats.by_rule.language_mirror", ("critic_stats", "by_rule", "language_mirror")),
    ("critic_stats.by_rule.details_before_metrics", ("critic_stats", "by_rule", "details_before_metrics")),
    ("critic_stats.by_rule.sync_then_status", ("critic_stats", "by_rule", "sync_then_status")),
    ("critic_stats.by_rule.pace_format_correct", ("critic_stats", "by_rule", "pace_format_correct")),
    ("critic_stats.by_rule.pace_comparison_directional", ("critic_stats", "by_rule", "pace_comparison_directional")),
    ("gates_stats.window_records", ("gates_stats", "window_records")),
    ("gates_stats.totals.fails", ("gates_stats", "totals", "fails")),
    ("gates_stats.totals.regenerate_success", ("gates_stats", "totals", "regenerate_success")),
    ("gates_stats.totals.regenerate_failed", ("gates_stats", "totals", "regenerate_failed")),
    ("gates_stats.totals.tool_forced_regenerate_success", ("gates_stats", "totals", "tool_forced_regenerate_success")),
    ("gates_stats.totals.tool_forced_regenerate_failed", ("gates_stats", "totals", "tool_forced_regenerate_failed")),
    ("gates_stats.rates.temporal_freshness.fail_rate", ("gates_stats", "rates", "temporal_freshness", "fail_rate")),
    ("gates_stats.rates.injury_persistence.fail_rate", ("gates_stats", "rates", "injury_persistence", "fail_rate")),
    ("gates_stats.rates.stats_grounding.fail_rate", ("gates_stats", "rates", "stats_grounding", "fail_rate")),
    ("gates_stats.rates.holistic_alert.fail_rate", ("gates_stats", "rates", "holistic_alert", "fail_rate")),
    ("gates_stats.rates.language_mirror.fail_rate", ("gates_stats", "rates", "language_mirror", "fail_rate")),
    ("prompt_metrics.window_minutes", ("prompt_metrics", "window_minutes")),
    ("prompt_metrics.total_responses_scanned", ("prompt_metrics", "total_responses_scanned")),
    ("prompt_metrics.total_violations", ("prompt_metrics", "total_violations")),
    ("prompt_metrics.violation_rate_pct", ("prompt_metrics", "violation_rate_pct")),
    ("cache_stats.window_calls", ("cache_stats", "window_calls")),
    ("cache_stats.cache_hit_rate_last_20", ("cache_stats", "cache_hit_rate_last_20")),
    ("cache_stats.itpm_last_60s", ("cache_stats", "itpm_last_60s")),
    ("cache_stats.itpm_last_300s_per_minute", ("cache_stats", "itpm_last_300s_per_minute")),
)


def _fetch_endpoint(client: httpx.Client, url: str) -> dict[str, Any] | str:
    """GET an admin endpoint. Returns parsed JSON dict on 200, error marker otherwise.

    The error marker is a string like ``<endpoint failed: status 503>`` and is
    stored under the section's slot in the merged JSON so each line that
    indexes into that section emits the marker rather than ``<missing>``.
    """
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        return f"<endpoint failed: transport error: {exc.__class__.__name__}>"
    if resp.status_code != 200:
        return f"<endpoint failed: status {resp.status_code}>"
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return "<endpoint failed: non-json body>"
    if not isinstance(data, dict):
        return "<endpoint failed: non-dict body>"
    return data


def fetch_all(api_url: str, window_minutes: int) -> dict[str, Any]:
    """Pull all four admin endpoints and return a merged dict.

    Each top-level key (``critic_stats``, ``gates_stats``, ``prompt_metrics``,
    ``cache_stats``) is either the parsed JSON dict or an error marker string.
    """
    base = api_url.rstrip("/")
    endpoints = {
        "critic_stats": f"{base}/admin/critic-stats",
        "gates_stats": f"{base}/admin/gates-stats",
        "prompt_metrics": f"{base}/admin/prompt-metrics?window_minutes={int(window_minutes)}",
        "cache_stats": f"{base}/admin/cache-stats",
    }
    merged: dict[str, Any] = {}
    with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
        for section, url in endpoints.items():
            merged[section] = _fetch_endpoint(client, url)
    return merged


def _lookup(merged: dict[str, Any], path: tuple[str, ...]) -> str:
    """Resolve a dotted path against the merged dict to a string for the fact pack.

    - If the top-level section is an error string, return that string for any
      key inside the section.
    - If any intermediate key is missing, return ``<missing>``.
    - Otherwise, return ``str(value)`` of the leaf.
    """
    if not path:
        return "<missing>"
    section_key = path[0]
    section = merged.get(section_key)
    if isinstance(section, str):
        # Endpoint failed: surface the failure marker on every line in that
        # section so the consumer can see exactly which lines are stale.
        return section
    if section is None:
        return "<missing>"

    cursor: Any = section
    for key in path[1:]:
        if not isinstance(cursor, dict) or key not in cursor:
            return "<missing>"
        cursor = cursor[key]
    return str(cursor)


def render_fact_pack(merged: dict[str, Any], *, pulled_at: dt.datetime) -> str:
    """Render the merged data as the literal block agents paste verbatim.

    The block format MUST stay stable: downstream agents parse it line by line.
    """
    ts = pulled_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"=== TELEMETRY FACT PACK (pulled {ts}) ==="]
    for label, path in _FACT_LINES:
        lines.append(f"{label}: {_lookup(merged, path)}")
    lines.append("=== END TELEMETRY FACT PACK ===")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull /admin/critic-stats, /admin/gates-stats, /admin/prompt-metrics, "
            "/admin/cache-stats and print a flat fact pack persona-test "
            "agents paste verbatim into reports."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API base URL. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=60,
        help="Window (minutes) for /admin/prompt-metrics. Default 60.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw merged JSON instead of the flat fact pack.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pulled_at = dt.datetime.now(dt.timezone.utc)
    try:
        merged = fetch_all(args.api_url, args.window_minutes)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: could not reach telemetry endpoints: {exc}\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(render_fact_pack(merged, pulled_at=pulled_at))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
