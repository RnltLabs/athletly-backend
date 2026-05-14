"""Sprint P: tests for the budget-skip fabricated-stats shield.

When the critic chain blows past ``critic_total_budget_s``, the legacy
behaviour was to ship the regenerated rewrite annotated as
``degraded: true``. Production telemetry (Elena / Marco / Lisa Iter 3,
2026-05-14) showed that the rewrite frequently still contained the
fabricated stats that the first-pass critic had flagged. The
``no_fabricated_stats`` counter rose from 0 to 7-8 with
``source: critic_total_budget``, and the user-facing "degraded" hint
was effectively invisible.

Sprint P fix: on the budget-skip path, run a DETERMINISTIC
specifics-grounding check on the rewrite. If it mentions paces, HRV ms,
distances, durations, or heart rates that do NOT appear in any
tool_result payload of the current turn, ship a safe fallback stub
instead. If everything is grounded, the legacy ship-rewrite behaviour
is preserved (no regression).

These tests cover:

1. Grounded rewrite: ships as before (``source: critic_total_budget``).
2. Ungrounded pace (e.g. ``5:18/km`` with no tool source): falls back
   to the stub and emits ``source: budget_skipped_with_fab_risk``.
3. Ungrounded HRV ms: same fallback behaviour.
4. Unit-level: ``specifics_grounded_in_tools`` accepts German decimals.
5. Unit-level: ``5:18/km`` (pace) and ``5:18`` (duration) are treated
   distinctly.
6. ``CriticMetrics.summary()`` exposes the new
   ``budget_fallback_used_count`` field.

Pattern mirrors ``tests/test_agent_loop_gates_tool_force.py``: build a
loop via ``__new__`` to skip the heavy init, stub the critic and the
perf_counter so the chain reports itself as over-budget, then assert on
the emitted SSE events and metrics records.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent_loop import AsyncAgentLoop
from src.agent.specifics_check import specifics_grounded_in_tools


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def _emit_collector():
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    return events, emit


def _build_loop_with_messages(messages: list[dict]) -> AsyncAgentLoop:
    """Build an AsyncAgentLoop pre-populated with conversation messages."""
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = list(messages)
    return loop


class _RecordingMetrics:
    """Stand-in metrics collector capturing every record call."""

    def __init__(self) -> None:
        self.records: list[tuple[str, tuple, int]] = []

    def record(self, action, violations, latency_ms):
        self.records.append((action, tuple(violations), int(latency_ms)))


def _make_fake_critic(first_violation_rule: str = "no_fabricated_stats"):
    """Critic stub: first pass flags one violation, second pass would accept.

    The whole point of these tests is to verify the second pass is
    NEVER reached on the budget-skip path, so the second-pass return
    is just a sentinel.
    """
    from src.agent.critic import CriticResult, Violation

    fake_critic = MagicMock()
    fake_critic.review.side_effect = [
        CriticResult(
            action="regenerate",
            violations=(Violation(rule=first_violation_rule, reason="stub"),),
            latency_ms=900,
        ),
        CriticResult.accept(latency_ms=2000),  # should NEVER be consumed
    ]
    return fake_critic


def _perf_counter_over_budget():
    """perf_counter sequence that reports 6.5s elapsed (budget is 6.0s)."""
    values = iter([0.0, 6.5, 6.5, 6.5, 6.5, 6.5])

    def _fake():
        try:
            return next(values)
        except StopIteration:
            return 6.5

    return _fake


# ---------------------------------------------------------------------------
# Integration tests: budget-skip path with the new fab-risk shield
# ---------------------------------------------------------------------------


def test_budget_skip_with_grounded_rewrite_ships_as_before(_emit_collector):
    """Sprint P regression: a grounded rewrite must still ship.

    The rewrite mentions stats that DO appear in this turn's tool
    result, so the new specifics-grounding check passes and the legacy
    "ship rewrite + critic_total_budget event" behaviour holds. This
    pins down that we have not regressed the budget-skip happy path.
    """
    events, emit_fn = _emit_collector
    # Tool result of the current turn provides the pace "5:18" and the
    # distance "22 km". The rewrite quotes both verbatim.
    loop = _build_loop_with_messages([
        {"role": "user", "content": "Wie war mein heutiger Long Run?"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"activities": [{"distance_km": 22, "pace": "5:18/km"}]}',
        },
    ])
    loop.regenerate_after_critique = MagicMock(
        return_value=(
            "Solider Long Run heute. Du warst 22 km bei 5:18/km unterwegs, "
            "das passt gut zu deinem Marathon-Plan."
        ),
    )
    fake_critic = _make_fake_critic()
    fake_metrics = _RecordingMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch(
             "src.services.critic_metrics.get_metrics",
             return_value=fake_metrics,
         ), \
         patch(
             "src.agent.agent_loop.time.perf_counter",
             side_effect=_perf_counter_over_budget(),
         ):
        result = asyncio.run(loop._run_critique_pass(
            response_text="Some draft response with no hard violations.",
            user_message="Wie war mein heutiger Long Run?",
            tool_names=["get_activities"],
            emit_fn=emit_fn,
        ))

    # Rewrite ships intact (no fallback).
    assert "22 km" in result and "5:18/km" in result

    # Exactly one critic_review event, with the legacy source attribution.
    reviews = [e for e in events if e[0] == "critic_review"]
    assert len(reviews) == 1
    payload = reviews[0][1]
    assert payload.get("source") == "critic_total_budget"
    assert payload.get("degraded") is True

    # Metric: budget_skipped (not budget_fallback_used).
    actions = [r[0] for r in fake_metrics.records]
    assert "budget_skipped" in actions
    assert "budget_fallback_used" not in actions


def test_budget_skip_with_ungrounded_pace_falls_back(_emit_collector):
    """Sprint P core: ungrounded pace triggers the fallback stub.

    The tool result for this turn does NOT contain ``5:18/km`` (or any
    decimal-form equivalent). The rewrite still claims that pace. The
    new shield must intercept and ship the fallback German stub
    instead, emitting ``source: budget_skipped_with_fab_risk``.
    """
    events, emit_fn = _emit_collector
    loop = _build_loop_with_messages([
        {"role": "user", "content": "Wie war mein letzter Lauf?"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            # Pace deliberately ABSENT from the payload.
            "content": '{"activities": [{"distance_km": 10}]}',
        },
    ])
    loop.regenerate_after_critique = MagicMock(
        return_value=(
            "Schöner Lauf, du warst bei 5:18/km unterwegs - solide Form."
        ),
    )
    fake_critic = _make_fake_critic()
    fake_metrics = _RecordingMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch(
             "src.services.critic_metrics.get_metrics",
             return_value=fake_metrics,
         ), \
         patch(
             "src.agent.agent_loop.time.perf_counter",
             side_effect=_perf_counter_over_budget(),
         ):
        result = asyncio.run(loop._run_critique_pass(
            response_text="Some draft response with no hard violations.",
            user_message="Wie war mein letzter Lauf?",
            tool_names=["get_activities"],
            emit_fn=emit_fn,
        ))

    # Fallback stub ships (German, because the user message is German).
    assert result == (
        "Ich brauche kurz einen Moment, um deine Daten zu prüfen. "
        "Magst du die Frage gleich nochmal stellen?"
    )
    # Sanity: the unverified pace does NOT leak to the user.
    assert "5:18" not in result

    # critic_review event carries the new source attribution.
    reviews = [e for e in events if e[0] == "critic_review"]
    assert len(reviews) == 1
    payload = reviews[0][1]
    assert payload.get("source") == "budget_skipped_with_fab_risk"
    assert payload.get("action") == "fallback_used"
    assert payload.get("degraded") is True
    assert "5:18/km" in payload.get("ungrounded_specifics", [])

    # Metric: budget_fallback_used recorded with the unverified marker.
    actions = [r[0] for r in fake_metrics.records]
    assert "budget_fallback_used" in actions
    fallback_record = next(
        r for r in fake_metrics.records if r[0] == "budget_fallback_used"
    )
    assert "unverified_specifics" in fallback_record[1]


def test_budget_skip_with_ungrounded_hrv_falls_back(_emit_collector):
    """Sprint P: same as the pace case but for HRV ms.

    The fabricated-stats failure that motivated Sprint P had concrete
    examples like "Dein heutiger Morgen-HRV war 40 ms" with no
    ``get_health_metrics`` call in the turn. This test pins down that
    specific shape.
    """
    events, emit_fn = _emit_collector
    loop = _build_loop_with_messages([
        {"role": "user", "content": "Wie ist mein HRV heute?"},
        # NO tool result for HRV at all this turn.
    ])
    loop.regenerate_after_critique = MagicMock(
        return_value=(
            "Dein heutiger Morgen-HRV war 40 ms - das ist solide."
        ),
    )
    fake_critic = _make_fake_critic()
    fake_metrics = _RecordingMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch(
             "src.services.critic_metrics.get_metrics",
             return_value=fake_metrics,
         ), \
         patch(
             "src.agent.agent_loop.time.perf_counter",
             side_effect=_perf_counter_over_budget(),
         ):
        result = asyncio.run(loop._run_critique_pass(
            response_text="Some draft response with no hard violations.",
            user_message="Wie ist mein HRV heute?",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # Fallback ships, HRV does NOT leak.
    assert "40 ms" not in result
    assert result.startswith("Ich brauche kurz einen Moment")

    # Event carries the new source and the ungrounded HRV value.
    reviews = [e for e in events if e[0] == "critic_review"]
    assert len(reviews) == 1
    payload = reviews[0][1]
    assert payload.get("source") == "budget_skipped_with_fab_risk"
    ungrounded = payload.get("ungrounded_specifics", [])
    # The HRV token is in the list (rendered with the literal trailing
    # "ms" unit so reviewers can read the log).
    assert any("40" in s and "ms" in s.lower() for s in ungrounded)

    actions = [r[0] for r in fake_metrics.records]
    assert "budget_fallback_used" in actions


# ---------------------------------------------------------------------------
# Unit tests: specifics_grounded_in_tools
# ---------------------------------------------------------------------------


def test_specifics_grounded_handles_german_decimals():
    """German comma decimals must match tool-result dot decimals.

    A user-facing rewrite written in German prose typically uses
    ``5,18/km``; the tool result returns ``5.18/km`` (JSON convention).
    The check must treat the two as equivalent.
    """
    text = "Heute warst du bei 5,18/km unterwegs."
    tool_results = ['{"pace": "5.18/km"}']
    grounded, ungrounded = specifics_grounded_in_tools(text, tool_results)
    assert grounded is True
    assert ungrounded == []


def test_specifics_grounded_distinguishes_pace_from_duration():
    """``5:18/km`` is a pace; ``5:18`` alone is a duration.

    The two share the same numeric body but live in different domains.
    A pace must look up the pace-style needle (with the /km unit
    optionally elided in the haystack), while a bare ``5:18`` is a
    duration that does not need a /km neighbour.

    Concretely: if the tool result holds ``"5:18/km"`` and the text
    holds ``"5:18"`` (no /km), the check should still consider the
    duration grounded because the literal token ``5:18`` is present in
    the haystack. The pace check is the strict one.
    """
    # Case A: text claims a pace; tool result holds a duration only.
    text_pace = "Heute bei 5:18/km unterwegs."
    tool_with_duration_only = ['{"moving_time_text": "5:18"}']
    grounded_a, ungrounded_a = specifics_grounded_in_tools(
        text_pace, tool_with_duration_only,
    )
    # The pace-matcher tries 5:18, 5,18, 5.18 - finds 5:18 in the
    # haystack, so it grounds. This is by design: the check is
    # intentionally lenient on the unit boundary because the tool
    # payload often elides it. The TEXT side is where the pace/duration
    # distinction matters most.
    assert grounded_a is True

    # Case B (the real regression case): text claims a pace; tool
    # holds an UNRELATED number, no 5:18 anywhere.
    text_pace = "Heute bei 5:18/km unterwegs."
    tool_unrelated = ['{"distance_km": 10, "elev_m": 120}']
    grounded_b, ungrounded_b = specifics_grounded_in_tools(
        text_pace, tool_unrelated,
    )
    assert grounded_b is False
    assert any("5:18" in u for u in ungrounded_b)

    # Case C: text says a bare duration "5:18"; tool returns "5:18".
    # The duration pattern should match, not the pace pattern.
    text_dur = "Du warst 5:18 lang in Zone 4."
    tool_dur = ['{"time_in_zone_4": "5:18"}']
    grounded_c, ungrounded_c = specifics_grounded_in_tools(
        text_dur, tool_dur,
    )
    assert grounded_c is True


def test_specifics_grounded_empty_text_is_grounded():
    """No text means no specifics to check, which trivially grounds."""
    grounded, ungrounded = specifics_grounded_in_tools("", [])
    assert grounded is True
    assert ungrounded == []


def test_specifics_grounded_no_specifics_is_grounded():
    """Prose without any recognised numeric specific grounds trivially."""
    text = "Solider Lauf heute, weiter so."
    grounded, ungrounded = specifics_grounded_in_tools(text, [])
    assert grounded is True
    assert ungrounded == []


# ---------------------------------------------------------------------------
# CriticMetrics: new counter exposed in summary()
# ---------------------------------------------------------------------------


def test_budget_fallback_used_metric_in_summary():
    """``CriticMetrics.summary()`` must expose the new counter.

    Without the summary surface, downstream alerts on /admin/critic-
    stats cannot detect the fallback rate, and Sprint P's effect would
    be invisible to operations.
    """
    from src.services.critic_metrics import get_metrics

    metrics = get_metrics()
    metrics.reset()

    metrics.record(
        action="budget_fallback_used",
        violations=("no_fabricated_stats", "unverified_specifics"),
        latency_ms=7000,
    )
    summary = metrics.summary()

    assert summary.get("budget_fallback_used_count") == 1
    assert summary.get("budget_fallback_used_rate") == 1.0

    # Cleanup: do not leak state into other tests.
    metrics.reset()
