"""Sprint F + M integration tests for gate / critic chain behavior.

Stubs ``chat_completion`` so the forced tool loop runs end-to-end in
process without an LLM provider. Verifies that:

1. A tool-required gate failure routes to the forced path and produces
   a passing rewrite when the model calls the required tools.
2. A forced path that still fails the gate records
   ``tool_forced_regenerate_failed`` AND emits ``critic_review`` with
   ``degraded=true``.
3. A forced path that raises returns the original response unchanged
   and records the failure metric.
4. A text-only gate failure does NOT enter the forced path.
5. Sprint M: the softened force-instruction preserves the substantive
   coaching content of a helpful knee-pain draft instead of deflecting
   to a "ich brauche Garmin-Login" reply.
6. Sprint M: the critic chain enforces ``critic_total_budget_s`` by
   skipping the second-pass LLM call once the chain has exceeded the
   budget. The skip path emits ``critic_review`` with ``degraded=true``
   and increments the ``budget_skipped`` counter in critic_metrics.

The tests mirror the pattern used in ``tests/test_critic.py``: build
an ``AsyncAgentLoop`` via ``__new__`` so the heavy ``__init__`` (tool
registry, supabase client, etc.) does not run, then attach the minimal
attributes ``_run_gates_pass`` and ``_handle_tool_required_gate_failure``
need.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent_loop import AsyncAgentLoop
from src.services.gates_metrics import (
    get_gates_metrics,
    reset_gates_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_text_response(text: str):
    """Build a minimal LLM response with text-only content."""
    msg = SimpleNamespace(
        content=text,
        tool_calls=None,
        reasoning_content=None,
        thinking=None,
    )
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _llm_tool_response(tool_name: str, args: str = "{}", tool_id: str = "call_1"):
    """Build a minimal LLM response that requests one client-tool call."""
    fn = SimpleNamespace(name=tool_name, arguments=args)
    tc = SimpleNamespace(id=tool_id, function=fn)
    msg = SimpleNamespace(
        content="",
        tool_calls=[tc],
        reasoning_content=None,
        thinking=None,
    )
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


@pytest.fixture(autouse=True)
def _reset_metrics():
    reset_gates_metrics()
    yield
    reset_gates_metrics()


@pytest.fixture
def _emit_collector():
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    return events, emit


def _build_loop():
    """Build an AsyncAgentLoop with the minimum attributes our path needs."""
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []
    loop._gates_regenerated_this_turn = False
    loop._active_alert_ids = ()
    loop._user_id = "test-user"
    loop._session_id = "test-session"
    loop.user_model = SimpleNamespace(user_id="test-user")
    loop.startup_context = ""
    loop.context = "default"
    # Empty deque of recent tool sets for Gate 3 windowing.
    from collections import deque

    loop._recent_tools_window = deque(maxlen=3)
    # Tool registry stub: empty tool list keeps the forced loop simple,
    # since we stub chat_completion to drive the tool_calls directly.
    loop.tools = SimpleNamespace(
        get_openai_tools=lambda defer_non_core=True: [],
    )
    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tool_forcing_regenerate_success_for_stats_grounding(_emit_collector):
    """Stats_grounding failure: forced retry calls get_activities, then passes.

    Uses a user message that triggers ONLY stats_grounding (not
    temporal_freshness), so the forced retry with a single read-tool call
    is sufficient to satisfy the gate on the second pass.
    """
    events, emit_fn = _emit_collector
    loop = _build_loop()
    # No temporal trigger word ("heute", "gerade", "eben") so only the
    # stats_grounding gate fires on the draft.
    user_msg = "wie waren meine letzten paar einheiten so im schnitt?"
    loop._messages = [{"role": "user", "content": user_msg}]

    # First chat_completion: model calls get_activities (in forced loop round 1).
    # Second chat_completion: model produces a clean final answer that
    # contains NO stats (sole way to satisfy stats_grounding without a
    # numeric token; the gate's pass condition is satisfied by the tool
    # being present in tools_called_this_turn so this is incidental).
    responses = [
        _llm_tool_response("get_activities", args="{}", tool_id="call_get_acts"),
        _llm_text_response(
            "Du warst zuletzt konstant unterwegs - solide Form, weiter so."
        ),
    ]

    def _stub_chat_completion(*a, **kw):
        return responses.pop(0)

    # Stub execute_with_budget so the tool "executes" without touching real services.
    def _stub_execute(_registry, tool_name, _args):
        return {"activities": [{"distance_km": 10, "pace": "5:00/km"}]}

    with patch("src.agent.agent_loop.chat_completion", side_effect=_stub_chat_completion), \
         patch(
             "src.agent.agent_loop.execute_with_budget",
             side_effect=_stub_execute,
         ), \
         patch(
             "src.agent.agent_loop.run_pre_hooks",
             return_value=SimpleNamespace(action="allow", reason=None, replace_args={}),
         ), \
         patch("src.agent.agent_loop.run_post_hooks", return_value=None), \
         patch("src.agent.agent_loop.build_runtime_context", return_value=""):

        # Draft response that triggers stats_grounding (mentions 10km, no read tool).
        draft = "Schoener Lauf, 10 km in 50:00 bei 5:00/km, Puls 152 sah solide aus."

        result = asyncio.run(
            loop._run_gates_pass(
                response_text=draft,
                user_message=user_msg,
                tool_names=[],  # no read tool before the regenerate
                emit_fn=emit_fn,
            )
        )

    # The new candidate response should be the model's clean rewrite.
    assert "solide" in result or "konstant" in result

    # Metric: tool_forced_regenerate_success recorded on stats_grounding.
    summary = get_gates_metrics().summary()
    assert summary["per_gate"]["stats_grounding"]["tool_forced_regenerate_success"] >= 1
    assert summary["per_gate"]["stats_grounding"]["regenerate_success"] >= 1


def test_tool_forcing_regenerate_failure_emits_degraded_critic_review(_emit_collector):
    """When the forced retry STILL fails the gate, emit degraded critic_review."""
    events, emit_fn = _emit_collector
    loop = _build_loop()
    user_msg = "wie waren meine letzten paar einheiten so im schnitt?"
    loop._messages = [{"role": "user", "content": user_msg}]

    # Forced loop: model produces a text response that STILL fabricates stats
    # (no tool calls during the forced rounds), so stats_grounding fires again.
    responses = [
        _llm_text_response(
            "Auch jetzt schaetze ich mal 10km in 50:00. War bestimmt gut."
        ),
    ]

    def _stub_chat_completion(*a, **kw):
        return responses.pop(0) if responses else _llm_text_response("")

    with patch("src.agent.agent_loop.chat_completion", side_effect=_stub_chat_completion), \
         patch(
             "src.agent.agent_loop.execute_with_budget",
             side_effect=lambda *a, **kw: {},
         ), \
         patch(
             "src.agent.agent_loop.run_pre_hooks",
             return_value=SimpleNamespace(action="allow", reason=None, replace_args={}),
         ), \
         patch("src.agent.agent_loop.run_post_hooks", return_value=None), \
         patch("src.agent.agent_loop.build_runtime_context", return_value=""):

        draft = "Schoener Lauf, 10 km in 50:00 bei 5:00/km."
        result = asyncio.run(
            loop._run_gates_pass(
                response_text=draft,
                user_message=user_msg,
                tool_names=[],
                emit_fn=emit_fn,
            )
        )

    # Forced retry produced text but no tools fired; gate still fails. The
    # rewrite ships and a critic_review with degraded=true is emitted.
    review_events = [e for e in events if e[0] == "critic_review"]
    assert len(review_events) == 1
    payload = review_events[0][1]
    assert payload.get("degraded") is True
    assert payload.get("source") == "response_gates_tool_forced"

    summary = get_gates_metrics().summary()
    assert summary["per_gate"]["stats_grounding"]["tool_forced_regenerate_failed"] >= 1
    assert summary["per_gate"]["stats_grounding"]["regenerate_failed"] >= 1


def test_tool_forcing_regenerate_chat_completion_raises_keeps_original(_emit_collector):
    """If chat_completion blows up during the forced loop, ship the original."""
    events, emit_fn = _emit_collector
    loop = _build_loop()
    user_msg = "wie waren meine letzten paar einheiten so im schnitt?"
    loop._messages = [{"role": "user", "content": user_msg}]

    def _boom(*a, **kw):
        raise RuntimeError("provider 503")

    with patch("src.agent.agent_loop.chat_completion", side_effect=_boom), \
         patch(
             "src.agent.agent_loop.execute_with_budget",
             side_effect=lambda *a, **kw: {},
         ), \
         patch(
             "src.agent.agent_loop.run_pre_hooks",
             return_value=SimpleNamespace(action="allow", reason=None, replace_args={}),
         ), \
         patch("src.agent.agent_loop.run_post_hooks", return_value=None), \
         patch("src.agent.agent_loop.build_runtime_context", return_value=""):

        draft = "Schoener Lauf, 10 km in 50:00 bei 5:00/km."
        result = asyncio.run(
            loop._run_gates_pass(
                response_text=draft,
                user_message=user_msg,
                tool_names=[],
                emit_fn=emit_fn,
            )
        )

    # Forced retry could not produce text - original ships.
    assert result == draft

    summary = get_gates_metrics().summary()
    # regenerate_failed AND tool_forced_regenerate_failed recorded.
    assert summary["per_gate"]["stats_grounding"]["regenerate_failed"] >= 1
    assert summary["per_gate"]["stats_grounding"]["tool_forced_regenerate_failed"] >= 1


def test_elena_turn6_knee_pain_softer_force_keeps_substantive_advice(_emit_collector):
    """Elena Iter 2 Turn 6 regression: a helpful knee-pain draft must NOT
    be replaced with a "ich brauche Garmin-Login" deflection after the
    tool-force regenerate.

    The original symptom: draft said "Knieschmerzen klingen nach ITB-
    Reizung, mach erstmal Pause" (helpful). Gate injury_persistence
    fired because no annotate_activity / append_to_journal was called.
    The forced retry deflected with "Ich brauche zuerst Zugang zu
    deinen Garmin-Daten". Worse than the original.

    Fix verified here: (a) the composed force-instruction explicitly
    forbids the deflection patterns and tells the model to preserve
    its earlier coaching content; (b) when the model follows that
    instruction (stubbed: calls annotate_activity, then emits a
    substantive coaching answer), the final text contains coaching
    keywords and contains NONE of the forbidden deflection phrases.
    """
    events, emit_fn = _emit_collector
    loop = _build_loop()
    user_msg = "Mein Knie tut weh nach dem Lauf, was soll ich tun?"
    loop._messages = [{"role": "user", "content": user_msg}]

    # Forced loop responses:
    # 1. Model calls annotate_activity (satisfies persistence requirement
    #    for injury_persistence gate).
    # 2. Model emits final text with substantive coaching content.
    coaching_text = (
        "Klingt nach einer leichten ITB-Reizung am Knie. Reduzier die "
        "Intensität jetzt, mach zwei bis drei Tage Pause oder nur "
        "ganz leichter Spaziergang, dann teste vorsichtig wieder. "
        "Wenn der Schmerz bleibt, sprechen wir nochmal."
    )
    responses = [
        _llm_tool_response(
            "annotate_activity",
            args='{"activity_id":"latest","note":"Knieschmerz nach Lauf"}',
            tool_id="call_annot",
        ),
        _llm_text_response(coaching_text),
    ]

    def _stub_chat_completion(*a, **kw):
        return responses.pop(0)

    def _stub_execute(_registry, tool_name, _args):
        return {"ok": True, "activity_id": "latest"}

    with patch("src.agent.agent_loop.chat_completion", side_effect=_stub_chat_completion), \
         patch(
             "src.agent.agent_loop.execute_with_budget",
             side_effect=_stub_execute,
         ), \
         patch(
             "src.agent.agent_loop.run_pre_hooks",
             return_value=SimpleNamespace(action="allow", reason=None, replace_args={}),
         ), \
         patch("src.agent.agent_loop.run_post_hooks", return_value=None), \
         patch("src.agent.agent_loop.build_runtime_context", return_value=""):

        helpful_draft = (
            "Klingt nach ITB-Reizung, mach erstmal Pause und schau dann "
            "wie es sich entwickelt."
        )
        result = asyncio.run(
            loop._run_gates_pass(
                response_text=helpful_draft,
                user_message=user_msg,
                tool_names=[],
                emit_fn=emit_fn,
            )
        )

    # Final text must keep substantive coaching content.
    lower = result.lower()
    coaching_hits = [
        kw for kw in ("itb", "knie", "pause", "reduzier", "leichter")
        if kw in lower
    ]
    assert coaching_hits, (
        f"Final text lost all coaching keywords. Got: {result!r}"
    )

    # Final text must NOT be a deflection.
    forbidden_phrases = (
        "logge dich ein",
        "log dich ein",
        "ich brauche zuerst",
        "ich habe keine daten",
        "please log in",
    )
    for phrase in forbidden_phrases:
        assert phrase not in lower, (
            f"Final text deflected with forbidden phrase {phrase!r}: {result!r}"
        )


def test_compose_force_instruction_forbids_deflection_for_injury():
    """Unit-level: the injury_persistence force instruction must tell the
    model to preserve coaching content and forbid the Garmin-login
    deflection. Pure check against the prompt string so a regression
    that softens the wording too far cannot land unnoticed.
    """
    from src.agent.response_gates import (
        GateResult,
        _compose_tool_force_instruction,
    )

    fail = GateResult(
        passed=False,
        gate_id="injury_persistence",
        reason="test",
        required_action=None,
    )
    text = _compose_tool_force_instruction((fail,))
    lower = text.lower()
    # Anti-deflection guidance.
    assert "ich brauche zuerst" in lower
    # Preserve coaching content guidance.
    assert "preserve" in lower or "keep the" in lower
    # Still names the required tool so the model knows what to call.
    assert "annotate_activity" in text
    assert "append_to_journal" in text


def test_text_only_gate_failure_uses_legacy_path(_emit_collector):
    """A language_mirror failure must NOT enter the forced tool loop.

    Uses a German user message that contains no temporal triggers
    ("heute", "gerade", "eben"), no sport nouns, and no injury terms,
    so only language_mirror fires.
    """
    events, emit_fn = _emit_collector
    loop = _build_loop()
    user_msg = (
        "Hallo, ich wuerde gerne wissen, wie das mit der Planung fuer "
        "die kommende Saison so aussieht und was du mir empfiehlst."
    )
    loop._messages = [{"role": "user", "content": user_msg}]

    # The text-only path calls regenerate_after_gates (a method on the loop).
    # We stub it to return a clean German rewrite that is short and does
    # NOT contain any stats-triggering numeric token.
    loop.regenerate_after_gates = MagicMock(
        return_value="Klar, lass uns das gemeinsam ueberlegen.",
    )

    # If chat_completion is touched, the test fails (text-only path uses the
    # stubbed regenerate_after_gates and the second-pass run_gates).
    with patch(
        "src.agent.agent_loop.chat_completion",
        side_effect=AssertionError("forced loop must NOT run for text-only gates"),
    ):
        # English response to a German user message triggers language_mirror only.
        draft = (
            "Sure, you can plan the season carefully and listen to your body, "
            "keep an open mind and adjust along the way."
        )
        result = asyncio.run(
            loop._run_gates_pass(
                response_text=draft,
                user_message=user_msg,
                tool_names=[],
                emit_fn=emit_fn,
            )
        )

    # regenerate_after_gates was called exactly once.
    loop.regenerate_after_gates.assert_called_once()

    # No tool_forced_* metrics recorded.
    summary = get_gates_metrics().summary()
    assert summary["totals"]["tool_forced_regenerate_success"] == 0
    assert summary["totals"]["tool_forced_regenerate_failed"] == 0


# ---------------------------------------------------------------------------
# Sprint M: critic chain total-budget cap
# ---------------------------------------------------------------------------


def test_critic_chain_skips_second_pass_when_budget_exceeded(_emit_collector):
    """Production telemetry showed the critic chain averaging 8.6s, well
    over the 4.0s per-call timeout, because first-pass + regenerate +
    second-pass compound serially. Sprint M caps the chain at
    ``critic_total_budget_s`` (6.0s).

    Scenario: stub the chain so first-pass + regenerate together have
    consumed 6.5s of wall-clock by the time the budget check runs. The
    second-pass critic.review (which would otherwise add another ~2s)
    MUST be skipped, the rewrite ships, and a critic_review SSE event
    with ``degraded=true`` plus ``source=critic_total_budget`` is
    emitted. The ``budget_skipped`` counter in critic_metrics is
    incremented.
    """
    from src.agent.agent_loop import AsyncAgentLoop
    from src.agent.critic import CriticResult, Violation
    from src.services.critic_metrics import get_metrics as get_critic_metrics

    events, emit_fn = _emit_collector
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []
    loop.regenerate_after_critique = MagicMock(
        return_value="Rewritten text without fabricated stats.",
    )

    fake_critic = MagicMock()
    # Only the FIRST pass should be invoked. The second pass is the
    # one we expect to be skipped, so its return value is set as a
    # sentinel that would fail the test if it were ever consumed.
    second_pass_sentinel = CriticResult.accept(latency_ms=2000)
    fake_critic.review.side_effect = [
        CriticResult(
            action="regenerate",
            violations=(Violation(rule="no_fabricated_stats", reason="x"),),
            latency_ms=900,
        ),
        second_pass_sentinel,
    ]

    # Drive perf_counter so the chain reports 6.5s elapsed at the
    # budget check, well over the 6.0s budget. First read = chain_start.
    perf_counter_values = iter([0.0, 6.5, 6.5, 6.5, 6.5])

    def _fake_perf_counter() -> float:
        try:
            return next(perf_counter_values)
        except StopIteration:
            return 6.5

    fake_metrics_records: list[tuple] = []

    class _RecordingMetrics:
        def record(self, action, violations, latency_ms):
            fake_metrics_records.append((action, tuple(violations), latency_ms))

    fake_metrics = _RecordingMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch(
             "src.services.critic_metrics.get_metrics",
             return_value=fake_metrics,
         ), \
         patch(
             "src.agent.agent_loop.time.perf_counter",
             side_effect=_fake_perf_counter,
         ):
        result = asyncio.run(loop._run_critique_pass(
            response_text="Some draft response without hard violations.",
            user_message="Hello coach",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # Rewrite ships.
    assert result == "Rewritten text without fabricated stats."

    # Second-pass critic.review must NOT have been invoked.
    assert fake_critic.review.call_count == 1, (
        "Second-pass critic.review was called even though the budget was "
        f"already exceeded; call_count={fake_critic.review.call_count}"
    )

    # critic_review SSE event with degraded=True and source attribution.
    review_events = [e for e in events if e[0] == "critic_review"]
    assert len(review_events) == 1
    payload = review_events[0][1]
    assert payload.get("degraded") is True
    assert payload.get("source") == "critic_total_budget"

    # budget_skipped counter incremented.
    actions = [c[0] for c in fake_metrics_records]
    assert "budget_skipped" in actions

    # And via the real metrics summary endpoint, the counter is exposed.
    # Use a fresh records list on the live singleton to confirm the
    # serialisation surface includes the new field.
    live_metrics = get_critic_metrics()
    live_metrics.reset()
    live_metrics.record(
        action="budget_skipped", violations=(), latency_ms=7000,
    )
    live_summary = live_metrics.summary()
    assert live_summary.get("budget_skipped_count") == 1
    assert live_summary.get("budget_skipped_rate") == 1.0
    live_metrics.reset()
