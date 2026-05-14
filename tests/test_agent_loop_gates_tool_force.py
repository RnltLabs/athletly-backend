"""Sprint F integration tests for the Tool-Forcing Regenerate path.

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
