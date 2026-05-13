"""Tests for spawn_subagent budget and round caps.

Covers Option A budget guard introduced after a single subagent invocation
chewed through 30-50k input tokens and tripped Anthropic Tier 1 rate
limits:

- SUBAGENT_INPUT_TOKEN_BUDGET halts the loop early when total prompt
  tokens accumulated across rounds exceeds the cap.
- SUBAGENT_MAX_ROUNDS caps the loop at 5 rounds even when each round
  spends few tokens.
- Natural exits (model returns final answer with no tool_calls) still
  return success and do NOT carry _budget_exhausted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.meta_tools import (
    SUBAGENT_INPUT_TOKEN_BUDGET,
    SUBAGENT_MAX_ROUNDS,
    register_meta_tools,
)
from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(tool_id: str, name: str, arguments: str = "{}") -> MagicMock:
    """Build a MagicMock mimicking LiteLLM's tool_call object."""
    tc = MagicMock()
    tc.id = tool_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_response(
    *,
    content: str | None = None,
    tool_calls: list | None = None,
    prompt_tokens: int = 0,
) -> MagicMock:
    """Build a MagicMock mimicking litellm.ModelResponse.

    - content + no tool_calls = natural final answer (loop exits).
    - tool_calls set = subagent will execute tools and continue.
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    return response


def _build_spawn_subagent():
    """Register meta tools on a fresh registry and return the spawn_subagent
    handler. Spawning a real user_model is unnecessary because the loop only
    touches user_model in the 'readonly' tools_scope branch."""
    registry = ToolRegistry()
    user_model = MagicMock()
    user_model.user_id = "test-user"
    register_meta_tools(registry, user_model)
    tool = registry._tools["spawn_subagent"]
    return tool.handler


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_rounds_is_five(self) -> None:
        assert SUBAGENT_MAX_ROUNDS == 5

    def test_token_budget_is_ten_thousand(self) -> None:
        assert SUBAGENT_INPUT_TOKEN_BUDGET == 10000


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


class TestBudgetGuard:
    @patch("src.agent.tools.meta_tools.chat_completion")
    def test_breaks_when_total_input_tokens_exceeds_budget(self, mock_chat) -> None:
        """After 3 rounds at 4000 tokens each (total = 12000), the loop
        should break with _budget_exhausted=True before issuing round 4."""
        tool_call = _make_tool_call("call_1", "web_search", '{"q": "x"}')
        # Each round returns tool_calls (so the loop keeps going) and
        # reports prompt_tokens=4000. The budget guard fires AFTER tokens
        # accumulate to >= 10000.
        mock_chat.return_value = _make_response(
            content="thinking...",
            tool_calls=[tool_call],
            prompt_tokens=4000,
        )

        spawn = _build_spawn_subagent()
        result = spawn(task="find something")

        # Round 1: total=4000 (<10000), continue
        # Round 2: total=8000 (<10000), continue
        # Round 3: total=12000 (>=10000), break with budget exhausted
        assert result.get("_budget_exhausted") is True
        assert result["rounds"] == 3
        assert result["_tokens_used"] == 12000
        assert "result" in result
        # Three chat_completion calls were made before budget tripped.
        assert mock_chat.call_count == 3

    @patch("src.agent.tools.meta_tools.chat_completion")
    def test_under_budget_does_not_break(self, mock_chat) -> None:
        """Two rounds at 4000 tokens (total=8000) stay under the 10k cap;
        the loop should keep going until rounds run out or a final answer
        arrives. We give round 3 a final answer to terminate cleanly."""
        tool_call = _make_tool_call("call_1", "web_search", '{"q": "x"}')

        responses = [
            _make_response(content=None, tool_calls=[tool_call], prompt_tokens=4000),
            _make_response(content=None, tool_calls=[tool_call], prompt_tokens=4000),
            # Final answer terminates the loop cleanly.
            _make_response(content="here is my answer", tool_calls=None, prompt_tokens=1000),
        ]
        mock_chat.side_effect = responses

        spawn = _build_spawn_subagent()
        result = spawn(task="find something")

        assert "_budget_exhausted" not in result
        assert result["result"] == "here is my answer"
        assert result["rounds"] == 3
        assert result["_tokens_used"] == 9000

    @patch("src.agent.tools.meta_tools.chat_completion")
    def test_natural_final_answer_returns_normally(self, mock_chat) -> None:
        """If round 2 returns no tool_calls, return the result normally and
        do NOT mark the response as budget-exhausted."""
        responses = [
            _make_response(
                content=None,
                tool_calls=[_make_tool_call("c1", "web_search", '{"q": "x"}')],
                prompt_tokens=3000,
            ),
            _make_response(
                content="final synthesized answer",
                tool_calls=None,
                prompt_tokens=3000,
            ),
        ]
        mock_chat.side_effect = responses

        spawn = _build_spawn_subagent()
        result = spawn(task="research X")

        assert "_budget_exhausted" not in result
        assert result["result"] == "final synthesized answer"
        assert result["rounds"] == 2
        assert result["_tokens_used"] == 6000

    @patch("src.agent.tools.meta_tools.chat_completion")
    def test_budget_exhausted_response_has_fallback_text(self, mock_chat) -> None:
        """When budget trips and the model returned no content this round,
        the result string should be the fallback halted-marker text."""
        tool_call = _make_tool_call("c1", "web_search", '{"q": "x"}')
        mock_chat.return_value = _make_response(
            content=None,  # empty content on the round that trips the budget
            tool_calls=[tool_call],
            prompt_tokens=6000,
        )

        spawn = _build_spawn_subagent()
        result = spawn(task="research X")

        assert result.get("_budget_exhausted") is True
        assert "halted" in result["result"]
        assert "budget" in result["result"]


# ---------------------------------------------------------------------------
# Max-rounds cap
# ---------------------------------------------------------------------------


class TestMaxRoundsCap:
    @patch("src.agent.tools.meta_tools.chat_completion")
    def test_stops_at_max_rounds_even_with_tiny_tokens(self, mock_chat) -> None:
        """Six rounds of tool calls with very small token counts should be
        capped at SUBAGENT_MAX_ROUNDS (= 5) by the loop bound, NOT by the
        budget guard."""
        tool_call = _make_tool_call("c1", "web_search", '{"q": "x"}')
        # All rounds: small token spend so budget never trips.
        mock_chat.return_value = _make_response(
            content=None,
            tool_calls=[tool_call],
            prompt_tokens=100,
        )

        spawn = _build_spawn_subagent()
        result = spawn(task="lots of tiny rounds")

        # The loop should exit after exactly SUBAGENT_MAX_ROUNDS iterations
        # via the max-rounds branch (error message), not via budget exhaustion.
        assert result["rounds"] == SUBAGENT_MAX_ROUNDS
        assert "_budget_exhausted" not in result
        assert "max rounds" in result.get("error", "")
        assert mock_chat.call_count == SUBAGENT_MAX_ROUNDS

    @patch("src.agent.tools.meta_tools.chat_completion")
    def test_caller_max_rounds_capped_at_global(self, mock_chat) -> None:
        """Even if the caller asks for max_rounds=20, the global cap wins."""
        tool_call = _make_tool_call("c1", "web_search", '{"q": "x"}')
        mock_chat.return_value = _make_response(
            content=None,
            tool_calls=[tool_call],
            prompt_tokens=50,
        )

        spawn = _build_spawn_subagent()
        result = spawn(task="t", max_rounds=20)

        assert result["rounds"] == SUBAGENT_MAX_ROUNDS
        assert mock_chat.call_count == SUBAGENT_MAX_ROUNDS
