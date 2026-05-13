"""Tests for Anthropic prompt-cache activation.

Covers the three root causes of the silent cache failure documented in
the fix:

1. STATIC_SYSTEM_PROMPT must clear Anthropic's 4096-token minimum
   cacheable size (~16400 chars conservatively).
2. ``_add_message_cache_breakpoints`` must attach cache_control to the
   last message block without mutating the input.
3. ``chat_completion(runtime_context=...)`` must add a SECOND system
   block (in addition to the static prompt) with its own
   cache_control breakpoint when targeting an Anthropic model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.llm import (
    _MIN_CACHE_CHARS,
    _add_message_cache_breakpoints,
    chat_completion,
)
from src.agent.system_prompt import STATIC_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 1. Static prompt size
# ---------------------------------------------------------------------------


class TestStaticPromptSize:
    def test_static_prompt_clears_anthropic_cache_threshold(self) -> None:
        """STATIC_SYSTEM_PROMPT must be long enough to be cacheable on
        claude-haiku-4-5 (>=4096 tokens, mapped to >=16400 chars)."""
        assert len(STATIC_SYSTEM_PROMPT) > 16400, (
            f"STATIC_SYSTEM_PROMPT is only {len(STATIC_SYSTEM_PROMPT)} "
            f"chars (~{len(STATIC_SYSTEM_PROMPT) // 4} tokens). Anthropic "
            f"silently ignores cache_control for prompts below the 4096 "
            f"token minimum; we need >=16400 chars to be safe."
        )

    def test_min_cache_chars_constant_matches_threshold(self) -> None:
        """The module-level threshold tracks the 4096-token rule."""
        assert _MIN_CACHE_CHARS >= 16384


# ---------------------------------------------------------------------------
# 2. _add_message_cache_breakpoints
# ---------------------------------------------------------------------------


class TestAddMessageCacheBreakpoints:
    def test_returns_input_unchanged_for_non_anthropic(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _add_message_cache_breakpoints(msgs, is_anthropic=False)
        assert result is msgs

    def test_returns_input_unchanged_for_short_history(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = _add_message_cache_breakpoints(msgs, is_anthropic=True)
        assert result is msgs

    def test_attaches_cache_control_to_last_message_block(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "last"},
        ]
        result = _add_message_cache_breakpoints(msgs, is_anthropic=True)

        # Last message content was rewrapped into a single-block list.
        last = result[-1]
        assert isinstance(last["content"], list)
        assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert last["content"][-1]["text"] == "last"

    def test_does_not_mutate_input_list(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "last"},
        ]
        _add_message_cache_breakpoints(msgs, is_anthropic=True)

        # Original messages still have their string content (no shared mutation).
        assert msgs[-1]["content"] == "last"
        assert isinstance(msgs[-1]["content"], str)

    def test_handles_tool_role_message(self) -> None:
        """role=tool messages need cache_control on their content too."""
        msgs = [
            {"role": "user", "content": "go"},
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": '{"result": "ok"}',
            },
        ]
        result = _add_message_cache_breakpoints(msgs, is_anthropic=True)
        last = result[-1]
        assert last["tool_call_id"] == "tc_1"  # preserved
        assert isinstance(last["content"], list)
        assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_preserves_existing_content_blocks(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "block-a"},
                    {"type": "text", "text": "block-b"},
                ],
            },
        ]
        result = _add_message_cache_breakpoints(msgs, is_anthropic=True)
        last_blocks = result[-1]["content"]
        assert len(last_blocks) == 2
        assert last_blocks[0] == {"type": "text", "text": "block-a"}
        assert last_blocks[1]["text"] == "block-b"
        assert last_blocks[1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# 3. chat_completion(runtime_context=...) request shape
# ---------------------------------------------------------------------------


def _mock_response() -> MagicMock:
    """Build a minimal litellm.ModelResponse-shaped mock."""
    resp = MagicMock()
    resp.usage = MagicMock(
        prompt_tokens=0,
        completion_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock(content="ok", tool_calls=None)
    return resp


class TestRuntimeContextSecondSystemBlock:
    @patch("src.agent.llm.litellm.completion")
    def test_anthropic_runtime_context_creates_second_cached_system_block(
        self, mock_completion
    ) -> None:
        """When called with runtime_context against an Anthropic model and
        a long-enough static system prompt, the request must contain TWO
        system blocks, each carrying its own ephemeral cache_control."""
        mock_completion.return_value = _mock_response()

        long_system_prompt = "S" * 20000  # > _MIN_CACHE_CHARS
        long_runtime_ctx = "R" * 1200  # > _MIN_RUNTIME_CTX_CACHE_CHARS

        chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt=long_system_prompt,
            runtime_context=long_runtime_ctx,
            model="anthropic/claude-haiku-4-5-20251001",
        )

        assert mock_completion.call_count == 1
        sent = mock_completion.call_args.kwargs["messages"]

        system_blocks = [m for m in sent if m.get("role") == "system"]
        assert len(system_blocks) == 2, (
            f"expected 2 system blocks, got {len(system_blocks)}: {system_blocks}"
        )

        # Both system blocks must be content-blocks lists carrying cache_control.
        for block in system_blocks:
            assert isinstance(block["content"], list)
            assert block["content"][0]["cache_control"] == {"type": "ephemeral"}

        # First block is the static system prompt; second is runtime context.
        assert system_blocks[0]["content"][0]["text"] == long_system_prompt
        assert system_blocks[1]["content"][0]["text"] == long_runtime_ctx

    @patch("src.agent.llm.litellm.completion")
    def test_non_anthropic_runtime_context_merged_into_first_user_message(
        self, mock_completion
    ) -> None:
        """For Gemini/OpenAI, runtime_context is merged into the first
        user message as a fallback (no second system block)."""
        mock_completion.return_value = _mock_response()

        chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="S" * 20000,
            runtime_context="R" * 1200,
            model="gemini/gemini-2.5-flash",
        )

        sent = mock_completion.call_args.kwargs["messages"]
        system_blocks = [m for m in sent if m.get("role") == "system"]
        # Only the static system prompt; runtime_context not in its own block.
        assert len(system_blocks) == 1

        user_blocks = [m for m in sent if m.get("role") == "user"]
        # Runtime context was prepended to the first user message.
        assert "R" * 1200 in user_blocks[0]["content"]
        assert "[CONTEXT]" in user_blocks[0]["content"]
