"""Tests for the auth-failure fallback path in _compress_with_llm.

When the configured compression provider rejects the call with an auth-
or BadRequest-shaped error (e.g. an invalid Gemini API key), the helper
must retry exactly once against the primary `MODEL` from src.agent.llm
before giving up.

These tests patch `src.agent.llm.chat_completion` to drive the side
effects deterministically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from litellm.exceptions import BadRequestError

from src.agent.tools import truncation
from src.agent.tools.truncation import _compress_with_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(content: str) -> MagicMock:
    """Build a fake chat_completion response with the given text content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _bad_request(message: str = "API_KEY_INVALID: invalid Gemini API key") -> BadRequestError:
    """Create a BadRequestError with positional args matching litellm's signature."""
    return BadRequestError(
        message=message,
        model="gemini/gemini-2.5-flash",
        llm_provider="gemini",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompressionFallback:
    """Verify the primary-model retry kicks in only on auth-style errors."""

    @patch("src.agent.tools.truncation._get_compression_model", return_value="gemini/gemini-2.5-flash")
    @patch("src.agent.llm.MODEL", "anthropic/claude-haiku-4-5-20251001")
    @patch("src.agent.llm.chat_completion")
    def test_fallback_used_when_primary_auth_fails(
        self, mock_chat, _mock_model
    ) -> None:
        """Bad key on Gemini falls back to the configured primary MODEL."""
        mock_chat.side_effect = [
            _bad_request(),
            _ok_response('{"summary": "fallback ok"}'),
        ]

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)

        assert result == '{"summary": "fallback ok"}'
        assert mock_chat.call_count == 2

        # First call goes to the compression model, second to the fallback.
        first_kwargs = mock_chat.call_args_list[0].kwargs
        second_kwargs = mock_chat.call_args_list[1].kwargs
        assert first_kwargs["model"] == "gemini/gemini-2.5-flash"
        assert second_kwargs["model"] == "anthropic/claude-haiku-4-5-20251001"

        # Prompt + budget must be identical across the two attempts.
        assert first_kwargs["messages"] == second_kwargs["messages"]
        assert first_kwargs["temperature"] == second_kwargs["temperature"] == 0.0

    @patch("src.agent.tools.truncation._get_compression_model", return_value="gemini/gemini-2.5-flash")
    @patch("src.agent.llm.MODEL", "anthropic/claude-haiku-4-5-20251001")
    @patch("src.agent.llm.chat_completion")
    def test_fallback_also_fails_returns_none(
        self, mock_chat, _mock_model
    ) -> None:
        """If both attempts fail with auth errors, the caller sees None."""
        mock_chat.side_effect = [
            _bad_request("API_KEY_INVALID on primary"),
            _bad_request("API_KEY_INVALID on fallback"),
        ]

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)

        assert result is None
        assert mock_chat.call_count == 2

    @patch("src.agent.tools.truncation._get_compression_model", return_value="gemini/gemini-2.5-flash")
    @patch("src.agent.llm.MODEL", "anthropic/claude-haiku-4-5-20251001")
    @patch("src.agent.llm.chat_completion")
    def test_no_fallback_when_primary_succeeds(
        self, mock_chat, _mock_model
    ) -> None:
        """A happy first call must not trigger any retry."""
        mock_chat.return_value = _ok_response('{"summary": "primary ok"}')

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)

        assert result == '{"summary": "primary ok"}'
        assert mock_chat.call_count == 1
        called_model = mock_chat.call_args.kwargs["model"]
        assert called_model == "gemini/gemini-2.5-flash"

    @patch("src.agent.tools.truncation._get_compression_model", return_value="gemini/gemini-2.5-flash")
    @patch("src.agent.llm.MODEL", "anthropic/claude-haiku-4-5-20251001")
    @patch("src.agent.llm.chat_completion")
    def test_non_auth_error_does_not_trigger_fallback(
        self, mock_chat, _mock_model
    ) -> None:
        """Generic exceptions (e.g. network) should NOT consume the fallback."""
        mock_chat.side_effect = RuntimeError("network blip")

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)

        assert result is None
        # Only the primary attempt was made.
        assert mock_chat.call_count == 1

    @patch("src.agent.tools.truncation._get_compression_model", return_value="gemini/gemini-2.5-flash")
    @patch("src.agent.llm.MODEL", "gemini/gemini-2.5-flash")
    @patch("src.agent.llm.chat_completion")
    def test_no_fallback_when_primary_equals_compression(
        self, mock_chat, _mock_model
    ) -> None:
        """If the fallback model is identical, we skip the retry to avoid a loop."""
        mock_chat.side_effect = _bad_request()

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)

        assert result is None
        assert mock_chat.call_count == 1


class TestCompressionModelSetting:
    """Surface-level check that the new setting is wired to env."""

    def test_default_compression_model(self) -> None:
        from src.config import Settings

        assert Settings().compression_model == "gemini/gemini-2.5-flash"

    def test_module_reads_settings(self) -> None:
        assert truncation._get_compression_model() == "gemini/gemini-2.5-flash"
