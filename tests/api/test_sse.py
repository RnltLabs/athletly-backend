"""Tests for src/api/sse.py — SSEEmitter typed event helpers.

Each test verifies that the correct SSE event name and data fields are produced
by the corresponding SSEEmitter static method.
"""

import json

import pytest

from src.api.sse import SSEEmitter


class TestSSEEmitter:
    def test_thinking_event(self) -> None:
        """SSEEmitter.thinking produces event='thinking' with a text payload."""
        evt = SSEEmitter.thinking("Analyzing your training load...")

        assert evt.event == "thinking"
        data = json.loads(evt.data)
        assert data["text"] == "Analyzing your training load..."

    def test_tool_hint_event(self) -> None:
        """SSEEmitter.tool_hint produces event='tool_hint' with name and args."""
        args = {"user_id": "abc-123", "limit": 5}
        evt = SSEEmitter.tool_hint(name="get_activities", args=args)

        assert evt.event == "tool_hint"
        data = json.loads(evt.data)
        assert data["name"] == "get_activities"
        assert data["args"] == args

    def test_tool_hint_empty_args(self) -> None:
        """SSEEmitter.tool_hint handles empty args dict."""
        evt = SSEEmitter.tool_hint(name="health_check", args={})

        data = json.loads(evt.data)
        assert data["args"] == {}

    def test_message_event(self) -> None:
        """SSEEmitter.message produces event='message' with the reply text."""
        reply = "Great effort today! Let's increase intensity next week."
        evt = SSEEmitter.message(reply)

        assert evt.event == "message"
        data = json.loads(evt.data)
        assert data["text"] == reply

    def test_error_event_with_defaults(self) -> None:
        """SSEEmitter.error includes message and default code='internal_error'."""
        evt = SSEEmitter.error("Something went wrong")

        assert evt.event == "error"
        data = json.loads(evt.data)
        assert data["message"] == "Something went wrong"
        assert data["code"] == "internal_error"

    def test_error_event_with_custom_code(self) -> None:
        """SSEEmitter.error uses the provided custom error code."""
        evt = SSEEmitter.error("Already in progress", code="concurrent_request")

        data = json.loads(evt.data)
        assert data["code"] == "concurrent_request"
        assert data["message"] == "Already in progress"

    def test_done_event(self) -> None:
        """SSEEmitter.done produces event='done' with an empty data payload."""
        evt = SSEEmitter.done()

        assert evt.event == "done"
        data = json.loads(evt.data)
        assert data == {}

    def test_usage_event(self) -> None:
        """SSEEmitter.usage produces event='usage' with token counts and model."""
        evt = SSEEmitter.usage(input_tokens=120, output_tokens=45, model="gemini/gemini-2.5-flash")

        assert evt.event == "usage"
        data = json.loads(evt.data)
        assert data["input_tokens"] == 120
        assert data["output_tokens"] == 45
        assert data["model"] == "gemini/gemini-2.5-flash"

    def test_format_event_returns_raw_sse_string(self) -> None:
        """SSEEmitter.format_event builds the correct raw SSE wire format."""
        raw = SSEEmitter.format_event("test_event", {"key": "value"})

        assert raw.startswith("event: test_event\n")
        assert "data:" in raw
        assert raw.endswith("\n\n")
        # The data line should contain valid JSON.
        data_line = [line for line in raw.splitlines() if line.startswith("data:")][0]
        json_part = data_line[len("data:"):]
        parsed = json.loads(json_part)
        assert parsed == {"key": "value"}
