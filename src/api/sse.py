"""SSE event emitter helpers for the Athletly streaming API.

All public methods return a ServerSentEvent ready to be yielded
from an async generator passed to EventSourceResponse.
"""

import json
from sse_starlette.sse import ServerSentEvent


class SSEEmitter:
    """Formats and emits typed Server-Sent Events."""

    @staticmethod
    def format_event(event_name: str, data: dict) -> str:
        """Serialise an SSE frame as a raw string (for testing / logging)."""
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event_name}\ndata: {payload}\n\n"

    # -- Typed helpers -------------------------------------------------------

    @staticmethod
    def thinking(text: str) -> ServerSentEvent:
        """Agent is reasoning — surface a thinking indicator in the UI."""
        return ServerSentEvent(
            event="thinking",
            data=json.dumps({"text": text}, ensure_ascii=False),
        )

    @staticmethod
    def tool_hint(name: str, args: dict) -> ServerSentEvent:
        """Agent is about to call a tool — lets the UI show a spinner label."""
        return ServerSentEvent(
            event="tool_hint",
            data=json.dumps({"name": name, "args": args}, ensure_ascii=False),
        )

    @staticmethod
    def message(text: str) -> ServerSentEvent:
        """Final coach reply text."""
        return ServerSentEvent(
            event="message",
            data=json.dumps({"text": text}, ensure_ascii=False),
        )

    @staticmethod
    def usage(input_tokens: int, output_tokens: int, model: str) -> ServerSentEvent:
        """Token usage metadata — emitted once at the end of each turn."""
        return ServerSentEvent(
            event="usage",
            data=json.dumps(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": model,
                },
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def error(message: str, code: str = "internal_error") -> ServerSentEvent:
        """Structured error event — client should surface this to the user."""
        return ServerSentEvent(
            event="error",
            data=json.dumps({"message": message, "code": code}, ensure_ascii=False),
        )

    @staticmethod
    def done() -> ServerSentEvent:
        """Sentinel event — signals the stream is finished."""
        return ServerSentEvent(
            event="done",
            data=json.dumps({}, ensure_ascii=False),
        )
