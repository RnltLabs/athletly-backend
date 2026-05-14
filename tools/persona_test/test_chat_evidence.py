"""Tests for the ``--print-evidence`` flag in ``tools/persona_test/chat.py``.

The EVIDENCE TRACE block is consumed verbatim by other persona-test agents,
so the shape must stay stable. These tests assert:

1. session_id is captured from the ``session_start`` event.
2. tools_called is a Python-list-like rendering of ordered tool_hint events
   with compact args. ``tool_hint`` is the actual SSE event emitted by
   ``src/api/sse.py``; the agent-internal ``tool_call`` trace event never
   reaches the wire.
3. response_text_verbatim matches the joined message event texts exactly
   (no truncation, multi-line preserved).
4. The usage line is present with the documented keys.
5. The critic_review event is rendered when present, ``null`` otherwise.
6. The block contains no markdown formatting (literal text only).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.persona_test.chat import (  # noqa: E402
    EvidenceTrace,
    _format_tool_args_compact,
    _iter_sse,
    _render_event,
    _render_evidence_block,
)


def _build_trace_from_events(events: list[tuple[str, dict]]) -> EvidenceTrace:
    """Replay a fake SSE event sequence into an EvidenceTrace.

    Pure dataclass population: no httpx, no I/O. Mirrors what ``_render_event``
    does for the trace, so we can test the renderer in isolation. ``tool_hint``
    is the wire event emitted by ``src/api/sse.py`` and is the only event we
    count toward ``tools_called``.
    """
    trace = EvidenceTrace()
    for event_type, payload in events:
        if event_type == "session_start":
            trace.session_id = payload.get("session_id")
        elif event_type == "tool_hint":
            trace.tools_called.append(
                (payload.get("name", "?"), payload.get("args") or {})
            )
        elif event_type == "message":
            text = payload.get("text", "")
            if text:
                trace.message_chunks.append(text)
        elif event_type == "usage":
            trace.usage = dict(payload)
        elif event_type == "critic_review":
            trace.critic_review_event = dict(payload)
    return trace


def test_evidence_block_basic_shape() -> None:
    """A representative trace renders with header, all required keys, footer."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "4e701f74-6790-4ac0-8aea-fd3f87875d96"}),
            ("tool_hint", {"name": "get_activities", "args": {"limit": 10}}),
            ("tool_hint", {"name": "sync_garmin_data", "args": {"mode": "auto"}}),
            (
                "message",
                {"text": "Sehe deinen 10er von heute frueh, 10.02 km in 50:14."},
            ),
            (
                "usage",
                {
                    "input_tokens": 850,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 180,
                    "model": "anthropic/claude-haiku-4-5",
                },
            ),
        ]
    )
    out = _render_evidence_block(trace)

    # Frame markers exactly as documented.
    assert out.startswith("=== EVIDENCE TRACE ===\n")
    assert out.rstrip("\n").endswith("=== END EVIDENCE ===")

    # Required lines, in the documented order.
    lines = out.splitlines()
    assert lines[0] == "=== EVIDENCE TRACE ==="
    assert lines[1] == "session_id: 4e701f74-6790-4ac0-8aea-fd3f87875d96"
    assert lines[2].startswith("tools_called: [")
    assert "get_activities(limit=10)" in lines[2]
    assert "sync_garmin_data(mode=auto)" in lines[2]
    # response_text_verbatim header line.
    assert lines[3] == "response_text_verbatim:"
    # Verbatim text.
    assert "Sehe deinen 10er von heute frueh, 10.02 km in 50:14." in out
    # Tokens line shape.
    assert (
        "tokens: input=850 cache_read=0 output=180 model=anthropic/claude-haiku-4-5"
        in out
    )
    # Critic review absent -> literal null.
    assert "critic_review_event: null" in out


def test_evidence_block_tools_called_preserves_order() -> None:
    """tools_called is the ordered tool_hint list, not a set."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("tool_hint", {"name": "A", "args": {}}),
            ("tool_hint", {"name": "B", "args": {}}),
            ("tool_hint", {"name": "A", "args": {}}),
            ("message", {"text": "ok"}),
            ("usage", {"input_tokens": 1, "output_tokens": 1, "model": "m"}),
        ]
    )
    out = _render_evidence_block(trace)
    tools_line = next(line for line in out.splitlines() if line.startswith("tools_called:"))
    # A, B, A in that order: A's index < B's index < second A's index.
    a_first = tools_line.find("A(")
    b_idx = tools_line.find("B(")
    a_second = tools_line.find("A(", b_idx)
    assert 0 <= a_first < b_idx < a_second


def test_evidence_block_verbatim_preserves_multiline() -> None:
    """response_text_verbatim must NOT truncate or collapse newlines."""
    multiline = "Zeile eins.\nZeile zwei mit Umlaut: ueber.\nZeile drei."
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("message", {"text": multiline}),
            ("usage", {"input_tokens": 0, "output_tokens": 0, "model": "m"}),
        ]
    )
    out = _render_evidence_block(trace)
    assert multiline in out
    # No ellipsis / truncation marker.
    assert "..." not in out.split("response_text_verbatim:\n", 1)[1].split("\ntokens:")[0]


def test_evidence_block_multiple_message_events_join() -> None:
    """Multiple message events join in order with no separator inserted."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("message", {"text": "Erstes Stueck. "}),
            ("message", {"text": "Zweites Stueck."}),
            ("usage", {"input_tokens": 0, "output_tokens": 0, "model": "m"}),
        ]
    )
    out = _render_evidence_block(trace)
    # The exact concatenation appears in the block.
    assert "Erstes Stueck. Zweites Stueck." in out


def test_evidence_block_critic_review_rendered_as_json() -> None:
    """When a critic_review event fires, the full payload is inlined as JSON."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("message", {"text": "Antwort."}),
            ("usage", {"input_tokens": 1, "output_tokens": 1, "model": "m"}),
            (
                "critic_review",
                {
                    "action": "regenerate",
                    "violations": ["no_em_dash"],
                    "latency_ms": 412,
                },
            ),
        ]
    )
    out = _render_evidence_block(trace)
    critic_line = next(
        line for line in out.splitlines() if line.startswith("critic_review_event:")
    )
    assert critic_line.startswith("critic_review_event: {")
    assert '"action":"regenerate"' in critic_line
    assert '"no_em_dash"' in critic_line


def test_evidence_block_no_markdown_formatting() -> None:
    """The block is plain literal text. No markdown headers / fences / bullets."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("tool_hint", {"name": "foo", "args": {}}),
            ("message", {"text": "Antwort."}),
            ("usage", {"input_tokens": 1, "output_tokens": 1, "model": "m"}),
        ]
    )
    out = _render_evidence_block(trace)
    # No markdown headers / fences / list markers anywhere in the block.
    for line in out.splitlines():
        assert not line.lstrip().startswith("#"), f"markdown header in: {line!r}"
        assert not line.lstrip().startswith("```"), f"code fence in: {line!r}"
        assert not line.lstrip().startswith("- "), f"bullet in: {line!r}"
        assert not line.lstrip().startswith("* "), f"bullet in: {line!r}"


def test_evidence_block_no_em_or_en_dash_in_skeleton() -> None:
    """The skeleton must never contain em-dash or en-dash chars (AI tell)."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("message", {"text": "Antwort."}),
            ("usage", {"input_tokens": 0, "output_tokens": 0, "model": "m"}),
        ]
    )
    out = _render_evidence_block(trace)
    assert "—" not in out  # em-dash
    assert "–" not in out  # en-dash


def test_format_tool_args_compact_truncates_at_60_chars() -> None:
    """The per-tool compact args string caps at 60 chars with ``...`` suffix."""
    long_args = {"query": "a" * 200, "limit": 10}
    rendered = _format_tool_args_compact(long_args)
    assert len(rendered) <= 60
    assert rendered.endswith("...")


def test_format_tool_args_compact_empty() -> None:
    """Empty args render as an empty string (not ``{}`` or ``None``)."""
    assert _format_tool_args_compact({}) == ""
    assert _format_tool_args_compact(None) == ""  # type: ignore[arg-type]


def test_evidence_block_handles_missing_usage() -> None:
    """When the SSE stream omits a usage event, the line still renders with defaults."""
    trace = _build_trace_from_events(
        [
            ("session_start", {"session_id": "sid"}),
            ("message", {"text": "Antwort."}),
        ]
    )
    out = _render_evidence_block(trace)
    assert "tokens: input=0 cache_read=0 output=0 model=?" in out


def test_evidence_block_handles_missing_session_id() -> None:
    """If session_start never fired, the line marks the absence rather than crashing."""
    trace = _build_trace_from_events(
        [
            ("message", {"text": "Antwort."}),
            ("usage", {"input_tokens": 0, "output_tokens": 0, "model": "m"}),
        ]
    )
    out = _render_evidence_block(trace)
    assert "session_id: <missing>" in out


# ---------------------------------------------------------------------------
# Integration: feed a fake SSE byte stream through the real event renderer
# and the real parser, populating a Trace via _render_event.
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Just enough of an httpx streaming response to drive ``_iter_sse``."""

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def iter_lines(self):
        # httpx 0.27+ semantics: yield decoded lines with no trailing newline.
        for line in self._raw.split("\n"):
            yield line


def test_iter_sse_plus_render_event_populates_trace(monkeypatch) -> None:
    """Drive the real parser + dispatcher with a representative SSE stream.

    The test stubs out ``write_session_id`` so it does not touch disk, then
    feeds a hand-crafted SSE byte buffer through the actual parser and the
    actual event renderer. Asserts the resulting EvidenceTrace contains all
    the structured fields downstream agents rely on.
    """
    written = []

    def _fake_write(slug: str, session_id: str) -> None:
        written.append((slug, session_id))

    monkeypatch.setattr("tools.persona_test.chat.write_session_id", _fake_write)

    sse_text = (
        "event: session_start\n"
        'data: {"session_id": "sid-abc"}\n'
        "\n"
        "event: tool_hint\n"
        'data: {"name": "get_activities", "args": {"limit": 5}}\n'
        "\n"
        "event: tool_hint\n"
        'data: {"name": "sync_garmin_data", "args": {"mode": "auto"}}\n'
        "\n"
        "event: message\n"
        'data: {"text": "Hallo Elena, sehe deinen Lauf."}\n'
        "\n"
        "event: usage\n"
        'data: {"input_tokens": 1200, "cache_read_input_tokens": 500, '
        '"output_tokens": 90, "model": "anthropic/claude-haiku-4-5"}\n'
        "\n"
        "event: critic_review\n"
        'data: {"action": "accept", "violations": []}\n'
        "\n"
        "event: done\n"
        "data: {}\n"
        "\n"
    )

    trace = EvidenceTrace()
    response = _FakeStreamResponse(sse_text)
    for event_type, payload in _iter_sse(response):
        if event_type == "done":
            break
        _render_event(
            event_type,
            payload,
            slug="_test_evidence",
            show_thinking=False,
            trace=trace,
        )

    # Session id captured from session_start.
    assert trace.session_id == "sid-abc"
    # tools_called preserves order; structure is list (per dataclass default).
    assert isinstance(trace.tools_called, list)
    assert [name for name, _ in trace.tools_called] == [
        "get_activities",
        "sync_garmin_data",
    ]
    # message text captured.
    assert trace.message_chunks == ["Hallo Elena, sehe deinen Lauf."]
    # usage captured with all keys.
    assert trace.usage is not None
    assert trace.usage["input_tokens"] == 1200
    assert trace.usage["cache_read_input_tokens"] == 500
    assert trace.usage["output_tokens"] == 90
    # critic_review captured.
    assert trace.critic_review_event == {"action": "accept", "violations": []}

    # And the session_id was passed to the persistence helper (which we stubbed).
    assert written == [("_test_evidence", "sid-abc")]

    # Finally render the block and make sure the verbatim text comes out.
    out = _render_evidence_block(trace)
    assert "Hallo Elena, sehe deinen Lauf." in out
    assert "session_id: sid-abc" in out
    # tools_called rendering shows both names in order.
    tools_line = next(
        line for line in out.splitlines() if line.startswith("tools_called:")
    )
    assert tools_line.index("get_activities") < tools_line.index("sync_garmin_data")


def test_real_sse_tool_hint_payload_populates_tools_called(monkeypatch) -> None:
    """Regression: real-shape ``tool_hint`` SSE frames must populate tools_called.

    Production was shipping ``tools_called: []`` because the trace populator
    listened on ``tool_call`` (an internal-only agent trace event) instead of
    ``tool_hint`` (the actual wire event from ``src/api/sse.py``). This test
    feeds a stream built from real ``SSEEmitter.tool_hint`` frames (with their
    ``display_label`` + ``group_id`` extras) and asserts the trace populates.
    """
    from src.api.sse import SSEEmitter

    monkeypatch.setattr(
        "tools.persona_test.chat.write_session_id", lambda slug, sid: None
    )

    # Build a real SSE byte stream using the production emitter.
    frames = [
        SSEEmitter.format_event("session_start", {"session_id": "sid-real"}),
        SSEEmitter.tool_hint(
            name="sync_garmin_data",
            args={"mode": "auto"},
            display_label="Garmin syncen",
            group_id="grp-1",
        ),
        SSEEmitter.tool_hint(
            name="get_activities",
            args={"limit": 10},
            display_label="Aktivitaeten laden",
            group_id="grp-1",
        ),
        SSEEmitter.format_event("message", {"text": "Lauf gesehen."}),
        SSEEmitter.format_event(
            "usage",
            {"input_tokens": 100, "output_tokens": 20, "model": "m"},
        ),
        SSEEmitter.format_event("done", {}),
    ]

    # ``ServerSentEvent`` and the raw string from ``format_event`` differ in
    # type. Coerce both to the wire-string form ``_iter_sse`` expects.
    sse_text_parts: list[str] = []
    for frame in frames:
        if isinstance(frame, str):
            sse_text_parts.append(frame)
        else:
            # ServerSentEvent: rebuild the wire frame from its public attrs.
            sse_text_parts.append(
                f"event: {frame.event}\ndata: {frame.data}\n\n"
            )
    sse_text = "".join(sse_text_parts)

    trace = EvidenceTrace()
    response = _FakeStreamResponse(sse_text)
    for event_type, payload in _iter_sse(response):
        if event_type == "done":
            break
        _render_event(
            event_type,
            payload,
            slug="_test_real_tool_hint",
            show_thinking=False,
            trace=trace,
        )

    # The whole point of the regression test: tools_called must not be empty.
    assert trace.tools_called, "tool_hint frames did not populate tools_called"
    assert [name for name, _ in trace.tools_called] == [
        "sync_garmin_data",
        "get_activities",
    ]

    out = _render_evidence_block(trace)
    tools_line = next(
        line for line in out.splitlines() if line.startswith("tools_called:")
    )
    assert tools_line != "tools_called: []"
    assert "sync_garmin_data" in tools_line
    assert "get_activities" in tools_line
