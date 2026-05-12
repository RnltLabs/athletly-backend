"""Chat command for athctl.

Drives the production ``AgentLoop`` directly (no HTTP/JWT round-trip)
so the test harness can iterate on prompts, tools, and beliefs without
re-deploying the API. Streams progress events with rich formatting and
optionally drops into a REPL for multi-turn sessions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import click
from rich.text import Text

from athctl.common import EXIT_BACKEND_DOWN, console, emit, require_active_user
from athctl.state import load_state, set_last_session_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress event styling
# ---------------------------------------------------------------------------

_EVENT_PREFIX = {
    "thinking": "[~]",
    "tool_call": "[->]",
    "tool_hint": "[->]",
    "tool_result": "[<-]",
    "tool_error": "[x]",
    "responding": "[ok]",
}

_EVENT_STYLE = {
    "thinking": "dim",
    "tool_call": "cyan",
    "tool_hint": "cyan",
    "tool_result": "dim",
    "tool_error": "red bold",
    "responding": "green bold",
}


def _emit_progress(event_type: str, detail: str) -> None:
    """Render one progress event to the console or as a JSON line."""
    if os.environ.get("ATHCTL_JSON") == "1":
        print(json.dumps({"event": event_type, "detail": detail}, ensure_ascii=False))
        return
    prefix = _EVENT_PREFIX.get(event_type, "[?]")
    style = _EVENT_STYLE.get(event_type, "")
    text = Text(f"{prefix} {detail[:300]}", style=style)
    console.print(text)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(help="Send a chat message to the agent as the active test user.")
@click.argument("message", required=False)
@click.option("--repl", is_flag=True, help="Enter interactive REPL after first turn.")
@click.option(
    "--new-session",
    is_flag=True,
    help="Start a fresh session instead of resuming the last one.",
)
@click.option("--session-id", default=None, help="Resume a specific session by id.")
@click.option(
    "--trace/--no-trace",
    default=True,
    help="Write JSONL trace to logs/traces/{session_id}.jsonl (default on).",
)
def chat(
    message: str | None,
    repl: bool,
    new_session: bool,
    session_id: str | None,
    trace: bool,
) -> None:
    """Send one chat message; with --repl, keep prompting until Ctrl+D / Ctrl+C."""
    uid = require_active_user()

    if not message and not repl:
        print(
            "Provide a message or use --repl for interactive mode.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        from src.agent.agent_loop import AgentLoop
        from src.db.user_model_db import UserModelDB
    except Exception as exc:
        print(f"Could not import agent stack: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    try:
        user_model = UserModelDB.load_or_create(uid)
    except Exception as exc:
        print(f"Could not load user model for {uid}: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    if trace:
        os.environ["ATHLETLY_TRACE_AGENT"] = "1"

    loop = AgentLoop(
        user_model=user_model,
        on_progress=_emit_progress,
        context="coach",
        trace_to_file=trace,
    )

    resume_id = session_id
    if not resume_id and not new_session:
        resume_id = load_state().get("last_session_id") or None

    if resume_id:
        try:
            active_session = loop.start_session(resume_session_id=resume_id)
            emit(f"Resumed session {active_session}")
        except Exception as exc:
            logger.warning("Resume failed, starting new session: %s", exc)
            active_session = loop.start_session()
            emit(f"Started new session {active_session}")
    else:
        active_session = loop.start_session()
        emit(f"Started session {active_session}")

    set_last_session_id(active_session)

    def _send(text: str) -> None:
        result = loop.process_message(text)
        emit(
            f"\n{result.response_text}\n",
            data={
                "session_id": active_session,
                "response_text": result.response_text,
                "tool_calls": result.tool_calls_made,
                "duration_ms": result.total_duration_ms,
            },
        )

    if message:
        _send(message)

    if not repl:
        return

    console.print("\n[dim]Entering REPL. Empty line or Ctrl+D / Ctrl+C exits.[/dim]\n")
    try:
        while True:
            try:
                line = console.input("[bold]you[/bold] > ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                break
            _send(line)
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting REPL.[/dim]")
