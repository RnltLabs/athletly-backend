"""Trace inspection commands for athctl.

Reads JSONL trace files emitted by ``AgentLoop`` when running with
``trace_to_file=True`` (or env var ``ATHLETLY_TRACE_AGENT=1``). Each
line is one event of type ``llm_call``, ``tool_call``, ``tool_result``,
or ``response``.
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from pathlib import Path

import click
from rich.text import Text

from athctl.common import console, emit
from athctl.state import load_state

TRACES_DIR = Path("logs/traces")


def _resolve_session_id(session_id: str | None) -> str:
    if session_id:
        return session_id
    sid = load_state().get("last_session_id")
    if not sid:
        print(
            "No session id. Pass one explicitly or run `athctl chat` first.",
            file=sys.stderr,
        )
        sys.exit(2)
    return sid


def _trace_path(session_id: str) -> Path:
    return TRACES_DIR / f"{session_id}.jsonl"


def _render_event(evt: dict) -> Text:
    etype = evt.get("type", "?")
    ts = evt.get("ts", "")
    data = evt.get("data", {})
    if etype == "llm_call":
        msg = f"LLM call ({data.get('messages', '?')} msgs, {data.get('tools', '?')} tools)"
        style = "magenta"
    elif etype == "tool_call":
        msg = f"-> {data.get('tool')}({json.dumps(data.get('args', {}), default=str)[:120]})"
        style = "cyan"
    elif etype == "tool_result":
        if data.get("error"):
            msg = f"x  {data.get('tool')} : {str(data.get('error'))[:120]}"
            style = "red"
        else:
            preview = str(data.get("result", ""))[:120]
            msg = f"<- {data.get('tool')} : {preview}"
            style = "dim"
    elif etype == "response":
        msg = f"ok {str(data.get('text', ''))[:200]}"
        style = "green"
    else:
        msg = f"{etype}: {json.dumps(data, default=str)[:200]}"
        style = ""
    return Text(f"[{ts}] {msg}", style=style)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@click.group(help="Inspect agent trace files (logs/traces/*.jsonl).")
def trace() -> None:
    """Trace operations."""


@trace.command("path", help="Print the path to the trace file for a session.")
@click.option("--session", "session_id", default=None)
def trace_path(session_id: str | None) -> None:
    sid = _resolve_session_id(session_id)
    p = _trace_path(sid)
    emit(str(p), data={"session_id": sid, "path": str(p), "exists": p.exists()})


@trace.command("tail", help="Follow a trace file as new events arrive.")
@click.option("--session", "session_id", default=None)
@click.option("--interval", default=0.5, help="Polling interval in seconds.")
def trace_tail(session_id: str | None, interval: float) -> None:
    sid = _resolve_session_id(session_id)
    path = _trace_path(sid)
    if not path.exists():
        print(
            f"Trace file {path} does not exist yet. "
            "Make sure ATHLETLY_TRACE_AGENT=1 and `athctl chat` is running.",
            file=sys.stderr,
        )
        sys.exit(4)

    pos = 0
    console.print(f"[dim]Tailing {path} (Ctrl+C to stop)[/dim]")
    try:
        while True:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(pos)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    console.print(_render_event(evt))
                pos = fh.tell()
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped tailing.[/dim]")


@trace.command("replay", help="Pretty-print a complete trace file.")
@click.argument("session_id", required=False)
def trace_replay(session_id: str | None) -> None:
    sid = _resolve_session_id(session_id)
    path = _trace_path(sid)
    if not path.exists():
        print(f"Trace file {path} does not exist.", file=sys.stderr)
        sys.exit(4)
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    for evt in events:
        console.print(_render_event(evt))
    summary = {
        "total_events": len(events),
        "llm_calls": sum(1 for e in events if e.get("type") == "llm_call"),
        "tool_calls": sum(1 for e in events if e.get("type") == "tool_call"),
        "tool_results": sum(1 for e in events if e.get("type") == "tool_result"),
        "tool_errors": sum(
            1 for e in events
            if e.get("type") == "tool_result" and e.get("data", {}).get("error")
        ),
        "responses": sum(1 for e in events if e.get("type") == "response"),
    }
    emit(
        "\nSummary: " + ", ".join(f"{k}={v}" for k, v in summary.items()),
        data={"session_id": sid, "summary": summary, "events": events},
    )


@trace.command("diff", help="Compare two trace files semantically.")
@click.argument("session_a")
@click.argument("session_b")
def trace_diff(session_a: str, session_b: str) -> None:
    def _signature(sid: str) -> list[str]:
        path = _trace_path(sid)
        if not path.exists():
            return []
        out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")
            data = evt.get("data", {})
            if etype == "tool_call":
                out.append(f"tool_call:{data.get('tool')}({json.dumps(data.get('args', {}), sort_keys=True, default=str)})")
            elif etype == "tool_result":
                out.append(
                    f"tool_result:{data.get('tool')}:"
                    f"{'error' if data.get('error') else 'ok'}"
                )
            elif etype == "response":
                out.append(f"response:{str(data.get('text', ''))[:80]}")
        return out

    a = _signature(session_a)
    b = _signature(session_b)
    diff = list(
        difflib.unified_diff(a, b, fromfile=session_a, tofile=session_b, n=2, lineterm="")
    )
    if not diff:
        emit("No semantic differences.")
        return
    for line in diff:
        if line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        else:
            console.print(line)
