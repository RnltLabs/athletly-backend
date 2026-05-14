"""Send one chat message as a persona and stream the coach response.

Usage::

    python tools/persona_test/chat.py elena "war eben laufen, 10km easy bei 5:00 HR 142"
    python tools/persona_test/chat.py elena "..." --new   # start a fresh chat session
    python tools/persona_test/chat.py elena "..." --api-url http://localhost:8000

The session id is persisted at .persona_test_session/<slug>.json so each
subsequent call continues the same conversation. Use --new to reset.

Exit codes:
    0 = success
    2 = auth / config error
    3 = HTTP error (4xx / 5xx)
    4 = SSE stream cut mid-response
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.persona_test._persona import list_personas, load_persona
from tools.persona_test._supabase import (
    clear_cached_jwt,
    read_session_id,
    sign_in_persona,
    write_session_id,
)

logger = logging.getLogger(__name__)


DEFAULT_API_URL = os.environ.get("ATHLETLY_API_URL", "http://localhost:8000")
# Generous timeout: the agent can take 60+ seconds on complex turns.
_HTTP_TIMEOUT_S = 180.0


def send_message(
    *,
    slug: str,
    message: str,
    api_url: str,
    new_session: bool = False,
    show_thinking: bool = False,
) -> int:
    """Send one chat message as the persona; render the SSE stream.

    Returns the exit code (0 on success).
    """
    persona = load_persona(slug)
    auth = sign_in_persona(slug)

    session_id = None if new_session else read_session_id(slug)

    body = {
        "message": message,
        "session_id": session_id or "new",
        "context": "coach",
    }
    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    url = api_url.rstrip("/") + "/chat"
    _print_user_line(persona.name, message)

    try:
        rc = _stream(url, headers, body, slug, show_thinking)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            # Cached JWT might have expired between the cache check and the
            # actual request. Force a re-sign-in and retry once.
            clear_cached_jwt(slug)
            auth = sign_in_persona(slug)
            headers["Authorization"] = f"Bearer {auth.access_token}"
            try:
                rc = _stream(url, headers, body, slug, show_thinking)
            except Exception as exc2:  # noqa: BLE001
                sys.stderr.write(f"ERROR (retry): {exc2}\n")
                return 3
        else:
            sys.stderr.write(f"HTTP {exc.response.status_code}: {exc.response.text}\n")
            return 3
    except httpx.HTTPError as exc:
        sys.stderr.write(f"HTTP error: {exc}\n")
        return 3
    return rc


# ---------------------------------------------------------------------------
# SSE streaming and event rendering
# ---------------------------------------------------------------------------


def _stream(
    url: str,
    headers: dict,
    body: dict,
    slug: str,
    show_thinking: bool,
) -> int:
    """POST to /chat and render the SSE stream live."""
    saw_done = False
    saw_message = False
    with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
        with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise httpx.HTTPStatusError(
                    f"{response.status_code}",
                    request=response.request,
                    response=response,
                )
            for event_type, payload in _iter_sse(response):
                if event_type == "done":
                    saw_done = True
                    break
                _render_event(event_type, payload, slug=slug, show_thinking=show_thinking)
                if event_type == "message":
                    saw_message = True
    if not saw_done:
        sys.stderr.write("WARNING: stream ended without `done` event\n")
        return 4
    if not saw_message:
        sys.stderr.write("WARNING: no coach message in stream\n")
    return 0


def _iter_sse(response):
    """Yield (event_type, parsed_payload) tuples from a live SSE response.

    Robust to chunked frames split across multiple lines / blocks.
    """
    event_type = "message"
    data_lines: list[str] = []
    for raw_line in response.iter_lines():
        # httpx 0.27+ yields decoded strings already.
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
        if line == "":
            # End of one SSE frame.
            if data_lines:
                data = "\n".join(data_lines)
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = {"raw": data}
                yield event_type, payload
            event_type = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        # Ignore comment lines (:) and id: lines.


def _render_event(event_type: str, payload: dict, *, slug: str, show_thinking: bool) -> None:
    """Render a single SSE event to stdout in a human-readable form."""
    if event_type == "session_start":
        sid = payload.get("session_id")
        if sid:
            write_session_id(slug, sid)
            sys.stdout.write(f"[session] {sid}\n")
            sys.stdout.flush()
        return

    if event_type == "thinking":
        if show_thinking:
            text = payload.get("text", "").strip()
            if text:
                sys.stdout.write(f"[thinking] {text}\n")
                sys.stdout.flush()
        return

    if event_type in ("tool_call", "tool_hint"):
        name = payload.get("name", "?")
        args = payload.get("args") or {}
        args_short = _short_args(args)
        sys.stdout.write(f"[tool] {name}({args_short})\n")
        sys.stdout.flush()
        return

    if event_type == "tool_group_start":
        sys.stdout.write("[tool group start]\n")
        sys.stdout.flush()
        return

    if event_type == "tool_group_end":
        sys.stdout.write("[tool group end]\n")
        sys.stdout.flush()
        return

    if event_type == "tool_result":
        # Tool results can be large - show a short preview only.
        name = payload.get("name", "?")
        summary = _short_payload(payload)
        sys.stdout.write(f"[tool result] {name}: {summary}\n")
        sys.stdout.flush()
        return

    if event_type == "tool_error":
        sys.stdout.write(f"[tool error] {payload}\n")
        sys.stdout.flush()
        return

    if event_type == "message":
        text = payload.get("text", "")
        if text:
            sys.stdout.write(f"\n< Coach: {text}\n\n")
            sys.stdout.flush()
        return

    if event_type == "ui_component":
        ui_type = payload.get("type", "?")
        ui_id = payload.get("id", "?")
        sys.stdout.write(f"[ui] {ui_type} ({ui_id})\n")
        sys.stdout.flush()
        return

    if event_type == "action_request":
        sys.stdout.write(
            f"[action] {payload.get('action_type', '?')}: {payload.get('label', '')}\n",
        )
        sys.stdout.flush()
        return

    if event_type == "usage":
        sys.stdout.write(
            "[usage] input={input} output={output} model={model}\n".format(
                input=payload.get("input_tokens", 0),
                output=payload.get("output_tokens", 0),
                model=payload.get("model", "?"),
            ),
        )
        sys.stdout.flush()
        return

    if event_type == "error":
        sys.stderr.write(
            "[error] {code}: {message}\n".format(
                code=payload.get("code", "?"),
                message=payload.get("message", "?"),
            ),
        )
        return

    if event_type == "pending_action":
        sys.stdout.write(f"[pending_action] {payload}\n")
        sys.stdout.flush()
        return

    if event_type == "onboarding_complete":
        sys.stdout.write("[onboarding_complete]\n")
        sys.stdout.flush()
        return

    if event_type == "critic_review":
        sys.stdout.write(f"[critic_review] {_short_payload(payload)}\n")
        sys.stdout.flush()
        return

    # Unknown event - show raw for diagnosis.
    sys.stdout.write(f"[{event_type}] {_short_payload(payload)}\n")
    sys.stdout.flush()


def _short_args(args: dict, max_len: int = 120) -> str:
    """Render a tool's args dict as a short k=v string."""
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            s = json.dumps(value, ensure_ascii=False)
        else:
            s = str(value)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{key}={s}")
    out = ", ".join(parts)
    if len(out) > max_len:
        out = out[: max_len - 3] + "..."
    return out


def _short_payload(payload: dict, max_len: int = 200) -> str:
    """Render a payload dict as a short JSON string."""
    s = json.dumps(payload, ensure_ascii=False)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _print_user_line(name: str, message: str) -> None:
    sys.stdout.write(f"> {name}: {message}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one chat message as a persona test user.",
    )
    parser.add_argument(
        "persona",
        help=f"Persona slug. Available: {list_personas()}",
    )
    parser.add_argument(
        "message",
        help="The user message to send (in the persona's voice).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API base URL. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new chat session (do not resume the persisted session id).",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Also render thinking events from the stream.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return send_message(
        slug=args.persona,
        message=args.message,
        api_url=args.api_url,
        new_session=args.new,
        show_thinking=args.show_thinking,
    )


if __name__ == "__main__":
    sys.exit(main())
