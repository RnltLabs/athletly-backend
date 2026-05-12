"""Shared helpers for athctl commands.

- Supabase client access (service-role, bypasses RLS)
- Rich-based pretty printing (respects ``ATHCTL_QUIET`` / ``ATHCTL_JSON``)
- Exit codes used uniformly across all commands
- ``require_active_user()`` guard for commands that need a test user
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from rich.console import Console

from athctl.state import get_active_user_id

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_AUTH = 2
EXIT_NETWORK = 3
EXIT_STATE = 4
EXIT_BACKEND_DOWN = 5


# ---------------------------------------------------------------------------
# Console / output helpers
# ---------------------------------------------------------------------------


def _quiet() -> bool:
    return os.environ.get("ATHCTL_QUIET") == "1"


def _json_mode() -> bool:
    return os.environ.get("ATHCTL_JSON") == "1"


# Single shared console - silenced when --quiet is active. We rebuild it
# lazily so the env var set by the top-level CLI options takes effect.
def _make_console() -> Console:
    return Console(quiet=_quiet(), highlight=False)


console: Console = _make_console()


def refresh_console() -> Console:
    """Rebuild the global ``console`` after env flags are applied.

    Click sets ``ATHCTL_QUIET`` / ``ATHCTL_JSON`` in the top-level group
    callback. By the time individual commands run we need a console that
    respects those flags.
    """
    global console  # noqa: PLW0603 -- intentional singleton refresh
    console = _make_console()
    return console


def as_json(data: Any) -> str:
    """Render *data* as a pretty JSON string (2-space indent, stable order)."""
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def emit(human: str, *, data: Any | None = None) -> None:
    """Print *human* to the console, or the JSON payload if ``--json``.

    When ``ATHCTL_JSON=1`` is set, ``data`` is dumped to stdout as JSON
    (if provided); otherwise the human-readable string is printed via
    the Rich console.
    """
    if _json_mode():
        if data is not None:
            print(as_json(data))
        return
    console.print(human)


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------


def get_supabase():  # noqa: ANN201 - return type is supabase.Client
    """Return the cached service-role Supabase client.

    Wraps ``src.db.client.get_supabase`` so callers don't need to know
    about the backend module layout.
    """
    from src.db.client import get_supabase as _backend_get_supabase

    return _backend_get_supabase()


# ---------------------------------------------------------------------------
# State guards
# ---------------------------------------------------------------------------


def require_active_user() -> str:
    """Return the active test user id or exit with a helpful message.

    Exits with ``EXIT_STATE`` if no active user is set in
    ``~/.athctl/state.json``. The exit message is plain stderr so it
    works in both human and JSON modes.
    """
    uid = get_active_user_id()
    if not uid:
        print(
            "No active test user. Run `athctl init` first to bootstrap one.",
            file=sys.stderr,
        )
        sys.exit(EXIT_STATE)
    return uid
