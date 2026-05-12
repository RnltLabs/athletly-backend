"""State-file management for athctl.

Stores the currently-active test user (and a little extra metadata) in
``~/.athctl/state.json`` so subsequent commands like ``athctl chat`` know
which user_id to operate on without re-prompting.

The state file is intentionally small and human-editable:

    {
      "active_user_id": "uuid-or-null",
      "active_user_email": "claude-test+TS@athletly.dev",
      "last_session_id": null,
      "created_at": "2026-05-12T08:15:00+00:00"
    }

All functions in this module are pure with respect to the file system in
the sense that they never raise on missing dir/file - they create or
return defaults idempotently.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path.home() / ".athctl"
STATE_FILE = STATE_DIR / "state.json"


def _empty_state() -> dict[str, Any]:
    """Return a fresh, empty state dict (immutable factory)."""
    return {
        "active_user_id": None,
        "active_user_email": None,
        "last_session_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _ensure_dir() -> None:
    """Create ``~/.athctl/`` if it does not yet exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, Any]:
    """Return the current state dict, creating defaults if missing.

    Never raises on missing file. On JSON parse errors the file is left
    untouched and a fresh default state is returned (the caller can
    decide whether to overwrite by calling ``save_state``).
    """
    _ensure_dir()
    if not STATE_FILE.exists():
        return _empty_state()
    try:
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                return _empty_state()
            return data
    except (json.JSONDecodeError, OSError):
        return _empty_state()


def save_state(state: dict[str, Any]) -> None:
    """Persist *state* to disk atomically (write-then-rename).

    Callers should treat the returned state from ``load_state`` as
    immutable: build a new dict (``{**old, "key": new_value}``) and pass
    it here rather than mutating the original.
    """
    _ensure_dir()
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(STATE_FILE)


def get_active_user_id() -> str | None:
    """Return the currently-active test user id, or ``None`` if unset."""
    return load_state().get("active_user_id")


def get_last_session_id() -> str | None:
    """Return the most recent chat session id, or ``None`` if unset."""
    return load_state().get("last_session_id")


def set_last_session_id(session_id: str | None) -> dict[str, Any]:
    """Persist *session_id* as the last chat session and return new state.

    Builds a new dict rather than mutating the loaded one.
    """
    current = load_state()
    new_state = {**current, "last_session_id": session_id}
    save_state(new_state)
    return new_state


def set_active_user(uid: str, email: str | None = None) -> dict[str, Any]:
    """Mark *uid* as the active test user and persist the new state.

    Returns the new state dict for inspection. Never mutates the
    previously-loaded state object - constructs a new one.
    """
    current = load_state()
    new_state = {
        **current,
        "active_user_id": uid,
        "active_user_email": email if email is not None else current.get("active_user_email"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(new_state)
    return new_state


def clear_active_user() -> dict[str, Any]:
    """Reset the active user fields, keeping the file in place."""
    current = load_state()
    new_state = {
        **current,
        "active_user_id": None,
        "active_user_email": None,
        "last_session_id": None,
    }
    save_state(new_state)
    return new_state
