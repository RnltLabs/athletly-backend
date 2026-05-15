"""Session lifecycle tools: 14-day rolling window over ``training_sessions``.

Replaces the 24-week locked-plan ``save_plan`` model. The coach plans the
next 14 days at a time, modulates as the athlete's life shifts, and
extends the window when there are fewer than three days of runway left.

Tools registered:
    - propose_sessions: persist a list of upcoming sessions for a window.
      Replaces existing planned rows in the same date range so the coach
      can re-propose freely. Emits a ``plan_preview`` UI component.
    - modify_session: edit date / modality / notes / payload. Substantive
      changes flip status to ``modulated`` and keep ``planned_payload``
      as the audit trail.
    - complete_session: mark a session done (or skipped) with an optional
      free-form athlete note.
    - get_session_window: read planned + recently-completed sessions in a
      bounded window. The coach calls this first to know what's already
      planned before proposing new work.
    - extend_window: append more sessions when the window is running low,
      typically called when fewer than three days remain.

The ``training_sessions`` table is created in
``supabase/migrations/20260515000000_training_sessions.sql``. The legacy
``plans`` table is left in place but no longer written by these tools.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date as _date_cls
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cap the number of sessions embedded in the inline ``plan_preview``
# component so a 14-day window does not bloat the SSE payload. The full
# set stays in the database; the UI component just truncates.
_MAX_INLINE_PREVIEW_SESSIONS = 30

# Substantive change fields. Editing any of these flips status to
# ``modulated`` so the athlete can see the prescription was reshaped.
_SUBSTANTIVE_FIELDS = frozenset({"date", "modality", "payload", "notes"})

# Valid status values mirror the CHECK constraint in the migration.
_VALID_STATUSES = frozenset({"planned", "completed", "skipped", "modulated"})

# Maximum window the coach can request in one call. Guards against
# accidentally pulling years of history into the prompt.
_MAX_WINDOW_DAYS_AHEAD = 60
_MAX_WINDOW_DAYS_BACK = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_iso() -> str:
    return _date_cls.today().isoformat()


def _parse_iso_date(value: str) -> _date_cls:
    """Parse an ISO YYYY-MM-DD string. Raises ValueError on bad input."""
    return _date_cls.fromisoformat(value)


def _normalize_modality(modality) -> str | None:
    """Trim and validate a modality string.

    Returns the cleaned modality or ``None`` if invalid (empty / not a
    string). Coach picks any free string: this only rejects empty/junk
    so we never write rows that violate the table CHECK constraint.
    """
    if not isinstance(modality, str):
        return None
    cleaned = modality.strip()
    if not cleaned:
        return None
    return cleaned


def _serialize_session_row(row: dict) -> dict:
    """Project a Supabase row into the public tool-result shape.

    Returns a fresh dict so callers cannot mutate the original.
    """
    return {
        "id": row.get("id"),
        "date": row.get("date"),
        "modality": row.get("modality"),
        "status": row.get("status"),
        "payload": row.get("payload") or {},
        "notes": row.get("notes") or "",
        "athlete_note": row.get("athlete_note") or "",
        "actual_activity_id": row.get("actual_activity_id"),
        "planned_payload": row.get("planned_payload") or {},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _build_plan_preview(window_sessions: list[dict]) -> dict:
    """Construct the ``plan_preview`` UI component for the inline card.

    The component type and prop shape stay STABLE so the React Native
    frontend does not need any code change when the schema migrates. The
    ``sessions`` array uses the historical key names (``day``, ``date``,
    ``sport``, ``name``, ``description``, ``duration_minutes``,
    ``intensity``) so existing renderUIComponent registries keep working.
    """
    inline_sessions: list[dict] = []
    truncated = len(window_sessions) > _MAX_INLINE_PREVIEW_SESSIONS
    for raw in window_sessions[:_MAX_INLINE_PREVIEW_SESSIONS]:
        payload = raw.get("payload") or {}
        date_str = raw.get("date")
        day_name = ""
        if isinstance(date_str, str):
            try:
                day_name = _parse_iso_date(date_str).strftime("%A").lower()
            except ValueError:
                day_name = ""
        slim = {
            "day": day_name,
            "date": date_str,
            "sport": raw.get("modality"),
            "name": payload.get("name") if isinstance(payload, dict) else None,
            "description": raw.get("notes") or (
                payload.get("description") if isinstance(payload, dict) else None
            ),
            "duration_minutes": (
                payload.get("duration_minutes") if isinstance(payload, dict) else None
            ),
            "intensity": (
                payload.get("intensity") if isinstance(payload, dict) else None
            ),
            "status": raw.get("status"),
        }
        # Drop keys whose value is None so the frontend's "absent vs
        # null" distinction is preserved.
        inline_sessions.append({k: v for k, v in slim.items() if v is not None})

    start_date = window_sessions[0].get("date") if window_sessions else None
    end_date = window_sessions[-1].get("date") if window_sessions else None

    props: dict = {
        "start_date": start_date,
        "end_date": end_date,
        "sessions": inline_sessions,
        "truncated_in_ui": truncated,
    }

    return {
        "type": "plan_preview",
        "id": uuid.uuid4().hex,
        "props": props,
    }


# ---------------------------------------------------------------------------
# File-backed fallback for environments without Supabase
# ---------------------------------------------------------------------------


def _file_store_path() -> Path:
    """Return the path to the file-backed training_sessions store.

    The fallback is used in CLI / test environments where Supabase is
    not configured. Each row is appended/updated in a single JSON file
    so the tests can round-trip without a network dependency.
    """
    base = Path("data/training_sessions")
    base.mkdir(parents=True, exist_ok=True)
    return base / "rows.json"


def _file_load_rows() -> list[dict]:
    path = _file_store_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text() or "[]")
    except json.JSONDecodeError:
        logger.warning("training_sessions store is corrupt; resetting")
        return []


def _file_write_rows(rows: list[dict]) -> None:
    _file_store_path().write_text(json.dumps(rows, indent=2, default=str))


# ---------------------------------------------------------------------------
# Persistence layer (Supabase + file fallback share the same shape)
# ---------------------------------------------------------------------------


def _insert_sessions_supabase(user_id: str, rows: list[dict]) -> list[dict]:
    from src.db.client import get_supabase

    db = get_supabase()
    result = db.table("training_sessions").insert(rows).execute()
    return result.data or []


def _delete_planned_range_supabase(
    user_id: str, start_date: str, end_date: str
) -> int:
    from src.db.client import get_supabase

    db = get_supabase()
    # Re-propose semantics: only ``planned`` rows are deleted. Completed
    # / skipped / modulated rows are history, not a draft, and stay put.
    result = (
        db.table("training_sessions")
        .delete()
        .eq("user_id", user_id)
        .eq("status", "planned")
        .gte("date", start_date)
        .lte("date", end_date)
        .execute()
    )
    deleted = result.data or []
    return len(deleted)


def _select_window_supabase(
    user_id: str, start_date: str, end_date: str
) -> list[dict]:
    from src.db.client import get_supabase

    db = get_supabase()
    result = (
        db.table("training_sessions")
        .select("*")
        .eq("user_id", user_id)
        .gte("date", start_date)
        .lte("date", end_date)
        .order("date", desc=False)
        .execute()
    )
    return result.data or []


def _select_session_supabase(user_id: str, session_id: str) -> dict | None:
    from src.db.client import get_supabase

    db = get_supabase()
    result = (
        db.table("training_sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("id", session_id)
        .maybe_single()
        .execute()
    )
    return result.data if result is not None else None


def _update_session_supabase(
    user_id: str, session_id: str, patch: dict
) -> dict | None:
    from src.db.client import get_supabase

    db = get_supabase()
    result = (
        db.table("training_sessions")
        .update(patch)
        .eq("user_id", user_id)
        .eq("id", session_id)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def _max_planned_date_supabase(user_id: str) -> str | None:
    from src.db.client import get_supabase

    db = get_supabase()
    result = (
        db.table("training_sessions")
        .select("date")
        .eq("user_id", user_id)
        .eq("status", "planned")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        return None
    return rows[0].get("date")


# ---------------------------------------------------------------------------
# Public registration
# ---------------------------------------------------------------------------


def register_session_tools(registry: ToolRegistry, user_id: str) -> None:
    """Register the session lifecycle tools bound to ``user_id``.

    The signature matches the legacy ``register_session_tools`` call
    site so ``registry.py`` does not need to change. The previous
    deprecated ``search_session_history`` helper has been retired in
    favour of the lifecycle tools.
    """
    settings = get_settings()
    use_supabase = bool(getattr(settings, "use_supabase", False))

    # ---- helpers bound to the closure -----------------------------------

    def _insert_sessions(rows: list[dict]) -> list[dict]:
        if use_supabase:
            return _insert_sessions_supabase(user_id, rows)
        store = _file_load_rows()
        for row in rows:
            new_row = {
                "id": uuid.uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **row,
            }
            store.append(new_row)
        _file_write_rows(store)
        # Return the newly inserted rows in insertion order.
        return store[-len(rows):]

    def _delete_planned_range(start_date: str, end_date: str) -> int:
        if use_supabase:
            return _delete_planned_range_supabase(user_id, start_date, end_date)
        store = _file_load_rows()
        kept: list[dict] = []
        removed = 0
        for row in store:
            if (
                row.get("user_id") == user_id
                and row.get("status") == "planned"
                and start_date <= str(row.get("date", "")) <= end_date
            ):
                removed += 1
                continue
            kept.append(row)
        _file_write_rows(kept)
        return removed

    def _select_window(start_date: str, end_date: str) -> list[dict]:
        if use_supabase:
            return _select_window_supabase(user_id, start_date, end_date)
        store = _file_load_rows()
        rows = [
            row for row in store
            if row.get("user_id") == user_id
            and start_date <= str(row.get("date", "")) <= end_date
        ]
        rows.sort(key=lambda r: str(r.get("date", "")))
        return rows

    def _select_session(session_id: str) -> dict | None:
        if use_supabase:
            return _select_session_supabase(user_id, session_id)
        for row in _file_load_rows():
            if row.get("user_id") == user_id and row.get("id") == session_id:
                return row
        return None

    def _update_session(session_id: str, patch: dict) -> dict | None:
        if use_supabase:
            return _update_session_supabase(user_id, session_id, patch)
        store = _file_load_rows()
        updated: dict | None = None
        for row in store:
            if row.get("user_id") == user_id and row.get("id") == session_id:
                row.update(patch)
                row["updated_at"] = datetime.now(timezone.utc).isoformat()
                updated = row
                break
        _file_write_rows(store)
        return updated

    def _max_planned_date() -> str | None:
        if use_supabase:
            return _max_planned_date_supabase(user_id)
        rows = [
            row for row in _file_load_rows()
            if row.get("user_id") == user_id and row.get("status") == "planned"
        ]
        if not rows:
            return None
        return max(str(r.get("date", "")) for r in rows)

    # ---- propose_sessions ------------------------------------------------

    def propose_sessions(
        start_date: str,
        end_date: str,
        sessions: list[dict],
    ) -> dict:
        """Persist a rolling window of upcoming training sessions.

        Replaces any existing ``planned`` rows in the given date range
        for this user so the coach can re-propose without manual
        deletion. Returns ``{count, sessions, plan_preview}`` and emits
        a ``plan_preview`` UI component the frontend renders inline.
        """
        try:
            _parse_iso_date(start_date)
            _parse_iso_date(end_date)
        except (TypeError, ValueError):
            return {
                "error": "start_date and end_date must be YYYY-MM-DD strings",
            }
        if start_date > end_date:
            return {"error": "start_date must be <= end_date"}

        if not isinstance(sessions, list) or not sessions:
            return {"error": "sessions must be a non-empty list of dicts"}

        prepared_rows: list[dict] = []
        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                return {
                    "error": f"sessions[{index}] is not a dict",
                }
            session_date = raw.get("date")
            if not isinstance(session_date, str):
                return {
                    "error": f"sessions[{index}].date is required (YYYY-MM-DD)",
                }
            try:
                _parse_iso_date(session_date)
            except ValueError:
                return {
                    "error": f"sessions[{index}].date is not a valid YYYY-MM-DD",
                }
            if not (start_date <= session_date <= end_date):
                return {
                    "error": (
                        f"sessions[{index}].date {session_date!r} is outside "
                        f"the window [{start_date}, {end_date}]"
                    ),
                }
            modality = _normalize_modality(raw.get("modality"))
            if modality is None:
                return {
                    "error": f"sessions[{index}].modality must be a non-empty string",
                }
            payload = raw.get("payload") or {}
            if not isinstance(payload, dict):
                return {
                    "error": f"sessions[{index}].payload must be an object",
                }
            notes = raw.get("notes") or ""
            if not isinstance(notes, str):
                return {
                    "error": f"sessions[{index}].notes must be a string",
                }
            prepared_rows.append({
                "user_id": user_id,
                "date": session_date,
                "modality": modality,
                "status": "planned",
                "payload": payload,
                "planned_payload": payload,
                "notes": notes,
                "athlete_note": "",
            })

        try:
            replaced = _delete_planned_range(start_date, end_date)
            inserted = _insert_sessions(prepared_rows)
        except Exception as exc:
            logger.error("propose_sessions persistence failed: %s", exc, exc_info=True)
            return {"error": f"persistence failed: {exc}"}

        serialized = [_serialize_session_row(r) for r in inserted]
        # Sort by date so the preview reads chronologically regardless
        # of what the coach passed in.
        serialized.sort(key=lambda r: str(r.get("date", "")))

        return {
            "count": len(serialized),
            "replaced_existing": replaced,
            "start_date": start_date,
            "end_date": end_date,
            "sessions": serialized,
            "_ui_component": _build_plan_preview(serialized),
        }

    registry.register(Tool(
        name="propose_sessions",
        description=(
            "Persist a 14-day rolling window of upcoming training "
            "sessions and render an inline plan_preview card. Each "
            "session is `{date: YYYY-MM-DD, modality: free string, "
            "payload?: object, notes?: string}`. ``modality`` is FREE "
            "text - pick what fits the athlete (run, bike, swim, gym, "
            "hyrox, skitour, yoga, mobility, whatever). ``payload`` is "
            "also free-form: put whatever makes sense for the modality "
            "(intervals, RPE, distance, gear cues, etc.). Calling "
            "propose_sessions twice for the same date range replaces "
            "all existing PLANNED sessions in that range so you can "
            "re-propose freely. Completed / skipped / modulated rows "
            "are never touched."
        ),
        handler=propose_sessions,
        parameters={
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Window start (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "Window end (YYYY-MM-DD), inclusive.",
                },
                "sessions": {
                    "type": "array",
                    "description": (
                        "Sessions to plan in this window. Each entry is "
                        "{date, modality, payload?, notes?}."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "modality": {"type": "string"},
                            "payload": {"type": "object"},
                            "notes": {"type": "string"},
                        },
                        "required": ["date", "modality"],
                    },
                },
            },
            "required": ["start_date", "end_date", "sessions"],
        },
        category="planning",
        display_label="Plane deine naechsten Tage",
    ))

    # ---- modify_session --------------------------------------------------

    def modify_session(session_id: str, change: dict) -> dict:
        """Edit a session and (when substantive) flip status to modulated.

        ``change`` is a partial dict over ``{date, modality, payload,
        notes}``. Editing any of those fields counts as substantive and
        sets ``status = 'modulated'`` so the audit trail captures the
        reshape. The original prescription stays in ``planned_payload``.
        """
        if not isinstance(session_id, str) or not session_id:
            return {"error": "session_id is required"}
        if not isinstance(change, dict) or not change:
            return {"error": "change must be a non-empty dict"}

        existing = _select_session(session_id)
        if existing is None:
            return {"error": f"session {session_id!r} not found"}

        patch: dict = {}
        substantive = False

        if "date" in change:
            new_date = change["date"]
            if not isinstance(new_date, str):
                return {"error": "change.date must be a YYYY-MM-DD string"}
            try:
                _parse_iso_date(new_date)
            except ValueError:
                return {"error": "change.date is not a valid YYYY-MM-DD"}
            if new_date != existing.get("date"):
                patch["date"] = new_date
                substantive = True

        if "modality" in change:
            modality = _normalize_modality(change.get("modality"))
            if modality is None:
                return {"error": "change.modality must be a non-empty string"}
            if modality != existing.get("modality"):
                patch["modality"] = modality
                substantive = True

        if "payload" in change:
            payload = change["payload"]
            if not isinstance(payload, dict):
                return {"error": "change.payload must be an object"}
            if payload != (existing.get("payload") or {}):
                patch["payload"] = payload
                substantive = True

        if "notes" in change:
            notes = change["notes"]
            if not isinstance(notes, str):
                return {"error": "change.notes must be a string"}
            if notes != (existing.get("notes") or ""):
                patch["notes"] = notes
                substantive = True

        if not patch:
            # No-op modify: return the existing row, do not flip status.
            return {
                "modified": False,
                "session": _serialize_session_row(existing),
            }

        if substantive and existing.get("status") == "planned":
            patch["status"] = "modulated"

        try:
            updated = _update_session(session_id, patch)
        except Exception as exc:
            logger.error("modify_session failed: %s", exc, exc_info=True)
            return {"error": f"persistence failed: {exc}"}

        if updated is None:
            return {"error": "update returned no row"}

        return {
            "modified": True,
            "substantive": substantive,
            "session": _serialize_session_row(updated),
        }

    registry.register(Tool(
        name="modify_session",
        description=(
            "Edit a previously proposed session. ``change`` is a "
            "partial dict over {date, modality, payload, notes}. Any "
            "substantive change flips status to 'modulated' and keeps "
            "the original prescription in ``planned_payload`` for the "
            "audit trail. Use this when the athlete needs to shift a "
            "day, swap modality, or tighten notes after life shifted."
        ),
        handler=modify_session,
        parameters={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "ID of the session to edit.",
                },
                "change": {
                    "type": "object",
                    "description": "Partial fields to update.",
                },
            },
            "required": ["session_id", "change"],
        },
        category="planning",
        display_label="Passe eine Einheit an",
    ))

    # ---- complete_session ------------------------------------------------

    def complete_session(
        session_id: str,
        athlete_note: str = "",
        status: str = "completed",
    ) -> dict:
        """Mark a session done or skipped with an optional athlete note."""
        if not isinstance(session_id, str) or not session_id:
            return {"error": "session_id is required"}
        if status not in {"completed", "skipped"}:
            return {
                "error": "status must be 'completed' or 'skipped'",
            }
        if not isinstance(athlete_note, str):
            return {"error": "athlete_note must be a string"}

        existing = _select_session(session_id)
        if existing is None:
            return {"error": f"session {session_id!r} not found"}

        patch = {
            "status": status,
            "athlete_note": athlete_note,
        }

        try:
            updated = _update_session(session_id, patch)
        except Exception as exc:
            logger.error("complete_session failed: %s", exc, exc_info=True)
            return {"error": f"persistence failed: {exc}"}

        if updated is None:
            return {"error": "update returned no row"}

        return {
            "completed": True,
            "session": _serialize_session_row(updated),
        }

    registry.register(Tool(
        name="complete_session",
        description=(
            "Mark a session done (status='completed') or skipped "
            "(status='skipped') with an optional free-form athlete "
            "note. The note is for the athlete's voice - their take "
            "on how the session went - and is preserved verbatim in "
            "the audit trail."
        ),
        handler=complete_session,
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "athlete_note": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["completed", "skipped"],
                },
            },
            "required": ["session_id"],
        },
        category="planning",
        display_label="Schliesse eine Einheit ab",
    ))

    # ---- get_session_window ----------------------------------------------

    def get_session_window(days_ahead: int = 14, days_back: int = 7) -> dict:
        """Return planned + recently-completed sessions in a window.

        ``days_ahead`` covers the rolling window the coach is curating;
        ``days_back`` is recent history so the coach can see what the
        athlete actually completed. Defaults give a 21-day view (14
        ahead, 7 back) which is the canonical context size.
        """
        try:
            days_ahead_i = int(days_ahead)
            days_back_i = int(days_back)
        except (TypeError, ValueError):
            return {"error": "days_ahead and days_back must be integers"}

        days_ahead_i = max(0, min(days_ahead_i, _MAX_WINDOW_DAYS_AHEAD))
        days_back_i = max(0, min(days_back_i, _MAX_WINDOW_DAYS_BACK))

        today = _date_cls.today()
        start = (today - timedelta(days=days_back_i)).isoformat()
        end = (today + timedelta(days=days_ahead_i)).isoformat()

        try:
            rows = _select_window(start, end)
        except Exception as exc:
            logger.error("get_session_window failed: %s", exc, exc_info=True)
            return {"error": f"read failed: {exc}"}

        serialized = [_serialize_session_row(r) for r in rows]
        planned = [s for s in serialized if s.get("status") == "planned"]
        completed = [s for s in serialized if s.get("status") == "completed"]
        skipped = [s for s in serialized if s.get("status") == "skipped"]
        modulated = [s for s in serialized if s.get("status") == "modulated"]

        return {
            "today": today.isoformat(),
            "window_start": start,
            "window_end": end,
            "count": len(serialized),
            "planned_count": len(planned),
            "completed_count": len(completed),
            "skipped_count": len(skipped),
            "modulated_count": len(modulated),
            "sessions": serialized,
        }

    registry.register(Tool(
        name="get_session_window",
        description=(
            "Read the athlete's 14-day rolling window: planned "
            "sessions ahead plus recently-completed history. Defaults "
            "to 14 days ahead + 7 days back. Use this BEFORE proposing "
            "new sessions so you know what is already on the schedule."
        ),
        handler=get_session_window,
        parameters={
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "How far forward to read (default 14).",
                },
                "days_back": {
                    "type": "integer",
                    "description": "How far back to read (default 7).",
                },
            },
        },
        category="planning",
        display_label="Lese deine naechsten Tage",
    ))

    # ---- extend_window ---------------------------------------------------

    def extend_window(days: int = 14, sessions: list[dict] | None = None) -> dict:
        """Append more planned sessions starting after the current end.

        Convenience wrapper around ``propose_sessions``: the coach
        calls it when the rolling window has fewer than three days of
        runway left. ``days`` is the length of the new chunk; ``sessions``
        must fall within the new chunk. When ``sessions`` is omitted
        this returns the suggested ``start_date`` / ``end_date`` so the
        coach can compose the prescription and call again.
        """
        try:
            days_i = int(days)
        except (TypeError, ValueError):
            return {"error": "days must be an integer"}
        days_i = max(1, min(days_i, _MAX_WINDOW_DAYS_AHEAD))

        try:
            current_end = _max_planned_date()
        except Exception as exc:
            logger.error("extend_window read failed: %s", exc, exc_info=True)
            return {"error": f"read failed: {exc}"}

        today = _date_cls.today()
        if current_end:
            try:
                end_date_obj = _parse_iso_date(current_end)
            except ValueError:
                end_date_obj = today
        else:
            end_date_obj = today - timedelta(days=1)

        new_start_obj = max(end_date_obj + timedelta(days=1), today)
        new_end_obj = new_start_obj + timedelta(days=days_i - 1)
        new_start = new_start_obj.isoformat()
        new_end = new_end_obj.isoformat()

        if not sessions:
            return {
                "extended": False,
                "suggested_start_date": new_start,
                "suggested_end_date": new_end,
                "message": (
                    "Call propose_sessions(start_date, end_date, sessions) "
                    "with this window to extend the schedule."
                ),
            }

        result = propose_sessions(new_start, new_end, sessions)
        if "error" in result:
            return result
        # Tag the response so the caller knows it came from extend_window.
        return {**result, "extended": True}

    registry.register(Tool(
        name="extend_window",
        description=(
            "Append another chunk to the rolling window. Call this "
            "when the current window has <3 days of runway left. With "
            "``sessions`` omitted it returns the suggested date range "
            "so you can reason about what to prescribe. With "
            "``sessions`` it appends them via propose_sessions."
        ),
        handler=extend_window,
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Length of the new chunk in days (default 14).",
                },
                "sessions": {
                    "type": "array",
                    "description": (
                        "Optional sessions to plan. When omitted, the "
                        "tool returns the suggested date range without "
                        "writing anything."
                    ),
                    "items": {"type": "object"},
                },
            },
        },
        category="planning",
        display_label="Erweitere dein Fenster",
    ))
