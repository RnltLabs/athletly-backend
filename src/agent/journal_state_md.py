"""Render the typed journal state (intents + programs + coach_notes) to
markdown for system-prompt injection.

The three typed tables in ``src/db/intents_db.py`` / ``programs_db.py`` /
``coach_notes_db.py`` are the source of truth. The agent never sees the
JSON rows directly: each turn we render a compact markdown block and
append it to the athlete journal that already lives in the system prompt.

Design notes:

* The output is *plain markdown* (no decorative dashes, no bold markers
  in body text). The system prompt allows markdown headings, but the
  body of each block stays clean so any quoted snippet survives the
  constitutional critic.
* Coach Notes are truncated to ``notes_limit`` rows (default 10). The
  agent can call ``list_coach_notes(limit=50)`` if it needs more.
* Empty categories produce empty strings; we never emit a stub heading
  with no body.

Hooked into :func:`src.agent.athlete_md.build_athlete_md` so the rendered
block is injected each turn.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# Default number of coach notes to include in the injected markdown block.
DEFAULT_NOTES_LIMIT = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_journal_state_md(
    user_id: str,
    *,
    notes_limit: int = DEFAULT_NOTES_LIMIT,
) -> str:
    """Render the three typed state blocks for ``user_id``.

    Returns the concatenated markdown string with leading/trailing blank
    lines trimmed. Returns ``""`` when the athlete has no intents, no
    programs, and no notes (empty stubs are intentionally suppressed -
    they only add tokens without information).
    """
    try:
        from src.db.coach_notes_db import list_coach_notes
        from src.db.intents_db import list_intents
        from src.db.programs_db import list_programs

        intents = list_intents(user_id, status_filter="active")
        programs = list_programs(user_id, active_only=True)
        notes = list_coach_notes(user_id, limit=notes_limit)
    except Exception:
        logger.exception("render_journal_state_md fetch failed for %s", user_id)
        return ""

    return _render_blocks(intents=intents, programs=programs, notes=notes)


# ---------------------------------------------------------------------------
# Pure rendering helpers (no DB)
# ---------------------------------------------------------------------------


def _render_blocks(
    *,
    intents: list[dict],
    programs: list[dict],
    notes: list[dict],
) -> str:
    """Compose the three sections into one block. Pure function for tests."""
    parts: list[str] = []

    intents_md = _render_intents(intents)
    if intents_md:
        parts.append(intents_md)

    programs_md = _render_programs(programs)
    if programs_md:
        parts.append(programs_md)

    notes_md = _render_coach_notes(notes)
    if notes_md:
        parts.append(notes_md)

    return "\n\n".join(parts).strip()


def _render_intents(intents: list[dict]) -> str:
    """Render the ``## Active Intents`` block.

    Returns ``""`` when ``intents`` is empty so we never inject an empty
    stub. One bullet per intent, ordered by priority asc then created_at desc
    (the DB query already returns them in that order).
    """
    if not intents:
        return ""

    lines = ["## Active Intents", ""]
    for row in intents:
        lines.append(_format_intent_bullet(row))
    return "\n".join(lines)


def _format_intent_bullet(row: dict) -> str:
    """Render one intent row as a single markdown bullet."""
    priority = _safe_int(row.get("priority"), default=3)
    type_label = (row.get("type") or "intent").strip()
    title = (row.get("title") or "").strip() or "(no title)"
    status = (row.get("status") or "active").strip()
    created = _format_date(row.get("created_at"))

    # Capitalise the type label for visual scanability but keep the raw
    # casing if the coach used something custom (e.g. "race_b").
    pretty_type = _title_case_type(type_label)

    meta_bits: list[str] = []
    meta_bits.append(f"status: {status}")
    if created:
        meta_bits.append(f"created {created}")

    payload_summary = _render_payload_summary(row.get("payload"))
    if payload_summary:
        meta_bits.append(payload_summary)

    meta = ", ".join(meta_bits)
    return f"- [Priority {priority} - {pretty_type}] {title} ({meta})"


def _render_programs(programs: list[dict]) -> str:
    """Render the ``## Active Programs`` block.

    Each program gets a ``### <label>`` subheading plus optional body
    composed from the JSON payload. Returns ``""`` if no programs.
    """
    if not programs:
        return ""

    lines = ["## Active Programs", ""]
    for row in programs:
        lines.extend(_format_program_block(row))
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_program_block(row: dict) -> list[str]:
    """Render one program row as a heading + optional body lines."""
    kind = (row.get("kind") or "program").strip()
    label = (row.get("label") or "").strip() or kind
    created = _format_date(row.get("created_at"))

    header_bits = [kind]
    if created:
        header_bits.append(f"since {created}")
    header = f"### {label} ({', '.join(header_bits)})"

    body_lines = _render_program_body(row.get("payload"))

    out: list[str] = [header]
    if body_lines:
        out.extend(body_lines)
    return out


def _render_program_body(payload) -> list[str]:
    """Render the JSON payload of a program as readable markdown lines.

    Convention: a ``notes`` key (str) renders as a free-form line.
    Any other scalar key renders as ``key: value``. Lists render as
    ``key: a, b, c``. Empty / None payloads produce no lines.
    """
    if not isinstance(payload, dict) or not payload:
        return []

    lines: list[str] = []

    notes = payload.get("notes")
    if isinstance(notes, str) and notes.strip():
        lines.append(f"Coach notes: {notes.strip()}")

    for key, value in payload.items():
        if key == "notes":
            continue
        rendered = _scalar_to_str(value)
        if rendered is None:
            continue
        # Convert snake_case key to "Title case" label.
        label = key.replace("_", " ").strip().capitalize()
        lines.append(f"{label}: {rendered}")

    return lines


def _render_coach_notes(notes: list[dict]) -> str:
    """Render the ``## Coach Notes`` block. Newest first."""
    if not notes:
        return ""

    limit = len(notes)
    lines = [f"## Coach Notes (last {limit})", ""]
    for row in notes:
        lines.append(_format_coach_note_bullet(row))
    return "\n".join(lines)


def _format_coach_note_bullet(row: dict) -> str:
    """One coach note bullet, with date prefix and optional scope tag."""
    date_str = _format_date(row.get("created_at")) or "unknown"
    content = (row.get("content") or "").strip()
    scope = (row.get("scope") or "").strip()
    if scope:
        return f"- {date_str}: {content} (scope: {scope})"
    return f"- {date_str}: {content}"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _safe_int(value, *, default: int) -> int:
    """Best-effort int coercion. Returns ``default`` on failure."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_date(value) -> str:
    """Render a Supabase timestamp as ``YYYY-MM-DD``. Empty string on failure."""
    if not value:
        return ""
    if isinstance(value, str):
        # Supabase timestamps look like ``2026-05-12T08:30:45.123456+00:00``.
        # We only want the date portion. Split on T first for cheap path.
        head = value.split("T", 1)[0]
        if len(head) == 10 and head.count("-") == 2:
            return head
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except ValueError:
            return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    return ""


def _title_case_type(raw: str) -> str:
    """Map a free-text intent type to a tidy display label.

    Known types (``event``, ``capacity``, ``habit``, ``recovery``) get a
    title-cased rendering. Custom types are passed through unchanged so
    coach-specific tags survive (e.g. ``race_b``).
    """
    lower = raw.lower()
    canonical = {
        "event": "Event",
        "capacity": "Capacity",
        "habit": "Habit",
        "recovery": "Recovery",
    }
    return canonical.get(lower, raw)


def _render_payload_summary(payload) -> str:
    """Render a one-line summary for an intent payload.

    We highlight a few common keys (``race_date``, ``target_time``,
    ``target_lift``) so the coach sees the essentials. Unknown keys are
    omitted to keep the bullet readable - the agent can call
    ``list_intents`` for the full JSON.
    """
    if not isinstance(payload, dict) or not payload:
        return ""

    bits: list[str] = []
    for key in ("race_date", "target_time", "target_lift", "target_value"):
        value = payload.get(key)
        rendered = _scalar_to_str(value)
        if rendered is not None:
            label = key.replace("_", " ")
            bits.append(f"{label} {rendered}")
    return ", ".join(bits)


def _scalar_to_str(value) -> str | None:
    """Convert a scalar / list-of-scalars value to a display string.

    Returns ``None`` when the value is missing, empty, or a non-renderable
    structured type (nested dict). Keeps the renderer defensive against
    arbitrary JSON shapes the coach might write.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, list):
        rendered_items: list[str] = []
        for item in value:
            piece = _scalar_to_str(item)
            if piece is not None:
                rendered_items.append(piece)
        if not rendered_items:
            return None
        return ", ".join(rendered_items)
    return None
