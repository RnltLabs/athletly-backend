"""Athlete journal: one markdown document per user, holding everything
identity-related the coach should know.

Replaces the multi-store identity layer (profiles columns + beliefs +
goal_history + athlete_md + preferences). Pragmatic single-source-of-
truth: read it once at the top of every turn, the agent has the full
picture.

Document structure (sections, in canonical order):

    # <Athlete Name>

    ## Identity
    Static facts: age, location, sports, lifestyle, body, constraints.

    ## Current Goal
    Event, date, target time, course facts, source/verification.

    ## Goal Timeline
    Append-only history of goal changes with reasoning.

    ## What I know about my training
    Performance facts: paces, PBs, VO2max estimates.

    ## Preferences
    How the athlete wants to be coached: output style, training time,
    surface, days off, etc.

    ## Open Threads
    Things to follow up on next turn: pain reports, mid-decision goals,
    'try this and see how it feels'.

    ## What we tried
    Coach learnings over time. What worked, what did not.

Sections are managed by tools. Missing sections are tolerated; the
coach can add them. Section order is preserved on update.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# Canonical section order. Sections not in this list are appended at end.
CANONICAL_SECTIONS = (
    "Identity",
    "Current Goal",
    "Goal Timeline",
    "What I know about my training",
    "Preferences",
    "Open Threads",
    "What we tried",
)


# ---------------------------------------------------------------------------
# Read / write the raw markdown
# ---------------------------------------------------------------------------


def read_journal(user_id: str) -> str:
    """Return the journal markdown for *user_id* (empty string if none)."""
    from src.db.client import get_supabase

    client = get_supabase()
    try:
        res = (
            client.table("athlete_journal")
            .select("content")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0].get("content") or ""
        return ""
    except Exception:
        logger.exception("read_journal failed for %s", user_id)
        return ""


def write_journal(user_id: str, content: str) -> None:
    """Upsert the full journal content for *user_id*."""
    from src.db.client import get_supabase

    client = get_supabase()
    client.table("athlete_journal").upsert(
        {
            "user_id": user_id,
            "content": content,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="user_id",
    ).execute()


# ---------------------------------------------------------------------------
# Section operations
# ---------------------------------------------------------------------------

_SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_sections(content: str) -> dict[str, str]:
    """Parse a journal into a dict of section_name -> body text.

    The title line ("# Name") is preserved under the key "_title". Body
    is everything between the heading and the next ``##`` heading,
    trimmed of leading/trailing whitespace.
    """
    sections: dict[str, str] = {}
    title_match = re.match(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    if title_match:
        sections["_title"] = title_match.group(1).strip()

    matches = list(_SECTION_HEADING.finditer(content))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        sections[name] = body
    return sections


def render_sections(sections: dict[str, str]) -> str:
    """Serialize sections dict back to markdown in canonical order."""
    lines: list[str] = []
    title = sections.get("_title")
    if title:
        lines.append(f"# {title}")
        lines.append("")

    written: set[str] = set()
    for name in CANONICAL_SECTIONS:
        if name in sections:
            lines.append(f"## {name}")
            lines.append("")
            if sections[name].strip():
                lines.append(sections[name].strip())
                lines.append("")
            written.add(name)

    # Any custom sections not in canonical order: append at the end.
    for name, body in sections.items():
        if name == "_title" or name in written:
            continue
        lines.append(f"## {name}")
        lines.append("")
        if body.strip():
            lines.append(body.strip())
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def update_section(user_id: str, section: str, content: str) -> str:
    """Replace the body of *section* (creates if missing). Returns new doc."""
    doc = read_journal(user_id)
    sections = parse_sections(doc)
    sections[section] = content.strip()
    new_doc = render_sections(sections)
    write_journal(user_id, new_doc)
    return new_doc


def append_to_section(user_id: str, section: str, content: str) -> str:
    """Append *content* as a new bullet to *section*. Returns new doc.

    If section is missing, creates it. Content is added as ``- ...``
    if it does not already start with a list marker.
    """
    doc = read_journal(user_id)
    sections = parse_sections(doc)
    line = content.strip()
    if not (line.startswith("- ") or line.startswith("* ")):
        line = f"- {line}"
    existing = sections.get(section, "").strip()
    sections[section] = f"{existing}\n{line}".strip() if existing else line
    new_doc = render_sections(sections)
    write_journal(user_id, new_doc)
    return new_doc


def remove_from_section(user_id: str, section: str, contains: str) -> str:
    """Remove lines from *section* that contain the substring *contains*.

    Used to resolve open threads: ``remove_from_section(uid, 'Open Threads',
    'knee pain')`` strips any matching bullet. Returns the new doc.
    """
    doc = read_journal(user_id)
    sections = parse_sections(doc)
    body = sections.get(section, "")
    if not body:
        return doc
    kept = [
        ln for ln in body.splitlines() if contains.lower() not in ln.lower()
    ]
    sections[section] = "\n".join(kept).strip()
    new_doc = render_sections(sections)
    write_journal(user_id, new_doc)
    return new_doc


# ---------------------------------------------------------------------------
# Initial bootstrap
# ---------------------------------------------------------------------------


def ensure_journal_exists(user_id: str, name: str | None = None) -> str:
    """Create an empty stub journal if the user does not yet have one."""
    current = read_journal(user_id)
    if current.strip():
        return current
    title = name or "Athlete"
    seed = (
        f"# {title}\n\n"
        f"## Identity\n\n"
        f"_(not yet filled in)_\n\n"
        f"## Current Goal\n\n"
        f"_(no active goal)_\n\n"
        f"## Goal Timeline\n\n"
        f"## What I know about my training\n\n"
        f"## Preferences\n\n"
        f"## Open Threads\n\n"
        f"## What we tried\n"
    )
    write_journal(user_id, seed)
    return seed
