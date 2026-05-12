"""Skill loader: scans playbooks/*.md files into in-memory Skill objects.

Frontmatter schema (YAML, between leading ``---`` lines):

    ---
    name: onboarding
    description: When the athlete has an incomplete profile, walk them
      through setup naturally - one question at a time.
    when_to_use: profiles.onboarding_complete = false
    required_tools:
      - update_profile
      - update_goal
      - add_belief
      - edit_athlete_memory
    ---

The body is plain markdown - the playbook itself. ``invoke_skill`` returns
the body verbatim so the agent reads it and follows the steps.

Loading is lazy + cached: first ``load_skills()`` scan; later calls are
free. Returns immutable ``Skill`` dataclasses. Failures (bad YAML,
missing fields) are logged and the skill is skipped, never crashes the
agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PLAYBOOKS_DIR = Path(__file__).parent / "playbooks"


@dataclass(frozen=True)
class Skill:
    """A loaded skill - frontmatter metadata + body playbook."""

    name: str
    description: str
    when_to_use: str = ""
    required_tools: tuple[str, ...] = field(default_factory=tuple)
    body: str = ""
    source_path: str = ""


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split markdown into (frontmatter_dict, body_text).

    Returns ({}, full_text) if no frontmatter block is present. Uses a
    tiny YAML-ish parser - we accept only ``key: value`` and ``key:``
    followed by indented ``- value`` lists. Full YAML would be overkill.
    """
    lines = text.splitlines(keepends=False)
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text

    meta_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:]).strip()

    data: dict = {}
    current_key: str | None = None
    for raw in meta_lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        # List item under current_key
        stripped = line.strip()
        if stripped.startswith("- ") and current_key:
            existing = data.get(current_key) or []
            if not isinstance(existing, list):
                existing = []
            existing.append(stripped[2:].strip())
            data[current_key] = existing
            continue
        # key: value or key: (list follows)
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = value
                current_key = None
            else:
                data[key] = []
                current_key = key
        # continuation line for multi-line value
        elif current_key and isinstance(data.get(current_key), str):
            data[current_key] = data[current_key] + " " + stripped
    return data, body


def _build_skill(path: Path, meta: dict, body: str) -> Skill | None:
    """Build a Skill dataclass from parsed frontmatter, or None on error."""
    name = (meta.get("name") or "").strip()
    description = (meta.get("description") or "").strip()
    if not name or not description:
        logger.warning(
            "Skill %s missing required name/description in frontmatter - skipped",
            path,
        )
        return None
    when_to_use = (meta.get("when_to_use") or "").strip()
    raw_tools = meta.get("required_tools") or []
    if isinstance(raw_tools, str):
        raw_tools = [t.strip() for t in raw_tools.split(",")]
    required_tools = tuple(t for t in raw_tools if t)
    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        required_tools=required_tools,
        body=body,
        source_path=str(path),
    )


@lru_cache(maxsize=1)
def load_skills() -> dict[str, Skill]:
    """Scan ``playbooks/`` and return a name -> Skill dict (cached).

    Cache is process-lifetime. To pick up changes during dev call
    ``load_skills.cache_clear()``.
    """
    result: dict[str, Skill] = {}
    if not PLAYBOOKS_DIR.is_dir():
        logger.info("No skills/playbooks directory at %s", PLAYBOOKS_DIR)
        return result
    for path in sorted(PLAYBOOKS_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            skill = _build_skill(path, meta, body)
            if skill is None:
                continue
            if skill.name in result:
                logger.warning(
                    "Duplicate skill name %s (skipping %s, keeping %s)",
                    skill.name, path, result[skill.name].source_path,
                )
                continue
            result[skill.name] = skill
        except Exception:
            logger.exception("Failed to load skill from %s", path)
    return result


def list_skills() -> list[Skill]:
    """Return all loaded skills as a stable-ordered list."""
    return sorted(load_skills().values(), key=lambda s: s.name)


def get_skill(name: str) -> Skill | None:
    """Return the named skill or None if it does not exist."""
    return load_skills().get(name)
