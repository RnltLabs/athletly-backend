"""Persona markdown loader.

Parses the YAML frontmatter + markdown sections into a typed dataclass.
Used by seed.py, reset.py, chat.py, and the fake-data generators.

No external deps: we lean on the stdlib + PyYAML which is already present
in the supabase-py transitive dependency tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Persona files live alongside this module under personas/.
_PERSONAS_DIR = Path(__file__).parent / "personas"


@dataclass(frozen=True)
class Persona:
    """Structured persona definition parsed from a markdown file."""

    slug: str
    name: str
    email: str
    age: int
    location: str
    sports: list[str]
    goal_event: str
    goal_date: str
    goal_target_time: str
    goal_type: str
    training_days_per_week: int
    max_session_minutes: int
    estimated_vo2max: float
    threshold_pace_min_km: float
    weekly_volume_km: float
    weeks_of_history: int
    rhr_baseline_bpm: int
    hrv_baseline_ms: float
    sleep_hours_baseline: float
    has_active_plan: bool
    # Optional / sport-specific fields.
    goal_pace_min_km: float | None = None
    ftp_watts: int | None = None
    swim_css_min_per_100m: float | None = None
    injury_history: list[dict] = field(default_factory=list)
    # Markdown sections keyed by H2 heading (e.g. "Identity", "Goal").
    sections: dict[str, str] = field(default_factory=dict)
    # Raw source path for diagnostics.
    source_path: str = ""


# Required frontmatter keys. Missing any of these is a hard error so that
# the seeder never silently produces a half-populated test user.
_REQUIRED_KEYS = (
    "slug",
    "name",
    "email",
    "age",
    "location",
    "sports",
    "goal_event",
    "goal_date",
    "goal_target_time",
    "goal_type",
    "training_days_per_week",
    "max_session_minutes",
    "estimated_vo2max",
    "threshold_pace_min_km",
    "weekly_volume_km",
    "weeks_of_history",
    "rhr_baseline_bpm",
    "hrv_baseline_ms",
    "sleep_hours_baseline",
    "has_active_plan",
)


def persona_path(slug: str) -> Path:
    """Resolve the markdown file path for a persona slug."""
    return _PERSONAS_DIR / f"{slug}.md"


def list_personas() -> list[str]:
    """Return all available persona slugs sorted alphabetically."""
    return sorted(p.stem for p in _PERSONAS_DIR.glob("*.md"))


def load_persona(slug: str) -> Persona:
    """Load and validate a persona definition by slug.

    Raises:
        FileNotFoundError: if no markdown file exists for *slug*.
        ValueError: if frontmatter is missing required keys.
    """
    path = persona_path(slug)
    if not path.exists():
        raise FileNotFoundError(
            f"No persona at {path}. Available: {list_personas()}"
        )

    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    sections = _parse_sections(body)

    missing = [k for k in _REQUIRED_KEYS if k not in frontmatter]
    if missing:
        raise ValueError(
            f"Persona {slug!r} is missing required frontmatter keys: {missing}"
        )

    return Persona(
        slug=str(frontmatter["slug"]),
        name=str(frontmatter["name"]),
        email=str(frontmatter["email"]),
        age=int(frontmatter["age"]),
        location=str(frontmatter["location"]),
        sports=list(frontmatter["sports"]),
        goal_event=str(frontmatter["goal_event"]),
        goal_date=str(frontmatter["goal_date"]),
        goal_target_time=str(frontmatter["goal_target_time"]),
        goal_type=str(frontmatter["goal_type"]),
        training_days_per_week=int(frontmatter["training_days_per_week"]),
        max_session_minutes=int(frontmatter["max_session_minutes"]),
        estimated_vo2max=float(frontmatter["estimated_vo2max"]),
        threshold_pace_min_km=float(frontmatter["threshold_pace_min_km"]),
        weekly_volume_km=float(frontmatter["weekly_volume_km"]),
        weeks_of_history=int(frontmatter["weeks_of_history"]),
        rhr_baseline_bpm=int(frontmatter["rhr_baseline_bpm"]),
        hrv_baseline_ms=float(frontmatter["hrv_baseline_ms"]),
        sleep_hours_baseline=float(frontmatter["sleep_hours_baseline"]),
        has_active_plan=bool(frontmatter["has_active_plan"]),
        goal_pace_min_km=_optional_float(frontmatter.get("goal_pace_min_km")),
        ftp_watts=_optional_int(frontmatter.get("ftp_watts")),
        swim_css_min_per_100m=_optional_float(
            frontmatter.get("swim_css_min_per_100m"),
        ),
        injury_history=list(frontmatter.get("injury_history") or []),
        sections=sections,
        source_path=str(path),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file with YAML frontmatter into (dict, body)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            "Persona file is missing YAML frontmatter (--- ... --- block at top)."
        )
    raw_yaml, body = match.group(1), match.group(2)
    parsed = yaml.safe_load(raw_yaml) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Persona frontmatter must be a YAML mapping.")
    return parsed, body


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _parse_sections(body: str) -> dict[str, str]:
    """Split the markdown body into sections keyed by H2 heading."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        sections[title] = body[start:end].strip()
    return sections


def _optional_float(value) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value) -> int | None:
    return int(value) if value is not None else None
