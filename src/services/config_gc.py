"""Config Store Garbage Collection & LLM Consolidation.

Runs at session start to clean up stale agent-defined configurations:
- Archives configs not updated in 90+ days.
- Logs a warning if active config count exceeds 60.
- Dedup check: detects name-based duplicates across config tables.
- LLM consolidation: when active count > 60, an LLM pass merges
  semantically similar metric_definitions (Visionplan 8.12 D).
- Semantic similarity helpers for lightweight formula comparison.

Usage::

    from src.services.config_gc import run_config_gc, compute_config_similarity

    run_config_gc(user_id)
    score = compute_config_similarity("hr * duration * 0.5", "hr*duration*0.50")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

# Thresholds
_LOW_CONFIDENCE_THRESHOLD = 0.5
_STALE_DAYS = int(os.environ.get("ATHLETLY_GC_STALE_DAYS", "30"))
_MAX_ACTIVE_CAP = int(os.environ.get("ATHLETLY_GC_MAX_ACTIVE_CAP", "60"))

# Semantic dedup threshold (Visionplan 8.12 D)
SIMILARITY_THRESHOLD = float(os.environ.get("ATHLETLY_GC_SIMILARITY_THRESHOLD", "0.88"))
_FORMULA_WEIGHT = float(os.environ.get("ATHLETLY_GC_FORMULA_WEIGHT", "0.7"))
_NAME_WEIGHT = float(os.environ.get("ATHLETLY_GC_NAME_WEIGHT", "0.3"))

# Config tables to scan for GC
_CONFIG_TABLES = (
    "metric_definitions",
    "eval_criteria",
    "session_schemas",
    "periodization_models",
    "proactive_trigger_rules",
)


# ---------------------------------------------------------------------------
# Semantic similarity helpers (stdlib only — no embedding dependencies)
# ---------------------------------------------------------------------------


def compute_config_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two text strings.

    Uses ``difflib.SequenceMatcher`` which implements a variant of the
    Ratcliff/Obershelp algorithm. Returns a float in [0.0, 1.0].

    Both inputs are normalized (stripped, lowercased, whitespace-collapsed)
    before comparison.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical).
    """
    normalized1 = _normalize_text(text1)
    normalized2 = _normalize_text(text2)

    # Both empty → identical
    if not normalized1 and not normalized2:
        return 1.0
    # One empty, one not → completely different
    if not normalized1 or not normalized2:
        return 0.0

    return SequenceMatcher(None, normalized1, normalized2).ratio()


def compute_weighted_config_similarity(
    formula1: str,
    formula2: str,
    name1: str = "",
    name2: str = "",
    desc1: str = "",
    desc2: str = "",
) -> float:
    """Compute weighted similarity combining formula and name+description.

    Weights: formula similarity (0.7) + name/description similarity (0.3).

    Args:
        formula1: First formula string.
        formula2: Second formula string.
        name1: First name string.
        name2: Second name string.
        desc1: First description string.
        desc2: Second description string.

    Returns:
        Weighted similarity score in [0.0, 1.0].
    """
    formula_sim = compute_config_similarity(formula1, formula2)

    # Combine name and description for the identity component
    identity1 = f"{name1} {desc1}".strip()
    identity2 = f"{name2} {desc2}".strip()
    identity_sim = compute_config_similarity(identity1, identity2)

    return (_FORMULA_WEIGHT * formula_sim) + (_NAME_WEIGHT * identity_sim)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: strip, lowercase, collapse whitespace."""
    return " ".join(text.strip().lower().split())


def _archive_stale_configs(user_id: str) -> int:
    """Archive configs that have been superseded or are very old.

    Marks configs with valid_until already set (superseded) as truly archived
    by doing nothing — they're already versioned.  This function focuses on
    active configs that are stale (very old, not updated).

    Returns:
        Number of configs archived.
    """
    archived = 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS * 3)).isoformat()

    try:
        db = get_supabase()
    except Exception:
        logger.debug("GC: Supabase unavailable", exc_info=True)
        return 0

    for table in _CONFIG_TABLES:
        try:
            result = (
                db.table(table)
                .select("id, updated_at")
                .eq("user_id", user_id)
                .is_("valid_until", "null")
                .lt("updated_at", cutoff)
                .execute()
            )
            stale_rows = result.data or []
            for row in stale_rows:
                db.table(table).update({
                    "valid_until": datetime.now(timezone.utc).isoformat(),
                }).eq("id", row["id"]).execute()
                archived += 1
        except Exception:
            logger.debug("GC scan skipped for %s", table, exc_info=True)

    return archived


def _count_active_configs(user_id: str) -> int:
    """Count all active (non-archived) configs across all tables."""
    total = 0

    try:
        db = get_supabase()
    except Exception:
        logger.debug("GC: Supabase unavailable", exc_info=True)
        return 0

    for table in _CONFIG_TABLES:
        try:
            result = (
                db.table(table)
                .select("id", count="exact")
                .eq("user_id", user_id)
                .is_("valid_until", "null")
                .execute()
            )
            total += result.count or 0
        except Exception:
            logger.debug("Config count skipped for %s", table, exc_info=True)

    return total


def _check_duplicates(user_id: str) -> list[dict]:
    """Find duplicate config names within each table.

    Returns a list of ``{"table": str, "name": str, "count": int}`` dicts
    for any name that appears more than once in active configs.
    """
    duplicates = []

    try:
        db = get_supabase()
    except Exception:
        logger.debug("GC: Supabase unavailable", exc_info=True)
        return []

    for table in _CONFIG_TABLES:
        try:
            # Fetch all active names
            name_col = "sport" if table == "session_schemas" else "name"
            result = (
                db.table(table)
                .select(name_col)
                .eq("user_id", user_id)
                .is_("valid_until", "null")
                .execute()
            )
            rows = result.data or []
            names = [r.get(name_col, "") for r in rows]
            # Count occurrences
            seen: dict[str, int] = {}
            for name in names:
                seen[name] = seen.get(name, 0) + 1
            for name, count in seen.items():
                if count > 1:
                    duplicates.append({
                        "table": table,
                        "name": name,
                        "count": count,
                    })
        except Exception:
            logger.debug("Duplicate check skipped for %s", table, exc_info=True)

    return duplicates


# ---------------------------------------------------------------------------
# LLM-based config consolidation (Visionplan 8.12 D)
# ---------------------------------------------------------------------------

_CONSOLIDATION_MODEL = "gemini/gemini-2.5-flash"

_CONSOLIDATION_PROMPT = (
    "You are a sports analytics expert. Below is a list of metric definitions "
    "for a single user. Identify groups of semantically similar or redundant "
    "metrics that should be merged into one.\n\n"
    "RULES:\n"
    "1. Only group metrics that measure the SAME thing (different names/formulas "
    "for the same concept).\n"
    "2. For each group, pick the BEST name to keep (most descriptive, most "
    "commonly used).\n"
    "3. List the others as 'archive' candidates.\n"
    "4. Provide a brief reason for each merge.\n"
    "5. If no metrics should be merged, return an empty merge_groups array.\n"
    "6. Output ONLY valid JSON — no markdown, no explanation outside JSON.\n\n"
    "OUTPUT FORMAT:\n"
    '{{"merge_groups": [{{"keep": "name_to_keep", '
    '"archive": ["name1", "name2"], "reason": "..."}}]}}\n\n'
    "METRICS:\n{metrics_text}"
)


def _fetch_active_metrics(user_id: str) -> list[dict]:
    """Fetch all active metric_definitions for a user.

    Returns a list of dicts with 'name', 'formula', 'description' keys.
    Returns an empty list on any DB error.
    """
    try:
        db = get_supabase()
        result = (
            db.table("metric_definitions")
            .select("name, formula, description")
            .eq("user_id", user_id)
            .is_("valid_until", "null")
            .execute()
        )
        return result.data or []
    except Exception:
        logger.debug("Failed to fetch metrics for consolidation", exc_info=True)
        return []


def _build_metrics_text(metrics: list[dict]) -> str:
    """Format metrics as numbered text for the LLM prompt."""
    lines = []
    for i, m in enumerate(metrics, 1):
        name = m.get("name", "")
        formula = m.get("formula", "")
        desc = m.get("description", "")
        lines.append(f"{i}. {name}: formula='{formula}', description='{desc}'")
    return "\n".join(lines)


def _parse_merge_groups(llm_content: str) -> list[dict]:
    """Parse the LLM response into a list of merge group dicts.

    Returns an empty list if the response is not valid JSON or
    does not contain the expected structure.
    """
    try:
        parsed = json.loads(llm_content)
    except (json.JSONDecodeError, TypeError):
        logger.debug("Consolidation LLM returned non-JSON: %s", str(llm_content)[:200])
        return []

    groups = parsed.get("merge_groups")
    if not isinstance(groups, list):
        return []

    # Validate each group has required keys
    valid_groups = []
    for group in groups:
        if (
            isinstance(group, dict)
            and isinstance(group.get("keep"), str)
            and isinstance(group.get("archive"), list)
            and all(isinstance(n, str) for n in group["archive"])
        ):
            valid_groups.append(group)

    return valid_groups


def _archive_config_by_name(user_id: str, name: str) -> bool:
    """Archive a single metric_definition by setting valid_until.

    Returns True if the config was found and archived, False otherwise.
    """
    try:
        db = get_supabase()
        now = datetime.now(timezone.utc).isoformat()
        result = (
            db.table("metric_definitions")
            .update({"valid_until": now})
            .eq("user_id", user_id)
            .eq("name", name)
            .is_("valid_until", "null")
            .execute()
        )
        return bool(result.data)
    except Exception:
        logger.debug("Failed to archive config '%s'", name, exc_info=True)
        return False


def _consolidate_configs(user_id: str) -> int:
    """Run an LLM consolidation pass on metric_definitions.

    Asks the LLM to identify semantically similar metrics and archives
    the redundant ones. Best-effort: returns 0 on any failure.

    Returns:
        Number of configs archived via consolidation.
    """
    metrics = _fetch_active_metrics(user_id)
    if len(metrics) < 2:
        return 0

    metrics_text = _build_metrics_text(metrics)
    prompt = _CONSOLIDATION_PROMPT.format(metrics_text=metrics_text)

    try:
        from src.agent.llm import chat_completion

        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=_CONSOLIDATION_MODEL,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("LLM consolidation call failed", exc_info=True)
        return 0

    merge_groups = _parse_merge_groups(content)
    if not merge_groups:
        return 0

    # Build a set of valid metric names for safety
    valid_names = {m.get("name", "") for m in metrics}
    consolidated = 0

    for group in merge_groups:
        for name_to_archive in group["archive"]:
            if name_to_archive not in valid_names:
                logger.debug("Skipping unknown metric '%s'", name_to_archive)
                continue
            if name_to_archive == group["keep"]:
                continue  # Safety: never archive the "keep" metric
            if _archive_config_by_name(user_id, name_to_archive):
                consolidated += 1
                logger.info(
                    "Consolidated metric '%s' (kept '%s'): %s",
                    name_to_archive, group["keep"], group.get("reason", ""),
                )

    return consolidated


def run_config_gc(user_id: str) -> dict:
    """Run garbage collection for a user's agent configs.

    Called at session start (non-blocking, best-effort).

    Args:
        user_id: UUID of the user.

    Returns:
        Dict with GC results: ``archived``, ``active_count``, ``duplicates``,
        ``warning`` (if active count exceeds cap).
    """
    try:
        archived = _archive_stale_configs(user_id)
        active_count = _count_active_configs(user_id)
        duplicates = _check_duplicates(user_id)

        result: dict = {
            "archived": archived,
            "active_count": active_count,
            "duplicates": duplicates,
        }

        if active_count > _MAX_ACTIVE_CAP:
            logger.warning(
                "User %s has %d active configs (cap %d) — running consolidation",
                user_id, active_count, _MAX_ACTIVE_CAP,
            )
            consolidated = _consolidate_configs(user_id)
            result = {**result, "consolidated": consolidated}
            if consolidated > 0:
                # Re-count after consolidation
                result = {**result, "active_count": _count_active_configs(user_id)}
            result = {
                **result,
                "warning": (
                    f"Active config count ({active_count}) exceeded cap ({_MAX_ACTIVE_CAP}). "
                    f"Consolidated {consolidated} redundant metrics."
                ),
            }

        if archived > 0:
            logger.info(
                "Config GC for user %s: archived %d stale configs", user_id, archived,
            )

        if duplicates:
            logger.info(
                "Config duplicates for user %s: %s", user_id, duplicates,
            )

        return result
    except Exception:
        logger.warning("Config GC failed for user %s", user_id, exc_info=True)
        return {"archived": 0, "active_count": 0, "duplicates": []}
