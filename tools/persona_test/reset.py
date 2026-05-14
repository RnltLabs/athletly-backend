"""Wipe a persona test user back to a clean state.

Usage::

    python tools/persona_test/reset.py elena         # soft: keeps auth user
    python tools/persona_test/reset.py elena --hard  # also deletes auth user

Safety:
- Refuses to operate on a user whose user_metadata.is_test is not True.
- Every delete filters on the resolved user_id; nothing is unfiltered.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Path bootstrap so the script is runnable both as a module and as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.persona_test._persona import list_personas, load_persona
from tools.persona_test._supabase import (
    assert_is_test_user,
    clear_cached_jwt,
    clear_session_id,
    delete_test_user,
    find_user_by_email,
    get_service_client,
)

logger = logging.getLogger(__name__)


# Tables to clean per user. Order matters for FK cascades, but all of these
# are scoped to user_id so order is mostly cosmetic.
_PER_USER_TABLES = (
    "session_messages",
    "sessions",
    "episodes",
    "plans",
    "pending_actions",
    "health_daily_metrics",
    "import_manifest",
    "activities",
    "athlete_journal",
    "profiles",
)


def reset_persona(slug: str, *, hard: bool) -> dict:
    """Reset all data for a persona test user. Returns a summary dict."""
    persona = load_persona(slug)
    client = get_service_client()

    user = find_user_by_email(client, persona.email)
    if user is None:
        return {
            "slug": slug,
            "user_id": None,
            "deleted": {},
            "auth_deleted": False,
            "note": "Auth user did not exist - nothing to do.",
        }

    assert_is_test_user(user)
    user_id = user["id"]

    deleted_counts: dict[str, int] = {}
    for table in _PER_USER_TABLES:
        try:
            res = (
                client.table(table)
                .delete()
                .eq("user_id", user_id)
                .execute()
            )
            deleted_counts[table] = len(res.data or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete from %s failed: %s", table, exc)
            deleted_counts[table] = -1

    auth_deleted = False
    if hard:
        delete_test_user(client, user_id)
        auth_deleted = True

    clear_cached_jwt(slug)
    clear_session_id(slug)

    return {
        "slug": slug,
        "user_id": user_id,
        "deleted": deleted_counts,
        "auth_deleted": auth_deleted,
    }


def _render(summary: dict) -> str:
    lines = [
        f"Persona:      {summary['slug']}",
        f"User id:      {summary['user_id']}",
        f"Auth deleted: {summary['auth_deleted']}",
        "Deleted rows:",
    ]
    if not summary.get("deleted"):
        lines.append("  (none)")
    else:
        for table, count in summary["deleted"].items():
            lines.append(f"  {table:<24} {count}")
    if summary.get("note"):
        lines.append(f"Note: {summary['note']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset a persona test user. Safe-by-default (keeps auth user).",
    )
    parser.add_argument(
        "persona",
        help=f"Persona slug. Available: {list_personas()}",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Also delete the auth user (so the next seed will recreate it).",
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
    summary = reset_persona(args.persona, hard=args.hard)
    sys.stdout.write(_render(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
