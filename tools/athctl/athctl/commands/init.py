"""`athctl init` - bootstrap a throwaway test user.

The init flow is the trust-anchor for the whole test harness:

1. Create a fresh Supabase auth user via the admin API.
2. Insert an empty ``profiles`` row for them.
3. Clone Roman's Garmin tokens onto the new user (no fresh login needed).
4. Verify the connection and run an initial sync of activities/daily/sleep.
5. Optionally import Strava bootstrap tokens from env.
6. Write the new user id into ``~/.athctl/state.json``.

All side-effecting steps surface their outcome immediately via Rich so a
human running the command can see what happened; ``--json`` mode emits a
single structured summary at the end.
"""

from __future__ import annotations

import logging
import secrets
import sys
import time
from datetime import datetime, timezone
from typing import Any

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from athctl.common import (
    EXIT_AUTH,
    EXIT_BACKEND_DOWN,
    EXIT_NETWORK,
    as_json,
    console,
    emit,
    get_supabase,
)
from athctl.state import set_active_user

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Source user whose Garmin tokens get cloned onto every test user.
# This is Roman's real Athletly account. Hardcoded on purpose - the test
# harness is single-tenant by design and the value is not a secret.
SOURCE_USER_ID = "d511a8ce-9a35-476e-bbc1-840f3e7ee81e"

DEFAULT_ACTIVITIES_DAYS = 180
DEFAULT_DAILY_DAYS = 30
DEFAULT_SLEEP_DAYS = 30

TEST_EMAIL_DOMAIN = "athletly.dev"
TEST_EMAIL_LOCAL = "claude-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_test_email() -> str:
    """Return a unique throwaway email for the new test user.

    Uses an integer second-timestamp to keep the address short and easy
    to grep for. A random suffix is appended in the unlikely case two
    inits race within the same second.
    """
    ts = int(time.time())
    suffix = secrets.token_hex(2)
    return f"{TEST_EMAIL_LOCAL}+{ts}{suffix}@{TEST_EMAIL_DOMAIN}"


def _create_auth_user(email: str) -> dict[str, Any]:
    """Create a Supabase auth user via the admin API.

    Returns ``{"id": uuid, "email": ..., "password": ...}`` on success.
    Exits the process with ``EXIT_AUTH`` if creation fails - without an
    auth user the rest of init has nothing to attach data to.
    """
    password = secrets.token_urlsafe(32)
    client = get_supabase()
    try:
        response = client.auth.admin.create_user(
            {
                "email": email,
                "password": password,
                "email_confirm": True,
            }
        )
    except Exception as exc:
        emit(
            f"[red]Failed to create Supabase auth user:[/red] {exc}",
            data={"step": "auth.create_user", "status": "error", "error": str(exc)},
        )
        sys.exit(EXIT_AUTH)

    user = getattr(response, "user", None) or (
        response.get("user") if isinstance(response, dict) else None
    )
    if user is None:
        emit(
            "[red]Supabase admin.create_user returned no user object.[/red]",
            data={"step": "auth.create_user", "status": "error", "error": "no user in response"},
        )
        sys.exit(EXIT_AUTH)

    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    if not user_id:
        emit(
            "[red]Supabase admin.create_user returned no user id.[/red]",
            data={"step": "auth.create_user", "status": "error", "error": "no user id"},
        )
        sys.exit(EXIT_AUTH)

    return {"id": str(user_id), "email": email, "password": password}


def _insert_profile(user_id: str) -> None:
    """Insert a minimal empty profile row for *user_id*.

    The real profiles table is mostly flat (goal_event / goal_target_date
    / training_days_per_week / ... as top-level columns) plus a few JSONB
    catch-alls (``goal``, ``meta``). For bootstrap we set the minimum
    that satisfies NOT NULL constraints and let the agent populate the
    rest during onboarding.
    """
    client = get_supabase()
    try:
        client.table("profiles").upsert(
            {"user_id": user_id, "sports": [], "onboarding_complete": False},
            on_conflict="user_id",
        ).execute()
    except Exception as exc:
        emit(
            f"[red]Failed to insert profile row:[/red] {exc}",
            data={"step": "profiles.insert", "status": "error", "error": str(exc)},
        )
        sys.exit(EXIT_BACKEND_DOWN)


def _clone_garmin_tokens(target_uid: str) -> dict[str, Any]:
    """Copy Roman's Garmin tokens onto *target_uid*.

    Returns the provider response. Exits with EXIT_AUTH on failure - a
    test user without a working Garmin connection is useless.
    """
    from src.services.providers.garmin import GarminProvider

    result = GarminProvider().clone_tokens_from(SOURCE_USER_ID, target_uid)
    if result.get("status") != "cloned":
        emit(
            f"[red]Garmin token clone failed:[/red] {result.get('error')}",
            data={"step": "garmin.clone", "status": "error", "error": result.get("error")},
        )
        sys.exit(EXIT_AUTH)
    return result


def _verify_garmin(target_uid: str) -> dict[str, Any]:
    """Verify the cloned session is actually usable.

    Returns the check_connection dict on success, exits EXIT_NETWORK on
    failure. We do not retry: if Garth can't restore the session right
    after a clone, something is deeply wrong.
    """
    from src.services.providers.garmin import GarminProvider

    check = GarminProvider().check_connection(target_uid)
    if check.get("status") != "active":
        emit(
            f"[red]Garmin session check failed:[/red] {check}",
            data={"step": "garmin.check", "status": "error", "result": check},
        )
        sys.exit(EXIT_NETWORK)
    return check


def _sync_garmin(target_uid: str) -> dict[str, Any]:
    """Run the three initial Garmin syncs with a visible progress bar."""
    from src.services.providers.garmin import GarminProvider

    provider = GarminProvider()
    results: dict[str, Any] = {}

    steps = [
        ("activities", lambda: provider.sync_activities(target_uid, days=DEFAULT_ACTIVITIES_DAYS)),
        ("daily", lambda: provider.sync_daily_stats(target_uid, days=DEFAULT_DAILY_DAYS)),
        ("sleep", lambda: provider.sync_sleep(target_uid, days=DEFAULT_SLEEP_DAYS)),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for label, fn in steps:
            task = progress.add_task(f"Syncing Garmin {label}...", total=None)
            try:
                results[label] = fn()
            except Exception as exc:
                results[label] = {"status": "error", "error": str(exc)}
                logger.exception("Garmin %s sync raised", label)
            progress.update(task, completed=1)

    return results


def _maybe_import_strava(target_uid: str) -> dict[str, Any]:
    """Import bootstrap Strava tokens if the env vars are set.

    Strava is optional - missing tokens are not an error, just skipped.
    Failures to import are logged as warnings; we never abort the init
    flow on a Strava problem.
    """
    from src.config import get_settings

    settings = get_settings()
    access = (settings.strava_test_access_token or "").strip()
    refresh = (settings.strava_test_refresh_token or "").strip()

    if not access or not refresh:
        return {"status": "skipped", "reason": "no bootstrap tokens in env"}

    try:
        from src.services.providers.strava import StravaProvider

        result = StravaProvider().import_bootstrap_tokens(target_uid, access, refresh)
        return result
    except Exception as exc:
        logger.warning("Strava bootstrap import failed: %s", exc)
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@click.command(
    "init",
    help=(
        "Bootstrap a fresh throwaway test user that reuses Roman's Garmin "
        "tokens and (optionally) imports Strava bootstrap tokens from env."
    ),
)
@click.option(
    "--name",
    "name",
    type=str,
    default=None,
    help="Optional display name to store on the test profile (purely informational).",
)
@click.option(
    "--reuse-existing",
    "reuse_existing",
    is_flag=True,
    default=False,
    help=(
        "If an active test user already exists in ~/.athctl/state.json, "
        "verify it instead of creating a new one. (Future task - currently "
        "always creates a new user.)"
    ),
)
def init(name: str | None, reuse_existing: bool) -> None:  # noqa: ARG001
    """Bootstrap a fresh test user end-to-end."""
    summary: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_user_id": SOURCE_USER_ID,
    }

    # 1. Auth user
    email = _generate_test_email()
    auth = _create_auth_user(email)
    new_uid = auth["id"]
    summary["test_user_id"] = new_uid
    summary["test_user_email"] = email
    emit(
        f"[green]Created Supabase auth user[/green] {new_uid} ({email})",
        data={"step": "auth.create_user", "status": "ok", "user_id": new_uid, "email": email},
    )

    # 2. Profile row
    _insert_profile(new_uid)
    emit(
        "[green]Inserted empty profile row[/green]",
        data={"step": "profiles.insert", "status": "ok"},
    )

    # 3. Clone Garmin tokens
    clone = _clone_garmin_tokens(new_uid)
    summary["garmin_clone"] = clone
    emit(
        f"[green]Cloned Garmin tokens[/green] from {SOURCE_USER_ID} "
        f"(display: {clone.get('display_name')})",
        data={"step": "garmin.clone", "status": "ok", "result": clone},
    )

    # 4. Verify Garmin connection
    check = _verify_garmin(new_uid)
    summary["garmin_check"] = check
    emit(
        f"[green]Verified Garmin session[/green] (active, {check.get('provider_user_id')})",
        data={"step": "garmin.check", "status": "ok", "result": check},
    )

    # 5. Initial sync
    syncs = _sync_garmin(new_uid)
    summary["garmin_syncs"] = syncs
    act = syncs.get("activities", {})
    daily = syncs.get("daily", {})
    sleep = syncs.get("sleep", {})
    emit(
        (
            f"[green]Garmin sync done:[/green] "
            f"activities synced={act.get('synced')} skipped={act.get('skipped')}, "
            f"daily synced={daily.get('synced')}, "
            f"sleep synced={sleep.get('synced')}"
        ),
        data={"step": "garmin.sync", "status": "ok", "result": syncs},
    )

    # 6. Optional Strava
    strava_result = _maybe_import_strava(new_uid)
    summary["strava"] = strava_result
    if strava_result.get("status") in {"imported", "ok"}:
        emit(
            f"[green]Strava imported[/green] (display: {strava_result.get('display_name')})",
            data={"step": "strava.import", "status": "ok", "result": strava_result},
        )
    elif strava_result.get("status") == "skipped":
        emit(
            "[yellow]Strava skipped:[/yellow] bootstrap tokens missing "
            "(STRAVA_TEST_ACCESS_TOKEN / STRAVA_TEST_REFRESH_TOKEN).",
            data={"step": "strava.import", "status": "skipped", "result": strava_result},
        )
    else:
        emit(
            f"[yellow]Strava import failed (non-fatal):[/yellow] "
            f"{strava_result.get('error')}",
            data={"step": "strava.import", "status": "error", "result": strava_result},
        )

    # 7. Persist state
    set_active_user(new_uid, email=email)
    summary["state_file"] = "~/.athctl/state.json"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["status"] = "ok"

    # 8. Final summary (single emit in both modes)
    final_human = (
        f"\n[bold green]Test user ready.[/bold green]\n"
        f"  user_id : {new_uid}\n"
        f"  email   : {email}\n"
        f"  garmin  : {check.get('provider_user_id')} "
        f"(activities synced={act.get('synced')}, daily={daily.get('synced')}, sleep={sleep.get('synced')})\n"
        f"  strava  : {strava_result.get('status')}\n"
        f"  state   : ~/.athctl/state.json\n\n"
        f"Next: [cyan]athctl chat \"Hi, who am I?\"[/cyan]"
    )
    emit(final_human, data=summary)

    # In JSON mode the final dict already went out via emit(); avoid double-print.
    # In quiet mode emit() returned silently, but we still want a single JSON
    # payload on stdout for scripting use cases - the standard emit handles that.
    _ = as_json  # keep the import live for future commands
