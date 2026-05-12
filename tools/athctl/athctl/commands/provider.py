"""Provider commands for athctl.

Exposes the registered provider abstraction (Garmin, Strava, ...) on the
CLI so the test harness can:

- list providers and their capabilities
- check connection status for the active user
- run an authentication flow
- sync activities / daily metrics / sleep
- disconnect (delete tokens)

The commands here delegate to the provider singletons in
``src.services.providers.registry``. They never reach into the database
directly; everything goes through the provider interface so behavior
matches what the backend agent sees.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from athctl.common import (
    EXIT_AUTH,
    EXIT_NETWORK,
    EXIT_STATE,
    as_json,
    console,
    emit,
)
from athctl.state import get_active_user_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# When --full is used, walk back in chunks of this many days so a single
# request does not try to fetch years of data at once.
FULL_SYNC_CHUNK_DAYS = 90

# Hard floor for --full backfills. Beyond this providers usually have
# nothing useful and the request just wastes API quota.
FULL_SYNC_FLOOR_DATE = date(2020, 1, 1)

# Sleep between chunks during --full to be gentle on provider rate limits.
FULL_SYNC_CHUNK_PAUSE_SECONDS = 2

# Default days for plain `provider sync <name>` with no flags.
DEFAULT_SYNC_DAYS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_provider_or_exit(name: str):
    """Return the provider singleton for *name* or exit with a helpful list."""
    from src.services.providers.registry import get_provider, list_providers

    try:
        return get_provider(name)
    except ValueError:
        known = ", ".join(sorted(p.name for p in list_providers()))
        emit(
            f"[red]Unknown provider {name!r}.[/red] Known: {known}",
            data={"status": "error", "error": f"unknown provider {name!r}", "known": known},
        )
        sys.exit(EXIT_STATE)


def _capabilities_to_dict(caps: Any) -> dict[str, Any]:
    """Render a ProviderCapabilities dataclass as a plain dict."""
    return {
        "activities": bool(getattr(caps, "activities", False)),
        "daily_metrics": bool(getattr(caps, "daily_metrics", False)),
        "sleep": bool(getattr(caps, "sleep", False)),
        "body_battery": bool(getattr(caps, "body_battery", False)),
        "vo2max": bool(getattr(caps, "vo2max", False)),
        "auth_flow": str(getattr(caps, "auth_flow", "unknown")),
    }


def _yesno(flag: bool) -> str:
    """Compact yes/no glyph for human-readable tables."""
    return "yes" if flag else "no"


def _resolve_days(days: int | None, since: str | None) -> int:
    """Resolve --days / --since into a single ``days`` integer.

    --since takes precedence over --days. The result is clamped to >= 1
    so providers always do at least one day of work.
    """
    if since:
        try:
            since_date = datetime.strptime(since, "%Y-%m-%d").date()
        except ValueError as exc:
            emit(
                f"[red]--since must be YYYY-MM-DD:[/red] {exc}",
                data={"status": "error", "error": f"invalid --since: {exc}"},
            )
            sys.exit(EXIT_STATE)
        delta_days = (datetime.now(timezone.utc).date() - since_date).days
        return max(1, delta_days)
    if days is not None:
        return max(1, int(days))
    return DEFAULT_SYNC_DAYS


def _full_sync_chunks(today: date) -> list[int]:
    """Return a list of `days` values for back-to-back full-backfill chunks.

    The first chunk is ``[today-90 .. today]``, the next
    ``[today-180 .. today-90]`` (still expressed as a `days=180` call
    against the provider since providers count back from today), and so
    on until we hit ``FULL_SYNC_FLOOR_DATE``.
    """
    chunks: list[int] = []
    cursor = today
    while cursor > FULL_SYNC_FLOOR_DATE:
        days_back = (today - cursor).days + FULL_SYNC_CHUNK_DAYS
        floor_days = (today - FULL_SYNC_FLOOR_DATE).days
        days_back = min(days_back, floor_days)
        chunks.append(days_back)
        cursor = cursor - timedelta(days=FULL_SYNC_CHUNK_DAYS)
    return chunks


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group(
    "provider",
    help="Manage external data providers (Garmin, Strava, ...).",
)
def provider() -> None:
    """Provider operations."""


# ---------------------------------------------------------------------------
# provider list
# ---------------------------------------------------------------------------


@provider.command("list", help="List registered providers and their capabilities.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Emit JSON.")
def provider_list(json_flag: bool) -> None:
    """List all providers with capabilities and (if user set) status."""
    from src.services.providers.registry import list_providers

    uid = get_active_user_id()
    rows: list[dict[str, Any]] = []
    for prov in list_providers():
        caps = _capabilities_to_dict(prov.capabilities)
        status_info: dict[str, Any] | None = None
        if uid:
            try:
                status_info = prov.check_connection(uid)
            except Exception as exc:
                logger.warning("check_connection(%s) raised: %s", prov.name, exc)
                status_info = {
                    "connected": False,
                    "status": "error",
                    "error": str(exc),
                }
        rows.append(
            {
                "name": prov.name,
                "capabilities": caps,
                "status": status_info,
            }
        )

    if json_flag:
        print(as_json({"active_user_id": uid, "providers": rows}))
        return

    table = Table(title="Providers", show_lines=False)
    table.add_column("Name", style="bold cyan")
    table.add_column("Activities")
    table.add_column("Daily")
    table.add_column("Sleep")
    table.add_column("Body Batt.")
    table.add_column("Auth flow")
    if uid:
        table.add_column("Status")

    for row in rows:
        caps = row["capabilities"]
        cells = [
            row["name"],
            _yesno(caps["activities"]),
            _yesno(caps["daily_metrics"]),
            _yesno(caps["sleep"]),
            _yesno(caps["body_battery"]),
            caps["auth_flow"],
        ]
        if uid:
            status = row["status"] or {}
            label = status.get("status", "?")
            connected = status.get("connected")
            if connected:
                cells.append(f"[green]{label}[/green]")
            elif label == "disconnected":
                cells.append(f"[yellow]{label}[/yellow]")
            else:
                cells.append(f"[red]{label}[/red]")
        table.add_row(*cells)

    console.print(table)
    if not uid:
        console.print("[yellow]No active user. Run `athctl init` first to see connection status.[/yellow]")


# ---------------------------------------------------------------------------
# provider connect
# ---------------------------------------------------------------------------


@provider.command("connect", help="Run the authentication flow for a provider.")
@click.argument("name", type=str)
@click.option(
    "--email",
    type=str,
    default=None,
    help="Garmin only: account email (prompted if omitted).",
)
@click.option(
    "--password",
    type=str,
    default=None,
    help="Garmin only: account password (prompted if omitted, hidden input).",
)
@click.option(
    "--redirect-uri",
    type=str,
    default="http://localhost",
    help="Strava only: OAuth redirect URI to use in the authorize URL.",
)
@click.option(
    "--code",
    type=str,
    default=None,
    help="Strava only: pre-supplied auth code (skips the interactive prompt).",
)
def provider_connect(
    name: str,
    email: str | None,
    password: str | None,
    redirect_uri: str,
    code: str | None,
) -> None:
    """Authenticate the active user with *name*."""
    uid = get_active_user_id()
    if not uid:
        emit(
            "[red]No active user. Run `athctl init` first.[/red]",
            data={"status": "error", "error": "no active user"},
        )
        sys.exit(EXIT_STATE)

    prov = _get_provider_or_exit(name)
    name_lower = name.lower()

    if name_lower == "garmin":
        from src.services.providers.garmin import GarminProvider

        if not isinstance(prov, GarminProvider):
            emit(
                "[red]Registry returned non-Garmin provider for 'garmin'.[/red]",
                data={"status": "error", "error": "registry mismatch"},
            )
            sys.exit(EXIT_STATE)
        if not email:
            email = click.prompt("Garmin email", type=str)
        if not password:
            password = click.prompt("Garmin password", type=str, hide_input=True)
        result = prov.authenticate(uid, email, password)
        if result.get("status") != "connected":
            emit(
                f"[red]Garmin auth failed:[/red] {result.get('error')}",
                data={"status": "error", "step": "garmin.authenticate", "result": result},
            )
            sys.exit(EXIT_AUTH)
        emit(
            f"[green]Garmin connected[/green] (display: {result.get('display_name')})",
            data={"status": "ok", "step": "garmin.authenticate", "result": result},
        )
        return

    if name_lower == "strava":
        from src.services.providers.strava import StravaProvider

        if not isinstance(prov, StravaProvider):
            emit(
                "[red]Registry returned non-Strava provider for 'strava'.[/red]",
                data={"status": "error", "error": "registry mismatch"},
            )
            sys.exit(EXIT_STATE)
        auth_url = prov.authorization_url(redirect_uri=redirect_uri)
        emit(
            (
                "[bold]Open this URL in a browser, approve access,[/bold] then paste "
                "the value of the `code=` query parameter from the redirected URL:\n"
                f"  {auth_url}\n"
            ),
            data={"status": "prompt", "step": "strava.authorize_url", "url": auth_url},
        )
        if not code:
            code = click.prompt("code", type=str)
        result = prov.exchange_code(uid, code)
        if result.get("status") != "connected":
            emit(
                f"[red]Strava auth failed:[/red] {result.get('error')}",
                data={"status": "error", "step": "strava.exchange_code", "result": result},
            )
            sys.exit(EXIT_AUTH)
        emit(
            f"[green]Strava connected[/green] (display: {result.get('display_name')})",
            data={"status": "ok", "step": "strava.exchange_code", "result": result},
        )
        return

    emit(
        f"[red]Provider {name!r} has no automated `connect` flow.[/red]",
        data={"status": "error", "error": f"no connect flow for {name!r}"},
    )
    sys.exit(EXIT_STATE)


# ---------------------------------------------------------------------------
# provider sync
# ---------------------------------------------------------------------------


def _run_one_sync_pass(prov, uid: str, days: int) -> dict[str, Any]:
    """Run sync_activities + capability-gated daily/sleep for *days*.

    Returns a structured dict per step so callers can sum across chunks.
    """
    out: dict[str, Any] = {"days": days}
    caps = prov.capabilities

    out["activities"] = prov.sync_activities(uid, days=days)
    if getattr(caps, "daily_metrics", False):
        out["daily"] = prov.sync_daily_stats(uid, days=days)
    else:
        out["daily"] = {"status": "skipped", "reason": "capability disabled"}
    if getattr(caps, "sleep", False):
        out["sleep"] = prov.sync_sleep(uid, days=days)
    else:
        out["sleep"] = {"status": "skipped", "reason": "capability disabled"}

    return out


def _print_sync_summary(name: str, label: str, result: dict[str, Any]) -> None:
    """Pretty-print one (activities|daily|sleep) sync result."""
    status = result.get("status", "?")
    synced = result.get("synced")
    skipped = result.get("skipped")
    reason = result.get("reason") or result.get("error")
    if status == "ok":
        synced_str = synced if synced is not None else "?"
        skipped_str = f", skipped={skipped}" if skipped is not None else ""
        emit(
            f"  [green]{name} {label}:[/green] synced={synced_str}{skipped_str}",
            data=None,
        )
    elif status == "skipped":
        emit(f"  [dim]{name} {label}: skipped ({reason})[/dim]", data=None)
    else:
        emit(f"  [red]{name} {label}: {status} ({reason})[/red]", data=None)


@provider.command("sync", help="Sync data for a provider (activities + capability-gated).")
@click.argument("name", type=str)
@click.option("--days", type=int, default=None, help="How many days back to sync.")
@click.option("--since", type=str, default=None, help="Sync since YYYY-MM-DD (overrides --days).")
@click.option("--full", "full_sync", is_flag=True, default=False, help="Backfill in 90-day chunks back to 2020-01-01.")
@click.option("--json", "json_flag", is_flag=True, default=False, help="Emit JSON result instead of human output.")
def provider_sync(
    name: str,
    days: int | None,
    since: str | None,
    full_sync: bool,
    json_flag: bool,
) -> None:
    """Sync activities/daily/sleep for *name* against the active user."""
    uid = get_active_user_id()
    if not uid:
        emit(
            "[red]No active user. Run `athctl init` first.[/red]",
            data={"status": "error", "error": "no active user"},
        )
        sys.exit(EXIT_STATE)

    prov = _get_provider_or_exit(name)

    if full_sync and (days is not None or since is not None):
        emit(
            "[yellow]--full overrides --days/--since.[/yellow]",
            data={"status": "warning", "message": "--full overrides --days/--since"},
        )

    aggregate: dict[str, Any] = {
        "provider": prov.name,
        "user_id": uid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [],
    }

    if full_sync:
        today = datetime.now(timezone.utc).date()
        chunks = _full_sync_chunks(today)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"Full backfill {prov.name} ({len(chunks)} chunks)...", total=len(chunks)
            )
            for idx, days_back in enumerate(chunks):
                progress.update(
                    task,
                    description=f"{prov.name} chunk {idx + 1}/{len(chunks)} (days={days_back})",
                )
                chunk_result = _run_one_sync_pass(prov, uid, days=days_back)
                aggregate["chunks"].append(chunk_result)
                progress.advance(task)
                if idx < len(chunks) - 1:
                    time.sleep(FULL_SYNC_CHUNK_PAUSE_SECONDS)
        aggregate["finished_at"] = datetime.now(timezone.utc).isoformat()

        if json_flag:
            print(as_json(aggregate))
            return

        total_act = sum(
            int((c.get("activities") or {}).get("synced") or 0) for c in aggregate["chunks"]
        )
        total_daily = sum(
            int((c.get("daily") or {}).get("synced") or 0) for c in aggregate["chunks"]
        )
        total_sleep = sum(
            int((c.get("sleep") or {}).get("synced") or 0) for c in aggregate["chunks"]
        )
        emit(
            (
                f"[bold green]{prov.name} full backfill done.[/bold green] "
                f"chunks={len(aggregate['chunks'])}, "
                f"activities={total_act}, daily={total_daily}, sleep={total_sleep}"
            ),
            data=None,
        )
        return

    resolved_days = _resolve_days(days, since)
    aggregate["days"] = resolved_days
    result = _run_one_sync_pass(prov, uid, days=resolved_days)
    aggregate["chunks"].append(result)
    aggregate["finished_at"] = datetime.now(timezone.utc).isoformat()

    if json_flag:
        print(as_json(aggregate))
        return

    emit(
        f"[bold]{prov.name} sync (days={resolved_days})[/bold]",
        data=None,
    )
    _print_sync_summary(prov.name, "activities", result.get("activities") or {})
    _print_sync_summary(prov.name, "daily", result.get("daily") or {})
    _print_sync_summary(prov.name, "sleep", result.get("sleep") or {})


# ---------------------------------------------------------------------------
# provider status
# ---------------------------------------------------------------------------


@provider.command("status", help="Show connection status for a provider.")
@click.argument("name", type=str)
@click.option("--json", "json_flag", is_flag=True, default=False, help="Emit JSON.")
def provider_status(name: str, json_flag: bool) -> None:
    """Check connection for *name* against the active user."""
    uid = get_active_user_id()
    if not uid:
        emit(
            "[red]No active user. Run `athctl init` first.[/red]",
            data={"status": "error", "error": "no active user"},
        )
        sys.exit(EXIT_STATE)

    prov = _get_provider_or_exit(name)
    try:
        result = prov.check_connection(uid)
    except Exception as exc:
        emit(
            f"[red]check_connection raised:[/red] {exc}",
            data={"status": "error", "error": str(exc)},
        )
        sys.exit(EXIT_NETWORK)

    if json_flag:
        print(as_json({"provider": prov.name, "user_id": uid, "result": result}))
        return

    connected = result.get("connected")
    label = result.get("status", "?")
    if connected:
        color = "green"
    elif label == "disconnected":
        color = "yellow"
    else:
        color = "red"
    emit(
        f"[{color}]{prov.name}: {label}[/{color}] (provider_user={result.get('provider_user_id')}, "
        f"last_sync={result.get('last_sync_at')})",
        data=None,
    )


# ---------------------------------------------------------------------------
# provider disconnect
# ---------------------------------------------------------------------------


@provider.command("disconnect", help="Delete stored tokens for a provider.")
@click.argument("name", type=str)
@click.option("--yes", "skip_confirm", is_flag=True, default=False, help="Skip confirmation prompt.")
def provider_disconnect(name: str, skip_confirm: bool) -> None:
    """Disconnect *name* for the active user."""
    uid = get_active_user_id()
    if not uid:
        emit(
            "[red]No active user. Run `athctl init` first.[/red]",
            data={"status": "error", "error": "no active user"},
        )
        sys.exit(EXIT_STATE)

    prov = _get_provider_or_exit(name)

    if not skip_confirm:
        confirmed = click.confirm(
            f"Disconnect {prov.name} for user {uid}? (deletes tokens and synced data)",
            default=False,
        )
        if not confirmed:
            emit("Aborted.", data={"status": "aborted"})
            return

    try:
        result = prov.disconnect(uid)
    except Exception as exc:
        emit(
            f"[red]disconnect raised:[/red] {exc}",
            data={"status": "error", "error": str(exc)},
        )
        sys.exit(EXIT_NETWORK)

    emit(
        f"[green]{prov.name} disconnected.[/green] result={result}",
        data={"status": "ok", "provider": prov.name, "result": result},
    )
