"""Top-level Click group for the ``athctl`` CLI.

Loads the backend ``.env`` (so SUPABASE_*, ANTHROPIC_API_KEY, STRAVA_*
are available), sets global flags via env vars, and wires up every
real subcommand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from athctl import __version__
from athctl.commands.chat import chat
from athctl.commands.init import init as init_command
from athctl.commands.provider import provider
from athctl.commands.reset import reset
from athctl.commands.snapshot import snapshot
from athctl.commands.state import state
from athctl.commands.trace import trace
from athctl.commands.trigger import trigger
from athctl.commands.validate import validate
from athctl.common import refresh_console


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def _load_backend_env() -> None:
    """Load the backend .env into the process environment."""
    explicit = os.environ.get("ATHLETLY_BACKEND_ROOT")
    if explicit:
        candidate = Path(explicit) / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            return
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            env = parent / ".env"
            if env.is_file():
                load_dotenv(env)
            return


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "athctl - test harness for the Athletly backend.\n\n"
        "Bootstraps a throwaway test user that reuses Roman's Garmin tokens, "
        "drives chat sessions, snapshots Supabase state, and validates "
        "agent behavior end-to-end."
    ),
)
@click.version_option(__version__, prog_name="athctl")
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON instead of human-friendly output.",
)
@click.option(
    "--quiet",
    "quiet_mode",
    is_flag=True,
    default=False,
    help="Suppress progress output (errors still go to stderr).",
)
def main(json_mode: bool, quiet_mode: bool) -> None:
    """Athletly test-harness CLI."""
    _load_backend_env()
    if json_mode:
        os.environ["ATHCTL_JSON"] = "1"
    if quiet_mode:
        os.environ["ATHCTL_QUIET"] = "1"
    refresh_console()


# ---------------------------------------------------------------------------
# Provider shortcuts (athctl garmin <cmd> -> athctl provider <name> <cmd>)
# ---------------------------------------------------------------------------


def _make_shortcut_group(provider_name: str) -> click.Group:
    """Build a shortcut group that forwards to `provider <name> <cmd>`."""

    @click.group(
        name=provider_name,
        help=f"{provider_name.capitalize()} shortcuts (alias for `provider {provider_name} ...`).",
    )
    def shortcut() -> None:
        pass

    @shortcut.command("status")
    @click.pass_context
    def _status(ctx: click.Context) -> None:
        ctx.invoke(provider.get_command(ctx, "status"), name=provider_name)

    @shortcut.command("connect")
    @click.pass_context
    def _connect(ctx: click.Context) -> None:
        ctx.invoke(provider.get_command(ctx, "connect"), name=provider_name)

    @shortcut.command("disconnect")
    @click.option("--yes", is_flag=True)
    @click.pass_context
    def _disconnect(ctx: click.Context, yes: bool) -> None:
        ctx.invoke(provider.get_command(ctx, "disconnect"), name=provider_name, yes=yes)

    @shortcut.command("sync")
    @click.option("--days", type=int, default=None)
    @click.option("--since", default=None)
    @click.option("--full", is_flag=True)
    @click.pass_context
    def _sync(ctx: click.Context, days: int | None, since: str | None, full: bool) -> None:
        ctx.invoke(
            provider.get_command(ctx, "sync"),
            name=provider_name,
            days=days,
            since=since,
            full=full,
        )

    return shortcut


garmin_shortcut = _make_shortcut_group("garmin")
strava_shortcut = _make_shortcut_group("strava")


# ---------------------------------------------------------------------------
# Wire everything up
# ---------------------------------------------------------------------------

main.add_command(init_command)
main.add_command(provider)
main.add_command(garmin_shortcut)
main.add_command(strava_shortcut)
main.add_command(state)
main.add_command(chat)
main.add_command(reset)
main.add_command(trigger)
main.add_command(snapshot)
main.add_command(trace)
main.add_command(validate)


if __name__ == "__main__":
    main()
