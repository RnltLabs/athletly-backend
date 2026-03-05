"""CLI interface for AgenticSports using Rich."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape

from src.agent.coach import generate_plan, save_plan
from src.agent.proactive import check_proactive_triggers, format_proactive_message
from src.agent.trajectory import assess_trajectory
from src.config import get_settings
from src.memory.episodes import list_episodes as list_episodes_file
from src.memory.profile import create_profile, save_profile, load_profile
from src.memory.user_model import UserModel
from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import compute_metric
from src.tools.activity_store import store_activity as store_activity_file
from src.tools.activity_store import list_activities as list_activities_file
from src.tools.activity_store import import_new_activities

console = Console()

def _get_known_sports() -> list[str]:
    """Get sports the athlete has actually done (from activity store)."""
    try:
        activities = list_activities_file()
        sports = sorted({a.get("sport", "unknown") for a in activities})
        return sports if sports else ["running", "cycling", "swimming"]  # empty-state hint
    except Exception:
        return []


def _get_backends():
    """Return (user_model, list_activities_fn, list_episodes_fn, store_activity_fn) based on config.

    When Supabase is configured with a user_id, returns DB-backed implementations.
    Otherwise falls back to legacy file-based storage.
    """
    settings = get_settings()
    if settings.use_supabase:
        from src.db import UserModelDB
        from src.db import list_activities as db_list_activities
        from src.db import list_episodes as db_list_episodes
        uid = settings.agenticsports_user_id
        user_model = UserModelDB.load_or_create(uid)
        return (
            user_model,
            lambda **kw: db_list_activities(uid, **kw),
            lambda **kw: db_list_episodes(uid, **kw),
        )
    else:
        user_model = UserModel.load_or_create()
        return (
            user_model,
            lambda **kw: list_activities_file(),
            lambda **kw: list_episodes_file(**kw),
        )


def _format_targets(targets: dict) -> str:
    """Format sport-specific targets as readable text.

    Renders target keys with their appropriate units:
      Running: "5:30-6:00/km" | Cycling: "180-200W" | Swimming: "1:45-1:55/100m"
      General: "Zone 2" | "RPE 6-7"
    """
    if not targets:
        return ""

    parts = []
    if "pace_min_km" in targets:
        parts.append(f"{targets['pace_min_km']}/km")
    if "power_watts" in targets:
        parts.append(f"{targets['power_watts']}W")
    if "pace_min_100m" in targets:
        parts.append(f"{targets['pace_min_100m']}/100m")
    if "cadence_rpm" in targets:
        parts.append(f"{targets['cadence_rpm']}rpm")
    if "hr_zone" in targets:
        parts.append(targets["hr_zone"])
    if "rpe" in targets:
        parts.append(f"RPE {targets['rpe']}")

    return " | ".join(parts)


def _format_steps(steps: list[dict]) -> str:
    """Format workout steps as readable lines for CLI display.

    Regular steps: "Warmup 15:00 @ 6:00-6:30/km | Zone 2"
    Repeat groups: "6x:" header with indented sub-steps.
    """
    lines = []
    for step in steps:
        step_type = step.get("type", "?")
        duration = step.get("duration", "")
        targets = step.get("targets", {})

        if step_type == "repeat":
            count = step.get("repeat_count", 1)
            lines.append(f"{count}x:")
            for sub in step.get("steps", []):
                sub_type = sub.get("type", "?").title()
                sub_dur = sub.get("duration", "")
                sub_targets = _format_targets(sub.get("targets", {}))
                target_str = f" @ {sub_targets}" if sub_targets else ""
                lines.append(f"  {sub_type} {sub_dur}{target_str}")
        else:
            target_str = _format_targets(targets)
            at_str = f" @ {target_str}" if target_str else ""
            lines.append(f"{step_type.title()} {duration}{at_str}")

    return "\n".join(lines)


def onboard_athlete() -> dict:
    """Interactive CLI to collect athlete info and create a profile."""
    console.print(
        Panel("[bold]Welcome to AgenticSports[/bold] - Your Autonomous AI Coach", style="blue")
    )

    # Sports
    known = _get_known_sports()
    if known:
        console.print("\nSports you've trained: " + ", ".join(known))
    else:
        console.print("\nEnter any sport (e.g. running, cycling, swimming, gym)")
    sports_input = Prompt.ask(
        "What sport(s) do you train?",
        default="running",
    )
    sports = [s.strip().lower() for s in sports_input.split(",")]

    # Goal event
    event = Prompt.ask("What's your goal event?", default="Half Marathon")

    # Target date
    target_date = Prompt.ask("Target date (YYYY-MM-DD)?", default="2026-08-15")

    # Target time
    target_time = Prompt.ask("Target time (H:MM:SS)?", default="1:45:00")

    # Training days
    training_days = IntPrompt.ask(
        "How many days per week can you train?", default=5
    )

    # Max session duration
    max_duration = IntPrompt.ask(
        "Max session duration in minutes?", default=90
    )

    profile = create_profile(
        sports=sports,
        event=event,
        target_date=target_date,
        target_time=target_time,
        training_days_per_week=training_days,
        max_session_minutes=max_duration,
    )

    path = save_profile(profile)
    console.print(f"\n[green]Profile saved to {path}[/green]")
    return profile


def display_plan(plan: dict) -> None:
    """Display a training plan as a Rich table."""
    table = Table(title="Weekly Training Plan", show_lines=True)
    table.add_column("Day", style="bold", width=10)
    table.add_column("Sport", width=10)
    table.add_column("Type", style="cyan", width=16)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Description", width=40)
    table.add_column("Notes", width=25)

    for session in plan.get("sessions", []):
        if session.get("steps"):
            # New structured format: render steps in Description column
            duration = session.get(
                "total_duration_minutes",
                session.get("duration_minutes", "?"),
            )
            description = _format_steps(session["steps"])
            notes = session.get("notes", "")
        else:
            # Old flat format: keep existing behavior
            duration = session.get("duration_minutes", "?")
            description = session.get("description", "")
            notes = session.get("notes", "")

        table.add_row(
            session.get("day", ""),
            session.get("sport", ""),
            session.get("type", ""),
            f"{duration} min",
            description,
            notes,
        )

    console.print(table)

    summary = plan.get("weekly_summary", {})
    if summary:
        console.print(
            Panel(
                f"Total sessions: {summary.get('total_sessions', '?')} | "
                f"Total duration: {summary.get('total_duration_minutes', '?')} min | "
                f"Focus: {summary.get('focus', 'N/A')}",
                title="Weekly Summary",
                style="green",
            )
        )


def import_activity(file_path: str) -> None:
    """Import a FIT file (or JSON fixture), compute metrics, and store it."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    console.print(f"[yellow]Parsing {path.name}...[/yellow]")
    activity = parse_fit_file(str(path))

    sport = activity.get("sport", "unknown")
    duration_sec = activity.get("duration_seconds", 0)
    duration_min = duration_sec / 60 if duration_sec else 0
    hr = activity.get("heart_rate", {})
    avg_hr = hr.get("avg") if hr else None
    distance = activity.get("distance_meters")

    console.print(f"  Sport: [cyan]{sport}[/cyan]")
    console.print(f"  Duration: [cyan]{duration_min:.0f} min[/cyan]")
    if distance:
        console.print(f"  Distance: [cyan]{distance / 1000:.1f} km[/cyan]")

    # Compute TRIMP via DB-defined formula if user_id is available
    settings = get_settings()
    if avg_hr and settings.use_supabase and settings.agenticsports_user_id:
        uid = settings.agenticsports_user_id
        variables = {
            "duration_minutes": duration_min,
            "avg_hr": avg_hr,
        }
        trimp = compute_metric(uid, "trimp", variables)
        if trimp is not None:
            console.print(f"  Avg HR: [cyan]{avg_hr} bpm[/cyan]")
            console.print(f"  TRIMP: [cyan]{trimp}[/cyan]")
            activity["trimp"] = trimp
        else:
            console.print(f"  Avg HR: [cyan]{avg_hr} bpm[/cyan]")
            console.print("  [dim]TRIMP metric not defined in DB[/dim]")
    elif avg_hr:
        console.print(f"  Avg HR: [cyan]{avg_hr} bpm[/cyan]")
        console.print("  [dim]TRIMP requires Supabase with metric definitions[/dim]")

    stored_path = store_activity(activity)
    console.print(f"\n[green]Activity stored: {stored_path}[/green]")


def format_import_report(imported: list[dict]) -> str:
    """Format a human-readable summary of imported activities.

    Examples:
        "Found 3 new activities: Running 8.2km, Cycling 45.0km, Strength 60min"
        "Found 1 new activity: Swimming 1.5km"
        "No new activities found."
    """
    if not imported:
        return "No new activities found."

    summaries = []
    for act in imported:
        sport = (act.get("sport") or "unknown").title()
        distance = act.get("distance_meters")
        duration_sec = act.get("duration_seconds")

        if distance and distance > 0:
            summaries.append(f"{sport} {distance / 1000:.1f}km")
        elif duration_sec and duration_sec > 0:
            summaries.append(f"{sport} {round(duration_sec / 60)}min")
        else:
            summaries.append(sport)

    count = len(imported)
    noun = "activity" if count == 1 else "activities"
    return f"Found {count} new {noun}: {', '.join(summaries)}"


def run_import() -> list[dict]:
    """Run auto-import pipeline and display results.

    Returns the list of newly imported activities.
    """
    imported = import_new_activities()
    report = format_import_report(imported)

    if imported:
        console.print(Panel(report, title="Activity Import", style="green"))
    else:
        console.print(f"[dim]{report}[/dim]")

    return imported


def run_assessment() -> None:
    """Run a full assessment cycle (deprecated -- use chat mode instead).

    The V2 AgentCore pipeline has been removed. Assessment is now handled
    by the agent loop in chat mode, which can analyse activities, compare
    them against the current plan, and propose adjustments conversationally.
    """
    console.print(
        "[yellow]The standalone --assess command has been retired.[/yellow]\n"
        "Use [bold]--chat[/bold] instead and ask the agent to assess your recent training."
    )


def _load_latest_plan() -> dict | None:
    """Load the most recent training plan from data/plans/."""
    import json as _json
    plans_dir = Path(__file__).parent.parent.parent / "data" / "plans"
    plan_files = sorted(plans_dir.glob("plan_*.json")) if plans_dir.exists() else []
    if not plan_files:
        return None
    return _json.loads(plan_files[-1].read_text())


def run_trajectory() -> None:
    """Show full trajectory assessment."""
    run_import()

    try:
        profile = load_profile()
    except FileNotFoundError:
        console.print("[red]No athlete profile found. Run onboarding first.[/red]")
        return

    plan = _load_latest_plan()
    if not plan:
        console.print("[red]No training plan found.[/red]")
        return

    _, list_acts_fn, list_eps_fn = _get_backends()
    activities = list_acts_fn()
    episodes = list_eps_fn()

    console.print("[yellow]Assessing training trajectory...[/yellow]\n")

    try:
        traj = assess_trajectory(profile, activities, episodes, plan)
    except Exception as e:
        console.print(f"[red]Trajectory assessment failed: {e}[/red]")
        return

    # Display trajectory
    goal = traj.get("goal", {})
    console.print(Panel(
        f"Event: [cyan]{goal.get('event', '?')}[/cyan]\n"
        f"Target: [cyan]{goal.get('target_time', '?')}[/cyan] by {goal.get('target_date', '?')}\n"
        f"Weeks remaining: [cyan]{goal.get('weeks_remaining', '?')}[/cyan]",
        title="Goal",
        style="blue",
    ))

    trajectory = traj.get("trajectory", {})
    on_track = trajectory.get("on_track", "unknown")
    status_color = "green" if on_track else "red"
    console.print(Panel(
        f"On track: [{status_color}]{on_track}[/{status_color}]\n"
        f"Predicted time: [cyan]{trajectory.get('predicted_race_time', '?')}[/cyan]\n"
        f"Confidence: [cyan]{traj.get('confidence', '?')}[/cyan]\n"
        f"{traj.get('confidence_explanation', '')}",
        title="Trajectory",
        style=status_color,
    ))

    recommendations = traj.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            console.print(f"  - {rec}")

    risks = traj.get("risks", [])
    if risks:
        console.print("\n[bold]Risks:[/bold]")
        for risk in risks:
            console.print(f"  [{risk.get('probability', '?')}] {risk.get('risk', '?')}")
            console.print(f"    Mitigation: {risk.get('mitigation', '?')}")


def run_status() -> None:
    """Quick status check with proactive messages."""
    run_import()

    try:
        profile = load_profile()
    except FileNotFoundError:
        console.print("[red]No athlete profile found.[/red]")
        return

    plan = _load_latest_plan()
    _, list_acts_fn, list_eps_fn = _get_backends()
    activities = list_acts_fn()
    episodes = list_eps_fn()

    console.print(Panel(
        f"Athlete: [cyan]{profile.get('name', 'Unknown')}[/cyan]\n"
        f"Goal: [cyan]{profile.get('goal', {}).get('event', '?')}[/cyan] "
        f"by {profile.get('goal', {}).get('target_date', '?')}\n"
        f"Activities: [cyan]{len(activities)}[/cyan] | "
        f"Episodes: [cyan]{len(episodes)}[/cyan]",
        title="AgenticSports Status",
        style="blue",
    ))

    if plan and activities and episodes:
        try:
            traj = assess_trajectory(profile, activities, episodes, plan)
            triggers = check_proactive_triggers(profile, activities, episodes, traj)

            if triggers:
                console.print("\n[bold]Messages for you:[/bold]")
                for trigger in triggers:
                    msg = format_proactive_message(trigger, profile)
                    priority = trigger.get("priority", "low")
                    color = {"high": "red", "medium": "yellow", "low": "green"}.get(priority, "white")
                    console.print(f"  [{color}]{msg}[/{color}]")
        except Exception as e:
            console.print(f"[dim]Could not generate trajectory: {e}[/dim]")


def run_chat() -> None:
    """Interactive chat mode with conversational coaching."""
    _run_chat()


def _run_chat() -> None:
    """Agent loop for interactive fitness coaching.

    The agent decides what to do via tools.
    Includes: startup optimization, plan display, import awareness,
    and proactive session-start analysis.
    """
    import json as _json
    from src.agent.agent_loop import AgentLoop
    from src.agent.startup_context import build_startup_context

    imported = run_import()
    user_model, _, _ = _get_backends()

    # Pre-compute startup context (Gap 5 -- instant greeting)
    startup_ctx = build_startup_context(user_model, imported=imported)

    # Progress callback for tool visibility
    def on_progress(event_type: str, detail: str):
        if event_type == "tool_call":
            console.print(f"  [dim]-> {detail}[/dim]")
        elif event_type == "tool_result":
            console.print(f"  [dim]   {detail[:120]}[/dim]")
        elif event_type == "tool_error":
            console.print(f"  [yellow]   ! {detail}[/yellow]")

    agent = AgentLoop(
        user_model=user_model,
        on_progress=on_progress,
        startup_context=startup_ctx,
    )
    agent.start_session()

    # Startup greeting
    is_new = not user_model.structured_core.get("sports")
    if is_new:
        greeting = (
            "Welcome to AgenticSports! I'm your autonomous AI coach -- "
            "I work with athletes across all sports, from running and cycling to basketball, "
            "swimming, CrossFit, and beyond.\n\n"
            "Tell me about yourself -- what's your name, what sport(s) do you do, "
            "what are you training for, and how does your typical training week look?"
        )
    else:
        # Proactive session-start analysis (Gap 8)
        startup_result = agent.process_message(
            "[SYSTEM] New session started. Greet the athlete by name using the "
            "pre-loaded session context. If there are new imports or recent "
            "activities, briefly mention notable observations (volume changes, "
            "new PRs, missed sessions). Keep it concise and warm. "
            "Check for any notable changes worth mentioning."
        )
        greeting = startup_result.response_text

    console.print(Panel(escape(greeting), title="AgenticSports Coach", style="blue"))
    agent.inject_context("model", greeting)

    # Main loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold]You[/bold]")
        except (KeyboardInterrupt, EOFError):
            user_input = "exit"

        if user_input.lower().strip() in ("exit", "quit", "q"):
            user_model.save()
            console.print("[dim]Session ended. See you next time![/dim]")
            break

        if user_input.lower().strip() == "plan":
            user_input = "Please create a new training plan for me."

        # Process through agent loop
        console.print()
        result = agent.process_message(user_input)

        # Display response
        console.print(Panel(escape(result.response_text), title="AgenticSports Coach", style="blue"))

        # Plan display integration (Gap 6)
        for turn in result.turns:
            if turn.tool_name == "save_plan" and turn.content:
                try:
                    save_result = _json.loads(turn.content)
                    if save_result.get("saved"):
                        plan_path = save_result.get("path")
                        if plan_path:
                            plan_data = _json.loads(Path(plan_path).read_text())
                            display_plan(plan_data)
                except (_json.JSONDecodeError, OSError, KeyError):
                    pass

        # Show tool usage stats
        if result.tool_calls_made > 0:
            console.print(
                f"  [dim]({result.tool_calls_made} tool calls, "
                f"{result.total_duration_ms}ms)[/dim]"
            )

        # If onboarding just completed, offer to create a plan (Gap 4b)
        if result.onboarding_just_completed:
            console.print(
                "\n[yellow]Profile complete! I can create your first training plan now.[/yellow]"
            )

    user_model.save()


def main(args: list[str] | None = None):
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        prog="agenticsports",
        description="AgenticSports - Autonomous AI Sports Coach",
    )
    parser.add_argument(
        "--import", dest="import_file", metavar="FILE",
        help="Import a single FIT file or JSON activity fixture (for bulk import, auto-import runs on startup)",
    )
    parser.add_argument(
        "--assess", action="store_true",
        help="Run assessment on latest activities vs current plan",
    )
    parser.add_argument(
        "--trajectory", action="store_true",
        help="Show full trajectory assessment",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Quick status check with proactive messages",
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Enter interactive chat mode (default when no flags given)",
    )
    parser.add_argument(
        "--onboard-legacy", action="store_true",
        help="Use legacy form-based onboarding (deprecated)",
    )

    parsed = parser.parse_args(args)

    if parsed.import_file:
        import_activity(parsed.import_file)
        return

    if parsed.assess:
        run_assessment()
        return

    if parsed.trajectory:
        run_trajectory()
        return

    if parsed.status:
        run_status()
        return

    if parsed.onboard_legacy:
        # Legacy form-based onboarding
        profile = onboard_athlete()
        console.print("\n[yellow]Generating your training plan...[/yellow]\n")
        _, list_acts_fn, list_eps_fn = _get_backends()
        activities = list_acts_fn()
        from src.memory.episodes import retrieve_relevant_episodes
        _episodes = list_eps_fn(limit=10)
        _relevant_eps = retrieve_relevant_episodes(
            {"goal": profile.get("goal", {}), "sports": profile.get("sports", [])},
            _episodes,
            max_results=5,
        )
        try:
            settings = get_settings()
            uid = settings.agenticsports_user_id if settings.use_supabase else ""
            plan = generate_plan(profile, activities=activities, relevant_episodes=_relevant_eps, user_id=uid)
        except ValueError as e:
            console.print(f"[red]Error generating plan: {e}[/red]")
            return
        except Exception as e:
            console.print(f"[red]Failed to connect to Gemini: {e}[/red]")
            return
        path = save_plan(plan)
        console.print(f"[green]Plan saved to {path}[/green]\n")
        display_plan(plan)
        return

    # Default: chat mode (same as --chat)
    run_chat()


if __name__ == "__main__":
    main()
