"""Trigger commands for athctl.

Manually fires background services (heartbeat, reflection, plan generation)
against the active test user so we can observe the resulting state changes
without waiting for the natural cadence.
"""

from __future__ import annotations

import logging
import sys

import click

from athctl.common import EXIT_BACKEND_DOWN, console, emit, require_active_user

logger = logging.getLogger(__name__)


@click.group(help="Trigger backend events (heartbeat, reflection, plan generation).")
def trigger() -> None:
    """Trigger backend events."""


@trigger.command("heartbeat", help="Run one heartbeat tick for the active user.")
def heartbeat() -> None:
    """Run heartbeat services for the active user only.

    The production HeartbeatService iterates all active users; here we
    invoke the per-user sub-checks directly so the test loop stays
    isolated.
    """
    uid = require_active_user()
    results: dict[str, str] = {}

    try:
        from src.agent.proactive import check_proactive_triggers
        proactive_result = check_proactive_triggers(uid)
        results["proactive"] = str(proactive_result)
    except Exception as exc:
        results["proactive_error"] = str(exc)
        logger.warning("Proactive check failed", exc_info=True)

    try:
        from src.agent.reflection import check_and_generate_reflections
        reflection_result = check_and_generate_reflections(uid)
        results["reflection"] = str(reflection_result)
    except Exception as exc:
        results["reflection_error"] = str(exc)
        logger.warning("Reflection check failed", exc_info=True)

    emit(
        f"Heartbeat tick complete for {uid}",
        data={"user_id": uid, **results},
    )
    for k, v in results.items():
        console.print(f"  {k}: {v}")


@trigger.command("reflection", help="Force a reflection pass for the active user.")
@click.option(
    "--force/--no-force",
    default=True,
    help="Bypass the natural reflection cadence checks.",
)
def reflection(force: bool) -> None:
    """Generate (or attempt to generate) an episodic reflection now."""
    uid = require_active_user()
    try:
        from src.agent.reflection import check_and_generate_reflections
    except Exception as exc:
        print(f"Could not import reflection module: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    try:
        result = check_and_generate_reflections(uid, force=force) if "force" in check_and_generate_reflections.__code__.co_varnames else check_and_generate_reflections(uid)
    except Exception as exc:
        print(f"Reflection failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    emit(
        f"Reflection result: {result}",
        data={"user_id": uid, "result": result},
    )


@trigger.command(
    "plan-generate",
    help="Trigger plan generation by sending a synthetic message to the agent.",
)
@click.option(
    "--prompt",
    default="Please generate or update my training plan based on my current goal and recent activities.",
    help="Override the synthetic prompt used to request plan generation.",
)
def plan_generate(prompt: str) -> None:
    """Drive plan generation through the agent loop.

    Plans are produced inside the agent (LLM + planning tools), so we
    cannot bypass it cleanly without duplicating logic. Instead we send
    a one-shot message that should cause the agent to call its planning
    tools and persist a new plan.
    """
    uid = require_active_user()

    try:
        from src.agent.agent_loop import AgentLoop
        from src.db.user_model_db import UserModelDB
    except Exception as exc:
        print(f"Could not import agent stack: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    try:
        user_model = UserModelDB.load_or_create(uid)
    except Exception as exc:
        print(f"Could not load user model: {exc}", file=sys.stderr)
        sys.exit(EXIT_BACKEND_DOWN)

    loop = AgentLoop(user_model=user_model, context="coach", trace_to_file=True)
    session_id = loop.start_session()
    emit(f"Started session {session_id}, sending plan-generation prompt...")
    result = loop.process_message(prompt)
    emit(
        f"Agent response:\n{result.response_text}",
        data={
            "session_id": session_id,
            "response_text": result.response_text,
            "tool_calls": result.tool_calls_made,
            "duration_ms": result.total_duration_ms,
        },
    )
