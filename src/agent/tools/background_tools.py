"""Background task tools (Claude Code run_in_background pattern).

The agent enqueues long ops (subagent spawn, macrocycle, weekly plan)
and gets back a job_id immediately. A worker thread runs the actual
handler via a fresh tool registry (thread-local state, fresh Supabase
client). The agent polls via check_background_job / list_my_jobs.

Kinds: 'subagent' | 'macrocycle' | 'plan' | 'test'.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings
from src.services.background_jobs import (
    enqueue_job,
    get_job,
    list_pending_jobs,
)

logger = logging.getLogger(__name__)


SUPPORTED_KINDS = ("subagent", "macrocycle", "plan", "test")


ESTIMATED_SECONDS = {
    "subagent": 25,
    "macrocycle": 55,
    "plan": 22,
    "test": 2,
}


def _resolve_user_id(user_model) -> str:
    """Return the user_id from user_model or fall back to settings."""
    uid = getattr(user_model, "user_id", None)
    if uid:
        return uid
    return get_settings().agenticsports_user_id


# ---------------------------------------------------------------------------
# Runners. Each takes (job_id, args) and returns a JSON-serialisable dict.
# Runners must NOT close over the main-thread user_model directly because
# they execute on a worker thread. They reload state via UserModelDB.
# ---------------------------------------------------------------------------


def _load_user_model(user_id: str):
    """Reload the user model inside a worker thread."""
    from src.db.user_model_db import UserModelDB
    return UserModelDB.load_or_create(user_id)


def _subagent_runner(job_id: str, args: dict) -> dict:
    """Worker: invoke spawn_subagent via a fresh registry."""
    user_id = args.get("user_id")
    if not user_id:
        return {"error": "missing user_id in args"}
    user_model = _load_user_model(user_id)
    registry = ToolRegistry()
    from src.agent.tools.meta_tools import register_meta_tools
    register_meta_tools(registry, user_model)
    return registry.execute("spawn_subagent", {
        "task": args.get("task", ""),
        "tools_scope": args.get("tools_scope", "research"),
        "max_rounds": args.get("max_rounds"),
        "context": args.get("context"),
    })


def _macrocycle_runner(job_id: str, args: dict) -> dict:
    """Worker: invoke create_macrocycle_plan via a fresh registry."""
    user_id = args.get("user_id")
    if not user_id:
        return {"error": "missing user_id in args"}
    user_model = _load_user_model(user_id)
    registry = ToolRegistry()
    from src.agent.tools.macrocycle_tools import register_macrocycle_tools
    register_macrocycle_tools(registry, user_model)
    call_args = {k: v for k, v in args.items() if k not in ("user_id",) and v is not None}
    return registry.execute("create_macrocycle_plan", call_args)


def _plan_runner(job_id: str, args: dict) -> dict:
    """Worker: invoke create_training_plan via a fresh registry."""
    user_id = args.get("user_id")
    if not user_id:
        return {"error": "missing user_id in args"}
    user_model = _load_user_model(user_id)
    registry = ToolRegistry()
    from src.agent.tools.planning_tools import register_planning_tools
    register_planning_tools(registry, user_model)
    call_args = {k: v for k, v in args.items() if k not in ("user_id",) and v is not None}
    return registry.execute("create_training_plan", call_args)


def _test_runner(job_id: str, args: dict) -> dict:
    """Worker: trivial sleep-based test runner for the E2E check."""
    import time
    seconds = float(args.get("seconds", 1))
    time.sleep(max(0.0, min(seconds, 5.0)))
    return {"ok": True, "slept": seconds, "echo": args.get("echo")}


_RUNNERS = {
    "subagent": _subagent_runner,
    "macrocycle": _macrocycle_runner,
    "plan": _plan_runner,
    "test": _test_runner,
}


# ---------------------------------------------------------------------------
# Tool registration.
# ---------------------------------------------------------------------------


def register_background_tools(registry: ToolRegistry, user_model) -> None:
    """Register the three background-task tools on the registry."""
    _user_id = _resolve_user_id(user_model)

    def enqueue_background_task(
        task: str,
        kind: str = "subagent",
        tools_scope: str = "research",
        args: dict | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Enqueue a long-running task; return a job_id immediately."""
        if kind not in SUPPORTED_KINDS:
            return {
                "status": "rejected",
                "error": f"Unknown kind '{kind}'. Allowed: {list(SUPPORTED_KINDS)}",
            }

        runner = _RUNNERS[kind]
        # Build per-kind args bundle that the runner can consume.
        if kind == "subagent":
            payload: dict[str, Any] = {
                "user_id": _user_id,
                "task": task,
                "tools_scope": tools_scope,
                **(args or {}),
            }
        elif kind == "macrocycle":
            payload = {
                "user_id": _user_id,
                "name": (args or {}).get("name") or task[:80] or "Macrocycle",
                **(args or {}),
            }
        elif kind == "plan":
            payload = {
                "user_id": _user_id,
                **(args or {}),
            }
            # Treat the `task` string as a focus hint if no focus given.
            if "focus" not in payload and task:
                payload["focus"] = task
        else:  # test
            payload = {"user_id": _user_id, **(args or {})}

        try:
            job_id = enqueue_job(
                user_id=_user_id,
                job_type=kind,
                args=payload,
                runner_fn=runner,
                session_id=session_id,
            )
        except Exception as exc:
            logger.exception("Failed to enqueue background task")
            return {"status": "rejected", "error": str(exc)}

        return {
            "status": "queued",
            "job_id": job_id,
            "kind": kind,
            "estimated_seconds": ESTIMATED_SECONDS.get(kind, 30),
        }

    registry.register(Tool(
        name="enqueue_background_task",
        description=(
            "Fork a long-running operation into the background and return "
            "IMMEDIATELY with a job_id. Mirrors Claude Code's Task tool with "
            "run_in_background=True. The user keeps chatting while a worker "
            "thread runs the actual work. On a later turn, call "
            "check_background_job(job_id) or list_my_jobs() to harvest the "
            "result and surface it to the user.\n\n"
            "WHEN TO USE:\n"
            "- The work would otherwise block the turn for >30 seconds AND\n"
            "- The user signalled patience or multitasking intent ('mach mal "
            "in Ruhe', 'lass das im Hintergrund laufen', 'waehrend ich was "
            "anderes mache') OR the user is mid-conversation and you want to "
            "keep them engaged on a different sub-task while heavy work runs.\n\n"
            "WHEN NOT TO USE (default = synchronous):\n"
            "- The user asked a focused question and is actively waiting. They "
            "expect immediate feedback, not a 'come back later' job_id.\n"
            "- The work is fast (<10 seconds). The overhead is not worth it.\n"
            "- The work has tight coupling to the next response (e.g. you "
            "would need the result THIS turn to answer the user). Block "
            "instead.\n\n"
            "KINDS:\n"
            "- 'subagent' (default): research subagent (web_search + web_fetch). "
            "Equivalent to spawn_subagent but async. Use for race lookups, "
            "methodology research, product comparisons.\n"
            "- 'macrocycle': run create_macrocycle_plan. Long (45-60s). Pass "
            "args={'name': '...', 'weeks': 16, 'start_date': 'YYYY-MM-DD', "
            "'periodization_model': '...'}.\n"
            "- 'plan': run create_training_plan. Medium (15-30s). Pass "
            "args={'focus': '...', 'macrocycle_week': N, 'feedback': '...', "
            "'sport_distribution': {...}}. The task string is used as focus "
            "if no focus is given in args.\n\n"
            "EXAMPLES:\n"
            "- enqueue_background_task(task='Find date, course, elevation "
            "for the Karlsruher Halbmarathon 2026', kind='subagent').\n"
            "- enqueue_background_task(task='', kind='macrocycle', args="
            "{'name': 'Berlin Marathon 2026', 'weeks': 16}).\n"
            "- enqueue_background_task(task='build phase, recovery low', "
            "kind='plan').\n\n"
            "Returns {'status': 'queued', 'job_id': '...', 'kind': '...', "
            "'estimated_seconds': N} on success. On rejection: {'status': "
            "'rejected', 'error': '...'}. NEVER fabricate a job_id - if the "
            "tool returned rejected, fall back to a synchronous call."
        ),
        handler=enqueue_background_task,
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "For kind='subagent': the research task string. "
                        "For kind='plan': used as 'focus' if args.focus omitted. "
                        "For kind='macrocycle': informational only (use args.name)."
                    ),
                },
                "kind": {
                    "type": "string",
                    "description": "Which operation to enqueue.",
                    "enum": ["subagent", "macrocycle", "plan"],
                },
                "tools_scope": {
                    "type": "string",
                    "description": (
                        "For kind='subagent' only: 'research' (web only) or "
                        "'readonly' (web + read-only DB tools)."
                    ),
                    "enum": ["research", "readonly"],
                    "nullable": True,
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Kind-specific arguments. macrocycle: {name, weeks, "
                        "start_date, periodization_model}. plan: {focus, "
                        "macrocycle_week, feedback, sport_distribution}. "
                        "subagent: {max_rounds, context}."
                    ),
                    "nullable": True,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session id that enqueued this job.",
                    "nullable": True,
                },
            },
            "required": ["task"],
        },
        category="meta",
    ))

    def check_background_job(job_id: str) -> dict:
        """Poll a background job by id and return its current row."""
        row = get_job(job_id)
        if not row:
            return {"error": f"No job with id {job_id}", "job_id": job_id}
        runtime = None
        started = row.get("started_at")
        completed = row.get("completed_at")
        if started:
            try:
                from datetime import datetime, timezone
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                end_dt = (
                    datetime.fromisoformat(completed.replace("Z", "+00:00"))
                    if completed else datetime.now(timezone.utc)
                )
                runtime = (end_dt - start_dt).total_seconds()
            except (ValueError, AttributeError):
                runtime = None
        return {
            "job_id": row.get("id"),
            "kind": row.get("job_type"),
            "status": row.get("status"),
            "result": row.get("result"),
            "error": row.get("error"),
            "runtime_seconds": runtime,
            "created_at": row.get("created_at"),
            "started_at": started,
            "completed_at": completed,
        }

    registry.register(Tool(
        name="check_background_job",
        description=(
            "Check the status of a single background job by id. Returns "
            "{status, result, error, runtime_seconds}. Status is one of "
            "'queued', 'running', 'done', 'failed'.\n\n"
            "USE THIS WHEN:\n"
            "- You previously called enqueue_background_task and want to "
            "see if the work finished before responding to the user.\n"
            "- list_my_jobs surfaced a job you want to inspect more closely.\n\n"
            "Returns 'result' only when status='done'. On 'failed' the "
            "'error' field has the exception message. On 'queued' or "
            "'running' result is null - tell the user it is still cooking "
            "and offer to keep going on something else."
        ),
        handler=check_background_job,
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job id returned by enqueue_background_task.",
                },
            },
            "required": ["job_id"],
        },
        category="meta",
    ))

    def list_my_jobs(limit: int = 10) -> dict:
        """Return recent jobs for the current user."""
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 10
        rows = list_pending_jobs(_user_id, limit=limit)
        summary = [
            {
                "job_id": r.get("id"),
                "kind": r.get("job_type"),
                "status": r.get("status"),
                "created_at": r.get("created_at"),
                "completed_at": r.get("completed_at"),
                "has_result": r.get("result") is not None,
                "error": r.get("error"),
            }
            for r in rows
        ]
        pending = [r for r in summary if r["status"] in ("queued", "running")]
        done = [r for r in summary if r["status"] == "done"]
        failed = [r for r in summary if r["status"] == "failed"]
        return {
            "count": len(summary),
            "pending": pending,
            "done": done,
            "failed": failed,
        }

    registry.register(Tool(
        name="list_my_jobs",
        description=(
            "List the user's recent background jobs (pending + done + "
            "failed). Returns a grouped summary so you can decide what to "
            "surface back to the user.\n\n"
            "CALL THIS AT THE START OF A TURN if:\n"
            "- You previously enqueued background work and have not yet "
            "harvested it.\n"
            "- The user asks something like 'was ist mit dem Plan?', 'ist "
            "das schon fertig?', 'hast du das gefunden?' - the answer "
            "likely lives in a job result.\n\n"
            "After calling list_my_jobs, call check_background_job(job_id) "
            "on any 'done' entries you want to fold into the response, then "
            "address them in your reply to the user. Don't dump the raw "
            "list - synthesise."
        ),
        handler=list_my_jobs,
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 10, max 50).",
                    "nullable": True,
                },
            },
        },
        category="meta",
    ))
