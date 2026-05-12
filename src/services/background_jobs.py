"""Background job worker for long-running agent tool calls.

Pattern: Claude Code's `run_in_background` from the Task tool. The
Coach agent enqueues an operation (subagent spawn, macrocycle plan
generation, weekly plan generation) and returns immediately with a
``job_id``. A bounded ThreadPoolExecutor runs the actual work in the
background. The agent later polls ``get_job`` or ``list_pending_jobs``
to surface results back to the user.

Design notes:
- Single process scope. This is a worker pool inside the FastAPI /
  CLI process, not a distributed queue. Good enough for single-tenant
  alpha; swap for Celery / Inngest when we need durability across
  restarts.
- Thread-safety: each worker thread builds its own Supabase client
  via ``create_client`` so we never share a session across threads.
  ``get_supabase`` (the cached singleton) is only used from the
  enqueue path (main thread / async event loop thread).
- Failure-mode: any exception inside the runner is caught and the
  row transitions to ``status='failed'`` with the error message. No
  job is ever left hanging in ``running``.
- Result truncation: results are stored as JSONB. We trim very large
  payloads so we do not blow up the DB; the agent can re-run the work
  if it really needs the full payload.

Public API:
- ``enqueue_job(user_id, job_type, args, runner_fn, session_id=None) -> job_id``
- ``get_job(job_id) -> dict | None``
- ``list_pending_jobs(user_id, limit=20) -> list[dict]``
- ``shutdown_executor()`` -- for tests / clean shutdown.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from supabase import Client, create_client

from src.config import get_settings
from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# A runner takes (job_id, args_dict) and returns a JSON-serialisable result.
RunnerFn = Callable[[str, dict], Any]


MAX_WORKERS = 4
RESULT_BYTE_LIMIT = 200_000  # ~200 KB JSON cap per result row.

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Return the singleton ThreadPoolExecutor, creating it lazily."""
    global _executor  # noqa: PLW0603 -- singleton
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=MAX_WORKERS,
                thread_name_prefix="bgjob",
            )
            logger.info("background_jobs ThreadPoolExecutor started (max=%d)", MAX_WORKERS)
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Shut down the worker pool. Used by tests; no-op if not started."""
    global _executor  # noqa: PLW0603
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=wait)
            _executor = None


def _build_worker_client() -> Client:
    """Build a fresh Supabase client for use inside a worker thread.

    We never share a Supabase client across threads because the underlying
    httpx session is not guaranteed to be safe for our usage.
    """
    settings = get_settings()
    key = settings.supabase_service_role_key or settings.supabase_anon_key
    if not key:
        raise RuntimeError("No Supabase key configured for background worker.")
    return create_client(settings.supabase_url, key)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_result(result: Any) -> Any:
    """Cap result payload size to keep DB rows reasonable."""
    try:
        payload = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {"_truncated": True, "_note": "result not JSON-serialisable"}
    if len(payload) <= RESULT_BYTE_LIMIT:
        return result
    return {
        "_truncated": True,
        "_note": f"Result exceeded {RESULT_BYTE_LIMIT} bytes; first chunk preserved.",
        "_preview": payload[:RESULT_BYTE_LIMIT],
    }


def enqueue_job(
    user_id: str,
    job_type: str,
    args: dict,
    runner_fn: RunnerFn,
    session_id: str | None = None,
) -> str:
    """Insert a queued job row and submit work to the executor.

    Args:
        user_id: Owning user UUID. Must satisfy the FK on auth.users.
        job_type: Short tag for routing ("subagent", "macrocycle",
            "plan", or app-defined).
        args: Input arguments. Stored as JSONB on the row and passed
            verbatim to ``runner_fn``.
        runner_fn: Callable that performs the work. Signature
            ``(job_id, args) -> result``. Exceptions are caught and
            recorded on the row.
        session_id: Optional session identifier (which conversation
            enqueued this) so the agent can attribute results.

    Returns:
        The job_id (UUID string) of the inserted row.
    """
    client = get_supabase()
    row = {
        "user_id": user_id,
        "job_type": job_type,
        "status": "queued",
        "input_args": args,
        "session_id": session_id,
    }
    insert = client.table("background_jobs").insert(row).execute()
    if not insert.data:
        raise RuntimeError("Failed to insert background_jobs row.")
    job_id = insert.data[0]["id"]

    executor = _get_executor()
    executor.submit(_run_job, job_id, args, runner_fn)
    logger.info(
        "Enqueued background job %s (type=%s, user=%s, session=%s)",
        job_id, job_type, user_id, session_id,
    )
    return job_id


def _run_job(job_id: str, args: dict, runner_fn: RunnerFn) -> None:
    """Worker entrypoint. Marks running, executes runner, records outcome."""
    client = _build_worker_client()
    try:
        client.table("background_jobs").update({
            "status": "running",
            "started_at": _now_iso(),
        }).eq("id", job_id).execute()
    except Exception:
        logger.exception("Failed to mark job %s as running", job_id)
        # Continue anyway; we still want to attempt the work.

    try:
        result = runner_fn(job_id, args)
        safe_result = _truncate_result(result)
        client.table("background_jobs").update({
            "status": "done",
            "result": safe_result,
            "completed_at": _now_iso(),
        }).eq("id", job_id).execute()
        logger.info("Background job %s completed", job_id)
    except Exception as exc:  # noqa: BLE001 -- worker boundary: record everything
        logger.exception("Background job %s failed", job_id)
        try:
            client.table("background_jobs").update({
                "status": "failed",
                "error": str(exc)[:4000],
                "completed_at": _now_iso(),
            }).eq("id", job_id).execute()
        except Exception:
            logger.exception("Also failed to record failure for job %s", job_id)


def get_job(job_id: str) -> dict | None:
    """Fetch a single job row by id, or None if not found."""
    client = get_supabase()
    resp = (
        client.table("background_jobs")
        .select("*")
        .eq("id", job_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return resp.data[0]


def list_pending_jobs(user_id: str, limit: int = 20) -> list[dict]:
    """Return queued, running, or recently finished jobs for the user.

    "Recently finished" = the latest ``limit`` rows after ordering by
    created_at DESC, regardless of status. The agent typically calls
    this once per turn to see what to surface back to the user.
    """
    client = get_supabase()
    resp = (
        client.table("background_jobs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []
