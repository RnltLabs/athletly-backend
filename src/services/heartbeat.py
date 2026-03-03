"""HeartbeatService — periodic background worker for proactive intelligence.

Runs every ``interval_seconds`` (default 30 min), scans all recently-active
users, and fires the proactive trigger check for each one.

Concurrency safety: before processing a user the service attempts to acquire
a Redis lock for that user's session.  If the lock is already held (meaning
the user is actively chatting) the tick skips that user to avoid interference.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Module-level Redis connection pool — created lazily on first use and reused
# across all lock operations for the lifetime of the process.
_redis_pool: "aioredis.Redis | None" = None  # noqa: F821  (type-only forward ref)

# How long a heartbeat lock is valid (seconds).
# Must be longer than the slowest proactive check.
_LOCK_TTL_SECONDS = 120

# How many users to process concurrently per tick.
_CONCURRENCY_LIMIT = 10

# Activity window: only process users active within this many days.
_ACTIVE_WINDOW_DAYS = 7


class HeartbeatService:
    """Asyncio-based periodic worker that runs proactive checks for all users."""

    def __init__(self, interval_seconds: int = 1800) -> None:
        self.interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        """Start the background loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="heartbeat")
        logger.info("HeartbeatService started (interval=%ds)", self.interval)

    async def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("HeartbeatService stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.error("Heartbeat tick error: %s", exc, exc_info=True)
            await asyncio.sleep(self.interval)

    async def _tick(self) -> None:
        """One heartbeat cycle — process all recently-active users."""
        now = datetime.now(timezone.utc)
        logger.info("Heartbeat tick at %s", now.isoformat())

        user_ids = await _fetch_active_user_ids()
        if not user_ids:
            logger.debug("No active users to process this tick")
            return

        logger.info("Heartbeat processing %d active users", len(user_ids))

        semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def process_one(user_id: str) -> None:
            async with semaphore:
                await _process_user(user_id)

        await asyncio.gather(*(process_one(uid) for uid in user_ids))


# ------------------------------------------------------------------
# User fetching
# ------------------------------------------------------------------


async def _fetch_active_user_ids() -> list[str]:
    """Query Supabase for users who have been active recently.

    Returns a list of user UUIDs.  Returns an empty list on any error so
    that a DB outage does not crash the heartbeat loop.
    """
    try:
        from src.db.client import get_async_supabase

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=_ACTIVE_WINDOW_DAYS)
        ).isoformat()

        client = await get_async_supabase()
        result = await (
            client.table("sessions")
            .select("user_id")
            .gte("last_active", cutoff)
            .execute()
        )

        rows: list[dict] = result.data or []
        # Deduplicate — one user may have many sessions.
        return list({row["user_id"] for row in rows if row.get("user_id")})
    except Exception as exc:
        logger.error("Failed to fetch active users: %s", exc)
        return []


# ------------------------------------------------------------------
# Per-user processing
# ------------------------------------------------------------------


async def _process_user(user_id: str) -> None:
    """Run proactive trigger check for a single user.

    Steps:
    1. Try to acquire a Redis lock (skip if user is chatting).
    2. Check proactive triggers from stored data.
    3. If triggers found, send a push notification.
    4. Release lock.
    """
    lock_key = f"heartbeat:lock:{user_id}"
    lock_acquired = await _try_acquire_lock(lock_key)
    if not lock_acquired:
        logger.debug("User %s is locked (chatting) — skipping", user_id)
        return

    try:
        triggers = await _check_triggers_for_user(user_id)
        if triggers:
            await _notify_user(user_id, triggers[0])
            logger.info(
                "Proactive trigger for user %s: %s",
                user_id,
                triggers[0].get("type"),
            )
    except Exception as exc:
        logger.error("Error processing user %s: %s", user_id, exc)
    finally:
        await _release_lock(lock_key)


async def _check_triggers_for_user(user_id: str) -> list[dict]:
    """Load user data from Supabase and evaluate proactive triggers.

    Returns a list of trigger dicts (may be empty).
    """
    try:
        from src.agent.proactive import check_proactive_triggers
        from src.db.client import get_async_supabase

        client = await get_async_supabase()

        # Fetch the most recent activity records.
        acts_result = await (
            client.table("activities")
            .select("*")
            .eq("user_id", user_id)
            .order("started_at", desc=True)
            .limit(20)
            .execute()
        )
        activities: list[dict] = acts_result.data or []

        # Fetch stored episodes.
        eps_result = await (
            client.table("episodes")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        episodes: list[dict] = eps_result.data or []

        # Fetch user profile.
        profile_result = await (
            client.table("profiles")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        athlete_profile: dict = (profile_result.data or {}) if profile_result else {}

        # Trigger check runs synchronously — offload to thread pool.
        loop = asyncio.get_running_loop()
        triggers = await loop.run_in_executor(
            None,
            check_proactive_triggers,
            athlete_profile,
            activities,
            episodes,
            {},  # trajectory placeholder
        )
        return triggers
    except Exception as exc:
        logger.error("Trigger check failed for user %s: %s", user_id, exc)
        return []


async def _notify_user(user_id: str, trigger: dict) -> None:
    """Send a push notification for the highest-priority trigger."""
    try:
        from src.agent.proactive import format_proactive_message
        from src.agent.tools.notification_tools import send_notification_async

        message_body = format_proactive_message(trigger, {})
        await send_notification_async(
            user_id=user_id,
            title="Your AI Coach",
            body=message_body,
            data={"trigger_type": trigger.get("type")},
        )
    except Exception as exc:
        logger.error("Notification failed for user %s: %s", user_id, exc)


# ------------------------------------------------------------------
# Redis lock helpers
# ------------------------------------------------------------------


async def _get_redis() -> "aioredis.Redis":
    """Return the shared Redis client, creating it once per process.

    Uses a connection pool backed by the configured ``redis_url``.
    Upstash Redis requires the ``rediss://`` (TLS) URL which is handled
    transparently by redis-py when the scheme is ``rediss``.
    """
    import redis.asyncio as aioredis
    from src.config import get_settings

    global _redis_pool  # noqa: PLW0603
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            # Single-connection pool is sufficient; heartbeat is not high-throughput.
            max_connections=5,
        )
    return _redis_pool


async def _try_acquire_lock(key: str) -> bool:
    """Try to set a Redis NX lock.  Returns True if acquired, False if busy."""
    try:
        client = await _get_redis()
        result = await client.set(key, "1", nx=True, ex=_LOCK_TTL_SECONDS)
        return result is True
    except Exception as exc:
        # If Redis is down, proceed without locking (best-effort).
        logger.warning("Redis lock unavailable (%s) — proceeding without lock", exc)
        return True


async def _release_lock(key: str) -> None:
    """Release a previously acquired Redis lock."""
    try:
        client = await _get_redis()
        await client.delete(key)
    except Exception as exc:
        logger.warning("Failed to release Redis lock %s: %s", key, exc)
