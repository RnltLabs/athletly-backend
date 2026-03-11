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
        self._tick_count: int = 0

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
        self._tick_count += 1
        logger.info("Heartbeat tick #%d at %s", self._tick_count, now.isoformat())

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

        # Self-improvement check every 12th tick (~6 hours at 30min interval)
        if self._tick_count % 12 == 0:
            await self._run_self_improvement(user_ids)

        # Episode consolidation check every 48th tick (~24 hours at 30min interval)
        if self._tick_count % 48 == 0:
            await self._run_episode_consolidation(user_ids)

    async def _run_episode_consolidation(self, user_ids: list[str]) -> None:
        """Check for unconsolidated months and consolidate them."""
        try:
            from src.services.episode_consolidation import (
                consolidate_month,
                get_unconsolidated_months,
            )

            for user_id in user_ids:
                try:
                    months = await get_unconsolidated_months(user_id)
                    for month in months[:2]:  # Max 2 months per tick
                        await consolidate_month(user_id, month)
                except Exception as exc:
                    logger.debug(
                        "Episode consolidation skip for %s: %s", user_id, exc,
                    )
        except Exception as exc:
            logger.error("Episode consolidation check failed: %s", exc)

    async def _run_self_improvement(self, user_ids: list[str]) -> None:
        """Queue self-improvement checks for users with metric definitions."""
        try:
            from src.agent.proactive import queue_proactive_message
            from src.db.agent_config_db import get_metric_definitions

            loop = asyncio.get_running_loop()

            for user_id in user_ids:
                try:
                    definitions = await loop.run_in_executor(
                        None, get_metric_definitions, user_id,
                    )
                    if definitions:
                        trigger = {
                            "type": "self_improvement_check",
                            "priority": "low",
                            "data": {
                                "metric_count": len(definitions),
                                "metric_names": [
                                    d.get("name", "") for d in definitions[:5]
                                ],
                            },
                        }
                        await loop.run_in_executor(
                            None,
                            lambda t=trigger: queue_proactive_message(
                                user_id=user_id,
                                trigger=t,
                                priority=0.3,
                            ),
                        )
                        logger.info(
                            "Self-improvement check queued for user %s (%d metrics)",
                            user_id, len(definitions),
                        )
                except Exception as exc:
                    logger.debug(
                        "Self-improvement skip for user %s: %s", user_id, exc,
                    )
        except Exception as exc:
            logger.error("Self-improvement check failed: %s", exc)


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
    4. Check silence-based conversation triggers and queue them.
    5. Release lock.
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

        # Silence-based conversation triggers (queue for later delivery).
        silence_queued = await _check_silence_triggers(user_id)
        if silence_queued:
            logger.info(
                "Silence triggers queued for user %s: %d message(s)",
                user_id,
                len(silence_queued),
            )

        # Unknown activity detection
        unknown_queued = await _detect_unknown_activities(user_id)
        if unknown_queued:
            logger.info(
                "Unknown activities queued for user %s: %d",
                user_id, len(unknown_queued),
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

        # Fetch the most recent activity records (agent + health sources).
        acts_result = await (
            client.table("activities")
            .select("*")
            .eq("user_id", user_id)
            .order("start_time", desc=True)
            .limit(20)
            .execute()
        )
        activities: list[dict] = acts_result.data or []

        # Also fetch health provider activities for cross-sport awareness.
        try:
            health_result = await (
                client.table("activities")
                .select("*")
                .eq("user_id", user_id)
                .in_("source", ["apple_health", "health_connect"])
                .order("start_time", desc=True)
                .limit(20)
                .execute()
            )
            health_acts: list[dict] = health_result.data or []
            # Normalize and merge (avoid duplicates via garmin_activity_id).
            agent_garmin_ids = {
                a["garmin_activity_id"]
                for a in activities
                if a.get("garmin_activity_id")
            }
            for ha in health_acts:
                if ha.get("external_id") not in agent_garmin_ids:
                    activities.append({
                        "sport": ha.get("activity_type", "unknown"),
                        "start_time": ha.get("start_time"),
                        "duration_seconds": ha.get("duration_seconds", 0),
                        "avg_hr": ha.get("avg_heart_rate"),
                        "max_hr": ha.get("max_heart_rate"),
                        "trimp": ha.get("training_load_trimp"),
                        "source": "health",
                    })
        except Exception as exc:
            logger.debug("Health activities fetch skipped: %s", exc)

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

        loop = asyncio.get_running_loop()

        # PHASE 4b: Dynamic trigger evaluation (agent-defined rules first)
        try:
            from src.agent.dynamic_triggers import evaluate_dynamic_triggers
            from src.db.health_data_db import get_merged_daily_metrics

            # Fetch merged daily metrics (Garmin + Health) synchronously.
            daily_metrics = await loop.run_in_executor(
                None,
                lambda: get_merged_daily_metrics(user_id, days=14),
            )

            dynamic_triggers = await loop.run_in_executor(
                None,
                evaluate_dynamic_triggers,
                user_id,
                activities,
                daily_metrics,
                athlete_profile,
            )

            if dynamic_triggers:
                return dynamic_triggers
        except Exception as exc:
            logger.debug("Dynamic trigger eval skipped for %s: %s", user_id, exc)

        # Fallback: hardcoded triggers
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


async def _fetch_last_interaction(user_id: str) -> str | None:
    """Query the sessions table for the user's most recent ``updated_at``.

    Returns an ISO timestamp string, or ``None`` if no sessions exist.
    """
    try:
        from src.db.client import get_async_supabase

        client = await get_async_supabase()
        result = await (
            client.table("sessions")
            .select("updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )

        rows: list[dict] = result.data or []
        if rows and rows[0].get("updated_at"):
            return str(rows[0]["updated_at"])
        return None
    except Exception as exc:
        logger.error("Failed to fetch last interaction for user %s: %s", user_id, exc)
        return None


async def _check_silence_triggers(user_id: str) -> list[dict]:
    """Check silence-based conversation triggers and queue any that fire.

    Queries the user's most recent session timestamp and passes it to
    ``check_conversation_triggers()``.  Any resulting triggers are queued
    via ``queue_proactive_message()`` for later delivery.

    Returns a list of queued message dicts (may be empty).
    Non-blocking: logs and returns [] on any error.
    """
    try:
        last_interaction = await _fetch_last_interaction(user_id)

        # No sessions at all — nothing to evaluate silence against.
        if last_interaction is None:
            return []

        from src.agent.proactive import (
            check_conversation_triggers,
            queue_proactive_message,
        )

        loop = asyncio.get_running_loop()

        # check_conversation_triggers is synchronous — offload to thread pool.
        triggers = await loop.run_in_executor(
            None,
            check_conversation_triggers,
            {},  # user_model_data (silence check only needs timestamp)
            last_interaction,
        )

        if not triggers:
            return []

        queued: list[dict] = []
        for trigger in triggers:
            # Use the trigger's urgency (from silence decay) as priority,
            # falling back to a moderate default.
            priority = trigger.get("urgency", 0.5)
            msg = await loop.run_in_executor(
                None,
                lambda t=trigger, p=priority: queue_proactive_message(
                    user_id=user_id,
                    trigger=t,
                    priority=p,
                ),
            )
            queued.append(msg)

        return queued
    except Exception as exc:
        logger.error("Silence trigger check failed for user %s: %s", user_id, exc)
        return []


async def _detect_unknown_activities(user_id: str) -> list[dict]:
    """Detect unclassified activities from the last 24 hours and queue triggers.

    Queries health_activities for activity_type IN ('unknown', 'other', 'uncategorized')
    within the last 24 hours. For each found, queues an unknown_activity trigger.
    """
    try:
        from src.agent.proactive import queue_proactive_message
        from src.db.client import get_async_supabase

        client = await get_async_supabase()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        result = await (
            client.table("activities")
            .select("id,activity_type,start_time,duration_seconds,distance_meters")
            .eq("user_id", user_id)
            .in_("source", ["apple_health", "health_connect"])
            .in_("activity_type", ["unknown", "other", "uncategorized"])
            .gte("start_time", cutoff)
            .execute()
        )

        unknown_acts = result.data or []
        if not unknown_acts:
            return []

        queued = []
        loop = asyncio.get_running_loop()
        for act in unknown_acts:
            duration_min = round((act.get("duration_seconds") or 0) / 60)
            trigger = {
                "type": "unknown_activity",
                "priority": "medium",
                "data": {
                    "activity_id": act.get("id"),
                    "start_time": act.get("start_time"),
                    "duration_minutes": duration_min,
                    "distance_meters": act.get("distance_meters"),
                },
            }
            msg = await loop.run_in_executor(
                None,
                lambda t=trigger: queue_proactive_message(
                    user_id=user_id,
                    trigger=t,
                    priority=0.6,
                ),
            )
            queued.append(msg)

        return queued
    except Exception as exc:
        logger.debug("Unknown activity detection skipped for %s: %s", user_id, exc)
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
