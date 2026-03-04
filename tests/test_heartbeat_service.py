"""Unit tests for src.services.heartbeat — HeartbeatService.

Covers:
- Lifecycle (start, stop, _running flag)
- _tick() — fetches active users, processes each in parallel, handles empty list
- _process_user() — acquire lock, check triggers, release lock, skip if locked
- _fetch_active_user_ids() — queries sessions, deduplicates user_ids
- _check_triggers_for_user() — loads activities + episodes + health + profile
- _notify_user() — formats message, calls send_notification_async
- Redis lock helpers — _try_acquire_lock (NX), _release_lock (delete), _get_redis
- Error handling — non-blocking (errors logged, don't crash loop)
- Concurrency — semaphore limit of 10

All async functions are mocked via AsyncMock.  Tests are synchronous
(unittest.TestCase) using asyncio.run() to drive async code under test.
"""

from __future__ import annotations

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.services.heartbeat import (
    HeartbeatService,
    _check_triggers_for_user,
    _fetch_active_user_ids,
    _get_redis,
    _notify_user,
    _process_user,
    _release_lock,
    _try_acquire_lock,
    _CONCURRENCY_LIMIT,
    _LOCK_TTL_SECONDS,
    _ACTIVE_WINDOW_DAYS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-heartbeat-test-uuid"
USER_ID_2 = "user-heartbeat-test-uuid-2"
USER_ID_3 = "user-heartbeat-test-uuid-3"


# ---------------------------------------------------------------------------
# Async-chain mock builder (mirrors project pattern from test_proactive_queue_db)
# ---------------------------------------------------------------------------


def _make_async_chain(rows: list[dict]) -> MagicMock:
    """Return a mock async query chain whose (await .execute()).data == *rows*.

    All intermediate query-builder methods return the same chain so any
    call order (eq, gte, order, limit, select, maybe_single, ...) works.
    """
    chain = MagicMock()
    result = MagicMock()
    result.data = rows

    for method_name in [
        "select",
        "eq",
        "gte",
        "lt",
        "order",
        "limit",
        "neq",
        "in_",
        "upsert",
        "update",
        "insert",
        "maybe_single",
        "desc",
    ]:
        getattr(chain, method_name).return_value = chain

    # .execute() must be awaitable (AsyncMock)
    chain.execute = AsyncMock(return_value=result)
    return chain


def _mock_async_supabase(table_chains: dict[str, MagicMock] | None = None) -> AsyncMock:
    """Return a mock async Supabase client.

    *table_chains* maps table names to chain mocks.  If a table name is not
    in the dict a default empty-result chain is used.
    """
    client = AsyncMock()
    table_chains = table_chains or {}

    def _table(name: str) -> MagicMock:
        return table_chains.get(name, _make_async_chain([]))

    client.table = MagicMock(side_effect=_table)
    return client


def _mock_redis(
    *,
    set_return: bool | None = True,
    raise_on_set: Exception | None = None,
    raise_on_delete: Exception | None = None,
) -> AsyncMock:
    """Return a mock redis.asyncio client."""
    redis = AsyncMock()
    if raise_on_set:
        redis.set = AsyncMock(side_effect=raise_on_set)
    else:
        redis.set = AsyncMock(return_value=set_return)
    if raise_on_delete:
        redis.delete = AsyncMock(side_effect=raise_on_delete)
    else:
        redis.delete = AsyncMock()
    return redis


# ===========================================================================
# 1. Lifecycle: start(), stop(), _running flag
# ===========================================================================


class TestHeartbeatServiceLifecycle(unittest.TestCase):
    """Tests for HeartbeatService construction, start, and stop."""

    def test_default_interval(self) -> None:
        svc = HeartbeatService()
        self.assertEqual(svc.interval, 1800)

    def test_custom_interval(self) -> None:
        svc = HeartbeatService(interval_seconds=60)
        self.assertEqual(svc.interval, 60)

    def test_initial_state_not_running(self) -> None:
        svc = HeartbeatService()
        self.assertFalse(svc._running)
        self.assertIsNone(svc._task)

    def test_start_sets_running_flag(self) -> None:
        svc = HeartbeatService()

        async def _run() -> None:
            # Patch _loop so it doesn't actually run forever.
            with patch.object(svc, "_loop", new_callable=AsyncMock):
                await svc.start()
                self.assertTrue(svc._running)
                self.assertIsNotNone(svc._task)
                # Clean up
                await svc.stop()

        asyncio.run(_run())

    def test_stop_clears_running_flag(self) -> None:
        svc = HeartbeatService()

        async def _run() -> None:
            with patch.object(svc, "_loop", new_callable=AsyncMock):
                await svc.start()
                await svc.stop()
                self.assertFalse(svc._running)

        asyncio.run(_run())

    def test_stop_cancels_task(self) -> None:
        svc = HeartbeatService()

        async def _run() -> None:
            with patch.object(svc, "_loop", new_callable=AsyncMock):
                await svc.start()
                task = svc._task
                await svc.stop()
                self.assertTrue(task.done())

        asyncio.run(_run())

    def test_stop_is_safe_when_not_started(self) -> None:
        """Calling stop() before start() must not raise."""
        svc = HeartbeatService()

        async def _run() -> None:
            await svc.stop()
            self.assertFalse(svc._running)

        asyncio.run(_run())


# ===========================================================================
# 2. _tick()
# ===========================================================================


class TestTick(unittest.TestCase):
    """Tests for HeartbeatService._tick()."""

    def test_tick_fetches_active_users_and_processes(self) -> None:
        """_tick() calls _fetch_active_user_ids then _process_user for each."""
        svc = HeartbeatService()

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._fetch_active_user_ids",
                    new_callable=AsyncMock,
                    return_value=[USER_ID, USER_ID_2],
                ) as mock_fetch,
                patch(
                    "src.services.heartbeat._process_user",
                    new_callable=AsyncMock,
                ) as mock_process,
            ):
                await svc._tick()
                mock_fetch.assert_awaited_once()
                self.assertEqual(mock_process.await_count, 2)

        asyncio.run(_run())

    def test_tick_returns_early_on_empty_user_list(self) -> None:
        """_tick() skips processing when no active users."""
        svc = HeartbeatService()

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._fetch_active_user_ids",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
                patch(
                    "src.services.heartbeat._process_user",
                    new_callable=AsyncMock,
                ) as mock_process,
            ):
                await svc._tick()
                mock_process.assert_not_awaited()

        asyncio.run(_run())

    def test_tick_uses_semaphore_for_concurrency(self) -> None:
        """_tick() creates a semaphore with _CONCURRENCY_LIMIT."""
        svc = HeartbeatService()
        user_ids = [f"user-{i}" for i in range(_CONCURRENCY_LIMIT + 5)]

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._fetch_active_user_ids",
                    new_callable=AsyncMock,
                    return_value=user_ids,
                ),
                patch(
                    "src.services.heartbeat._process_user",
                    new_callable=AsyncMock,
                ) as mock_process,
            ):
                await svc._tick()
                self.assertEqual(mock_process.await_count, len(user_ids))

        asyncio.run(_run())

    def test_tick_processes_all_users_via_gather(self) -> None:
        """All user IDs passed to _process_user via asyncio.gather."""
        svc = HeartbeatService()

        async def _run() -> None:
            called_ids: list[str] = []

            async def _capture_process(user_id: str) -> None:
                called_ids.append(user_id)

            with (
                patch(
                    "src.services.heartbeat._fetch_active_user_ids",
                    new_callable=AsyncMock,
                    return_value=[USER_ID, USER_ID_2, USER_ID_3],
                ),
                patch(
                    "src.services.heartbeat._process_user",
                    side_effect=_capture_process,
                ),
            ):
                await svc._tick()
                self.assertEqual(sorted(called_ids), sorted([USER_ID, USER_ID_2, USER_ID_3]))

        asyncio.run(_run())


# ===========================================================================
# 3. _process_user()
# ===========================================================================


class TestProcessUser(unittest.TestCase):
    """Tests for _process_user()."""

    def test_acquires_lock_calls_triggers_releases_lock(self) -> None:
        trigger = {"type": "goal_at_risk", "priority": "high", "data": {}}

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_lock,
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                    return_value=[trigger],
                ) as mock_check,
                patch(
                    "src.services.heartbeat._notify_user",
                    new_callable=AsyncMock,
                ) as mock_notify,
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ) as mock_release,
            ):
                await _process_user(USER_ID)
                mock_lock.assert_awaited_once_with(f"heartbeat:lock:{USER_ID}")
                mock_check.assert_awaited_once_with(USER_ID)
                mock_notify.assert_awaited_once_with(USER_ID, trigger)
                mock_release.assert_awaited_once_with(f"heartbeat:lock:{USER_ID}")

        asyncio.run(_run())

    def test_skips_user_when_lock_not_acquired(self) -> None:
        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                ) as mock_check,
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ) as mock_release,
            ):
                await _process_user(USER_ID)
                mock_check.assert_not_awaited()
                mock_release.assert_not_awaited()

        asyncio.run(_run())

    def test_no_notification_when_no_triggers(self) -> None:
        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
                patch(
                    "src.services.heartbeat._notify_user",
                    new_callable=AsyncMock,
                ) as mock_notify,
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ),
            ):
                await _process_user(USER_ID)
                mock_notify.assert_not_awaited()

        asyncio.run(_run())

    def test_lock_released_even_on_exception(self) -> None:
        """Lock must be released in the finally block even if triggers raise."""

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("trigger crash"),
                ),
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ) as mock_release,
            ):
                # Should NOT raise — error is caught internally.
                await _process_user(USER_ID)
                mock_release.assert_awaited_once_with(f"heartbeat:lock:{USER_ID}")

        asyncio.run(_run())

    def test_only_first_trigger_sent(self) -> None:
        """_process_user sends notification for triggers[0] only."""
        trigger_1 = {"type": "goal_at_risk", "priority": "high", "data": {}}
        trigger_2 = {"type": "on_track", "priority": "low", "data": {}}

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                    return_value=[trigger_1, trigger_2],
                ),
                patch(
                    "src.services.heartbeat._notify_user",
                    new_callable=AsyncMock,
                ) as mock_notify,
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ),
            ):
                await _process_user(USER_ID)
                mock_notify.assert_awaited_once_with(USER_ID, trigger_1)

        asyncio.run(_run())

    def test_error_in_process_user_does_not_propagate(self) -> None:
        """_process_user swallows exceptions — heartbeat loop must not crash."""

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                    side_effect=ValueError("boom"),
                ),
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ),
            ):
                # Must not raise.
                await _process_user(USER_ID)

        asyncio.run(_run())


# ===========================================================================
# 4. _fetch_active_user_ids()
# ===========================================================================


class TestFetchActiveUserIds(unittest.TestCase):
    """Tests for _fetch_active_user_ids()."""

    def test_returns_user_ids_from_sessions(self) -> None:
        sessions_chain = _make_async_chain([
            {"user_id": USER_ID},
            {"user_id": USER_ID_2},
        ])
        client = _mock_async_supabase({"sessions": sessions_chain})

        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                return_value=client,
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        self.assertEqual(sorted(result), sorted([USER_ID, USER_ID_2]))

    def test_deduplicates_user_ids(self) -> None:
        """Multiple sessions for the same user yield a single entry."""
        sessions_chain = _make_async_chain([
            {"user_id": USER_ID},
            {"user_id": USER_ID},
            {"user_id": USER_ID},
        ])
        client = _mock_async_supabase({"sessions": sessions_chain})

        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                return_value=client,
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], USER_ID)

    def test_returns_empty_list_when_no_sessions(self) -> None:
        sessions_chain = _make_async_chain([])
        client = _mock_async_supabase({"sessions": sessions_chain})

        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                return_value=client,
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        self.assertEqual(result, [])

    def test_returns_empty_list_on_supabase_error(self) -> None:
        """DB errors must not propagate — return empty list instead."""

        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Supabase down"),
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        self.assertEqual(result, [])

    def test_filters_out_null_user_ids(self) -> None:
        """Rows with user_id=None should be excluded."""
        sessions_chain = _make_async_chain([
            {"user_id": USER_ID},
            {"user_id": None},
            {"user_id": ""},
        ])
        client = _mock_async_supabase({"sessions": sessions_chain})

        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                return_value=client,
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        # None and "" are both falsy, so they should be filtered out.
        self.assertEqual(result, [USER_ID])

    def test_queries_sessions_table(self) -> None:
        """Must query the 'sessions' table."""
        sessions_chain = _make_async_chain([])
        client = _mock_async_supabase({"sessions": sessions_chain})

        async def _run() -> None:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                return_value=client,
            ):
                await _fetch_active_user_ids()
            client.table.assert_called_with("sessions")

        asyncio.run(_run())

    def test_result_data_none_returns_empty_list(self) -> None:
        """When result.data is None, should not crash."""
        chain = MagicMock()
        result_mock = MagicMock()
        result_mock.data = None
        for m in ["select", "eq", "gte", "lt", "order", "limit"]:
            getattr(chain, m).return_value = chain
        chain.execute = AsyncMock(return_value=result_mock)

        client = _mock_async_supabase({"sessions": chain})

        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                return_value=client,
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        self.assertEqual(result, [])


# ===========================================================================
# 5. _check_triggers_for_user()
# ===========================================================================


class TestCheckTriggersForUser(unittest.TestCase):
    """Tests for _check_triggers_for_user()."""

    def _build_supabase(
        self,
        activities: list[dict] | None = None,
        health_activities: list[dict] | None = None,
        episodes: list[dict] | None = None,
        profile: dict | None = None,
    ) -> AsyncMock:
        """Build a multi-table mock Supabase client."""
        act_chain = _make_async_chain(activities or [])
        health_chain = _make_async_chain(health_activities or [])
        eps_chain = _make_async_chain(episodes or [])

        # profiles uses maybe_single — need special handling
        profile_chain = MagicMock()
        profile_result = MagicMock()
        profile_result.data = profile or {}
        for m in ["select", "eq", "gte", "lt", "order", "limit", "maybe_single"]:
            getattr(profile_chain, m).return_value = profile_chain
        profile_chain.execute = AsyncMock(return_value=profile_result)

        return _mock_async_supabase({
            "activities": act_chain,
            "health_activities": health_chain,
            "episodes": eps_chain,
            "profiles": profile_chain,
        })

    def test_returns_triggers_list(self) -> None:
        client = self._build_supabase()
        triggers = [{"type": "low_activity", "priority": "high", "data": {}}]

        async def _run() -> list[dict]:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=triggers,
                ),
            ):
                return await _check_triggers_for_user(USER_ID)

        result = asyncio.run(_run())
        self.assertEqual(result, triggers)

    def test_returns_empty_list_on_error(self) -> None:
        """Any exception in data fetching returns empty list."""

        async def _run() -> list[dict]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                side_effect=RuntimeError("DB down"),
            ):
                return await _check_triggers_for_user(USER_ID)

        result = asyncio.run(_run())
        self.assertEqual(result, [])

    def test_fetches_activities_episodes_and_profile(self) -> None:
        """Must query activities, episodes, and profiles tables."""
        client = self._build_supabase()

        async def _run() -> None:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=[],
                ),
            ):
                await _check_triggers_for_user(USER_ID)

            table_calls = [c[0][0] for c in client.table.call_args_list]
            self.assertIn("activities", table_calls)
            self.assertIn("episodes", table_calls)
            self.assertIn("profiles", table_calls)

        asyncio.run(_run())

    def test_fetches_health_activities(self) -> None:
        """Must also query health_activities table."""
        client = self._build_supabase()

        async def _run() -> None:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=[],
                ),
            ):
                await _check_triggers_for_user(USER_ID)

            table_calls = [c[0][0] for c in client.table.call_args_list]
            self.assertIn("health_activities", table_calls)

        asyncio.run(_run())

    def test_calls_check_proactive_triggers_with_correct_args(self) -> None:
        profile = {"name": "Test Runner", "sports": ["running"]}
        activities = [{"sport": "running", "start_time": "2026-03-01"}]
        episodes = [{"type": "training_plan", "created_at": "2026-03-01"}]

        client = self._build_supabase(
            activities=activities,
            episodes=episodes,
            profile=profile,
        )

        async def _run() -> None:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=[],
                ) as mock_check,
            ):
                await _check_triggers_for_user(USER_ID)
                mock_check.assert_called_once()
                args = mock_check.call_args[0]
                # Args: (athlete_profile, activities, episodes, trajectory_placeholder)
                self.assertEqual(args[0], profile)
                self.assertIsInstance(args[1], list)  # activities (merged)
                self.assertEqual(args[2], episodes)
                self.assertEqual(args[3], {})  # trajectory placeholder

        asyncio.run(_run())

    def test_health_activities_merged_without_duplicates(self) -> None:
        """Health activities with matching external_id should not be duplicated."""
        activities = [
            {
                "sport": "running",
                "garmin_activity_id": "garmin-001",
                "start_time": "2026-03-01",
            }
        ]
        health_activities = [
            {
                "external_id": "garmin-001",  # duplicate — skip
                "activity_type": "running",
                "start_time": "2026-03-01",
            },
            {
                "external_id": "garmin-002",  # new — merge
                "activity_type": "cycling",
                "start_time": "2026-03-02",
                "duration_seconds": 3600,
                "avg_heart_rate": 140,
                "max_heart_rate": 165,
                "training_load_trimp": 80,
            },
        ]

        client = self._build_supabase(
            activities=activities,
            health_activities=health_activities,
        )

        async def _run() -> None:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=[],
                ) as mock_check,
            ):
                await _check_triggers_for_user(USER_ID)
                merged = mock_check.call_args[0][1]
                # 1 agent activity + 1 merged health activity = 2 total
                self.assertEqual(len(merged), 2)
                # Verify the merged health activity has source="health"
                health_merged = [a for a in merged if a.get("source") == "health"]
                self.assertEqual(len(health_merged), 1)
                self.assertEqual(health_merged[0]["sport"], "cycling")

        asyncio.run(_run())

    def test_health_activities_fetch_failure_does_not_crash(self) -> None:
        """If health_activities table query fails, processing continues."""
        activities = [{"sport": "running", "start_time": "2026-03-01"}]

        # Build a special client where health_activities raises
        act_chain = _make_async_chain(activities)
        eps_chain = _make_async_chain([])

        health_chain = MagicMock()
        for m in ["select", "eq", "gte", "lt", "order", "limit"]:
            getattr(health_chain, m).return_value = health_chain
        health_chain.execute = AsyncMock(side_effect=RuntimeError("table not found"))

        profile_chain = MagicMock()
        profile_result = MagicMock()
        profile_result.data = {}
        for m in ["select", "eq", "gte", "lt", "order", "limit", "maybe_single"]:
            getattr(profile_chain, m).return_value = profile_chain
        profile_chain.execute = AsyncMock(return_value=profile_result)

        client = _mock_async_supabase({
            "activities": act_chain,
            "health_activities": health_chain,
            "episodes": eps_chain,
            "profiles": profile_chain,
        })

        async def _run() -> list[dict]:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=[],
                ),
            ):
                return await _check_triggers_for_user(USER_ID)

        # Must not raise
        result = asyncio.run(_run())
        self.assertEqual(result, [])

    def test_profile_none_result_yields_empty_dict(self) -> None:
        """If profile_result is None-ish, athlete_profile should be {}."""
        client = self._build_supabase(profile=None)

        async def _run() -> None:
            with (
                patch(
                    "src.db.client.get_async_supabase",
                    new_callable=AsyncMock,
                    return_value=client,
                ),
                patch(
                    "src.agent.proactive.check_proactive_triggers",
                    return_value=[],
                ) as mock_check,
            ):
                await _check_triggers_for_user(USER_ID)
                profile_arg = mock_check.call_args[0][0]
                self.assertIsInstance(profile_arg, dict)

        asyncio.run(_run())


# ===========================================================================
# 6. _notify_user()
# ===========================================================================


class TestNotifyUser(unittest.TestCase):
    """Tests for _notify_user()."""

    def test_sends_notification_with_formatted_message(self) -> None:
        trigger = {"type": "goal_at_risk", "priority": "high", "data": {}}

        async def _run() -> None:
            with (
                patch(
                    "src.agent.proactive.format_proactive_message",
                    return_value="Your goal is at risk!",
                ) as mock_format,
                patch(
                    "src.agent.tools.notification_tools.send_notification_async",
                    new_callable=AsyncMock,
                ) as mock_send,
            ):
                await _notify_user(USER_ID, trigger)
                mock_format.assert_called_once_with(trigger, {})
                mock_send.assert_awaited_once_with(
                    user_id=USER_ID,
                    title="Your AI Coach",
                    body="Your goal is at risk!",
                    data={"trigger_type": "goal_at_risk"},
                )

        asyncio.run(_run())

    def test_does_not_propagate_exception(self) -> None:
        """Notification failures must be swallowed."""
        trigger = {"type": "on_track", "data": {}}

        async def _run() -> None:
            with (
                patch(
                    "src.agent.proactive.format_proactive_message",
                    side_effect=RuntimeError("format error"),
                ),
            ):
                # Must not raise.
                await _notify_user(USER_ID, trigger)

        asyncio.run(_run())

    def test_trigger_type_in_notification_data(self) -> None:
        trigger = {"type": "low_activity", "data": {}}

        async def _run() -> None:
            with (
                patch(
                    "src.agent.proactive.format_proactive_message",
                    return_value="Move more!",
                ),
                patch(
                    "src.agent.tools.notification_tools.send_notification_async",
                    new_callable=AsyncMock,
                ) as mock_send,
            ):
                await _notify_user(USER_ID, trigger)
                sent_data = mock_send.call_args[1]["data"]
                self.assertEqual(sent_data["trigger_type"], "low_activity")

        asyncio.run(_run())

    def test_trigger_without_type_key(self) -> None:
        """Trigger dict missing 'type' should not crash — uses .get()."""
        trigger = {"data": {"metric": 42}}

        async def _run() -> None:
            with (
                patch(
                    "src.agent.proactive.format_proactive_message",
                    return_value="Check in",
                ),
                patch(
                    "src.agent.tools.notification_tools.send_notification_async",
                    new_callable=AsyncMock,
                ) as mock_send,
            ):
                await _notify_user(USER_ID, trigger)
                sent_data = mock_send.call_args[1]["data"]
                # trigger.get("type") returns None when key missing
                self.assertIsNone(sent_data["trigger_type"])

        asyncio.run(_run())


# ===========================================================================
# 7. Redis lock helpers
# ===========================================================================


class TestRedisLockHelpers(unittest.TestCase):
    """Tests for _try_acquire_lock, _release_lock, _get_redis."""

    def test_try_acquire_lock_success(self) -> None:
        redis = _mock_redis(set_return=True)

        async def _run() -> bool:
            with patch(
                "src.services.heartbeat._get_redis",
                new_callable=AsyncMock,
                return_value=redis,
            ):
                return await _try_acquire_lock("test:lock:key")

        result = asyncio.run(_run())
        self.assertTrue(result)
        redis.set.assert_awaited_once_with(
            "test:lock:key", "1", nx=True, ex=_LOCK_TTL_SECONDS,
        )

    def test_try_acquire_lock_already_held(self) -> None:
        redis = _mock_redis(set_return=None)  # Redis SET NX returns None if key exists

        async def _run() -> bool:
            with patch(
                "src.services.heartbeat._get_redis",
                new_callable=AsyncMock,
                return_value=redis,
            ):
                return await _try_acquire_lock("test:lock:key")

        result = asyncio.run(_run())
        self.assertFalse(result)

    def test_try_acquire_lock_redis_down_returns_true(self) -> None:
        """If Redis is unavailable, proceed without lock (best-effort)."""

        async def _run() -> bool:
            with patch(
                "src.services.heartbeat._get_redis",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Redis down"),
            ):
                return await _try_acquire_lock("test:lock:key")

        result = asyncio.run(_run())
        self.assertTrue(result)

    def test_try_acquire_lock_redis_set_error_returns_true(self) -> None:
        """If Redis set() raises, return True (best-effort)."""
        redis = _mock_redis(raise_on_set=RuntimeError("connection reset"))

        async def _run() -> bool:
            with patch(
                "src.services.heartbeat._get_redis",
                new_callable=AsyncMock,
                return_value=redis,
            ):
                return await _try_acquire_lock("test:lock:key")

        result = asyncio.run(_run())
        self.assertTrue(result)

    def test_release_lock_calls_delete(self) -> None:
        redis = _mock_redis()

        async def _run() -> None:
            with patch(
                "src.services.heartbeat._get_redis",
                new_callable=AsyncMock,
                return_value=redis,
            ):
                await _release_lock("test:lock:key")
            redis.delete.assert_awaited_once_with("test:lock:key")

        asyncio.run(_run())

    def test_release_lock_swallows_redis_error(self) -> None:
        """If Redis delete fails, the error is logged but not raised."""
        redis = _mock_redis(raise_on_delete=ConnectionError("Redis gone"))

        async def _run() -> None:
            with patch(
                "src.services.heartbeat._get_redis",
                new_callable=AsyncMock,
                return_value=redis,
            ):
                # Must not raise.
                await _release_lock("test:lock:key")

        asyncio.run(_run())

    def test_get_redis_creates_singleton(self) -> None:
        """_get_redis should create pool on first call and reuse on second."""
        mock_from_url = MagicMock(return_value=MagicMock())

        async def _run() -> None:
            import src.services.heartbeat as hb_module
            # Reset module-level pool.
            original_pool = hb_module._redis_pool
            hb_module._redis_pool = None

            try:
                with (
                    patch("redis.asyncio.from_url", mock_from_url),
                    patch("src.config.get_settings") as mock_settings,
                ):
                    mock_settings.return_value.redis_url = "redis://localhost:6379"
                    r1 = await _get_redis()
                    r2 = await _get_redis()
                    # from_url should only be called once (singleton).
                    mock_from_url.assert_called_once()
                    self.assertIs(r1, r2)
            finally:
                hb_module._redis_pool = original_pool

        asyncio.run(_run())

    def test_lock_ttl_constant(self) -> None:
        self.assertEqual(_LOCK_TTL_SECONDS, 120)


# ===========================================================================
# 8. Error handling — non-blocking
# ===========================================================================


class TestErrorHandling(unittest.TestCase):
    """Errors in any stage must not crash the heartbeat loop."""

    def test_tick_error_does_not_crash_loop(self) -> None:
        """_loop catches exceptions from _tick and continues."""
        svc = HeartbeatService(interval_seconds=0)
        tick_count = 0

        async def _counting_tick() -> None:
            nonlocal tick_count
            tick_count += 1
            if tick_count == 1:
                raise RuntimeError("first tick boom")
            # On second call, stop the loop.
            svc._running = False

        async def _run() -> None:
            with patch.object(svc, "_tick", side_effect=_counting_tick):
                svc._running = True
                # Run loop directly (not via start() to avoid task overhead)
                await svc._loop()
            # Loop ran twice: first tick errored, second tick stopped.
            self.assertEqual(tick_count, 2)

        asyncio.run(_run())

    def test_fetch_active_users_error_returns_empty(self) -> None:
        async def _run() -> list[str]:
            with patch(
                "src.db.client.get_async_supabase",
                new_callable=AsyncMock,
                side_effect=Exception("total DB failure"),
            ):
                return await _fetch_active_user_ids()

        result = asyncio.run(_run())
        self.assertEqual(result, [])

    def test_notify_user_error_logged_not_raised(self) -> None:
        trigger = {"type": "test", "data": {}}

        async def _run() -> None:
            with (
                patch(
                    "src.agent.proactive.format_proactive_message",
                    return_value="msg",
                ),
                patch(
                    "src.agent.tools.notification_tools.send_notification_async",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("push service down"),
                ),
            ):
                # Must not raise.
                await _notify_user(USER_ID, trigger)

        asyncio.run(_run())


# ===========================================================================
# 9. Concurrency — semaphore limit
# ===========================================================================


class TestConcurrency(unittest.TestCase):
    """Tests for concurrency control in _tick()."""

    def test_concurrency_limit_constant(self) -> None:
        self.assertEqual(_CONCURRENCY_LIMIT, 10)

    def test_active_window_days_constant(self) -> None:
        self.assertEqual(_ACTIVE_WINDOW_DAYS, 7)

    def test_semaphore_limits_concurrent_processing(self) -> None:
        """At most _CONCURRENCY_LIMIT users should be processed concurrently."""
        svc = HeartbeatService()
        max_concurrent = 0
        current_concurrent = 0

        async def _slow_process(user_id: str) -> None:
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
            # Yield control to simulate actual async work.
            await asyncio.sleep(0.01)
            current_concurrent -= 1

        user_ids = [f"user-{i}" for i in range(25)]

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._fetch_active_user_ids",
                    new_callable=AsyncMock,
                    return_value=user_ids,
                ),
                patch(
                    "src.services.heartbeat._process_user",
                    side_effect=_slow_process,
                ),
            ):
                await svc._tick()
            self.assertLessEqual(max_concurrent, _CONCURRENCY_LIMIT)

        asyncio.run(_run())

    def test_all_users_processed_despite_semaphore(self) -> None:
        """Even with the semaphore, all users must eventually be processed."""
        svc = HeartbeatService()
        processed: list[str] = []

        async def _track_process(user_id: str) -> None:
            processed.append(user_id)
            await asyncio.sleep(0.001)

        user_ids = [f"user-{i}" for i in range(20)]

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._fetch_active_user_ids",
                    new_callable=AsyncMock,
                    return_value=user_ids,
                ),
                patch(
                    "src.services.heartbeat._process_user",
                    side_effect=_track_process,
                ),
            ):
                await svc._tick()
            self.assertEqual(sorted(processed), sorted(user_ids))

        asyncio.run(_run())


# ===========================================================================
# 10. _loop() integration
# ===========================================================================


class TestLoop(unittest.TestCase):
    """Tests for the internal _loop() method."""

    def test_loop_exits_when_running_cleared(self) -> None:
        svc = HeartbeatService(interval_seconds=0)
        call_count = 0

        async def _stop_tick() -> None:
            nonlocal call_count
            call_count += 1
            svc._running = False

        async def _run() -> None:
            svc._running = True
            with patch.object(svc, "_tick", side_effect=_stop_tick):
                await svc._loop()
            self.assertEqual(call_count, 1)

        asyncio.run(_run())

    def test_loop_sleeps_for_interval(self) -> None:
        svc = HeartbeatService(interval_seconds=42)

        async def _stop_tick() -> None:
            svc._running = False

        async def _run() -> None:
            svc._running = True
            with (
                patch.object(svc, "_tick", side_effect=_stop_tick),
                patch("src.services.heartbeat.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            ):
                await svc._loop()
                mock_sleep.assert_awaited_once_with(42)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
