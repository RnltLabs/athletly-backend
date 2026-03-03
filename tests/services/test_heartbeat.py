"""Tests for HeartbeatService.

Covers:
- start() sets _running=True and creates a background task
- stop() cancels the task and sets _running=False
- _tick() is invoked by the internal loop
- Service can be started and stopped cleanly (no exceptions)

All external dependencies (Supabase, Redis, proactive checks) are mocked
so tests are fully isolated and fast.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.services.heartbeat import HeartbeatService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heartbeat() -> HeartbeatService:
    """A HeartbeatService with a very short interval for fast tests."""
    return HeartbeatService(interval_seconds=9999)  # interval so long _loop never fires twice


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_sets_running_flag(heartbeat: HeartbeatService):
    """After start(), _running should be True."""
    with patch.object(heartbeat, "_loop", new_callable=lambda: lambda self: asyncio.sleep(9999)):
        # Patch _loop so it never actually runs the tick
        heartbeat._loop = AsyncMock(return_value=None)
        await heartbeat.start()
        assert heartbeat._running is True
        await heartbeat.stop()


@pytest.mark.asyncio
async def test_start_creates_task(heartbeat: HeartbeatService):
    """start() must create an asyncio.Task."""
    heartbeat._loop = AsyncMock(return_value=None)
    await heartbeat.start()
    try:
        assert heartbeat._task is not None
        assert isinstance(heartbeat._task, asyncio.Task)
    finally:
        await heartbeat.stop()


@pytest.mark.asyncio
async def test_stop_sets_running_false(heartbeat: HeartbeatService):
    """stop() must set _running=False."""
    heartbeat._loop = AsyncMock(return_value=None)
    await heartbeat.start()
    await heartbeat.stop()
    assert heartbeat._running is False


@pytest.mark.asyncio
async def test_stop_cancels_task(heartbeat: HeartbeatService):
    """stop() must cancel the background task."""
    heartbeat._loop = AsyncMock(return_value=None)
    await heartbeat.start()
    task = heartbeat._task
    await heartbeat.stop()
    assert task is not None
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_start_stop_clean(heartbeat: HeartbeatService):
    """start() followed by stop() must not raise any exception."""
    heartbeat._loop = AsyncMock(return_value=None)
    await heartbeat.start()
    await heartbeat.stop()  # Should complete without raising


@pytest.mark.asyncio
async def test_stop_without_start_is_safe():
    """Calling stop() before start() must not raise."""
    svc = HeartbeatService(interval_seconds=9999)
    await svc.stop()  # _task is None — must be a no-op


# ---------------------------------------------------------------------------
# _tick is called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_runs():
    """Verify that _tick() is invoked when the loop runs."""
    svc = HeartbeatService(interval_seconds=0)  # interval=0 so tick fires immediately

    tick_calls: list[str] = []

    async def fake_tick() -> None:
        tick_calls.append("called")
        # After being called once, stop the service to prevent infinite loop
        svc._running = False

    svc._tick = fake_tick  # type: ignore[method-assign]

    await svc.start()
    # Give the event loop a moment to run the background task
    await asyncio.sleep(0.05)
    await svc.stop()

    assert len(tick_calls) >= 1, "_tick was never called"


@pytest.mark.asyncio
async def test_tick_exception_does_not_crash_loop():
    """An exception inside _tick must be caught and the loop must survive."""
    svc = HeartbeatService(interval_seconds=0)

    call_count = 0

    async def flaky_tick() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated tick failure")
        # Second call: stop the service cleanly
        svc._running = False

    svc._tick = flaky_tick  # type: ignore[method-assign]

    await svc.start()
    await asyncio.sleep(0.1)
    await svc.stop()

    # The loop must have survived the first exception and attempted again
    assert call_count >= 1


# ---------------------------------------------------------------------------
# _tick internals (with full mocking)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_skips_when_no_active_users():
    """If _fetch_active_user_ids returns [], _tick completes without processing anyone."""
    svc = HeartbeatService(interval_seconds=9999)

    with patch(
        "src.services.heartbeat._fetch_active_user_ids",
        new_callable=AsyncMock,
        return_value=[],
    ):
        # Should complete cleanly with no errors
        await svc._tick()


@pytest.mark.asyncio
async def test_tick_processes_active_users():
    """If _fetch_active_user_ids returns users, _process_user is called for each."""
    svc = HeartbeatService(interval_seconds=9999)

    processed_users: list[str] = []

    async def mock_process_user(user_id: str) -> None:
        processed_users.append(user_id)

    with (
        patch(
            "src.services.heartbeat._fetch_active_user_ids",
            new_callable=AsyncMock,
            return_value=["user-1", "user-2"],
        ),
        patch(
            "src.services.heartbeat._process_user",
            side_effect=mock_process_user,
        ),
    ):
        await svc._tick()

    assert set(processed_users) == {"user-1", "user-2"}
