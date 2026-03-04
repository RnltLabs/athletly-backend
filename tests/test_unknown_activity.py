"""Unit tests for Unknown Activity Flow — detection, classification, and formatting.

Covers:
- _detect_unknown_activities() — finds unclassified activities, queues triggers, handles errors
- classify_activity() — updates records, handles not-found and missing params
- format_proactive_message() — unknown_activity message format
- _process_user() — integration with unknown activity detection

All DB calls are mocked via unittest.mock.patch.  No real Supabase calls.
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "test-user-unknown-activity"
ACTIVITY_UUID = "act-uuid-1234-5678"


# ---------------------------------------------------------------------------
# Helpers — immutable factory functions
# ---------------------------------------------------------------------------


def _make_unknown_activity(
    activity_id: str = ACTIVITY_UUID,
    activity_type: str = "unknown",
    start_time: str | None = None,
    duration_seconds: int = 2400,
    distance_meters: float | None = 5000.0,
) -> dict:
    """Create a test unknown health_activity dict."""
    now = datetime.now(timezone.utc)
    return {
        "id": activity_id,
        "activity_type": activity_type,
        "start_time": start_time or (now - timedelta(hours=3)).isoformat(),
        "duration_seconds": duration_seconds,
        "distance_meters": distance_meters,
    }


def _make_async_chain(rows: list[dict]) -> MagicMock:
    """Return a mock async query chain whose (await .execute()).data == *rows*.

    All intermediate query-builder methods return the same chain so any
    call order works.
    """
    chain = MagicMock()
    result = MagicMock()
    result.data = rows

    for method_name in [
        "select", "eq", "gte", "lt", "order", "limit",
        "neq", "in_", "upsert", "update", "insert", "maybe_single",
    ]:
        getattr(chain, method_name).return_value = chain

    chain.execute = AsyncMock(return_value=result)
    return chain


def _mock_async_supabase(table_chains: dict[str, MagicMock] | None = None) -> AsyncMock:
    """Return a mock async Supabase client."""
    client = AsyncMock()
    table_chains = table_chains or {}

    def _table(name: str) -> MagicMock:
        return table_chains.get(name, _make_async_chain([]))

    client.table = MagicMock(side_effect=_table)
    return client


def _make_settings(user_id: str = USER_ID) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = user_id
    return s


def _register_analysis_tools():
    """Register analysis tools and return the registry."""
    from src.agent.tools.analysis_tools import register_analysis_tools
    from src.agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_analysis_tools(registry)
    return registry


# ---------------------------------------------------------------------------
# Tests: _detect_unknown_activities
# ---------------------------------------------------------------------------


class TestDetectUnknownFindsUnclassified(unittest.TestCase):
    """Mock Supabase to return unknown activities, verify queue call."""

    def test_detect_unknown_finds_unclassified(self) -> None:
        from src.services.heartbeat import _detect_unknown_activities

        unknown_act = _make_unknown_activity()
        health_chain = _make_async_chain([unknown_act])
        mock_client = _mock_async_supabase({"health_activities": health_chain})

        queued_msg = {"id": "msg-1", "trigger_type": "unknown_activity"}

        async def _run() -> list[dict]:
            with (
                patch("src.db.client.get_async_supabase", return_value=mock_client),
                patch(
                    "src.agent.proactive.queue_proactive_message",
                    return_value=queued_msg,
                ) as mock_queue,
            ):
                result = await _detect_unknown_activities(USER_ID)
                # Verify queue was called
                mock_queue.assert_called_once()
                call_kwargs = mock_queue.call_args
                self.assertEqual(call_kwargs.kwargs["user_id"], USER_ID)
                self.assertEqual(
                    call_kwargs.kwargs["trigger"]["type"], "unknown_activity"
                )
                return result

        result = asyncio.run(_run())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["trigger_type"], "unknown_activity")


class TestDetectUnknownIgnoresClassified(unittest.TestCase):
    """Mock Supabase returning empty (no unknown), verify no queue."""

    def test_detect_unknown_ignores_classified(self) -> None:
        from src.services.heartbeat import _detect_unknown_activities

        health_chain = _make_async_chain([])
        mock_client = _mock_async_supabase({"health_activities": health_chain})

        async def _run() -> list[dict]:
            with patch(
                "src.db.client.get_async_supabase", return_value=mock_client
            ):
                return await _detect_unknown_activities(USER_ID)

        result = asyncio.run(_run())
        self.assertEqual(result, [])


class TestDetectUnknownHandlesDbError(unittest.TestCase):
    """Supabase throws, returns empty list."""

    def test_detect_unknown_handles_db_error(self) -> None:
        from src.services.heartbeat import _detect_unknown_activities

        async def _run() -> list[dict]:
            with patch(
                "src.db.client.get_async_supabase",
                side_effect=Exception("DB connection failed"),
            ):
                return await _detect_unknown_activities(USER_ID)

        result = asyncio.run(_run())
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Tests: classify_activity
# ---------------------------------------------------------------------------


class TestClassifyActivityUpdatesRecord(unittest.TestCase):
    """Mock Supabase update, verify correct call."""

    def test_classify_activity_updates_record(self) -> None:
        registry = _register_analysis_tools()
        classify_tool = registry._tools["classify_activity"]
        self.assertIsNotNone(classify_tool)

        mock_result = MagicMock()
        mock_result.data = [{"id": ACTIVITY_UUID, "activity_type": "running"}]

        mock_table = MagicMock()
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = mock_result

        mock_client = MagicMock()
        mock_client.table.return_value = mock_table

        with (
            patch("src.config.get_settings", return_value=_make_settings()),
            patch("src.db.client.get_supabase", return_value=mock_client),
        ):
            result = classify_tool.handler(activity_id=ACTIVITY_UUID, sport="Running")

        self.assertEqual(result["status"], "classified")
        self.assertEqual(result["activity_id"], ACTIVITY_UUID)
        self.assertEqual(result["sport"], "running")
        mock_table.update.assert_called_once_with({"activity_type": "running"})


class TestClassifyActivityNotFound(unittest.TestCase):
    """Mock empty result, returns not_found."""

    def test_classify_activity_not_found(self) -> None:
        registry = _register_analysis_tools()
        classify_tool = registry._tools["classify_activity"]

        mock_result = MagicMock()
        mock_result.data = []

        mock_table = MagicMock()
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = mock_result

        mock_client = MagicMock()
        mock_client.table.return_value = mock_table

        with (
            patch("src.config.get_settings", return_value=_make_settings()),
            patch("src.db.client.get_supabase", return_value=mock_client),
        ):
            result = classify_tool.handler(
                activity_id="nonexistent-id", sport="cycling"
            )

        self.assertEqual(result["status"], "not_found")
        self.assertIn("not found", result["message"].lower())


class TestClassifyActivityMissingParams(unittest.TestCase):
    """Empty activity_id/sport returns error."""

    def test_classify_activity_missing_activity_id(self) -> None:
        registry = _register_analysis_tools()
        classify_tool = registry._tools["classify_activity"]

        with patch("src.config.get_settings", return_value=_make_settings()):
            result = classify_tool.handler(activity_id="", sport="running")

        self.assertEqual(result["status"], "error")
        self.assertIn("required", result["message"].lower())

    def test_classify_activity_missing_sport(self) -> None:
        registry = _register_analysis_tools()
        classify_tool = registry._tools["classify_activity"]

        with patch("src.config.get_settings", return_value=_make_settings()):
            result = classify_tool.handler(activity_id=ACTIVITY_UUID, sport="")

        self.assertEqual(result["status"], "error")
        self.assertIn("required", result["message"].lower())


# ---------------------------------------------------------------------------
# Tests: format_proactive_message — unknown_activity
# ---------------------------------------------------------------------------


class TestFormatUnknownActivityMessage(unittest.TestCase):
    """Verify message format includes time and duration."""

    def test_format_unknown_activity_message(self) -> None:
        from src.agent.proactive import format_proactive_message

        trigger = {
            "type": "unknown_activity",
            "priority": "medium",
            "data": {
                "activity_id": ACTIVITY_UUID,
                "start_time": "2026-03-04T10:00:00+00:00",
                "duration_minutes": 40,
                "distance_meters": 5000.0,
            },
        }

        msg = format_proactive_message(trigger, {})

        self.assertIn("2026-03-04T10:00:00+00:00", msg)
        self.assertIn("40", msg)
        self.assertIn("unclassified", msg.lower())
        self.assertIn("sport", msg.lower())


class TestFormatUnknownActivityMessageMissingData(unittest.TestCase):
    """Graceful with missing fields."""

    def test_format_unknown_activity_message_missing_data(self) -> None:
        from src.agent.proactive import format_proactive_message

        trigger = {
            "type": "unknown_activity",
            "priority": "medium",
            "data": {},
        }

        msg = format_proactive_message(trigger, {})

        # Should use defaults ("recently" and "?") and not crash
        self.assertIn("recently", msg)
        self.assertIn("?", msg)
        self.assertIn("unclassified", msg.lower())


# ---------------------------------------------------------------------------
# Tests: _process_user calls _detect_unknown_activities
# ---------------------------------------------------------------------------


class TestHeartbeatCallsDetectUnknown(unittest.TestCase):
    """Mock _detect_unknown_activities, verify it's called from _process_user."""

    def test_heartbeat_calls_detect_unknown(self) -> None:
        from src.services.heartbeat import _process_user

        async def _run() -> None:
            with (
                patch(
                    "src.services.heartbeat._try_acquire_lock",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
                patch(
                    "src.services.heartbeat._release_lock",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.heartbeat._check_triggers_for_user",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
                patch(
                    "src.services.heartbeat._check_silence_triggers",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
                patch(
                    "src.services.heartbeat._detect_unknown_activities",
                    new_callable=AsyncMock,
                    return_value=[],
                ) as mock_detect,
            ):
                await _process_user(USER_ID)
                mock_detect.assert_called_once_with(USER_ID)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
