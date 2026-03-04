"""Tests for notification tools — send_notification + spawn_background_task.

Covers:
- Tool registration and schema
- send_notification: token lookup, Expo API call, error handling
- spawn_background_task: task ID generation, async/sync context
- Helper functions: payload building, push token lookup
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.agent.tools.notification_tools import (
    _build_expo_payload,
    _post_to_expo,
    send_notification_async,
)
from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

USER_ID = "test-user-456"


def _make_registry() -> ToolRegistry:
    """Register notification tools and return the registry."""
    registry = ToolRegistry()
    from src.agent.tools.notification_tools import register_notification_tools

    register_notification_tools(registry, USER_ID)
    return registry


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Verify tools are registered with correct names and schemas."""

    def test_send_notification_registered(self) -> None:
        registry = _make_registry()
        tools = registry.get_openai_tools()
        names = [t["function"]["name"] for t in tools]
        assert "send_notification" in names

    def test_spawn_background_task_registered(self) -> None:
        registry = _make_registry()
        tools = registry.get_openai_tools()
        names = [t["function"]["name"] for t in tools]
        assert "spawn_background_task" in names

    def test_send_notification_required_params(self) -> None:
        registry = _make_registry()
        tools = registry.get_openai_tools()
        tool = next(t for t in tools if t["function"]["name"] == "send_notification")
        assert set(tool["function"]["parameters"]["required"]) == {"title", "body"}

    def test_spawn_background_task_required_params(self) -> None:
        registry = _make_registry()
        tools = registry.get_openai_tools()
        tool = next(t for t in tools if t["function"]["name"] == "spawn_background_task")
        assert tool["function"]["parameters"]["required"] == ["instruction"]


# ---------------------------------------------------------------------------
# _build_expo_payload tests
# ---------------------------------------------------------------------------


class TestBuildExpoPayload:
    """Test Expo push payload construction (pure function)."""

    def test_basic_payload(self) -> None:
        result = _build_expo_payload("ExponentPushToken[abc]", "Title", "Body", None)
        assert result == {
            "to": "ExponentPushToken[abc]",
            "title": "Title",
            "body": "Body",
            "sound": "default",
        }

    def test_payload_with_data(self) -> None:
        result = _build_expo_payload(
            "ExponentPushToken[abc]", "T", "B", {"screen": "coach"}
        )
        assert result["data"] == {"screen": "coach"}
        assert result["to"] == "ExponentPushToken[abc]"

    def test_payload_immutability(self) -> None:
        """Calling twice returns independent dicts."""
        a = _build_expo_payload("tok1", "A", "B", None)
        b = _build_expo_payload("tok2", "C", "D", {"x": 1})
        assert a["to"] != b["to"]
        assert "data" not in a
        assert "data" in b


# ---------------------------------------------------------------------------
# send_notification_async tests
# ---------------------------------------------------------------------------


class TestSendNotificationAsync:
    """Test the async notification sending pipeline."""

    @pytest.mark.asyncio
    async def test_no_push_token_skips(self) -> None:
        """When no push token exists, returns graceful no-op."""
        with patch(
            "src.agent.tools.notification_tools._lookup_push_token",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await send_notification_async(USER_ID, "Hi", "Body")
            assert result["sent"] is False
            assert result["reason"] == "no_push_token"

    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        """When token exists and Expo API succeeds."""
        with (
            patch(
                "src.agent.tools.notification_tools._lookup_push_token",
                new_callable=AsyncMock,
                return_value="ExponentPushToken[xyz]",
            ),
            patch(
                "src.agent.tools.notification_tools._post_to_expo",
                new_callable=AsyncMock,
                return_value={"sent": True, "ticket": {"id": "ticket-1"}},
            ) as mock_post,
        ):
            result = await send_notification_async(USER_ID, "Title", "Body", {"k": "v"})
            assert result["sent"] is True
            # Verify payload was built correctly
            call_args = mock_post.call_args[0][0]
            assert call_args["to"] == "ExponentPushToken[xyz]"
            assert call_args["title"] == "Title"
            assert call_args["data"] == {"k": "v"}


# ---------------------------------------------------------------------------
# _post_to_expo tests
# ---------------------------------------------------------------------------


class TestPostToExpo:
    """Test the Expo Push API HTTP call."""

    @pytest.mark.asyncio
    async def test_successful_post(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"id": "receipt-abc"}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _post_to_expo({"to": "tok", "title": "T", "body": "B"})
            assert result["sent"] is True
            assert result["ticket"] == {"id": "receipt-abc"}

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            resp = MagicMock()
            resp.status_code = 429
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "rate limited", request=MagicMock(), response=resp
            )
            instance.post.return_value = resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _post_to_expo({"to": "tok", "title": "T", "body": "B"})
            assert result["sent"] is False
            assert "429" in result["error"]

    @pytest.mark.asyncio
    async def test_network_error(self) -> None:
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _post_to_expo({"to": "tok", "title": "T", "body": "B"})
            assert result["sent"] is False
            assert "error" in result


# ---------------------------------------------------------------------------
# send_notification tool (sync wrapper) tests
# ---------------------------------------------------------------------------


class TestSendNotificationTool:
    """Test the sync tool handler via registry.execute()."""

    def test_returns_sent_in_turn_flag(self) -> None:
        """Result always includes _sent_in_turn=True."""
        registry = _make_registry()
        with (
            patch(
                "src.agent.tools.notification_tools._lookup_push_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = registry.execute(
                "send_notification", {"title": "Hi", "body": "Test"}
            )
            assert result["_sent_in_turn"] is True

    def test_missing_required_params(self) -> None:
        """Missing 'body' returns error from registry."""
        registry = _make_registry()
        result = registry.execute("send_notification", {"title": "Hi"})
        # Registry catches TypeError for missing args
        assert "error" in result


# ---------------------------------------------------------------------------
# spawn_background_task tests
# ---------------------------------------------------------------------------


class TestSpawnBackgroundTask:
    """Test the background task spawning tool."""

    def test_returns_task_id(self) -> None:
        registry = _make_registry()
        result = registry.execute(
            "spawn_background_task", {"instruction": "Analyze last week"}
        )
        assert result["spawned"] is True
        assert result["task_id"].startswith("bg_")

    def test_task_id_unique(self) -> None:
        registry = _make_registry()
        r1 = registry.execute(
            "spawn_background_task", {"instruction": "Task 1"}
        )
        r2 = registry.execute(
            "spawn_background_task", {"instruction": "Task 2"}
        )
        assert r1["task_id"] != r2["task_id"]

    def test_missing_instruction(self) -> None:
        registry = _make_registry()
        result = registry.execute("spawn_background_task", {})
        assert "error" in result
