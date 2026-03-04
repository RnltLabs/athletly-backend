"""Tests for Phase 3 onboarding API + SSE + tool integration.

Tests cover:
    - ChatRequest context field validation
    - SSE onboarding_complete event emission
    - complete_onboarding tool logic
    - Context forwarding through the full chain
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.agent.agent_loop import AgentResult
from src.api.sse import SSEEmitter
from src.config import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JWT_SECRET = "test-onboarding-secret-minimum-32-bytes-for-hs256-compliance"
TEST_USER_ID = "onboarding-test-user-uuid-001"
TEST_SESSION_ID = "onboarding-session-abc-123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_settings() -> Settings:
    return Settings(
        supabase_jwt_secret=JWT_SECRET,
        redis_url="redis://localhost:6379",
        cors_origins="*",
        heartbeat_interval_seconds=99999,
    )


def _make_token(secret: str = JWT_SECRET) -> str:
    now = int(time.time())
    payload = {
        "sub": TEST_USER_ID,
        "email": "onboarding@test.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    return pyjwt.encode(payload, secret, algorithm="HS256")


def _parse_sse_events(raw: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of {'event': str, 'data': dict} dicts."""
    events: list[dict] = []
    current_event: dict = {}
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            try:
                current_event["data"] = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                current_event["data"] = line[len("data:"):].strip()
        elif line == "" and current_event:
            events.append(current_event)
            current_event = {}
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def onboarding_app():
    """FastAPI app with all external dependencies mocked."""
    test_settings = _make_test_settings()

    mock_heartbeat = MagicMock()
    mock_heartbeat.start = AsyncMock()
    mock_heartbeat.stop = AsyncMock()

    with patch("src.services.heartbeat.HeartbeatService", return_value=mock_heartbeat):
        with patch("src.config.get_settings", return_value=test_settings):
            with patch("src.api.auth.get_settings", return_value=test_settings):
                with patch("src.api.routers.chat.get_settings", return_value=test_settings):
                    from src.api.main import create_app
                    app = create_app()

    return app


@pytest_asyncio.fixture
async def onboarding_client(onboarding_app):
    transport = ASGITransport(app=onboarding_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ---------------------------------------------------------------------------
# ChatRequest context field validation
# ---------------------------------------------------------------------------


class TestChatRequestContext:
    """Test that the context field validates correctly."""

    @pytest.mark.asyncio
    async def test_context_default_is_coach(self, onboarding_client: AsyncClient) -> None:
        """POST /chat without context field defaults to 'coach'."""
        token = _make_token()
        test_settings = _make_test_settings()

        mock_loop = MagicMock()
        mock_loop.start_session.return_value = TEST_SESSION_ID

        async def _mock_process(user_message: str, emit_fn):
            await emit_fn("message", {"text": "Hello!"})
            return AgentResult(response_text="Hello!")

        mock_loop.process_message_sse = _mock_process
        mock_user_model = MagicMock()

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat.get_settings", return_value=test_settings):
                with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                    with patch(
                        "src.api.routers.chat.AsyncAgentLoop",
                        return_value=mock_loop,
                    ) as mock_cls:
                        with patch(
                            "src.api.routers.chat._load_user_model",
                            new=AsyncMock(return_value=mock_user_model),
                        ):
                            response = await onboarding_client.post(
                                "/chat",
                                json={"message": "Hello"},
                                headers={"Authorization": f"Bearer {token}"},
                            )

        assert response.status_code == 200
        mock_cls.assert_called_once_with(user_model=mock_user_model, context="coach")

    @pytest.mark.asyncio
    async def test_context_onboarding_forwarded(self, onboarding_client: AsyncClient) -> None:
        """POST /chat with context='onboarding' forwards it to AsyncAgentLoop."""
        token = _make_token()
        test_settings = _make_test_settings()

        mock_loop = MagicMock()
        mock_loop.start_session.return_value = TEST_SESSION_ID

        async def _mock_process(user_message: str, emit_fn):
            await emit_fn("message", {"text": "Welcome!"})
            return AgentResult(response_text="Welcome!")

        mock_loop.process_message_sse = _mock_process
        mock_user_model = MagicMock()

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat.get_settings", return_value=test_settings):
                with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                    with patch(
                        "src.api.routers.chat.AsyncAgentLoop",
                        return_value=mock_loop,
                    ) as mock_cls:
                        with patch(
                            "src.api.routers.chat._load_user_model",
                            new=AsyncMock(return_value=mock_user_model),
                        ):
                            response = await onboarding_client.post(
                                "/chat",
                                json={
                                    "message": "Hallo, ich bin neu hier",
                                    "context": "onboarding",
                                },
                                headers={"Authorization": f"Bearer {token}"},
                            )

        assert response.status_code == 200
        mock_cls.assert_called_once_with(user_model=mock_user_model, context="onboarding")

    @pytest.mark.asyncio
    async def test_context_invalid_value_rejected(self, onboarding_client: AsyncClient) -> None:
        """POST /chat with invalid context returns 422."""
        token = _make_token()
        test_settings = _make_test_settings()

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat.get_settings", return_value=test_settings):
                response = await onboarding_client.post(
                    "/chat",
                    json={"message": "Hello", "context": "invalid"},
                    headers={"Authorization": f"Bearer {token}"},
                )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# SSE onboarding_complete event
# ---------------------------------------------------------------------------


class TestOnboardingCompleteSSE:
    """Test that onboarding_complete SSE event is emitted correctly."""

    def test_sse_emitter_onboarding_complete(self) -> None:
        """SSEEmitter.onboarding_complete produces the correct event."""
        evt = SSEEmitter.onboarding_complete()

        assert evt.event == "onboarding_complete"
        data = json.loads(evt.data)
        assert data["onboarding_complete"] is True

    @pytest.mark.asyncio
    async def test_onboarding_complete_emitted_in_stream(
        self, onboarding_client: AsyncClient
    ) -> None:
        """When agent result has onboarding_just_completed=True, SSE stream includes the event."""
        token = _make_token()
        test_settings = _make_test_settings()

        mock_loop = MagicMock()
        mock_loop.start_session.return_value = TEST_SESSION_ID

        async def _mock_process(user_message: str, emit_fn):
            await emit_fn("message", {"text": "Setup complete!"})
            return AgentResult(
                response_text="Setup complete!",
                onboarding_just_completed=True,
            )

        mock_loop.process_message_sse = _mock_process
        mock_user_model = MagicMock()

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat.get_settings", return_value=test_settings):
                with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                    with patch(
                        "src.api.routers.chat.AsyncAgentLoop",
                        return_value=mock_loop,
                    ):
                        with patch(
                            "src.api.routers.chat._load_user_model",
                            new=AsyncMock(return_value=mock_user_model),
                        ):
                            response = await onboarding_client.post(
                                "/chat",
                                json={
                                    "message": "Alles klar!",
                                    "context": "onboarding",
                                },
                                headers={"Authorization": f"Bearer {token}"},
                            )

        assert response.status_code == 200
        events = _parse_sse_events(response.content)
        event_names = [e["event"] for e in events]

        assert "onboarding_complete" in event_names
        assert "message" in event_names
        assert "done" in event_names

        oc_evt = next(e for e in events if e["event"] == "onboarding_complete")
        assert oc_evt["data"]["onboarding_complete"] is True

    @pytest.mark.asyncio
    async def test_no_onboarding_complete_for_coach_mode(
        self, onboarding_client: AsyncClient
    ) -> None:
        """In coach mode with onboarding_just_completed=False, no onboarding_complete event."""
        token = _make_token()
        test_settings = _make_test_settings()

        mock_loop = MagicMock()
        mock_loop.start_session.return_value = TEST_SESSION_ID

        async def _mock_process(user_message: str, emit_fn):
            await emit_fn("message", {"text": "Coach reply."})
            return AgentResult(
                response_text="Coach reply.",
                onboarding_just_completed=False,
            )

        mock_loop.process_message_sse = _mock_process
        mock_user_model = MagicMock()

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat.get_settings", return_value=test_settings):
                with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                    with patch(
                        "src.api.routers.chat.AsyncAgentLoop",
                        return_value=mock_loop,
                    ):
                        with patch(
                            "src.api.routers.chat._load_user_model",
                            new=AsyncMock(return_value=mock_user_model),
                        ):
                            response = await onboarding_client.post(
                                "/chat",
                                json={"message": "How was my week?"},
                                headers={"Authorization": f"Bearer {token}"},
                            )

        events = _parse_sse_events(response.content)
        event_names = [e["event"] for e in events]

        assert "onboarding_complete" not in event_names


# ---------------------------------------------------------------------------
# complete_onboarding tool
# ---------------------------------------------------------------------------


class TestCompleteOnboardingTool:
    """Test the complete_onboarding tool logic."""

    def _make_user_model(
        self,
        sports: list | None = None,
        goal_event: str | None = None,
    ) -> MagicMock:
        profile = {
            "name": "TestUser",
            "sports": sports or [],
            "goal": {"event": goal_event},
            "constraints": {"training_days_per_week": 4, "max_session_minutes": 60},
        }
        mock = MagicMock()
        mock.project_profile.return_value = profile
        mock.user_id = TEST_USER_ID
        mock.meta = {}
        return mock

    def test_complete_onboarding_success(self) -> None:
        """complete_onboarding succeeds when sports and goal are present."""
        user_model = self._make_user_model(
            sports=["running"],
            goal_event="Marathon Berlin 2026",
        )

        with patch("src.agent.tools.onboarding_tools.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                use_supabase=False,
                agenticsports_user_id=TEST_USER_ID,
            )

            from src.agent.tools.onboarding_tools import register_onboarding_tools
            from src.agent.tools.registry import ToolRegistry

            registry = ToolRegistry()
            register_onboarding_tools(registry, user_model)

            result = registry.execute("complete_onboarding", {})

        assert result["status"] == "success"
        assert result["onboarding_complete"] is True
        assert user_model.meta["_onboarding_complete"] is True
        user_model.save.assert_called_once()

    def test_complete_onboarding_missing_sports(self) -> None:
        """complete_onboarding fails when sports are missing."""
        user_model = self._make_user_model(
            sports=[],
            goal_event="Get fit",
        )

        with patch("src.agent.tools.onboarding_tools.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                use_supabase=False,
                agenticsports_user_id=TEST_USER_ID,
            )

            from src.agent.tools.onboarding_tools import register_onboarding_tools
            from src.agent.tools.registry import ToolRegistry

            registry = ToolRegistry()
            register_onboarding_tools(registry, user_model)

            result = registry.execute("complete_onboarding", {})

        assert result["status"] == "error"
        assert "sports" in result["missing"]

    def test_complete_onboarding_missing_goal(self) -> None:
        """complete_onboarding fails when goal is missing."""
        user_model = self._make_user_model(
            sports=["cycling"],
            goal_event=None,
        )

        with patch("src.agent.tools.onboarding_tools.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                use_supabase=False,
                agenticsports_user_id=TEST_USER_ID,
            )

            from src.agent.tools.onboarding_tools import register_onboarding_tools
            from src.agent.tools.registry import ToolRegistry

            registry = ToolRegistry()
            register_onboarding_tools(registry, user_model)

            result = registry.execute("complete_onboarding", {})

        assert result["status"] == "error"
        assert "goal" in result["missing"]

    def test_complete_onboarding_missing_both(self) -> None:
        """complete_onboarding fails when both sports and goal are missing."""
        user_model = self._make_user_model(sports=[], goal_event=None)

        with patch("src.agent.tools.onboarding_tools.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                use_supabase=False,
                agenticsports_user_id=TEST_USER_ID,
            )

            from src.agent.tools.onboarding_tools import register_onboarding_tools
            from src.agent.tools.registry import ToolRegistry

            registry = ToolRegistry()
            register_onboarding_tools(registry, user_model)

            result = registry.execute("complete_onboarding", {})

        assert result["status"] == "error"
        assert "sports" in result["missing"]
        assert "goal" in result["missing"]
