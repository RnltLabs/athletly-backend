"""Tests for the chat SSE endpoint and confirm endpoint.

Routes under test:
    POST /chat          — returns an SSE event stream
    POST /chat/confirm  — stores a user confirmation in Redis (or in-process)

Strategy:
    - AsyncAgentLoop, UserModelDB, and Redis are fully mocked.
    - httpx.AsyncClient with ASGITransport drives the FastAPI app.
    - SSE frames are collected by reading the raw response bytes.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import Settings

# ---------------------------------------------------------------------------
# Constants (duplicated locally so these tests are self-contained)
# ---------------------------------------------------------------------------

JWT_SECRET = "test-chat-endpoint-secret-minimum-32-bytes-for-hs256-compliance"
TEST_USER_ID = "chat-test-user-uuid-999"
TEST_SESSION_ID = "session-abc-123"


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


def _make_token(secret: str = JWT_SECRET, offset: int = 3600) -> str:
    now = int(time.time())
    payload = {
        "sub": TEST_USER_ID,
        "email": "chat@test.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": now,
        "exp": now + offset,
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
# App fixture (local, not from conftest, to isolate patch scope)
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_app():
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
async def chat_client(chat_app):
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(chat_client: AsyncClient) -> None:
    """GET /health returns status=ok and a version string."""
    response = await chat_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------------------------------------------------------------------------
# Authentication guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_requires_auth_no_header(chat_client: AsyncClient) -> None:
    """POST /chat without an Authorization header returns 401."""
    response = await chat_client.post(
        "/chat",
        json={"message": "Hello coach"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_requires_auth_wrong_secret(chat_client: AsyncClient) -> None:
    """POST /chat with a token signed by a wrong secret returns 401."""
    bad_token = _make_token(secret="totally-wrong-secret")
    response = await chat_client.post(
        "/chat",
        json={"message": "Hello coach"},
        headers={"Authorization": f"Bearer {bad_token}"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_requires_auth_expired_token(chat_client: AsyncClient) -> None:
    """POST /chat with an expired token returns 401."""
    expired = _make_token(offset=-3600)
    response = await chat_client.post(
        "/chat",
        json={"message": "Hello coach"},
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


def _make_mock_agent_loop(events: list[tuple[str, dict]]):
    """Return a mock AsyncAgentLoop that emits the given events then a message."""
    from src.agent.agent_loop import AgentResult

    async def _mock_process_message_sse(user_message: str, emit_fn) -> AgentResult:
        for event_type, data in events:
            await emit_fn(event_type, data)
        await emit_fn("message", {"text": "Here is your coaching reply."})
        return AgentResult(response_text="Here is your coaching reply.")

    mock_loop = MagicMock()
    mock_loop.start_session.return_value = TEST_SESSION_ID
    mock_loop.process_message_sse = _mock_process_message_sse
    return mock_loop


@pytest.mark.asyncio
async def test_chat_returns_sse_stream(chat_client: AsyncClient) -> None:
    """POST /chat with a valid token streams thinking, tool_hint, message, done events."""
    token = _make_token()
    test_settings = _make_test_settings()

    mock_loop = _make_mock_agent_loop([
        ("thinking", {"text": "Analyzing request..."}),
        ("tool_hint", {"name": "get_activities", "args": {"limit": 5}}),
    ])

    mock_user_model = MagicMock()

    with patch("src.api.auth.get_settings", return_value=test_settings):
        with patch("src.api.routers.chat.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                with patch("src.api.routers.chat.AsyncAgentLoop", return_value=mock_loop):
                    with patch(
                        "src.api.routers.chat._load_user_model",
                        new=AsyncMock(return_value=mock_user_model),
                    ):
                        response = await chat_client.post(
                            "/chat",
                            json={"message": "Give me a training plan", "session_id": TEST_SESSION_ID},
                            headers={"Authorization": f"Bearer {token}"},
                        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse_events(response.content)
    event_names = [e["event"] for e in events]

    assert "thinking" in event_names
    assert "tool_hint" in event_names
    assert "message" in event_names
    assert "done" in event_names

    # Verify message content
    message_evt = next(e for e in events if e["event"] == "message")
    assert message_evt["data"]["text"] == "Here is your coaching reply."

    # done must be last
    assert event_names[-1] == "done"


@pytest.mark.asyncio
async def test_chat_error_not_persisted(chat_client: AsyncClient) -> None:
    """When the agent raises, an error SSE event is emitted and done follows — no crash."""
    token = _make_token()
    test_settings = _make_test_settings()

    async def _failing_process(user_message: str, emit_fn):
        raise RuntimeError("Simulated agent failure")

    mock_loop = MagicMock()
    mock_loop.start_session.return_value = TEST_SESSION_ID
    mock_loop.process_message_sse = _failing_process

    mock_user_model = MagicMock()

    with patch("src.api.auth.get_settings", return_value=test_settings):
        with patch("src.api.routers.chat.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                with patch("src.api.routers.chat.AsyncAgentLoop", return_value=mock_loop):
                    with patch(
                        "src.api.routers.chat._load_user_model",
                        new=AsyncMock(return_value=mock_user_model),
                    ):
                        response = await chat_client.post(
                            "/chat",
                            json={"message": "Trigger failure"},
                            headers={"Authorization": f"Bearer {token}"},
                        )

    assert response.status_code == 200  # SSE transport stays 200
    events = _parse_sse_events(response.content)
    event_names = [e["event"] for e in events]

    assert "error" in event_names
    assert "done" in event_names
    assert event_names[-1] == "done"

    error_evt = next(e for e in events if e["event"] == "error")
    assert "code" in error_evt["data"]


# ---------------------------------------------------------------------------
# /chat/confirm endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_endpoint_with_redis(chat_client: AsyncClient) -> None:
    """POST /chat/confirm stores a confirmation in Redis and returns status=ok."""
    token = _make_token()
    test_settings = _make_test_settings()

    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(return_value=True)

    with patch("src.api.auth.get_settings", return_value=test_settings):
        with patch("src.api.routers.chat.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=mock_redis)):
                response = await chat_client.post(
                    "/chat/confirm",
                    json={
                        "session_id": TEST_SESSION_ID,
                        "action_id": "overwrite-plan-001",
                        "confirmed": True,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"

    # Verify Redis.set was called with the expected key pattern.
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    key_arg = call_args[0][0]
    assert key_arg == f"confirm:{TEST_SESSION_ID}:overwrite-plan-001"


@pytest.mark.asyncio
async def test_confirm_endpoint_without_redis(chat_client: AsyncClient) -> None:
    """POST /chat/confirm falls back gracefully when Redis is unavailable."""
    token = _make_token()
    test_settings = _make_test_settings()

    with patch("src.api.auth.get_settings", return_value=test_settings):
        with patch("src.api.routers.chat.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=None)):
                response = await chat_client.post(
                    "/chat/confirm",
                    json={
                        "session_id": TEST_SESSION_ID,
                        "action_id": "plan-action-002",
                        "confirmed": False,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )

    # Should still return 200 using in-process fallback.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_confirm_requires_auth(chat_client: AsyncClient) -> None:
    """POST /chat/confirm without a token returns 401."""
    response = await chat_client.post(
        "/chat/confirm",
        json={
            "session_id": "s1",
            "action_id": "a1",
            "confirmed": True,
        },
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Concurrent request lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_request_returns_error_event(chat_client: AsyncClient) -> None:
    """When a user lock is held, the SSE stream immediately emits a concurrent_request error."""
    token = _make_token()
    test_settings = _make_test_settings()

    # Simulate a Redis lock already held by returning None for acquire (NX fails).
    mock_redis = AsyncMock()
    # set with nx=True returns None when key already exists.
    mock_redis.set = AsyncMock(return_value=None)

    with patch("src.api.auth.get_settings", return_value=test_settings):
        with patch("src.api.routers.chat.get_settings", return_value=test_settings):
            with patch("src.api.routers.chat._get_redis", new=AsyncMock(return_value=mock_redis)):
                response = await chat_client.post(
                    "/chat",
                    json={"message": "Hello again"},
                    headers={"Authorization": f"Bearer {token}"},
                )

    assert response.status_code == 200
    events = _parse_sse_events(response.content)
    event_names = [e["event"] for e in events]

    assert "error" in event_names
    error_evt = next(e for e in events if e["event"] == "error")
    assert error_evt["data"]["code"] == "concurrent_request"
    assert "done" in event_names
