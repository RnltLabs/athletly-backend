"""Shared fixtures for the API test suite.

Provides:
- test_app: a FastAPI app with lifespan bypassed and settings monkeypatched
- test_client: an httpx.AsyncClient backed by ASGI transport
- jwt_secret: the shared HS256 test secret
- valid_jwt: a freshly-signed JWT accepted by verify_jwt
- expired_jwt: a JWT whose exp is in the past
"""

import time
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JWT_SECRET = "test-super-secret-jwt-key-for-testing-only-minimum-32-bytes"
TEST_USER_ID = "user-uuid-test-1234"
TEST_USER_EMAIL = "coach@athletly.test"


# ---------------------------------------------------------------------------
# Settings override
# ---------------------------------------------------------------------------


def _make_test_settings() -> Settings:
    """Return a Settings instance with known test values (no .env loaded)."""
    return Settings(
        supabase_jwt_secret=JWT_SECRET,
        redis_url="redis://localhost:6379",
        cors_origins="*",
        heartbeat_interval_seconds=99999,  # effectively disabled
        gemini_api_key="test-gemini-key",
        supabase_url="https://test.supabase.co",
        supabase_anon_key="test-anon-key",
        supabase_service_role_key="test-service-role-key",
    )


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jwt_secret() -> str:
    return JWT_SECRET


@pytest.fixture
def valid_jwt(jwt_secret: str) -> str:
    """A valid HS256 JWT with the Supabase `authenticated` audience."""
    now = int(time.time())
    payload = {
        "sub": TEST_USER_ID,
        "email": TEST_USER_EMAIL,
        "role": "authenticated",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
    }
    return pyjwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def expired_jwt(jwt_secret: str) -> str:
    """A JWT whose exp is already in the past."""
    past = int(time.time()) - 7200
    payload = {
        "sub": TEST_USER_ID,
        "email": TEST_USER_EMAIL,
        "role": "authenticated",
        "aud": "authenticated",
        "iat": past - 3600,
        "exp": past,
    }
    return pyjwt.encode(payload, jwt_secret, algorithm="HS256")


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app(monkeypatch) -> FastAPI:
    """FastAPI test app with lifespan bypassed and settings monkeypatched."""
    # Override cached settings so the real .env is never consulted.
    test_settings = _make_test_settings()
    monkeypatch.setattr("src.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("src.api.auth.get_settings", lambda: test_settings)
    monkeypatch.setattr("src.api.routers.chat.get_settings", lambda: test_settings)

    # Patch HeartbeatService so lifespan does not attempt real Supabase/Redis.
    mock_heartbeat = MagicMock()
    mock_heartbeat.start = AsyncMock()
    mock_heartbeat.stop = AsyncMock()

    with patch("src.services.heartbeat.HeartbeatService", return_value=mock_heartbeat):
        from src.api.main import create_app

        app = create_app()

    return app


@pytest_asyncio.fixture
async def test_client(test_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """httpx.AsyncClient using ASGI transport — no real network calls."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
