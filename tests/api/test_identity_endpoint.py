"""Tests for the GET /profile/identity endpoint.

Route under test:
    GET /profile/identity - returns the structured "Wie Athletly mich sieht" payload.

Strategy:
    - Patch the journal reader and UserModelDB.load_or_create so no real
      Supabase calls are made.
    - Use the shared conftest auth fixtures (valid_jwt, test_client).
"""

import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import Settings

JWT_SECRET = "test-identity-secret-minimum-32-bytes-for-hs256-compliance!"
TEST_USER_ID = "identity-test-user-uuid-001"


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
        "email": "identity@test.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": now,
        "exp": now + offset,
    }
    return pyjwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def identity_app():
    """FastAPI app with settings patched. Data sources mocked per-test."""
    test_settings = _make_test_settings()

    with patch("src.config.get_settings", return_value=test_settings):
        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch(
                "src.api.routers.chat.get_settings", return_value=test_settings
            ):
                from src.api.main import create_app
                app = create_app()
    return app


@pytest_asyncio.fixture
async def identity_client(identity_app):
    transport = ASGITransport(app=identity_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


def _full_journal() -> str:
    return (
        "# Roman\n\n"
        "## Identity\n\n"
        "27 Jahre, Karlsruhe, Laeufer.\n\n"
        "## Current Goal\n\n"
        "Karlsruher Halbmarathon 20.09.2026, Ziel 1:24:00.\n\n"
        "## Goal Timeline\n\n"
        "- 2026-03-01 wechsel von Marathon zu Halbmarathon\n\n"
        "## What I know about my training\n\n"
        "Threshold 4:10 min/km, VO2max etwa 56.\n\n"
        "## Preferences\n\n"
        "Lange Laeufe am Sonntag.\n\n"
        "## Open Threads\n\n"
        "- Knie beobachten\n\n"
        "## What we tried\n\n"
        "- Drei Tempo-Einheiten pro Woche waren zuviel.\n"
    )


def _full_profile() -> dict:
    return {
        "name": "Roman",
        "sports": ["running", "cycling"],
        "goal": {
            "event": "Karlsruher Halbmarathon",
            "target_date": "2026-09-20",
            "target_time": "1:24:00",
        },
        "fitness": {
            "estimated_vo2max": 56,
            "threshold_pace_min_km": 4.1,
            "weekly_volume_km": 60,
            "ftp_watts": None,
            "trend": "improving",
        },
        "constraints": {
            "training_days_per_week": 5,
            "max_session_minutes": 90,
            "available_sports": ["running", "cycling"],
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-05-14T08:00:00+00:00",
    }


def _empty_profile() -> dict:
    return {
        "name": "Athlete",
        "sports": [],
        "goal": {"event": None, "target_date": None, "target_time": None},
        "fitness": {
            "estimated_vo2max": None,
            "threshold_pace_min_km": None,
            "weekly_volume_km": None,
            "trend": "unknown",
        },
        "constraints": {
            "training_days_per_week": 5,
            "max_session_minutes": 90,
            "available_sports": [],
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _build_supabase_mock(updated_at_value: str | None) -> MagicMock:
    """Build a chainable Supabase client mock for the athlete_journal lookup."""
    execute_result = MagicMock()
    execute_result.data = (
        [{"updated_at": updated_at_value}] if updated_at_value is not None else []
    )

    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = execute_result

    client = MagicMock()
    client.table.return_value = chain
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetIdentity:
    """End-to-end tests for GET /profile/identity."""

    @pytest.mark.asyncio
    async def test_full_journal_and_structured(
        self, identity_client: AsyncClient
    ) -> None:
        """All journal sections + structured fields are populated."""
        token = _make_token()
        test_settings = _make_test_settings()

        mock_model = MagicMock()
        mock_model.project_profile.return_value = _full_profile()

        supabase = _build_supabase_mock("2026-05-14T08:00:00+00:00")

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch(
                "src.agent.athlete_journal.read_journal",
                return_value=_full_journal(),
            ):
                with patch(
                    "src.db.user_model_db.UserModelDB.load_or_create",
                    return_value=mock_model,
                ):
                    with patch(
                        "src.api.routers.profile.get_supabase",
                        return_value=supabase,
                    ):
                        response = await identity_client.get(
                            "/profile/identity",
                            headers={"Authorization": f"Bearer {token}"},
                        )

        assert response.status_code == 200, response.text
        body = response.json()

        assert body["athlete_name"] == "Roman"
        assert body["last_updated_at"] == "2026-05-14T08:00:00+00:00"

        # Sections: stable canonical order, all populated.
        keys = [s["key"] for s in body["sections"]]
        assert keys == [
            "identity",
            "current_goal",
            "goal_timeline",
            "training",
            "preferences",
            "open_threads",
            "what_we_tried",
        ]
        for section in body["sections"]:
            assert section["is_empty"] is False
            assert section["body"].strip() != ""

        # Identity section has the German title.
        identity_section = next(
            s for s in body["sections"] if s["key"] == "identity"
        )
        assert identity_section["title"] == "Identität"

        structured = body["structured"]
        assert structured["sports"] == ["running", "cycling"]
        assert structured["training_days_per_week"] == 5
        assert structured["max_session_minutes"] == 90
        assert structured["fitness"]["vo2max_estimate"] == 56
        assert structured["fitness"]["threshold_pace_min_km"] == 4.1
        assert structured["fitness"]["weekly_volume_km"] == 60
        assert structured["fitness"]["ftp_watts"] is None
        assert structured["goal"]["event"] == "Karlsruher Halbmarathon"
        assert structured["goal"]["target_date"] == "2026-09-20"
        assert structured["goal"]["target_time"] == "1:24:00"

    @pytest.mark.asyncio
    async def test_empty_journal_returns_empty_sections(
        self, identity_client: AsyncClient
    ) -> None:
        """An athlete without a journal still gets all sections as empty."""
        token = _make_token()
        test_settings = _make_test_settings()

        mock_model = MagicMock()
        mock_model.project_profile.return_value = _empty_profile()

        supabase = _build_supabase_mock(None)

        with patch("src.api.auth.get_settings", return_value=test_settings):
            with patch(
                "src.agent.athlete_journal.read_journal", return_value=""
            ):
                with patch(
                    "src.db.user_model_db.UserModelDB.load_or_create",
                    return_value=mock_model,
                ):
                    with patch(
                        "src.api.routers.profile.get_supabase",
                        return_value=supabase,
                    ):
                        response = await identity_client.get(
                            "/profile/identity",
                            headers={"Authorization": f"Bearer {token}"},
                        )

        assert response.status_code == 200, response.text
        body = response.json()

        assert body["athlete_name"] is None
        assert body["last_updated_at"] is None

        # Every canonical section is still emitted, marked empty.
        assert len(body["sections"]) == 7
        for section in body["sections"]:
            assert section["is_empty"] is True
            assert section["body"] == ""

        structured = body["structured"]
        assert structured["sports"] == []
        assert structured["goal"]["event"] is None
        assert structured["fitness"]["vo2max_estimate"] is None

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(
        self, identity_client: AsyncClient
    ) -> None:
        """No Authorization header returns 401 from the JWT dependency."""
        test_settings = _make_test_settings()
        with patch("src.api.auth.get_settings", return_value=test_settings):
            response = await identity_client.get("/profile/identity")
        assert response.status_code == 401
