"""Tests for GET /profile/identity/widgets.

The endpoint composes:
    1. journal markdown read (mocked)
    2. structured profile read (mocked)
    3. Redis cache get/set (mocked or fakeredis)
    4. LLM call (mocked)
    5. fallback widget assembly (exercised on LLM failure)

Tests cover the happy path, the cache hit, and the LLM failure fallback.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import Settings

JWT_SECRET = "test-identity-widgets-secret-minimum-32-bytes-for-hs256-pass!"
TEST_USER_ID = "identity-widgets-user-uuid-001"


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
        "email": "widgets@test.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": now,
        "exp": now + offset,
    }
    return pyjwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def widgets_app():
    """FastAPI app with settings patched."""
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
async def widgets_client(widgets_app):
    transport = ASGITransport(app=widgets_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def _sample_journal() -> str:
    return (
        "# Roman\n\n"
        "## Identity\n\n"
        "27 Jahre, Karlsruhe, Laeufer.\n\n"
        "## Current Goal\n\n"
        "Karlsruher Halbmarathon 20.09.2026, Ziel 1:24:00.\n\n"
        "## What I know about my training\n\n"
        "Threshold 4:10 min/km, VO2max etwa 56.\n"
    )


def _sample_profile() -> dict[str, Any]:
    return {
        "name": "Roman",
        "sports": ["running"],
        "goal": {
            "event": "Karlsruher Halbmarathon",
            "target_date": "2026-09-20",
            "target_time": "01:24:00",
        },
        "fitness": {
            "estimated_vo2max": 56,
            "threshold_pace_min_km": 4.1,
            "weekly_volume_km": 60,
            "ftp_watts": None,
        },
        "constraints": {
            "training_days_per_week": 5,
            "max_session_minutes": 90,
        },
    }


def _llm_widgets_payload() -> dict[str, Any]:
    """A representative LLM JSON output."""
    return {
        "widgets": [
            {
                "type": "profile_header",
                "props": {
                    "name": "Roman",
                    "subtitle": "27 Jahre - Karlsruhe",
                    "sport_badges": ["Laufen"],
                },
            },
            {
                "type": "hero_goal",
                "props": {
                    "event": "Karlsruher Halbmarathon",
                    "target_date": "2026-09-20",
                    "target_time": "01:24:00",
                    "pace": "3:58 /km",
                    "source": "athlete journal",
                },
            },
            {
                "type": "stat_grid",
                "props": {
                    "title": "Performance",
                    "stats": [
                        {
                            "label": "Threshold-Pace",
                            "value": "4:10",
                            "unit": "/km",
                            "icon": "timer",
                        },
                        {
                            "label": "VO2max",
                            "value": "56",
                            "unit": "ml/kg/min",
                            "icon": "activity",
                        },
                    ],
                },
            },
        ],
        "edit_hint_per_widget": {
            "0": "Lass uns mein Profil anpassen: ",
            "1": "Lass uns mein Ziel anpassen: ",
            "2": "Lass uns meine Trainingswerte anpassen: ",
        },
    }


def _make_llm_response(content: str) -> MagicMock:
    """Mimic litellm.ModelResponse shape: .choices[0].message.content."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class _FakeAsyncRedis:
    """Minimal in-memory async Redis double for cache hit tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls = 0
        self.get_calls = 0

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        self.get_calls += 1
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.set_calls += 1
        self.store[key] = value

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetIdentityWidgets:
    """End-to-end tests for GET /profile/identity/widgets."""

    @pytest.mark.asyncio
    async def test_returns_valid_widgets_on_llm_success(
        self, widgets_client: AsyncClient
    ) -> None:
        token = _make_token()
        test_settings = _make_test_settings()
        mock_model = MagicMock()
        mock_model.project_profile.return_value = _sample_profile()
        llm_response = _make_llm_response(json.dumps(_llm_widgets_payload()))

        with (
            patch("src.api.auth.get_settings", return_value=test_settings),
            patch(
                "src.agent.athlete_journal.read_journal",
                return_value=_sample_journal(),
            ),
            patch(
                "src.db.user_model_db.UserModelDB.load_or_create",
                return_value=mock_model,
            ),
            patch(
                "src.services.identity_widgets._get_redis",
                AsyncMock(return_value=None),
            ),
            patch(
                "src.agent.llm.chat_completion",
                return_value=llm_response,
            ),
        ):
            response = await widgets_client.get(
                "/profile/identity/widgets",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["cache_hit"] is False
        assert isinstance(body["generated_at"], str) and body["generated_at"]
        assert len(body["widgets"]) == 3

        # Widget shape and ordering preserved.
        types = [w["type"] for w in body["widgets"]]
        assert types == ["profile_header", "hero_goal", "stat_grid"]

        header = body["widgets"][0]
        assert header["props"]["name"] == "Roman"
        assert header["props"]["sport_badges"] == ["Laufen"]

        # Post-processing: leading zero stripped, countdown_days computed.
        hero = body["widgets"][1]
        assert hero["props"]["target_time"] == "1:24:00"
        assert hero["props"]["target_date"] == "2026-09-20"
        assert isinstance(hero["props"]["countdown_days"], int)
        assert hero["props"]["countdown_days"] >= 0

        # Edit hints normalized to int keys (FastAPI serializes back to str
        # in the JSON envelope, but values are intact).
        hints = body["edit_hint_per_widget"]
        assert len(hints) == 3
        for hint in hints.values():
            assert hint.startswith("Lass uns")
            assert "anpassen" in hint

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_call(
        self, widgets_client: AsyncClient
    ) -> None:
        """Second call with identical journal/profile hits the cache."""
        token = _make_token()
        test_settings = _make_test_settings()
        mock_model = MagicMock()
        mock_model.project_profile.return_value = _sample_profile()
        llm_response = _make_llm_response(json.dumps(_llm_widgets_payload()))

        fake_redis = _FakeAsyncRedis()
        # Use a normal AsyncMock that returns our fake.
        async def fake_get_redis():
            return fake_redis

        call_counter = {"n": 0}

        def counting_chat_completion(*args, **kwargs):
            call_counter["n"] += 1
            return llm_response

        with (
            patch("src.api.auth.get_settings", return_value=test_settings),
            patch(
                "src.agent.athlete_journal.read_journal",
                return_value=_sample_journal(),
            ),
            patch(
                "src.db.user_model_db.UserModelDB.load_or_create",
                return_value=mock_model,
            ),
            patch(
                "src.services.identity_widgets._get_redis",
                side_effect=fake_get_redis,
            ),
            patch(
                "src.agent.llm.chat_completion",
                side_effect=counting_chat_completion,
            ),
        ):
            first = await widgets_client.get(
                "/profile/identity/widgets",
                headers={"Authorization": f"Bearer {token}"},
            )
            second = await widgets_client.get(
                "/profile/identity/widgets",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text

        first_body = first.json()
        second_body = second.json()

        assert first_body["cache_hit"] is False
        assert second_body["cache_hit"] is True

        # LLM should have been called exactly once across both requests.
        assert call_counter["n"] == 1
        # Redis got one set (write) and at least two gets (one miss + one hit).
        assert fake_redis.set_calls == 1
        assert fake_redis.get_calls >= 2

        # Widget content identical between calls.
        assert first_body["widgets"] == second_body["widgets"]

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_structured_widgets(
        self, widgets_client: AsyncClient
    ) -> None:
        """When the LLM raises, the service falls back to deterministic widgets."""
        token = _make_token()
        test_settings = _make_test_settings()
        mock_model = MagicMock()
        mock_model.project_profile.return_value = _sample_profile()

        with (
            patch("src.api.auth.get_settings", return_value=test_settings),
            patch(
                "src.agent.athlete_journal.read_journal",
                return_value=_sample_journal(),
            ),
            patch(
                "src.db.user_model_db.UserModelDB.load_or_create",
                return_value=mock_model,
            ),
            patch(
                "src.services.identity_widgets._get_redis",
                AsyncMock(return_value=None),
            ),
            patch(
                "src.agent.llm.chat_completion",
                side_effect=RuntimeError("LLM provider down"),
            ),
        ):
            response = await widgets_client.get(
                "/profile/identity/widgets",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["cache_hit"] is False
        widgets = body["widgets"]
        assert len(widgets) >= 1

        # Fallback ALWAYS emits a profile_header.
        types = [w["type"] for w in widgets]
        assert types[0] == "profile_header"
        assert widgets[0]["props"]["name"] == "Roman"

        # Goal was present in structured profile, so a hero_goal must follow.
        assert "hero_goal" in types
        hero = next(w for w in widgets if w["type"] == "hero_goal")
        assert hero["props"]["event"] == "Karlsruher Halbmarathon"
        # Leading zero stripped in fallback as well.
        assert hero["props"]["target_time"] == "1:24:00"
        assert hero["props"]["target_date"] == "2026-09-20"

        # Stat grid populated from structured fitness fields.
        assert "stat_grid" in types
        grid = next(w for w in widgets if w["type"] == "stat_grid")
        labels = [s["label"] for s in grid["props"]["stats"]]
        assert "VO2max" in labels
        assert "Threshold-Pace" in labels

        # Edit hints present for every widget.
        hints = body["edit_hint_per_widget"]
        assert len(hints) == len(widgets)
        for hint in hints.values():
            assert hint.startswith("Lass uns")

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(
        self, widgets_client: AsyncClient
    ) -> None:
        test_settings = _make_test_settings()
        with patch("src.api.auth.get_settings", return_value=test_settings):
            response = await widgets_client.get("/profile/identity/widgets")
        assert response.status_code == 401
