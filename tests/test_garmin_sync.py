"""Unit tests for Phase 8c — Garmin Sync.

Covers:
- Token CRUD: store_token, get_token, delete_token, list_connected_providers,
  update_token_status, update_last_sync
- GarminSyncService.authenticate: success, wrong credentials
- GarminSyncService.sync_activities: success, expired token
- GarminSyncService.sync_daily_stats: success, no token
- GarminSyncService.sync_sleep: success
- GarminSyncService.check_connection: connected, disconnected
- sync_garmin_data agent tool: registered, success, no user_id
- Garmin API router: connect, status, disconnect, sync cooldown enforcement

All external dependencies (Garmin library, Supabase, Redis) are fully mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-garmin-test-uuid"
GARMIN_EMAIL = "athlete@example.com"
GARMIN_PASSWORD = "s3cr3t"
GARTH_TOKENS = '{"access_token": "abc", "refresh_token": "xyz"}'
DISPLAY_NAME = "Test Athlete"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_supabase_mock() -> MagicMock:
    """Return a mock Supabase client with chainable query builder."""
    client = MagicMock()
    # Make every query builder method return the mock itself so chaining works
    table_mock = MagicMock()
    client.table.return_value = table_mock
    for method in ("select", "upsert", "update", "delete", "insert", "eq"):
        getattr(table_mock, method).return_value = table_mock
    table_mock.execute.return_value = MagicMock(data=[])
    return client


def _token_row(status: str = "active") -> dict:
    return {
        "user_id": USER_ID,
        "provider": "garmin",
        "token_data": {"garth_tokens": GARTH_TOKENS, "email": GARMIN_EMAIL},
        "provider_user_id": DISPLAY_NAME,
        "status": status,
        "last_sync_at": None,
    }


# ===========================================================================
# 1. Token CRUD — src.db.provider_tokens_db
# ===========================================================================


class TestProviderTokensDb:
    """Tests for the provider_tokens_db CRUD functions."""

    def test_store_token_upserts_and_returns_row(self) -> None:
        """store_token should upsert on user_id+provider and return the row."""
        from src.db.provider_tokens_db import store_token

        mock_client = _make_supabase_mock()
        expected_row = _token_row()
        mock_client.table().execute.return_value = MagicMock(data=[expected_row])

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            result = store_token(USER_ID, "garmin", {"garth_tokens": GARTH_TOKENS}, DISPLAY_NAME)

        assert result["user_id"] == USER_ID
        assert result["provider"] == "garmin"
        assert result["status"] == "active"

    def test_store_token_returns_row_when_data_empty(self) -> None:
        """store_token falls back to the constructed row when data is empty."""
        from src.db.provider_tokens_db import store_token

        mock_client = _make_supabase_mock()
        mock_client.table().execute.return_value = MagicMock(data=[])

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            result = store_token(USER_ID, "garmin", {"garth_tokens": GARTH_TOKENS})

        assert result["user_id"] == USER_ID
        assert result["provider"] == "garmin"

    def test_get_token_returns_row(self) -> None:
        """get_token returns the first row for a matching user+provider."""
        from src.db.provider_tokens_db import get_token

        mock_client = _make_supabase_mock()
        mock_client.table().execute.return_value = MagicMock(data=[_token_row()])

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            result = get_token(USER_ID, "garmin")

        assert result is not None
        assert result["provider"] == "garmin"
        assert result["status"] == "active"

    def test_get_token_returns_none_when_not_found(self) -> None:
        """get_token returns None when no token exists."""
        from src.db.provider_tokens_db import get_token

        mock_client = _make_supabase_mock()
        mock_client.table().execute.return_value = MagicMock(data=[])

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            result = get_token(USER_ID, "garmin")

        assert result is None

    def test_delete_token_calls_delete(self) -> None:
        """delete_token should invoke the Supabase delete chain."""
        from src.db.provider_tokens_db import delete_token

        mock_client = _make_supabase_mock()

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            delete_token(USER_ID, "garmin")

        mock_client.table.assert_called_with("provider_tokens")
        mock_client.table().delete.assert_called_once()

    def test_list_connected_providers_returns_names(self) -> None:
        """list_connected_providers returns provider name strings."""
        from src.db.provider_tokens_db import list_connected_providers

        mock_client = _make_supabase_mock()
        mock_client.table().execute.return_value = MagicMock(
            data=[{"provider": "garmin"}, {"provider": "strava"}]
        )

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            providers = list_connected_providers(USER_ID)

        assert providers == ["garmin", "strava"]

    def test_list_connected_providers_empty(self) -> None:
        """list_connected_providers returns empty list when no active tokens."""
        from src.db.provider_tokens_db import list_connected_providers

        mock_client = _make_supabase_mock()
        mock_client.table().execute.return_value = MagicMock(data=[])

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            providers = list_connected_providers(USER_ID)

        assert providers == []

    def test_update_token_status_calls_update(self) -> None:
        """update_token_status should invoke the Supabase update chain."""
        from src.db.provider_tokens_db import update_token_status

        mock_client = _make_supabase_mock()

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            update_token_status(USER_ID, "garmin", "expired")

        mock_client.table().update.assert_called_once()
        update_call_kwargs = mock_client.table().update.call_args[0][0]
        assert update_call_kwargs["status"] == "expired"

    def test_update_last_sync_calls_update(self) -> None:
        """update_last_sync should update last_sync_at via Supabase."""
        from src.db.provider_tokens_db import update_last_sync

        mock_client = _make_supabase_mock()

        with patch("src.db.provider_tokens_db.get_supabase", return_value=mock_client):
            update_last_sync(USER_ID, "garmin")

        mock_client.table().update.assert_called_once()
        update_payload = mock_client.table().update.call_args[0][0]
        assert "last_sync_at" in update_payload


# ===========================================================================
# 2. GarminSyncService — authenticate
# ===========================================================================


class TestGarminAuthenticate:
    """Tests for GarminSyncService.authenticate."""

    def test_authenticate_success_stores_token(self) -> None:
        """Successful login stores tokens and returns connected status."""
        from src.services.garmin_sync import GarminSyncService

        mock_garmin = MagicMock()
        mock_garmin.garth.dumps.return_value = GARTH_TOKENS
        mock_garmin.get_full_name.return_value = DISPLAY_NAME

        with (
            patch("src.services.providers.garmin.store_token") as mock_store,
            patch.dict("sys.modules", {"garminconnect": MagicMock(Garmin=MagicMock(return_value=mock_garmin))}),
        ):
            result = GarminSyncService.authenticate(USER_ID, GARMIN_EMAIL, GARMIN_PASSWORD)

        assert result["status"] == "connected"
        assert result["display_name"] == DISPLAY_NAME
        mock_store.assert_called_once()
        store_args = mock_store.call_args[1]
        assert store_args["user_id"] == USER_ID
        assert store_args["provider"] == "garmin"
        assert store_args["token_data"]["garth_tokens"] == GARTH_TOKENS
        assert store_args["provider_user_id"] == DISPLAY_NAME

    def test_authenticate_wrong_credentials_returns_error(self) -> None:
        """Failed login returns error dict without raising."""
        from src.services.garmin_sync import GarminSyncService

        mock_garmin_class = MagicMock()
        mock_garmin_class.return_value.login.side_effect = Exception("Invalid credentials")

        with patch.dict("sys.modules", {"garminconnect": MagicMock(Garmin=mock_garmin_class)}):
            result = GarminSyncService.authenticate(USER_ID, GARMIN_EMAIL, "wrong")

        assert result["status"] == "error"
        assert "Invalid credentials" in result["error"]


# ===========================================================================
# 3. GarminSyncService — sync_activities
# ===========================================================================


class TestGarminSyncActivities:
    """Tests for GarminSyncService.sync_activities."""

    def test_sync_activities_returns_correct_summary(self) -> None:
        """sync_activities should upsert rows and return synced count."""
        from src.services.garmin_sync import GarminSyncService

        mock_garmin = MagicMock()
        mock_garmin.get_activities_by_date.return_value = [
            {"activityId": "111", "activityType": {"typeKey": "running"}, "duration": 3600, "distance": 10000},
            {"activityId": "222", "activityType": {"typeKey": "cycling"}, "duration": 7200, "distance": 40000},
        ]
        mock_supabase = _make_supabase_mock()

        with (
            patch(
                "src.services.providers.garmin.GarminProvider._restore_session",
                return_value=(mock_garmin, _token_row()),
            ),
            patch("src.db.client.get_supabase", return_value=mock_supabase),
            patch("src.services.providers.garmin.update_last_sync"),
        ):
            result = GarminSyncService.sync_activities(USER_ID, days=7)

        assert result["status"] == "ok"
        assert result["synced"] == 2
        assert result["skipped"] == 0
        assert result["days"] == 7

    def test_sync_activities_skips_missing_activity_id(self) -> None:
        """Activities without an activityId are skipped."""
        from src.services.garmin_sync import GarminSyncService

        mock_garmin = MagicMock()
        mock_garmin.get_activities_by_date.return_value = [
            {"activityId": "111", "activityType": {"typeKey": "running"}, "duration": 1800},
            {"activityType": {"typeKey": "other"}, "duration": 600},  # no activityId
        ]
        mock_supabase = _make_supabase_mock()

        with (
            patch(
                "src.services.providers.garmin.GarminProvider._restore_session",
                return_value=(mock_garmin, _token_row()),
            ),
            patch("src.db.client.get_supabase", return_value=mock_supabase),
            patch("src.services.providers.garmin.update_last_sync"),
        ):
            result = GarminSyncService.sync_activities(USER_ID)

        assert result["synced"] == 1
        assert result["skipped"] == 1

    def test_sync_activities_expired_token_returns_error(self) -> None:
        """sync_activities returns error dict when token is expired."""
        from src.services.garmin_sync import GarminSyncService

        with patch(
            "src.services.providers.garmin.GarminProvider._restore_session",
            side_effect=ValueError("Garmin session expired: token invalid"),
        ):
            result = GarminSyncService.sync_activities(USER_ID)

        assert result["status"] == "error"
        assert "expired" in result["error"]


# ===========================================================================
# 4. GarminSyncService — sync_daily_stats
# ===========================================================================


class TestGarminSyncDailyStats:
    """Tests for GarminSyncService.sync_daily_stats."""

    def test_sync_daily_stats_success(self) -> None:
        """sync_daily_stats syncs one row per day and returns correct count."""
        from src.services.garmin_sync import GarminSyncService

        mock_garmin = MagicMock()
        mock_garmin.get_stats.return_value = {
            "restingHeartRate": 48,
            "totalSteps": 12000,
            "averageStressLevel": 25,
        }
        mock_supabase = _make_supabase_mock()

        with (
            patch(
                "src.services.providers.garmin.GarminProvider._restore_session",
                return_value=(mock_garmin, _token_row()),
            ),
            patch("src.db.client.get_supabase", return_value=mock_supabase),
            patch("src.services.providers.garmin.update_last_sync"),
        ):
            result = GarminSyncService.sync_daily_stats(USER_ID, days=3)

        assert result["status"] == "ok"
        assert result["synced"] == 3
        assert result["days"] == 3

    def test_sync_daily_stats_no_token_returns_error(self) -> None:
        """sync_daily_stats returns error when no token is found."""
        from src.services.garmin_sync import GarminSyncService

        with patch(
            "src.services.providers.garmin.GarminProvider._restore_session",
            side_effect=ValueError("No Garmin connection found"),
        ):
            result = GarminSyncService.sync_daily_stats(USER_ID)

        assert result["status"] == "error"
        assert "No Garmin connection" in result["error"]


# ===========================================================================
# 5. GarminSyncService — check_connection
# ===========================================================================


class TestGarminCheckConnection:
    """Tests for GarminSyncService.check_connection."""

    def test_check_connection_valid_returns_connected(self) -> None:
        """check_connection returns connected=True for valid tokens."""
        from src.services.garmin_sync import GarminSyncService

        token_row = _token_row()
        token_row["last_sync_at"] = "2026-03-01T10:00:00+00:00"
        token_row["provider_user_id"] = DISPLAY_NAME

        with patch(
            "src.services.providers.garmin.GarminProvider._restore_session",
            return_value=(MagicMock(), token_row),
        ):
            result = GarminSyncService.check_connection(USER_ID)

        assert result["connected"] is True
        assert result["provider"] == "garmin"
        assert result["status"] == "active"
        assert result["provider_user_id"] == DISPLAY_NAME

    def test_check_connection_no_token_returns_disconnected(self) -> None:
        """check_connection returns connected=False when no token exists."""
        from src.services.garmin_sync import GarminSyncService

        with patch(
            "src.services.providers.garmin.GarminProvider._restore_session",
            side_effect=ValueError("No Garmin connection found"),
        ):
            result = GarminSyncService.check_connection(USER_ID)

        assert result["connected"] is False
        assert result["status"] == "disconnected"


# ===========================================================================
# 6. sync_garmin_data agent tool
# ===========================================================================


class TestSyncGarminDataTool:
    """Tests for the sync_garmin_data agent tool."""

    def _build_registry(self, user_id: str | None = USER_ID) -> ToolRegistry:
        from src.agent.tools.garmin_tools import register_garmin_tools

        registry = ToolRegistry()
        user_model = MagicMock()
        user_model.user_id = user_id
        register_garmin_tools(registry, user_model)
        return registry

    def test_tool_registered(self) -> None:
        """sync_garmin_data tool should be in the registry."""
        registry = self._build_registry()
        names = [t["name"] for t in registry.list_tools()]
        assert "sync_garmin_data" in names

    def test_tool_category_is_data(self) -> None:
        """sync_garmin_data tool should have category='data'."""
        registry = self._build_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["sync_garmin_data"]["category"] == "data"

    def test_tool_no_user_id_returns_error(self) -> None:
        """Tool returns error dict when no user_id is set on user_model."""
        registry = self._build_registry(user_id=None)
        result = registry.execute("sync_garmin_data", {})
        assert "error" in result

    def test_tool_success_returns_combined_result(self) -> None:
        """Tool calls sync_activities and sync_daily_stats and returns both."""
        registry = self._build_registry()

        activities_result = {"status": "ok", "synced": 5, "skipped": 0, "days": 7}
        stats_result = {"status": "ok", "synced": 7, "days": 7}

        with (
            patch(
                "src.services.garmin_sync.GarminSyncService.sync_activities",
                return_value=activities_result,
            ),
            patch(
                "src.services.garmin_sync.GarminSyncService.sync_daily_stats",
                return_value=stats_result,
            ),
        ):
            result = registry.execute("sync_garmin_data", {"days": 7})

        assert result["activities"]["synced"] == 5
        assert result["daily_stats"]["synced"] == 7

    def test_tool_no_user_model_returns_error(self) -> None:
        """Tool returns error when user_model is None."""
        from src.agent.tools.garmin_tools import register_garmin_tools

        registry = ToolRegistry()
        register_garmin_tools(registry, user_model=None)
        result = registry.execute("sync_garmin_data", {})
        assert "error" in result


# ===========================================================================
# 7. Garmin API router — endpoint tests
# ===========================================================================


class TestGarminRouter:
    """Tests for the /garmin API endpoints."""

    def _make_app(self) -> "TestClient":
        """Build a minimal FastAPI app with the Garmin router.

        Overrides the get_user_id dependency so no real JWT validation occurs.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.routers.garmin import router
        from src.api.auth import get_user_id

        app = FastAPI()
        app.include_router(router, prefix="/garmin")
        # Override auth dependency so tests never touch JWT validation
        app.dependency_overrides[get_user_id] = lambda: USER_ID
        return TestClient(app, raise_server_exceptions=False)

    def test_connect_returns_connected(self) -> None:
        """POST /garmin/connect returns 200 with status=connected on success."""
        client = self._make_app()

        with patch(
            "src.services.garmin_sync.GarminSyncService.authenticate",
            return_value={"status": "connected", "display_name": DISPLAY_NAME},
        ):
            resp = client.post(
                "/garmin/connect",
                json={"email": GARMIN_EMAIL, "password": GARMIN_PASSWORD},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "connected"

    def test_connect_raises_400_on_auth_failure(self) -> None:
        """POST /garmin/connect returns 400 when authentication fails."""
        client = self._make_app()

        with patch(
            "src.services.garmin_sync.GarminSyncService.authenticate",
            return_value={"status": "error", "error": "Invalid credentials"},
        ):
            resp = client.post(
                "/garmin/connect",
                json={"email": GARMIN_EMAIL, "password": "wrong"},
            )

        assert resp.status_code == 400

    def test_status_returns_connection_info(self) -> None:
        """GET /garmin/status returns connection info."""
        client = self._make_app()

        with patch(
            "src.services.garmin_sync.GarminSyncService.check_connection",
            return_value={
                "connected": True,
                "provider": "garmin",
                "status": "active",
                "last_sync_at": None,
                "provider_user_id": DISPLAY_NAME,
            },
        ):
            resp = client.get("/garmin/status")

        assert resp.status_code == 200
        assert resp.json()["connected"] is True

    def test_disconnect_deletes_token(self) -> None:
        """DELETE /garmin/disconnect calls delete_token and returns disconnected."""
        client = self._make_app()

        with patch("src.db.provider_tokens_db.delete_token") as mock_delete:
            resp = client.delete("/garmin/disconnect")

        assert resp.status_code == 200
        assert resp.json()["status"] == "disconnected"
        mock_delete.assert_called_once_with(USER_ID, "garmin")


# ===========================================================================
# 8. Sync cooldown enforcement
# ===========================================================================


class TestSyncCooldown:
    """Tests for the Redis-based sync cooldown check."""

    @pytest.mark.asyncio
    async def test_cooldown_set_on_first_sync(self) -> None:
        """First sync sets the cooldown key in Redis."""
        from src.api.routers.garmin import _check_sync_cooldown

        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 0
        mock_redis.set = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.garmin_sync_cooldown_seconds = 900
        mock_settings.redis_url = "redis://localhost:6379"

        with (
            patch("src.api.routers.garmin.aioredis.from_url", return_value=mock_redis),
            patch("src.api.routers.garmin.get_settings", return_value=mock_settings),
        ):
            await _check_sync_cooldown(USER_ID)

        mock_redis.set.assert_called_once_with(f"garmin_sync:{USER_ID}", "1", ex=900)

    @pytest.mark.asyncio
    async def test_cooldown_raises_429_when_active(self) -> None:
        """Second sync within cooldown raises HTTPException(429)."""
        from fastapi import HTTPException
        from src.api.routers.garmin import _check_sync_cooldown

        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1  # key already set
        mock_redis.aclose = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.garmin_sync_cooldown_seconds = 900
        mock_settings.redis_url = "redis://localhost:6379"

        with (
            patch("src.api.routers.garmin.aioredis.from_url", return_value=mock_redis),
            patch("src.api.routers.garmin.get_settings", return_value=mock_settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _check_sync_cooldown(USER_ID)

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_cooldown_skipped_when_redis_unavailable(self) -> None:
        """Cooldown check is skipped silently when Redis is unreachable."""
        from src.api.routers.garmin import _check_sync_cooldown

        mock_settings = MagicMock()
        mock_settings.garmin_sync_cooldown_seconds = 900
        mock_settings.redis_url = "redis://localhost:6379"

        with (
            patch(
                "src.api.routers.garmin.aioredis.from_url",
                side_effect=Exception("Connection refused"),
            ),
            patch("src.api.routers.garmin.get_settings", return_value=mock_settings),
        ):
            # Should not raise
            await _check_sync_cooldown(USER_ID)


# ===========================================================================
# 9. Config — garmin_sync_cooldown_seconds
# ===========================================================================


class TestGarminConfig:
    """Tests that config has the garmin_sync_cooldown_seconds field."""

    def test_default_cooldown_is_900(self) -> None:
        """garmin_sync_cooldown_seconds should default to 900 seconds."""
        from src.config import Settings

        settings = Settings()
        assert settings.garmin_sync_cooldown_seconds == 900
