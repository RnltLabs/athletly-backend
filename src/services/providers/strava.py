"""Strava provider — OAuth2 with refresh-token flow.

Strava uses standard OAuth2: a long-lived ``refresh_token`` (per user) lets
us mint short-lived ``access_token``s for the API.  We persist the token
triple ``(access_token, refresh_token, expires_at)`` in ``provider_tokens``
and refresh transparently before each API call.

The OAuth app credentials (``client_id`` / ``client_secret``) live in
environment / settings, not in the per-user token row.

Capabilities: activities only.  Strava does not expose sleep, HRV, or body
battery — those remain Garmin-only in the system.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from src.config import get_settings
from src.db.provider_tokens_db import (
    get_token,
    store_token,
    update_last_sync,
)
from src.services.providers.base import Provider, ProviderCapabilities

logger = logging.getLogger(__name__)

STRAVA_API_BASE = "https://www.strava.com/api/v3"
STRAVA_OAUTH_TOKEN_URL = "https://www.strava.com/api/v3/oauth/token"
STRAVA_OAUTH_AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
REQUIRED_SCOPES = "read,activity:read_all"


class StravaProvider(Provider):
    """Strava integration via OAuth2 refresh-token flow."""

    name = "strava"
    capabilities = ProviderCapabilities(
        activities=True,
        daily_metrics=False,
        sleep=False,
        body_battery=False,
        vo2max=False,
        auth_flow="oauth2_browser",
    )

    # ------------------------------------------------------------------
    # Authorization URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def authorization_url(redirect_uri: str = "http://localhost") -> str:
        """Build the URL the user clicks to authorize the Strava app.

        After authorizing, Strava redirects to ``redirect_uri`` with
        ``?code=<auth_code>``.  Pass that code to ``exchange_code``.
        """
        settings = get_settings()
        return (
            f"{STRAVA_OAUTH_AUTHORIZE_URL}"
            f"?client_id={settings.strava_client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri}"
            f"&approval_prompt=force"
            f"&scope={REQUIRED_SCOPES}"
        )

    # ------------------------------------------------------------------
    # Token exchange and refresh
    # ------------------------------------------------------------------

    def exchange_code(self, user_id: str, code: str) -> dict:
        """Exchange a one-time auth code for access/refresh tokens.

        Called after the user clicks the authorization URL and pastes back
        the ``code=`` query parameter.
        """
        settings = get_settings()
        try:
            response = httpx.post(
                STRAVA_OAUTH_TOKEN_URL,
                data={
                    "client_id": settings.strava_client_id,
                    "client_secret": settings.strava_client_secret,
                    "grant_type": "authorization_code",
                    "code": code,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {"status": "error", "error": f"Token exchange failed: {exc}"}

        return self._persist_token(user_id, payload)

    def import_bootstrap_tokens(
        self, user_id: str, access_token: str, refresh_token: str
    ) -> dict:
        """Import a pre-existing access+refresh token pair into supabase.

        Used to seed a test user with tokens obtained out-of-band (e.g.
        from the Strava developer console). The tokens are immediately
        refreshed so that we know they are valid and to capture the new
        ``expires_at``.
        """
        store_token(
            user_id=user_id,
            provider=self.name,
            token_data={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": 0,
            },
            provider_user_id=None,
        )
        refreshed = self._refresh(user_id)
        if refreshed.get("status") != "ok":
            return refreshed
        athlete = self._fetch_athlete(user_id)
        if athlete.get("status") == "ok":
            display = (
                f"{athlete['data'].get('firstname', '')} "
                f"{athlete['data'].get('lastname', '')}".strip()
            )
            store_token(
                user_id=user_id,
                provider=self.name,
                token_data=get_token(user_id, self.name)["token_data"],
                provider_user_id=display or None,
            )
            return {"status": "imported", "display_name": display}
        return {"status": "imported", "display_name": None}

    def _persist_token(self, user_id: str, payload: dict) -> dict:
        """Persist a token response from /oauth/token to provider_tokens."""
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        expires_at = payload.get("expires_at")
        if not access or not refresh:
            return {"status": "error", "error": "Strava token response missing fields"}
        display = None
        athlete = payload.get("athlete") or {}
        if athlete:
            display = (
                f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
                or None
            )
        store_token(
            user_id=user_id,
            provider=self.name,
            token_data={
                "access_token": access,
                "refresh_token": refresh,
                "expires_at": expires_at,
            },
            provider_user_id=display,
        )
        return {"status": "connected", "display_name": display}

    def _refresh(self, user_id: str) -> dict:
        """Use the stored refresh_token to mint a fresh access_token."""
        settings = get_settings()
        token_row = get_token(user_id, self.name)
        if not token_row:
            return {"status": "error", "error": "No Strava connection"}
        refresh = token_row["token_data"].get("refresh_token")
        if not refresh:
            return {"status": "error", "error": "Missing refresh_token"}
        try:
            response = httpx.post(
                STRAVA_OAUTH_TOKEN_URL,
                data={
                    "client_id": settings.strava_client_id,
                    "client_secret": settings.strava_client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {"status": "error", "error": f"Refresh failed: {exc}"}

        store_token(
            user_id=user_id,
            provider=self.name,
            token_data={
                "access_token": payload["access_token"],
                "refresh_token": payload["refresh_token"],
                "expires_at": payload["expires_at"],
            },
            provider_user_id=token_row.get("provider_user_id"),
        )
        return {"status": "ok"}

    def _access_token(self, user_id: str) -> str:
        """Return a valid access_token, refreshing if needed."""
        token_row = get_token(user_id, self.name)
        if not token_row:
            raise ValueError("No Strava connection")
        expires_at = token_row["token_data"].get("expires_at", 0) or 0
        if expires_at - int(time.time()) < 300:
            result = self._refresh(user_id)
            if result.get("status") != "ok":
                raise ValueError(result.get("error", "Token refresh failed"))
            token_row = get_token(user_id, self.name)
        return token_row["token_data"]["access_token"]

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _fetch_athlete(self, user_id: str) -> dict:
        try:
            access = self._access_token(user_id)
            response = httpx.get(
                f"{STRAVA_API_BASE}/athlete",
                headers={"Authorization": f"Bearer {access}"},
                timeout=15.0,
            )
            response.raise_for_status()
            return {"status": "ok", "data": response.json()}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def check_connection(self, user_id: str) -> dict:
        token_row = get_token(user_id, self.name)
        if not token_row:
            return {"connected": False, "provider": self.name, "status": "disconnected"}
        athlete = self._fetch_athlete(user_id)
        if athlete.get("status") == "ok":
            return {
                "connected": True,
                "provider": self.name,
                "status": "active",
                "last_sync_at": token_row.get("last_sync_at"),
                "provider_user_id": token_row.get("provider_user_id"),
            }
        return {
            "connected": False,
            "provider": self.name,
            "status": "error",
            "error": athlete.get("error"),
        }

    def sync_activities(self, user_id: str, days: int = 7) -> dict:
        try:
            access = self._access_token(user_id)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        after_ts = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        )

        synced = 0
        skipped = 0
        page = 1
        from src.db.client import get_supabase

        client = get_supabase()
        try:
            while True:
                response = httpx.get(
                    f"{STRAVA_API_BASE}/athlete/activities",
                    headers={"Authorization": f"Bearer {access}"},
                    params={"after": after_ts, "per_page": 200, "page": page},
                    timeout=30.0,
                )
                if response.status_code == 401:
                    return {
                        "status": "error",
                        "error": "Strava unauthorized — check scope (need activity:read_all)",
                    }
                response.raise_for_status()
                acts = response.json()
                if not acts:
                    break
                for act in acts:
                    strava_id = str(act.get("id", ""))
                    if not strava_id:
                        skipped += 1
                        continue
                    avg_speed = act.get("average_speed")
                    avg_pace = None
                    if avg_speed and avg_speed > 0:
                        avg_pace = round(1000 / (avg_speed * 60), 2)
                    row = {
                        "user_id": user_id,
                        "sport": _strava_sport_to_canonical(act.get("type")),
                        "start_time": act.get("start_date_local"),
                        "duration_seconds": act.get("moving_time"),
                        "distance_meters": act.get("distance"),
                        "avg_hr": act.get("average_heartrate"),
                        "max_hr": act.get("max_heartrate"),
                        "calories": act.get("calories"),
                        "avg_pace_min_km": avg_pace,
                        "elevation_gain_m": act.get("total_elevation_gain"),
                        "source": self.name,
                        "strava_activity_id": strava_id,
                        "raw_data": act,
                    }
                    client.table("activities").upsert(
                        row, on_conflict="user_id,strava_activity_id"
                    ).execute()
                    synced += 1
                if len(acts) < 200:
                    break
                page += 1

            update_last_sync(user_id, self.name)
            return {"status": "ok", "synced": synced, "skipped": skipped, "days": days}
        except Exception as exc:
            logger.exception("Strava sync_activities failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}


def _strava_sport_to_canonical(strava_type: str | None) -> str:
    """Map Strava activity type names to the canonical sport vocabulary.

    Garmin uses ``running``, ``road_biking``, ``strength_training`` etc.
    Strava uses ``Run``, ``Ride``, ``WeightTraining`` etc. We normalize
    Strava → Garmin-style so downstream code sees a single vocabulary.
    """
    if not strava_type:
        return "unknown"
    mapping = {
        "Run": "running",
        "TrailRun": "trail_running",
        "VirtualRun": "treadmill_running",
        "Ride": "road_biking",
        "VirtualRide": "virtual_ride",
        "MountainBikeRide": "mountain_biking",
        "GravelRide": "gravel_cycling",
        "Swim": "swimming",
        "WeightTraining": "strength_training",
        "Workout": "strength_training",
        "Yoga": "yoga",
        "Walk": "walking",
        "Hike": "hiking",
        "Rowing": "rowing",
        "Elliptical": "elliptical",
    }
    return mapping.get(strava_type, strava_type.lower())
