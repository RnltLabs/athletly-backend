"""Garmin provider — wraps python-garminconnect / Garth OAuth.

Authenticates with email + password (no MFA support in current upstream lib),
persists Garth session tokens in ``provider_tokens``, and syncs activities,
daily health metrics, and sleep into local Supabase tables.

All garminconnect imports are deferred inside methods so the library remains
an optional dependency — importing this module does not require it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.db.provider_tokens_db import (
    get_token,
    store_token,
    update_last_sync,
    update_token_status,
)
from src.services.providers.base import Provider, ProviderCapabilities

logger = logging.getLogger(__name__)


class GarminProvider(Provider):
    """Garmin Connect integration via python-garminconnect (Garth)."""

    name = "garmin"
    capabilities = ProviderCapabilities(
        activities=True,
        daily_metrics=True,
        sleep=True,
        body_battery=True,
        vo2max=True,
        auth_flow="password_optional_mfa",
    )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, user_id: str, email: str, password: str) -> dict:
        """Authenticate with Garmin and persist Garth session tokens.

        Args:
            user_id: Supabase user UUID this connection belongs to.
            email: Garmin Connect account email.
            password: Garmin Connect account password.

        Returns:
            ``{"status": "connected", "display_name": "..."}`` on success,
            ``{"status": "error", "error": "..."}`` on failure.
        """
        try:
            from garminconnect import Garmin

            garmin = Garmin(email, password)
            garmin.login()

            token_data_str = garmin.garth.dumps()
            display_name: str = email
            if hasattr(garmin, "get_full_name"):
                try:
                    display_name = garmin.get_full_name() or email
                except Exception:
                    pass

            store_token(
                user_id=user_id,
                provider=self.name,
                token_data={"garth_tokens": token_data_str, "email": email},
                provider_user_id=display_name,
            )
            logger.info("Garmin authenticated for user %s (%s)", user_id, display_name)
            return {"status": "connected", "display_name": display_name}
        except Exception as exc:
            logger.warning("Garmin auth failed for user %s: %s", user_id, exc)
            return {"status": "error", "error": str(exc)}

    def clone_tokens_from(self, source_user_id: str, target_user_id: str) -> dict:
        """Copy an existing Garmin token from one Athletly user to another.

        Used by the test harness: a dedicated test user reuses the real
        user's already-authenticated Garmin session without going through
        another login/MFA dance.

        Args:
            source_user_id: User whose Garmin tokens are already valid.
            target_user_id: User to copy tokens to.

        Returns:
            ``{"status": "cloned", "display_name": "..."}`` or error dict.
        """
        source = get_token(source_user_id, self.name)
        if not source:
            return {
                "status": "error",
                "error": f"Source user {source_user_id} has no Garmin connection to clone",
            }
        try:
            store_token(
                user_id=target_user_id,
                provider=self.name,
                token_data=source["token_data"],
                provider_user_id=source.get("provider_user_id"),
            )
            return {
                "status": "cloned",
                "display_name": source.get("provider_user_id"),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Session restore
    # ------------------------------------------------------------------

    def _restore_session(self, user_id: str) -> tuple[Any, dict]:
        """Return a logged-in ``Garmin`` instance from stored tokens.

        Raises ``ValueError`` if no token exists, the token is not active,
        or the session cannot be restored. Marks the token as expired in
        the latter case.
        """
        token_row = get_token(user_id, self.name)
        if not token_row:
            raise ValueError("No Garmin connection found")
        if token_row.get("status") != "active":
            raise ValueError(f"Garmin connection is {token_row.get('status')}")
        try:
            from garminconnect import Garmin

            garmin = Garmin()
            garth_tokens = token_row["token_data"]["garth_tokens"]
            garmin.login(tokenstore=garth_tokens)
            return garmin, token_row
        except Exception as exc:
            update_token_status(user_id, self.name, "expired")
            raise ValueError(f"Garmin session expired: {exc}")

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def check_connection(self, user_id: str) -> dict:
        try:
            _, token_row = self._restore_session(user_id)
            return {
                "connected": True,
                "provider": self.name,
                "status": "active",
                "last_sync_at": token_row.get("last_sync_at"),
                "provider_user_id": token_row.get("provider_user_id"),
            }
        except ValueError:
            return {"connected": False, "provider": self.name, "status": "disconnected"}
        except Exception:
            return {"connected": False, "provider": self.name, "status": "error"}

    def sync_activities(self, user_id: str, days: int = 7) -> dict:
        try:
            garmin, _ = self._restore_session(user_id)
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days)

            activities = garmin.get_activities_by_date(
                start_date.isoformat(), end_date.isoformat()
            )

            from src.db.client import get_supabase

            client = get_supabase()
            synced = 0
            skipped = 0
            for act in activities or []:
                garmin_id = str(act.get("activityId", ""))
                if not garmin_id:
                    skipped += 1
                    continue

                avg_speed = act.get("averageSpeed")
                avg_pace = None
                if avg_speed and avg_speed > 0:
                    avg_pace = round(1000 / (avg_speed * 60), 2)

                def _int(val: object) -> int | None:
                    return int(val) if val is not None else None

                row = {
                    "user_id": user_id,
                    "sport": act.get("activityType", {}).get("typeKey", "unknown"),
                    "start_time": act.get("startTimeLocal"),
                    "duration_seconds": _int(act.get("duration")),
                    "distance_meters": act.get("distance"),
                    "avg_hr": _int(act.get("averageHR")),
                    "max_hr": _int(act.get("maxHR")),
                    "calories": _int(act.get("calories")),
                    "training_effect": act.get("aerobicTrainingEffect"),
                    "vo2max_activity": act.get("vO2MaxValue"),
                    "avg_pace_min_km": avg_pace,
                    "elevation_gain_m": act.get("elevationGain"),
                    "source": self.name,
                    "garmin_activity_id": garmin_id,
                    "raw_data": act,
                }
                client.table("activities").upsert(
                    row, on_conflict="user_id,garmin_activity_id"
                ).execute()
                synced += 1

            update_last_sync(user_id, self.name)
            return {"status": "ok", "synced": synced, "skipped": skipped, "days": days}
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Garmin sync_activities failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}

    def sync_daily_stats(self, user_id: str, days: int = 7) -> dict:
        try:
            garmin, _ = self._restore_session(user_id)
            from src.db.client import get_supabase

            client = get_supabase()
            synced = 0

            for i in range(days):
                day = (
                    datetime.now(timezone.utc).date() - timedelta(days=i)
                ).isoformat()
                try:
                    stats = garmin.get_stats(day)
                    if not stats:
                        continue

                    def _int(val: object) -> int | None:
                        return int(val) if val is not None else None

                    row: dict = {
                        "user_id": user_id,
                        "date": day,
                        "source": self.name,
                        "resting_heart_rate": _int(stats.get("restingHeartRate")),
                        "steps": _int(stats.get("totalSteps")),
                        "stress_avg": _int(stats.get("averageStressLevel")),
                        "hrv_avg": (stats.get("hrvSummary") or {}).get("weeklyAvg"),
                        "body_battery_high": _int(stats.get("bodyBatteryChargedValue")),
                        "body_battery_low": _int(stats.get("bodyBatteryDrainedValue")),
                        "active_calories": _int(stats.get("activeKilocalories")),
                        "total_calories": _int(stats.get("totalKilocalories")),
                        "floors_climbed": _int(stats.get("floorsAscended")),
                    }

                    try:
                        spo2 = garmin.get_spo2_data(day)
                        if spo2:
                            row["spo2_avg"] = spo2.get("averageSpO2") or spo2.get(
                                "allDaySpO2", {}
                            ).get("averageSpO2Value")
                    except Exception:
                        logger.debug("SpO2 unavailable for %s", day)

                    try:
                        resp = garmin.get_respiration_data(day)
                        if resp:
                            row["respiration_avg"] = resp.get("avgWakingRespirationValue")
                    except Exception:
                        logger.debug("Respiration unavailable for %s", day)

                    try:
                        maxm = garmin.get_max_metrics(day)
                        if maxm and isinstance(maxm, dict):
                            generic = maxm.get("generic", {}) or {}
                            row["vo2max"] = generic.get("vo2MaxPreciseValue") or generic.get(
                                "vo2MaxValue"
                            )
                        elif isinstance(maxm, list) and len(maxm) > 0:
                            first = maxm[0]
                            generic = first.get("generic", {}) or {}
                            row["vo2max"] = generic.get("vo2MaxPreciseValue") or generic.get(
                                "vo2MaxValue"
                            )
                    except Exception:
                        logger.debug("VO2Max unavailable for %s", day)

                    try:
                        im = garmin.get_intensity_minutes_data(day)
                        if im:
                            moderate = im.get("moderateIntensityMinutes", 0) or 0
                            vigorous = im.get("vigorousIntensityMinutes", 0) or 0
                            row["intensity_minutes"] = moderate + vigorous
                    except Exception:
                        logger.debug("Intensity minutes unavailable for %s", day)

                    if not row.get("floors_climbed"):
                        try:
                            floors = garmin.get_floors(day)
                            if floors:
                                row["floors_climbed"] = floors.get("floorsAscended")
                        except Exception:
                            pass

                    row["raw_data"] = stats
                    client.table("health_daily_metrics").upsert(
                        row, on_conflict="user_id,date,source"
                    ).execute()
                    synced += 1
                except Exception:
                    logger.debug(
                        "Failed to sync daily stats for %s", day, exc_info=True
                    )

            update_last_sync(user_id, self.name)
            return {"status": "ok", "synced": synced, "days": days}
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Garmin sync_daily_stats failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}

    def sync_sleep(self, user_id: str, days: int = 7) -> dict:
        try:
            garmin, _ = self._restore_session(user_id)
            from src.db.client import get_supabase

            client = get_supabase()
            synced = 0

            for i in range(days):
                day = (
                    datetime.now(timezone.utc).date() - timedelta(days=i)
                ).isoformat()
                try:
                    sleep = garmin.get_sleep_data(day)
                    if not sleep:
                        continue

                    daily_dto = sleep.get("dailySleepDTO") or {}
                    sleep_score = (
                        (daily_dto.get("sleepScores") or {})
                        .get("overall", {})
                        .get("value")
                    )
                    sleep_seconds = daily_dto.get("sleepTimeSeconds")
                    sleep_minutes = (
                        round(sleep_seconds / 60, 1)
                        if sleep_seconds is not None
                        else None
                    )

                    def _s2m(key: str) -> float | None:
                        val = daily_dto.get(key)
                        return round(val / 60, 1) if val is not None else None

                    client.table("health_daily_metrics").upsert(
                        {
                            "user_id": user_id,
                            "date": day,
                            "source": self.name,
                            "sleep_score": sleep_score,
                            "sleep_duration_minutes": sleep_minutes,
                            "sleep_deep_minutes": _s2m("deepSleepSeconds"),
                            "sleep_light_minutes": _s2m("lightSleepSeconds"),
                            "sleep_rem_minutes": _s2m("remSleepSeconds"),
                            "sleep_awake_minutes": _s2m("awakeSleepSeconds"),
                        },
                        on_conflict="user_id,date,source",
                    ).execute()
                    synced += 1
                except Exception:
                    logger.debug("Failed to sync sleep for %s", day, exc_info=True)

            return {"status": "ok", "synced": synced, "days": days}
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Garmin sync_sleep failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}

    def disconnect(self, user_id: str) -> dict:
        """Delete tokens AND all synced Garmin data for this user."""
        from src.db.client import get_supabase
        from src.db.provider_tokens_db import delete_token

        db = get_supabase()
        db.table("activities").delete().eq("user_id", user_id).eq(
            "source", self.name
        ).execute()
        db.table("health_daily_metrics").delete().eq("user_id", user_id).eq(
            "source", self.name
        ).execute()
        delete_token(user_id, self.name)
        return {"status": "disconnected"}
