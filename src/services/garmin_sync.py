"""Garmin sync service — interim solution using python-garminconnect.

Authenticates with Garmin Connect using email + password (Garth OAuth2 under
the hood), persists session tokens in provider_tokens, and syncs activities /
daily health stats into the local Supabase tables.

All garminconnect imports are deferred inside functions so the library is an
optional dependency that does not break the server when garminconnect is absent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.provider_tokens_db import (
    get_token,
    store_token,
    update_last_sync,
    update_token_status,
)

logger = logging.getLogger(__name__)


class GarminSyncService:
    """Sync Garmin data via python-garminconnect (interim solution)."""

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def authenticate(user_id: str, email: str, password: str) -> dict:
        """Authenticate with Garmin and persist session tokens.

        Args:
            user_id: Supabase user UUID.
            email: Garmin Connect account email.
            password: Garmin Connect account password.

        Returns:
            Dict with ``status`` ("connected" | "error") and optional fields.
        """
        try:
            from garminconnect import Garmin  # optional dep
            garmin = Garmin(email, password)
            garmin.login()

            # Serialize Garth OAuth2 tokens to a JSON-safe string
            token_data_str = garmin.garth.dumps()

            display_name: str = email
            if hasattr(garmin, "get_full_name"):
                try:
                    display_name = garmin.get_full_name() or email
                except Exception:
                    pass

            store_token(
                user_id=user_id,
                provider="garmin",
                token_data={"garth_tokens": token_data_str, "email": email},
                provider_user_id=display_name,
            )
            logger.info("Garmin authenticated for user %s (%s)", user_id, display_name)
            return {"status": "connected", "display_name": display_name}
        except Exception as exc:
            logger.warning("Garmin auth failed for user %s: %s", user_id, exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Session restoration
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_session(user_id: str):
        """Restore a Garmin session from stored tokens.

        Returns:
            Tuple of (Garmin instance, token_row dict).

        Raises:
            ValueError: If no token is found, token is not active, or session
                        could not be restored (marks token as expired).
        """
        token_row = get_token(user_id, "garmin")
        if not token_row:
            raise ValueError("No Garmin connection found")
        if token_row.get("status") != "active":
            raise ValueError(f"Garmin connection is {token_row.get('status')}")
        try:
            from garminconnect import Garmin  # optional dep
            garmin = Garmin()
            garth_tokens = token_row["token_data"]["garth_tokens"]
            garmin.login(tokenstore=garth_tokens)
            return garmin, token_row
        except Exception as exc:
            update_token_status(user_id, "garmin", "expired")
            raise ValueError(f"Garmin session expired: {exc}")

    # ------------------------------------------------------------------
    # Sync: activities
    # ------------------------------------------------------------------

    @staticmethod
    def sync_activities(user_id: str, days: int = 7) -> dict:
        """Sync recent activities from Garmin into the activities table.

        Extracts training_effect, vo2max, calories, pace, and elevation
        in addition to the base fields.
        """
        try:
            garmin, _ = GarminSyncService._restore_session(user_id)
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

                # Pace: convert avg speed (m/s) to min/km
                avg_speed = act.get("averageSpeed")
                avg_pace = None
                if avg_speed and avg_speed > 0:
                    avg_pace = round(1000 / (avg_speed * 60), 2)

                row = {
                    "user_id": user_id,
                    "sport": act.get("activityType", {}).get("typeKey", "unknown"),
                    "start_time": act.get("startTimeLocal"),
                    "duration_seconds": int(act.get("duration", 0)),
                    "distance_meters": act.get("distance"),
                    "avg_hr": act.get("averageHR"),
                    "max_hr": act.get("maxHR"),
                    "calories": act.get("calories"),
                    "training_effect": act.get("trainingEffectLabel") and act.get("aerobicTrainingEffect"),
                    "vo2max_activity": act.get("vO2MaxValue"),
                    "avg_pace_min_km": avg_pace,
                    "elevation_gain_m": act.get("elevationGain"),
                    "source": "garmin",
                    "garmin_activity_id": garmin_id,
                    "raw_data": act,
                }
                client.table("activities").upsert(
                    row, on_conflict="user_id,garmin_activity_id"
                ).execute()
                synced += 1

            update_last_sync(user_id, "garmin")
            return {"status": "ok", "synced": synced, "skipped": skipped, "days": days}
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Garmin sync_activities failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Sync: daily stats
    # ------------------------------------------------------------------

    @staticmethod
    def sync_daily_stats(user_id: str, days: int = 7) -> dict:
        """Sync daily health stats from Garmin.

        Fetches get_stats (base metrics), get_spo2_data, get_respiration_data,
        get_max_metrics (VO2Max), get_floors, and get_intensity_minutes_data
        for each day and merges them into a single upsert per day.
        """
        try:
            garmin, _ = GarminSyncService._restore_session(user_id)
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

                    row: dict = {
                        "user_id": user_id,
                        "date": day,
                        "source": "garmin",
                        "resting_heart_rate": stats.get("restingHeartRate"),
                        "steps": stats.get("totalSteps"),
                        "stress_avg": stats.get("averageStressLevel"),
                        "hrv_avg": (
                            stats.get("hrvSummary") or {}
                        ).get("weeklyAvg"),
                        "body_battery_high": stats.get("bodyBatteryChargedValue"),
                        "body_battery_low": stats.get("bodyBatteryDrainedValue"),
                        "active_calories": stats.get("activeKilocalories"),
                        "total_calories": stats.get("totalKilocalories"),
                        "floors_climbed": stats.get("floorsAscended"),
                    }

                    # -- SpO2 -------------------------------------------------
                    try:
                        spo2 = garmin.get_spo2_data(day)
                        if spo2:
                            row["spo2_avg"] = (
                                spo2.get("averageSpO2")
                                or spo2.get("allDaySpO2", {}).get("averageSpO2Value")
                            )
                    except Exception:
                        logger.debug("SpO2 unavailable for %s", day)

                    # -- Respiration ------------------------------------------
                    try:
                        resp = garmin.get_respiration_data(day)
                        if resp:
                            row["respiration_avg"] = resp.get("avgWakingRespirationValue")
                    except Exception:
                        logger.debug("Respiration unavailable for %s", day)

                    # -- VO2Max -----------------------------------------------
                    try:
                        maxm = garmin.get_max_metrics(day)
                        if maxm and isinstance(maxm, dict):
                            generic = maxm.get("generic", {}) or {}
                            row["vo2max"] = generic.get("vo2MaxPreciseValue") or generic.get("vo2MaxValue")
                        elif isinstance(maxm, list) and len(maxm) > 0:
                            first = maxm[0]
                            generic = first.get("generic", {}) or {}
                            row["vo2max"] = generic.get("vo2MaxPreciseValue") or generic.get("vo2MaxValue")
                    except Exception:
                        logger.debug("VO2Max unavailable for %s", day)

                    # -- Intensity minutes ------------------------------------
                    try:
                        im = garmin.get_intensity_minutes_data(day)
                        if im:
                            moderate = im.get("moderateIntensityMinutes", 0) or 0
                            vigorous = im.get("vigorousIntensityMinutes", 0) or 0
                            row["intensity_minutes"] = moderate + vigorous
                    except Exception:
                        logger.debug("Intensity minutes unavailable for %s", day)

                    # -- Floors (fallback if not in stats) --------------------
                    if not row.get("floors_climbed"):
                        try:
                            floors = garmin.get_floors(day)
                            if floors:
                                row["floors_climbed"] = floors.get("floorsAscended")
                        except Exception:
                            pass

                    row["raw_data"] = stats
                    client.table("health_daily_metrics").upsert(
                        row, on_conflict="user_id,date,source",
                    ).execute()
                    synced += 1
                except Exception:
                    logger.debug(
                        "Failed to sync daily stats for %s", day, exc_info=True
                    )

            update_last_sync(user_id, "garmin")
            return {"status": "ok", "synced": synced, "days": days}
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Garmin sync_daily_stats failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Sync: sleep
    # ------------------------------------------------------------------

    @staticmethod
    def sync_sleep(user_id: str, days: int = 7) -> dict:
        """Sync sleep data including stages from Garmin.

        Extracts total duration, sleep score, and per-stage durations
        (deep, light, REM, awake) from the dailySleepDTO.
        """
        try:
            garmin, _ = GarminSyncService._restore_session(user_id)
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

                    # Sleep stages (seconds → minutes)
                    def _s2m(key: str) -> float | None:
                        val = daily_dto.get(key)
                        return round(val / 60, 1) if val is not None else None

                    client.table("health_daily_metrics").upsert(
                        {
                            "user_id": user_id,
                            "date": day,
                            "source": "garmin",
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
                    logger.debug(
                        "Failed to sync sleep for %s", day, exc_info=True
                    )

            return {"status": "ok", "synced": synced, "days": days}
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.exception("Garmin sync_sleep failed for user %s", user_id)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    @staticmethod
    def check_connection(user_id: str) -> dict:
        """Check whether stored Garmin tokens are still valid.

        Returns:
            Dict with ``connected`` (bool), ``provider``, ``status``, and
            optional ``last_sync_at`` / ``provider_user_id`` fields.
        """
        try:
            _, token_row = GarminSyncService._restore_session(user_id)
            return {
                "connected": True,
                "provider": "garmin",
                "status": "active",
                "last_sync_at": token_row.get("last_sync_at"),
                "provider_user_id": token_row.get("provider_user_id"),
            }
        except ValueError:
            return {
                "connected": False,
                "provider": "garmin",
                "status": "disconnected",
            }
        except Exception:
            return {
                "connected": False,
                "provider": "garmin",
                "status": "error",
            }
