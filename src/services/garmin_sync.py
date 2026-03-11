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
            garmin.garth.loads(token_row["token_data"]["garth_tokens"])
            garmin.login()  # refresh tokens if needed
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

        Args:
            user_id: Supabase user UUID.
            days: How many days back to fetch (1–30).

        Returns:
            Summary dict with ``status``, ``synced``, ``skipped``, ``days``.
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

                row = {
                    "user_id": user_id,
                    "sport": act.get("activityType", {}).get("typeKey", "unknown"),
                    "start_time": act.get("startTimeLocal"),
                    "duration_seconds": int(act.get("duration", 0)),
                    "distance_meters": act.get("distance"),
                    "avg_hr": act.get("averageHR"),
                    "max_hr": act.get("maxHR"),
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
        """Sync daily health stats (steps, HR, stress) from Garmin.

        Args:
            user_id: Supabase user UUID.
            days: How many days back to fetch (1–30).

        Returns:
            Summary dict with ``status``, ``synced``, ``days``.
        """
        try:
            garmin, _ = GarminSyncService._restore_session(user_id)
            synced = 0

            for i in range(days):
                day = (
                    datetime.now(timezone.utc).date() - timedelta(days=i)
                ).isoformat()
                try:
                    stats = garmin.get_stats(day)
                    if stats:
                        from src.db.client import get_supabase
                        client = get_supabase()
                        client.table("health_daily_metrics").upsert(
                            {
                                "user_id": user_id,
                                "date": day,
                                "source": "garmin",
                                "resting_heart_rate": stats.get("restingHeartRate"),
                                "steps": stats.get("totalSteps"),
                                "stress_avg": stats.get("averageStressLevel"),
                                "hrv_avg": (
                                    stats.get("hrvSummary", {}) or {}
                                ).get("weeklyAvg"),
                                "body_battery_high": stats.get(
                                    "bodyBatteryChargedValue"
                                ),
                                "body_battery_low": stats.get(
                                    "bodyBatteryDrainedValue"
                                ),
                                "active_calories": stats.get(
                                    "activeKilocalories"
                                ),
                                "total_calories": stats.get(
                                    "totalKilocalories"
                                ),
                                "raw_data": stats,
                            },
                            on_conflict="user_id,date,source",
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
        """Sync sleep data from Garmin.

        Args:
            user_id: Supabase user UUID.
            days: How many days back to fetch (1–30).

        Returns:
            Summary dict with ``status``, ``synced``, ``days``.
        """
        try:
            garmin, _ = GarminSyncService._restore_session(user_id)
            synced = 0

            for i in range(days):
                day = (
                    datetime.now(timezone.utc).date() - timedelta(days=i)
                ).isoformat()
                try:
                    sleep = garmin.get_sleep_data(day)
                    if sleep:
                        from src.db.client import get_supabase
                        client = get_supabase()
                        daily_dto = sleep.get("dailySleepDTO", {})
                        sleep_score = (
                            daily_dto.get("sleepScores", {})
                            .get("overall", {})
                            .get("value")
                        )
                        sleep_seconds = daily_dto.get("sleepTimeSeconds")
                        sleep_minutes = (
                            sleep_seconds / 60
                            if sleep_seconds is not None
                            else None
                        )
                        client.table("health_daily_metrics").upsert(
                            {
                                "user_id": user_id,
                                "date": day,
                                "source": "garmin",
                                "sleep_score": sleep_score,
                                "sleep_duration_minutes": sleep_minutes,
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
