"""Data access tools -- read-only access to athlete data.

These are the equivalent of Claude Code's Read, Grep, Glob tools.
The agent uses these to inspect data BEFORE deciding what to do.

Activity dicts may arrive in two formats:
- Flat columns from the DB (avg_hr, max_hr, avg_pace_min_km, etc.)
- Nested dicts from the file store (heart_rate.avg, pace.avg_min_per_km, etc.)
The accessors below try flat keys first, then fall back to nested.
"""

from src.agent.tools.registry import Tool, ToolRegistry
from src.agent.tools.format_helpers import pace_to_mmss, minutes_to_hms
from src.config import get_settings


def register_data_tools(registry: ToolRegistry, user_model):
    """Register all data access tools."""
    _settings = get_settings()

    def get_athlete_profile() -> dict:
        """Get the current athlete profile."""
        from src.utils.pace_format import decimal_min_to_mmss
        profile = user_model.project_profile()
        # Pre-format fitness decimals to a _pretty twin so the LLM cannot
        # misread the raw decimal as mm:ss (Sprint C fix). The decimal
        # stays in place for backwards compat / internal math.
        fitness = profile.get("fitness")
        if isinstance(fitness, dict):
            threshold_pretty = decimal_min_to_mmss(
                fitness.get("threshold_pace_min_km")
            )
            if threshold_pretty is not None:
                fitness["threshold_pace_pretty"] = threshold_pretty
        profile["_has_activities"] = bool(profile.get("sports"))
        profile["_onboarding_complete"] = bool(
            profile.get("sports") and
            profile.get("goal", {}).get("event") and
            profile.get("constraints", {}).get("training_days_per_week")
        )
        return profile

    _athlete_profile_description = (
        "Return structured profile: sports, goal (event/date/target_time), "
        "constraints (training_days_per_week, max_session_minutes, "
        "available_sports), fitness (VO2max, threshold_pace, weekly_volume, "
        "FTP), plus flags (_has_activities, _onboarding_complete). Read at "
        "start of any planning task. Avoid for free-form beliefs (use "
        "get_beliefs), activities (get_activities), or active plan "
        "(get_current_plan). Fields may be null in early onboarding."
    )

    registry.register(Tool(
        name="get_athlete_profile",
        description=_athlete_profile_description,
        handler=get_athlete_profile,
        parameters={},
        category="data",
        display_label="Lese dein Profil",
    ))

    # DEPRECATED: get_user_profile replaced by get_athlete_profile. NOT registered.
    # _DISABLED_get_user_profile_registration = lambda: registry.register(Tool(
    #     name="get_user_profile",
    #     description="Alias for get_athlete_profile. " + _athlete_profile_description,
    #     handler=get_athlete_profile,
    #     parameters={},
    #     category="data",
    # ))

    def get_activities(limit: int = 10, sport: str = None, days: int = None) -> dict:
        """Get recent training activities."""
        from datetime import datetime, timedelta

        if _settings.use_supabase:
            from src.db import list_activities as db_list_activities
            uid = user_model.user_id if user_model else _settings.agenticsports_user_id
            activities = db_list_activities(uid, limit=100)
        else:
            from src.tools.activity_store import list_activities
            activities = list_activities()

        if sport:
            activities = [a for a in activities if a.get("sport", "").lower() == sport.lower()]

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            activities = [
                a for a in activities
                if _parse_datetime(a.get("start_time", "2000-01-01")) > cutoff
            ]

        # Most recent first, apply limit
        activities = sorted(
            activities,
            key=lambda a: a.get("start_time", ""),
            reverse=True,
        )[:limit]

        result = {
            "count": len(activities),
            "activities": [],
        }

        # Sports for which pace is meaningful in the list view. Everything
        # else (strength, yoga, etc.) omits avg_pace_pretty to keep entries
        # tight - the agent can still fetch full pace data via
        # get_activity_details(activity_id).
        _PACE_SPORTS = {
            "running", "trail_running", "treadmill_running",
            "walking", "hiking",
            "cycling", "road_biking", "mountain_biking", "indoor_cycling",
            "swimming", "open_water_swimming", "lap_swimming",
            "rowing", "indoor_rowing",
        }

        for act in activities:
            pace_data = act.get("pace", {}) or {}
            duration_seconds = act.get("duration_seconds", 0) or 0
            distance_meters = act.get("distance_meters") or 0
            duration_min = round(duration_seconds / 60, 1)

            sport = act.get("sport", "unknown")
            include_pace = sport.lower() in _PACE_SPORTS

            entry: dict = {
                "id": act.get("id") or act.get("garmin_activity_id"),
                "date": act.get("start_time", "")[:10],
                "sport": sport,
                "duration_pretty": minutes_to_hms(duration_min),
                "distance_km": round(distance_meters / 1000, 2) if distance_meters else None,
            }

            if include_pace:
                pace_decimal = (
                    act.get("avg_pace_min_km")
                    or pace_data.get("avg_min_per_km")
                    or pace_data.get("avg_min_per_100m")
                )
                # Compute pretty pace from duration/distance directly to
                # avoid the ~0.5s precision loss of round(pace, 2). Falls
                # back to the rounded decimal if either is missing.
                if duration_seconds and distance_meters:
                    sec_per_km = (duration_seconds / distance_meters) * 1000
                    pretty_pace = f"{int(sec_per_km // 60)}:{int(round(sec_per_km % 60)):02d}"
                else:
                    pretty_pace = pace_to_mmss(pace_decimal)
                if pretty_pace:
                    entry["avg_pace_pretty"] = pretty_pace

            result["activities"].append(entry)

        return result

    registry.register(Tool(
        name="get_activities",
        description=(
            "Return a SLIM list of recent activities, newest first, with "
            "optional sport and days filters. Each item carries ONLY: id, "
            "date (YYYY-MM-DD), sport, duration_pretty (h:mm:ss), "
            "distance_km, and avg_pace_pretty (m:ss/km, only for "
            "pace-relevant sports). Use this to identify and pick "
            "activities. For HR zone distribution, training effect, "
            "power data, running form, elevation, TRIMP, calories, "
            "decimal pace/HR - call get_activity_details(activity_id). "
            "limit default 10 (use 30-50 for trends, 3-5 for last "
            "context); days e.g. 7 for last week."
        ),
        handler=get_activities,
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of activities to return (default 10)",
                },
                "sport": {
                    "type": "string",
                    "description": "Filter by sport (e.g., 'running', 'cycling'). Omit for all sports.",
                    "nullable": True,
                },
                "days": {
                    "type": "integer",
                    "description": "Only activities from the last N days. Omit for no time filter.",
                    "nullable": True,
                },
            },
        },
        category="data",
        display_label="Lese deine letzten Workouts",
    ))

    def get_activity_details(activity_id: str) -> dict:
        """Return all coaching-relevant fields for a single activity.

        Pulls from the raw provider payload so the agent can see metrics
        beyond the summary list: training effect (aerobic + anaerobic),
        HR zone distribution, power data, running form, elevation, GAP.
        """
        if not _settings.use_supabase:
            return {"error": "Supabase not configured"}

        from src.db.client import get_supabase
        sb = get_supabase()
        uid = user_model.user_id if user_model else _settings.agenticsports_user_id

        # Try by primary id first, then garmin_activity_id (string).
        row = None
        try:
            r = sb.table("activities").select("*").eq("id", activity_id).eq("user_id", uid).limit(1).execute()
            if r.data:
                row = r.data[0]
        except Exception:
            pass
        if row is None:
            try:
                r = sb.table("activities").select("*").eq("garmin_activity_id", str(activity_id)).eq("user_id", uid).limit(1).execute()
                if r.data:
                    row = r.data[0]
            except Exception:
                pass

        if row is None:
            return {"error": f"No activity found with id={activity_id}"}

        raw = row.get("raw_data") or {}
        duration_s = row.get("duration_seconds") or 0
        distance_m = row.get("distance_meters") or 0

        # Pretty pace from raw values (no rounding cascade).
        pretty_pace = None
        if duration_s and distance_m:
            sec_per_km = (duration_s / distance_m) * 1000
            pretty_pace = f"{int(sec_per_km // 60)}:{int(round(sec_per_km % 60)):02d}"

        # HR zone distribution: turn raw seconds-per-zone into pct + pretty.
        hr_zones_seconds = {
            f"zone_{i}": raw.get(f"hrTimeInZone_{i}")
            for i in range(1, 6)
            if raw.get(f"hrTimeInZone_{i}") is not None
        }
        hr_total = sum(hr_zones_seconds.values()) if hr_zones_seconds else 0
        hr_zones_pct = {
            k: round((v / hr_total) * 100, 1) for k, v in hr_zones_seconds.items()
        } if hr_total else None

        power_zones_seconds = {
            f"zone_{i}": raw.get(f"powerTimeInZone_{i}")
            for i in range(1, 6)
            if raw.get(f"powerTimeInZone_{i}") is not None
        }

        def _msps_to_kmh(speed_mps: float | None) -> float | None:
            if speed_mps is None:
                return None
            try:
                return round(float(speed_mps) * 3.6, 1)
            except (TypeError, ValueError):
                return None

        avg_speed = raw.get("averageSpeed")
        max_speed = raw.get("maxSpeed")
        gap_speed = raw.get("avgGradeAdjustedSpeed")

        return {
            "id": row.get("id"),
            "garmin_activity_id": row.get("garmin_activity_id"),
            "date": (row.get("start_time") or "")[:10],
            "sport": row.get("sport"),
            "duration_pretty": minutes_to_hms(duration_s / 60) if duration_s else None,
            "distance_km": round(distance_m / 1000, 2) if distance_m else None,
            "avg_pace_pretty": pretty_pace,
            "max_speed_kmh": _msps_to_kmh(max_speed),
            "avg_grade_adjusted_speed_kmh": _msps_to_kmh(gap_speed),
            "avg_hr": row.get("avg_hr"),
            "max_hr": row.get("max_hr"),
            "hr_zones_seconds": hr_zones_seconds or None,
            "hr_zones_pct": hr_zones_pct,
            "training_effect": {
                "aerobic": row.get("training_effect") or raw.get("aerobicTrainingEffect"),
                "anaerobic": raw.get("anaerobicTrainingEffect"),
                "label": raw.get("trainingEffectLabel"),
                "aerobic_message": raw.get("aerobicTrainingEffectMessage"),
                "anaerobic_message": raw.get("anaerobicTrainingEffectMessage"),
            },
            "vo2max": row.get("vo2max_activity") or raw.get("vO2MaxValue"),
            "activity_training_load": raw.get("activityTrainingLoad"),
            "calories": row.get("calories"),
            "power": {
                "avg": raw.get("avgPower"),
                "normalized": raw.get("normPower"),
                "max": raw.get("maxPower"),
                "zones_seconds": power_zones_seconds or None,
            } if raw.get("avgPower") else None,
            "running_form": {
                "avg_cadence_spm": raw.get("averageRunningCadenceInStepsPerMinute"),
                "max_cadence_spm": raw.get("maxRunningCadenceInStepsPerMinute"),
                "avg_stride_length_cm": raw.get("avgStrideLength"),
                "avg_vertical_oscillation_cm": raw.get("avgVerticalOscillation"),
                "avg_ground_contact_time_ms": raw.get("avgGroundContactTime"),
            } if raw.get("averageRunningCadenceInStepsPerMinute") else None,
            "elevation": {
                "gain_m": row.get("elevation_gain_m") or raw.get("elevationGain"),
                "loss_m": raw.get("elevationLoss"),
                "min_m": raw.get("minElevation"),
                "max_m": raw.get("maxElevation"),
                "avg_m": raw.get("avgElevation"),
            } if (row.get("elevation_gain_m") or raw.get("elevationGain")) else None,
            "moving_time_pretty": minutes_to_hms(raw.get("movingDuration") / 60) if raw.get("movingDuration") else None,
        }

    registry.register(Tool(
        name="get_activity_details",
        description=(
            "Pull all coaching-relevant fields for ONE activity by id. "
            "Returns: pace, training effect (aerobic + anaerobic + label), "
            "HR zone distribution (seconds + percentages), power data + "
            "zones, running form (cadence, stride, vertical oscillation), "
            "elevation profile (gain, loss, min/max/avg), VO2max from "
            "device. Use this to deep-dive ONE workout - e.g. analyse a "
            "race, evaluate a tempo run, or diagnose a hard week. Do NOT "
            "call for every activity in a list - get_activities is the "
            "compact summary tool."
        ),
        handler=get_activity_details,
        parameters={
            "type": "object",
            "properties": {
                "activity_id": {
                    "type": "string",
                    "description": "Activity id (uuid from id column or string garmin_activity_id).",
                },
            },
            "required": ["activity_id"],
        },
        category="data",
        display_label="Schaue mir den Lauf genauer an",
    ))

    def get_current_plan() -> dict:
        """Get the current active training plan."""
        if _settings.use_supabase:
            from src.db import get_active_plan
            uid = user_model.user_id if user_model else _settings.agenticsports_user_id
            plan_row = get_active_plan(uid)
            if not plan_row:
                return {"plan": None, "message": "No training plans exist yet."}
            plan_data = plan_row.get("plan_data", {})
            return {
                "plan": plan_data,
                "id": plan_row["id"],
                "sessions_count": len(plan_data.get("sessions", [])),
                "training_phase": plan_data.get("training_phase", "unknown"),
            }
        else:
            from pathlib import Path
            import json
            plans_dir = Path("data/plans")
            if not plans_dir.exists():
                return {"plan": None, "message": "No training plans exist yet."}
            plan_files = sorted(plans_dir.glob("plan_*.json"), reverse=True)
            if not plan_files:
                return {"plan": None, "message": "No training plans exist yet."}
            latest = json.loads(plan_files[0].read_text())
            return {
                "plan": latest,
                "file": str(plan_files[0]),
                "sessions_count": len(latest.get("sessions", [])),
                "training_phase": latest.get("training_phase", "unknown"),
            }

    registry.register(Tool(
        name="get_current_plan",
        description=(
            "Get the most recent training plan. Returns the full plan with sessions, "
            "phase, and metadata. Returns null if no plan exists yet."
        ),
        handler=get_current_plan,
        parameters={},
        category="data",
        display_label="Lese deinen aktuellen Plan",
    ))

    def get_past_plans(limit: int = 5) -> dict:
        """Get previously generated plans."""
        if _settings.use_supabase:
            from src.db import list_plans
            rows = list_plans(_settings.agenticsports_user_id, limit=limit)
            plans = []
            for r in rows:
                pd = r.get("plan_data", {})
                plans.append({
                    "id": r["id"],
                    "date": r.get("created_at", "")[:10],
                    "phase": pd.get("training_phase", "unknown"),
                    "sessions": len(pd.get("sessions", [])),
                    "evaluation_score": r.get("evaluation_score"),
                })
            return {"plans": plans, "count": len(plans)}
        else:
            from pathlib import Path
            import json
            plans_dir = Path("data/plans")
            if not plans_dir.exists():
                return {"plans": [], "count": 0}
            plan_files = sorted(plans_dir.glob("plan_*.json"), reverse=True)[:limit]
            plans = []
            for f in plan_files:
                try:
                    plan = json.loads(f.read_text())
                    plans.append({
                        "file": f.name,
                        "date": f.stem.replace("plan_", ""),
                        "phase": plan.get("training_phase", "unknown"),
                        "sessions": len(plan.get("sessions", [])),
                        "evaluation_score": plan.get("_evaluation", {}).get("score"),
                    })
                except (json.JSONDecodeError, OSError):
                    continue
            return {"plans": plans, "count": len(plans)}

    # DEPRECATED: get_past_plans replaced by get_plan_history. NOT registered.
    # registry.register(Tool(
    #     name="get_past_plans",
    #     description=(
    #         "Get a list of previously generated training plans with their dates, "
    #         "phases, session counts, and evaluation scores. Useful for understanding "
    #         "training history and progression."
    #     ),
    #     handler=get_past_plans,
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "limit": {
    #                 "type": "integer",
    #                 "description": "Maximum plans to return (default 5)",
    #             },
    #         },
    #     },
    #     category="data",
    # ))

    def get_beliefs(category: str = None, min_confidence: float = 0.0) -> dict:
        """Get current beliefs about the athlete."""
        beliefs = user_model.get_active_beliefs(min_confidence=min_confidence)

        if category:
            beliefs = [b for b in beliefs if b.get("category") == category]

        return {
            "count": len(beliefs),
            "beliefs": [
                {
                    "id": b.get("id"),
                    "text": b.get("text"),
                    "category": b.get("category"),
                    "confidence": round(b.get("confidence", 0), 2),
                    "source": b.get("source", "conversation"),
                }
                for b in beliefs
            ],
        }

    # DEPRECATED: get_beliefs - beliefs table dropped. NOT registered.
    # registry.register(Tool(
    #     name="get_beliefs",
    #     description=(
    #         "Get recorded beliefs about the athlete (scheduling, fitness, constraints, "
    #         "physical, motivation, history, preference, personality). "
    #         "Beliefs are things the coach has learned about the athlete through conversation. "
    #         "Use this to recall what you know before giving advice."
    #     ),
    #     handler=get_beliefs,
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "category": {
    #                 "type": "string",
    #                 "description": "Filter by category. Omit for all categories.",
    #                 "nullable": True,
    #                 "enum": ["scheduling", "fitness", "constraint", "physical",
    #                          "motivation", "history", "preference", "personality"],
    #             },
    #             "min_confidence": {
    #                 "type": "number",
    #                 "description": "Minimum confidence threshold (0.0-1.0, default 0.0)",
    #             },
    #         },
    #     },
    #     category="data",
    # ))


def _parse_datetime(dt_str: str):
    """Parse an ISO datetime string, handling timezone-aware strings."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return datetime(2000, 1, 1)
