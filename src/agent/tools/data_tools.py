"""Data access tools -- read-only access to athlete data.

These are the equivalent of Claude Code's Read, Grep, Glob tools.
The agent uses these to inspect data BEFORE deciding what to do.

FIX vs Blueprint: Activity dicts use nested keys (heart_rate.avg, pace.avg_min_per_km,
zone_distribution) not flat keys (avg_heart_rate, avg_pace_min_km, hr_zone_distribution).
"""

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings


def register_data_tools(registry: ToolRegistry, user_model):
    """Register all data access tools."""
    _settings = get_settings()

    def get_athlete_profile() -> dict:
        """Get the current athlete profile."""
        profile = user_model.project_profile()
        profile["_has_activities"] = bool(profile.get("sports"))
        profile["_onboarding_complete"] = bool(
            profile.get("sports") and
            profile.get("goal", {}).get("event") and
            profile.get("constraints", {}).get("training_days_per_week")
        )
        return profile

    registry.register(Tool(
        name="get_athlete_profile",
        description=(
            "Get the athlete's current profile including sports, goals, constraints, "
            "and fitness data. Use this FIRST when you need to understand who the athlete "
            "is and what they want. Returns null fields for info not yet gathered."
        ),
        handler=get_athlete_profile,
        parameters={},
        category="data",
    ))

    def get_activities(limit: int = 10, sport: str = None, days: int = None) -> dict:
        """Get recent training activities."""
        from datetime import datetime, timedelta

        if _settings.use_supabase:
            from src.db import list_activities as db_list_activities
            activities = db_list_activities(_settings.agenticsports_user_id, limit=100)
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
        for act in activities:
            # Extract from nested structure (actual codebase format)
            hr_data = act.get("heart_rate", {}) or {}
            pace_data = act.get("pace", {}) or {}
            zone_data = act.get("zone_distribution", {}) or {}

            entry = {
                "date": act.get("start_time", "")[:10],
                "sport": act.get("sport", "unknown"),
                "sub_sport": act.get("sub_sport"),
                "duration_minutes": round(act.get("duration_seconds", 0) / 60, 1),
                "distance_km": round(act.get("distance_meters", 0) / 1000, 2) if act.get("distance_meters") else None,
                "avg_hr": hr_data.get("avg"),
                "max_hr": hr_data.get("max"),
                "avg_pace_min_km": pace_data.get("avg_min_per_km") or pace_data.get("avg_min_per_100m"),
                "trimp": act.get("trimp"),
                "hr_zones": zone_data if zone_data else None,
                "calories": act.get("calories"),
            }

            # Add power data if available
            power_data = act.get("power", {}) or {}
            if power_data.get("avg_watts"):
                entry["avg_watts"] = power_data["avg_watts"]
                entry["normalized_watts"] = power_data.get("normalized_watts")

            result["activities"].append(entry)

        return result

    registry.register(Tool(
        name="get_activities",
        description=(
            "Get recent training activities with optional filtering by sport or time range. "
            "Returns date, sport, duration, distance, HR, pace, TRIMP for each activity. "
            "Use this to understand what the athlete has been doing recently."
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
    ))

    def get_current_plan() -> dict:
        """Get the current active training plan."""
        if _settings.use_supabase:
            from src.db import get_active_plan
            plan_row = get_active_plan(_settings.agenticsports_user_id)
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

    registry.register(Tool(
        name="get_past_plans",
        description=(
            "Get a list of previously generated training plans with their dates, "
            "phases, session counts, and evaluation scores. Useful for understanding "
            "training history and progression."
        ),
        handler=get_past_plans,
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum plans to return (default 5)",
                },
            },
        },
        category="data",
    ))

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

    registry.register(Tool(
        name="get_beliefs",
        description=(
            "Get recorded beliefs about the athlete (scheduling, fitness, constraints, "
            "physical, motivation, history, preference, personality). "
            "Beliefs are things the coach has learned about the athlete through conversation. "
            "Use this to recall what you know before giving advice."
        ),
        handler=get_beliefs,
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category. Omit for all categories.",
                    "nullable": True,
                    "enum": ["scheduling", "fitness", "constraint", "physical",
                             "motivation", "history", "preference", "personality"],
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0-1.0, default 0.0)",
                },
            },
        },
        category="data",
    ))


def _parse_datetime(dt_str: str):
    """Parse an ISO datetime string, handling timezone-aware strings."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return datetime(2000, 1, 1)
