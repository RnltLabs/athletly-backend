"""Plan Adapter -- normalize agent plan output to weekly_plans.days JSONB format.

The agent produces free-form plan dicts. This adapter enforces the schema
that the weekly_plans table expects, validating required fields and
normalising values before persistence.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
VALID_INTENSITIES = frozenset({"low", "moderate", "high"})


def _unwrap_agent_plan(plan: dict) -> dict:
    """Unwrap LLM wrapper structures before adapting to weekly format."""
    # {"result": "<json string>"} wrapper
    if list(plan.keys()) == ["result"] and isinstance(plan.get("result"), str):
        from src.agent.json_utils import extract_json
        try:
            inner = extract_json(plan["result"])
            if isinstance(inner, dict):
                logger.info("plan_adapter: unwrapped 'result' string wrapper")
                return _unwrap_agent_plan(inner)
        except (ValueError, TypeError):
            pass
    return plan


def adapt_plan_to_weekly_format(agent_plan: dict) -> dict:
    """Convert an agent plan dict to the weekly_plans.days JSONB format.

    Expected input (any shape the agent produces):
        {
            "monday": {"sessions": [{"type": "run", "intensity": "moderate", ...}]},
            "tuesday": {"sessions": []},   # rest day
            ...
        }
        OR
        {
            "days": {"monday": {"sessions": [...]}, ...}
        }

    Returns a validated dict keyed by lowercase day names. Every day in
    VALID_DAYS is present; missing days become rest days (sessions: []).

    Raises:
        ValueError: if a session is missing required fields or has invalid values.
    """
    # Unwrap any {"result": "..."} wrappers from LLM drift
    agent_plan = _unwrap_agent_plan(agent_plan)
    raw_days = _extract_days(agent_plan)
    result: dict = {}

    for day in VALID_DAYS:
        day_data = raw_days.get(day, {})
        sessions = day_data.get("sessions", [])
        validated = [_validate_session(s, day) for s in sessions]
        result[day] = {"sessions": validated}

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_days(agent_plan: dict) -> dict:
    """Pull the day-keyed dict out of whatever shape the agent returned."""
    # If the plan has a top-level "days" key, unwrap it
    if "days" in agent_plan and isinstance(agent_plan["days"], dict):
        return {k.lower(): v for k, v in agent_plan["days"].items()}

    # If the top-level keys are day names, use them directly
    day_keys = {k.lower() for k in agent_plan} & set(VALID_DAYS)
    if day_keys:
        return {k.lower(): v for k, v in agent_plan.items() if k.lower() in VALID_DAYS}

    # Use the shared session extractor to handle all LLM schema variants
    from src.agent.plan_evaluator import extract_sessions_from_plan
    sessions = extract_sessions_from_plan(agent_plan)
    if sessions:
        return _distribute_sessions_to_days(sessions)

    return {}


def _validate_session(session: dict, day: str) -> dict:
    """Validate and normalise a single session dict.

    Raises ValueError on missing required fields or invalid intensity.
    Returns a new dict (immutable pattern — never mutates input).
    """
    session_type = session.get("type") or session.get("session_type")
    if not session_type:
        raise ValueError(f"Session on {day} is missing 'type' field: {session}")

    intensity = (session.get("intensity") or "moderate").lower().strip()
    if intensity not in VALID_INTENSITIES:
        raise ValueError(
            f"Session on {day} has invalid intensity '{intensity}'. "
            f"Must be one of: {sorted(VALID_INTENSITIES)}"
        )

    return {
        "type": str(session_type).lower().strip(),
        "intensity": intensity,
        "duration_min": int(session.get("duration_min") or session.get("duration_minutes") or 0),
        "description": str(session.get("description") or ""),
        **{
            k: v for k, v in session.items()
            if k not in {"type", "session_type", "intensity", "duration_min",
                         "duration_minutes", "description"}
        },
    }


def _distribute_sessions_to_days(sessions: list[dict]) -> dict:
    """Best-effort: assign sessions to days using their 'day' field or sequentially."""
    days: dict = {d: {"sessions": []} for d in VALID_DAYS}

    for session in sessions:
        day = (session.get("day") or session.get("d") or "").lower().strip()
        if day in days:
            days[day]["sessions"].append(session)
        else:
            # Assign to the first day that has no session yet
            for candidate in VALID_DAYS:
                if not days[candidate]["sessions"]:
                    days[candidate]["sessions"].append(session)
                    break

    return days
