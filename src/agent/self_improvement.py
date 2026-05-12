"""Self-Improvement-Cycle: the coach learns from outcomes.

The visionplan.md describes a five-step cycle the coach runs in the
background:

  Discover -> Define -> Compute -> Verify -> Adjust

This module implements the trigger points that fire it:

1. **After every plan save**: auto-evaluate the plan, flag low scores
   so the coach can revise next turn.
2. **Daily heartbeat check**: compute plan compliance, update beliefs
   when adherence is consistently low or fitness drifts off projection.
3. **After every Garmin sync**: trajectory-check the goal projection
   and surface anomalies (HRV crash, RHR spike, missed-key-session).

Each trigger is intentionally narrow. The "intelligence" lives in
beliefs and update_profile calls, not in a giant orchestration prompt.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# Score below which a saved plan should be flagged for revision.
LOW_PLAN_SCORE_THRESHOLD = 70


# ---------------------------------------------------------------------------
# Trigger 1: post-save plan auto-evaluation
# ---------------------------------------------------------------------------


def post_save_plan_evaluation(
    args: dict, result: Any, ctx: "Any"
) -> None:  # uses HookContext, typed loose to avoid import cycle
    """Post-hook for save_plan: evaluate quality, persist score back.

    Runs synchronously - the next chat turn will see the updated
    evaluation_score/feedback on the plan row. Errors are swallowed
    (logged) so the save itself is never retroactively broken.
    """
    if not isinstance(result, dict) or not result.get("saved"):
        return
    plan_id = result.get("id")
    if not plan_id:
        return

    try:
        from src.db.client import get_supabase

        client = get_supabase()
        row = (
            client.table("plans")
            .select("plan_data, evaluation_score")
            .eq("id", plan_id)
            .execute()
        )
        if not row.data:
            return
        plan_data = row.data[0].get("plan_data") or {}
        existing_score = row.data[0].get("evaluation_score")
        # Skip if the agent already attached an evaluation (manual path).
        if existing_score is not None:
            return

        from src.agent.plan_evaluator import evaluate_plan

        profile = ctx.user_model.project_profile()
        beliefs = ctx.user_model.get_active_beliefs(min_confidence=0.6)
        eval_result = evaluate_plan(
            plan=plan_data,
            profile=profile,
            user_id=ctx.user_id,
            beliefs=beliefs,
        )
        # evaluate_plan returns a PlanEvaluation TypedDict or dataclass
        if hasattr(eval_result, "__dict__"):
            score = getattr(eval_result, "score", None)
            feedback_raw = (
                getattr(eval_result, "feedback", None)
                or getattr(eval_result, "issues", None)
            )
        else:
            score = eval_result.get("score") if isinstance(eval_result, dict) else None
            feedback_raw = (
                eval_result.get("feedback") or eval_result.get("issues")
                if isinstance(eval_result, dict) else None
            )
        feedback = feedback_raw
        if isinstance(feedback, list):
            feedback = "; ".join(str(f) for f in feedback)

        client.table("plans").update({
            "evaluation_score": score,
            "evaluation_feedback": feedback,
        }).eq("id", plan_id).execute()

        logger.info(
            "Auto-evaluated plan %s for user %s: score=%s",
            plan_id, ctx.user_id, score,
        )
    except Exception:
        logger.exception("post_save_plan_evaluation failed")


# ---------------------------------------------------------------------------
# Trigger 2: heartbeat compliance + trajectory check
# ---------------------------------------------------------------------------


def heartbeat_self_improvement_tick(user_id: str) -> dict:
    """One self-improvement tick for *user_id*. Idempotent.

    Returns a small dict describing what fired - logged but not surfaced
    in chat unless one of the sub-checks raises a flag.
    """
    summary: dict[str, Any] = {"user_id": user_id, "actions": []}

    try:
        compliance_action = _check_plan_compliance(user_id)
        if compliance_action:
            summary["actions"].append(compliance_action)
    except Exception:
        logger.exception("compliance check failed")

    try:
        trajectory_action = _check_trajectory_anomaly(user_id)
        if trajectory_action:
            summary["actions"].append(trajectory_action)
    except Exception:
        logger.exception("trajectory check failed")

    return summary


def _check_plan_compliance(user_id: str) -> dict | None:
    """Compare last 7 days of activities to planned sessions.

    Flags a belief when compliance < 60% three weeks in a row.
    """
    from src.db.client import get_supabase

    client = get_supabase()

    plan_res = (
        client.table("plans")
        .select("plan_data, created_at")
        .eq("user_id", user_id)
        .eq("active", True)
        .limit(1)
        .execute()
    )
    if not plan_res.data:
        return None
    plan_data = plan_res.data[0].get("plan_data") or {}
    planned_sessions = plan_data.get("sessions") or []
    if not planned_sessions:
        return None

    since = (date.today() - timedelta(days=7)).isoformat()
    act_res = (
        client.table("activities")
        .select("sport, start_time")
        .eq("user_id", user_id)
        .gte("start_time", since)
        .execute()
    )
    completed_count = len(act_res.data or [])
    expected = len(planned_sessions)
    if expected == 0:
        return None
    compliance = completed_count / expected

    if compliance < 0.6:
        try:
            client.table("beliefs").insert({
                "user_id": user_id,
                "text": (
                    f"Plan compliance last 7 days: "
                    f"{completed_count}/{expected} sessions completed "
                    f"({compliance*100:.0f}%). Consider lighter plan or "
                    "investigate barriers."
                ),
                "category": "fitness",
                "confidence": 0.85,
                "source": "self_improvement",
                "active": True,
            }).execute()
        except Exception:
            logger.exception("could not insert compliance belief")
        return {
            "type": "compliance_low",
            "completed": completed_count,
            "expected": expected,
            "ratio": compliance,
        }
    return None


def _check_trajectory_anomaly(user_id: str) -> dict | None:
    """Detect recent HRV / RHR anomaly suggesting accumulated fatigue.

    Compares last 3 days to prior 7. Flags belief if HRV drops >15%
    or RHR rises >7%.
    """
    from src.db.client import get_supabase

    client = get_supabase()
    end_day = date.today()
    recent_start = (end_day - timedelta(days=3)).isoformat()
    prior_start = (end_day - timedelta(days=10)).isoformat()

    res = (
        client.table("health_daily_metrics")
        .select("date, hrv_avg, resting_heart_rate")
        .eq("user_id", user_id)
        .gte("date", prior_start)
        .order("date")
        .execute()
    )
    rows = res.data or []
    if len(rows) < 5:
        return None
    recent = [r for r in rows if r.get("date") >= recent_start]
    prior = [r for r in rows if r.get("date") < recent_start]
    if not recent or not prior:
        return None

    def _avg(lst: list[dict], key: str) -> float | None:
        vals = [r.get(key) for r in lst if r.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    recent_hrv = _avg(recent, "hrv_avg")
    prior_hrv = _avg(prior, "hrv_avg")
    recent_rhr = _avg(recent, "resting_heart_rate")
    prior_rhr = _avg(prior, "resting_heart_rate")

    flags: list[str] = []
    if recent_hrv and prior_hrv and prior_hrv > 0:
        delta = (recent_hrv - prior_hrv) / prior_hrv * 100
        if delta < -15:
            flags.append(f"HRV dropped {abs(delta):.0f}% vs 7-day avg")
    if recent_rhr and prior_rhr and prior_rhr > 0:
        delta = (recent_rhr - prior_rhr) / prior_rhr * 100
        if delta > 7:
            flags.append(f"RHR rose {delta:.0f}% vs 7-day avg")

    if not flags:
        return None

    text = (
        "Recovery anomaly: "
        + "; ".join(flags)
        + ". Consider lighter intensity or extra recovery day."
    )
    try:
        client.table("beliefs").insert({
            "user_id": user_id,
            "text": text,
            "category": "fitness",
            "confidence": 0.85,
            "source": "self_improvement",
            "active": True,
        }).execute()
    except Exception:
        logger.exception("could not insert anomaly belief")
    return {"type": "recovery_anomaly", "flags": flags}
