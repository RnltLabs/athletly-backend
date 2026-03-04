"""Onboarding tools -- agent tool to mark onboarding as complete.

The agent calls `complete_onboarding` after gathering all required profile
data during the onboarding conversation. This tool:
1. Validates that minimum requirements are met (sports, goal present)
2. Sets `profiles.onboarding_complete = true` in Supabase
3. Sets `user_model.meta["_onboarding_complete"] = True`
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_onboarding_tools(registry: ToolRegistry, user_model) -> None:
    """Register onboarding tools into the registry."""
    _settings = get_settings()

    def _get_user_id() -> str:
        if user_model is not None and hasattr(user_model, "user_id"):
            return user_model.user_id
        return _settings.agenticsports_user_id

    def complete_onboarding() -> dict:
        """Mark the onboarding as complete after validating minimum requirements."""
        profile = user_model.project_profile()

        # Validate minimum requirements
        missing = []
        if not profile.get("sports"):
            missing.append("sports")

        goal = profile.get("goal") or {}
        if isinstance(goal, dict) and not goal.get("event"):
            missing.append("goal")

        if missing:
            return {
                "status": "error",
                "error": f"Cannot complete onboarding: missing {', '.join(missing)}",
                "missing": missing,
            }

        # Gate 2: Validate configs + plan exist before allowing completion
        if _settings.use_supabase:
            try:
                from src.db.client import get_supabase

                uid = _get_user_id()

                schemas = get_supabase().table("agent_configs").select("id").eq(
                    "user_id", uid
                ).eq("config_type", "session_schema").limit(1).execute()
                if not schemas.data:
                    missing.append("session_schemas (call define_session_schema first)")

                metrics = get_supabase().table("agent_configs").select("id").eq(
                    "user_id", uid
                ).eq("config_type", "metric").limit(1).execute()
                if not metrics.data:
                    missing.append("metrics (call define_metric first)")

                plans = get_supabase().table("weekly_plans").select("id").eq(
                    "user_id", uid
                ).eq("status", "active").limit(1).execute()
                if not plans.data:
                    missing.append("training_plan (call create_training_plan + save_plan first)")
            except Exception:
                logger.warning("Config gate DB check failed", exc_info=True)

        if missing:
            return {
                "status": "error",
                "error": f"Cannot complete onboarding: missing {', '.join(missing)}",
                "missing": missing,
            }

        # Set onboarding_complete in Supabase profiles table
        if _settings.use_supabase:
            try:
                from src.db.client import get_supabase

                get_supabase().table("profiles").update(
                    {"onboarding_complete": True}
                ).eq("user_id", _get_user_id()).execute()
                logger.info("Set profiles.onboarding_complete=true for user %s", _get_user_id())
            except Exception:
                logger.warning("Failed to update profiles.onboarding_complete", exc_info=True)

        # Set flag in user_model meta (immutable update)
        user_model.meta = {**user_model.meta, "_onboarding_complete": True}
        user_model.save()

        logger.info("Onboarding complete for user %s", _get_user_id())
        return {"status": "success", "onboarding_complete": True}

    registry.register(Tool(
        name="complete_onboarding",
        description=(
            "Mark the onboarding process as complete. Call this ONLY after you have "
            "gathered the minimum required information (at least one sport and one goal), "
            "defined session schemas, metrics, eval criteria, created and saved the first "
            "training plan. This sets the user's onboarding_complete flag to true."
        ),
        handler=complete_onboarding,
        parameters={
            "type": "object",
            "properties": {},
        },
        category="onboarding",
    ))
