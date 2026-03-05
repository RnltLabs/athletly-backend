"""Plan adjustment tools -- modify an existing training plan.

Unlike ``create_training_plan`` which generates a plan from scratch,
``adjust_plan`` reads the current active plan, applies natural-language
adjustment instructions via an LLM call, and saves the modified version.

This follows the NanoBot principle: the tool is generic infrastructure.
All sport-specific reasoning happens in the LLM -- no hardcoded rules,
no hardcoded formulas.

Tool registered:
    - adjust_plan: Read current plan, apply adjustments, save result.
"""

import json
import logging
from datetime import datetime

from src.agent.llm import chat_completion
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)

_MAX_INSTRUCTION_LENGTH = 2000


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_plan_adjust_tools(registry: ToolRegistry, user_model) -> None:
    """Register plan adjustment tools bound to the given user_model."""
    _settings = get_settings()

    def _resolve_user_id() -> str:
        if hasattr(user_model, "user_id"):
            return user_model.user_id
        return _settings.agenticsports_user_id

    def adjust_plan(
        instructions: str,
        plan_id: str = "",
    ) -> dict:
        """Adjust an existing training plan based on natural-language instructions.

        Reads the current active plan (or a specific plan by ID), sends it
        to the LLM together with the adjustment instructions, and saves
        the modified plan as a new version.

        The LLM decides all modifications -- this tool contains zero
        sport-specific logic.
        """
        if not instructions or not instructions.strip():
            return {"error": "instructions must not be empty"}

        if len(instructions) > _MAX_INSTRUCTION_LENGTH:
            return {
                "error": f"instructions too long (max {_MAX_INSTRUCTION_LENGTH} characters)",
            }

        if not _settings.use_supabase:
            return {"error": "Supabase not configured"}

        user_id = _resolve_user_id()

        # 1. Load the plan to adjust
        current_plan = _load_plan(user_id, plan_id.strip() if plan_id else "")
        if current_plan is None:
            return {"error": "No active plan found. Create a plan first."}

        plan_data = current_plan.get("plan_data", current_plan)
        source_plan_id = current_plan.get("id", "")

        # 2. Build context for the LLM
        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)

        # 3. Ask the LLM to produce the adjusted plan
        try:
            adjusted_plan = _generate_adjustment(
                plan_data=plan_data,
                instructions=instructions.strip(),
                profile=profile,
                beliefs=beliefs,
            )
        except Exception as exc:
            logger.error("adjust_plan LLM call failed: %s", exc)
            return {"error": f"Plan adjustment failed: {exc}"}

        # 4. Annotate the adjusted plan with metadata
        adjusted_plan["_adjusted_at"] = datetime.now().isoformat()
        adjusted_plan["_adjustment_instructions"] = instructions.strip()
        adjusted_plan["_source_plan_id"] = source_plan_id

        # 5. Save as new active plan
        try:
            from src.db.plans_db import store_plan

            row = store_plan(user_id, adjusted_plan)
            return {
                "adjusted": True,
                "new_plan_id": row["id"],
                "source_plan_id": source_plan_id,
                "plan": adjusted_plan,
            }
        except Exception as exc:
            logger.error("adjust_plan save failed: %s", exc)
            return {
                "error": f"Adjustment generated but save failed: {exc}",
                "plan": adjusted_plan,
            }

    registry.register(Tool(
        name="adjust_plan",
        description=(
            "Adjust an existing training plan based on athlete feedback or "
            "changed circumstances. Unlike create_training_plan (which generates "
            "from scratch), this reads the current plan and modifies specific "
            "aspects while preserving the overall structure. "
            "Examples: 'Tuesday is too hard, make it easier', "
            "'Add a rest day on Wednesday', 'Shift long run to Sunday'. "
            "The adjusted plan is saved as a new version automatically."
        ),
        handler=adjust_plan,
        parameters={
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what to change in the plan. "
                        "Be specific about which sessions, days, or aspects to modify."
                    ),
                },
                "plan_id": {
                    "type": "string",
                    "description": (
                        "ID of a specific plan to adjust. "
                        "Leave empty to adjust the current active plan."
                    ),
                    "nullable": True,
                },
            },
            "required": ["instructions"],
        },
        category="planning",
    ))


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------


def _load_plan(user_id: str, plan_id: str) -> dict | None:
    """Load a plan from the database.

    If plan_id is provided, loads that specific plan.
    Otherwise loads the current active plan.

    Returns the plan row dict or None.
    """
    from src.db.plans_db import get_active_plan

    if plan_id:
        from src.db.client import get_supabase

        result = (
            get_supabase()
            .table("plans")
            .select("*")
            .eq("id", plan_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        return result.data if result is not None else None

    return get_active_plan(user_id)


def _generate_adjustment(
    plan_data: dict,
    instructions: str,
    profile: dict,
    beliefs: list[dict],
) -> dict:
    """Call the LLM to produce an adjusted version of the plan.

    The LLM receives the full current plan, the athlete profile/beliefs
    for context, and the adjustment instructions. It returns the complete
    modified plan as JSON.

    No sport-specific rules are injected here -- the LLM reasons about
    the domain autonomously.

    Args:
        plan_data: The current plan JSON.
        instructions: What to change.
        profile: Athlete profile dict.
        beliefs: Active beliefs list.

    Returns:
        The adjusted plan as a dict.

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
    """
    belief_lines = "\n".join(
        f"- [{b.get('category', '?')}] {b.get('text', '')}"
        for b in beliefs
    ) if beliefs else "(no beliefs recorded yet)"

    system_prompt = (
        "You are a plan adjustment assistant. You receive an existing training "
        "plan and instructions for how to modify it. Your job is to apply the "
        "requested changes while keeping the rest of the plan intact and "
        "coherent. Return ONLY the complete adjusted plan as valid JSON — "
        "same structure as the input plan. Do not add commentary outside the JSON."
    )

    user_prompt = (
        f"CURRENT PLAN:\n{json.dumps(plan_data, indent=2, ensure_ascii=False)}\n\n"
        f"ATHLETE PROFILE:\n{json.dumps(profile, indent=2, ensure_ascii=False)}\n\n"
        f"KNOWN BELIEFS:\n{belief_lines}\n\n"
        f"ADJUSTMENT INSTRUCTIONS:\n{instructions}\n\n"
        "Return the complete adjusted plan as JSON. Preserve the overall "
        "structure. Only change what the instructions ask for."
    )

    response = chat_completion(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.4,
    )

    raw_content = response.choices[0].message.content.strip()
    adjusted = extract_json(raw_content)

    if not isinstance(adjusted, dict):
        raise ValueError(
            f"LLM returned non-dict type ({type(adjusted).__name__}) "
            "instead of a plan object"
        )

    return adjusted
