"""Training coach agent: generates weekly plans via LiteLLM."""

import json
from datetime import datetime
from pathlib import Path

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion
from src.agent.prompts import build_coach_system_prompt, build_plan_prompt

PLANS_DIR = Path(__file__).parent.parent.parent / "data" / "plans"


def generate_plan(
    profile: dict,
    beliefs: list[dict] | None = None,
    activities: list[dict] | None = None,
    relevant_episodes: list[dict] | None = None,
    user_id: str = "",
) -> dict:
    """Send athlete profile to LLM and return a structured training plan.

    Args:
        profile: Athlete profile dict (from UserModel.project_profile()).
        beliefs: Active beliefs to inject as coach's notes for personalization.
        activities: Optional activity list for data-derived target generation.
                    When provided, per-sport performance data is injected into
                    the prompt so the LLM can set athlete-specific targets.
        relevant_episodes: Past episode reflections to inform planning.
        user_id: User ID for loading session schemas from DB.

    Raises ValueError if the response is not valid JSON.
    """
    user_prompt = build_plan_prompt(
        profile, beliefs=beliefs, activities=activities,
        relevant_episodes=relevant_episodes,
    )

    response = chat_completion(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=build_coach_system_prompt(user_id),
        temperature=0.7,
    )

    text = response.choices[0].message.content.strip()
    return extract_json(text)


def save_plan(plan: dict) -> Path:
    """Save a training plan to data/plans/ with a timestamp filename."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = PLANS_DIR / f"plan_{timestamp}.json"
    path.write_text(json.dumps(plan, indent=2))
    return path
