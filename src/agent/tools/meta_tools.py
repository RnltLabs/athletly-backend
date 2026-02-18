"""Meta tools -- agent self-management and sub-agents.

spawn_specialist is the equivalent of Claude Code's Task tool.
get_session_context provides conversation metadata.
"""

import json
from src.agent.llm import chat_completion
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry


def register_meta_tools(registry: ToolRegistry, user_model):
    """Register meta/utility tools."""

    def spawn_specialist(type: str, task: str, context: dict = None) -> dict:
        """Spawn a specialist sub-agent for complex analysis."""
        specialist_prompts = {
            "data_analyst": (
                "You are a sports data analyst. Analyze the provided training data "
                "and produce structured insights. Focus on: training load trends, "
                "recovery status, performance changes, and gaps. "
                "Respond with ONLY a valid JSON object."
            ),
            "domain_expert": (
                "You are a sports science expert and exercise physiologist. "
                "Given the athlete's sport(s) and goal, provide sport-specific "
                "training methodology guidance: periodization phase, energy systems, "
                "session types, and safety considerations. "
                "Respond with ONLY a valid JSON object."
            ),
            "safety_reviewer": (
                "You are a sports medicine safety reviewer. Analyze the athlete's "
                "profile and training for safety concerns: overtraining risk, "
                "injury risk, youth considerations, medical referral needs. "
                "Be thorough but not alarmist. "
                "Respond with ONLY a valid JSON object."
            ),
        }

        if type not in specialist_prompts:
            return {"error": f"Unknown specialist: {type}. Available: {list(specialist_prompts.keys())}"}

        context_str = json.dumps(context or {}, ensure_ascii=False, indent=2)
        prompt = f"TASK: {task}\n\nCONTEXT:\n{context_str}"

        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=specialist_prompts[type],
            temperature=0.3,
        )

        try:
            result = extract_json(response.choices[0].message.content.strip())
        except (ValueError, Exception):
            result = {"raw_response": response.choices[0].message.content.strip()[:2000]}

        return {"specialist": type, "result": result}

    registry.register(Tool(
        name="spawn_specialist",
        description=(
            "Spawn a specialist sub-agent for complex multi-step analysis. "
            "Available specialists:\n"
            "- data_analyst: Analyze training data trends, load, recovery\n"
            "- domain_expert: Sport-specific methodology and periodization guidance\n"
            "- safety_reviewer: Check for safety concerns, overtraining, medical referrals\n"
            "Use this when you need deep expertise for a specific aspect. "
            "Pass relevant context so the specialist has what it needs."
        ),
        handler=spawn_specialist,
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Specialist type",
                    "enum": ["data_analyst", "domain_expert", "safety_reviewer"],
                },
                "task": {
                    "type": "string",
                    "description": "What you want the specialist to analyze",
                },
                "context": {
                    "type": "object",
                    "description": "Relevant context for the specialist (profile, data, etc.)",
                    "nullable": True,
                },
            },
            "required": ["type", "task"],
        },
        category="meta",
    ))

    def get_session_context() -> dict:
        """Get conversation metadata."""
        profile = user_model.project_profile()
        return {
            "athlete_name": profile.get("name", "Athlete"),
            "sports": profile.get("sports", []),
            "has_plan": bool(profile.get("sports")),
            "onboarding_complete": bool(
                profile.get("sports") and
                profile.get("goal", {}).get("event") and
                profile.get("constraints", {}).get("training_days_per_week")
            ),
            "belief_count": len(user_model.get_active_beliefs()),
        }

    registry.register(Tool(
        name="get_session_context",
        description=(
            "Get metadata about the current session: athlete name, sports, "
            "whether onboarding is complete, belief count. "
            "Use this at the start of a session to understand context."
        ),
        handler=get_session_context,
        parameters={},
        category="meta",
    ))
