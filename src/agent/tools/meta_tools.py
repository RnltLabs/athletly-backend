"""Meta tools -- agent self-management and sub-agents.

spawn_subagent is the equivalent of Claude Code's Task tool with the
general-purpose subagent_type: a focused, tool-using agent that runs a
short loop for a self-contained task and returns a synthesized answer.

spawn_specialist is a lighter-weight single-shot LLM call for narrow
analytic tasks where you just need a structured opinion (no tool use).
"""

import json
import logging

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion
from src.agent.tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


# Hard limit on subagent iterations to bound cost / latency.
SUBAGENT_MAX_ROUNDS = 8


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
            "Single-shot LLM call to a specialist persona (no tools). Returns "
            "structured JSON opinion on a narrow analytic question. Cheaper "
            "than spawn_subagent (no tool loop). Types: data_analyst (interpret "
            "training data), domain_expert (sports-science methodology check), "
            "safety_reviewer (injury/overtraining/youth risk sweep). Use when "
            "you have data in hand and want a focused lens. Avoid when you need "
            "external info (use spawn_subagent). CRITICAL: pass everything via "
            "context dict; specialist sees nothing else."
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

    # ------------------------------------------------------------------
    # spawn_subagent - generic tool-using subagent (Claude Code pattern)
    # ------------------------------------------------------------------

    def spawn_subagent(
        task: str,
        tools_scope: str = "research",
        max_rounds: int | None = None,
        context: dict | None = None,
    ) -> dict:
        """Spawn a general-purpose subagent with restricted tool access.

        The subagent runs its own LLM + tool loop until it produces a
        final non-tool answer (or hits ``max_rounds``). Returns the
        synthesized answer plus a short trace of tool calls used. The
        main agent gets only the result - the subagent's intermediate
        steps do NOT bloat the main context.

        tools_scope determines which tools the subagent gets:
        - "research" (default): web_search + web_fetch only. Best for
          looking up external facts: races, methodologies, products,
          rule changes, current information.
        - "readonly": web_search + web_fetch + read-only DB tools
          (get_activities, get_daily_metrics, ...). Best for deep
          analysis on existing data without polluting main context.

        Returns:
            {"result": str, "tool_calls": [{"tool", "args"}], "rounds": int}
            or {"error": str} if the subagent could not produce a result.
        """
        from src.agent.tools.registry import ToolRegistry as _Registry
        from src.agent.tools.research_tools import register_research_tools

        sub_registry = _Registry()
        register_research_tools(sub_registry)

        if tools_scope == "readonly":
            # Add safe read-only tools too.
            from src.agent.tools.data_tools import register_data_tools
            from src.agent.tools.health_tools import register_health_tools
            from src.agent.tools.analysis_tools import register_analysis_tools
            try:
                register_data_tools(sub_registry, user_model)
                from src.config import get_settings as _gs
                uid = getattr(user_model, "user_id", None) or _gs().agenticsports_user_id
                register_health_tools(sub_registry, user_id=uid)
                register_analysis_tools(sub_registry)
            except Exception as exc:
                logger.debug("Subagent readonly tool registration partial: %s", exc)

        sub_system_prompt = (
            "You are a focused research subagent. You have a SINGLE task and "
            "a small set of tools. Use tools as needed, then return a concise "
            "final answer.\n\n"
            "Rules:\n"
            "- Be honest about uncertainty. If you cannot verify a fact, say so.\n"
            "- Prefer naming sources (URLs from web_fetch, or 'trained knowledge').\n"
            "- Return ONLY plain text (or JSON if the task explicitly asks for it).\n"
            "- Stop as soon as you have a sufficient answer. Do not over-research.\n"
            "- If the question is ambiguous, return what you found and a "
            "  follow-up question for the user."
        )

        messages: list[dict] = [{
            "role": "user",
            "content": f"TASK: {task}\n\nCONTEXT: {json.dumps(context or {}, ensure_ascii=False, default=str)}",
        }]

        rounds = 0
        tool_trace: list[dict] = []
        bound = max_rounds if max_rounds and max_rounds > 0 else SUBAGENT_MAX_ROUNDS
        bound = min(bound, SUBAGENT_MAX_ROUNDS)

        openai_tools = sub_registry.get_openai_tools()

        while rounds < bound:
            rounds += 1
            response = chat_completion(
                messages=messages,
                system_prompt=sub_system_prompt,
                tools=openai_tools,
                temperature=0.3,
            )
            if not response.choices:
                return {"error": "subagent: empty LLM response", "rounds": rounds}

            msg = response.choices[0].message
            tcalls = msg.tool_calls
            if not tcalls:
                return {
                    "result": (msg.content or "").strip(),
                    "tool_calls": tool_trace,
                    "rounds": rounds,
                }

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tcalls
                ],
            })

            for tc in tcalls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                result = sub_registry.execute(tc.function.name, args)
                tool_trace.append({"tool": tc.function.name, "args": args})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:2000],
                })

        return {
            "error": "subagent: hit max rounds without final answer",
            "tool_calls": tool_trace,
            "rounds": rounds,
        }

    registry.register(Tool(
        name="spawn_subagent",
        description=(
            "Launch a subagent with its own LLM + tool loop for a self-contained "
            "task; returns only the final answer (intermediate tool calls do not "
            "pollute main context). Use for external info you lack: race "
            "details, methodology lookups, gear specs, ambiguity research. "
            "Avoid for built-in knowledge, in-context analysis (use "
            "spawn_specialist), or single web_search/web_fetch calls. "
            "tools_scope: 'research' (web only, default) or 'readonly' (adds "
            "read-only DB tools). Write task as if briefing a stranger: state "
            "what to find, format wanted, any known specifics."
        ),
        handler=spawn_subagent,
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Clear instruction for the subagent. Be specific - "
                        "the subagent has no other context. Examples: "
                        "'Find the date, course, and elevation for the "
                        "Karlsruher Halbmarathon 2026', 'Look up best "
                        "practices for marathon taper in last 2 weeks'."
                    ),
                },
                "tools_scope": {
                    "type": "string",
                    "description": (
                        "Tool subset for the subagent. 'research' = web only "
                        "(default). 'readonly' = web + read-only DB tools."
                    ),
                    "enum": ["research", "readonly"],
                    "nullable": True,
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Cap subagent rounds (default 5, max 8).",
                    "nullable": True,
                },
                "context": {
                    "type": "object",
                    "description": "Optional extra context for the subagent.",
                    "nullable": True,
                },
            },
            "required": ["task"],
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
