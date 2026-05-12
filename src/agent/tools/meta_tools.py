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
            "Single-shot LLM call to a specialist persona with no tool access. "
            "Returns a structured JSON opinion on a narrow analytic question. "
            "Cheaper and faster than spawn_subagent because there is no tool loop - "
            "use this when you want a focused expert perspective on data you "
            "already have in hand.\n\n"
            "AVAILABLE SPECIALISTS:\n"
            "- data_analyst: analyzes a block of training data for load trends, "
            "recovery status, performance changes, gaps. Best for 'what does this "
            "data tell us?'.\n"
            "- domain_expert: sports-science methodology guidance for a sport + "
            "goal combination: energy systems, periodization phase fit, session "
            "type rationale, safety considerations. Best for 'is this approach "
            "scientifically sound?'.\n"
            "- safety_reviewer: sports-medicine safety check on profile + "
            "training: overtraining risk, injury risk, youth considerations, "
            "medical referral need. Best for vulnerability checks before "
            "committing to a hard plan, especially for youth athletes, returning-"
            "from-injury, or unusually high volumes.\n\n"
            "WHEN TO USE:\n"
            "- You have data and want a focused interpretation lens.\n"
            "- Pre-commit safety sweep on a high-load macrocycle or build phase.\n"
            "- You want a second opinion grounded in a specific discipline "
            "without burning tokens on tool use.\n\n"
            "WHEN NOT TO USE:\n"
            "- You need external information you do not have: use spawn_subagent "
            "(it can web_search). spawn_specialist has NO tools.\n"
            "- The question is too broad for a single-shot answer: break it up "
            "or use spawn_subagent.\n"
            "- For routine coaching reasoning the main agent can handle directly "
            "with its own context.\n\n"
            "CRITICAL: pass everything the specialist needs in the context dict. "
            "The specialist sees ONLY the task string and the context dict - no "
            "athlete profile, no beliefs, no activity history unless you put "
            "them in context.\n\n"
            "EXAMPLES:\n"
            "- spawn_specialist(type='data_analyst', task='Identify load and "
            "recovery patterns in the last 14 days', context={'activities': [..],\n"
            " 'health_summary': {..}}).\n"
            "- spawn_specialist(type='safety_reviewer', task='Assess this "
            "16-week marathon build for a 17-year-old', context={'profile': {..},\n"
            " 'macrocycle': {..}}).\n"
            "- spawn_specialist(type='domain_expert', task='Is a polarized 80/20 "
            "split appropriate for a sub-3 marathon goal at current fitness?', "
            "context={'profile': {..}, 'recent_fitness': {..}}).\n\n"
            "Returns {'specialist': <type>, 'result': <JSON dict>}. If the model "
            "fails to return clean JSON, 'result' falls back to "
            "{'raw_response': '...'} - parse defensively."
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
            "Launch a focused subagent that runs its OWN LLM + tool loop to "
            "complete a self-contained task, then returns ONLY the final "
            "synthesized answer. Equivalent to Claude Code's Task tool with the "
            "general-purpose subagent type. The subagent's intermediate tool "
            "calls do not enter the main agent's context - this is the whole "
            "point: keep your context clean of noisy intermediate steps.\n\n"
            "USE PROACTIVELY when you need external information you do not "
            "already have:\n"
            "- A specific race or event the athlete named (e.g. 'Halbmarathon "
            "Karlsruhe', 'Berlin Marathon 2027', 'Ironman Hamburg'): research "
            "exact date, course profile, elevation, distance variants, "
            "registration window, qualifying standards.\n"
            "- Current training methodology, sports-science papers, recovery "
            "protocols, nutrition guidelines you are uncertain about.\n"
            "- Product / gear specs before recommending equipment.\n"
            "- Disambiguation when the user mentions something ambiguous: do "
            "the research, then ASK the user to confirm before acting on it.\n\n"
            "WHEN NOT TO USE:\n"
            "- For questions you can answer from built-in knowledge with "
            "reasonable confidence: just answer directly.\n"
            "- For analysis on data you already have in context: use "
            "spawn_specialist (cheaper, no tool loop) or reason directly.\n"
            "- For trivial one-URL lookups where you already know the URL: use "
            "web_fetch directly.\n"
            "- For a single quick web search where you will need the raw "
            "results: use web_search directly - subagent overhead is not "
            "justified.\n\n"
            "TOOLS SCOPE:\n"
            "- 'research' (default): web_search + web_fetch only. Best for "
            "external knowledge lookups.\n"
            "- 'readonly': adds read-only DB tools (get_activities, "
            "get_daily_metrics, analysis_tools). Use when the question requires "
            "DEEP analysis of the athlete's own data AND the analysis would "
            "otherwise pollute the main context with raw data dumps.\n\n"
            "WRITING THE TASK PROMPT (CRITICAL):\n"
            "The subagent starts with ZERO context about the athlete or the "
            "conversation. Brief it like a smart researcher who just walked in:\n"
            "- State what to find out, why it matters, and the format you want "
            "back ('Report a 3-line summary: date, course, elevation').\n"
            "- Include any specifics you already know (year, athlete's "
            "constraints) that narrow the search.\n"
            "- Be explicit about scope: what is in, what is out.\n"
            "- Do NOT write 'based on your findings, decide X' - that delegates "
            "synthesis. Ask for facts, then synthesize yourself.\n\n"
            "EXAMPLES:\n"
            "- spawn_subagent(task='Find the date, course profile, elevation "
            "gain, and registration link for the Karlsruher Halbmarathon 2026. "
            "Return a 4-bullet summary plus the official URL.')\n"
            "- spawn_subagent(task='Look up evidence-based best practices for "
            "marathon taper in the final 2 weeks: volume reduction percent, "
            "intensity retention, key sessions. Cite sources.')\n"
            "- spawn_subagent(task='Analyze this athlete\\'s last 60 days of "
            "running activities for signs of accumulated fatigue or "
            "overreaching. Report top 3 patterns.', tools_scope='readonly')\n\n"
            "AFTER THE SUBAGENT RETURNS: decide whether to confirm with the "
            "athlete ('Meinst du den Halbmarathon Karlsruhe am 1. Mai 2026?') "
            "or fold the facts directly into your response. Cite sources when "
            "the subagent provided URLs.\n\n"
            "PARAMETERS: max_rounds caps cost (default 5, hard ceiling 8). "
            "context dict is passed verbatim to the subagent as JSON - use it "
            "for extra data the subagent needs but cannot research itself.\n\n"
            "Returns {'result': '<final synthesized text>', 'tool_calls': "
            "[{tool, args}, ...], 'rounds': int}. On failure to converge: "
            "{'error': '...', 'tool_calls': [...], 'rounds': int}."
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
