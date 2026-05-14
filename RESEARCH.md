# Feature 4: Plan-and-Execute Pattern - Research

Date: 2026-05-14
Branch: worktree-agent-a5df813ba1b869155
Author: Lead engineer (Athletly)

## 1. LangChain Plan-and-Execute (Q2 2026 canonical)

The Plan-and-Execute pattern in LangChain has two roles:

- **Planner**: an LLM prompt that decomposes a high-level goal into an ordered list of steps. Runs ONCE up front. Output is structured (list of subtasks, dependencies, constraints acknowledged).
- **Executor**: iterates over the plan. For each step, performs the action (tool call, sub-LLM call, function execution). The executor can be a simple function-caller or a ReAct sub-loop scoped to a single step.

Since LangChain v1.0 (Oct 2025) the canonical building block is `create_agent` running on the LangGraph runtime, and in March 2026 LangChain shipped **Deep Agents**, an agent harness on top of LangGraph that bundles plan-and-execute defaults (file-system scratchpad, sub-agent context isolation, durable execution). Deep Agents is a packaging of the pattern, not a new runtime: under the hood it is still a graph of planner + executor nodes.

Key Q2 2026 takeaway: the planner does not have to be an "agent" with tools. It is just a structured-output LLM call that returns a plan dict. The executor is the part that uses tools. This is exactly the shape we want for training plan generation.

Sources:
- https://blog.langchain.com/planning-agents/
- https://www.langchain.com/blog/planning-agents
- https://docs.langchain.com/oss/python/langchain/agents
- https://www.marktechpost.com/2026/03/15/langchain-releases-deep-agents-a-structured-runtime-for-planning-memory-and-context-isolation-in-multi-step-ai-agents/
- https://medium.com/@visakhpadmanabhan7/plan-and-execute-in-langchain-handling-complexity-with-structure-b5972dbce577

## 2. Hierarchical task decomposition in coaching tools

Coaching platforms structure plans in a near-identical 3-level hierarchy:

- **Macro / Phase**: 4 to 16 week blocks named by training intent. TrainerRoad uses Base (12 weeks), Build (8 weeks), Specialty (8 weeks). Final Surge plans (marketplace) follow the same Base/Build/Peak/Taper layout. Running plans (Daniels, Pfitzinger, Hansons) all encode periodization as a small fixed set of phases with target weekly volume and intensity distribution per phase.
- **Meso / Week**: a "weekly template" maps weekday slots (Mon-Sun) to slot types (easy, quality, long, rest, cross). The slot vocabulary is small (5-7 labels). Volume and intensity targets shift by phase but the weekly skeleton stays stable.
- **Micro / Session**: the actual workout: warm-up, main set, cool-down, target pace/HR/power, total duration. Only at this level does sport-specific detail appear (paces, watts, swim intervals).

This 3-level shape maps cleanly onto a plan-and-execute split:
- Planner outputs Macro + Meso (phases + weekly template + constraints acknowledged).
- Executor fills in Micro (one session per (week, weekday) slot).

Sources:
- https://www.trainerroad.com/blog/how-to-follow-a-trainerroad-training-plan/
- https://www.finalsurge.com/trainingplans
- https://support.finalsurge.com/hc/en-us/articles/4408535325207-Creating-and-Editing-Training-Plans
- https://sunriserunco.com/running-plans/

## 3. Anthropic context engineering / sub-agent architecture (Q2 2026)

The Anthropic "Effective Context Engineering" article frames the relevant pattern as **sub-agent architecture**:

> "The main agent coordinates with a high-level plan while subagents perform deep technical work or use tools to find relevant information."

> "Each sub-agent can explore extensively but returns only condensed summaries (typically 1,000-2,000 tokens)."

> "A clear separation of concerns: the detailed search context remains isolated within sub-agents, while the lead agent focuses on synthesizing and analyzing the results."

The article notes sub-agent / multi-agent setups "excel for complex research and analysis where parallel exploration pays dividends" and reports that the Anthropic multi-agent research system "showed a substantial improvement over single-agent systems on complex research tasks."

How this maps to plan generation: training plan generation IS analysis (athlete state, goal, constraints) followed by parallelizable per-section work (one session per slot). The planner is the lead, the executor calls are scoped sub-tasks. Critically, the executor only needs the OUTLINE plus athlete constraints in context, not the full athlete history that the planner saw. That isolation is the point.

Source:
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

## 4. When does Plan-and-Execute beat one-shot?

Synthesizing across sources:

**Plan-and-Execute wins when**:
- Output is multi-step / multi-section and steps are largely INDEPENDENT once the outline is fixed (the per-day session does not need to "see" the other days, only the slot constraints).
- Output exceeds a context-coherence budget for one-shot generation. Empirically: a 12 to 16 week running plan is roughly 60 to 110 sessions, each 80 to 150 tokens of structured detail, plus 1k to 2k tokens of phase reasoning. That puts the one-shot output at 6k to 17k tokens. Models reliably degrade in coherence and start dropping required schema fields above ~4k tokens of structured output in one go.
- Cost of a mistake is high (here: a saved plan is the canonical artefact the athlete tracks against for months).

**One-shot wins when**:
- Output is short (one week, 5 to 7 sessions, ~700 to 1500 tokens) - the planning overhead does not pay off.
- The task is exploratory and you do not yet know the decomposition (browse-and-react beats plan-and-execute here).
- Latency budget is tight (one-shot is one round-trip; plan-and-execute is 2 to N round-trips).

**Threshold metric we adopt for Athletly**: total session count.

- `session_count = duration_weeks * training_days_per_week`
- `session_count >= 14` (roughly 2 weeks at 7 sessions, or 3 weeks at 5 sessions): use Plan-and-Execute.
- `session_count < 14`: inline (current behaviour, one LLM call from the main coach).

This crosses both the "more than a couple of weeks" intuition AND the empirical token-coherence threshold. It is also forced for Pro: free tier is capped at 1 week so it never crosses the threshold.

Sources:
- https://medium.com/@dzianisv/vibe-engineering-langchains-tool-calling-agent-vs-react-agent-and-modern-llm-agent-architectures-bdd480347692
- https://byaiteam.com/blog/2025/12/09/ai-agent-planning-react-vs-plan-and-execute-for-reliability/
- https://medium.com/@shubham.ksingh.cer14/plan-and-execute-ai-agents-architecture-f6c60b5b9598

## 5. Latency trade-off

LLM inference is a 2 to 20 second sink per call. The 10 second mark is the UX cliff where users start losing trust in a static loading state.

Concrete budgets for plan generation:

- One-shot 16-week inline plan today: 1 main LLM call. ~12 to 25 seconds (long structured output).
- Plan-and-Execute, sequential executor: 1 planner call (3 to 6 seconds, small output) + N executor calls (1 to 3 seconds each for a single session). For 16 weeks of 5 days = 80 sessions, sequential is unacceptable.
- Plan-and-Execute, **chunked by week**: 1 planner call + 16 executor calls (one per week, 1 to 3 seconds each). Total wall-clock ~25 to 55 seconds, similar to one-shot but with streaming.
- Plan-and-Execute, **parallel by week**: 1 planner call + max(per-week latency) with concurrent executor calls. Total wall-clock ~6 to 12 seconds. Best UX.

Mitigation when waiting:
- Stream intermediate progress events to the UI (planner_done, week_1_done, week_2_done, ...). This converts "static load" into "visible progress", and per UX research extends user tolerance well past 10 seconds.

We chose week-level chunking, with optional parallelism via `concurrent.futures.ThreadPoolExecutor` since LiteLLM is sync-friendly. For Phase 1 we ship SEQUENTIAL week execution (simpler, single failure mode) with progress events, and leave parallel execution behind a feature flag.

Sources:
- https://www.uxtigers.com/post/think-time-ux
- https://medium.com/@raj-srivastava/understanding-latency-in-multi-agent-genai-systems-1000dd34f6c4
- https://medium.com/google-cloud/the-art-of-fast-agents-14-strategies-to-fix-latency-07a1e1dfebf9
- https://www.digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput

## 6. Structured intermediate format

YES: the planner produces a strict JSON outline that the executor reads slot by slot.

Best practice across 2026 LangChain / Deep Agents posts: the planner uses structured output (JSON schema enforcement via the provider's response_format / structured output mode, or Pydantic schema in LangChain). The executor receives the OUTLINE plus a single slot descriptor ((week_index, weekday, phase_label, slot_type, target_minutes, intensity_distribution)) and emits ONE session dict matching `save_plan`'s `sessions[]` schema.

Two consequences:
1. The executor system prompt is much shorter (no need to teach periodization, only "fill in this one slot").
2. The executor can be a cheaper model than the planner (Haiku, not Sonnet) without losing plan-level coherence, because coherence lives in the OUTLINE, not in the per-session prose.

This is exactly the cost-asymmetry the Product Owner directive bakes in: Sonnet for the planner, Haiku for the executor.

Sources:
- https://blog.langchain.com/planning-agents/
- https://apxml.com/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents/plan-and-execute-agents
- https://www.marktechpost.com/2026/03/15/langchain-releases-deep-agents-a-structured-runtime-for-planning-memory-and-context-isolation-in-multi-step-ai-agents/

## Summary table

| Question | Answer |
|---|---|
| Pattern | Planner (1 LLM call, Sonnet, structured outline) -> Executor (1 LLM call per week, Haiku, fills sessions) -> Validator -> save_plan |
| Trigger | `duration_weeks * training_days_per_week >= 14` AND user is Pro |
| Free tier | Inline, Haiku, cap to 1 week |
| Planner output | JSON: phases[], weekly_template, constraints_acknowledged, total_weeks |
| Executor input | outline + (week_index, weekday) slot descriptor |
| Latency target | <15s with streaming for 16-week plan (sequential), <8s parallel |
| Cost target | ~$0.02 to $0.05 per Pro plan generation (1 Sonnet + 16 Haiku calls) |
