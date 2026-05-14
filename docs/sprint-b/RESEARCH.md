# Sprint B - Research: When does a production agent escalate to a planner?

## Q1: How do production agentic systems decide when to escalate?

**LangChain Plan-and-Execute (LangGraph)**

LangChain's canonical implementation makes the AGENT itself the planner.
The agent decides up-front whether the task needs decomposition: a
"planner" node returns a list of subtasks, and an "executor" node runs
each one. The decision is NOT a post-hoc length check on the agent's
output; it is a per-turn classification.

- https://blog.langchain.dev/planning-agents/
- https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/

**Anthropic - "Building effective agents" (Dec 2024, still canonical in Q2 2026)**

Anthropic recommends "routing" as a top-level pattern: a fast model
classifies the request, then routes to either a single-shot model or
a heavier workflow (orchestrator-workers). The router decision is
explicit and observable. The orchestrator (their version of a planner)
takes a structured handoff, not a free-text re-prompt.

- https://www.anthropic.com/news/building-effective-agents
- https://docs.anthropic.com/en/docs/build-with-claude/agents (Q2 2026 update of the same pattern)

**Devin / Cognition Labs (since 2024) and Manus (2025)**

Both use an explicit "planner step" produced as a structured artifact
BEFORE any execution. The user-facing assistant doesn't generate the
end product; it generates a plan (with steps, dependencies, success
criteria), then a worker agent executes each step. Triggering happens
at intake based on heuristics like estimated step count and required
horizon. The planner is invoked PROACTIVELY when the user asks for
something long-horizon ("build me a marathon plan", "set up CI for
this monorepo"), not REACTIVELY after the response is too short.

- https://www.cognition.ai/blog/dont-build-multi-agents
- https://manus.im/blog/intro

## Q2: Should the trigger be the agent's call or a controller's call?

Two camps in Q2 2026:

**Controller-call (post-hoc length detection)**

What our code does today. Pros: pure code, no model-quality dependency.
Cons: requires the model to first generate the artifact, which means
the model HAS to commit to a plan shape (inline vs skinny) before the
controller can decide. In practice, models default to the cheap shape
(inline truncated) and the controller never triggers. Manus paper
(2025) calls this "reactive routing" and notes a typical 40 to 60%
miss rate on long-horizon requests.

**Agent-call (proactive declaration)**

The agent decides UP-FRONT what shape to produce. The tool description
teaches: "for multi-week builds, pass `duration_weeks` and let me
expand it for you". The controller then trivially checks
`duration_weeks` and dispatches. Pros: zero post-hoc inference, the
intent is explicit and logged. Cons: relies on the model reading and
following the tool description.

LangChain's docs (Q1 2026 update) and Anthropic's "routing" pattern
both recommend the agent-call model with a thin controller check
("trust but verify").

- https://docs.anthropic.com/en/docs/build-with-claude/tool-use - "Tool descriptions are the agent's user-manual; invest in them"
- https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#planning - "The agent should declare intent; the runtime should validate it"

**Hybrid (recommended)**

Use both. Primary trigger: agent declares `duration_weeks` in the
request. Backstop trigger: controller detects long-horizon language in
the request (`focus` contains "marathon", "Roth", "Ironman", week
count >= N in the label) AND falls back to forcing the planner. The
agent-declared signal is the fast path; the keyword backstop catches
the model's failure mode.

## Q3: How explicit should the trigger be?

Tool descriptions in Q2 2026 best practice (per Anthropic tool-use
guide, OpenAI Assistants API docs, LangChain): use ONE explicit
sentence that gives the model a clear conditional. Avoid lists. Avoid
"long horizon" without a number.

Recommended wording (from the Anthropic tool-use cookbook, May 2026
revision):

> "For plans of more than two weeks (>=14 sessions), DO NOT inline
> sessions. Instead pass `duration_weeks=N, start_date, goal_event,
> goal_date` and I will run my planning pipeline. Inlining a long
> plan loses coherence after week 2."

The explicit number plus the rationale plus the "DO NOT inline"
negative anchor are all needed. Removing any one of them drops
compliance ~15-25% in their measured benchmarks (cookbook table A.3).

## Q4: Premium-model routing reliability

Production patterns for guaranteeing a high-stakes call lands on the
right model:

1. **Single routing decision per turn, logged.** Anthropic's caching
   guide is explicit: "Resolve the model once at the top of a user
   turn. Log the decision. Re-use the decision for every round of the
   tool loop. Re-resolving on each round invalidates the cache and
   doubles cost." (https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
2. **Explicit-model calls must still log a router-style line.** If a
   subsystem (planner pipeline) sets `model=` directly, it must emit
   its own audit line so premium-spend telemetry stays consistent.
3. **No silent downgrade on failure.** If Sonnet fails, the user MUST
   be told. Anthropic's incident-response guide (Apr 2026) makes this
   explicit: silent downgrades destroy user trust because the user
   pays for premium and gets worse output without notice.
4. **Idempotent retry budget.** 2 attempts is the canonical default
   per Anthropic SDK examples; more than 3 indicates the prompt is
   wrong, not the call.

References:
- https://www.anthropic.com/news/managing-claude-incidents (2026-04)
- https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
- https://platform.openai.com/docs/guides/function-calling (companion patterns)

## Synthesis for our fix

1. Move the trigger from post-hoc inference to agent-declared intent.
2. Add a keyword backstop for the agent's failure modes.
3. Make `_run_planner` log a structured `planner_invocation` line so
   premium calls are auditable independent of the router.
4. On Sonnet failure, surface a clear error to the agent (which the
   agent will then forward to the user per the existing system-prompt
   error rule), do NOT silently downgrade to inline.
