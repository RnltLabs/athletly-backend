# Feature 3: Reflexion Loop - Research

Date: 2026-05-14
Author: Lead engineer (Feature 3)

## 1. Reflexion paper (Shinn et al., 2023)

Source: "Reflexion: Language Agents with Verbal Reinforcement Learning",
Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao. NeurIPS 2023.
- arXiv: https://arxiv.org/abs/2303.11366
- Project page: https://noahshinn.com/reflexion

Core mechanism (three agents in a loop):
1. Actor (LLM) generates an action or trajectory.
2. Evaluator (scalar reward or LLM-as-judge) scores the trajectory.
3. Self-Reflection module reads the trajectory + score and produces a
   verbal lesson in natural language. The lesson is appended to a
   long-term episodic memory.
4. On the next attempt the Actor sees the lessons in its context and
   acts differently.

What gets reflected on:
- The whole trajectory (state, action, observation, reward) of the
  most recent attempt.
- The lesson is "verbal" reinforcement: text the next run can read,
  not a gradient.
- Reflexion deliberately keeps memory bounded (e.g. last 1 to 3
  lessons) to avoid prompt bloat; older lessons get summarised or
  dropped.

Adaptation for a long-running coach (us):
- We do not have a binary reward signal. Our "evaluator" is the
  conversation itself: user corrections, contradiction signals,
  re-asked questions, tool failures.
- The lesson store grows over weeks not over retries. So we cannot
  keep "all lessons in prompt"; we need retrieval (next section).

## 2. Q2 2026 production examples of agent self-improvement loops

### Generative Agents (Park et al., 2023, "Generative Agents:
Interactive Simulacra of Human Behavior", https://arxiv.org/abs/2304.03442)

- Each agent maintains a memory stream of timestamped natural-language
  observations.
- Daily reflection cycle: when accumulated "importance score" of new
  observations crosses a threshold (~150), the agent prompts itself
  with "given the recent observations, what are the most salient
  high-level questions we can answer?" and then "what do these
  observations say about <athlete>?". The answers are stored back as
  reflections (synthetic memories) with their own importance score.
- Retrieval uses a weighted sum of recency, importance, and
  semantic relevance.
- Pattern we adopt: reflection produces durable, higher-level
  "insights" that themselves enter the memory store and can be
  retrieved alongside raw observations.

### Mem0 (Q2 2026 architecture, https://mem0.ai/blog/state-of-ai-agent-memory-2026)

- Read for the architecture only; we are NOT using their paid SaaS
  (user directive).
- Stages they describe:
  1. EXTRACT: LLM scans conversation for "memorable facts".
  2. UPDATE: dedupe / supersede existing memories.
  3. STORE: text plus vector embedding.
  4. RETRIEVE: top-K vector similarity, optionally re-ranked.
- They distinguish "Factual", "Episodic", and "Semantic" memory.
- Lesson for us: separate WHAT happened (already covered by our
  `episodes` table and `session_messages`) from the abstracted
  LESSON ("athlete dislikes Sunday long runs"). Reflexion targets
  the second tier.
- Their hallucination guard: the extract step asks the LLM to quote
  the conversation span that justifies the memory. We borrow this
  idea (see Phase 2, evidence field).

### Anthropic "Context engineering for AI agents" (Sep 2025,
https://www.anthropic.com/news/context-engineering-for-ai-agents)

Key takeaways relevant to a reflection loop:
- Treat context as a budget, not a backpack. Inject only what the
  current turn needs.
- Prefer many small, retrieval-keyed notes over one big "everything
  we know" blob.
- Use a recency + relevance retrieval policy; do not let stale notes
  drown current ones.
- For long-running agents, schedule periodic "self-edit" passes that
  compact or supersede older notes. (We map this to our 90-day decay
  rule.)
- Anthropic also recommends running the reflection on a CHEAPER
  model (Haiku) since it is doing structured extraction, not
  reasoning under uncertainty. Matches our product directive.

### Letta / MemGPT (https://github.com/letta-ai/letta)

- Open-source long-term memory layer. Same Extract -> Store ->
  Retrieve loop, but with explicit "archival" vs "core" tiers.
- Trigger for archival memory write is also LLM-driven, similar to
  what we propose.

## 3. Signals that best indicate "lesson learned"

Ranked by usefulness for our coach domain:

1. Explicit user correction. Patterns like "No, that's wrong",
   "I told you yesterday I cannot run on Mondays", "Actually it was
   12k not 10k". Highest-signal: the user is literally teaching us.
2. User repeats themselves across sessions. Same complaint or
   preference surfacing twice = stable signal.
3. Tool call failures or timeouts followed by user frustration.
   E.g. agent suggests a plan that violates a constraint we did
   not know about.
4. Long tool chains that yielded no useful answer. Indicates a gap
   in the agent's mental model of the athlete.
5. Sudden plan changes the user accepts (e.g. they confirm a
   recovery week we proposed). The acceptance itself is a positive
   signal worth remembering.
6. Strongly positive sentiment ("yes, that worked great"). We can
   add a confirmation token to lessons.

We do not blindly use sentiment, because endurance coaching often
has the user push back even when the coach is right.

## 4. Storage format: raw markdown vs structured taxonomy

Trade-offs:

- Raw markdown notes: cheap, lossless, mirrors how the LLM thinks.
  Bad for retrieval: vector recall over free-form text plus
  duplicates accumulate fast.
- Fully typed taxonomy: hard to design up front; brittle when the
  domain shifts; forces the LLM to pick categories that may not fit.
- Hybrid (what we pick): a tiny structured envelope (topic,
  observation, lesson, applicable_to, evidence, confidence,
  source_session_id) plus a free-text `lesson` body for the
  natural-language insight. Embedding is computed over the `lesson`
  body plus `topic` plus `applicable_to`.

The structured envelope gives us reliable retrieval filters
(applicable_to = "running"), the free-text body keeps the LLM
expressive, and the evidence pointer lets us audit any lesson back
to the source turn.

## 5. Retrieval: how does next session pick relevant lessons

Three-stage policy at session start:

1. Compute the user query embedding (or the runtime-context
   embedding) using the existing Gemini `text-embedding-004`
   model (already wired up via `UserModelDB._generate_embedding`).
2. pgvector cosine search against the user's lessons, weighted by
   `(confidence * recency_decay)`.
3. Take top-K (K = 5 default, 3 on cold start) and inject them into
   `build_runtime_context` as a short "Lessons learned about this
   athlete" block, capped at ~600 tokens.

This mirrors `match_beliefs()`, so the implementation is a small
RPC clone with a different table.

Cold-start fallback when there are fewer than 10 lessons: return all
of them (matches the same fallback pattern in
`UserModelDB.find_similar_beliefs`).

## 6. Privacy and hallucination guards

Threats:
- Reflection LLM invents facts ("Athlete said they injured their
  knee" when the user never said that).
- Reflection records sensitive medical statements that the user did
  not intend to enshrine.

Mitigations:

- Prompt-level: the reflection prompt explicitly says "Do NOT invent
  facts the user did not state. If unsure, return an empty list.
  Quote the user's own words as evidence."
- Schema-level: every lesson stores an `evidence` field containing
  a direct quote (or quotes) from the session. We post-validate
  that the quote substring is actually present in the source
  messages; if not, we drop the lesson.
- Domain-level: the prompt forbids storing PII categories
  (full name, address, phone, financial info) and medical
  diagnoses. Symptoms ("knee soreness on long runs") are allowed
  because they directly affect coaching.
- Rate-level: free tier is capped to one reflection run per user
  per day (and per month batch), so a single bad run cannot flood
  the store.
- Reversal: lessons are soft-deleted (`active = false`) instead of
  hard-deleted, so we can audit and roll back.

## References

- Shinn et al. 2023, Reflexion: https://arxiv.org/abs/2303.11366
- Park et al. 2023, Generative Agents: https://arxiv.org/abs/2304.03442
- Mem0 state-of-ai-agent-memory 2026: https://mem0.ai/blog/state-of-ai-agent-memory-2026
- Anthropic context engineering 2025: https://www.anthropic.com/news/context-engineering-for-ai-agents
- Anthropic prompt caching 2026: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Letta / MemGPT: https://github.com/letta-ai/letta
- pgvector: https://github.com/pgvector/pgvector
