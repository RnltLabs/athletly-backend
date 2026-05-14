# RESEARCH: Premium Routing + Sport-Math Compute Tool

Sprint E. Trigger: Lisa (Triathletin) test. Coach hallucinated NP=352W from
FTP=240W. Algebraically impossible (normalized power on a sustainable
interval cannot exceed FTP by ~46%). Symptom: Haiku 4.5 reasoning collapsed
on FTP/NP/CSS math under multi-step load. Cross-sport context (Roth race
strategy, watts/kg vs body weight, pacing-vs-nutrition split) came out
generic, not personalized.

## 1. When do production AI agents in Q2 2026 escalate to deeper-thinking models?

The dominant production pattern is "tiered routing": cheap default model
for routine turns, deeper model triggered on demand. The trigger is almost
always one of three: explicit user signal, heuristic classification of the
turn, or a controller that escalates after a quality check fails.

- Anthropic, model selection guidance (Q2 2026):
  `https://docs.claude.com/en/docs/about-claude/models/overview` recommends
  Haiku for high-volume tool turns and Sonnet for "complex analysis, math,
  multi-step planning". Extended thinking is positioned as the escalation
  step when the task needs step-by-step verification of intermediate work.
- Anthropic Extended Thinking docs:
  `https://docs.claude.com/en/docs/build-with-claude/extended-thinking`.
  Extended thinking emits a separate reasoning block before the answer.
  Crucially for our bug: arithmetic errors get caught in the thinking
  block before the model commits to a wrong final number.
- LangChain "router chains" and LangGraph "conditional edges":
  `https://python.langchain.com/docs/how_to/routing/` and
  `https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges`.
  The canonical pattern is a deterministic classifier (regex, keyword
  match, small LLM) that picks the next node. Production teams report
  cleaner observability and lower cost with deterministic rules first,
  LLM classifier only as a fallback.
- OpenAI Routing guide (gpt-4.1 era, still current):
  `https://platform.openai.com/docs/guides/prompt-engineering/routing`.
  Same shape: classifier -> route to specialist model. They explicitly
  warn against using the expensive model as the classifier (latency +
  cost negate the savings).
- Cursor and Cognition (Devin) public posts (2025/2026): both describe
  two-tier setups. Cursor uses cheap completion for inline edits, a
  deeper model when the prompt requests "fix this complex bug" or when
  the diff touches > N files. Devin (Cognition blog,
  `https://cognition.ai/blog/`) uses planning at the start of each task
  on a larger model, executes individual steps on a cheaper one.

Heuristic conclusion: deterministic keyword/pattern trigger is the dominant
pattern in 2026 for routing decisions. LLM-classifier routing exists but
is reserved for ambiguous cases where keywords fail. False positives on
the keyword trigger are tolerable as long as the rule is "narrow and
specific". False negatives (missing a hard query) are the failure mode
we are actually trying to fix.

## 2. Tool-augmented math: how do agents like Claude-Code-with-Python or Devin handle calculations?

The pattern: never trust LLM arithmetic. Route every non-trivial
calculation through a deterministic tool whose result the LLM consumes
verbatim.

- Anthropic tool use docs:
  `https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview`.
  The recommended pattern for "math the model gets wrong" is a
  function-call tool that wraps the calculation. The model passes
  arguments, the tool returns the answer, the model quotes it.
- Anthropic code execution (server-side Python sandbox):
  `https://docs.claude.com/en/docs/agents-and-tools/tool-use/code-execution-tool`.
  This is the heavyweight version: model writes Python, server runs it,
  returns the result. Overkill for fixed formulas; appropriate when the
  agent needs to compose novel calculations on the fly.
- Claude Code itself (from `https://docs.claude.com/en/docs/claude-code/`)
  exposes `Bash` to run `python -c "..."` for ad-hoc computation. The
  in-product preset for "calculations" routes to Python rather than
  the model.
- Cognition Devin: documented in their public engineering posts as using
  a `python_execute` tool for any arithmetic step, with the agent
  forbidden by system prompt from computing inline.
- LangChain Tools docs: same pattern in their math-tools recipe
  (`https://python.langchain.com/docs/integrations/tools/llm_math/`).
  They explicitly note "LLMs are notoriously bad at math; this tool
  exists because GPT-4 cannot reliably multiply 4-digit numbers".

For our use case (a fixed set of sport-science formulas), the right
shape is a structured tool with named formulas (`vdot_from_race_time`,
`ftp_zones`, etc.), not a free-form Python sandbox. Pure functions,
unit tested, deterministic. The agent picks the formula name and
passes the inputs; the tool returns the numbers plus a one-line
interpretation. System prompt enforces: "for any FTP/NP/VDOT/CSS/HR
calculation, call `compute_sport_math` instead of computing inline."

## 3. Sport-science formulas: canonical references

### VDOT (Jack Daniels)

Source: Jack Daniels, "Daniels' Running Formula", 3rd edition (2014),
Human Kinetics. ISBN 978-1450431835. Tables widely reproduced online,
e.g. `https://vdoto2.com/calculator/` and
`https://runsmartproject.com/calculator/`.

VDOT is a unitless number representing the runner's effective VO2max
based on race performance. The Daniels formula maps race velocity (in
m/min) to VO2 demand, then divides by percent-of-VO2max sustained for
the race duration.

Velocity to VO2 (Daniels 2014, p. 80):

  VO2 (ml/kg/min) = -4.60 + 0.182258 * v + 0.000104 * v^2

  where v = race distance (m) / race time (min)

Percent VO2max sustained for the race time:

  %VO2max = 0.8 + 0.1894393 * exp(-0.012778 * t) + 0.2989558 * exp(-0.1932605 * t)

  where t = race time in minutes

VDOT = VO2 / %VO2max.

Reference VDOT values (Daniels table, p. 84+):
- 5k 20:00 -> VDOT ~49.2
- 10k 40:00 -> VDOT ~52.0
- 5k 18:00 -> VDOT ~54.8
- Marathon 3:00:00 -> VDOT ~54.4

### Daniels training paces from VDOT

Easy, Marathon, Threshold, Interval, Repetition paces. From Daniels'
Running Formula table 5.2 (and digitized at
`https://vdoto2.com/calculator/`):

For VDOT 50:
- Easy: 5:11 to 5:42 /km
- Marathon: 4:34 /km
- Threshold: 4:18 /km
- Interval: 3:59 /km
- Repetition: 3:42 /km

The pace function is the inverse of the VO2 formula at the target
intensity percentage:
- Easy: 65 to 79% VO2max
- Marathon: 80%
- Threshold: 88%
- Interval: 97.5 to 100%
- Repetition: 105 to 110%

### FTP zones (Coggan)

Source: Andrew Coggan, "Training and Racing with a Power Meter", 3rd
edition (2019), VeloPress. ISBN 978-1937715922. Reproduced at
`https://www.trainingpeaks.com/learn/articles/power-training-levels/`
and `https://www.fascat.com/power-zones/`.

7 zones as percent of FTP:
- Z1 Active Recovery: < 56% FTP
- Z2 Endurance: 56% to 75% FTP
- Z3 Tempo: 76% to 90% FTP
- Z4 Lactate Threshold: 91% to 105% FTP
- Z5 VO2max: 106% to 120% FTP
- Z6 Anaerobic Capacity: 121% to 150% FTP
- Z7 Neuromuscular Power: > 150% FTP

### Normalized Power (NP, Coggan)

Source: Coggan, "Normalized Power: definition", available at
`https://www.trainingpeaks.com/blog/normalized-power-intensity-factor-training-stress/`.

Algorithm:
1. Compute 30-second rolling average power across the workout.
2. Raise each rolling-average value to the 4th power.
3. Take the mean of those 4th-power values.
4. Take the 4th root.

NP = ( mean( (rolling_30s_avg_power)^4 ) )^(1/4)

Key property for our Lisa bug: NP on a sustained sub-threshold workout
CANNOT meaningfully exceed FTP. For a 60-minute steady ride, NP is
bounded above by FTP plus a few watts (variability cost). NP > 1.10 *
FTP implies either supra-threshold intervals or genuinely unsustainable
work; NP = 352 from FTP 240 (ratio 1.47) is mathematically inconsistent
with a sustainable interval.

### Critical Swim Speed (CSS)

Source: Swim Smooth method, "Critical Swim Speed test":
`https://www.swimsmooth.com/improve/intermediate/critical-swim-speed`.
Also in Mike Ricci's USAT coaching curriculum.

CSS_pace_per_100m = (T400 - T200) / 2

where T400 and T200 are 400m and 200m time-trial seconds.

Pool pace sets relative to CSS:
- Easy pace: CSS + 12 sec/100m
- Threshold pace: CSS
- Race pace (Olympic distance): CSS - 2 to 5 sec/100m
- 200/400m repeats: CSS - 5 to 10 sec/100m

### Karvonen HR zones

Source: Karvonen, M. J., et al. (1957). "The effects of training on
heart rate; a longitudinal study." Annales medicinae experimentalis et
biologiae Fenniae, 35(3), 307 to 315. Adapted widely; ACSM Guidelines
for Exercise Testing and Prescription, 11th ed. (2021).
`https://www.acsm.org/`.

HR_target = ((HR_max - HR_rest) * intensity_pct) + HR_rest

5 zones:
- Z1 Recovery: 50% to 60% HRR
- Z2 Aerobic Base: 60% to 70% HRR
- Z3 Tempo: 70% to 80% HRR
- Z4 Lactate Threshold: 80% to 90% HRR
- Z5 VO2max: 90% to 100% HRR

### Pace from goal time

Trivial:
  pace_per_km_seconds = goal_time_seconds / distance_km

But the agent gets it wrong under load (off-by-one minute, swap of
m/s and km/h). Worth a tool entry for safety.

## 4. Detection heuristics: pattern matching vs LLM classifier

Production teams converge on the same answer in 2025/2026: deterministic
keyword matching first, LLM classifier only as a fallback for ambiguity.

Reasoning:
- **Latency**: regex matching is microseconds; an LLM classifier costs a
  full round-trip (200-800ms with Haiku) before the actual answer call
  begins.
- **Cost**: LLM classifier on every turn adds 30-50% to per-turn token
  count (system prompt + classifier overhead). The whole point of
  tier-routing is to save money on routine turns.
- **Observability**: regex matches are loggable as keyword lists; LLM
  classifier verdicts are opaque ("model said complex"). When debugging
  why a turn was misrouted, the regex log shows you exactly which token
  fired.
- **Determinism**: tests can pin the routing decision per input string.
  An LLM classifier shifts with model version.

Failure modes of keyword routing:
- **False positives**: "I had a triathlon last year" mentions triathlon
  but is not a hard query. Mitigation: pair keyword with intent signal
  (race-strategy verbs like "pacing", "nutrition", "strategy", explicit
  formula references).
- **False negatives**: athlete uses synonyms or describes the situation
  without trigger words. Mitigation: pair sport-keywords with math-only
  fallback ("VDOT", "FTP", "NP", number-with-unit-watts patterns).

For our Sprint E target, keyword routing is the right move. Specific
triggers identified:

A. Multi-sport context keywords (triathlon, ironman, hyrox, duathlon)
   PLUS race-strategy verb (pacing, strategie, strategy, ernaehrung,
   nutrition, fueling, taper, peak)

B. Power/pace math symbols (FTP, NP, VDOT, CSS, Karvonen, watts/kg,
   threshold pace, lactate threshold)

C. Long-horizon plan signal (8 plus weeks, marathon plan, ironman plan,
   periodization, build phase, peak phase)

D. Explicit number-comparison ("ist X Watt realistisch", "wie schnell
   muss ich laufen fuer Sub-3", "kann ich 240 NP halten")

A or B or (C and D) -> tier="complex". This catches the Lisa case
(triathlon + race strategy + FTP math) without firing on simple turns
like "wie fuehle ich mich heute".

## Citations summary

- Daniels J. Daniels' Running Formula, 3rd ed. Human Kinetics, 2014.
- Coggan AR, Allen H. Training and Racing with a Power Meter, 3rd ed.
  VeloPress, 2019.
- Karvonen MJ et al. Ann Med Exp Biol Fenn 1957; 35(3):307-315.
- ACSM Guidelines for Exercise Testing and Prescription, 11th ed., 2021.
- Anthropic docs: claude.com/en/docs/about-claude/models/overview
- Anthropic Extended Thinking: claude.com/en/docs/build-with-claude/extended-thinking
- Anthropic tool use: claude.com/en/docs/agents-and-tools/tool-use/overview
- LangChain routing: python.langchain.com/docs/how_to/routing/
- LangGraph conditional edges: langchain-ai.github.io/langgraph/concepts/low_level/
- OpenAI routing: platform.openai.com/docs/guides/prompt-engineering/routing
- TrainingPeaks Coggan zones: trainingpeaks.com/learn/articles/power-training-levels/
- TrainingPeaks NP/IF/TSS: trainingpeaks.com/blog/normalized-power-intensity-factor-training-stress/
- Swim Smooth CSS: swimsmooth.com/improve/intermediate/critical-swim-speed
- VDOT calculator: vdoto2.com/calculator/
- RunSmart calculator: runsmartproject.com/calculator/
