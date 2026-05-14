# LIVE_DIAGNOSIS: Iter 2 Sprint H

## Setup

Verified on hetzner production container `athletly-api-1`:

- container is running, served from /app on commit `1597119`
- code-path under inspection: `src/agent/agent_loop.py`, `src/agent/llm.py`,
  `src/agent/complexity_detector.py`, `src/agent/planner.py`,
  `src/agent/prompts/{planner_system,executor_system}.md`,
  `src/agent/system_prompt.py`

## 1. Lisa's stored plan (latest, focus contains "Roth")

Queried from supabase `plans` table, ordered desc, first row:

```
FOCUS:           "8-week Ironman taper to 11:30 - ITB-protected running strategy"
TOTAL_SESSIONS:  48
BY_SPORT:        running=36, cycling=9, swimming=3
DURATION_WEEKS:  8
GOAL_EVENT:      "Challenge Roth"
WEEKLY_TEMPLATE: monday=easy, tuesday=quality, wednesday=easy,
                 thursday=quality, friday=rest, saturday=long_or_quality,
                 sunday=long
```

Observations:

- 48 sessions is the MATHEMATICAL output of 8 weeks x 6 non-rest days. The
  user expected 96 (8 weeks x 12 sessions = 1.7 sessions/day). The
  weekday template is 6 training days; 6 x 8 = 48 is internally
  consistent. So "48 vs 96 expected" is partially a planner UX/communi-
  cation issue: the athlete asked for triathlon training (sport mix)
  and the planner produced 1 session/day even though triathletes
  typically double up (swim early + run/bike later).
- Sport distribution is heavily skewed: 75% running, 19% cycling, 6%
  swimming. For a Roth Langdistanz this is wildly wrong. Swimming for
  3.8 km in race needs roughly 2-3 swims per week (24 total over 8 wk),
  not 3. Cycling for 180 km in race needs at least 2-3 bike sessions
  per week. **Lisa's plan would not get her to the Roth finish line.**
- Focus label says "Ironman taper" - phase labels in outline say "Build,
  Recovery, Peak, Race-Prep, Taper" which is correct macro structure
  but the COMPOSED focus string conflates Build/Peak weeks with "taper".
- No mention of distance in focus. The user reports the assistant said
  "Ironman 70.3" - which is the half (~113 km), but Roth is Langdistanz
  (full, ~226 km). This is the distance-terminology confusion.

## 2. Complexity detector firing (Bug B)

Ran `needs_complex_reasoning` against Lisa's typical turn-1 message:

```
"8-Wochen Build bis Roth Langdistanz, ich brauche einen Plan."
```

Detector returns `is_complex=True` with matched keywords
`["langdistanz", "8-week plan"]`. This is correct - Group A
(`langdistanz`) ALONE escalates by the documented rule.

## 3. Production logs - what actually happens

Searched container logs for `complex chat_completion failed` and found
the smoking gun. Every single complex turn raises:

```
litellm.BadRequestError: AnthropicException - {"type":"error",
"error":{"type":"invalid_request_error",
"message":"`temperature` may only be set to 1 when thinking is enabled."}}
```

This was repeated for 7+ separate request_ids in the past 48h:
`req_011Cb2R4yAmL1zFH6atuUZ7b`, `req_011Cb2R8ptJVfiThooTtKGwn`,
`req_011Cb2RJs3ogiji2zunWRVGX`, `req_011Cb2RTz3jxyfYkqa1iMzKq`,
`req_011Cb2Rj8TRdpabYHvi8MgAa`, `req_011Cb2Rksc21VZCKoTTyauyo`,
`req_011Cb2RnWN8csZWjB6xJcNge`.

Every single one logs `complex chat_completion failed (...), retrying
as routine`. So:

- complexity_detector fires correctly: `selected_tier = "complex"`
- model_router resolves correctly: Sonnet + thinking_budget=2048
- chat_completion sends `thinking={type: enabled, budget_tokens: 2048}`
  AND `temperature=AGENT_TEMPERATURE` (=0.7)
- Anthropic Extended Thinking REQUIRES `temperature=1` when thinking is
  enabled. The 0.7 makes Anthropic reject the request.
- The `try/except` in agent_loop catches the failure and retries with
  `tier="routine"`, which lands on Haiku.

**This is why every one of Lisa's 6 turns burned Haiku even though
complexity_detector did escalate.** Tier propagation IS plumbed
correctly to chat_completion - the bug is at the LLM-API layer:
incompatible parameter combination.

## 4. Planner prompt - does it know about sport mix?

`src/agent/prompts/planner_system.md` line 31-39:

```
"weekly_template": {
  "monday": "<easy|quality|long|long_or_quality|cross|rest>",
  ...
}
```

The slot vocabulary defines INTENSITY/DURATION TYPE only. There is no
sport field. The executor picks sport in `_sanitize_executor_sessions`:

```python
# planner.py line 746-748
sport = (entry.get("sport") or available_sports[0] or "running").lower()
if available_sports and sport not in [s.lower() for s in available_sports]:
    sport = available_sports[0].lower()
```

`available_sports[0]` for Lisa is "running" (alphabetical order from
`["running", "cycling", "swimming"]` after profile load). When the
executor's free-text guess fails (Haiku doesn't always emit a `sport`
field), the fallback is always "running". This is exactly why 36 of 48
sessions are running.

The planner prompt is single-sport-blind. The executor prompt says
`sport` must be in `available_sports` but does not require a
distribution. There is no validation that swim/bike actually appear in
the plan.

## 5. complexity_detector unit test - Lisa's exact turn-1

Need to add an explicit unit test for Lisa's actual production
message shape, namely the one with `Langdistanz Roth 8 Wochen`.
Current tests cover `triathlon`, `Ironman`, `70.3`, `Roth focus`
but NOT the precise phrase the user typed in production.

## Summary of two distinct bugs

| Bug | Location | Effect | Fix |
|-----|----------|--------|-----|
| A. Sport mix | planner_system.md + executor + sanitize | 75% of sessions are running | Make weekly_template sport-aware + post-assembly validation + regenerate |
| B. Tier propagation | llm.py / agent_loop.py | Sonnet+thinking call rejected, fallback to Haiku | Force temperature=1.0 when thinking_budget>0 |

Plus a distance-terminology gap that needs explicit definitions in the
system prompt so the assistant stops mixing Langdistanz with 70.3.
