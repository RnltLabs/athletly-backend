# RESEARCH: Multi-Sport Plan Structure for Long Course Triathlon

## Question

How do production triathlon coaching apps structure the WEEKLY
distribution of swim / bike / run sessions for a long-course
(Langdistanz / Ironman) build phase, and how do they expose that
structure in their planner data model?

## Sources

1. TrainingPeaks "Ironman Training Plans" library
   - https://www.trainingpeaks.com/training-plans/ironman/
   - Plan structure summary (downloaded sample of Joe Friel's
     "Triathlete's Training Bible" 24-week Ironman plan; cross-checked
     against 12-week and 8-week peak builds).
2. Joe Friel, "The Triathlete's Training Bible", 5th ed., section on
   Periodization for Long Course (chapter 7) and Weekly Routines for
   the multi-discipline athlete (chapter 8).
3. Final Surge plan templates
   - https://www.finalsurge.com/training-plans/
   - Specifically the Long Course Build templates from coaches Hillary
     Biscay and Mike Plumb.
4. 80/20 Endurance: "Triathlon Training Plans"
   - https://www.8020endurance.com/8020-triathlon-plans/
   - Matt Fitzgerald's Ironman Level 1 / Level 2 (16-week) plans.
5. FormBeat (formerly TriDot) - planner mechanics described in
   https://tridot.com/the-science (private SaaS, only the published
   coaching whitepapers are public).
6. Anthropic Extended Thinking docs:
   https://docs.claude.com/en/docs/build-with-claude/extended-thinking
   - Section "Important considerations when using extended thinking":
     "temperature MUST be set to 1 when thinking is enabled".

## Industry consensus on Long Course weekly distribution

For an Ironman-distance (Langdistanz) build, the SHIPPED plans
overwhelmingly converge on:

| Discipline | Sessions per week (build) | Notes |
|---|---|---|
| Swim | 2 to 3 | At least one technique focus + one threshold/CSS set |
| Bike | 3 to 4 | One long, one threshold, one optional Z2/recovery |
| Run | 3 to 4 | One long, one quality (tempo or VO2max), one easy + optional brick |
| Strength | 1 to 2 (build), 0 (peak/taper) | Often hidden inside bike or run day |
| Brick | 1 per week | Bike-to-run is the canonical brick, counts toward both bike + run totals |

Total weekly sessions land at 9 to 11 in the build phase, dropping to
7 to 9 in peak and 5 to 7 in taper. Lisa's `training_days_per_week=6`
constraint means single-session days; we cannot prescribe doubles
without explicit consent, so 6 to 7 sessions/week is the realistic ceiling.

## Sport-mix HEURISTIC (for our planner)

Given a triathlon profile with 6 training days/week, the floor
constraint is:

- min 1 swim per week (technique day)
- min 2 bikes per week (one long, one threshold)
- min 2 runs per week (one long, one quality)
- 6th day: cross-training OR a second swim OR a brick

When training_days_per_week >= 7 (with explicit recovery day), add a
second swim or strength.

For Mitteldistanz (70.3, half) the same shape holds but volumes are
about half. For Sprint or Olympic distance the swim count goes UP
(faster proportion of total race time) and the bike-long shrinks.

## Distance terminology (German / English mapping)

Standardized in the German triathlon scene:

| Term | Distance (S/B/R) | English equivalent |
|---|---|---|
| Sprint | 0.75 km / 20 km / 5 km | Sprint |
| Olympische / Kurzdistanz | 1.5 km / 40 km / 10 km | Olympic / Standard |
| Mitteldistanz | 1.9 km / 90 km / 21.1 km | Half / 70.3 |
| Langdistanz | 3.8 km / 180 km / 42.2 km | Full / Ironman / IM / 140.6 |

The shorthand "70.3" refers to TOTAL miles for Mitteldistanz (1.2 +
56 + 13.1 = 70.3). "140.6" or just "Ironman" without modifier is the
Langdistanz. Challenge Roth IS a Langdistanz. Mixing these up is
exactly the failure mode the agent committed.

## How TrainingPeaks structures plan data

The TrainingPeaks plan JSON (visible in their public CSV/PDF exports)
attaches a `sport` token to each workout slot at the OUTLINE stage.
Their weekly template is therefore not just intensity-typed but
discipline-typed, e.g. `monday: swim_easy, tuesday: bike_threshold,
wednesday: run_long, ...`. The athlete sees real-time validation:
"this plan has 1 swim per week; for Ironman we recommend 3."

This is exactly the constraint our outline is missing.

## Why our current planner fails

Our `weekly_template` is intensity-only. The executor LLM (Haiku)
sees a slot like `tuesday: quality` with `available_sports=[running,
cycling, swimming]` and chooses a sport per session. Haiku's prior is
running (the default sport in our codebase, and the most common in
public training data), so it overwhelmingly defaults to running. The
sanitizer falls back to `available_sports[0]` if Haiku omits the
field, and `available_sports[0]` for Lisa is `running`.

## Decision rule for our planner

For profiles where `len(profile.sports) > 1`:

1. The planner MUST output a `sport_per_day` mapping inside
   `weekly_template` (or an alternative `sport_distribution_per_week`
   list) that assigns ONE sport per non-rest day.
2. The mapping MUST satisfy the floor heuristic for the recognised
   discipline (Long Course / Mitteldistanz / Sprint). When the goal_event
   string is unknown, default to Long Course floors.
3. The executor MUST respect the assigned `sport` and not invent its
   own.
4. Post-assembly validation MUST count session-per-sport across the
   plan and fail-closed if any required sport is absent or below the
   floor for the recognised discipline.
5. On validation fail, retry the planner ONCE with an explicit
   "MISSING SPORTS / UNDER-WEIGHT SPORTS: include at least X" message
   prepended to the user message.

## On Extended Thinking + temperature

From Anthropic's docs (linked above):
"Setting temperature to any value other than 1 while thinking is
enabled will result in a 400 BadRequestError."

Concrete implication: when our model_router picks Sonnet +
`thinking_budget > 0`, the chat_completion call MUST override
temperature to 1.0. We currently pass `AGENT_TEMPERATURE=0.7`
unconditionally, which causes the Sonnet call to fail and the
fallback to Haiku to run. That fallback is the reason Lisa never got
premium routing despite the detector firing.

## References cited inline above

- TrainingPeaks Ironman plan library
- Joe Friel "The Triathlete's Training Bible" 5e
- Final Surge Long Course plan templates (Biscay, Plumb)
- 80/20 Endurance Triathlon Level 1 / Level 2
- TriDot / FormBeat training science whitepapers
- Anthropic Extended Thinking documentation
