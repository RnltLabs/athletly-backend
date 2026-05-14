# Sprint B - Live Diagnosis: Lisa's missing Plan-and-Execute trigger

Diagnosis run on Hetzner production container `athletly-api-1`.
Today: 2026-05-14.

## Lisa's profile

- `user_id`: `a9ac9632-fddc-49d3-bacd-984210a275e7`
- `name`: Lisa Brandt (test persona, is_test_persona=true)
- `goal_event`: Challenge Roth, target_date 2026-07-06, target_time 11:30:00
- `training_days_per_week`: 6
- `max_session_minutes`: 240
- `available_sports`: running, cycling, swimming
- `estimated_vo2max`: 56
- `threshold_pace_min_km`: 4.25

## Active plan she actually got

Plan id `6df773cb-3488-4a05-a13c-e9bd58ad7d2c`, saved 2026-05-14T11:11:13 UTC.

Key facts read from `plans.plan_data`:

| Field | Value |
|---|---|
| focus | "Build 2: Langdistanz-Spezifik (8 Wochen bis Roth)" |
| start_date | 2026-05-18 |
| duration_weeks | None (absent from dict) |
| session_count | 16 |
| unique_dates | 14 |
| date range | 2026-05-18 to 2026-05-31 (= 2 weeks) |
| has `_generation_meta` | False |
| has `outline` | False |
| has `_saved_at` | True |
| top-level keys | `_saved_at`, `focus`, `sessions`, `start_date` |

The label says "8 Wochen bis Roth" but the plan dict only covers 2 weeks
(14 unique dates from 2026-05-18 through 2026-05-31). The agent
announced an 8-week build, then only saved the first 2 weeks inline.

`_generation_meta` is absent: the Plan-and-Execute pipeline never ran.
If it had, `_assemble_plan` would have stamped `_generation_meta.mode =
"plan_and_execute"` plus an `outline` block (planner.py:653-677). Both
are missing, so the inline (one-shot Haiku) path is the only one that
fired.

Also: the immediately-prior plan row (`1fb265f5...`, created 1 minute
earlier) has the SKINNY request shape: keys = `goal_date, goal_event,
goal_target_time, meta, weeks`. That is the shape the planner pipeline
wants: agent saved it briefly, then overwrote with the inline 2-week
Haiku dict 67 seconds later. Hint that the agent oscillated between
skinny and full shapes.

## Logs

`docker logs athletly-api-1 --since 3h | grep -iE "model_router decision|premium|sonnet"`: **no matches**.

The router's `log_choice()` line `model_router decision tier=... model=...
thinking=... premium=...` (model_router.py:113-121) never fired with
`premium=True` during Lisa's session. Every chat_completion ran on
Haiku.

Other noise (unrelated to this bug, noted for context):
- Repeated `Embedding failed after 2 retries (provider=gemini/text-embedding-004)` and `LLM compression failed model=gemini/gemini-2.5-flash error=... API_KEY_INVALID`.
  Gemini API key is broken in prod. Embeddings/compression both downgrade. Not the cause of THIS bug but worth flagging upstream.
- Several `Critic call timed out after 1.5 s, failing open`. Tier 1 ITPM pressure (Anthropic 50k limit).

## Conclusion

The Plan-and-Execute pipeline never fired for Lisa. The plan stored in
the database is a pure inline Haiku dump of just the first 2 weeks of
what was meant to be an 8-week build. Neither the heuristic nor the
planner produced anything. Root-cause analysis follows in
`ROOT_CAUSE.md`.
