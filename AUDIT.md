# Sprint C Audit: Decimal-Pace Surfaces

Goal: every LLM-facing surface that emits a decimal-minute pace must be replaced (or paired) with a pretty mm:ss string. The bug Elena hit was `threshold_pace_min_km: 4.50` being rendered to the athlete as "4:50/km" because the agent had no pre-formatted variant in the runtime context block.

The existing "pretty pace" infrastructure in `src/agent/tools/format_helpers.py` (`pace_to_mmss`, `minutes_to_hms`, `minutes_to_hm`) covers `get_activities` and `get_activity_details` results but stops at the profile / runtime-context boundary.

## Surfaces inspected

| # | File | Surface | Decimal field | LLM-facing? | Action |
|---|------|---------|--------------|-------------|--------|
| 1 | `src/agent/system_prompt.py` (line 712, `build_runtime_context`) | `# Current Athlete` runtime block: `Threshold pace: {threshold_pace} min/km` | `fitness.threshold_pace_min_km` (float) | YES, every turn | FIX: pretty-format via `decimal_min_to_mmss`. Replace `min/km` with ` /km`. |
| 2 | `src/agent/tools/data_tools.py` (line 21, `get_athlete_profile` handler) | Whole profile dict returned to LLM via tool call; includes `fitness.threshold_pace_min_km` as raw decimal | `fitness.threshold_pace_min_km` | YES, on demand | FIX: post-process `project_profile()` output to add `threshold_pace_pretty` and `threshold_pace_min_km_pretty` keys. Keep raw decimal for backwards compat. |
| 3 | `src/memory/user_model.py` (line 279, `get_model_summary`) | LLM prompt-injection block: `threshold {threshold_pace_min_km} min/km` | `fitness.threshold_pace_min_km` | YES (legacy injector still callable) | FIX: pretty-format. |
| 4 | `src/db/user_model_db.py` (line 782, supabase mirror of `get_model_summary`) | Same as #3, but pulled from Supabase | `threshold_pace_min_km` | YES | FIX: pretty-format. |
| 5 | `src/agent/planner.py` (line 377, `_build_planner_prompt`) | Planner-LLM input: `threshold_pace_min_km: {value}` | `fitness.threshold_pace_min_km` | YES (planner agent) | FIX: emit pretty-format alongside, e.g. `threshold_pace: 4:30 /km`. |
| 6 | `src/agent/athlete_journal.py` (markdown blob) | Free-form markdown the coach writes via `update_journal_section` / `append_to_journal` | None (coach-authored) | Indirect: coach writes what it sees | NO ACTION (coach already shown pretty strings via #1; if it writes a decimal it is a downstream symptom, not a source). |
| 7 | `src/agent/athlete_md.py` (training + recovery live blocks) | Live aggregates (total km, total min, sleep, HR) | All already int or already aggregated; no pace decimals | YES, but no pace | NO ACTION. |
| 8 | `src/services/identity_widgets.py` (line 1227) | Frontend widget JSON props (`StatGridItem.value=str(threshold)`) | `threshold_pace_min_km` | NO (frontend widget, not LLM) | NO ACTION for Sprint C scope. Tracked as separate fix-it. |
| 9 | `src/tools/fitness_tracker.py` | Writes decimal into DB | `threshold_pace_min_km` | NO (DB write) | NO ACTION. DB stays decimal per spec. |
| 10 | `src/memory/episodes.py` (line 35, episode template string) | Documentation literal `"-0:05/km or +0:03/km or stable"` | None | NO | NO ACTION. |

## Summary

LLM-facing surfaces that need pretty formatting: 5 (rows 1, 2, 3, 4, 5).
LLM-facing surfaces already safe (pretty-only): get_activities, get_activity_details, athlete_md live blocks.
DB / frontend surfaces unchanged: identity_widgets, fitness_tracker DB writes, raw activity columns.

## Naming convention adopted

- Raw decimal field name is unchanged (`threshold_pace_min_km`).
- Pre-formatted twin lives next to the decimal under the suffix `_pretty` for tool-result dicts: `threshold_pace_pretty` (the value the LLM should quote).
- For runtime-context text blocks (`build_runtime_context`, `get_model_summary`, `_build_planner_prompt`): emit only the pretty form. The decimal does not appear in the text the LLM sees.
