---
name: goal_change
description: Handle a goal change cleanly - confirm intent, research the new event, atomically switch the goal, archive the old macrocycle, and offer to rebuild the plan. Invoke whenever the athlete signals a switch to a different target race, date, or time.
when_to_use: Athlete says something like "I want to switch to X instead", "let's change the goal", "I'm moving the date", "new target time".
required_tools:
  - update_goal
  - spawn_subagent
  - create_macrocycle_plan
  - save_macrocycle
  - append_to_journal
---

# Goal Change Workflow

A goal change is a high-impact moment. The old plan becomes stale, the
athlete's mental model shifts, and downstream things (eval criteria,
beliefs about race-pace) may need revision. Treat it deliberately.

## Process

1. **Confirm the change explicitly**. Quote back what you understood:
   "Du moechtest also vom XYZ zum ABC wechseln, target time NEW. Richtig?"
   Wait for confirmation. People sometimes brainstorm out loud without
   committing.

2. **If the new event is a specific named race**, research it first:
   ```
   spawn_subagent(task="Find date, distance, elevation, registration
   window, typical course details for <event name>. Return concrete
   facts only.")
   ```
   You need the date BEFORE calling update_goal so the target_date is
   correct.

3. **Atomically switch the goal**:
   ```
   update_goal(
     event=<new event name>,
     target_date=<YYYY-MM-DD>,
     target_time=<HH:MM:SS or null>,
     event_facts=<course details, distance, elevation, surface>,
     source=<where the facts came from, e.g. 'spawn_subagent research
             2026-05-12 + athlete confirmed'>,
     reasoning=<one short sentence WHY the athlete changed: faster
                course, longer timeline, different city, etc.>
   )
   ```
   The journal's "Current Goal" section is rewritten and a new bullet
   is appended to "Goal Timeline" for future reflection - "why did we
   abandon the old plan?". Always include event_facts, source, reasoning.

4. **Acknowledge cascading effects to the athlete**:
   - The old macrocycle has been archived (`update_goal` does this).
   - The journal's "Current Goal" is rewritten; "Goal Timeline" gets a
     bullet so the change is auditable.
   - The training plan needs to be rebuilt.

5. **Offer next steps without forcing them**. Athlete might want to
   pause and think:
   "Die neue Macrocycle und der Plan koennen wir jetzt aufbauen, oder
   du schaust dir das in Ruhe an und wir machen es naechstes Mal.
   Was passt dir?"

6. **If athlete says yes, rebuild**:
   - `create_macrocycle_plan(name=..., weeks=..., start_date=today)`
   - `save_macrocycle()` (uses cached draft)
   - `create_training_plan(macrocycle_week=1)`
   - `evaluate_plan(plan=<from cache>)` - if score < 70, regenerate
     with feedback before saving
   - `save_plan()` (uses cached draft)
   - Summarize the new plan briefly for the athlete

## Edge cases

- **Just a date shift, same event**: still use update_goal with the new
  date. The old macrocycle is archived even though event name is
  identical - because the timeline is different.

- **Soft musing, not a commitment**: do NOT call update_goal. Use
  `append_to_journal(section="Open Threads", entry="<date>: Athlete
  considering switching to <X> - revisit next session.")` to remember
  the idea. Revisit later.

- **Athlete wants to keep the old macrocycle**: do NOT call update_goal.
  Instead update the target_time only if that's what changed, or capture
  the new context with `append_to_journal(section="Open Threads", ...)`.
  The user must EXPLICITLY commit before archiving the old plan.

- **No active macrocycle currently**: update_goal still works. The
  cascade just skips the archive step. Build the new macrocycle from
  scratch.

## Anti-patterns

- Calling update_goal without confirmation. The athlete might just be
  thinking out loud.
- Skipping the research step for unfamiliar events. Don't invent dates.
- Forgetting to include reasoning. The audit log loses context.
- Rebuilding the plan before the athlete approves the goal change.
