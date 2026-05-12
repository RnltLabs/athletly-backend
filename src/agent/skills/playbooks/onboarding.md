---
name: onboarding
description: Guide a new athlete through profile setup in natural conversation. Invoke when profiles.onboarding_complete is false or when key identity fields are empty (no name, no sports, no goal). One question at a time, warm tone, no checklist feel.
when_to_use: Athlete is new or has incomplete core profile (missing name, sports, or goal). NOT a separate mode - just a workflow the coach follows inline.
required_tools:
  - update_profile
  - update_goal
  - update_journal_section
  - append_to_journal
  - spawn_subagent
---

# Onboarding Workflow

You are onboarding a new athlete. The goal is to fill in the core profile
fields while making it feel like a normal conversation, not a form.

## Required core fields (in priority order)

1. **Name** - profile.name
2. **Sports** - profile.sports (array)
3. **Goal event** - profile.goal_event + target_date + target_time
4. **Training constraints** - profile.training_days_per_week + profile.max_session_minutes

## Process

1. **Inspect what is missing**. The athlete profile block in your runtime
   context shows current state. Identify the first unfilled field in the
   priority order above.

2. **Ask ONE question naturally** about that field. Conversational, not
   form-style. Examples:
   - "Wie soll ich dich nennen?" (not "Bitte gib deinen Namen ein.")
   - "Was sind deine Hauptsportarten?"
   - "Hast du ein konkretes Ziel - ein Rennen, ein Event?"
   - "Wie viele Tage in der Woche kannst du realistisch trainieren?"

3. **When the athlete answers**, call the matching tool IMMEDIATELY:
   - Name: `update_profile(field="name", value="...")`
   - Sports: `update_profile(field="sports", value=["running", "cycling", ...])`
   - Goal event: if the user names a specific race, first
     `spawn_subagent(task="Find date, distance, course details for <event>")`
     to verify, then `update_goal(event=..., target_date=..., target_time=..., reasoning=...)`
   - Training days: `update_profile(field="constraints.training_days_per_week", value=N)`
   - Max session: `update_profile(field="constraints.max_session_minutes", value=N)`

4. **Capture extra signals in the journal** alongside the structured updates.
   - Identity / lifestyle / body facts ("32, lebt in Karlsruhe, ein Kind"):
     `update_journal_section(section="Identity", content="...")` or
     `append_to_journal(section="Identity", entry="...")`.
   - Coaching preferences ("laufe lieber morgens, mag keinen Asphalt"):
     `update_journal_section(section="Preferences", content="...")` or
     `append_to_journal(section="Preferences", entry="...")`.
   - Anything that should be checked next session (injury, decision pending):
     `append_to_journal(section="Open Threads", entry="<date>: ...")`.

5. **Move to next missing field**. Repeat until all four required fields
   are filled.

6. **When complete**, do NOT announce "onboarding done". Just transition
   naturally into coaching: confirm the picture briefly, then ask what
   they want to do next ("Soll ich dir mal einen ersten Trainingsplan
   bauen?"). The transition is INVISIBLE to the athlete.

## Anti-patterns to avoid

- Asking multiple questions at once. One question, one answer.
- Treating it like a form. "Schritt 3 von 5" is wrong.
- Skipping the tool call. Every answer gets persisted immediately.
- Generic chitchat that adds no progress. Every turn should advance.
- Calling save_macrocycle/create_training_plan before the four core fields
  are filled. Onboarding first, planning second.

## Disambiguation

If the athlete names a sport you cannot uniquely classify (e.g. "Trail")
or an event you cannot identify (e.g. "Heidelberg Marathon"), use
`spawn_subagent` to research before calling update_profile/update_goal.
Do NOT guess.

## Success criteria

The skill has succeeded when:
- profile.name is set
- profile.sports has >= 1 entry
- profile.goal_event, goal_target_date, goal_target_time are all set
- profile.training_days_per_week is set
- profile.max_session_minutes is set

After that, set onboarding_complete=true via
`update_profile(field="onboarding_complete", value=true)` (note: this
field is a bool, the tool accepts strings that get parsed).
