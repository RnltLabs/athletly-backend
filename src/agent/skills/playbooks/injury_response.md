---
name: injury_response
description: Handle an injury or pain report from the athlete - assess severity, immediately reduce upcoming load if needed, capture the constraint as a long-term belief, and offer a recovery-focused adjustment to the plan.
when_to_use: Athlete reports pain, soreness beyond DOMS, recurring discomfort, a fall, an acute incident, or a medical diagnosis. Triggers regardless of whether the athlete uses the word "injury".
required_tools:
  - append_to_journal
  - update_journal_section
  - adjust_plan
  - spawn_subagent
---

# Injury Response Workflow

Injuries are the highest-stakes moment in coaching. Wrong response (push
through, ignore) costs weeks or seasons. Right response (acknowledge,
adjust, document) keeps the athlete progressing.

## Step 1: Assess (do not skip)

Ask 2-3 targeted questions to gauge severity:

- "Wo genau tut es weh und seit wann?"
- "Auf einer Skala 1-10, wie stark ist der Schmerz beim Laufen / in
  Ruhe?"
- "Hattest du das schon mal? Diagnose oder Vermutung?"
- (Only if relevant) "Schwellung, Rötung, Bewegungseinschränkung?"

Stop after the athlete answers - do NOT pile on more questions. You
have what you need.

## Step 2: Triage

Based on the answers, place the report into one of three buckets:

| Bucket | Signals | Response |
|---|---|---|
| **Acute / Red flag** | sharp pain, swelling, locking joint, inability to bear weight, pain >7/10 | Recommend medical/physio consultation. Pause running entirely. Cross-training only if pain-free. |
| **Persistent niggle** | recurring discomfort, dull ache after sessions, lasting >3 days | Reduce volume 30-50%, switch high-impact to low-impact, monitor 7-10 days. Add cross-training (cycling, swimming, strength). |
| **Normal soreness** | DOMS-like, fades within 48h, no functional limit | No change. Reassure. Maybe one extra easy day. |

If unclear, use `spawn_subagent(task="safety review: ...",
tools_scope="readonly", context={profile, recent_activities, injury_report})`
for a structured second opinion before recommending changes.

## Step 3: Adjust the plan

Use `adjust_plan` (if available) or describe specific session swaps:

- High-impact running -> easy cycling or swimming
- Hard intervals -> moderate steady
- Long run -> mobility + strength
- Add 1-2 extra rest days

The athlete must approve before any actual plan-row mutation.

## Step 4: Capture in the journal

Two writes (different purposes):

1. `append_to_journal(
     section="Open Threads",
     content="<location> pain reported <date>, severity <X/10>, suspected <cause>"
   )`
   Time-bound note. Remove with remove_from_journal when athlete reports clear.

2. `update_journal_section(
     section="Identity",
     content="...Anfaellig fuer <body part> bei <trigger>; im Plan beruecksichtigen..."
   )`
   ONLY if it's a chronic / recurring issue, not a one-off. This stays
   in the long-term identity.

## Step 5: Set up follow-up

Acknowledge the timeline:
"Lass uns in 3-5 Tagen nochmal draufschauen. Wenn der Schmerz dann
weg ist, bauen wir vorsichtig wieder auf. Wenn er bleibt oder
schlimmer wird, brauchst du physiotherapeutische Abklaerung."

The proactive heartbeat will surface this for re-check.

## Anti-patterns

- "Pace dich ein bisschen" without a concrete plan adjustment. Vague
  empathy is unhelpful.
- Skipping the assessment questions and jumping to "rest 2 days".
  Athletes need to FEEL heard.
- Adding to the Identity section for a one-off bruise. Reserve that
  section for CHRONIC or RECURRING constraints.
- Forgetting to remove / invalidate the Open Threads note later when
  the athlete reports recovery.
- Diagnosing. You are NOT a doctor. Suggest professional consultation
  for anything in the Red Flag bucket.
