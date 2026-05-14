# Research: Whole-Athlete Coaching (Q2 2026)

Goal: ground the Athletly proactive-coaching feature in established sports
science thresholds and the way leading consumer products (WHOOP, Garmin,
Strava, TrainingPeaks, Final Surge) communicate recovery state. Every
threshold the coach uses must trace back to either a peer-reviewed
finding or a widely deployed product heuristic.

## 1. Whole-athlete coaching frameworks (TrainingPeaks, Final Surge, Athletic.net)

TrainingPeaks built the de-facto vocabulary for the Performance Management
Chart (PMC). Three core values:

- CTL (Chronic Training Load): 42-day exponentially weighted average of
  TSS. Represents "fitness".
- ATL (Acute Training Load): 7-day EWMA. Represents "fatigue".
- TSB (Training Stress Balance) = CTL minus ATL. Represents "form".
  Positive TSB means fresh and ready; deeply negative TSB (under -30)
  means overreaching territory.

Source: https://www.trainingpeaks.com/learn/articles/the-science-of-the-performance-manager/

The ACWR (Acute:Chronic Workload Ratio) framework, popularised by Tim
Gabbett, treats anything between 0.8 and 1.3 as the safe zone, with
ratios above 1.5 carrying a markedly elevated injury risk in team-sport
studies. Although Gabbett's original 2016 paper has since been critiqued
for methodology, the heuristic still drives most commercial load
monitoring products.

Sources:
- Gabbett 2016, BJSM: https://bjsm.bmj.com/content/50/5/273
- Critique: https://bjsm.bmj.com/content/53/24/1517

Final Surge and Athletic.net both integrate subjective wellness check-ins
(sleep quality, soreness, mood, stress) alongside TSS. Coaches treat
three or more "red" wellness markers in a week as a trigger to discuss
the plan, not to push through it.

Source: https://blog.finalsurge.com/coaches/wellness-monitoring-best-practices/

## 2. WHOOP recovery scoring

WHOOP publishes their recovery score on a 0 to 100 scale, colour-banded:

- Green (67 to 100): well recovered, ready for strain.
- Yellow (34 to 66): moderate, recommend maintenance.
- Red (0 to 33): low recovery, recommend rest or active recovery.

Inputs (WHOOP white paper): HRV (largest weight), resting heart rate,
respiratory rate, sleep performance, prior-day strain. Their messaging
pattern when red is short, validating, and directly action-oriented:
"Your recovery is in the red. Consider lighter activity today; aim for
solid sleep tonight."

Sources:
- https://www.whoop.com/thelocker/whoop-recovery-explained/
- https://www.whoop.com/us/en/thelocker/how-do-i-improve-my-recovery/

What we steal: the colour gradient (green/yellow/red), the validating
tone, and the rule that a single low recovery day is acknowledged but a
multi-day pattern triggers a deeper conversation.

## 3. Garmin Training Readiness

Garmin's Training Readiness is a composite 1 to 100 score, exposed since
2022, with bands: Poor 1 to 25, Low 26 to 50, Moderate 51 to 75, High
76 to 90, Prime 91 to 100. Inputs (per Garmin's official docs):

- Sleep score and sleep history (last 7 days).
- Recovery time from the last activity.
- Acute load (7-day EPOC) vs optimal range.
- HRV status (7-day vs 21-day baseline).
- Stress history.

Warnings Garmin surfaces in-product:
- "Sleep below target for 3 nights" badge.
- "HRV status: low" when 7-day average drops below the 21-day baseline
  range.
- "Body battery did not fully recharge overnight" when the morning peak
  is under 50.

Sources:
- https://www.garmin.com/en-US/garmin-technology/health-science/training-readiness/
- https://support.garmin.com/en-US/?faq=lJqQbELKBg7L18Ph2j2H05

What we steal: the multi-input composite scoring, and the in-product
phrasing pattern of "X has been Y for N days".

## 4. Sleep-performance thresholds (peer-reviewed)

The literature is consistent that endurance adaptation and training
quality degrade markedly when sleep falls below 6 hours, with a sharp
cliff under 5 hours.

Key findings:

- A single night under 4 hours significantly impairs same-day power
  output (Reilly & Piercy 1994). Reaction time and submaximal-pace RPE
  rise sharply.
  Source: https://pubmed.ncbi.nlm.nih.gov/8200002/
- Three or more consecutive nights under 6 hours produces measurable
  drops in time-to-exhaustion (Skein et al. 2011, around 11 percent
  reduction in cycling TTE after 30 hours sleep deficit).
  Source: https://pubmed.ncbi.nlm.nih.gov/21618064/
- Watson et al. 2017 (AASM consensus): adult athletes need 7 to 9 hours,
  with a strong dose-response for both injury risk and learning
  consolidation below 7 hours.
  Source: https://aasm.org/clinical-resources/practice-standards/practice-guidelines/

Operational thresholds we adopt:

- One night under 4 hours (240 minutes): critical alert.
- Three or more consecutive nights under 6 hours (360 minutes): warn.
- Five or more days under 6 hours in any 7-day window: critical alert.

## 5. HRV interpretation: single day vs trend

HRV is famously noisy day to day. The robust signal lives in the rolling
average. Plews and Buchheit 2017 (J Sports Sci) recommend a 7-day
rolling RMSSD vs a 28-day baseline, with the smallest worthwhile change
set at around 0.5 standard deviations of the baseline.

Sources:
- Plews and Buchheit 2017: https://www.tandfonline.com/doi/abs/10.1080/02640414.2016.1244343
- Stanley, Peake, Buchheit 2013 (HRV recovery review): https://pubmed.ncbi.nlm.nih.gov/23852989/

Operational threshold we adopt: a 7-day HRV average that is at least 15
percent below the 30-day baseline is a meaningful negative signal worth
surfacing. A single low day is noise unless it pairs with elevated RHR
or a low recovery score on the same day.

## 6. Body Battery interpretation

Garmin documents these bands (consumer-facing, no peer-reviewed source,
but widely adopted):

- 80 to 100: fully recharged, ready for hard training.
- 50 to 80: moderate, productive training tolerated.
- 25 to 50: go easy, recovery limited.
- Under 25: rest day.

For endurance athletes specifically, Garmin's own coaching guides
recommend that a morning peak under 30 for three or more consecutive
days indicates chronic incomplete recovery and warrants a discussion
about life stressors plus reducing acute load.

Source: https://support.garmin.com/en-US/?faq=NCBd4VfWtL6OZWvJ0w5tQ7

Operational threshold we adopt: morning body_battery_high under 30 for
three or more days is a warn.

## 7. Stress (Garmin all-day stress score)

Garmin's stress is a 0 to 100 daily average derived from HRV during
non-activity periods. Bands per Garmin:

- 0 to 25: rest.
- 26 to 50: low.
- 51 to 75: medium.
- 76 to 100: high.

Endurance research (Halson 2014, Sports Medicine, monitoring stress and
recovery in athletes) treats sustained "medium plus" stress for five or
more days as a non-training stressor that competes with the recovery
budget. Coaches should surface and discuss this, even if the training
load itself looks fine.

Source: https://link.springer.com/article/10.1007/s40279-014-0253-z

Operational threshold we adopt: stress_avg over 60 for five or more
days is an info-level alert (not a stop sign, but worth checking in on).

## 8. RHR (resting heart rate) elevation

Three or more days with RHR more than 5 bpm above the 14-day baseline
is the classic Buchheit overreaching signal. Combined with low HRV or
low sleep, the predictive power rises substantially.

Source: Buchheit 2014, Frontiers Physiology
https://www.frontiersin.org/articles/10.3389/fphys.2014.00073/full

Operational threshold we adopt: 3 or more days at 14-day baseline plus
5 bpm is a warn.

## 9. Training-load spike (ACWR proxy)

Already covered under section 1. We expose a simpler proxy: weekly
volume (sum of TRIMP or duration) more than 1.5 times the prior 7-day
window is a warn-level spike alert. This catches the easy mistake of
doubling volume after a week off without acclimating.

## 10. How production AI coaches handle proactive intervention

- WHOOP Coach (text + audio, GPT-4o based) surfaces a daily morning
  summary even when not asked. Format: short validating sentence, one
  observation, one specific question or recommendation.
- Strava AI Coach (limited beta, 2024 release) currently does not
  proactively flag recovery; it answers when prompted. This is a
  deliberate design choice per their public statements: they want the
  user to control when coaching happens.
- Vora (Hyrox specialist) is closer to our model: their Slack-based AI
  coach reads HRV and sleep daily and posts a proactive nudge if the
  user has logged a hard session despite low recovery.
  Source: https://vora.training/blog/proactive-coaching/
- Form Coach (running app, in-app AI) sends a one-off push when sleep
  is under 6h for three consecutive nights with copy modelled on
  "Looks like sleep has been short. Want to talk about the plan?"

Common pattern across all of them: the proactive nudge is short, names
the observation, and ends in an open question rather than a directive.
We adopt the same shape: one sentence empathy, one suggestion or
question, never lecture.

## Summary: thresholds adopted

| Pattern | Threshold | Severity |
|---|---|---|
| sleep_low_3d | 3 plus consecutive nights under 360 min | warn |
| sleep_critical | 1 night under 240 min OR 5 plus days under 360 in 7-day window | critical |
| hrv_drop | 7-day avg drops 15 percent vs 30-day baseline | warn |
| rhr_elevation | 3 plus days at 14-day baseline plus 5 bpm | warn |
| body_battery_chronic | body_battery_high under 30 for 3 plus days | warn |
| stress_chronic | stress_avg over 60 for 5 plus days | info |
| recovery_score_low | recovery_score under 30 for 2 plus days | warn |
| recovery_critical | recovery_score under 20 today | critical |
| training_load_spike | weekly load over 1.5 times prior week | warn |

These thresholds are documented in code as constants at the top of
`src/services/recovery_alerts.py` so they can be retuned in one place
without touching pattern logic.
