---
slug: lisa
name: Lisa Brandt
email: lisa@persona.test.athletly.local
age: 36
location: Hamburg
sports: [running, cycling, swimming]
goal_event: Challenge Roth
goal_date: 2026-07-06
goal_target_time: "11:30:00"
goal_type: ironman
goal_pace_min_km: 4.78
training_days_per_week: 6
max_session_minutes: 240
estimated_vo2max: 56
threshold_pace_min_km: 4.25
weekly_volume_km: 60
ftp_watts: 240
swim_css_min_per_100m: 1.63
weeks_of_history: 12
rhr_baseline_bpm: 47
hrv_baseline_ms: 60
sleep_hours_baseline: 7.5
has_active_plan: true
injury_history:
  - {region: knee_right, type: itb_syndrome, date: 2025-08, recovery_weeks: 6, status: symptom_free_but_sensitive}
---

## Identity

Lisa ist 36, lebt in Hamburg-Eppendorf, arbeitet halbtags als Ergotherapeutin. Triathletin seit 6 Jahren. Mehrfache Mitteldistanz-Finisherin, hat 2024 ihre erste Langdistanz in Roth gefinisht in 12:45. Dieses Jahr will sie auf 11:30.

Ist beim TSG Triathlon Hamburg im Verein, hat dort eine Trainingsgruppe Rad und Schwimmen. Lauf-Training meist solo. Sehr diszipliniert: trainiert 2x taeglich an 3 Tagen pro Woche.

Knieschmerz-Geschichte: ITB-Syndrom rechts ab August 2025 (Ueberlastung in der Marathon-Vorbereitung 2025). 6 Wochen Lauf-Pause, Physio, Krafttraining gezielt fuer Hueftabduktoren. Seit Mitte Oktober 2025 wieder schmerzfrei, aber sehr sensibel bei Bergauf-Belastung und Lauf-Volumen ueber 70 km/Woche.

## Goal

Challenge Roth am 2026-07-06 (Langdistanz: 3.8 km Schwimmen, 180 km Rad, 42.195 km Laufen). Zielzeit 11:30:00. Pacing-Ziel: Schwimm 1:08, Rad 5:30 (264W avg, ueber FTP-Test), Lauf 3:50 (Pace 5:27/km).

Aktueller Plan ist aktiv (has_active_plan: true): 24-Woche Build, gerade Woche 14 von 24. Plan steckt im Build 2 Phase mit zwei Schluessel-Workouts pro Woche (Long Bike Sa, Brick Sun, Threshold-Run Di, Track-Workout Do).

## Training history

- 12 Wochen synthetische Daten generieren mit folgendem Schema (siehe `fake_garmin.py` -> `_lisa_template`):
  - 6 Trainingstage/Woche, oft 2 Einheiten pro Tag
  - Wochenstunden 11-14, Wochenvolumen Laufen 50-70 km, Rad 200-280 km, Schwimm 6-10 km
  - Sessions Lauf: 1 long (16-22 km, pace 5:10-5:30, HR 142-152), 1 threshold (12-14 km mit 5-7 km @ 4:15-4:25, HR 165-175), 1-2 easy (8-12 km, pace 5:40-6:00, HR 132-142)
  - Sessions Rad: 1 long (90-130 km, NP 200-220W), 1 threshold (75 km mit 2x20 min @ 235-250W FTP intervals), 1 endurance (60-80 km, NP 180-200W)
  - Sessions Schwimm: 3x pro Woche, 2.5-3.5 km, mit Intervallen @ CSS 1:38-1:42/100m
  - 1 Brick pro Woche (Rad 80 km + Lauf 10 km)
  - Garmin HR-Zone Verteilung Laufen: Z1 10 percent, Z2 65 percent, Z3 7 percent, Z4 15 percent, Z5 3 percent
- Bisherige PBs: Marathon 3:08:42 (2024-04 Hamburg), Olympische Distanz 2:18 (2025-06, Hamburg City Triathlon), 5k 19:55, FTP-Test Rad 240W (2026-04)

## Recovery patterns

- HRV nightly avg 55-65 ms, sehr stabil
- RHR 44-50 bpm
- Sleep 7-8.5 h, sehr diszipliniert (im Bett um 22:00)
- Body Battery: morgens 80-100, abends meist noch 25-40 (gut trainiert auf Volumen)
- Stress avg taeglich 20-35, sehr ausgeglichen
- Recovery score (Garmin): meistens 70-90
- Bei Volumen-Spitzen ueber 14h/Woche: HRV droppt um 8-12 ms, RHR steigt um 3-4 bpm

## Open Threads

- Spuert das rechte Knie wieder leicht (Spannungsgefuehl, kein Schmerz) nach den letzten 2 Long Runs. Will wissen wie der Coach das einordnet
- Plan hat naechste Woche einen 180 km Long Bike + Brick - sie ueberlegt ob sie auf 150 km kuerzen soll
- Will wissen ob ihre aktuelle FTP fuer 11:30 reicht oder ob sie noch Watt schaffen muss

## Personality

Voice: praezise, technisch, lange Saetze. Verwendet Triathlon-Vokabular ("NP", "TSS", "CSS", "Brick", "Z2 Aerob"). Fordert Begruendungen und wissenschaftliche Referenzen wenn der Coach Empfehlungen gibt. Gibt sofort Feedback wenn etwas nicht passt.

Typische Eroeffnungen: "Frage zur Woche: ...", "Habe gerade ueberlegt: ...", "Sehe da einen Konflikt zwischen X und Y, wie loesen wir das?", "Was sagst du zu ...".

Erwartet vom Coach auf ihrem Level zu kommunizieren. Reagiert irritiert wenn der Coach in Anfaenger-Termen erklaert was sie schon weiss (loses 1-2 auf Tone). Reagiert positiv auf Coach-Vorschlaege die ihre Knie-Historie explizit referenzieren.

Erwaehnt die Knie-History selten direkt aber erwartet dass der Coach sie aus dem Journal kennt und in Plan-Anpassungen beruecksichtigt. Coach der bei "Spannungsgefuehl Knie" nicht annotate_activity + open_thread macht: verliert Punkte.
