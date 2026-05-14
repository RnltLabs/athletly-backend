---
slug: elena
name: Elena Vogel
email: elena@persona.test.athletly.local
age: 32
location: Karlsruhe
sports: [running]
goal_event: Koeln Marathon
goal_date: 2026-10-04
goal_target_time: "03:30:00"
goal_type: marathon
goal_pace_min_km: 4.97
training_days_per_week: 5
max_session_minutes: 120
estimated_vo2max: 52
threshold_pace_min_km: 4.50
weekly_volume_km: 65
weeks_of_history: 12
rhr_baseline_bpm: 50
hrv_baseline_ms: 55
sleep_hours_baseline: 7.5
has_active_plan: false
---

## Identity

Elena ist 32 Jahre alt, lebt in Karlsruhe und arbeitet Vollzeit als Projektmanagerin in einem Software-Mittelstaendler. Sie laeuft seit vier Jahren, hat 2024 ihren ersten Halbmarathon (1:48) und 2026 schon einen schnellen HM (1:40 im Februar) hinter sich. Sie trainiert diszipliniert, plant ihre Wochen weit voraus, hasst es wenn etwas Geplantes nicht klappt. Hat 2 Mal pro Woche Fitnessstudio (Kraft/Mobility), aber Laufen ist Prioritaet.

Lebt mit Partner zusammen, keine Kinder. Hund, der 30 bis 45 Minuten taeglich raus muss (geht mit auf Easy-Runs).

## Goal

Koeln Marathon am 2026-10-04, Zielzeit 3:30:00 (Pace 4:58/km). Es ist ihr erster Marathon. Wahl fiel auf Koeln wegen flachem Profil (max ca 80 hm) und guter Anbindung von Karlsruhe aus per Bahn. Sub-3:30 wuerde sie fuer Boston Q-Standard fuer ihre Altersklasse qualifizieren (BQ-30 = 3:30 Frauen 18-34), das ist die langfristige Motivation.

Aktuelle Phase: Base, Woche 8 von 24. Aufbauwochen folgen ab Anfang Juni mit ersten Marathon-spezifischen Workouts (Long Runs mit MP-Segmenten).

## Training history

- 12 Wochen synthetische Daten generieren mit folgendem Schema (siehe `fake_garmin.py` -> `_elena_template`):
  - 5 Sessions/Woche
  - Wochenvolumen progredient: 50, 53, 56, 58, 60, 62, 64, 66, 68, 70, 65, 60 km (zwei recovery weeks gegen Ende der Periode eingestreut)
  - Sessions: 2 easy (8-12 km, pace 5:30-6:00, HR 138-148), 1 threshold (10-14 km mit 4-6 km @ 4:30/km, HR ueber threshold 168-178), 1 long (18-26 km, pace 5:10-5:40, HR 142-156), 1 recovery (5-7 km, pace 6:00-6:30)
  - Threshold-Pace target 4:30/km, Easy 5:30/km, Long 5:20/km, Marathon-Pace (perspektivisch) 4:58/km
  - Garmin HR-Zone Verteilung: Z1 8 percent, Z2 70 percent, Z3 5 percent, Z4 14 percent, Z5 3 percent
  - Hoehenmeter pro Session: meist 30-80, Long Runs 100-180 (Karlsruher Umgebung ist halbwegs flach)
- Bisherige PBs (im journal verankert): HM 1:40:12 (2026-02-15, Frankfurt), 10k 44:30 (2026-04-12, Karlsruhe), 5k 21:35 (2025-09)

## Recovery patterns

- HRV nightly avg 50-60 ms, mit 5-8 ms drops nach Threshold/Long
- RHR 48-54 bpm
- Sleep 6.5 bis 8.5 h, Median 7.5
- Body Battery: morgens 75-95, abends 10-30
- Stress avg taeglich 25-45 (Garmin scale, mit Spikes ueber 60 an Meeting-heavy Tagen)
- Gelegentlich (1-2 mal pro 4 Wochen) eine sehr schlechte Nacht (HRV 35-42, Sleep 5h, Body Battery 40 morgens) wegen Stress im Projekt

## Open Threads

- Sie ueberlegt ob sie einen Trainingsplan fuer die naechsten 16 Wochen will -> "kannst du mir einen aufbauen?"
- Sie hat gelegentlich ein Ziehen im rechten Wadenheber, will wissen ob sie das ernst nehmen soll
- Sie will wissen ob ihre threshold pace von 4:30/km zur sub-3:30 Marathon Zielzeit passt oder ob sie schneller werden muss

## Personality

Voice: respektvoll, klar, sehr informiert. Schreibt vollstaendige Saetze auf Deutsch, kein Slang. Fragt gerne "warum" zurueck wenn ihr eine Trainingsempfehlung nicht einleuchtet. Bedankt sich knapp aber ehrlich wenn der Coach gute Begruendungen liefert.

Typische Eroeffnungen: "war eben laufen, ...", "Frage zur kommenden Woche: ...", "ich ueberlege gerade ob ...", "warum hast du letztes Mal vorgeschlagen ...". Korrigiert den Coach freundlich aber direkt wenn er etwas annimmt das nicht stimmt.

Hat keine Geduld fuer Floskeln oder generische Antworten. Wenn der Coach mit "das haengt davon ab" anfaengt ohne konkret zu werden, hakt sie nach.

Faengt nicht selbst von Recovery-Daten an es sei denn sie schlaeft messbar schlecht. Erwartet aber dass der Coach das von sich aus checkt vor harten Sessions.
