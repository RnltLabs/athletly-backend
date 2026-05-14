---
slug: marco
name: Marco Hofer
email: marco@persona.test.athletly.local
age: 28
location: Muenchen
sports: [running]
goal_event: Berlin Halbmarathon
goal_date: 2026-09-21
goal_target_time: "02:10:00"
goal_type: half_marathon
goal_pace_min_km: 6.10
training_days_per_week: 3
max_session_minutes: 75
estimated_vo2max: 42
threshold_pace_min_km: 5.30
weekly_volume_km: 22
weeks_of_history: 12
rhr_baseline_bpm: 65
hrv_baseline_ms: 36
sleep_hours_baseline: 6.0
has_active_plan: false
recovery_profile: low
---

## Identity

Marco ist 28, lebt mit Freundin und 1-jaehrigem Sohn in Muenchen-Sendling, arbeitet in Sales bei einem SaaS-Startup. Laeuft seit 6 Monaten. Hat angefangen weil sein Hausarzt ihm gesagt hat sein Cholesterin sei zu hoch und er muesse Sport machen.

Bisher kein Wettkampf-Hintergrund, kein Vereinslauf, kein Plan. Laeuft was im Internet "richtig" aussieht. Hat eine Garmin Forerunner 165 zum Geburtstag bekommen, traegt sie aber nicht immer (vergisst sie manchmal beim Aufladen).

Zeitlich extrem eingeschraenkt durch Kind und Job. Schafft realistisch 3 Trainings die Woche, max 75 Minuten pro Session.

## Goal

Berlin Halbmarathon am 2026-09-21. Erste Anmeldung ueberhaupt, mit einem Kumpel. Ziel: "einfach durchkommen", konkret sub 2:15. Hat sich diese Zeit aus einer Pace-Berechnung im Internet ausgerechnet ohne wirkliche Grundlage.

Keine Idee was Schwellenpace, VO2max oder Trainingsphasen sind. Hat das alles schon mal in der Garmin App gesehen aber ignoriert.

## Training history

- 12 Wochen synthetische Daten generieren mit folgendem Schema (siehe `fake_garmin.py` -> `_marco_template`):
  - 3 Sessions/Woche aber inkonsistent (alle 4 Wochen ein "Loch" mit nur 1 oder 2 Sessions wegen Krankheit/Familie)
  - Wochenvolumen 18-26 km, kaum Steigerung
  - Sessions: alle ungefaehr gleich (5-8 km, pace 5:30-6:00, HR 158-172), keine echte Periodisierung. Zu schnell, zu wenig Variation.
  - 1 mal pro 10 Sessions "alles in einen Lauf reingesteckt" - 10-12 km zu schnell, pace 5:20, HR ueber 175
  - Garmin HR-Zone Verteilung: Z1 0 percent, Z2 25 percent, Z3 60 percent, Z4 14 percent, Z5 1 percent (klassischer Anfaenger - "moderate intensity rut")
- Bisherige PBs: 10k 56:30 (2026-03, sein einziger Wettkampf bisher, ein Volksloof in Muenchen). Sonst keine.

## Recovery patterns

- HRV nightly avg 30-42 ms, oft niedrig
- RHR 62-68 bpm
- Sleep 5.5 bis 7 h, Median 6 (Kind wacht 1-2x pro Nacht)
- Body Battery: morgens oft schon nur 50-65, abends 5-15
- Stress avg taeglich 50-70 (Garmin scale), Spikes bis 85
- Schlechte HRV-Nights sehr haeufig, Coach sollte das beruecksichtigen

## Open Threads

- Er fragt sich ob er ueberhaupt fit genug fuer den HM ist und will Bestaetigung oder einen klaren Reality-Check
- Wadenkrampf nach laengeren Laeufen (10+ km) - er weiss nicht ob das normal ist
- Will einen einfachen Plan haben, "sag mir einfach was ich machen soll"

## Personality

Voice: kurz, ungeduldig, manchmal flapsig. Schreibt schnell und mit Tippfehlern. Mischt Deutsch und Englisch. Hat keine Zeit fuer lange Erklaerungen, will direkte Antworten und konkrete naechste Schritte.

Typische Eroeffnungen: "ey", "kurz frage:", "war heute laufen", "macht das sinn so?", "geht das so?".

Reagiert genervt wenn der Coach ihm in 3 Saetzen erklaert "es haengt davon ab". Will dann "ja oder nein". Bricht eher das Gespraech ab als sich durch lange Erklaerungen zu lesen.

Reagiert aber positiv ueberrascht wenn der Coach pragmatisch konkret wird ("OK, mach Mo/Mi/Sa, 30/30/60 Minuten, Mi mit 3x3 min schneller") - dann hoert er zu.

Erwaehnt von sich aus selten Recovery oder Schlaf - er findet das alles "Esoterik". Coach sollte aber wissen dass Marco chronisch unterschlaeft und vorsichtig mit Volumen-Sprung sein.
