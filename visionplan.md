# Athletly — Vollstaendige Product Vision & Roadmap

> Dieses Dokument ist die Single Source of Truth fuer die Athletly (ehemals AgenticSports) v2 Entwicklung.
> Es enthaelt alle Designentscheidungen, technischen Details und den vollstaendigen Kontext
> damit eine neue Session nahtlos weiterarbeiten kann.

---

## 0. Anleitung fuer neue Sessions

### Was ist dieses Dokument?
Dies ist die vollstaendige Product Vision fuer **Athletly** — eine Mobile-First Multi-Sport AI Coach App. Zwei existierende Projekte werden zusammengefuehrt: ein Python AI Agent (AgenticSports) und eine React Native App (athletly-app).

### Aktueller Status
- **Phase**: Phasen 1–8 weitgehend umgesetzt (Agent Loop, Tools, Plan Generator, Episodic Memory, Proactive Triggers, Multi-User Readiness, Cost Monitoring, Garmin Sync)
- **Naechster Schritt**: Verbleibende Luecken schliessen, End-to-End Tests, Production Hardening
- **athletly-backend** (`~/Development/athletly/athletly-backend`): Python FastAPI Backend — GitHub: `RnltLabs/athletly-backend`
- **athletly-app** (`~/Development/athletly/athletly-app`): React Native/Expo App (V2) — GitHub: `RnltLabs/athletly-app`
- **athletly-v1**: Archiviert auf GitHub (`RnltLabs/athletly-v1`)
- **Supabase**: Laufende Instanz (URL + Keys in athletly-app `.env`)

### Wichtigste Regel beim Lesen
**Das fundamentale Designprinzip** (Sektion 3) durchdringt ALLES: Die App ist eine generische Plattform. Der Agent ist die gesamte Intelligenz. NICHTS ist hardcodiert — keine Sportarten, keine Formeln, keine Regeln, keine Bewertungskriterien. Wenn du Code schreibst und merkst dass du etwas sport-spezifisches hardcodest: STOPP. Das muss der Agent zur Laufzeit entscheiden.

### Referenzen im Dokument
- **ClawdBot / PicoClaw / OpenClaw**: Agentic AI Framework (430k+ LOC). Athletly uebernimmt die Kern-Prinzipien (Agent + Tools + Chat = Universal Interface) in vereinfachter Form. Nicht uebernommen: Gateway Control Plane, Lane Queue, Block Streaming, Session Bootstrap Files — diese sind fuer Multi-Tenant Enterprise-Scale relevant, nicht fuer unser MVP
- **NanoBot** (github.com/HKUDS/nanobot): "The Ultra-Lightweight OpenClaw" (~3,935 LOC). Direkte technische Referenz fuer: `_bus_progress` Pattern (Thinking + Tool Hint Events), HeartbeatService (proaktive Intelligenz), System Prompt Caching, Error-Response-Handling, MessageTool (Mid-Turn Push), SpawnTool (Background Subagents)
- **MoltBot**: Anderes Projekt von Roman. Gleiche Agent-Architektur (LLM + Tools + Chat). Zeigt dass das Pattern funktioniert
- **Claude Code**: Referenz fuer UX. So wie Claude Code jede Frage ueber Code beantwortet, soll Athletly jede Frage ueber Training beantworten — direkt im Chat, ohne auf andere Screens zu verweisen

### Sprache
- **App-UI**: Deutsch (bereits so gebaut in athletly-app)
- **Agent/Coach**: Spricht die Sprache des Users (erkennt automatisch)
- **Code/Docs**: Englisch

---

## 1. Kontext & Ausgangslage

### 1.1 Wer ist der User?

**Roman** — ambitionierter Hobby-Athlet:
- Sportarten: Laufen (Halbmarathon-Ziel), Rennrad, Schwimmen (1x/Woche), Gym (Muskelaufbau)
- 73kg, 180cm, lean
- Problem: Runna/Garmin Coach sind Single-Sport. Kein Tool koordiniert ALLE Sportarten
- Will keinen echten Coach bezahlen, aber einen intelligenten AI Coach der alles sieht

### 1.2 Zwei existierende Projekte

**Projekt 1: athletly-backend** (ehemals AgenticSports) — Das Hirn
- Repo: `RnltLabs/athletly-backend` — Pfad: `~/Development/athletly/athletly-backend`
- Tech: Python, CLI-basiert, 12,353 LOC
- LLM: Gemini 2.5 Flash via LiteLLM (model-agnostic)
- Daten: 5,975 FIT Files, 434 Aktivitaeten, 11 Sportarten erkannt
- Kern-Features:
  - **Agent Loop** (`src/agent/agent_loop.py`): 25 Tool-Rounds, Context Compression bei >40 Messages
  - **LLM** (`src/agent/llm.py`): LiteLLM, Thinking Budget 10k Tokens fuer Gemini 2.5
  - **Plan Generator** (`src/agent/prompts.py`, `src/agent/tools/planning_tools.py`): Generiert Wochenplan als JSON
  - **Plan Evaluator** (`src/agent/plan_evaluator.py`): 6 Kriterien (Sport Distribution 25%, Target Specificity 20%, Constraint Compliance 20%, Volume Progression 15%, Session Variety 10%, Recovery Balance 10%), Score >= 70/100, bis zu 3 Iterationen
  - **Training Assessment** (`src/agent/assessment.py`): Plan vs. Actual Vergleich, Compliance Score
  - **Episodic Memory** (`src/agent/reflection.py`): Weekly Reflections, Lessons, Patterns
  - **User Model** (`src/tools/user_model.py`): Belief-basiert, Confidence-Weighted
  - **TRIMP** (`src/tools/metrics.py`): Banister-Formel, HR-Zonen Karvonen — HARDCODIERT, muss generisch werden
  - **Fitness Tracking** (`src/tools/fitness_tracker.py`): Threshold Pace, Volume — LAUFZENTRIERT, muss generisch werden
  - **Trajectory Assessment** (`src/agent/trajectory.py`): Goal-Projektion mit Confidence Score
  - **Proactive Triggers** (`src/agent/proactive.py`): Fatigue Warning, Goal at Risk, etc. — HARDCODIERTE Trigger, muessen Agent-definiert werden
  - **FIT Parser** (`src/tools/fit_parser.py`): Binary FIT → JSON, 11 Sport-Normalisierungen
  - **System Prompt** (`src/agent/system_prompt.py`): Dynamisch generiert pro Turn, inkl. Startup-Context
  - **Config** (`src/config.py`): Pydantic Settings v2, `agenticsports_model` default Gemini 2.5 Flash
  - **Startup** (`src/agent/startup.py`): Goal Type Inference, Activity Context Building
  - **Activity Context** (`src/tools/activity_context.py`): Per-Sport Performance Summaries (letzte 28-56 Tage)
- Storage: JSON Files (`data/`) oder optional Supabase
- WICHTIG: Vieles ist sport-spezifisch hardcodiert (TRIMP, HR-Zonen, Lauf-Pace, Plan-Schema). Muss generisch werden

**Projekt 2: athletly-app** (V2) — Der Koerper
- Repo: `RnltLabs/athletly-app` — Pfad: `~/Development/athletly/athletly-app`
- Tech: React Native 0.81.5, Expo SDK 54, React 19.1, TypeScript
- Routing: expo-router 6 (file-based)
- State: Zustand v5 (6 Stores: user, chat, weeklyPlan, theme, sports, healthProvider)
- UI: NativeWind v4 + Custom StyleSheet, Reanimated v4, expo-blur, expo-linear-gradient
- Icons: Ionicons via @expo/vector-icons
- Backend: Supabase (Auth, PostgreSQL, Realtime, Edge Functions)
- Bundle ID: com.athletly.app, Scheme: athletlyapp
- EAS Build konfiguriert (dev, preview, production)
- **Screens:**
  - `app/(auth)/`: Login, Register, Forgot Password (Supabase Auth)
  - `app/(onboarding)/`: 9 Schritte — welcome, coach-name, sports, goals, training-mode, health-provider, garmin-connect, physical-data, ready
  - `app/(tabs)/index.tsx`: "Heute" — Tages-Session Card, Wochenprogress, Recovery Insight
  - `app/(tabs)/plan.tsx`: Trainingsplan — WeekCalendarStrip + DayDetailCard (Split-View)
  - `app/(tabs)/coach.tsx`: Chat — Inverted FlatList, SSE Streaming, Agent Status, Checkpoints
  - `app/(tabs)/tracking.tsx`: Aktivitaet tracken — Sport-Auswahl, Gym Body Parts
  - `app/(tabs)/profile.tsx`: Profil — 12 Sub-Screens (Appearance, Connected Services, etc.)
  - `app/workout/live.tsx`: Live Workout Timer
  - `app/workout/summary.tsx`: Post-Workout Summary
- **Key Components:**
  - `components/plan/WeekCalendarStrip.tsx`: 7 Tage, SVG Progress Rings, Sport-Icons
  - `components/plan/DayDetailCard.tsx`: Glassmorphism, Session Details, Coach Tip, Actions
  - `components/plan/WeeklySummaryCard.tsx`: Sport-Verteilung, Stunden, Coach Message
  - `components/chat/`: Chat UI, Typing Indicator, Agent Status
  - `components/ui/`: Button (5 Variants), Card (4 Variants), Badge, Input, ProgressRing, Skeleton, BottomSheet, etc.
- **Supabase Edge Functions** (aktueller Agent — wird ersetzt):
  - `supabase/functions/chat-stream/`: SSE Streaming Chat
  - `supabase/functions/chat/`: Non-Streaming Chat
  - `supabase/functions/confirm-action/`: Checkpoint Confirm
  - `supabase/functions/_shared/agent/`: ReAct Agent (TypeScript)
  - `supabase/functions/_shared/tools/`: plan.ts, activity.ts, week-summary.ts, session-management.ts, user-preferences.ts, research.ts
- **DB Tabellen** (aus Migrations):
  - users, garmin_activities, garmin_daily_stats, goals, training_plan, weekly_plans, chat_messages, agent_status, health_activities, push_tokens, sport_research_cache, rate_limits
- **Health Integration:**
  - Garmin: `lib/services/garminService.ts`
  - Apple Health: `@kingstinct/react-native-healthkit`
  - Android Health Connect: `react-native-health-connect`
- **Streaming:** `react-native-sse` (EventSource Polyfill)
- **Design System:**
  - Primary: Royal Blue #2563EB (HubFit-inspired design)
  - Dark Theme: Background #0C1115, Surface #151C21
  - 8 Theme Presets, Light/Dark/System Mode
  - Sport-spezifische Farben in tailwind.config.js
- **Stores:**
  - `store/userStore.ts`: Auth, Preferences, Coach Name/Style, Physical Data
  - `store/chatStore.ts`: Messages, Session ID, Checkpoint State, Realtime Status
  - `store/weeklyPlanStore.ts`: Weekly Plan CRUD, Week Navigation, Supabase Sync, Persist
  - `store/themeStore.ts`: Color Preset + Mode, Persist
  - `store/sportsStore.ts`: User Sports (Training vs. Tracking Mode)
  - `store/onboardingStore.ts`: Step Tracking
  - `store/healthProviderStore.ts`: Health Provider Connection
  - `store/metricsStore.ts`: Cached Health Metrics

### 1.3 Ziel

**Das Hirn (Python Agent) in den Koerper (React Native App) verpflanzen — und beides radikal generisch machen. Dann als Produkt launchen.**

---

## 2. Product Vision

> **Athletly ist dein persoenlicher AI Coach fuer JEDEN Sport — als Mobile App.**

Ein Agent der alle deine Sportarten versteht, einen integrierten Trainingsplan erstellt, sich automatisch anpasst, und dich wie ein echter Coach begleitet. Die App ist eine generische Plattform. Der Agent ist die gesamte Intelligenz.

---

## 3. Das Fundamentale Designprinzip

### Die App ist die Buehne. Der Agent ist der Kuenstler.

**Nichts — wirklich NICHTS — ist in der App oder im Backend hardcodiert:**

| Was NICHT hardcodiert sein darf | Stattdessen |
|---|---|
| Sportarten | Agent erkennt aus Freitext/Sprache |
| Trainingsregeln (z.B. "kein hartes Laufen nach Beintag") | Agent reasoning |
| Berechnungsformeln (TRIMP, HR-Zonen, FTP, etc.) | Agent entdeckt + definiert Formeln, System rechnet |
| Plan-Bewertungskriterien | Agent definiert was einen guten Plan ausmacht |
| Produktempfehlungen (Affiliate) | Agent recherchiert + empfiehlt pro Session |
| Periodisierungs-Schemata | Agent waehlt passendes Schema pro Ziel |
| Session-Strukturen (Warmup→Work→Cooldown) | Agent definiert Struktur pro Sport |
| Ziel-Kategorien (race_target, routine, etc.) | Agent leitet Zieltyp ab |
| Proactive Trigger Bedingungen | Agent definiert eigene Trigger-Regeln |
| Onboarding-Fragen | Agent fuehrt dynamisches Gespraech |
| Welche Health-Daten relevant sind | Agent entscheidet basierend auf Sportart + Ziel |

**Die App stellt nur bereit:**
- Generische UI-Komponenten (Kalender, Cards, Chat, Listen)
- Daten-Sync (Garmin, Apple Health, Health Connect)
- Generic Calculation Engine (fuehrt Agent-definierte Formeln aus)
- Push Notifications, Voice Input, Auth

**Metapher**: Die App ist ein leeres Notizbuch mit Taschenrechner. Der Agent ist der Sportwissenschaftler der es fuellt, benutzt und anpasst.

### Agent Configuration Store

Technisches Herzstuck: Der Agent speichert ALLE seine Entscheidungen als strukturierte Konfigurationen in der DB. Die App und die Calc Engine lesen diese Configs und fuehren aus.

```
Agent Configuration Store (Supabase Tabellen):

metric_definitions
├─ id, user_id, sport_context, metric_name
├─ formula_definition (JSON: variables, expression, output_unit)
├─ explanation (warum diese Formel)
├─ source (Agent-Reasoning, Research, etc.)
├─ created_at, updated_at
└─ Beispiel: { sport: "running", metric: "training_load",
│    formula: "duration_min * delta_hr_ratio * 0.64 * exp(1.92 * delta_hr_ratio)",
│    explanation: "Banister TRIMP — bewaehrte Methode fuer Ausdauer-Load-Berechnung" }

eval_criteria
├─ id, user_id, criteria_name, weight, description
├─ evaluation_prompt (wie der Agent dieses Kriterium bewertet)
└─ Beispiel: { name: "cross_sport_balance", weight: 0.25,
│    description: "Keine schweren Beinbelastungen an aufeinanderfolgenden Tagen" }

session_schemas
├─ id, user_id, sport, schema_definition (JSON: erlaubte Phasen, Struktur)
└─ Beispiel: { sport: "bouldering", phases: ["warmup", "technique", "projects", "cooldown"] }

periodization_models
├─ id, user_id, goal_context, model_definition (JSON: Phasen, Dauer, Fokus)
└─ Beispiel: { goal: "half_marathon", phases: [
│    { name: "base", weeks: 6, focus: "aerobic_volume" },
│    { name: "build", weeks: 4, focus: "threshold_work" }, ...] }

proactive_trigger_rules
├─ id, user_id, trigger_name, condition (JSON), action, cooldown_hours
└─ Beispiel: { name: "missed_session", condition: { missed_count: ">= 2", window: "7d" },
│    action: "suggest_replan" }

product_recommendations
├─ id, user_id, session_id, product_name, product_url, reason, affiliate_tag
└─ Generiert vom Agent als Teil der Plangeneration
```

### Self-Improvement Zyklus

1. Agent ENTDECKT (Reasoning + ggf. Web-Research) was fuer einen Sport relevant ist
2. Agent DEFINIERT strukturierte Konfiguration und speichert sie
3. System RECHNET / BEWERTET / EMPFIEHLT basierend auf den Configs
4. Agent PRUEFT ob die Ergebnisse sinnvoll sind (z.B. "Load-Werte korrelieren mit gefuehlter Ermuedung?")
5. Agent PASST AN wenn noetig (neue Formel, andere Gewichtung, bessere Trigger)
6. Agent ERKLAERT dem User auf Nachfrage warum er so rechnet/bewertet/empfiehlt

---

## 4. Weitere Designprinzipien

### Chat = Single Source of Truth
- Der Chat ist das **Eingangstor zu ALLEM**
- "Was steht morgen an?" → Coach antwortet direkt mit dem Training (sagt NICHT "schau auf die Plan-Seite")
- Daten, Plan, Fragen, Anpassungen, Statistiken — alles ueber den Chat erreichbar
- **Wie Claude Code**: Frag irgendwas → bekomm eine echte, vollstaendige Antwort
- Gleiche Architektur wie MoltBot/ClawBot — Agent mit Tools der auf alle Daten zugreifen kann
- "Warum berechnest du meinen Load so?" → Agent erklaert seine eigene Formel

### Der Coach als Vertrauensperson
- Baut eine echte Vertrauensbasis zum User auf
- Kennt den User: Sportarten, Ziele, Leistungslevel, Vorlieben, Schwaechen
- Kommuniziert wie ein Freund/Coach, nicht wie eine App
- Push-Nachrichten fuehlen sich an wie WhatsApp-Nachrichten vom Trainer
- Diese Vertrauensbasis ist die Grundlage fuer Monetarisierung (Affiliate)

---

## 5. Zielgruppe

**Ambitionierte Hobby-Athleten**:
- Machen mehrere Sportarten parallel
- Wollen Fortschritte machen und Ziele erreichen
- Koennen/wollen sich keinen echten Coach leisten
- Betreiben Sport amateurmaessig aber mit Ehrgeiz
- Brauchen einen intelligenten Plan der alles koordiniert
- Keine Profi-Athleten, keine Anfaenger — die Mitte

---

## 6. Monetarisierung

### Saeule 1: Subscriptions
- Gestaffeltes Abo-Modell (Tiers TBD — nach MVP Validierung)

### Saeule 2: Kontextuelle Affiliate-Empfehlungen (Agent-Driven)
- **Amazon Affiliate Produkte** direkt bei passenden Trainingseinheiten
- **Komplett Agent-gesteuert** — kein Mapping, keine Zuordnungstabelle im Code
- Teil der Plangeneration: Agent bekommt Sessions, recherchiert passende Produkte, speichert Empfehlungen in `product_recommendations`
- Nutzt Kontext: Session-Details, Sport, Intensitaet, User-Profil, Ziele
- **Niedrige Kaufhemmschwelle**: Produkte die ein Normalo-Athlet sowieso braucht
- Empfehlungen erscheinen in der App unterhalb der Session-Details in der Tagesansicht
- Agent kann fuer JEDEN Sport Empfehlungen ableiten — auch fuer Sportarten die nie jemand vorhergesehen hat

---

## 7. Core Features (Detail)

### 7.1 Voice-First Onboarding (Dynamischer Companion)
- **Kein fester Step-Flow** — der Agent fuehrt ein dynamisches Gespraech
- Das aktuelle 9-Schritt Onboarding (`app/(onboarding)/`) wird durch einen Chat-basierten Companion ersetzt
- **Spracheingabe als primaerer Input**: Native iOS Dictation / Android Speech API (kostenlos, offline)
- Tippen geht auch
- Mikrofon-Button im Chat
- Der Agent entscheidet welche Fragen er stellt basierend auf dem was der User sagt
- Agent erkennt Sportarten aus Freitext ("Ich mache Laufen und Rennrad"), fragt nach Zielen
- Agent entscheidet wann genug Info fuer den ersten Plan vorhanden ist
- Alles was im Onboarding gesagt wird, ist spaeter im Chat abrufbar (gespeichert als Chat-History + Beliefs)
- Technisch: Gleicher `POST /chat` Endpoint wie der Coach-Chat. Der Onboarding-Status wird im Request-Body als `{ "context": "onboarding" }` mitgeschickt. Der Agent erhaelt diesen Kontext als Teil der User-Message und weiss dadurch, dass er im Onboarding-Modus ist und gezielt Profildaten sammeln soll

### 7.2 Multi-Sport Trainingsplan (Generisch)
- Wochenplan der ALLE Sportarten des Users koordiniert
- Agent entscheidet: Cross-Sport Ermuedung, Reihenfolge, Intensitaetsverteilung
- Agent leitet ab: Zieltypen, Prioritaeten, Session-Strukturen
- Agent definiert: Bewertungskriterien fuer Planqualitaet (in `eval_criteria`)
- Auto-generiert + User reviewed/akzeptiert
- Plan-Schema ist generisch (JSON): Sport, Typ, Dauer, Schritte — aber WELCHE Schritte haengt vom Agent ab
- Darstellung in der App: WeekCalendarStrip (oben) + DayDetailCard (unten) — schon gebaut

### 7.3 Chat als Universal-Interface
- Eigener Tab in der App (schon gebaut: `app/(tabs)/coach.tsx`)
- Agent hat Tools mit Zugriff auf: Plan, Aktivitaeten, Profil, Health-Daten, Agent Configs
- Streaming Responses via SSE (schon implementiert in App mit `react-native-sse`)
- Kontextuelle Quick-Actions bei Sessions: "Warum dieses Training?", "Zu hart", "Verschieben"
- User kann alles fragen: Plan, Daten, Metriken, Erklaerungen zu Formeln

### 7.4 Unbekannte Aktivitaeten → Coach fragt nach
- Aktivitaet von Garmin/Apple Health gesynct aber Sport-Typ unklar
- → Push Notification: "Hey, ich habe ein Training erkannt"
- → Im Chat erscheint die Frage (wie WhatsApp vom Trainer)
- → User antwortet: "Das war Padel"
- → Agent klassifiziert die Aktivitaet
- → Wenn Sport NEU: Agent recherchiert Metriken, definiert Formeln, definiert Session-Schemas
- → Plan wird ggf. angepasst

### 7.5 Multi-Provider Health Data (Generisch)
- **Garmin**: Alle verfuegbaren Daten (Aktivitaeten, Schlaf, Stress, Body Battery, HRV, Schritte)
- **Apple Health**: ALLES was verfuegbar ist — generisch alle Datentypen einlesen
- **Android Health Connect**: Equivalent zu Apple Health
- App inventarisiert generisch welche Datentypen verfuegbar sind und synct ALLE
- Detaillierte Aktivitaetsdaten: HR-Verlauf, Pace, Power, Kadenz, Zonen, Splits
- Agent entscheidet welche Health-Daten fuer die Planung relevant sind (nicht hardcodiert)
- Bereits teilweise implementiert in athletly-app (Garmin Service, HealthKit, Health Connect)

### 7.6 Adaptive Replanning (Human-in-the-Loop)
- Agent erkennt Abweichungen vom Plan (eigene Trigger-Regeln in `proactive_trigger_rules`)
- Schlaegt Anpassung vor → Push Notification → Chat-Nachricht
- User bestaetigt oder lehnt ab in der App (Checkpoint-System schon gebaut)
- Langfristiges Lernen ueber Episodic Memory
- Agent verbessert seine Trigger-Regeln basierend auf User-Feedback

---

## 8. Architecture (Detail)

### 8.1 Uebersicht

```
┌─────────────────────────────────────────┐
│      React Native App (athletly-app)    │
│                                         │
│  GENERISCHE PLATTFORM:                  │
│  • Kalender/Plan Darstellung            │
│  • Chat Interface (SSE Streaming)       │
│  • Session Cards + Product Cards        │
│  • Health Sync (Garmin/Apple/Android)   │
│  • Voice Input (Native STT)            │
│  • Push Notifications                   │
│  (Keine Sport-Logik, keine Regeln,      │
│   keine Formeln)                        │
└────────┬──────────────┬─────────────────┘
         │              │
    Supabase Direct     │  Python FastAPI (Hetzner/Docker)
    ─────────────       │  ──────────────────────────────
    • Auth (JWT)        │  DIE INTELLIGENZ:
    • Realtime          │  • Agent Loop
    • Storage           │  • Generic Calculation Engine
    • Simple CRUD       │  • Plan Generation + Evaluation
    • Push webhooks     │  • Training Assessment
                        │  • Episodic Memory + Reflection
                        │  • Trajectory Assessment
                        │  • Proactive Engine
                        │  • LiteLLM (any model)
                        │  • Agent Config Management Tools
         │              │
         +----- DB -----+
                |
        Supabase PostgreSQL (RLS)
        ┌────────────────────────────────┐
        │ USER DATA:                     │
        │  users, activities,            │
        │  health_data, plans,           │
        │  chat_messages, episodes       │
        │                                │
        │ AGENT CONFIGURATION STORE:     │
        │  metric_definitions,           │
        │  eval_criteria,                │
        │  session_schemas,              │
        │  periodization_models,         │
        │  proactive_trigger_rules,      │
        │  product_recommendations       │
        └────────────────────────────────┘
```

### 8.2 Warum Hybrid (Supabase + Python)?

| Supabase liefert | Python liefert |
|---|---|
| Auth (JWT, OAuth, Social Login) | Multi-Round Agent Loop (25 Rounds, 30-60s+) |
| PostgreSQL mit RLS | Plan Evaluator-Optimizer (iterativ) |
| Realtime Subscriptions | Episodic Memory + Reflections |
| Push via Webhooks | Generic Calculation Engine |
| File Storage | LiteLLM (beliebiges Model) |
| Managed, zero ops | Keine Timeout-Limits |

**Supabase Edge Functions (aktueller TS Agent) wird ERSETZT durch Python FastAPI.**

### 8.3 API Design — Ein Endpoint, Ein Agent (ClawdBot/NanoBot-Prinzip)

**Fundamentale Regel**: Es gibt EINEN Chat-Endpoint. Der Agent entscheidet via Tools was zu tun ist. Keine dedizierten REST-Endpoints fuer spezifische Aktionen. Der Chat IST die API.

```
POST /chat              → SSE Stream (DER universelle Endpoint — alles geht durch den Agent)
POST /chat/confirm      → Checkpoint-Bestaetigung (Human-in-the-Loop Response, wird dem Agent als User-Message injiziert, z.B. "[CHECKPOINT] User hat bestaetigt: Plan KW 11 akzeptiert")
POST /webhook/activity  → Supabase Webhook bei neuer Aktivitaet (triggert Agent proaktiv)

Auth: Supabase JWT Verification bei jedem Request
```

**Warum nur EIN Endpoint?**

| Alte Vision (FALSCH) | NanoBot/ClawdBot-Prinzip (RICHTIG) |
|---|---|
| `POST /plan/generate` | User sagt im Chat "Erstell mir einen Plan" → Agent nutzt `create_plan` Tool |
| `POST /plan/adjust` | User sagt "Der Dienstag ist zu hart" → Agent nutzt `adjust_plan` Tool |
| `POST /activity/classify` | Agent fragt proaktiv "Was war das fuer ein Training?" → User antwortet im Chat |
| `GET /plan/current` | App liest direkt aus Supabase (`weekly_plans` Tabelle) — kein Agent noetig |
| `GET /metrics/{user_id}` | App liest direkt aus Supabase — Agent hat die Werte bereits berechnet und gespeichert |
| `GET /config/{user_id}` | App liest direkt aus Supabase (`metric_definitions` etc.) |

**Direkte Supabase-Reads (kein Agent noetig):**
- Plan-Darstellung im Kalender → App liest `weekly_plans` direkt aus Supabase
- Metriken-Anzeige → App liest berechnete Werte direkt aus Supabase
- Agent-Configs fuer UI → App liest `metric_definitions`, `session_schemas` direkt aus Supabase

**Alles andere geht durch den Chat:**
- Plan generieren, anpassen, bewerten
- Aktivitaeten klassifizieren
- Metriken erklaeren
- Profil aktualisieren
- Configs definieren/aendern

### 8.4 Generic Calculation Engine (`evalidate`)

Die Calc Engine fuehrt Agent-definierte Formeln sicher aus. Entscheidung: **`evalidate`** — Whitelist-basierter AST-Parser, schnellste sichere Option (0.33s/1M ops), aktiv maintained (Maerz 2026).

- **Input**: Formel-Definition (aus `metric_definitions`) + Rohdaten (aus `activities`/`health_data`)
- **Output**: Berechneter Wert
- **KEIN eval()** — `evalidate` blockiert ALLES was nicht explizit erlaubt ist (Whitelist-by-default)
- **Unterstuetzte Operationen**: Arithmetik, exp(), log(), min(), max(), avg(), Aggregationen ueber Zeitreihen
- **Bulk-Berechnung**: Kompiliert Formel einmal, evaluiert schnell ueber hunderte Aktivitaeten

**Setup:**
```python
from evalidate import Expr, base_eval_model
import math

calc_model = base_eval_model.clone()
calc_model.nodes.append('Call')
calc_model.imported_functions.update({
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "min": min, "max": max, "pow": pow,
    "avg": lambda lst: sum(lst) / len(lst),
    "sum": sum, "count": len,
})

# Agent definiert Formel, Engine rechnet sicher
expr = Expr("duration_min * delta_hr_ratio * 0.64 * exp(1.92 * delta_hr_ratio)", model=calc_model)
result = expr.eval({"duration_min": 45.0, "delta_hr_ratio": 1.1})
```

**Warum `evalidate` und nicht andere?**

| Library | Sicherheit | Performance (1M ops) | Status |
|---|---|---|---|
| **evalidate** | **Whitelist-AST** | **0.33s** | Maerz 2026 |
| simpleeval | Blacklist-AST | 1.82s | Nov 2024 |
| asteval | Blacklist (CVE-2025!) | 26.1s | Dez 2025 |
| numexpr | UNSICHER (nutzt eval()) | <0.1s | CVE-2023 |
| py_expression_eval | Custom Parser | unbekannt | Unmaintained seit 2021 |

**Provider-Metriken als Ergaenzung:** Garmin liefert Training Effect, VO2max, Body Battery, HRV. Apple Health liefert Cardio Fitness, Walking HR Average, HRV. Der Agent muss nicht alles selbst berechnen — er nutzt vorhandene Provider-Werte und definiert nur ergaenzende Formeln (z.B. TRIMP wo Garmin es nicht liefert). Self-Improvement: Agent vergleicht eigene Werte mit Provider-Werten als natuerlicher Feedback-Loop.

### 8.5 Data Flows

1. **Health Sync**: Garmin/Apple Health/Health Connect → App → Supabase (generisch, alle Datentypen)
2. **Neue Sportart**: User sagt "Ich mache jetzt Klettern" → Agent recherchiert → definiert Metriken, Formeln, Schemas → speichert in Agent Config Store
3. **Plan Generation**: Auto-Trigger oder Request → FastAPI → Agent liest Configs + Daten → generiert Plan → Plan in DB → Realtime → App zeigt Plan. *(Produktempfehlungen als Teil der Plangeneration erst ab Phase 6)*
4. **Coach Chat**: User tippt/spricht → Native STT → Text → FastAPI SSE → Agent mit Tools → Streaming Response → App
5. **Unbekannte Aktivitaet**: Sync → Sport-Typ unklar → Supabase Trigger → Push + Chat-Nachricht → User antwortet → Agent klassifiziert + ggf. neue Configs
6. **Replan**: Agent prueft eigene Trigger-Regeln → Abweichung erkannt → Vorschlag generiert → Push → User oeffnet App → bestaetigt/lehnt ab
7. **Self-Improvement**: Agent prueft periodisch ob Formeln/Kriterien sinnvolle Ergebnisse liefern → passt Configs an

### 8.6 App ↔ Backend Vertrag: Plan-Datenformat

**Zusaetzliche Tabellen (neu, muessen in Phase 1 erstellt werden):**

```sql
-- Chat Session Management
CREATE TABLE public.chat_sessions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id),
  context     TEXT DEFAULT 'coach',  -- 'coach' | 'onboarding'
  started_at  TIMESTAMPTZ DEFAULT NOW(),
  ended_at    TIMESTAMPTZ,
  summary     TEXT,                  -- 3-Satz Auto-Summary bei Session-Ende
  tags        TEXT[] DEFAULT '{}',   -- Keyword-Tags fuer Cross-Session-Search
  turn_count  INT DEFAULT 0
);

-- Berechnete Metriken (Agent schreibt via Calc Engine, App liest direkt)
CREATE TABLE public.calculated_metrics (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id),
  activity_id UUID REFERENCES public.garmin_activities(id),
  metric_name TEXT NOT NULL,          -- z.B. "trimp", "training_effect"
  value       DOUBLE PRECISION NOT NULL,
  unit        TEXT,                   -- z.B. "points", "min/km"
  formula_id  UUID REFERENCES public.metric_definitions(id),  -- welche Formel wurde benutzt
  source      TEXT DEFAULT 'agent',  -- 'agent' | 'garmin' | 'apple_health'
  calculated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(activity_id, metric_name, source)
);

-- Checkpoint / Pending Actions
CREATE TABLE public.pending_actions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id),
  session_id  UUID REFERENCES public.chat_sessions(id),
  action_type TEXT NOT NULL,         -- 'plan_confirm' | 'activity_classify' | 'replan'
  preview     JSONB,                 -- Was dem User angezeigt wird
  checkpoint_type TEXT DEFAULT 'HARD', -- 'SOFT' (auto-accept) | 'HARD' (muss bestaetigt)
  status      TEXT DEFAULT 'pending', -- 'pending' | 'confirmed' | 'rejected' | 'expired'
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  resolved_at TIMESTAMPTZ,
  auto_expire_at TIMESTAMPTZ         -- Fuer SOFT Checkpoints
);

-- Generische Health-Daten (Multi-Provider)
CREATE TABLE public.health_data (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id),
  provider    TEXT NOT NULL,         -- 'garmin' | 'apple_health' | 'health_connect'
  data_type   TEXT NOT NULL,         -- 'sleep' | 'hrv' | 'stress' | 'body_battery' | 'steps' | 'resting_hr'
  value       JSONB NOT NULL,        -- Generisches JSON — Schema variiert pro data_type
  recorded_at TIMESTAMPTZ NOT NULL,  -- Wann die Messung stattfand
  synced_at   TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, provider, data_type, recorded_at)
);

-- Push Notification History
CREATE TABLE public.push_notifications (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id),
  title       TEXT,
  body        TEXT NOT NULL,
  data        JSONB DEFAULT '{}',    -- Deep-Link Info, Action Type etc.
  trigger     TEXT NOT NULL,         -- 'heartbeat' | 'activity_sync' | 'agent_message'
  status      TEXT DEFAULT 'sent',   -- 'sent' | 'delivered' | 'failed'
  sent_at     TIMESTAMPTZ DEFAULT NOW(),
  expo_receipt_id TEXT               -- Expo Push Receipt ID fuer Delivery Tracking
);

-- Monatliche Episodic Consolidations
CREATE TABLE public.episode_consolidations (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES public.users(id),
  month       TEXT NOT NULL,         -- '2026-03'
  persistent_patterns TEXT[] DEFAULT '{}',
  anomalies   TEXT[] DEFAULT '{}',
  fitness_trajectory JSONB,
  coaching_lessons TEXT[] DEFAULT '{}',
  source_episode_ids UUID[] DEFAULT '{}',
  confidence  DOUBLE PRECISION DEFAULT 0.8,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, month)
);
```

**KRITISCH**: Die App-Komponenten (WeekCalendarStrip, DayDetailCard) erwarten ein bestimmtes Datenformat. Das Python Backend MUSS Plaene in diesem Format in die `weekly_plans` Tabelle schreiben.

**Supabase Tabelle `weekly_plans`:**
```sql
CREATE TABLE public.weekly_plans (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID NOT NULL REFERENCES public.users(id),
  week_start DATE NOT NULL,        -- Montag der Woche
  days       JSONB NOT NULL,       -- Array von Tagesobjekten (Schema unten)
  status     TEXT DEFAULT 'active', -- 'active' | 'completed' | 'replaced'
  created_by TEXT DEFAULT 'ai',    -- 'ai' | 'user'
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, week_start)
);
```

**`days` JSONB Schema (was der Agent produzieren muss):**
```json
[
  {
    "date": "2026-03-09",
    "day_name": "Montag",
    "sessions": [
      {
        "sport": "running",
        "duration_minutes": 45,
        "intensity": "high",
        "session_type": "intervals",
        "description": "5x1km Intervalle mit 400m Trabpause...",
        "details": {
          "pace_range": { "min": "5:30", "max": "4:30" },
          "hr_zone": "zone4",
          "intervals": {
            "sets": 5,
            "work_duration": 1000,
            "work_unit": "meters",
            "rest_duration": 400,
            "rest_unit": "meters",
            "target_pace": "4:30"
          },
          "distance_km": 8.2
        }
      }
    ]
  },
  {
    "date": "2026-03-10",
    "day_name": "Dienstag",
    "sessions": []
  }
]
```

**Wichtige Regeln:**
- `sessions: []` (leeres Array) → App zeigt RestDay
- Aktuell wird nur `sessions[0]` angezeigt (App ignoriert weitere Sessions pro Tag — TODO: Multi-Session pro Tag ermoeglichen)
- `session_type` IMMER explizit senden (Fallback-Logik in der App ist fragil)
- `intensity` muss genau `'low' | 'moderate' | 'high'` sein
- `sport` ist ein freier String — die App mapped ihn auf Icons/Farben (unbekannte Sports bekommen Default-Icon)
- `coachMessage` und `reasoning` fehlen aktuell im DB-Schema — werden in der App hardcodiert. **Muss als Top-Level Columns oder in JSONB ergaenzt werden**
- Die App queried: `.eq('status', 'active').order('created_at', { ascending: false }).limit(1).single()`

**Gym-spezifische Details:**
```json
{
  "sport": "gym",
  "session_type": "lower_body",
  "details": {
    "session_type": "lower_body",
    "target_rpe": 7
  }
}
```

### 8.7 App ↔ Backend Vertrag: Chat-SSE-Protokoll (NanoBot `_bus_progress` Pattern)

Die App nutzt `react-native-sse` um SSE Events vom Backend zu empfangen. Das Protokoll folgt dem **NanoBot `_bus_progress` Pattern**: Kein Token-Level-Streaming. Stattdessen feuert der Agent bei jedem Tool-Call ein Progress-Event, damit der User in Echtzeit sieht was der Coach tut.

**Request:**
```
POST /chat
Headers: Authorization: Bearer <supabase_jwt>
Body: { "message": "<user text>", "session_id": "<uuid>" }
Response: SSE Stream
```

**SSE Event-Typen (event name → data shape):**

| Event | Data | Zweck |
|---|---|---|
| `start` | `{ session_id: string }` | Stream geoeffnet |
| `thinking` | `{ content: string }` | Agent-Reasoning (Thinking-Content vor Tool-Call, `<think>` Tags gestrippt) |
| `tool_hint` | `{ tool: string, hint: string }` | Was der Agent gerade tut: `{ tool: "get_activities", hint: "Lade deine Laufhistorie der letzten 30 Tage..." }` |
| `message` | `{ content: string, checkpoint?: { id, type, preview }, session_id }` | Finale Antwort |
| `usage` | `{ provider, model, input_tokens, output_tokens, cost_usd, latency_ms }` | Kosten-Tracking |
| `error` | `{ message: string }` | Fehler |
| `done` | (leer) | Stream beendet, Connection schliessen |

**App-seitige Darstellung der Progress Events:**
- `thinking` → Transiente "Coach denkt nach..." Anzeige (verschwindet bei naechstem Event)
- `tool_hint` → Status-Chip unter dem Chat: "📊 Analysiere Herzfrequenz-Daten...", "📅 Erstelle Wochenplan..."
- `message` → Finale Chat-Bubble vom Coach

**Beispiel-Flow eines Agent-Turns:**
```
event: start        → { session_id: "abc-123" }
event: thinking     → { content: "User will Plan fuer naechste Woche. Ich schaue mir zuerst die letzten Aktivitaeten an..." }
event: tool_hint    → { tool: "get_activities", hint: "Lade Trainingshistorie..." }
event: tool_hint    → { tool: "get_agent_config", hint: "Pruefe Metrik-Definitionen..." }
event: thinking     → { content: "Basierend auf 3 Laeufen und 2 Gym-Sessions letzte Woche sollte ich die Belastung reduzieren..." }
event: tool_hint    → { tool: "create_plan", hint: "Erstelle Wochenplan..." }
event: message      → { content: "Hier ist dein Plan fuer naechste Woche! ...", checkpoint: { id: "cp-1", type: "HARD", preview: "Wochenplan KW 11" } }
event: usage        → { provider: "gemini", model: "gemini-2.5-flash", input_tokens: 8500, output_tokens: 1200, cost_usd: 0.003 }
event: done
```

**Checkpoint-System (Human-in-the-Loop):**
- Agent will eine Aktion ausfuehren (z.B. Plan erstellen) → sendet `message` mit `checkpoint`
- `checkpoint.type`: `'SOFT'` (Auto-Accept nach Timeout) oder `'HARD'` (muss bestaetigt werden)
- App zeigt Confirm/Reject UI
- User bestaetigt → App ruft `POST /chat/confirm` auf → wird dem Agent als User-Message injiziert, z.B. "[CHECKPOINT] User hat bestaetigt: Plan KW 11 akzeptiert"
- Supabase Tabelle `pending_actions` speichert den Checkpoint-Status

### 8.8 Agent Tool Inventory

Der Agent braucht diese Tools um alle Features zu ermoeglichen:

**Daten-Zugriff (Read):**
- `get_user_profile` — Profil, Sportarten, Ziele, Beliefs
- `get_activities` — Aktivitaeten mit Filtern (Sport, Zeitraum, Limit)
- `get_health_data` — Sleep, HRV, Stress, Body Battery etc.
- `get_current_plan` — Aktuellen Wochenplan
- `get_plan_history` — Vergangene Plaene
- `get_episodes` — Episodic Memory (Reflections)
- `get_agent_config` — Eigene Konfigurationen (Formeln, Kriterien, etc.)

**Aktionen (Write):**
- `create_plan` — Wochenplan generieren und in DB schreiben
- `adjust_plan` — Bestehenden Plan anpassen
- `update_profile` — User-Profil aktualisieren (Beliefs, Ziele)
- `classify_activity` — Unbekannte Aktivitaet klassifizieren

**Agent Config Management:**
- `define_metric` — Neue Metrik/Formel definieren und speichern
- `define_eval_criteria` — Bewertungskriterien definieren
- `define_session_schema` — Session-Struktur fuer einen Sport definieren
- `define_periodization` — Periodisierungs-Modell definieren
- `define_trigger_rule` — Proaktive Trigger-Regel definieren
- `update_config` — Bestehende Config anpassen (Self-Improvement)

**Empfehlungen:**
- `recommend_products` — Affiliate-Produkte fuer eine Session recherchieren

**Berechnung:**
- `calculate_metric` — Metrik berechnen via Generic Calc Engine (nutzt Agent-definierte Formel)
- `calculate_bulk_metrics` — Metriken ueber Zeitraum aggregieren

**Kommunikation (NanoBot MessageTool-Pattern):**
- `send_notification` — Push-Nachricht an den User senden (mid-turn, ohne auf finale Antwort zu warten)
  - Parameter: `{ content: string, title?: string, deep_link?: string }`
  - Nutzt Expo Push Notifications (`exp.host` API — kein SDK noetig, kostenlos, unbegrenzt)
  - Turn-Level Deduplication: Wenn `send_notification` gefeuert hat, wird die automatische Finale-Antwort unterdrueckt (NanoBot `_sent_in_turn` Pattern)
  - Push Token wird bei App-Start registriert und in `push_tokens` Tabelle gespeichert
  - Beispiel: HeartbeatService erkennt 3 verpasste Sessions → Agent ruft `send_notification({ content: "Hey, ich hab gemerkt du hast 3 Trainings ausgelassen. Sollen wir den Plan anpassen?", title: "Coach Update" })` auf

**Hintergrund-Aufgaben (NanoBot SpawnTool-Pattern):**
- `spawn_background_task` — Lange Aufgabe an einen Hintergrund-Subagent delegieren
  - Parameter: `{ task: string, label?: string }`
  - Gibt sofort eine Task-ID zurueck — Agent kann dem User antworten ohne zu blockieren
  - Subagent laeuft asynchron als `asyncio.Task` mit eigenem Agent-Loop (max 15 Iterationen, nur Read/Write/Calc Tools, kein MessageTool/SpawnTool)
  - Ergebnis wird als `InboundMessage(channel="system", sender_id="subagent")` zurueck an den Haupt-Agent gemeldet
  - Haupt-Agent fasst Ergebnis natuerlichsprachlich zusammen und sendet es per `send_notification` an den User
  - Cleanup: `done_callback` auf dem `asyncio.Task` raeumt auf, `/stop` cancelt alle laufenden Subagents
  - **Use Cases**: Plan-Generation (30-60s), Multi-Wochen-Analyse, Bulk-Metrik-Berechnung
  - Beispiel: User sagt "Erstell mir einen Plan" → Agent: "Ich arbeite daran, dauert ca. 30 Sekunden!" → spawnt Background-Task → User kann weiter chatten → Push wenn fertig

**Fehlerbehandlung bei Tools (NanoBot-Pattern):**
- Wenn ein Tool eine Exception wirft, wird die Exception NICHT als Raw-Stack-Trace zurueckgegeben
- Stattdessen: natuerlichsprachliche Fehlermeldung + Instruktion "Analysiere den Fehler und versuche einen anderen Ansatz."
- Dies ermoeglicht LLM Self-Correction statt Endlosschleifen bei wiederholten Tool-Fehlern

**Memory (NanoBot Consolidation-Pattern):**
- `search_session_history` — Vergangene Chat-Sessions durchsuchen (Keyword-Suche ueber Session-Summaries)
  - Jede beendete Session wird automatisch als 3-Satz-Summary + Tags in einem Session-Index gespeichert (~150 Tokens pro Session)
  - Ermoeglicht Cross-Session-Retrieval: "Hat der Athlet jemals ueber Knieprobleme gesprochen?"

### 8.9 Cold Start / Bootstrap Strategie

**Problem**: Bei Tabula rasa hat der Agent beim ersten User KEINE Konfigurationen. Aber der User erwartet schnell einen Plan.

**Loesung: Progressive Configuration**

1. **Onboarding-Chat** (2-3 Minuten): Agent lernt Sportarten + Ziele
2. **Sofort nach Onboarding**: Agent definiert Basis-Configs fuer die genannten Sportarten
   - Fuer bekannte Sportarten (Laufen, Radfahren, etc.) weiss das LLM die gaengigen Formeln
   - Agent ruft `define_metric`, `define_session_schema` etc. auf
   - Dauert ca. 1 Agent-Loop Durchgang (~10-30 Sekunden)
3. **Erster Plan**: Wird generiert mit den frisch definierten Configs
4. **Progressive Verfeinerung**: Nach den ersten echten Aktivitaeten verfeinert der Agent seine Configs

**Warum das funktioniert**: Das LLM (Gemini/Claude/GPT) KENNT bereits Sportformeln wie TRIMP, HR-Zonen etc. aus seinem Training. Es muss sie nicht "entdecken" im Sinne von Recherche — es muss sie nur als strukturierte Configs in die DB schreiben. Das geht schnell.

**Fuer exotische Sportarten** (Klettern, Reiten, etc.): Agent nutzt sein Wissen + ggf. Reasoning um sinnvolle Basis-Metriken abzuleiten. Verfeinert spaeter wenn echte Daten kommen.

### 8.10 Proaktive Intelligenz — HeartbeatService (NanoBot-Pattern)

**Alte Vision (FALSCH)**: 4 separate Background Workers (Auto-Plan Generator, Activity Processor, Proactive Checker, Health Data Processor) als eigene Infrastruktur.

**NanoBot/ClawdBot-Prinzip (RICHTIG)**: Der SELBE Agent wacht periodisch auf, prueft was zu tun ist, und handelt. Keine separate Worker-Infrastruktur. Ein Agent, viele Aufgaben.

**HeartbeatService:**
```python
# Wacht alle 30 Minuten auf (konfigurierbar)
# Der Agent bekommt eine System-Message: "Heartbeat: pruefe ob Aktionen noetig sind"
# Der Agent entscheidet SELBST via Reasoning + Tools was zu tun ist:
#   - Ist ein neuer Wochenplan faellig? → create_plan Tool
#   - Gibt es unverarbeitete Aktivitaeten? → calculate_metric Tool
#   - Sind Trigger-Regeln verletzt? → Push Notification via message Tool
#   - Neue Health-Daten da? → Kontext aktualisieren
# Wenn nichts zu tun ist → Agent bleibt still (keine Spam-Nachrichten)
```

| Aufgabe | Trigger | Wie der Agent es handhabt |
|---|---|---|
| **Neuer Wochenplan** | Heartbeat erkennt: kein aktiver Plan fuer aktuelle Woche | Agent nutzt `create_plan` Tool → Push an User via `message` Tool |
| **Neue Aktivitaet** | Supabase Webhook → `/webhook/activity` → Agent-Message | Agent nutzt `calculate_metric` + `get_agent_config` Tools → prueft Planabweichung |
| **Proaktive Checks** | Heartbeat (z.B. taeglich 18:00) | Agent liest eigene `proactive_trigger_rules` → evaluiert → sendet Push wenn noetig |
| **Health-Daten** | Heartbeat erkennt neue Daten seit letztem Check | Agent entscheidet welche Daten planungsrelevant sind |

**Implementierung**: `asyncio`-basierter HeartbeatService im FastAPI-Prozess. Kein Celery, kein APScheduler. Der Agent-Loop ist der einzige Worker.

**Entscheidungsmechanismus (NanoBot v0.1.4 Pattern):**
Der HeartbeatService nutzt ein virtuelles Tool `heartbeat_decision` mit strukturierter Antwort:
- `{ action: "skip" }` — nichts zu tun, Agent bleibt still
- `{ action: "run", summary: "Pruefe verpasste Sessions" }` — Agent fuehrt vollen Loop aus

Dies vermeidet fragiles Text-Parsing der Agent-Antwort. Der HeartbeatService prueft bei jedem 30-Min-Intervall ALLE Trigger-Typen inkl. zeitbasierter Regeln (z.B. "ist 18:00 seit letztem Daily Check vergangen?"). Kein separater Scheduler noetig.

**NanoBot MessageTool-Pattern fuer Push:**
- Agent kann **mid-turn** Nachrichten an den User senden (Push Notification)
- Turn-Level Deduplication: Wenn Agent bereits via MessageTool gesendet hat, wird die automatische Finale-Antwort unterdrueckt
- Heartbeat ist "truly silent" wenn nichts zu tun ist — keine Spam-Messages

---

### 8.11 Agentic Architecture Principles (ClawdBot/NanoBot-Alignment)

Dieses Projekt folgt explizit dem **ClawdBot/PicoClaw/MoltBot/NanoBot Architektur-Pattern**. Jede technische Entscheidung wird gegen diese Prinzipien geprueft:

**Prinzip 1: Ein Agent, Ein Endpoint**
- Es gibt EINEN Chat-Endpoint (`POST /chat`). Alles geht durch den Agent.
- Keine dedizierten REST-Endpoints fuer spezifische Aktionen.
- Der Agent entscheidet via Tools was zu tun ist. Der User kommuniziert ausschliesslich ueber Chat.
- Direkte DB-Reads (Plan im Kalender, Metriken-Anzeige) gehen an Supabase vorbei — dafuer braucht man keinen Agent.

**Prinzip 2: System Prompt Caching (NanoBot-Pattern)**
- Der System Prompt ist **vollstaendig statisch** — keine Runtime-Daten (Datum, User-ID, Sportarten) im System Prompt.
- Runtime-Kontext wird als **separate User-Message** am Anfang jedes Turns injiziert.
- Dadurch cached der LLM-Provider (Gemini, Anthropic) den System Prompt automatisch → weniger Latenz, weniger Kosten.
- **Beispiel**: System Prompt = "Du bist Athletly, ein AI Coach...". User-Message = `[CONTEXT] Date: 2026-03-03, User: Roman, Sports: running/cycling/gym, Active Plan: KW 10`

**Prinzip 3: Error Responses nie persistieren (NanoBot-Pattern)**
- Wenn ein Agent-Turn mit einem Fehler endet (finish_reason == "error", Tool-Exception, Timeout), wird die fehlerhafte Response **NICHT** in die Session History geschrieben.
- Verhindert Context-Poisoning: Ein fehlerhafter Turn wuerde sonst alle nachfolgenden Turns kontaminieren.
- Der User bekommt eine freundliche Fehlermeldung, aber die History bleibt sauber.

**Prinzip 4: HeartbeatService statt Background Workers**
- Keine separate Worker-Infrastruktur (kein Celery, kein APScheduler).
- Der SELBE Agent mit den SELBEN Tools wacht periodisch auf und prueft was zu tun ist.
- Wenn nichts zu tun ist → Agent bleibt still.

**Prinzip 5: Progress Events statt Token-Streaming**
- Kein Token-Level-Streaming (partial tokens an die UI).
- Stattdessen: Bei jedem Tool-Call feuert ein **Thinking + Tool Hint** Event.
- Der User sieht in Echtzeit was der Coach tut, bevor die finale Antwort kommt.

**Prinzip 6: Tool Autonomy**
- Der Agent sieht alle 15+ Tools und entscheidet autonom welche er aufruft.
- Kein Router, kein Workflow-Graph, keine vordefinierten Tool-Ketten.
- Gleicher Loop wie NanoBot/ClawdBot: `while not done → LLM(tools) → execute tool calls → repeat`
- **Safety Guard**: Maximal 25 Iterationen pro Agent-Turn. Bei Erreichen des Limits wird eine freundliche Fehlermeldung gesendet (nicht in History persistiert). Subagents (SpawnTool): max 15 Iterationen.

**Prinzip 7: Tabula Rasa — Agent definiert alles zur Laufzeit**
- Keine vordefinierten Sportarten, Formeln, Regeln, Bewertungskriterien.
- Der Agent entdeckt, definiert, speichert und verbessert alles selbst ueber den Agent Configuration Store.
- Token-Kosten fuer initiale Formel-Definition: ~500 Tokens bei Gemini Flash = $0.0002 pro User. Vernachlaessigbar.

### 8.12 Context & Memory Management

Langfristige Nutzung (Monate/Jahre) erfordert aktives Memory Management. Ohne dies waechst der Kontext unkontrolliert und die LLM-Kosten explodieren.

**A) Tool Result Truncation (Token-Budget pro Tool-Output)**

Grosse Tool-Outputs (z.B. `get_activities(days=90)` = 50 Aktivitaeten mit HR-Daten = 8.000-15.000 Tokens) muessen vor dem Einfuegen in den LLM-Kontext komprimiert werden.

| Tool | Rohes Output | Token-Budget | Kompression |
|---|---|---|---|
| `get_activities(days=90)` | 8.000-15.000 Tokens | 1.500 | 5-10x |
| `get_activities(days=7)` | 500-1.500 Tokens | keine Kompression | 1x |
| `analyze_training_load` | 2.000-4.000 Tokens | 800 | 2.5-5x |
| `web_search` | 3.000-6.000 Tokens | 1.200 | 2.5-5x |
| `create_training_plan` | 1.500-3.000 Tokens | 2.000 | keine Kompression |
| Default | — | 2.000 | — |

**Implementierung**: `execute_with_budget()` Wrapper im Tool Registry. Wenn das Token-Estimate (len/4) das Budget uebersteigt, wird eine LLM-basierte Zusammenfassung generiert. Fast-Path: Unter Budget → kein Extra-LLM-Call.

**B) Session-History Consolidation (Cross-Session Memory)**

Problem: Nach 100+ Chat-Sessions ueber Monate ist relevanter Kontext aus alten Sessions nicht auffindbar.

Loesung (NanoBot Two-File-Pattern):
1. **Session Summary Index**: Jede beendete Session wird automatisch als 3-Satz-Summary + Tags in `session_index.jsonl` gespeichert (~150 Tokens/Session)
2. **`search_session_history` Tool**: Agent kann per Keyword-Suche vergangene Sessions durchsuchen
3. **UserModel Beliefs**: Wichtige Fakten aus Sessions werden als Beliefs extrahiert und ueberleben ueber Sessions hinweg (bereits implementiert)

Token-Budget: 100 Sessions × 150 Tokens = 15KB Index-Datei. Suche ueber 5 Sessions = 750 Tokens. Vergleich: Eine volle Session laden = 2.000-8.000 Tokens.

**C) Episodic Memory Consolidation (Langzeit-Trainings-Patterns)**

Problem: Nach 50+ woeechentlichen Reflections (1 Jahr) waechst die Episodic Memory unkontrolliert.

Loesung (SimpleMem-inspiriert — 26.4% F1 Improvement bei 30x Token-Reduktion):
1. **Monatliche Konsolidierung**: Wenn >= 3 woeechentliche Episodes fuer einen Monat existieren, generiert ein LLM-Call eine monatliche Zusammenfassung
2. **Zwei-Tier Retrieval**: Erst monatliche Summaries (sehr kompakt, ~200 Tokens), dann bei Bedarf woeechentliche Details
3. **Persistente Patterns**: Muster die in 3+ Wochen auftreten werden in die monatliche Konsolidierung uebernommen und koennen als UserModel Beliefs promotet werden

Token-Budget: 3 Monats-Summaries × 200 Tokens = 600 Tokens + 4 aktuelle Wochen-Episodes × 200 Tokens = 800 Tokens. Total: ~1.400 Tokens fuer 6+ Monate Trainingshistorie.

**D) Agent Config Store — Versioning & Garbage Collection**

Problem: Agent Self-Improvement generiert ueber Monate 50+ Metric-Definitions, viele davon obsolet.

Loesung:
1. **Semantische Deduplikation**: Vor dem Einfuegen neuer Configs wird `cosine_similarity` gegen bestehende geprueft. Bei Similarity > 0.88 wird die alte Config superseded statt eine neue angelegt
2. **Bi-temporal Versioning**: Jede Config hat `valid_from` + `valid_until`. Alte Versionen werden archiviert, nicht geloescht
3. **Max Active Cap**: Wenn active_count > 60, triggert ein LLM-Konsolidierungs-Pass der semantisch aehnliche Configs merged
4. **GC bei Session-Start**: Archiviere Session-scoped Beliefs (> 1 Tag), Low-Confidence Beliefs (< 0.5, > 30 Tage)

**E) In-Session Context Compression (bereits implementiert)**

Bestehend in `agent_loop.py`: `_compress_history()` bei > 40 Messages. Letzte 4 Rounds bleiben vollstaendig, aeltere werden zusammengefasst.

Ergaenzung (Active Context Compression Pattern): System Prompt Hinweis "Nach 8+ Tool-Calls ohne User-Antwort: Fasse deine Erkenntnisse zusammen." Verhindert unkontrolliertes Tool-Call-Wachstum innerhalb einer Session.

---

### 8.13 Security & Infrastructure

**A) Supabase JWT Verification**
- Library: `PyJWT>=2.8` + `cryptography>=41` (NICHT python-jose — CVE bekannt)
- Tier A (HS256, legacy): Direkte Verifikation gegen `SUPABASE_JWT_SECRET`
- Tier B (ES256, empfohlen): `PyJWKClient` mit JWKS Endpoint (`{supabase_url}/auth/v1/.well-known/jwks.json`), 5-Min Cache
- Validierte Claims: `sub` (User-UUID), `role` (authenticated), `aud` (authenticated), `exp` (automatisch)

**B) Rate Limiting (Phase 1, NICHT Phase 8!)**
- Zwei-Schicht-Ansatz:
  - Layer 1: Request-Count via `slowapi` + Redis (`10/minute; 100/hour` pro User)
  - Layer 2: Per-User LLM Token Budget in Redis (Sliding Window, z.B. 500.000 Tokens/Tag)
- Redis laeuft lokal auf Hetzner (`apt install redis-server`), kein separater Service noetig

**C) Webhook Authentication**
- `/webhook/activity` nutzt Shared Secret in Custom Header: `X-Webhook-Secret`
- Verifikation: `hmac.compare_digest()` (Timing-Attack sicher)
- Timestamp-Validierung: Events aelter als 5 Minuten werden rejected (Replay-Schutz)

**D) Row-Level Security (RLS) fuer Agent Config Tabellen**
- `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` auf ALLEN Agent Config Tabellen
- Authenticated Users: NUR `SELECT` Policy (`auth.uid() = user_id`)
- Kein `INSERT/UPDATE/DELETE` fuer Authenticated (nur service_role schreibt)
- `service_role` bypassed RLS automatisch — FastAPI Backend nutzt service_role fuer alle Writes

**E) Concurrent Agent Loop Protection**
- Problem: HeartbeatService + User-Chat gleichzeitig fuer denselben User
- Loesung: Redis Distributed Lock pro User (`agent_loop:lock:{user_id}`)
  - `blocking=False` — wenn Lock belegt, wird Heartbeat uebersprungen (nicht blockiert)
  - `timeout=300s` — Auto-Release bei Prozess-Crash
  - Redis wird ohnehin fuer Rate Limiting gebraucht — kein Extra-Service

**F) Database Connection Strategy**
- `supabase-py` Async Client fuer PostgREST Operationen (kein asyncpg direkt)
- Direkte Postgres-Verbindung (Port 5432) nur wenn PostgREST nicht reicht
- Connection Pooling: Von Supabase gemanaged (PostgREST). Kein pgbouncer noetig fuer Single-Server

**G) Error Handling Szenarien**

| Szenario | Verhalten |
|---|---|
| HeartbeatService Agent-Call schlaegt fehl | Retry nach 5 Min, max 3 Retries, dann Skip bis naechster Heartbeat |
| Concurrent Heartbeat + User Chat | Heartbeat uebersprungen (Redis Lock nicht acquired), User-Chat hat Prioritaet |
| Calc Engine Formel-Fehler (Division/0, NaN) | `evalidate` faengt Exceptions → Agent erhaelt Fehlermeldung → kann Formel korrigieren |
| LLM Provider Timeout/Down | LiteLLM Fallback-Chain: Gemini Flash → Claude Haiku → Fehler-Nachricht an User |
| Supabase Connection Loss mid-turn | Tool-Error wird zurueckgegeben → Agent antwortet mit "Ich kann gerade nicht auf deine Daten zugreifen" → Error nicht in History |
| SSE Connection Drop | App reconnected, fragt letzte Message aus `chat_messages` Tabelle ab. Kein Retry des Agent-Turns |

---

## 9. Alle getroffenen Entscheidungen

| Entscheidung | Wahl | Begruendung |
|---|---|---|
| Frontend | Bestehende athletly-app (React Native / Expo) | Schon gebaut, komplette UI vorhanden |
| Agent Backend | Python FastAPI | AgenticSports Agent-Logik wiederverwenden, LiteLLM, keine Timeout-Limits |
| Data/Auth/Realtime | Supabase | Managed Auth, RLS, Realtime — schon in App integriert |
| LLM | LiteLLM — beliebig wechselbar | Gemini, Claude, GPT, Llama — User/Admin kann wechseln |
| Gym Detail Level | Session-Typ ("Beine", "Oberkoerper Push") | Keine konkreten Uebungen/Sets/Reps. Agent entscheidet Session-Typen |
| Plan Interaction | Auto-generiert + User reviewed | Plan kommt automatisch, User muss akzeptieren |
| Prioritaeten | Agent entscheidet Strategie | Keine festen Modi — Agent waehlt beste Priorisierung |
| Planabweichung | Agent schlaegt an, User bestaetigt | Human-in-the-Loop via Checkpoint-System |
| Coach Chat | Eigener Tab + kontextuelle Quick-Actions | Chat fuer alles, Quick-Actions bei Sessions |
| Zielgruppe | Produkt-Ambition (Multi-User) | Erstmal fuer Roman, aber skalierbar gebaut |
| Hardcoded Rules | KEINE | Alles Agent-Reasoning + Agent-definierte Configs |
| Hardcoded Formulas | KEINE | Agent entdeckt + definiert, System rechnet |
| Voice Input | Native Speech-to-Text (iOS/Android) | Kostenlos, offline moeglich |
| Onboarding | Dynamischer Companion (Chat-basiert) | Kein fester Step-Flow, Agent fuehrt Gespraech |
| Affiliate | Agent-gesteuert, Teil der Plangeneration | Keine Produkt-Zuordnungstabellen |
| Subscription Pricing | Spaeter (nach MVP) | Erstmal Produkt bauen, dann monetarisieren |
| Deploy Python Backend | Hetzner Server mit Docker | Roman hat eigenen Server, Zugang via `ssh hetzner` |
| App Name | Athletly | — |
| Initiale Agent Configs | Tabula rasa — Agent entdeckt alles neu | Keine Migration, kein Seed. Agent startet bei Null |
| Supabase-Only fuer Agent? | NEIN — zu limitiert | Edge Functions: 150s Timeout, Cold Starts, kein Multi-Round Agent Loop |
| Garmin API | Noch kein Access | Apple Health + FIT Files als Fallback |
| Calc Engine | `evalidate` (Whitelist-AST) | Schnellste sichere Option (0.33s/1M ops), aktiv maintained, kein eval() |
| API Design | EIN Chat-Endpoint (NanoBot-Prinzip) | Keine dedizierten REST-Endpoints. Chat IST die API. Direkte DB-Reads fuer UI |
| SSE Streaming | NanoBot `_bus_progress` Pattern | Thinking + Tool Hint Events statt Token-Level-Streaming |
| Background Workers | HeartbeatService (NanoBot-Pattern) | SELBER Agent wacht auf, kein Celery/APScheduler |
| System Prompt | Statisch/cacheable (NanoBot-Pattern) | Runtime-Kontext als User-Message, nicht im System Prompt |
| Error Handling | Errors nie in History persistieren | Verhindert Context-Poisoning (NanoBot-Pattern) |
| Provider-Metriken | Garmin/Apple Health Werte direkt nutzen | Agent berechnet nur ergaenzende Formeln, Self-Improvement ueber Korrelation |
| Architektur-Referenz | ClawdBot/NanoBot/MoltBot/PicoClaw | Alle technischen Entscheidungen gegen diese Prinzipien geprueft |
| Push Notifications | Expo Push Notifications (`exp.host` API) | Kostenlos, unbegrenzt, kein SDK noetig, automatische APNs/FCM Zertifikate |
| JWT Verification | `PyJWT>=2.8` + `cryptography` | NICHT python-jose (CVE). HS256 fuer Start, ES256 Migration spaeter |
| Rate Limiting | `slowapi` + Redis (Request Count) + Custom Token Budget | Zwei-Schicht-Ansatz, Phase 1a (NICHT Phase 8!) |
| Concurrent Agent Protection | Redis Distributed Lock pro User | `blocking=False`, `timeout=300s`, Heartbeat wird uebersprungen wenn User chattet |
| DB Connection | `supabase-py` Async Client (PostgREST) | Kein asyncpg direkt, kein pgbouncer fuer Single-Server |
| LLM Fallback | LiteLLM Fallback-Chain | Gemini Flash → Claude Haiku → Fehler-Nachricht |
| Tool Result Truncation | Token-Budget pro Tool-Output | Grosse Outputs werden LLM-komprimiert, Fast-Path fuer kleine Outputs |
| Memory Consolidation | Drei-Tier: Session-Index + Monatliche Episodes + Belief-Dedup | NanoBot + SimpleMem-inspiriert |
| MessageTool | NanoBot `_sent_in_turn` Pattern | Turn-Level Dedup, Agent kann mid-turn Push senden |
| SpawnTool | NanoBot Background Subagent Pattern | Async Task-ID, max 15 Iterationen, Cleanup via done_callback |
| Webhook Auth | Shared Secret + `hmac.compare_digest` | Timestamp-Validierung (5 Min), Replay-Schutz |
| Phase 1 Struktur | 3 Sub-Phasen (1a/1b/1c) | Jede Sub-Phase eigenstaendig testbar und lieferbar |
| Checkpoint Injection | User-Message (NICHT System-Message) | Agent versteht WER entschieden hat |

---

## 10. Was wiederverwendet wird (mit Refactoring-Status)

### Aus athletly-backend (Python, ehemals AgenticSports)

| Modul | Datei | Status | Was zu tun ist |
|---|---|---|---|
| Agent Loop | `src/agent/agent_loop.py` | ✅ Wiederverwendbar | DB statt File-Storage |
| LLM Integration | `src/agent/llm.py` | ✅ Wiederverwendbar | Bereits LiteLLM |
| Episodic Memory | `src/agent/reflection.py` | ✅ Wiederverwendbar | DB statt File-Storage |
| User Model (Beliefs) | `src/tools/user_model.py` | ✅ Wiederverwendbar | DB statt File-Storage |
| System Prompt | `src/agent/system_prompt.py` | 🔧 **Komplett umbauen** | Aktuell dynamisch (f-strings mit Datum, User, Sportarten). Muss vollstaendig statisch werden — ALLE Runtime-Daten raus, als separate User-Message injizieren (NanoBot Prinzip 2). Aufwand hoeher als "Anpassen" |
| Plan Generator | `src/agent/prompts.py` | 🔧 Anpassen | Generisches Plan-Schema, Agent-definierte Struktur |
| Plan Evaluator | `src/agent/plan_evaluator.py` | 🔧 Umbauen | Hardcodierte 6 Kriterien → Agent-definierte Kriterien aus DB |
| Training Assessment | `src/agent/assessment.py` | 🔧 Anpassen | Generischer machen |
| TRIMP/HR-Zonen | `src/tools/metrics.py` | 🔧 Umbauen | Raus aus Code → Agent Config + Calc Engine |
| Fitness Tracking | `src/tools/fitness_tracker.py` | 🔧 Umbauen | Lauf-zentriert → generisch via Agent Configs |
| Trajectory | `src/agent/trajectory.py` | 🔧 Anpassen | Generischer machen |
| Proactive Triggers | `src/agent/proactive.py` | 🔧 Umbauen | Hardcodierte Trigger → Agent-definierte Regeln aus DB |
| FIT Parser | `src/tools/fit_parser.py` | ✅ Wiederverwendbar | Bleibt als Fallback |
| Activity Context | `src/tools/activity_context.py` | 🔧 Anpassen | Generischer machen, alle Sportarten |
| Startup | `src/agent/startup.py` | 🔧 Anpassen | Goal Type Inference generischer |
| Config | `src/config.py` | ✅ Wiederverwendbar | Erweitern fuer FastAPI |

### Aus athletly-v1 (React Native, jetzt archiviert)

| Modul | Status | Was zu tun ist |
|---|---|---|
| 5-Tab Navigation | ✅ Behalten | — |
| WeekCalendarStrip | ✅ Behalten | Generisch (liest Plan aus DB) |
| DayDetailCard | ✅ Behalten | Product Cards hinzufuegen |
| Coach Chat (SSE) | 🔧 Umleiten | Endpoint von Edge Functions → FastAPI |
| Checkpoint System | ✅ Behalten | Funktioniert mit neuem Backend |
| Supabase Auth | ✅ Behalten | — |
| Supabase Realtime | ✅ Behalten | — |
| Garmin Integration | ✅ Behalten | — |
| Apple Health | ✅ Behalten | Mehr Datentypen einlesen |
| Health Connect | ✅ Behalten | Mehr Datentypen einlesen |
| Zustand Stores | 🔧 Anpassen | Multi-Goal, Agent Configs |
| Onboarding (9 Steps) | 🔧 Ersetzen | Dynamischer Chat-Companion |
| Edge Functions Agent | ❌ Ersetzen | Durch Python FastAPI |
| Design System | ✅ Behalten | — |
| Push Notifications | ✅ Behalten | — |

---

## 11. Roadmap (Detail)

Jede Phase liefert eigenstaendig Wert und ist testbar.

### Phase 1a: Core Chat Endpoint End-to-End
**Ziel**: Funktionierender `POST /chat` SSE Endpoint — User kann mit dem Agent chatten, Agent hat Zugriff auf DB

**Tasks:**
- [ ] FastAPI Server Setup mit Supabase JWT Verification (`PyJWT>=2.8` + `cryptography`, HS256 Tier A)
- [ ] DB Schema erstellen: `chat_sessions`, `chat_messages` (falls nicht vorhanden), `pending_actions`
- [ ] Agent Loop von AgenticSports portieren: Supabase statt JSON-Files, `supabase-py` Async Client
- [ ] System Prompt **komplett umbauen**: Statisch/cacheable — ALLE Runtime-Daten (Datum, User, Sportarten) raus, als separate User-Message injizieren (Sektion 8.11 Prinzip 2)
- [ ] Error Responses NIE in Session History persistieren (Sektion 8.11 Prinzip 3)
- [ ] EIN Chat-Endpoint: `POST /chat` (SSE) — NanoBot/ClawdBot-Prinzip
- [ ] SSE Event-Protokoll: start, thinking, tool_hint, message, usage, error, done (Sektion 8.7)
- [ ] Checkpoint-Confirm Endpoint: `POST /chat/confirm` (User-Message Injection)
- [ ] Basis Rate Limiting: `slowapi` + Redis (`10/min; 100/hour` pro User) + Per-User Token Budget
- [ ] Concurrent Agent Loop Protection: Redis Lock pro User (Sektion 8.13 E)
- [ ] Tool Result Truncation: `execute_with_budget()` Wrapper (Sektion 8.12 A)
- [ ] **Test**: User kann via SSE chatten, Agent antwortet mit Thinking + Tool Hint Events, Errors werden nicht persistiert, Rate Limiting greift

### Phase 1b: Agent Config Store + Calc Engine
**Ziel**: Agent kann eigenstaendig Metriken definieren, Formeln werden sicher berechnet

**Tasks:**
- [ ] DB Schema: Agent Configuration Store Tabellen (metric_definitions, eval_criteria, session_schemas, periodization_models, proactive_trigger_rules, product_recommendations)
- [ ] RLS Policies auf allen Agent Config Tabellen (SELECT fuer authenticated, Writes nur service_role)
- [ ] Generic Calculation Engine mit `evalidate` implementieren (Whitelist-AST, kein eval(), Error Handling fuer Division/0/NaN)
- [ ] DB Schema: `calculated_metrics` Tabelle fuer berechnete Werte
- [ ] Agent-Tools fuer Config-Management: `define_metric`, `define_eval_criteria`, `define_session_schema`, `get_config`, `update_config`
- [ ] Agent-Tool: `calculate_metric`, `calculate_bulk_metrics` (nutzen Calc Engine)
- [ ] KEINE Migration von hardcodierten Formeln — Tabula rasa
- [ ] Plan-Output muss exakt dem `weekly_plans.days` JSONB-Schema entsprechen (Sektion 8.6)
- [ ] DB Schema Fixes: `weekly_plans` um `coach_message` + `reasoning` Columns erweitern
- [ ] **Test**: Agent definiert eigenstaendig TRIMP-Formel via `define_metric`, Calc Engine berechnet korrekt, Plan landet in Supabase (korrektes Schema)

### Phase 1c: Proaktive Intelligenz + Deploy
**Ziel**: HeartbeatService laeuft, Push Notifications funktionieren, Backend ist auf Hetzner deployed

**Tasks:**
- [ ] DB Schema: `push_notifications`, `health_data` Tabellen
- [ ] Agent-Tool: `send_notification` (Expo Push via `exp.host` API, Turn-Level Deduplication)
- [ ] Agent-Tool: `spawn_background_task` (async Subagent fuer Plan-Generation)
- [ ] HeartbeatService: asyncio-basiert, konfigurierbare Intervalle, Redis Lock gegen Concurrent Loops
- [ ] Activity Webhook: `POST /webhook/activity` (Shared Secret Auth, Timestamp-Validierung)
- [ ] LiteLLM Fallback-Chain konfigurieren: Gemini Flash → Claude Haiku → Fehler
- [ ] Session Summary Index: Auto-Summary bei Session-Ende + `search_session_history` Tool
- [ ] Dockerfile + docker-compose.yml (FastAPI + Redis)
- [ ] Deploy auf Hetzner Server (Docker Container, `ssh hetzner`)
- [ ] **Test**: HeartbeatService erkennt fehlende Aktivitaet, Agent sendet Push via `send_notification`, Plan-Generation laeuft als Background Task

### Phase 2: App ← Python Backend
**Ziel**: athletly-app nutzt Python Chat-Endpoint statt Edge Functions. Alles geht durch den Chat.

**Tasks:**
- [ ] `useChatStream` Hook in App auf FastAPI `POST /chat` SSE Endpoint umleiten
- [ ] App verarbeitet neue SSE Events: `thinking` (transiente Anzeige), `tool_hint` (Status-Chips), `message` (finale Antwort)
- [ ] Plan-Darstellung liest direkt aus Supabase `weekly_plans` (kein dedizierter Plan-API-Endpoint noetig)
- [ ] Checkpoint/Confirm Flow: App ruft `POST /chat/confirm` auf statt alten Endpoint
- [ ] Agent-Status via SSE `thinking`/`tool_hint` Events (ersetzen `agent_status` Realtime)
- [ ] Edge Functions Agent deaktivieren/entfernen
- [ ] App liest Agent-Configs direkt aus Supabase fuer Darstellung (z.B. welche Metriken anzeigen)
- [ ] Fix: App Multi-Session pro Tag ermoeglichen (aktuell nur sessions[0])
- [ ] Fix: coachMessage + reasoning aus DB lesen statt hardcodiert
- [ ] Fix: Rest Day reason aus DB lesen statt hardcodiert "Erholungstag"
- [ ] **Test**: Chat + Plangenerierung funktioniert komplett in der App via Python Backend, Thinking/Tool Hint Events werden angezeigt, Multi-Session Tage

### Phase 3: Dynamischer Companion + Voice
**Ziel**: Modernes Onboarding als Gespraech mit Spracheingabe

**Tasks:**
- [ ] 9-Schritt Onboarding Flow ersetzen durch Chat-basierten Companion Screen
- [ ] Native Speech-to-Text Integration (expo-speech oder native Module)
- [ ] Mikrofon-Button im Chat + Onboarding
- [ ] Agent Onboarding-Logik: Dynamisches Gespraech, erkennt Sportarten aus Freitext
- [ ] Agent definiert initial Configs basierend auf Onboarding-Infos (Metriken, Schemas)
- [ ] Erster Plan wird generiert sobald Agent genug Infos hat
- [ ] **Test**: Neuer User spricht/tippt frei, Agent baut Profil auf, erster Plan wird generiert

### Phase 4: Multi-Sport Intelligence
**Ziel**: Agent koordiniert ueber Sportarten hinweg, Self-Improvement

**Tasks:**
- [ ] Agent-Tools erweitern: Cross-Sport Reasoning mit Health-Daten im Kontext
- [ ] Unbekannte-Aktivitaet-Flow (Push → Chat → Klassifizierung → ggf. neue Configs)
- [ ] Adaptive Replanning mit Checkpoint-Bestaetigung
- [ ] Agent definiert eigene Proactive Trigger Regeln (statt hardcodierte)
- [ ] Self-Improvement Loop: Agent prueft ob Formeln/Kriterien funktionieren
- [ ] **Test**: Multi-Sport Wochenplan der Ermuedung beruecksichtigt, unbekannte Aktivitaet wird erkannt und klassifiziert

### Phase 5: Generischer Health Data Import
**Ziel**: Alle verfuegbaren Daten von allen Quellen nutzen

**Tasks:**
- [ ] Apple Health: Alle verfuegbaren Datentypen inventarisieren und generisch einlesen
- [ ] Garmin: Alle verfuegbaren Metriken synchen
- [ ] Health Connect: Equivalent zu Apple Health
- [ ] Agent entscheidet welche Health-Daten fuer Planung relevant sind
- [ ] Health-Daten in Agent-Kontext einbauen (Schlaf, Stress, HRV → Planungsrelevanz)
- [ ] **Test**: Schlaf-Daten beeinflussen Planungsempfehlung

### Phase 6: Affiliate & Monetarisierung
**Ziel**: Agent-gesteuerte Produktempfehlungen

**Tasks:**
- [ ] Agent-Tool: `recommend_products` (recherchiert + empfiehlt)
- [ ] Amazon Affiliate Integration (Affiliate Tags, Product URLs)
- [ ] Product Card Component in der App (unter Session-Details in DayDetailCard)
- [ ] Subscription Tier System implementieren (Details TBD)
- [ ] **Test**: Plan wird generiert mit passenden Produktempfehlungen pro Session

### Phase 7: Long-Term Intelligence
**Ziel**: Coach denkt in Monaten

**Tasks:**
- [ ] Agent definiert Periodisierungs-Modelle in `periodization_models`
- [ ] Trend-Erkennung ueber Wochen/Monate (Agent analysiert Episodic Memory)
- [ ] Proaktive Langzeit-Hinweise via Push
- [ ] Goal Trajectory per Sport (generisch, nicht nur Laufen)
- [ ] DB Schema: `episode_consolidations` Tabelle + monatliche Konsolidierungslogik (Sektion 8.12 C)
- [ ] **Test**: 12-Wochen Vorschau mit phasengerechter Planung

### Phase 8: Multi-User, Polish & Launch
**Ziel**: App Store Ready

**Tasks:**
- [ ] Multi-User RLS Feintuning
- [ ] Garmin OAuth pro User
- [ ] Settings Screen erweitern
- [ ] Rate Limiting + Cost Monitoring (LLM-Kosten)
- [ ] App Store Submission (iOS + Android)
- [ ] **Test**: Zweiter User kann sich anmelden und eigenen Coach nutzen

---

## 12. Geklärte Fragen

| Frage | Antwort |
|---|---|
| **Garmin API** | Noch kein Access. Apple Health + FIT Files als Fallback |
| **Repo-Struktur** | **Separate Repos**: `RnltLabs/athletly-app` (Frontend V2) + `RnltLabs/athletly-backend` (Python FastAPI). V1 archiviert als `RnltLabs/athletly-v1`. Getrennte Deploy-Zyklen, klare Verantwortung. Backend wird auf Hetzner deployed, App via EAS Build |
| **Deploy-Target** | **Hetzner Server mit Docker**. Zugang via `ssh hetzner` |
| **App Name** | **Athletly** |
| **Calc Engine Parser** | **`evalidate`** — Whitelist-AST, 0.33s/1M ops, Maerz 2026 Release. Alternativen evaluiert und verworfen (simpleeval, asteval/CVE, numexpr/unsicher, py_expression_eval/unmaintained) |
| **Initiale Agent Configs** | Agent entdeckt ALLES komplett neu. Keine Migration von hardcodierten Formeln. Tabula rasa. Token-Kosten: ~500 Tokens = $0.0002 pro User |
| **API Design** | EIN Chat-Endpoint nach NanoBot/ClawdBot-Prinzip. Keine `/plan/generate` etc. Alles geht durch den Agent via Chat |
| **Streaming/Latenz** | NanoBot `_bus_progress` Pattern: Thinking + Tool Hint Events bei jedem Tool-Call. Kein Token-Streaming |
| **Background Workers** | HeartbeatService statt 4 separate Workers. SELBER Agent, keine extra Infrastruktur |
| **Provider-Metriken** | Garmin/Apple Health liefern VO2max, Training Effect, HRV etc. Agent nutzt diese + ergaenzende eigene Formeln |
| **Push Notifications** | Expo Push Notifications (`exp.host` API). Kostenlos, unbegrenzt, kein SDK. Automatische APNs/FCM Zertifikate via EAS |
| **JWT Library** | `PyJWT>=2.8` + `cryptography`. NICHT python-jose (CVE). HS256 Tier A fuer Start, ES256/JWKS Migration spaeter |
| **Rate Limiting** | `slowapi` + Redis (Request Count) + Custom Token Budget in Redis. In Phase 1a, NICHT Phase 8 |
| **Concurrent Agent Loops** | Redis Distributed Lock pro User. Heartbeat uebersprungen wenn User chattet. `timeout=300s` Auto-Release |
| **DB Connection** | `supabase-py` Async Client (PostgREST). Kein asyncpg, kein pgbouncer fuer Single-Server |
| **LLM Fallback** | LiteLLM Fallback-Chain: Gemini Flash → Claude Haiku → Fehler-Nachricht |
| **Tool Result Truncation** | Token-Budget pro Tool-Output (1.500 fuer Activities, 2.000 Default). LLM-komprimiert wenn uebers Budget |
| **Memory Consolidation** | Session-Index (150 Tokens/Session) + Monatliche Episode-Consolidation + Belief-Deduplikation (cosine > 0.88) |
| **Checkpoint Injection** | User-Message, NICHT System-Message. Agent versteht WER entschieden hat |
| **Webhook Auth** | Shared Secret in Custom Header + `hmac.compare_digest` + Timestamp-Validierung (5 Min Replay-Schutz) |
| **Phase 1 Split** | 3 Sub-Phasen: 1a (Core Chat E2E + Security), 1b (Config Store + Calc Engine), 1c (Proaktiv + Deploy) |
| **Fehlende DB-Tabellen** | 6 neue Tabellen definiert: chat_sessions, calculated_metrics, pending_actions, health_data, push_notifications, episode_consolidations |

---

## 13. Romans spezifisches Beispiel-Profil (fuer Tests)

Zum Testen ob der Agent korrekt arbeitet:

- **Laufen**: Halbmarathon, Zielzeit ~1:45, 3x/Woche
- **Rennrad**: Leistung verbessern + Spass, 2x/Woche
- **Schwimmen**: Dranbleiben, 1x/Woche
- **Gym**: Muskelaufbau, 73kg/180cm lean, ~3x/Woche, Session-Typen (Beine, Oberkoerper, etc.)
- **Ruhetag**: Sonntag bevorzugt
- **Max Sessions/Tag**: 2
- **Verfuegbare Tage**: Mo-Sa

Der Agent sollte z.B. erkennen: Nach Gym-Beintraining am Montag sollte Dienstag kein harter Lauf-Intervall sein. Schwimmen kann an einem Tag mit leichtem Lauf kombiniert werden. etc.

---

## 14. Bekannte Probleme & technische Schulden im bestehenden Code

### In athletly-v1 (Legacy Frontend — archiviert)
1. **Multi-Session pro Tag**: App zeigt nur `sessions[0]` — weitere Sessions werden ignoriert. Muss fuer Multi-Sport (z.B. morgens Lauf + abends Gym) gefixt werden
2. **`coachMessage` + `reasoning` hardcodiert**: App setzt "Dein personalisierter Trainingsplan ist bereit!" als Default. Backend kann diese Felder nicht ueber die DB setzen. → Top-Level Columns in `weekly_plans` ergaenzen
3. **Rest Day `reason` hardcodiert**: Immer "Erholungstag" — Backend kann keinen Custom-Grund senden. → In `days` JSONB ergaenzen
4. **`sessionType` Fallback fragil**: Wenn `session_type` fehlt, rät die App basierend auf `intensity`. → Backend muss IMMER `session_type` explizit senden
5. **Sport-Icons/Farben sind hardcodiert**: `tailwind.config.js` hat feste Sport-Farben. Fuer unbekannte Sports gibt es Default-Werte, aber das Mapping sollte langfristig dynamisch sein
6. **Onboarding ist 9 feste Schritte**: Muss durch dynamischen Companion ersetzt werden (Phase 3)

### In athletly-backend (Python, ehemals AgenticSports)
1. **TRIMP/HR-Zonen hardcodiert** in `src/tools/metrics.py` — muss komplett raus, durch Agent Config + Calc Engine ersetzt
2. **Plan Evaluator hat 6 feste Kriterien** — muessen durch Agent-definierte Kriterien aus DB ersetzt werden
3. **System Prompt enthält sport-spezifische Expertise-Listen** — muss generisch werden
4. **Fitness Tracking ist lauf-zentriert** (`threshold_pace_min_km`, `weekly_volume_km`) — muss generisch via Agent Configs
5. **Proactive Triggers sind hardcodiert** — muessen durch Agent-definierte Regeln ersetzt werden
6. **Storage ist File-basiert (JSON)** — muss auf Supabase umgestellt werden (teilweise schon vorbereitet)

---

## 15. Risiken & Mitigationen

| Risiko | Impact | Mitigation |
|---|---|---|
| LLM-Reasoning Qualitaet fuer Cross-Sport | Hoch | Strukturierter Kontext in Prompts, Plan-Evaluator als Quality Gate, iterative Verbesserung |
| Cold Start Latenz (erster Plan) | Mittel | Progressive Configuration (Sektion 8.9), LLM kennt Sportformeln bereits |
| LLM-Kosten bei Scale | Mittel | Gemini Flash ist guenstig (~$0.01/Plan), Caching, Config-Reuse ueber Sessions |
| Generic Calc Engine Sicherheit | ~~Hoch~~ Geloest | `evalidate` Whitelist-AST — blockiert alles was nicht explizit erlaubt ist |
| Garmin API Access | Mittel | Apple Health + FIT Files als Fallback, beide schon implementiert |
| App Plan-Schema Mismatch | Hoch | Exakter Vertrag dokumentiert (Sektion 8.6), Adapter-Layer im Backend |
| Agent Config Drift | Niedrig | Config Versioning (created_at/updated_at), Agent prueft Ergebnisse |
