<div align="center">
  <h1>Athletly Backend</h1>
  <h3>Autonomous AI Sports Coach — Depth over Breadth</h3>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.12-blue" alt="Python">
    <img src="https://img.shields.io/badge/LLM-Gemini_2.5_Flash-orange" alt="LLM">
    <img src="https://img.shields.io/badge/architecture-agentic_loop-red" alt="Architecture">
    <img src="https://img.shields.io/badge/API-FastAPI-009688" alt="FastAPI">
    <img src="https://img.shields.io/badge/DB-Supabase_%2B_pgvector-3ECF8E" alt="Supabase">
    <img src="https://img.shields.io/badge/streaming-SSE-blueviolet" alt="SSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

---

An autonomous coaching engine that reasons through an agentic loop with 23 specialized tools, belief-driven memory with pgvector embeddings, and real-time SSE streaming. Built on FastAPI with Supabase multi-user persistence, Redis concurrency control, and LiteLLM as the provider-agnostic LLM gateway.

> **Why "Depth over Breadth"?** Most agent frameworks go wide: many platforms, many providers, generic tools. Athletly goes **deep**: one domain, 23 specialized tools, belief-driven memory, probabilistic athlete modeling. The LLM makes every coaching decision — but the math is always correct.

---

## System Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        APP["Mobile App<br/>(Flutter)"]
    end

    subgraph Gateway["API Gateway"]
        direction TB
        FASTAPI["FastAPI<br/>Uvicorn ASGI"]
        AUTH["JWT Auth<br/>ES256 / HS256"]
        RATE["Rate Limiter<br/>slowapi"]
        SSE["SSE Stream<br/>EventSource"]
    end

    subgraph Core["Cognitive Engine"]
        LOOP["Agentic Loop"]
        PROMPT["System Prompt<br/>Static + Runtime Context"]
        TOOLS["Tool Registry<br/>23 Modules"]
        CALC["CalcEngine<br/>Safe Formula Eval"]
    end

    subgraph Intelligence["AI Layer · LiteLLM"]
        LITELLM["LiteLLM<br/>Provider Router"]
        GEMINI["Gemini 2.5 Flash<br/>(Primary)"]
        CLAUDE["Claude Haiku<br/>(Fallback)"]
        EMBED["text-embedding-004<br/>768-dim Vectors"]
    end

    subgraph Persistence["Data Layer"]
        SUPA[("Supabase<br/>PostgreSQL + pgvector<br/>+ Row-Level Security")]
        REDIS[("Redis 7<br/>Locks · Cache · Cooldowns")]
    end

    subgraph Workers["Background Services"]
        HEART["HeartbeatService<br/>30 min cycle"]
        GARMIN["GarminSyncService<br/>OAuth + Metrics"]
        SUMMARY["SessionSummarizer<br/>Context Compression"]
        EPISODE["EpisodeConsolidation<br/>Meta-Learning"]
        USAGE["UsageTracker<br/>Token Budget"]
    end

    subgraph Health["Health Providers"]
        GC["Garmin Connect"]
        AH["Apple Health"]
        HC["Health Connect"]
    end

    APP -->|"POST /chat · JWT"| FASTAPI
    FASTAPI --> AUTH
    FASTAPI --> RATE
    FASTAPI -->|"EventSourceResponse"| SSE
    SSE -->|"thinking · tool · message"| APP

    FASTAPI --> LOOP
    LOOP <--> LITELLM
    LOOP --> TOOLS
    TOOLS --> CALC
    PROMPT --> LOOP

    LITELLM --> GEMINI
    LITELLM -.->|"on failure"| CLAUDE
    LITELLM --> EMBED

    TOOLS --> SUPA
    TOOLS --> REDIS
    LOOP --> SUPA
    AUTH --> SUPA

    HEART --> SUPA
    HEART --> REDIS
    GARMIN --> GC
    GARMIN --> SUPA
    SUMMARY --> LITELLM

    GC -.-> SUPA
    AH -.-> SUPA
    HC -.-> SUPA

    style FASTAPI fill:#009688,stroke:#fff,color:#fff
    style LOOP fill:#e94560,stroke:#fff,color:#fff
    style LITELLM fill:#1a1a2e,stroke:#e94560,color:#fff
    style GEMINI fill:#4285F4,stroke:#fff,color:#fff
    style CLAUDE fill:#D97757,stroke:#fff,color:#fff
    style EMBED fill:#0f3460,stroke:#533483,color:#fff
    style TOOLS fill:#16213e,stroke:#0f3460,color:#fff
    style CALC fill:#533483,stroke:#e94560,color:#fff
    style SUPA fill:#3ECF8E,stroke:#fff,color:#fff
    style REDIS fill:#DC382D,stroke:#fff,color:#fff
    style HEART fill:#16213e,stroke:#0f3460,color:#fff
    style SSE fill:#7C3AED,stroke:#fff,color:#fff
```

---

## Core Principle: Code Computes, LLM Reasons

The agent defines metrics, formulas, and evaluation criteria at runtime through tools. A sandboxed expression engine (`CalcEngine` via `evalidate`) evaluates them deterministically — no hardcoded sport logic, no LLM-hallucinated math.

```mermaid
flowchart LR
    subgraph LLM["LLM Decides"]
        A["define_metric(<br/>name='trimp',<br/>formula='duration * avg_hr * 0.64 * e^(1.92 * avg_hr/max_hr)')"]
    end

    subgraph Engine["CalcEngine Evaluates"]
        B["evalidate sandbox<br/>whitelist: math ops only<br/>no imports, no I/O"]
    end

    subgraph Result["Deterministic Output"]
        C["TRIMP = 142.7"]
    end

    A -->|"formula string"| B
    B -->|"float result"| C

    style A fill:#e94560,stroke:#fff,color:#fff
    style B fill:#533483,stroke:#e94560,color:#fff
    style C fill:#22C55E,stroke:#fff,color:#fff
```

**What this means in practice:**

| Concern | Who Handles It | How |
|---|---|---|
| Which metrics matter for this sport? | LLM | `define_metric()`, `define_eval_criteria()` |
| Calculate TRIMP from heart rate data | CalcEngine | Sandboxed formula evaluation |
| Is this plan good enough? | LLM | Scores against agent-defined criteria |
| What's my threshold pace? | CalcEngine | Jack Daniels formula, agent-defined |
| Should I adjust intensity this week? | LLM | Analyzes recovery + load + beliefs |

---

## Agentic Loop

Inspired by [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — the LLM sees all 23 tools and autonomously decides what to call, when, and in what order.

```mermaid
flowchart TD
    START(["User Message"]) --> CTX["Build Runtime Context<br/>profile · 7d summary · beliefs · recovery"]
    CTX --> LLM["LLM Call<br/>system_prompt + messages + 23 tools"]
    LLM --> DECIDE{Tool calls<br/>in response?}

    DECIDE -->|"Yes"| EXEC["Execute Tools<br/>in parallel where possible"]
    EXEC --> EMIT_T["Emit SSE: tool_hint → tool_result"]
    EMIT_T --> LLM

    DECIDE -->|"No"| RESP["Final Response"]
    RESP --> EMIT_M["Emit SSE: message"]
    EMIT_M --> SAVE["Persist session + extract beliefs"]
    SAVE --> DONE(["SSE: done"])

    EXEC -->|"error after 3 retries"| FALLBACK["Fallback: tool-free mode"]
    FALLBACK --> LLM

    style START fill:#16213e,stroke:#0f3460,color:#fff
    style LLM fill:#e94560,stroke:#fff,color:#fff
    style EXEC fill:#533483,stroke:#e94560,color:#fff
    style RESP fill:#0f3460,stroke:#533483,color:#fff
    style SAVE fill:#3ECF8E,stroke:#fff,color:#fff
    style DONE fill:#16213e,stroke:#0f3460,color:#fff
```

**Loop constraints:**
- Max **25 tool rounds** per message (prevents infinite loops)
- **Context compression** at 40 messages (older turns summarized, last 4 kept verbatim)
- **Tool output truncation** at 2KB per tool call
- **Daily token budget** per user (500K tokens default, tracked via `UsageTracker`)

---

## LLM Layer: LiteLLM Provider Abstraction

LiteLLM provides a unified OpenAI-compatible interface across all providers. One `chat_completion()` call works with Gemini, Claude, or OpenAI — switching models requires only a config change.

```mermaid
flowchart LR
    subgraph App["Application"]
        CC["chat_completion(<br/>messages, tools,<br/>temperature)"]
    end

    subgraph LiteLLM["LiteLLM Router"]
        ROUTER["litellm.completion()<br/>drop_params=True<br/>provider auto-detect"]
    end

    subgraph Providers["LLM Providers"]
        G["Gemini 2.5 Flash<br/>+ thinking budget<br/>(10K tokens)"]
        C["Claude Haiku 4.5<br/>(fallback)"]
        O["OpenAI GPT-4o<br/>(optional)"]
    end

    subgraph Embed["Embedding"]
        E["google.genai.Client<br/>text-embedding-004<br/>768 dimensions"]
    end

    CC --> ROUTER
    ROUTER -->|"gemini/"| G
    ROUTER -.->|"on failure"| C
    ROUTER -.->|"optional"| O

    CC -.->|"belief similarity"| E

    style ROUTER fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#4285F4,stroke:#fff,color:#fff
    style C fill:#D97757,stroke:#fff,color:#fff
    style O fill:#10A37F,stroke:#fff,color:#fff
    style E fill:#0f3460,stroke:#533483,color:#fff
```

**Model selection strategy:**

| Model | Use Case | Temperature |
|---|---|---|
| Gemini 2.5 Flash | Main agent loop, plan generation, coaching | 0.7 |
| Gemini 2.0 Flash | Voice parsing (onboarding, fast extraction) | 0.1 |
| Claude Haiku 4.5 | Automatic fallback when Gemini fails | inherited |
| text-embedding-004 | Belief similarity search (pgvector cosine) | — |

**Gemini 2.5 special handling:** When tools + large system prompts are combined, a `thinking` budget of 10,000 tokens is injected to prevent empty responses — a Gemini-specific optimization handled transparently by the LLM layer.

---

## Belief-Driven Memory System

Every piece of athlete knowledge is a **belief** — not a static field. Beliefs carry confidence scores, pgvector embeddings, outcome tracking, and temporal metadata. They strengthen on confirmation and decay on contradiction.

```mermaid
flowchart TD
    subgraph Extract["Extraction · Every Message"]
        MSG["Athlete: 'Mein Halbmarathon-Ziel ist<br/>unter 1:30 im September'"] --> LLM_EX["LLM extracts beliefs"]
        LLM_EX --> B1["add_belief(<br/>text='HM Ziel: sub 1:30',<br/>category='motivation',<br/>confidence=0.9)"]
        LLM_EX --> B2["update_profile(<br/>goal.target_time='1:30:00',<br/>goal.target_date='2026-09')"]
    end

    subgraph Store["Storage · pgvector"]
        B1 --> EMB["Gemini Embedding<br/>768-dim vector"]
        EMB --> PG[("beliefs table<br/>cosine similarity<br/>match_beliefs() RPC")]
        B2 --> PROF[("profiles table<br/>structured_core JSONB")]
    end

    subgraph Retrieve["Retrieval · Per Session"]
        PG -->|"active beliefs<br/>confidence ≥ 0.6"| CTX["Runtime Context<br/>injected as first message"]
        PROF --> CTX
    end

    subgraph Learn["Learning · Outcome Tracking"]
        CTX --> PLAN["Plan uses belief"]
        PLAN -->|"athlete confirms"| UP["confidence ++<br/>utility ++"]
        PLAN -->|"athlete contradicts"| DOWN["confidence --<br/>superseded_by = new_belief"]
        DOWN -.->|"confidence < 0.3"| ARCHIVE["Auto-archive"]
    end

    style LLM_EX fill:#e94560,stroke:#fff,color:#fff
    style PG fill:#3ECF8E,stroke:#fff,color:#fff
    style EMB fill:#0f3460,stroke:#533483,color:#fff
    style CTX fill:#533483,stroke:#e94560,color:#fff
    style ARCHIVE fill:#6B7280,stroke:#fff,color:#fff
```

**Belief schema:**

| Field | Type | Purpose |
|---|---|---|
| `text` | string | Human-readable belief content |
| `category` | enum | `preference` · `constraint` · `fitness` · `physical` · `motivation` · `history` · `scheduling` · `personality` · `meta` |
| `confidence` | float 0.0–1.0 | Strength of belief, updated on confirm/contradict |
| `embedding` | vector(768) | Gemini `text-embedding-004` for similarity search |
| `stability` | enum | `stable` · `evolving` · `transient` |
| `durability` | enum | `global` · `seasonal` · `episode` · `session` |
| `outcome_history` | jsonb[] | Tracks confirm/contradict events over time |

---

## Real-Time SSE Streaming

The chat endpoint streams every stage of the agent's reasoning to the client — not just the final answer. Tool calls, intermediate results, and thinking are all visible in real time.

```mermaid
sequenceDiagram
    participant App as Mobile App
    participant API as FastAPI
    participant Lock as Redis Lock
    participant Agent as Agent Loop
    participant LLM as Gemini 2.5 Flash
    participant Tools as Tool Registry
    participant DB as Supabase

    App->>+API: POST /chat {message, session_id, JWT}
    API->>Lock: SET agent_loop:lock:{user_id} NX EX 300
    API->>DB: UserModelDB.load_or_create(user_id)
    API->>+Agent: AsyncAgentLoop.process_message_sse()
    API-->>App: SSE: session_start {session_id}

    rect rgb(30, 30, 50)
        Note over Agent,Tools: Tool Round 1
        Agent->>+LLM: system_prompt + context + messages + tools
        LLM-->>-Agent: tool_calls: [get_activities, get_daily_metrics]
        Agent-->>App: SSE: tool_hint {name, args}
        Agent->>+Tools: execute(get_activities, {days: 7})
        Tools->>DB: SELECT FROM activities...
        Tools-->>-Agent: {sessions: 5, distance_km: 42}
        Agent-->>App: SSE: tool_result {name, preview}
    end

    rect rgb(30, 30, 50)
        Note over Agent,Tools: Tool Round 2
        Agent->>+LLM: messages + tool_results
        LLM-->>-Agent: tool_calls: [create_training_plan]
        Agent-->>App: SSE: tool_hint {name, args}
        Agent->>+Tools: execute(create_training_plan, {...})
        Tools->>LLM: generate_plan(coach_prompt, context)
        Tools-->>-Agent: {plan: {...}, evaluation_score: 82}
        Agent-->>App: SSE: tool_result {name, preview}
    end

    Agent->>+LLM: messages + tool_results
    LLM-->>-Agent: "Hier ist dein Trainingsplan..."
    Agent-->>App: SSE: message {text}
    Agent->>DB: save session + messages + beliefs
    Agent-->>App: SSE: usage {tokens}
    Agent-->>-App: SSE: done

    API->>Lock: DEL agent_loop:lock:{user_id}
    deactivate API
```

**SSE event types:**

| Event | When | Payload |
|---|---|---|
| `session_start` | Immediately | `{session_id}` |
| `thinking` | LLM reasoning | `{text}` |
| `tool_hint` | Before tool execution | `{name, args}` |
| `tool_result` | After tool execution | `{name, preview}` |
| `tool_error` | Tool failure | `{name, error}` |
| `message` | Final response | `{text}` |
| `pending_action` | Checkpoint proposal | `{action_id, type, description}` |
| `usage` | Token accounting | `{input_tokens, output_tokens}` |
| `done` | Stream complete | `{}` |

---

## Prompt Architecture: Static + Runtime Split

The system prompt is split into a **static** component (identical for all users, LLM-cacheable) and a **runtime context** (per-user, per-request).

```mermaid
flowchart LR
    subgraph Static["Static System Prompt · Cacheable"]
        direction TB
        ID["Identity:<br/>'You are Athletly, an AI coaching agent'"]
        RULES["Rules:<br/>tool usage patterns,<br/>belief extraction mandate,<br/>language detection"]
        SAFETY["Safety:<br/>never guess, always verify,<br/>use tools to check data"]
    end

    subgraph Runtime["Runtime Context · Per Request"]
        direction TB
        DATE["Current date"]
        PROFILE["Athlete profile:<br/>name, sports, goals,<br/>fitness metrics"]
        BELIEFS["Active beliefs:<br/>confidence ≥ 0.6"]
        HEALTH["Recovery status:<br/>sleep, HRV, stress,<br/>body battery"]
        LOAD["7-day training load:<br/>sessions, TRIMP,<br/>volume by sport"]
        MACRO["Macrocycle week:<br/>phase, focus, targets"]
    end

    subgraph Injection["Message Assembly"]
        SYS["system: STATIC_SYSTEM_PROMPT"]
        USR1["user: [CONTEXT] runtime block"]
        USR2["user: athlete's actual message"]
    end

    Static --> SYS
    Runtime --> USR1
    SYS --> Injection
    USR1 --> Injection
    USR2 --> Injection

    style Static fill:#16213e,stroke:#0f3460,color:#fff
    style Runtime fill:#533483,stroke:#e94560,color:#fff
    style Injection fill:#e94560,stroke:#fff,color:#fff
```

**Why split?** The static prompt (~1,600 lines) is identical across all requests. LLM providers can cache it, reducing latency and cost. The runtime context changes per user and session — injected fresh every turn.

---

## Tool System

23 tools organized into domain-specific categories. The LLM autonomously selects which tools to call — there is no router, no hardcoded orchestration.

```mermaid
graph LR
    subgraph Registry["Tool Registry"]
        direction TB
        REG["register_tools()<br/>→ OpenAI function schema"]
    end

    subgraph Data["Data & Analysis"]
        D1["get_athlete_profile"]
        D2["get_activities"]
        D3["analyze_training_load"]
        D4["compare_plan_vs_actual"]
        D5["get_weekly_summary"]
    end

    subgraph Planning["Planning & Evaluation"]
        P1["create_training_plan"]
        P2["evaluate_plan"]
        P3["save_plan"]
        P4["create_macrocycle"]
        P5["assess_goal_trajectory"]
    end

    subgraph Memory["Memory & Config"]
        M1["add_belief"]
        M2["update_profile"]
        M3["define_metric"]
        M4["define_eval_criteria"]
        M5["define_session_schema"]
    end

    subgraph Health["Health & Wearables"]
        H1["get_daily_metrics"]
        H2["get_health_inventory"]
        H3["analyze_health_trends"]
        H4["sync_garmin_data"]
    end

    subgraph Meta["Checkpoint & Meta"]
        X1["propose_plan_change"]
        X2["web_search"]
        X3["recommend_products"]
        X4["send_notification"]
        X5["complete_onboarding"]
    end

    REG --> Data
    REG --> Planning
    REG --> Memory
    REG --> Health
    REG --> Meta

    style REG fill:#e94560,stroke:#fff,color:#fff
    style Data fill:#16213e,stroke:#0f3460,color:#fff
    style Planning fill:#0f3460,stroke:#533483,color:#fff
    style Memory fill:#533483,stroke:#e94560,color:#fff
    style Health fill:#22C55E,stroke:#000,color:#000
    style Meta fill:#1a1a2e,stroke:#e94560,color:#fff
```

**How tools integrate with the LLM:**
- Tools are registered as OpenAI-compatible function schemas
- LLM receives the full tool list every turn and decides which to call
- Tool execution results are appended as `tool_result` messages
- Multi-tool calls within a single LLM response are executed in parallel where possible

---

## Plan Generation & Evaluation Loop

Training plans go through a generate-evaluate-regenerate cycle until quality meets the threshold.

```mermaid
flowchart TD
    START["Agent: create_training_plan()"] --> GATHER["Gather context:<br/>profile, beliefs, activities,<br/>recovery, macrocycle week"]
    GATHER --> GEN["LLM generates plan<br/>(coach system prompt, T=0.7)"]
    GEN --> EVAL["evaluate_plan()<br/>score against agent-defined criteria"]
    EVAL --> CHECK{Score ≥ 70?}

    CHECK -->|"Yes"| SAVE["save_plan()<br/>deactivate previous, store new"]
    CHECK -->|"No"| FEEDBACK["Inject evaluation feedback<br/>as regeneration context"]
    FEEDBACK --> GEN

    FEEDBACK -->|"3rd attempt"| ACCEPT["Accept best available<br/>(with caveats noted)"]
    ACCEPT --> SAVE

    SAVE --> RESPOND["Agent presents plan<br/>to athlete via SSE"]

    style GEN fill:#e94560,stroke:#fff,color:#fff
    style EVAL fill:#533483,stroke:#e94560,color:#fff
    style SAVE fill:#3ECF8E,stroke:#fff,color:#fff
    style CHECK fill:#F59E0B,stroke:#000,color:#000
```

**Evaluation dimensions** (agent-defined at runtime via `define_eval_criteria`):
- Volume appropriateness for fitness level
- Intensity distribution (80/20 rule compliance)
- Recovery integration (rest days, easy sessions)
- Goal alignment (specificity for target event)
- Progressive overload (week-over-week progression)
- Constraint compliance (available days, max duration)

---

## Proactive Intelligence

The `HeartbeatService` runs every 30 minutes, scanning active users for conditions that warrant proactive outreach — without the athlete asking.

```mermaid
flowchart TD
    TICK["HeartbeatService._tick()<br/>every 30 min"] --> FETCH["Fetch active users<br/>(sessions.last_active ≥ 7d ago)"]
    FETCH --> EACH["For each user<br/>(semaphore: 10 concurrent)"]

    EACH --> LOCK{"Redis lock<br/>available?"}
    LOCK -->|"No (user chatting)"| SKIP["Skip user"]
    LOCK -->|"Yes"| LOAD["Load context:<br/>activities, metrics, episodes"]

    LOAD --> TRIGGERS{"Evaluate<br/>trigger rules"}

    TRIGGERS -->|"agent-defined rules<br/>(CalcEngine formulas)"| DYNAMIC["Dynamic triggers:<br/>'avg_hrv_7d < 35 AND<br/>total_sessions_7d >= 6'"]
    TRIGGERS -->|"built-in checks"| BUILTIN["Silence detection<br/>Unknown activities<br/>Goal timeline risk"]

    DYNAMIC --> QUEUE["Queue proactive message<br/>(priority: 0.0–1.0)"]
    BUILTIN --> QUEUE
    QUEUE --> DELIVER["Deliver via<br/>push / chat inbox"]

    EACH -->|"every 12 ticks (~6h)"| SELF["Self-improvement check:<br/>review metric definitions"]
    EACH -->|"every 48 ticks (~24h)"| CONSOL["Episode consolidation:<br/>weekly → monthly synthesis"]

    style TICK fill:#16213e,stroke:#0f3460,color:#fff
    style TRIGGERS fill:#e94560,stroke:#fff,color:#fff
    style DYNAMIC fill:#533483,stroke:#e94560,color:#fff
    style QUEUE fill:#F59E0B,stroke:#000,color:#000
    style DELIVER fill:#22C55E,stroke:#fff,color:#fff
```

**Trigger examples:**

| Trigger | Condition | Priority |
|---|---|---|
| High fatigue warning | `avg_hrv_7d < 35 AND total_sessions_7d >= 6` | HIGH |
| Goal at risk | Projected finish time > target by >5% | HIGH |
| Missed session pattern | 3+ consecutive planned sessions skipped | MEDIUM |
| Fitness improving | New personal best detected | LOW |
| Silence | No interaction for 5+ days | MEDIUM |

---

## Onboarding Pipeline

Voice-first onboarding: the client captures speech, sends the transcript, and the backend extracts structured data — then bootstraps the entire coaching system.

```mermaid
sequenceDiagram
    participant App as Mobile App
    participant API as FastAPI
    participant LLM as Gemini 2.0 Flash
    participant Agent as Onboarding Agent
    participant DB as Supabase

    Note over App,API: Phase 1: Voice Extraction (no auth required)
    App->>+API: POST /api/onboarding/parse-voice<br/>{text: "Ich laufe und fahre Rad", step: "sport"}
    API->>+LLM: Extract sports from German transcript
    LLM-->>-API: {items: ["Laufen", "Radfahren"]}
    API-->>-App: ParseVoiceResponse

    App->>+API: POST /api/onboarding/parse-voice<br/>{text: "Halbmarathon unter 1:30 im Sept", step: "goals"}
    API->>+LLM: Extract goals + structured data
    LLM-->>-API: {items: ["Halbmarathon"], structured: {event, date, target_time}}
    API-->>-App: ParseVoiceResponse

    Note over App,DB: Phase 2: Profile Setup (auth required)
    App->>+API: POST /api/onboarding/setup<br/>{sports, goals, available_days, wearable}
    API->>DB: UPDATE profiles SET sports, goal, meta
    API-->>App: {status: "ok"}

    Note over Agent,DB: Phase 3: Agent Bootstrap (async background)
    API->>+Agent: _trigger_onboarding_agent(user_id)
    deactivate API
    Agent->>DB: Read profile + beliefs
    Agent->>Agent: define_session_schema (per sport)
    Agent->>Agent: define_metric (TRIMP, pace zones, HR zones)
    Agent->>Agent: define_eval_criteria (plan quality rules)
    Agent->>Agent: define_periodization (phase structure)
    Agent->>Agent: define_trigger_rules (proactive conditions)
    Agent->>Agent: create_macrocycle (if goal ≥ 8 weeks)
    Agent->>Agent: create_training_plan → evaluate_plan
    Agent->>Agent: save_plan
    Agent->>Agent: recommend_products
    Agent->>Agent: complete_onboarding()
    deactivate Agent
```

---

## Data Architecture

```mermaid
erDiagram
    profiles ||--o{ beliefs : "has beliefs"
    profiles ||--o{ sessions : "has sessions"
    profiles ||--o{ activities : "logs activities"
    profiles ||--o{ plans : "has plans"
    profiles ||--o{ episodes : "has episodes"
    profiles ||--o{ health_daily_metrics : "tracks recovery"
    profiles ||--o{ provider_tokens : "connected wearables"
    profiles ||--o{ pending_actions : "awaiting confirmation"
    profiles ||--o{ proactive_queue : "queued messages"

    profiles {
        uuid user_id PK
        text name
        jsonb sports
        jsonb goal "event, date, target_time"
        jsonb fitness "vo2max, threshold_pace, trend"
        jsonb constraints "days_per_week, max_minutes"
        jsonb meta "wearable, available_days"
    }

    beliefs {
        uuid id PK
        uuid user_id FK
        text text
        text category "preference|constraint|fitness|..."
        float confidence "0.0 - 1.0"
        vector embedding "768-dim pgvector"
        jsonb outcome_history
        text stability "stable|evolving|transient"
        bool active
    }

    sessions {
        uuid id PK
        uuid user_id FK
        text context "coach|onboarding"
        text compressed_summary
        int turn_count
        int tool_calls_total
    }

    session_messages {
        uuid id PK
        uuid session_id FK
        text role "user|model|tool_call|tool_result"
        text content "max 8KB"
        jsonb meta
    }

    activities {
        uuid id PK
        uuid user_id FK
        text sport
        int duration_seconds
        float distance_meters
        int avg_hr
        float trimp
        text source "manual|garmin|webhook"
    }

    plans {
        uuid id PK
        uuid user_id FK
        jsonb plan_data "weeks, sessions, steps"
        int evaluation_score "0-100"
        text evaluation_feedback
        bool active
    }

    health_daily_metrics {
        uuid id PK
        uuid user_id FK
        date metric_date
        float sleep_seconds
        float hrv
        float stress_level
        float body_battery
        int steps
    }

    episodes {
        uuid id PK
        uuid user_id FK
        text episode_type "weekly|monthly|quarterly"
        text summary
        jsonb insights
    }

    sessions ||--o{ session_messages : "contains"
```

**Key data patterns:**
- **Row-Level Security (RLS)** on every table — each user only sees their own data
- **pgvector** for belief similarity search via `match_beliefs()` PostgreSQL RPC
- **Immutable writes** — updates return new objects, never mutate in place
- **Import deduplication** via SHA-256 file hashing in `import_manifest`
- **Partial unique indexes** prevent duplicate pending actions and proactive messages

---

## Concurrency & Resilience

```mermaid
flowchart LR
    subgraph Request["Incoming Request"]
        REQ["POST /chat"]
    end

    subgraph Locking["Distributed Locking"]
        REDIS_LOCK["Redis: SET NX EX 300<br/>key: agent_loop:lock:{user_id}"]
        FALLBACK["In-process dict fallback<br/>(if Redis unavailable)"]
    end

    subgraph Processing["Agent Processing"]
        AGENT["AsyncAgentLoop"]
    end

    subgraph Budget["Budget Enforcement"]
        USAGE_CHECK["check_budget(user_id)<br/>daily_usage < 500K tokens"]
    end

    REQ --> USAGE_CHECK
    USAGE_CHECK -->|"OK"| REDIS_LOCK
    USAGE_CHECK -->|"exceeded"| REJECT_429["HTTP 429"]
    REDIS_LOCK -->|"acquired"| AGENT
    REDIS_LOCK -->|"already locked"| REJECT_CONCURRENT["HTTP 429: concurrent_request"]
    REDIS_LOCK -.->|"Redis down"| FALLBACK
    FALLBACK --> AGENT

    style REDIS_LOCK fill:#DC382D,stroke:#fff,color:#fff
    style AGENT fill:#e94560,stroke:#fff,color:#fff
    style FALLBACK fill:#F59E0B,stroke:#000,color:#000
    style REJECT_429 fill:#6B7280,stroke:#fff,color:#fff
```

**Resilience philosophy: fail-open for non-critical, fail-closed for critical.**

| Component | On Failure | Strategy |
|---|---|---|
| Redis | In-process dict lock | Graceful degradation |
| LLM (Gemini) | Try Claude Haiku fallback | `chat_completion_with_fallback()` |
| Usage tracking | Log at DEBUG, continue | Fail-open |
| Session summarizer | Skip summary, continue | Fail-open |
| JWT verification | HTTP 401 | Fail-closed |
| HMAC webhook signature | HTTP 401 | Fail-closed |

---

## Security Architecture

```mermaid
flowchart TD
    subgraph Auth["Authentication"]
        JWT["Supabase JWT<br/>ES256 (JWKS) + HS256 (legacy)"]
        HMAC["Webhook HMAC-SHA256<br/>+ timestamp replay protection"]
    end

    subgraph RateLimit["Rate Limiting"]
        IP["Per-IP: 10/min, 100/hour<br/>(slowapi + Redis)"]
        USER["Per-User: 500K tokens/day<br/>(UsageTracker)"]
        COOLDOWN["Per-Action: 15min Garmin sync<br/>(Redis TTL)"]
    end

    subgraph Isolation["Data Isolation"]
        RLS["Supabase RLS<br/>user_id column filter"]
        LOCK["Redis distributed locks<br/>one active session per user"]
    end

    subgraph Network["Network Security"]
        CORS["CORS origin whitelist"]
        PROXY["Localhost-only binding<br/>(reverse proxy expected)"]
        DOCKER["Docker: no-dev,<br/>slim base, frozen lockfile"]
    end

    Auth --> Isolation
    RateLimit --> Isolation

    style JWT fill:#3ECF8E,stroke:#fff,color:#fff
    style HMAC fill:#0f3460,stroke:#533483,color:#fff
    style RLS fill:#3ECF8E,stroke:#fff,color:#fff
    style LOCK fill:#DC382D,stroke:#fff,color:#fff
```

---

## Background Services

| Service | Interval | Purpose | Failure Mode |
|---|---|---|---|
| **HeartbeatService** | 30 min | Proactive trigger detection, episode consolidation, self-improvement | Log + continue |
| **GarminSyncService** | On demand | Garmin Connect OAuth, activity + metrics sync | HTTP error to client |
| **SessionSummarizer** | Per new session | LLM-compress previous session for context efficiency | Skip silently |
| **EpisodeConsolidation** | ~24h (via heartbeat) | Synthesize weekly reflections → monthly reviews → promote patterns to beliefs | Log + skip month |
| **UsageTracker** | Per LLM call | Token accounting, model-specific pricing, budget enforcement | Fail-open |
| **ConfigGC** | Per new session | Remove stale agent-defined configs | Log + skip |

---

## Deployment

```mermaid
flowchart LR
    subgraph Dev["Development"]
        CODE["git push main"]
    end

    subgraph CI["GitHub Actions"]
        TRIGGER["workflow_dispatch<br/>or push to main"]
        SSH["SSH to Hetzner<br/>(appleboy/ssh-action)"]
    end

    subgraph Server["Hetzner VPS"]
        PULL["git pull origin main"]
        BUILD["docker compose build<br/>--no-cache api"]
        UP["docker compose up -d"]
        PRUNE["docker image prune -f"]
    end

    subgraph Stack["Runtime Stack"]
        NGINX["Nginx<br/>Reverse Proxy"]
        API_C["API Container<br/>Python 3.12-slim + uv"]
        REDIS_C["Redis 7 Alpine<br/>AOF persistence"]
    end

    subgraph External["External Services"]
        SUPA_E["Supabase<br/>(managed PostgreSQL)"]
        UPSTASH["Upstash Redis<br/>(production cache)"]
    end

    CODE --> TRIGGER
    TRIGGER --> SSH
    SSH --> PULL --> BUILD --> UP --> PRUNE

    NGINX -->|":8000"| API_C
    API_C --> REDIS_C
    API_C --> SUPA_E
    API_C --> UPSTASH

    style TRIGGER fill:#16213e,stroke:#0f3460,color:#fff
    style API_C fill:#009688,stroke:#fff,color:#fff
    style REDIS_C fill:#DC382D,stroke:#fff,color:#fff
    style SUPA_E fill:#3ECF8E,stroke:#fff,color:#fff
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Runtime** | Python 3.12+, uv | Language + package management |
| **API** | FastAPI + Uvicorn | ASGI server, async-native |
| **Streaming** | SSE (sse-starlette) | Real-time event streaming |
| **LLM Gateway** | LiteLLM | Provider-agnostic LLM calls (Gemini, Claude, OpenAI) |
| **Primary Model** | Gemini 2.5 Flash | Coaching agent, plan generation |
| **Embeddings** | Gemini text-embedding-004 | 768-dim belief similarity search |
| **Database** | Supabase (PostgreSQL + pgvector) | Persistence, RLS, vector search |
| **Concurrency** | Redis 7 | Distributed locks, cooldowns, confirmations |
| **Auth** | Supabase JWT (ES256/HS256) | User authentication |
| **Rate Limiting** | slowapi + Redis | Per-IP and per-user throttling |
| **Wearables** | garminconnect (Garth) | Garmin Connect OAuth + data sync |
| **Formula Engine** | evalidate (CalcEngine) | Sandboxed math expression evaluation |
| **Search** | BM25 (bm25s) + pgvector cosine | Hybrid belief retrieval |
| **Containerization** | Docker + docker-compose | Reproducible deployment |
| **CI/CD** | GitHub Actions | SSH-based deploy to Hetzner |

---

## Project Structure

```
src/
├── api/                          # API Gateway
│   ├── main.py                  #   App factory, CORS, rate limiting, lifespan
│   ├── auth.py                  #   Supabase JWT (ES256 + HS256)
│   ├── rate_limiter.py          #   slowapi + Redis/in-memory
│   ├── sse.py                   #   SSE event helpers
│   └── routers/
│       ├── chat.py              #   POST /chat (SSE), POST /chat/confirm
│       ├── onboarding.py        #   POST /parse-voice, POST /setup
│       ├── webhook.py           #   POST /webhook/activity (HMAC)
│       └── garmin.py            #   Garmin Connect OAuth + sync
│
├── agent/                        # Cognitive Engine
│   ├── agent_loop.py            #   Core agentic loop (Claude Code pattern)
│   ├── llm.py                   #   LiteLLM wrapper + fallback chain
│   ├── system_prompt.py         #   Static prompt + runtime context builder
│   ├── coach.py                 #   Plan generation (coach system prompt)
│   ├── plan_evaluator.py        #   Plan scoring (agent-defined criteria)
│   ├── assessment.py            #   Training assessment (plan vs actual)
│   ├── reflection.py            #   Episodic reflections + meta-belief extraction
│   ├── proactive.py             #   Trigger detection engine
│   ├── dynamic_triggers.py      #   CalcEngine-based trigger rules
│   └── tools/                   #   23 Tool Modules
│       ├── registry.py          #     Registration + OpenAI schema generation
│       ├── data_tools.py        #     Profile, activities, plans, beliefs
│       ├── analysis_tools.py    #     Training load, plan adherence
│       ├── planning_tools.py    #     Plan creation, evaluation, macrocycle
│       ├── memory_tools.py      #     Belief management, profile updates
│       ├── config_tools.py      #     Runtime metric/criteria definitions
│       ├── health_tools.py      #     Daily metrics, health inventory
│       ├── checkpoint_tools.py  #     Async user confirmation flow
│       └── ...                  #     research, garmin, product, notification
│
├── db/                           # Data Access Layer (19 modules)
│   ├── client.py                #   Supabase singleton (sync + async)
│   ├── user_model_db.py         #   Profiles + beliefs (pgvector)
│   ├── session_store_db.py      #   Sessions + message history
│   ├── activity_store_db.py     #   Activities + FIT import manifest
│   ├── plans_db.py              #   Training plans + evaluation scores
│   ├── health_data_db.py        #   Garmin/Apple Health/Health Connect
│   ├── agent_config_db.py       #   Runtime-defined configs
│   └── ...                      #   episodes, proactive_queue, provider_tokens
│
├── services/                     # Background Workers
│   ├── heartbeat.py             #   30-min proactive trigger loop
│   ├── garmin_sync.py           #   Garmin Connect OAuth + data sync
│   ├── session_summarizer.py    #   LLM session compression
│   ├── episode_consolidation.py #   Weekly → monthly synthesis
│   ├── usage_tracker.py         #   Token budget enforcement
│   └── health_context.py        #   Recovery context builder
│
├── calc/
│   └── engine.py                #   evalidate expression sandbox
│
├── memory/                       # User Model Abstractions
│   ├── user_model.py            #   Structured core + belief interface
│   └── episodes.py              #   Episode storage helpers
│
└── config.py                     # Pydantic Settings v2
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Code computes, LLM reasons** | Agent defines formulas via tools, CalcEngine evaluates them safely. No hardcoded sport logic, no hallucinated math. |
| **Single agent, not a swarm** | One coach who knows you well > five generic assistants. Specialist sub-agents spawn only for focused analysis. |
| **23 tools, no router** | LLM autonomously selects the right tools each turn. No hardcoded orchestration. |
| **Static + runtime prompt split** | Static prompt is LLM-cacheable (cost reduction). Runtime context injected fresh per request. |
| **Belief-driven memory** | Confidence decays on contradiction, strengthens on confirmation. Not just key-value storage. |
| **LiteLLM over direct SDK** | Provider-agnostic. Switch from Gemini to Claude by changing one env var. |
| **Supabase + RLS** | Multi-user isolation at database layer. Service role for backend ops. |
| **Redis with in-process fallback** | Distributed locks when available, graceful degradation when not. |
| **SSE over WebSockets** | Simpler protocol, unidirectional streaming, better proxy compatibility. |
| **Fail-open for non-critical** | Usage tracking, summarization, consolidation — never block the chat. |

---

## License

MIT — see [LICENSE](LICENSE).
