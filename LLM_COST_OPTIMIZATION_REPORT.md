# AgenticSports -- LLM-Kostenoptimierung: Strategiebericht

> **Datum:** 17. Februar 2026
> **Autor:** Lead Product Developer (synthetisiert aus 5 Spezialisten-Research-Reports)
> **Zielgruppe:** Founder / CTO
> **Status:** Entscheidungsvorlage

---

## Executive Summary

AgenticSports kostet heute ~**$0,01 pro Nachricht** mit Gemini 2.5 Flash. Bei 1.000 Nutzern (50K Nachrichten/Monat) ergibt das **~$819/Monat LLM-Kosten** -- bei geplanten Einnahmen von ~$600/Monat (40x Athlete + 10x Pro). Das Modell ist **nicht profitabel**, weil 950 Free-Tier-Nutzer $570/Monat an LLM-Kosten erzeugen. Die gute Nachricht: Mit 7 konkreten Optimierungen lassen sich die Kosten um **60-75% senken** und die Coaching-Qualitaet bei komplexen Anfragen gleichzeitig steigern. Die drei wichtigsten Hebel sind: (1) Google Context Caching fuer den System-Prompt ($215/Monat Ersparnis, 1 Tag Aufwand), (2) Model-Tiering -- Free-Tier auf Gemini Flash Lite ($475/Monat Ersparnis, 1 Tag), und (3) Tool-Result Truncation + Observation Masking ($126/Monat, 2 Tage). Zusammen bringen diese drei Massnahmen das Unternehmen in die Profitabilitaet.

---

## 1. Ist-Zustand: Kostenanalyse

### Aktuelle Architektur

```
Jede Nachricht --> agent_loop.py --> Gemini 2.5 Flash ($0.30/$2.50 per M tokens)
                   |-- System-Prompt: ~2.800 Tokens (rebuild pro Request)
                   |-- Tool Declarations: ~2.500 Tokens (14 Tools, immer alle gesendet)
                   |-- 2-4 LLM-Calls pro Nachricht (Tool-Rounds)
                   |-- Kein Caching, kein Routing, kein Truncation
```

### Kosten pro Nachricht (Ist-Zustand)

| Komponente | Input Tokens | Output Tokens | Kosten |
|---|---|---|---|
| LLM-Call 1 (System + History + Tools) | ~5.000 | ~800 | $0,0035 |
| LLM-Call 2 (Tool Results + Antwort) | ~6.000 | ~1.000 | $0,0043 |
| LLM-Call 3 (40% der Nachrichten) | ~7.000 | ~800 | $0,0041 |
| **Gewichteter Durchschnitt** | **~15.000** | **~2.200** | **~$0,010** |

### Token-Budget bei 50K Nachrichten/Monat

| Komponente | Tokens/Call | Calls/Msg | Monatlich |
|---|---|---|---|
| System-Prompt | 2.800 | 3 avg | 420M |
| Tool Declarations | 2.500 | 3 | 375M |
| Conversation History | 5.000 avg | 3 | 750M |
| User Message | 50 | 1 | 2,5M |
| Tool Results (kumuliert) | 2.400 avg | 2 | 240M |
| **Total Input** | | | **~2.270M** |
| Model Output (Tool Calls) | 200 | 2 | 20M |
| Model Output (Antwort) | 700 | 1 | 35M |
| **Total Output** | | | **~55M** |

### Monatliche Kosten bei Scale

| Nutzer | Nachrichten/Monat | Input (M) | Output (M) | **LLM-Kosten** |
|---|---|---|---|---|
| 100 | 5.000 | 227M | 5,5M | **$82** |
| 1.000 | 50.000 | 2.270M | 55M | **$819** |
| 10.000 | 500.000 | 22.700M | 550M | **$8.190** |

### Problemfelder

1. **Free-Tier ist der Kostentreiber:** 950 Free-User x $0,60/Monat = **$570** -- das sind 70% der gesamten LLM-Kosten.
2. **System-Prompt wird redundant neu verarbeitet:** 2.800 Tokens x 3 Calls x 50K Msgs = 420M Tokens/Monat nur fuer den Prompt.
3. **Tool Results sind untrunkiert:** Rohe JSON-Antworten (800-2.000 Tokens pro Tool-Call) werden vollstaendig in den Kontext gestopft.
4. **Kein Model-Routing:** Eine einfache Begruessungsnachricht kostet genauso viel wie eine komplexe Trainingsplan-Erstellung.
5. **Revenue Gap:** $819 LLM-Kosten vs. $600 Revenue = **-$219/Monat Verlust** bei 1.000 Nutzern.

---

## 2. Die 7 Optimierungs-Strategien (nach Impact/Aufwand sortiert)

---

### Strategie 1: Google Context Caching (P0 -- sofort)

**Was es ist:** Google bietet fuer Gemini 2.5 Modelle explizites Context Caching an. Der System-Prompt (~2.800 Tokens) und die Tool Declarations (~2.500 Tokens) werden einmal pro Session gecacht. Alle folgenden LLM-Calls in derselben Session lesen den gecachten Prefix mit **90% Rabatt** (statt $0,30/M nur $0,03/M fuer gecachte Tokens). AgenticSports macht durchschnittlich 3 Calls pro Nachricht -- der Cache amortisiert sich ab dem 2. Call.

**Ersparnis:**
- 795M Input-Tokens/Monat werden zu Cached Reads
- Von $238,50 auf $23,85 fuer den Prefix-Anteil
- **$214,62/Monat Ersparnis (26% der Gesamtkosten)**
- Storage-Kosten: $0,03/Monat (vernachlaessigbar)

**Aufwand:** 1 Tag (~20 LOC)

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/agent/agent_loop.py` -- Cache-Erstellung und -Nutzung

**Code-Beispiel:**

```python
from google import genai

class AgentLoop:
    def __init__(self, ...):
        self._cache = None

    def _ensure_cache(self):
        if self._cache is None:
            system_prompt = build_system_prompt(self.user_model, ...)
            tool_declarations = self.tools.get_declarations()
            self._cache = self.client.caches.create(
                model=MODEL,
                config={
                    "system_instruction": system_prompt,
                    "tools": [genai.types.Tool(
                        function_declarations=tool_declarations
                    )],
                    "ttl": "3600s",  # 1 Stunde
                },
            )
        return self._cache

    # Im generate_content-Call:
    response = self.client.models.generate_content(
        model=MODEL,
        contents=self._messages,
        config=genai.types.GenerateContentConfig(
            cached_content=self._cache.name,  # <-- NEU
            temperature=AGENT_TEMPERATURE,
        ),
    )
```

**Risiken:**
- Cache wird invalide, wenn sich der System-Prompt aendert (z.B. neuer Belief). Loesung: Cache pro Session erstellen, bei User-Model-Aenderung neu erstellen.
- Dynamic System Prompts (Strategie 4) sind **inkompatibel** mit Context Caching -- man muss sich fuer eines entscheiden. **Empfehlung: Context Caching gewinnt klar** (90% Rabatt auf 5.300 Tokens > manuelle Prompt-Kuerzung um 800 Tokens).
- Minimum 2.048 Tokens fuer Caching erforderlich. AgenticSports hat ~5.300 -- kein Problem.

**WICHTIG:** Diese Strategie ist **die einzelne wirkungsvollste Massnahme**. Sofort implementieren.

---

### Strategie 2: Tool-Result Truncation + Observation Masking (P0)

**Was es ist:** Zwei verwandte Optimierungen fuer den Konversations-Kontext:

**(a) Tool-Result Truncation:** Tool-Ergebnisse werden vor der Rueckgabe an den Kontext beschnitten. Null-Werte, leere Arrays und Metadaten-Felder werden entfernt. Fuer bestimmte Tools gibt es spezifische Komprimierung (z.B. `get_activities` liefert nur essentielle Felder, `get_current_plan` liefert Session-Summaries statt voller Struktur).

**(b) Observation Masking:** Basierend auf JetBrains-Research (NeurIPS 2025): In aelteren Konversationsrunden werden Tool-Ergebnisse durch einzeilige Zusammenfassungen ersetzt (`[get_activities: ok]`), waehrend User-Messages und Model-Reasoning erhalten bleiben. Dies halbiert die History-Kosten bei gleicher Agent-Performance.

**Ersparnis:**
- Tool-Result Truncation: 120M Input-Tokens/Monat = **$36/Monat (4,4%)**
- Observation Masking: 300M Input-Tokens/Monat = **$90/Monat (11%)**
- **Kombiniert: $126/Monat (15,4%)**

**Aufwand:** 2 Tage (~90 LOC)

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/agent/agent_loop.py` -- `_compress_history()` ersetzen + Truncation-Logik

**Code-Beispiel (Truncation):**

```python
def _truncate_tool_result(self, tool_name: str, result: dict) -> dict:
    # Null-Werte entfernen
    result = {k: v for k, v in result.items() if v is not None}

    if tool_name == "get_activities":
        for act in result.get("activities", []):
            for key in ["hr_zones", "calories", "sub_sport"]:
                act.pop(key, None)

    elif tool_name == "get_current_plan":
        plan = result.get("plan", {})
        result["plan"] = {
            "sessions_count": len(plan.get("sessions", [])),
            "session_summaries": [
                f"{s['day']}: {s['sport']} - {s['type']} ({s.get('total_duration_minutes', '?')}min)"
                for s in plan.get("sessions", [])
            ],
        }
    return result
```

**Code-Beispiel (Observation Masking):**

```python
def _compress_history(self):
    if len(self._messages) <= COMPRESSION_THRESHOLD:
        return
    keep_from = self._find_keep_boundary()
    masked = []
    for msg in self._messages[:keep_from]:
        if msg.role == "user" and self._has_function_response(msg):
            summaries = []
            for part in msg.parts:
                if hasattr(part, 'function_response') and part.function_response:
                    summaries.append(f"[{part.function_response.name}: ok]")
            masked.append(Content(role="user", parts=[Part(text=" | ".join(summaries))]))
        else:
            masked.append(msg)
    self._messages = masked + self._messages[keep_from:]
```

**Risiken:**
- Zu aggressive Truncation kann dem Modell wichtige Daten entziehen. Loesung: Token-Budgets pro Tool definieren (z.B. `get_activities`: max 800 Tokens, `create_training_plan`: max 1.500 Tokens).
- Observation Masking in den letzten 4 Runden NICHT anwenden -- nur auf aeltere History.

---

### Strategie 3: Tiered Model Routing via LiteLLM (P1)

**Was es ist:** Nicht jede Nachricht braucht Gemini 2.5 Flash. Ein Hybrid-Classifier (Regex-Regeln + guenstiger LLM-Fallback) kategorisiert eingehende Nachrichten in drei Tiers und routet sie zum passenden Modell. Der groesste Hebel ist **Model-Downgrade fuer Free-Tier-Nutzer** auf Gemini Flash Lite ($0,10/$0,40 statt $0,30/$2,50).

| Tier | Modell | Input/Output pro M | Anteil | Beispiel |
|---|---|---|---|---|
| T1 Simple | Gemini 2.0 Flash | $0,10 / $0,40 | ~40% | "Was ist mein naechstes Training?" |
| T2 Coaching | Gemini 2.5 Flash | $0,30 / $2,50 | ~35% | "Wie war meine Erholung?" |
| T3 Complex | Gemini 2.5 Flash (spaeter: Claude Sonnet 4.5) | $0,30 / $2,50 | ~25% | "Erstelle 12-Wochen-Marathonplan" |

**Zusaetzlich: Free-Tier auf Flash Lite** ($0,10/$0,40) fuer ALLE Anfragen:
- Aktuelle Free-Tier-Kosten: 950 User x $0,60 = **$570/Monat**
- Mit Flash Lite: 950 User x $0,10 = **$95/Monat**
- **Ersparnis: $475/Monat allein durch Free-Tier-Downgrade**

**Gesamt-Ersparnis:** $475 (Free-Tier-Downgrade) + ~$30 (T1-Routing fuer Paid-User) = **~$505/Monat**

**Aufwand:** 2-3 Tage (Phase 1: Classifier + Multi-Model in google-genai). Kein SDK-Wechsel noetig.

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/agent/llm.py` -- Multi-Model-Definitionen
- `/Users/roman/Development/AgenticSports/src/agent/agent_loop.py` -- Routing vor generate_content
- **Neue Datei:** `/Users/roman/Development/AgenticSports/src/agent/router.py` -- Hybrid-Classifier

**Code-Beispiel (router.py):**

```python
import re

MODELS = {
    "T1_SIMPLE":   "gemini-2.0-flash",         # $0.10/$0.40
    "T2_COACHING": "gemini-2.5-flash",          # $0.30/$2.50
    "T3_COMPLEX":  "gemini-2.5-flash",          # spaeter Claude Sonnet 4.5
}

def classify_request(message: str, tier: str = "paid") -> str:
    if tier == "free":
        return "T1_SIMPLE"  # Free-Tier immer guenstigstes Modell

    msg_lower = message.lower()
    # T3: Komplexe Planungs-Signale
    complex_signals = [
        r"\b(\d+)[\s-]*(week|wochen|month|monat)",
        r"periodiz", r"training\s*plan", r"trainingsplan",
        r"race\s*strategy", r"wettkampf.*strategie", r"taper",
    ]
    if any(re.search(p, msg_lower) for p in complex_signals):
        return "T3_COMPLEX"

    # T1: Einfache Lookups
    simple_signals = [
        r"(what|show|tell).*next\s*(workout|session)",
        r"(was|naechst).*training",
        r"^(hi|hello|hallo|hey|moin|servus)\b",
    ]
    if any(re.search(p, msg_lower) for p in simple_signals):
        return "T1_SIMPLE"

    return "T2_COACHING"  # Default
```

**Risiken:**
- Regelbasierter Classifier ist brittle -- "Wie war meine Erholung nach dem harten Intervalltraining und ich habe in 3 Tagen ein Rennen?" wird als T2 klassifiziert, braucht aber moeglicherweise T3-Qualitaet.
- Loesung: Hybrid-Ansatz mit LLM-Fallback fuer ambigue Nachrichten (Gemini Flash Lite Classifier-Call kostet $0,000012 pro Anfrage = $0,36/Monat bei 1.000 Msgs/Tag).
- T3 mit Claude Sonnet 4.5 erfordert spaeter einen SDK-Wechsel oder LiteLLM-Migration.

---

### Strategie 4: Prompt/Token-Optimierung (P1)

**Was es ist:** Mehrere kleinere Optimierungen, die zusammen signifikant wirken:

**(a) max_tokens Tuning:** Verschiedene Output-Limits je nach Anfrage-Typ setzen (Greeting: 300, Quick Answer: 500, Plan: 1.200, Tool-Call-Only: 200). Spart 10-15% Output-Tokens.

**(b) RAG fuer Beliefs:** Statt alle aktiven Beliefs zu laden, nur die top-10 relevantesten per Embedding-Similarity liefern. Spart ~1.500 Tokens pro `get_beliefs()`-Call bei 100+ Beliefs.

**(c) Tool Description Compression:** Aktuelle Beschreibungen sind verbose (~120 Tokens pro Tool). Komprimierte Versionen (~35 Tokens) sparen ~800 Tokens pro Call. **Aber:** Nur sinnvoll ohne Context Caching (da bei Caching die Descriptions im Cache liegen und der 90%-Rabatt gilt).

**Ersparnis:**
- max_tokens: **$21/Monat**
- RAG fuer Beliefs: **$23/Monat**
- Tool Description Compression: ~$36/Monat (aber nur relevant wenn kein Context Caching)
- **Kombiniert mit Context Caching: ~$44/Monat**

**Aufwand:** 1-2 Tage (~35 LOC)

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/agent/agent_loop.py` -- max_tokens Config
- `/Users/roman/Development/AgenticSports/src/agent/tools/data_tools.py` -- RAG-basierte Belief-Abfrage

**Code-Beispiel (max_tokens):**

```python
def _get_max_tokens(self, round_num: int, intent: str) -> int:
    if round_num > 0:  # Tool-Call-Round
        return 200
    return {"greeting": 300, "data_query": 500, "plan_request": 1200}.get(intent, 800)
```

**Risiken:** Zu restriktive max_tokens koennen Antworten abschneiden. Grosszuegig bemessen und per Monitoring nachjustieren.

---

### Strategie 5: Lokale Embeddings (BGE-M3) (P1)

**Was es ist:** AgenticSports nutzt aktuell `text-embedding-004` (Google API) fuer Belief-Similarity-Search. Der Wechsel auf das Open-Source-Modell **BGE-M3** (lokal via `sentence-transformers`) eliminiert alle Embedding-API-Kosten und funktioniert offline. BGE-M3 unterstuetzt 100+ Sprachen (inkl. Deutsch), hat 1024 Dimensionen und einen MTEB-Score von 63.0 (vs. 64.6 fuer OpenAI text-3-large).

**Ersparnis:**
- Aktuell: ~$0,01/1K Zeichen fuer Embeddings
- Neu: $0 (lokal)
- Bei Scale: **$10-30/Monat** (waechst mit Belief-Anzahl und User-Zahl)
- Wichtiger: **Keine Latenz-Abhaengigkeit** von externen APIs fuer Similarity-Search

**Aufwand:** 1 Tag (~20 LOC)

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/memory/user_model.py` -- `embed_belief()` und `find_similar_beliefs()` ersetzen

**Code-Beispiel:**

```python
# Vorher (API-Call):
response = client.models.embed_content(model=EMBEDDING_MODEL, content=text)
embedding = response.embedding

# Nachher (lokal):
from sentence_transformers import SentenceTransformer
_embed_model = SentenceTransformer('BAAI/bge-m3')

def embed_belief(text: str) -> list[float]:
    return _embed_model.encode(text).tolist()
```

**Risiken:**
- Erstes Laden des Modells dauert ~5 Sekunden und benoetigt ~1,5 GB RAM. Loesung: Einmal beim Server-Start laden.
- Auf dem Server (FastAPI) kein Problem; auf Mobile (React Native) nicht direkt nutzbar -- dort weiterhin API-Embeddings oder On-Device-Alternative (Phase 2).
- Bestehende Belief-Embeddings muessen einmalig neu berechnet werden (Migration).

---

### Strategie 6: Cost Control & Rate Limiting (P2)

**Was es ist:** Ein Dual-Layer-System aus sichtbaren Message-Limits (fuer User verstaendlich) und unsichtbaren Token-Budgets (fuer Kostencontrolle):

**Layer 1: Message Limits (User-facing)**

| Tier | Nachrichten/Tag | Nachrichten/Stunde | Max Tool Rounds |
|---|---|---|---|
| Free | 5 | 3 | 5 |
| Athlete ($9,99) | 50 | 20 | 15 |
| Pro ($19,99) | Unbegrenzt | 60 | 25 |

**Layer 2: Token Budgets (intern)**

| Tier | Monatliches Input Budget | Monatliches Output Budget | Kosten-Cap |
|---|---|---|---|
| Free | 1.200.000 | 200.000 | ~$0,86 |
| Athlete | 9.000.000 | 1.500.000 | ~$6,45 |
| Pro | 30.000.000 | 5.000.000 | ~$21,50 |

**Graceful Degradation statt Hard Cutoff:**

| Budget verbraucht | Aktion |
|---|---|
| 0-70% | Volle Qualitaet |
| 70-90% | Tool Rounds von 25 auf 10 reduzieren, Plan-Evaluation ueberspringen |
| 90-100% | Model-Downgrade auf Flash Lite |
| 100%+ | Nur Text-Antworten (keine Tool Calls), Upgrade-Hinweis |
| 120%+ | Tageslimit erreicht |

**Monitoring mit Helicone:** Ein-Zeilen-URL-Aenderung in `llm.py`, Free-Tier bis 10K Requests/Monat. Bietet Token-Tracking, Cost-Dashboards, Alerting.

**Ersparnis:** Direkte Ersparnis schwer zu beziffern, aber verhindert Kostenueberraschungen. Alert-Schwellen: >$5/Tag Warning, >$15/Tag Auto-Caching, einzelner User >$2/Tag = Abuse-Flag.

**Aufwand:** 2-3 Tage

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/agent/llm.py` -- Helicone-Proxy
- `/Users/roman/Development/AgenticSports/src/agent/agent_loop.py` -- Usage-Tracking aus `response.usage_metadata`
- **Neue Datei:** Usage-Meter / Budget-Enforcement-Modul (FastAPI-Backend)

**Code-Beispiel (Usage Tracking):**

```python
# Nach jedem generate_content-Call:
usage = response.usage_metadata
self.usage_meter.record(
    user_id=self.user_id,
    input_tokens=usage.prompt_token_count,
    output_tokens=usage.candidates_token_count,
    cached_tokens=getattr(usage, 'cached_content_token_count', 0),
)
```

**Risiken:** Zu aggressive Limits fuehren zu User-Frustration. Empfehlung: Grosszuegig starten, per Monitoring nachjustieren. Ein Athlet mitten in einem Gespraech ueber eine Verletzungswarnung darf nicht abrupt abgeschnitten werden.

---

### Strategie 7: Fine-Tuning fuer Coaching-Domaene (P3)

**Was es ist:** Ein kleineres Open-Source-Modell (Qwen3-32B oder Llama 3.1 8B) wird per QLoRA auf coaching-spezifische Daten fein-abgestimmt. Das Ergebnis: ein Modell, das deutsche Coaching-Terminologie (Periodisierung, Herzfrequenz-Zonen, Trainingsbelastung) nativ versteht, weniger Tokens fuer die gleiche Qualitaet braucht, und auf Hosted-Plattformen (Together.ai, Groq) fuer $0,20-0,90/M Tokens laeuft.

**Zwei Pfade:**

**(a) Subtask Fine-Tuning (empfohlen fuer jetzt):**

| Subtask | Aktuell | Fine-Tuned Alternative | Ersparnis/Call |
|---|---|---|---|
| Fatigue Detection | LLM-Call ($0,005-0,01) | Fine-tuned 3B Classifier ($0,0001) | 98% |
| Plan Evaluation Scoring | LLM-Call ($0,005-0,01) | Fine-tuned 3B Scorer ($0,0001) | 98% |
| Belief Extraction | Im Agent Loop | Fine-tuned NER (1B, $0,0001) | 95% |

**(b) Full Model Replacement (spaeter, ab 2.500+ Users):**

| | Gemini Flash (API) | Self-Hosted Fine-Tuned 8B |
|---|---|---|
| Pro-Nachricht-Kosten | $0,010 | ~$0,002 |
| Monatliches Hosting (GPU) | $0 | $300-500 |
| Fine-Tuning (einmalig) | $0 | $50-200 (QLoRA) |
| Break-Even | -- | **~167 DAU (~500+ Total Users)** |

**Ersparnis:**
- Subtask Fine-Tuning: **$50-100/Monat** bei 1.000 Nutzern
- Full Replacement: **$300-400/Monat** ab 2.500+ Nutzern

**Aufwand:**
- Subtask Fine-Tuning: 2-4 Wochen (inkl. Datenerstellung)
- Full Replacement: 2-3 Monate (Datenpipeline, Training, Evaluation, Deployment)

**Betroffene Dateien:**
- `/Users/roman/Development/AgenticSports/src/agent/proactive.py` -- `_detect_fatigue_llm()` durch Classifier ersetzen
- `/Users/roman/Development/AgenticSports/src/agent/plan_evaluator.py` -- Scoring durch Fine-Tuned Model
- `/Users/roman/Development/AgenticSports/src/memory/user_model.py` -- Belief Extraction

**Daten-Strategie:**
1. 5.000-10.000 synthetische Coaching-Gespraeche mit Gemini Flash generieren (~$15)
2. QLoRA Fine-Tune von Qwen3-32B auf Together.ai (~$20-40)
3. Deploy auf Groq oder Together.ai ($0,20-0,30/M Tokens)
4. Kontinuierliche Verbesserung mit echten User-Gespraechen (mit Consent)

**Risiken:**
- Fine-tuned Models koennen halluzinieren, wenn Out-of-Distribution-Anfragen kommen.
- Qualitaets-Monitoring ist essentiell (LLM-as-Judge mit Claude Sonnet, ~$2/Tag fuer 50 Evaluierungen).
- Fuer das Free-Tier akzeptabel, fuer Pro-Tier sollte weiterhin Gemini 2.5 Flash als Fallback verfuegbar sein.

---

## 3. Was NICHT funktioniert (und warum)

### Semantic Response Caching -- Warum schlecht fuer AgenticSports

Semantic Caching (Antworten basierend auf semantischer Aehnlichkeit der Frage aus dem Cache liefern) wird oft mit 67% Hit-Rate und 73% Kostenersparnis beworben. Diese Zahlen stammen aus **stateless Chatbot-Workloads** (FAQ-Bots, Customer Service). AgenticSports ist fundamental anders:

1. **Personalisierung:** Jeder Athlet hat ein einzigartiges User-Model, Belief-Set und Aktivitaets-History. Es gibt keinen "Shared Cache" zwischen Usern.
2. **Zeitliche Sensitivitaet:** "Was soll ich heute trainieren?" am Montag vs. Dienstag erfordert voellig unterschiedliche Antworten -- selbst fuer den gleichen User.
3. **Tool-Abhaengigkeit:** Der Agent ruft `get_activities`, `calculate_fitness_metrics`, `get_active_beliefs` auf, bevor er antwortet. Die Antwort haengt von dynamischen Tool-Ergebnissen ab.
4. **Agent Loop:** Ein typischer Turn umfasst 3-10 Tool Calls. Caching der finalen Antwort ueberspringt die gesamte Datenerhebung, die die Antwort akkurat macht.

**Realistische Schaetzung fuer AgenticSports: 5-15% Hit Rate, 3-10% Kostenersparnis.** Bei mittlerer Implementierungskomplexitaet ist der ROI schlecht.

**Schlimmer noch: Falsch-positive Cache-Hits sind gefaehrlich.** Eine gecachte Erholungs-Bewertung von Montag ist am Mittwoch nach zwei harten Trainingseinheiten potenziell schaedlich. Im Coaching-Kontext kann ein Stale Response zu Uebertraining fuehren.

### GPTCache -- Warum nicht geeignet

GPTCache (Zilliz) cached basierend auf **Query-Text allein**, nicht auf dem vollen Kontext (System-Prompt, User-Model, Tool-Results). Zwei identische Fragen ("Erhole ich mich gut?") mit unterschiedlichem Athleten-Status erhalten die gleiche gecachte Antwort.

Zusaetzliche Probleme:
- Kein Support fuer Tool-Call-Chains -- cached nur die finale Text-Antwort
- AgenticSports nutzt google-genai SDK, GPTCache hat nur Adapter fuer OpenAI/LangChain
- Wuerde einen Custom Wrapper um `client.models.generate_content()` erfordern

### LiteLLM Semantic Cache -- Warum nicht geeignet

- Semantic Cache funktioniert nur mit Requests unter 8.191 Input-Tokens und maximal 4 Messages. AgenticSports ueberschreitet beide Limits regelmaessig.
- Cache Key enthaelt keinen User-Model-State-Hash -- zwei verschiedene User mit der gleichen Frage bekommen die gleiche gecachte Antwort.
- Erfordert Redis Stack mit RediSearch-Modul -- operativer Overhead fuer minimalen Nutzen.

### Die richtige Alternative: "Cache the Prompt, not the Response"

Statt Antworten zu cachen, wird der **Prompt-Prefix** gecacht (= Google Context Caching, Strategie 1). Das LLM generiert weiterhin eine frische Antwort basierend auf aktuellen Tool-Ergebnissen, aber die System-Prompt-Verarbeitung wird zu 90% guenstiger. **Null Risiko fuer stale Responses.**

---

## 4. Kostenprojektion: Vorher/Nachher

### Monatliche Kosten nach Nutzer-Zahl

**Ohne Optimierung (Ist-Zustand):**

| Nutzer | Free (95%) | Athlete (4%) | Pro (1%) | Monatl. LLM-Kosten | Revenue | Profit |
|---|---|---|---|---|---|---|
| 100 | $57 | $12 | $7,50 | **$77** | $60 | **-$17** |
| 1.000 | $570 | $120 | $75 | **$765** | $600 | **-$165** |
| 10.000 | $5.700 | $1.200 | $750 | **$7.650** | $6.000 | **-$1.650** |

**Mit Optimierung (alle 6 Strategien ohne Fine-Tuning):**

| Optimierung | Monatl. Ersparnis |
|---|---|
| Google Context Caching | $215 |
| Tool Truncation + Observation Masking | $126 |
| Free-Tier auf Flash Lite | $475 |
| T1-Routing fuer Paid-User | $30 |
| max_tokens + RAG Beliefs | $44 |
| **Gesamt-Ersparnis** | **$890 (~73%)** |

| Nutzer | Free (95%) | Athlete (4%) | Pro (1%) | Monatl. LLM-Kosten | Revenue | Profit |
|---|---|---|---|---|---|---|
| 100 | $9,50 | $8 | $5 | **$23** | $60 | **+$37** |
| 1.000 | $95 | $80 | $50 | **$225** | $600 | **+$375** |
| 10.000 | $950 | $800 | $500 | **$2.250** | $6.000 | **+$3.750** |

### Break-Even-Analyse

| Szenario | Break-Even Paying Users | Break-Even Total Users (5% Conversion) |
|---|---|---|
| Ohne Optimierung | ~85 zahlende User | ~1.700 |
| Mit Optimierung (P0+P1) | ~25 zahlende User | ~500 |
| Mit allen Optimierungen | ~20 zahlende User | ~400 |

**Fazit:** Mit den Optimierungen wird AgenticSports **bereits ab ~400 Total Users profitabel** (bei 5% Conversion Rate = 20 zahlende User). Ohne Optimierungen erst ab ~1.700 Users.

### Kosten pro User pro Tier (optimiert)

| Tier | Msgs/Monat | LLM-Kosten/User | Revenue/User | Brutto-Marge |
|---|---|---|---|---|
| Free | 60 | **$0,10** (Flash Lite) | $0 | -$0,10 |
| Athlete | 300 | **$2,00** (cached + truncated) | $9,99 | **80%** |
| Pro | 750 | **$5,00** (cached + truncated) | $19,99 | **75%** |

---

## 5. Empfohlene Implementierungsreihenfolge

### Phase 0: Foundation (Woche 1) -- Sofortige Wirkung

| Tag | Aufgabe | Dateien | Ersparnis |
|---|---|---|---|
| 1 | Google Context Caching implementieren | `agent_loop.py` | $215/Mo |
| 2 | Free-Tier auf Flash Lite routen | `llm.py`, neues `router.py` | $475/Mo |
| 2 | Usage-Tracking aus `response.usage_metadata` | `agent_loop.py` | Monitoring |

**Dependencies:** Keine. Sofort machbar.
**Kumulierte Ersparnis nach Woche 1: ~$690/Monat (58%)**

### Phase 1: Token-Optimierung (Woche 2-3)

| Tag | Aufgabe | Dateien | Ersparnis |
|---|---|---|---|
| 3-4 | Tool-Result Truncation | `agent_loop.py` | $36/Mo |
| 4-5 | Observation Masking (replace _compress_history) | `agent_loop.py` | $90/Mo |
| 5 | max_tokens Tuning pro Intent | `agent_loop.py` | $21/Mo |
| 6 | BGE-M3 lokale Embeddings | `user_model.py` | $10-30/Mo |
| 6 | RAG fuer Beliefs (top-10 statt alle) | `data_tools.py` | $23/Mo |

**Dependencies:** Phase 0 muss abgeschlossen sein (Context Caching diktiert, dass Dynamic Prompts und Tool Filtering NICHT implementiert werden).
**Kumulierte Ersparnis nach Phase 1: ~$890/Monat (73%)**

### Phase 2: Monetarisierung & Control (Woche 3-4)

| Tag | Aufgabe | Dateien | Ersparnis |
|---|---|---|---|
| 7-8 | Helicone-Proxy integrieren | `llm.py` | Monitoring |
| 8-9 | Rate Limiting (Message + Token Budgets) | Neues Modul (FastAPI) | Caps Worst-Case |
| 9-10 | Cost-Alerting (>$5/Tag, >$15/Tag) | Backend | Praevention |
| 10 | Graceful Degradation bei Budget-Ueberschreitung | `agent_loop.py` | Praevention |

**Dependencies:** Usage-Tracking aus Phase 0.

### Phase 3: Quality & Future (Monat 2-3)

| Woche | Aufgabe | Aufwand | Benefit |
|---|---|---|---|
| 5-6 | T2/T3-Routing fuer Paid-Tier (Hybrid Classifier) | 2-3 Tage | Qualitaet auf Complex Queries |
| 6-8 | Subtask Fine-Tuning (Fatigue, Plan Scoring) | 2-4 Wochen | $50-100/Mo |
| 8+ | Claude Sonnet 4.5 fuer T3 (Plan-Erstellung) | 1-2 Wochen | Massiver Quality-Uplift |
| 10+ | LiteLLM-Migration fuer Multi-Provider | 3-5 Tage | Infrastruktur-Flexibilitaet |

**Dependencies:** Phase 2 abgeschlossen, Produktions-Daten fuer Quality-Monitoring.

### Visualisierung der Timeline

```
Woche 1        Woche 2-3       Woche 3-4       Monat 2-3
[Phase 0]       [Phase 1]       [Phase 2]       [Phase 3]
Context Cache   Truncation      Helicone        T2/T3 Routing
Free=FlashLite  Obs. Masking    Rate Limits     Subtask FT
Usage Track     max_tokens      Alerting        Claude T3
                BGE-M3          Degradation     LiteLLM
                RAG Beliefs

Ersparnis:      Ersparnis:      Ersparnis:      Ersparnis:
$690/Mo         +$180/Mo        Praevention     +$50-100/Mo
(kumulativ)     =$890/Mo                        + Quality
```

---

## 6. Revenue-Modell & Unit Economics

### Kosten pro User pro Tier (nach Optimierung)

| | Free | Athlete ($9,99) | Pro ($19,99) |
|---|---|---|---|
| Nachrichten/Monat | 60 | 300 | 750 |
| LLM-Kosten/User | $0,10 | $2,00 | $5,00 |
| Embedding-Kosten | $0 (lokal) | $0 (lokal) | $0 (lokal) |
| Infrastruktur (anteilig) | $0,05 | $0,20 | $0,40 |
| Stripe Gebuehren | -- | $0,60 | $0,90 |
| **Total COGS/User** | **$0,15** | **$2,80** | **$6,30** |
| **Brutto-Marge** | **-$0,15** | **$7,19 (72%)** | **$13,69 (68%)** |

### Revenue bei 1.000 Nutzern (5% Conversion)

| Posten | Monatlich |
|---|---|
| 40x Athlete a $9,99 | $399,60 |
| 10x Pro a $19,99 | $199,90 |
| **Total Revenue** | **$599,50** |
| Free-Tier LLM (950 x $0,10) | -$95,00 |
| Athlete LLM (40 x $2,00) | -$80,00 |
| Pro LLM (10 x $5,00) | -$50,00 |
| Infrastruktur (Supabase, Hosting) | -$50,00 |
| Stripe (2,9% + $0,30) | -$32,00 |
| **Total COGS** | **-$307,00** |
| **Brutto-Profit** | **+$292,50 (49%)** |

### Wann lohnt sich Fine-Tuning?

| Metrik | Wert |
|---|---|
| Fine-Tuning Kosten (einmalig) | $50-200 |
| Monatliche Ersparnis (Subtask FT) | $50-100 |
| **Amortisierung** | **1-2 Monate** |
| Monatliche Ersparnis (Full Replacement) | $300-400 |
| Hosting-Kosten (GPU) | $300-500/Monat |
| **Break-Even DAU fuer Full Replacement** | **~167 DAU (~500+ Total Users)** |

**Empfehlung:**
- **Subtask Fine-Tuning (P3):** Sofort rentabel, amortisiert in 1-2 Monaten. Starten sobald Phase 1 abgeschlossen.
- **Full Model Replacement:** Erst ab 2.500+ Total Users oekonomisch sinnvoll. Nicht vor 2027 relevant.

### Langfristige Skalierung (10.000 Users)

| Posten | Ohne Optimierung | Mit Optimierung | Mit Fine-Tuning |
|---|---|---|---|
| LLM-Kosten | $7.650/Mo | $2.250/Mo | $1.500/Mo |
| Revenue | $6.000/Mo | $6.000/Mo | $6.000/Mo |
| Brutto-Profit | **-$1.650** | **+$3.750** | **+$4.500** |
| Brutto-Marge | Negativ | **63%** | **75%** |

---

## Anhang: Modell-Preisvergleich (Februar 2026)

| Modell | Provider | Input $/M | Output $/M | Bester Einsatz |
|---|---|---|---|---|
| Gemini 2.5 Flash-Lite | Google | $0,10 | $0,40 | Free-Tier, Classification |
| Gemini 2.0 Flash | Google | $0,10 | $0,40 | T1 Simple Lookups |
| DeepSeek V3 | DeepSeek | $0,14 | $0,28 | Kostenoptimal (aber GDPR-Risiko) |
| GPT-4o-mini | OpenAI | $0,15 | $0,60 | Classification, Simple Q&A |
| Qwen 3 72B | Together/Groq | $0,20-0,90 | $0,60-0,90 | Bestes Open-Source fuer Agents |
| Gemini 2.5 Flash | Google | $0,30 | $2,50 | Standard Coaching (aktuell) |
| Claude Haiku 3.5 | Anthropic | $0,80 | $4,00 | Schnelle Qualitaets-Antworten |
| Mistral Large 2 | Mistral | $2,00 | $6,00 | Bestes Deutsch (81,6% MMLU) |
| GPT-4o | OpenAI | $2,50 | $10,00 | Complex Planning |
| Claude Sonnet 4.5 | Anthropic | $3,00 | $15,00 | Deep Reasoning, T3 Planning |

---

## Anhang: Anti-Patterns -- Was man NICHT tun sollte

1. **Semantic Response Caching implementieren** -- Der ROI ist fuer personalisierte Agent-Loops minimal (5-15% Hit Rate), die Implementierung komplex, und das Risiko fuer stale Coaching-Advice hoch.

2. **Dynamic System Prompts UND Context Caching** -- Diese beiden Strategien sind inkompatibel. Context Caching gewinnt klar (90% Rabatt vs. 29% Prompt-Kuerzung).

3. **DeepSeek V3 fuer Produktion nutzen** -- Trotz der attraktiven Preise ($0,14/$0,28) ist die API in China gehostet. GDPR-Compliance fuer deutsche Nutzer ist nicht gewaehrleistet.

4. **LiteLLM-Migration als erstes** -- Der SDK-Wechsel von google-genai zu LiteLLM kostet 3-5 Tage und bringt allein keine Kostenersparnis. Erst implementieren, wenn Multi-Provider-Routing (Phase 3) tatsaechlich benoetigt wird.

5. **Unlimited fuer alle Tiers** -- Ohne Token-Budgets und Rate-Limits kann ein einzelner Power-User $50+/Monat an LLM-Kosten verursachen.

---

*Dieser Bericht basiert auf 5 spezialisierten Research-Reports (Semantic Caching, Tiered Model Routing, Prompt/Token Optimization, Cost Control & Monetization, Open-Source/Alternative LLMs) mit insgesamt 50+ akademischen und industriellen Quellen, Stand Februar 2026.*
