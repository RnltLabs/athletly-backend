# Feature 6 Design: Prompt A/B Metrics and STRICT Rule Telemetry

## Goals (from PO directive)

1. Track WHICH STRICT rules from `system_prompt.py` get violated, HOW often, under WHAT conditions.
2. Zero user-facing cost: regex-only in the prod path, no extra LLM calls.
3. Internal observability tool, accessed via `/admin/prompt-metrics`.
4. Scaffold for future A/B testing of prompt variants.

## Non-goals (this PR)

- LLM-as-judge for the "hard" rules (fabricated stats, mid-response language switch). Scaffold the data shape so it can be added later, but no detector and no offline worker in this PR.
- Multi-variant routing. Add the `PromptVariant` enum and let every record carry a `variant` tag, but the routing function returns `DEFAULT` for everyone.
- UI for the dashboard. The endpoint returns JSON; future PR can wire a frontend.

## Architecture

### Component diagram (ASCII)

```
agent_loop.py (process_message)
   |
   v final response text
   |
+--+--------------------------------+
|  prompt_metrics.scan_response()    |  <- regex detectors, in-process, sync
+--+--------------------------------+
   |
   v ViolationRecord (immutable)
   |
+--+--------------------------------+
|  PromptMetrics.record()           |  <- singleton, ring buffer, lock
+--+--------------------------------+
   |
   +-- in-memory deque(maxlen=N)     <- live stats, always on
   +-- (optional) supabase upsert    <- historical, when use_supabase=true
   +-- logger.warning if rate > 5%   <- alerting
   
admin.py
   GET /admin/prompt-metrics?window_minutes=60
       reads PromptMetrics.summary(window_seconds=...)
       returns JSON
```

### Files

| File | Status | Purpose |
|---|---|---|
| `src/services/prompt_metrics.py` | NEW | Detectors + ring buffer + singleton |
| `src/agent/agent_loop.py` | EDIT | Hook scan on final response (one place: after `result.response_text = ...`) |
| `src/api/routers/admin.py` | EDIT | Add `/admin/prompt-metrics` endpoint |
| `supabase/migrations/20260514000000_prompt_violations.sql` | NEW | Optional persistent storage table |
| `tests/test_prompt_metrics.py` | NEW | Tests for detectors + store + endpoint |

### Detector module

```python
# Module structure (pseudocode)

@dataclass(frozen=True)
class Violation:
    rule_id: str          # "no_em_dash", "no_markdown_header", etc.
    severity: str         # "strict" | "warn"
    snippet: str          # ~80 char excerpt around the match
    char_offset: int

DETECTORS: list[Detector] = [
    EmDashDetector(),
    EnDashDetector(),
    MarkdownHeaderDetector(),
    MarkdownBoldDetector(),
    MarkdownItalicDetector(),
    GermanAsciiTransliterationDetector(),
]

def scan_response(text: str, language_hint: str | None = None) -> list[Violation]:
    """Run all detectors; return found violations. Pure function."""
```

Detectors are individual classes / callables that:
1. Take `text` and optional `language_hint`.
2. Return a list of `Violation` (0 to N).
3. Are pure: no I/O, no LLM calls.

### Detection rules implemented

| Rule ID | Severity | Regex | Notes |
|---|---|---|---|
| `no_em_dash` | strict | `—` (U+2014) | Direct; Roman has banned this character. |
| `no_en_dash` | strict | `–` (U+2013) | Direct. |
| `no_markdown_header` | strict | `^#{1,6}\s` (MULTILINE) | Skips lines starting with `#` inside fenced code blocks (rare in our output but safe). |
| `no_markdown_bold` | strict | `\*\*[^*\n]+\*\*` | Slight FP risk on legit `**` strings; accepted. |
| `no_markdown_italic` | warn | `(?<![*\\])\*[^*\n]+\*(?![*])` | Lower confidence; mark as `warn`. |
| `german_ascii_transliteration` | strict | `\b(ueber|fuer|Praeferenz|Maerz|Laeufer|Groesse|ueberfaellig|grosser|stoerung|ueberraschung)\w*\b` | Only fires when `language_hint == "de"` OR when the rest of the text contains German tells (>=3 German function words from a short list). |

Hard rules deferred (commented in source):
- `no_fabricated_stats`: requires tool-trace inspection.
- `no_language_switch`: requires per-segment language detection on the response.
- `no_trend_below_5_sessions`: requires tool-trace inspection.

### Metric store

```python
@dataclass(frozen=True)
class ViolationRecord:
    ts: float
    model: str
    variant: str           # PromptVariant value
    rule_id: str
    severity: str
    response_length: int
    snippet: str

class PromptMetrics:
    """Singleton. Ring buffer + lock, mirrors CacheTelemetry."""
    def record(rec: ViolationRecord) -> None: ...
    def record_response(model, variant, text, violations) -> None:
        # Records 1 row per violation, plus tracks total response count
        # for rate calculation.
    def summary(window_seconds: int = 3600) -> dict: ...
    def violation_rate(window_seconds: int = 60) -> float: ...
```

Two counters live alongside the violation ring:
- `_response_count`: total responses scanned (for rate denominator).
- `_violation_records`: deque of `ViolationRecord` (for breakdowns).

Both are protected by the same `threading.Lock`.

### Endpoint shape

`GET /admin/prompt-metrics?window_minutes=60` returns:

```json
{
  "window_minutes": 60,
  "total_responses_scanned": 1842,
  "total_violations": 23,
  "violation_rate_pct": 1.25,
  "by_rule": {
    "no_em_dash": {"count": 12, "severity": "strict", "last_seen_ts": 1715712345.6},
    "no_markdown_header": {"count": 7, "severity": "strict", "last_seen_ts": 1715712301.1},
    "german_ascii_transliteration": {"count": 4, "severity": "strict", "last_seen_ts": 1715712289.4}
  },
  "by_model": {
    "anthropic/claude-haiku-4-5": {"responses": 1500, "violations": 18},
    "gemini/gemini-2.5-flash": {"responses": 342, "violations": 5}
  },
  "by_variant": {
    "default": {"responses": 1842, "violations": 23}
  },
  "recent_samples": [
    {"rule_id": "no_em_dash", "snippet": "...laufen war gut [U+2014] du solltest...", "model": "...", "variant": "default", "ts": 1715712345.6}
  ]
}
```

### A/B scaffold

```python
class PromptVariant(StrEnum):
    DEFAULT = "default"
    # Future: VARIANT_B = "variant_b", etc.

def resolve_variant(user_id: str | None) -> PromptVariant:
    """Return the variant the user is routed to.
    
    Today: always DEFAULT. Future: hash user_id, route by percentage.
    """
    return PromptVariant.DEFAULT
```

The agent loop passes `resolve_variant(user_id)` into `record_response`. When variant B exists, this routing function gains a single `if` branch, and `build_system_prompt` learns to take the variant arg.

### Alerting

After each `record_response`, the store checks `violation_rate(60)`. If it exceeds 5%, emit:

```
logger.warning("Prompt violation rate spike: %s%% over last 60s (responses=%d, violations=%d, top rule=%s)",
               rate_pct, responses, violations, top_rule)
```

Throttled: only one warning per 60s window via a `_last_warn_ts` field. Otherwise log spam during incidents.

### Persistence (optional)

A migration creates `public.prompt_violations`. If `settings.use_supabase` is true and the Supabase client is available, `record_response` fires a non-blocking insert. Failures log at debug and never block the agent loop.

Schema:
```sql
prompt_violations (
    id uuid pk default gen_random_uuid(),
    ts timestamptz default now(),
    user_id uuid nullable,           -- nullable to avoid coupling to user_model
    model text not null,
    variant text not null,
    rule_id text not null,
    severity text not null,
    response_length int not null,
    snippet text not null            -- truncated to 200 chars
)
```

Indexed on `(ts desc)` and `(rule_id, ts desc)` and `(variant, ts desc)`.

RLS: service-role-only, mirroring `background_jobs`.

### Integration with agent_loop

Single hook site at line ~963 in `agent_loop.py`, right after `result.response_text` is finalized and before `_save_turn`. It is wrapped in `try / except` and log-debug-on-failure, identical to the `track_usage` pattern at line 624-627. Telemetry must never break the agent.

```python
# After result.response_text is finalized:
try:
    from src.services.prompt_metrics import scan_and_record
    scan_and_record(
        text=result.response_text,
        model=MODEL,
        user_id=self._user_id,
    )
except Exception:
    pass  # Non-critical, never block agent loop
```

### Thread safety

Same pattern as `CacheTelemetry`: one `threading.Lock` for the deque, the counters, and the warn timestamp. Reads take the lock to snapshot a list, then release before computing. The endpoint handler is async but the underlying store calls are blocking and microsecond-scale, so we do NOT release the GIL or spawn a thread - direct call from the async handler is fine.

### Performance budget

- Detector pass: target <1ms on a 2KB response. Compiled regexes only.
- Record + lock: O(1) deque append, microseconds.
- Endpoint: O(N) scan of the ring buffer (max 500 records), microseconds.

The async event loop never blocks more than ~2ms per response on this code path.

### Backwards compatibility

- No user-visible behavior changes.
- No new required config; Supabase persistence is opt-in via existing `use_supabase` flag.
- Existing `/admin/cache-stats` endpoint untouched.
- Detector failures are caught and ignored; agent loop runs identically with or without the hook.

### Test plan

`tests/test_prompt_metrics.py` covers:

1. Each detector returns the expected `Violation` for a known-bad string, and zero for a known-good string.
2. `scan_response` runs all detectors and aggregates.
3. `PromptMetrics.record_response` increments counts, populates by_rule / by_model / by_variant.
4. `violation_rate(60)` respects the time cutoff (patched `time.time`).
5. The alert WARN fires exactly once when rate exceeds 5% within the 60s throttle window.
6. The `/admin/prompt-metrics` endpoint returns the documented JSON shape via `TestClient`.
7. Singleton identity (`get_prompt_metrics() is get_prompt_metrics()`).
8. German transliteration detector only fires with `language_hint="de"` or German content cues.
9. Detector throughput: 1000 scans of a 2KB response complete in <1s (smoke check, not a hard SLA).
