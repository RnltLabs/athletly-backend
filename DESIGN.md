# DESIGN: Premium Routing + Sport-Math Compute Tool

Two complementary changes:

1. Deterministic detector at message receipt. When the message looks like
   it needs deep reasoning, route every chat_completion call of this turn
   to `tier="complex"` (Sonnet + extended thinking).
2. A `compute_sport_math` tool. The agent calls it instead of doing
   arithmetic. Pure Python functions, fully tested, returns verified
   numbers plus a one-line interpretation.

Together: Sonnet thinks deeper about WHAT to compute, the tool guarantees
the numbers are RIGHT. Lisa's NP=352W hallucination becomes impossible.

## 1. Premium routing detector

### File

`src/agent/complexity_detector.py` (new)

### Public interface

```python
def needs_complex_reasoning(
    user_message: str,
    has_recent_long_plan_request: bool = False,
) -> tuple[bool, list[str]]:
    """Return (True, matched_keywords) if the message needs Sonnet+thinking.

    Pure function. No I/O. Case-insensitive keyword matching with word
    boundaries to avoid substring false positives.
    """
```

Tuple return shape: the boolean drives routing; the keyword list lands in
the agent_loop log so we can audit why a turn escalated.

### Trigger groups (composable)

The detector fires on ANY of these groups matching:

Group A. Cross-sport / multi-sport context:
- Sport tokens: `triathlon`, `triathlet*`, `ironman`, `langdistanz`,
  `mitteldistanz`, `halfironman`, `half-ironman`, `70.3`, `hyrox`,
  `duathlon`, `aquathlon`

Group B. Race-strategy intent:
- Verb/noun tokens: `pacing`, `pacingstrategie`, `race strategy`,
  `rennstrategie`, `wettkampfstrategie`, `nutrition`, `ernaehrung`,
  `ernährung`, `fueling`, `verpflegung`, `taper`, `tapering`, `peak`,
  `peaking`, `deload`

Group C. Sport-math symbols:
- Formula tokens: `FTP`, `NP`, `normalized power`, `VDOT`, `CSS`,
  `Karvonen`, `watts/kg`, `w/kg`, `threshold pace`, `lactate threshold`,
  `LT2`, `LTHR`

Group D. Long-horizon plan / periodization:
- Tokens: `periodisierung`, `periodization`, `marathon plan`,
  `ironman plan`, `aufbauplan`, `trainingsblock`, `mesozyklus`,
  `makrozyklus`, regex `\b\d{1,2}\s*(woch|wks?|weeks?)\b` where the
  number is at least 8

Group E. Explicit number-comparison or arithmetic question:
- Regex `\b\d{2,4}\s*(w|watts?|watt)\b` (e.g. "240 W")
- Regex `\b\d{1,2}[.:]\d{2}\b` near pace/time keywords
- Phrase tokens: `ist .* realistisch`, `wie schnell`, `wie viele watt`,
  `kann ich .* halten`, `was bedeutet`, `rechne`, `berechne`

### Decision rule

```
complex = A or (B and (A or C or D))
       or (C and D)
       or (C and E)
       or (D and E)
       or A             # multi-sport ALWAYS escalates
```

Equivalently, three "ANY one suffices" rules:
- Multi-sport sport tokens (Group A) alone
- Race-strategy intent (B) PLUS sport context (A or C or D)
- Sport-math symbols (C) PLUS long-plan or number-comparison (D or E)
- Long-plan (D) PLUS number-comparison (E)

This catches the Lisa case (A + B + C all hit) without firing on
"how do I feel today" (no group matches), "I ran 8km this morning"
(no group matches), "tell me about my last activity" (no group matches).

### False-positive guardrails

- Word-boundary matching only. "Iron-fisted" does not match "ironman".
- Sport-token alone (Group A) escalates UNCONDITIONALLY. We accept the
  ~10% false-positive rate from someone mentioning a past triathlon
  casually, because the cost asymmetry favors deep reasoning when
  multi-sport context is present at all.
- Casing irrelevant: `compile(..., re.IGNORECASE)`.
- Stopword check: if the entire message is < 8 chars, never escalate
  (avoids "FTP?" alone burning Sonnet).

### Telemetry

Detector returns `(bool, [matched_keywords])`. Agent loop logs:

```
complexity_detector: complex_triggered keywords=[triathlon, FTP, pacing]
```

or

```
complexity_detector: routine keywords=[]
```

The structured key `complex_triggered` is greppable for cost dashboards
("how many Sonnet calls did the detector cause yesterday").

## 2. Compute tool

### File

`src/agent/tools/compute_tools.py` (new). Pure Python, no DB, no I/O.
Tested in `tests/test_compute_tools.py` against published reference
table values (Daniels VDOT table 5.2; Coggan zones for FTP 240W).

### Tool name

`compute_sport_math`

### Tool dispatch

The tool takes two parameters:

- `formula`: string, one of the seven supported formula names
- `inputs`: dict, formula-specific input

Each formula maps to a private pure function. Unknown formula name
returns `{"error": "unknown formula 'X'", "supported": [...list]}` so
the agent can self-correct.

### Supported formulas

1. `vdot_from_race_time(distance_km, time_seconds) -> {"vdot": float}`
   Inverse of Daniels velocity-to-VO2. Reference: 5k 20:00 -> 49.2.

2. `paces_from_vdot(vdot) -> {"easy_per_km_seconds": float, "marathon_per_km_seconds": float, "threshold_per_km_seconds": float, "interval_per_km_seconds": float, "repetition_per_km_seconds": float, "easy_per_km": "mm:ss", ...}`
   Reference: VDOT 50 -> threshold 4:18/km, marathon 4:34/km.

3. `ftp_zones(ftp_watts) -> {"z1": [low, high], "z2": [low, high], ..., "z7": [low, high], "interpretation": "Coggan 7-zone..."}`
   Reference: FTP 240W -> Z2 134-180, Z4 218-252, Z5 254-288.

4. `np_from_intervals(intervals: list) -> {"np_watts": float, "warning": str|None}`
   Compute NP from `[[duration_s, watts], ...]` segments. Uses Coggan's
   30s rolling-average plus 4th-power algorithm. Adds a warning when
   NP > 1.10 * mean_power (signals high variability) and a separate
   `sanity` field if implausibly high vs supplied FTP (when caller
   passes `ftp_watts` in `inputs`).

5. `hr_zones_karvonen(rhr, max_hr) -> {"z1": [low, high], ..., "z5": [low, high]}`
   5-zone HRR bands. Reference: RHR 50, HRmax 190 -> Z2 134-148.

6. `css_paces(css_pace_per_100m) -> {"easy_per_100m": "mm:ss", "threshold_per_100m": "mm:ss", "race_per_100m": "mm:ss", "splits": {"100m": ..., "200m": ..., "400m": ..., "1000m": ..., "1500m": ...}}`
   Pool pace targets for set distances at CSS race pace.

7. `pace_target_from_goal(distance_km, target_time_seconds) -> {"pace_per_km_seconds": float, "pace_per_km": "mm:ss", "speed_kmh": float}`
   Trivial division. Included because LLMs get it wrong under load.

### Return shape contract

Every formula returns a dict with `status="success"` and the result keys
listed. On any input error: `{"status": "error", "message": "..."}`. The
agent sees both shapes and reacts appropriately.

### Sanity warnings

The tool exists to catch the Lisa bug. Two specific sanity checks:

- `np_from_intervals`: if `inputs` contains an `ftp_watts` field and the
  computed NP > 1.15 * FTP, the returned dict includes
  `"sanity_warning": "NP (X W) exceeds 1.15 x FTP (Y W). This is only
  plausible for short supra-threshold intervals; verify the interval
  data is correct."`. The agent sees this warning before it can reply.

- `ftp_zones`: if FTP < 80 W or > 500 W, returns the zones but adds
  `"sanity_warning": "FTP outside typical adult range (80 to 500 W)..."`.

These warnings are not blocking. They are signals to the LLM. With the
Sonnet+thinking tier they will be incorporated; on Haiku they at least
add structured truth into the conversation history.

### Tool description (for the LLM)

> Compute sport-science formulas with verified, deterministic math. For
> ANY race-pace, training-zone, NP, FTP, VDOT, CSS, or HR-zone
> calculation, call this tool instead of computing inline. The tool's
> output is the source of truth; quote its numbers verbatim. Supported
> formulas: `vdot_from_race_time`, `paces_from_vdot`, `ftp_zones`,
> `np_from_intervals`, `hr_zones_karvonen`, `css_paces`,
> `pace_target_from_goal`. Pass `formula` and `inputs`.

Tool category: `analysis` (read-only, deterministic, safe).

Display label (German UI string): `Berechne Sport-Werte`.

### CORE_TOOL_NAMES decision

Add `compute_sport_math` to `CORE_TOOL_NAMES`. Reasoning: the STRICT
system-prompt rule will tell the agent to call this for any FTP/NP/VDOT
calculation. Without the schema always loaded, the agent has to
tool_search round-trip exactly when the math discipline matters most.
That defeats the rule. Cost: ~80 extra tokens per turn (one tool schema).
Worth it.

## 3. Wiring in `agent_loop.py`

### Where

`AgentLoop.process_message`, immediately after the user message is
appended to `self._messages` and BEFORE the loop's first chat_completion.

### What

```python
from src.agent.complexity_detector import needs_complex_reasoning

is_complex, matched = needs_complex_reasoning(user_message)
if is_complex:
    logger.info(
        "complexity_detector: complex_triggered keywords=%s",
        matched,
    )
    selected_tier = "complex"
else:
    selected_tier = "routine"
```

Then change EVERY `chat_completion(...tier="routine")` in this method
to `tier=selected_tier`. Both the main loop call and the empty-response
tool-free fallback. The runtime_context block and tool list are
unchanged; only the tier flips.

### Stickiness within a turn

The tier is set ONCE at message receipt and reused for every tool round
within the turn. Mid-turn switching is intentionally NOT supported (it
would invalidate the Anthropic prompt cache).

### Sonnet failure fallback

The Anthropic-key safety net in `llm.py` already falls back to
`MODEL` (Gemini) when `ANTHROPIC_API_KEY` is missing. That covers
local dev. For runtime Sonnet failures (rate limit exhausted,
overload), the existing rate-limit retry policy handles 429s; on
final exhaustion the exception propagates to the agent loop's
existing error-handling path.

Per spec: when the COMPLEX call fails, we annotate the response
with an explicit fallback notice. New helper in agent_loop:

```python
def _retry_complex_as_routine(self, *args, **kwargs):
    """Single retry with tier='routine' if complex call raises.

    Returns (response, fallback_used: bool).
    """
```

When `fallback_used` is True, we prepend an annotation to the final
response text:

`"[Hinweis: Tiefere Analyse war kurz nicht verfuegbar, hier meine beste
Antwort mit einfachem Reasoning.]\n\n"`

(German umlaut umlauts in the user-facing string, as the rule requires.)

This annotation only fires when the detector said "complex" AND the
Sonnet call actually failed and we retried on Haiku. Routine turns never
see this.

### Cost telemetry

The existing `_log_cache_usage` already emits `premium_call` log lines
for Sonnet hits. We add a counter line at the detector layer:

```
complexity_detector_decision=complex|routine keywords=[...] user_id=...
```

so a daily grep gives the routing-decision distribution independent of
the actual LLM call outcome.

## 4. System prompt change

Add a single STRICT block in `src/agent/system_prompt.py` near the
existing math STRICT rules:

```
STRICT: SPORT MATH DISCIPLINE. For ANY calculation involving FTP, NP,
normalized power, VDOT, CSS, Karvonen HR zones, watts/kg, threshold
pace, race pace, training zones, or interval pacing, you MUST call
`compute_sport_math(formula=..., inputs=...)` and quote the returned
numbers verbatim. NEVER compute these yourself, NEVER estimate
inline. NP can NEVER exceed FTP by more than ~10 percent on a
sustainable interval; if the tool returns a sanity_warning, surface
it to the athlete instead of suppressing it.
```

This rule is loud and specific. The agent reads it on every cached
prompt request.

## 5. Tests

### `tests/test_complexity_detector.py`

Triggers (must return True):
- "Ich bereite mich auf Roth vor. Wie pace ich die Langdistanz?" (A + B)
- "Mein FTP ist 240 W, kann ich 280 W NP halten ueber 1h?" (C + E)
- "Plane meinen 16-Wochen Marathonaufbau" (D + implicit math)
- "Triathlon Ernaehrung waehrend Race" (A + B)
- "Wie berechne ich CSS aus 400m und 200m?" (C)
- "Hyrox Pacing Strategie" (A + B)

Non-triggers (must return False):
- "Wie fuehle ich mich heute?"
- "Hallo, mein Name ist Lisa"
- "Ich war gestern laufen, 8 km easy"
- "Zeig mir meine letzten Aktivitaeten"
- "FTP?" (too short, < 8 chars)
- "Mein Triathlon ist letztes Jahr gewesen, kein Stress" - this is an
  edge case; sport-keyword alone DOES escalate per rule. We accept this
  false positive deliberately, but test it for documentation.

### `tests/test_compute_tools.py`

Reference-table accuracy tests (within 1% of published values):

- VDOT 5k 20:00 -> 49 plus/minus 1
- VDOT 10k 40:00 -> 52 plus/minus 1
- VDOT 5k 18:00 -> 55 plus/minus 1
- VDOT 50 -> threshold pace 4:18/km plus/minus 5 sec
- FTP 240 -> Z4 218 to 252 W (exact bounds)
- FTP 240 -> Z5 254 to 288 W (exact bounds)
- NP from constant 200W for 60 min: NP == 200 W exactly
- NP from [200W * 30min, 300W * 30min]: NP > mean (240W), < 300W,
  near the textbook value 261 W plus/minus 3
- Karvonen RHR 50, HRmax 190: Z2 134-148 exact
- CSS from T400=400s T200=180s: CSS = (400-180)/2 = 110 sec/100m
- Pace 10km in 50:00: 5:00/km exact
- Sanity warning fires when NP > 1.15 * FTP

Tests must run without external services (pure functions).

## 6. Cost / impact estimate

- Detector fires on roughly 5 to 15% of turns based on language patterns
  in current sessions (rough estimate; verifiable after a week of
  telemetry).
- Sonnet 4.6 costs ~3x Haiku 4.5 per token. With extended thinking
  budget=2048 we add ~2048 thinking-output tokens to the bill on
  triggered turns.
- Net cost increase: ~10 to 25% of total agent spend, assuming current
  Sonnet thinking is rarely invoked elsewhere.
- Math accuracy: 100% on the seven supported formulas (deterministic).
- Lisa-test specifically: with both changes, the agent (a) escalates the
  turn to Sonnet+thinking, (b) calls `compute_sport_math` for any FTP
  zone, NP estimate, or pace question, (c) gets the real number back
  instead of confabulating. NP=352 from FTP=240 becomes impossible: the
  tool either rejects bad inputs or returns a sanity warning.
