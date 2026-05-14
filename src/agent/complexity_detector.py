"""Deterministic complexity detector for premium routing.

When the user message looks like it needs deep, multi-step reasoning -
cross-sport context, race strategy, sport-science math, long-horizon
planning - this module returns True and the agent loop pins the
turn's tier to "complex" (Sonnet + extended thinking). The thinking
step is where arithmetic and step-ordering errors get caught before
they reach the athlete.

Design goals:
- Pure function. No I/O, no DB, no LLM call. Microseconds per turn.
- Deterministic. Same input -> same output, every time. Tests pin
  individual cases by string.
- Auditable. Returns the list of matched keywords so the agent loop
  log line shows EXACTLY why it escalated.
- Tight on false positives. A casual mention of "I felt tired today"
  must NOT escalate, otherwise we burn Sonnet on routine turns.

The keyword groups and decision rule are documented in DESIGN.md
section 1. Reference: research/Sport-science formulas (Daniels,
Coggan, Karvonen, Swim Smooth) and routing patterns from Anthropic /
LangChain / OpenAI docs (Q2 2026).
"""

from __future__ import annotations

import re
from typing import Pattern

# Minimum message length before we bother running the detector. Avoids
# escalating one-word messages like "FTP?" that carry no real query.
_MIN_MESSAGE_LEN = 8


def _kw_regex(keywords: list[str]) -> Pattern[str]:
    """Build a case-insensitive, word-boundary keyword union regex.

    Each keyword is escaped so substrings like "70.3" (with the dot)
    match literally. Word boundaries on both sides prevent substring
    false positives ("iron" should NOT match inside "ironing").
    """
    escaped = [re.escape(k) for k in keywords]
    pattern = r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


# Group A: multi-sport / cross-sport context tokens.
# Sport-specific words. Triathlon-flavoured queries almost always need
# the multi-discipline reasoning Sonnet provides.
_GROUP_A_KEYWORDS: list[str] = [
    "triathlon", "triathlet", "triathletin", "triathleten",
    "ironman", "iron-man",
    "langdistanz", "mitteldistanz", "kurzdistanz",
    "halfironman", "half-ironman", "70.3",
    "hyrox",
    "duathlon", "aquathlon",
]
_GROUP_A_RE = _kw_regex(_GROUP_A_KEYWORDS)

# Group B: race-strategy intent verbs and nouns. Words that signal
# the athlete wants strategy/planning advice, not a status check.
_GROUP_B_KEYWORDS: list[str] = [
    "pacing", "pacing-strategie", "pacingstrategie",
    "race strategy", "racestrategie",
    "rennstrategie", "wettkampfstrategie", "wettkampfplan",
    "nutrition", "ernaehrung", "ernährung",
    "fueling", "verpflegung", "verpflegungsstrategie",
    "taper", "tapering", "tapere",
    "peak", "peaking",
    "deload",
]
_GROUP_B_RE = _kw_regex(_GROUP_B_KEYWORDS)

# Group C: sport-math symbols. Direct formula references. Catching
# these is the WHOLE point of the bug fix: Haiku confabulated NP from
# FTP. Anytime these tokens appear, we want Sonnet thinking.
_GROUP_C_KEYWORDS: list[str] = [
    "FTP",
    "NP", "normalized power", "normalisierte leistung",
    "VDOT",
    "CSS",
    "Karvonen",
    "watts/kg", "watt/kg", "w/kg", "watts per kg",
    "threshold pace", "schwellentempo", "schwellenpace",
    "lactate threshold", "laktatschwelle",
    "LT2", "LTHR",
]
_GROUP_C_RE = _kw_regex(_GROUP_C_KEYWORDS)

# Group D: long-horizon plan / periodization. Word tokens (regex for
# the explicit weeks-count below).
_GROUP_D_KEYWORDS: list[str] = [
    "periodisierung", "periodization",
    "marathon plan", "marathonplan", "marathon-plan",
    "marathon-aufbau", "marathonaufbau", "marathon aufbau",
    "marathon-aufbauplan", "marathonaufbauplan",
    "ironman plan", "ironmanplan",
    "aufbauplan", "aufbau-plan",
    "trainingsblock", "trainings-block",
    "trainingsplan", "trainings-plan",
    "mesozyklus", "makrozyklus", "mikrozyklus",
    "build phase", "build-phase", "buildphase",
    "peak phase", "peak-phase",
    "base phase", "base-phase",
]
_GROUP_D_RE = _kw_regex(_GROUP_D_KEYWORDS)

# Weeks-count regex: "8 weeks", "12 wks", "16-woechige", "16 Wochen".
# German nouns are capitalized so we explicitly match both cases via the
# IGNORECASE flag. A planning horizon of 8 plus weeks consistently
# exceeds what Haiku can plan without dropping or duplicating sessions
# across phases. Shorter horizons (4 to 7 weeks) escalate only when
# combined with a plan-related word; see _SHORT_HORIZON_WEEKS_MIN below.
_GROUP_D_WEEKS_RE = re.compile(
    r"\b(\d{1,2})\s*-?\s*(woch|wks?|weeks?|woechig|wöchig)\w*",
    re.IGNORECASE,
)

# Threshold for "long horizon alone is enough" - 8 weeks plus escalates
# even without explicit plan keywords ("plane mir die naechsten 12 Wochen").
_LONG_HORIZON_WEEKS_MIN = 8

# Threshold for "short horizon combined with plan keyword escalates"
# ("Plan fuer die naechsten 4 Wochen Marathon-Aufbau"). 4 weeks is the
# shortest meaningful mesocycle; anything below is a single week's
# adjustment and stays routine.
_SHORT_HORIZON_WEEKS_MIN = 4

# Plan-intent words: presence of any of these PLUS a >=4 week horizon
# escalates. Kept separate from Group D keyword list so the keyword
# audit trail still distinguishes "explicit plan term matched" from
# "horizon + plan intent matched".
_PLAN_INTENT_RE = re.compile(
    r"(?<!\w)("
    r"plan|plane|planen|planung|"
    r"aufbau|aufbauen|"
    r"build|building|"
    r"phase|phasen|"
    r"block|bloecke|bloeke|bloecke|"
    r"trainingsplan|trainings-plan|"
    r"vorbereitung|vorbereiten|"
    r"strukturieren?|strukturiert"
    r")(?!\w)",
    re.IGNORECASE,
)

# Marathon / Ironman event names commonly used in German running scene.
# Bare event mention (e.g. "Koeln Marathon", "Berlin Marathon") is a
# strong long-horizon hint even without an explicit plan keyword - a
# user dropping the event name almost always wants planning help.
_EVENT_NAME_RE = re.compile(
    r"(?<!\w)("
    # Generic distance terms that imply long-horizon planning when paired
    # with a plan intent. "marathon" alone is too noisy (could be a
    # marathon recap), but "Koeln Marathon" or "Berlin Marathon" or any
    # city + Marathon pattern is a clear race target.
    r"berlin\s+marathon|koeln\s+marathon|k(?:o|oe|ö)ln\s+marathon|"
    r"hamburg\s+marathon|muenchen\s+marathon|m(?:u|ue|ü)nchen\s+marathon|"
    r"frankfurt\s+marathon|wien\s+marathon|"
    r"boston\s+marathon|new\s+york\s+marathon|nyc\s+marathon|"
    r"london\s+marathon|chicago\s+marathon|tokyo\s+marathon|"
    r"ironman\s+\w+|"
    r"roth(?:\s+langdistanz)?|"
    r"challenge\s+\w+"
    r")(?!\w)",
    re.IGNORECASE,
)

# Marathon proximity rule: "Marathon" (bare) plus "Plan" (bare) within
# this many characters of each other implies a long-horizon plan
# request even when the user does not use a compound noun. Catches
# Elena's "trainiere auf den Marathon ... Plan bitte" phrasing.
_MARATHON_PROXIMITY_CHARS = 80
_MARATHON_RE = re.compile(r"(?<!\w)marathon(?!\w)", re.IGNORECASE)
_PLAN_BARE_RE = re.compile(
    r"(?<!\w)(plan|plane|planen|aufbau|aufbauplan)(?!\w)",
    re.IGNORECASE,
)

# Group E: explicit number-comparison or arithmetic question.
# Patterns that signal the athlete is asking for a computed answer.
_GROUP_E_WATT_RE = re.compile(
    r"\b\d{2,4}\s*(w|watts?|watt)\b",
    re.IGNORECASE,
)
_GROUP_E_PACE_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\b",  # mm:ss or mm.ss style
)
_GROUP_E_PHRASE_RE = re.compile(
    r"("
    r"ist\s+\S+\s+realistisch"
    r"|wie\s+schnell"
    r"|wie\s+viele?\s+watt"
    r"|kann\s+ich\s+\S+\s+halten"
    r"|was\s+bedeutet"
    r"|berechne|rechne|kalkuliere"
    r"|sub-?\d+(:\d+)?"
    r")",
    re.IGNORECASE,
)


def _find_all(pattern: Pattern[str], text: str) -> list[str]:
    """Return every distinct match (lowercased) for *pattern* in *text*.

    Used to build the keyword audit trail. We only care that a keyword
    matched, not how many times.
    """
    seen: list[str] = []
    for m in pattern.finditer(text):
        token = m.group(0).strip().lower()
        if token and token not in seen:
            seen.append(token)
    return seen


def needs_complex_reasoning(
    user_message: str,
    has_recent_long_plan_request: bool = False,
) -> tuple[bool, list[str]]:
    """Decide whether *user_message* needs Sonnet + extended thinking.

    Returns:
        (is_complex, matched_keywords). The keyword list is the audit
        trail; an empty list with is_complex=False means no group
        matched. With is_complex=True, the list contains the actual
        tokens (lowercased) that triggered each group, in the order
        A, B, C, D, E.

    The decision rule (from DESIGN.md):
        complex = A
               or (B and (A or C or D))
               or (C and (D or E))
               or (D and E)
               or has_recent_long_plan_request

    Group A alone is enough: multi-sport keywords ALWAYS escalate
    because cross-sport reasoning routinely trips Haiku.
    """
    if not isinstance(user_message, str) or len(user_message.strip()) < _MIN_MESSAGE_LEN:
        return False, []

    text = user_message

    matched_a = _find_all(_GROUP_A_RE, text)
    matched_b = _find_all(_GROUP_B_RE, text)
    matched_c = _find_all(_GROUP_C_RE, text)
    matched_d = _find_all(_GROUP_D_RE, text)

    # Plan-intent words. Used both for the short-horizon-plus-plan combo
    # below and for the Marathon-proximity rule further down. Stored as
    # a flag rather than appended to matched_d so the audit trail only
    # gains entries when the keyword ACTUALLY drove the escalation.
    has_plan_intent = bool(_PLAN_INTENT_RE.search(text))

    # Group D also fires on a numeric week horizon detected via regex.
    # Long horizon alone (>=8 weeks) escalates immediately. Shorter
    # horizons (4 to 7 weeks) escalate only when combined with a plan
    # intent word so "4 Wochen verletzt" stays routine but
    # "Plan fuer die naechsten 4 Wochen Aufbau" escalates.
    weeks_hits = _GROUP_D_WEEKS_RE.findall(text)
    long_horizon_weeks = False
    for hit in weeks_hits:
        try:
            n = int(hit[0])
        except (TypeError, ValueError):
            continue
        if n >= _LONG_HORIZON_WEEKS_MIN:
            long_horizon_weeks = True
            matched_d.append(f"{n}-week plan")
            break
        if n >= _SHORT_HORIZON_WEEKS_MIN and has_plan_intent:
            long_horizon_weeks = True
            matched_d.append(f"{n}-week plan + plan_intent")
            break

    # Event-name allow-list: "Koeln Marathon", "Berlin Marathon",
    # "Ironman Frankfurt", "Roth Langdistanz", ... All strong
    # long-horizon hints when paired with a plan intent. Bare event
    # mentions without any plan word stay routine to avoid escalating
    # casual race chatter ("war beim Berlin Marathon dabei").
    event_matches = _find_all(_EVENT_NAME_RE, text)
    if event_matches and has_plan_intent:
        for ev in event_matches:
            token = f"event:{ev}"
            if token not in matched_d:
                matched_d.append(token)
        long_horizon_weeks = True  # treat as Group D long-horizon hit

    # Marathon-proximity rule: "Marathon" and "Plan/Aufbau" within
    # _MARATHON_PROXIMITY_CHARS chars of each other. Catches Elena's
    # exact phrasing "trainiere auf den Marathon am 4. Oktober, Plan
    # bitte fuer die naechsten 16 Wochen" - the keyword list would
    # require literal "marathon plan" (two consecutive words) which
    # never appears in natural German.
    m_match = _MARATHON_RE.search(text)
    p_match = _PLAN_BARE_RE.search(text)
    if m_match and p_match:
        distance = abs(m_match.start() - p_match.start())
        if distance <= _MARATHON_PROXIMITY_CHARS:
            token = "marathon+plan_proximity"
            if token not in matched_d:
                matched_d.append(token)
            long_horizon_weeks = True

    # Group E: numeric / phrase hints.
    matched_e: list[str] = []
    for m in _GROUP_E_WATT_RE.finditer(text):
        token = m.group(0).strip().lower()
        if token not in matched_e:
            matched_e.append(token)
    for m in _GROUP_E_PACE_RE.finditer(text):
        token = m.group(0).strip().lower()
        if token not in matched_e:
            matched_e.append(token)
    for m in _GROUP_E_PHRASE_RE.finditer(text):
        token = m.group(0).strip().lower()
        if token not in matched_e:
            matched_e.append(token)

    has_a = bool(matched_a)
    has_b = bool(matched_b)
    has_c = bool(matched_c)
    has_d = bool(matched_d) or long_horizon_weeks
    has_e = bool(matched_e)

    # Pace hits alone are too noisy ("ich lief 5:00") - require pace
    # to combine with a sport-math symbol (Group C) or planning context
    # (Group D) before contributing to escalation.
    pace_only = has_e and not has_a and not has_b and not has_c and not has_d
    if pace_only:
        has_e = False
        matched_e = []

    # A long-horizon plan signal (Group D) is sufficient on its own:
    # "16 Wochen Marathonaufbau" is exactly the periodization case Haiku
    # mishandles. Multi-sport tokens (A) also escalate unconditionally.
    is_complex = (
        has_a
        or has_d
        or (has_b and (has_a or has_c or has_d))
        or (has_c and (has_d or has_e))
        or has_recent_long_plan_request
    )

    # Build audit trail in deterministic order (A, B, C, D, E).
    matched_all: list[str] = []
    for bucket in (matched_a, matched_b, matched_c, matched_d, matched_e):
        for token in bucket:
            if token not in matched_all:
                matched_all.append(token)

    if has_recent_long_plan_request and "recent_long_plan_request" not in matched_all:
        matched_all.append("recent_long_plan_request")

    return is_complex, matched_all
