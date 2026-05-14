"""Prompt rule violation telemetry for STRICT rules in system_prompt.py.

Mirrors the cache_telemetry.py pattern: singleton, threading.Lock,
in-memory ring buffer. Optional Supabase persistence when enabled.

The detectors are pure regex (zero LLM cost, microsecond runtime).
They cover the easy STRICT rules:
    - em-dash (U+2014)
    - en-dash (U+2013)
    - Markdown headers
    - Markdown bold and italic
    - German ASCII transliteration ('ueber' instead of 'ueber'... ahem, 'ue' instead of umlaut)
    - Decimal-pace leak (Sprint C): "4.50/km" instead of "4:30/km"

Hard rules (fabricated stats, mid-response language switch, claims
without enough data) are deferred to a future LLM-as-judge worker that
will operate offline on sampled conversations. They are NOT detected
in the production path.

Hook into the agent loop via :func:`scan_and_record` after the final
response text is captured (see agent_loop.py around line 963).

A/B testing scaffold lives in :class:`PromptVariant` and
:func:`resolve_variant`. Today every user gets ``DEFAULT``; flipping a
percentage of traffic to a new variant is a one-line change.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A/B testing scaffold
# ---------------------------------------------------------------------------


class PromptVariant(str, Enum):
    """Enumerated prompt variants for A/B testing.

    Today: only DEFAULT. Future: add VARIANT_B, VARIANT_C, etc., and
    route a percentage of users by ``resolve_variant``.
    """

    DEFAULT = "default"


def resolve_variant(user_id: str | None) -> PromptVariant:
    """Return the prompt variant for a given user.

    Args:
        user_id: The athlete's user id. Used to deterministically route
            them to a variant. ``None`` falls back to DEFAULT.

    Returns:
        The :class:`PromptVariant` this user should see.

    Today every user gets DEFAULT. To split traffic 50/50 later::

        if user_id and (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 2) == 0:
            return PromptVariant.VARIANT_B
        return PromptVariant.DEFAULT
    """
    return PromptVariant.DEFAULT


# ---------------------------------------------------------------------------
# Violation data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """A single detected rule violation. Pure value type, no I/O."""

    rule_id: str
    severity: str  # "strict" or "warn"
    snippet: str   # short excerpt around the match (max ~80 chars)
    char_offset: int


@dataclass(frozen=True)
class ViolationRecord:
    """Immutable snapshot of a violation, tagged with model + variant + time."""

    ts: float
    model: str
    variant: str
    rule_id: str
    severity: str
    response_length: int
    snippet: str


# ---------------------------------------------------------------------------
# Detectors -- pure regex, no I/O, no LLM
# ---------------------------------------------------------------------------


_EM_DASH_RE = re.compile("—")
_EN_DASH_RE = re.compile("–")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
_MD_ITALIC_RE = re.compile(r"(?<![*\\])\*[^*\n]+\*(?!\*)")

# Decimal-pace leak detector (Sprint C, extended Sprint K). Catches the
# exact failure mode from the Elena bug: a coach response that emits a
# decimal-minutes pace like "4.50/km" or "4.50 min/km" instead of the
# correct mm:ss form "4:30/km". The /km qualifier makes the context
# unambiguous regardless of the minute digit, so we accept any single-
# or double-digit whole minute. The decimal must have exactly ONE or TWO
# digits to keep the pattern specific to pace notation (vs. "4.5km" the
# distance, which we tolerate but only as plain km, not "min/km" pace).
#
# Sprint K: accept BOTH period and comma as the decimal separator so the
# German notation "4,50/km" is also caught. Without this, the gate was
# blind to the most likely failure mode for a DE coach response.
_DECIMAL_PACE_RE = re.compile(
    r"\b\d{1,2}[.,]\d{1,2}\s*(?:min\s*)?/\s*km\b",
    re.IGNORECASE,
)

# German ASCII transliteration: the German-context detector fires on
# words that read like umlaut transliteration. We avoid English false
# positives ("user", "feature") by requiring the words to be
# unambiguously German (no English homograph).
_GERMAN_TRANSLIT_RE = re.compile(
    r"\b("
    r"ueberfaellig|ueberfaelliger|ueberfaelligen|"
    r"Praeferenz|Praeferenzen|"
    r"Maerz|"
    r"Laeufer|Laeuferin|"
    r"Groesse|Groessen|"
    r"fuer|fuers|"
    r"ueberraschung|Ueberraschung|"
    r"Stoerung|Stoerungen|"
    r"Maennchen|Maedchen|"
    r"Erlaeuterung|"
    r"haetten|koennten|wuerden|moechten"
    r")\b",
    re.IGNORECASE,
)

# Cheap German-context heuristic: if the text contains >=3 of these
# common German function words, we assume the response is German and
# fire the transliteration detector even without an explicit hint.
_GERMAN_HINT_WORDS = {
    "und", "ist", "nicht", "auch", "mit", "fuer", "fur", "das",
    "den", "dem", "die", "der", "ein", "eine", "einen", "auf",
    "aber", "wenn", "schon", "noch", "war", "haben", "habe", "hast",
    "kann", "soll", "muss", "wirst", "werden", "wir", "ich", "du",
    "dein", "deine", "mein", "meine", "sie", "es", "doch", "mal",
}
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _snippet_around(text: str, start: int, end: int, ctx: int = 30) -> str:
    """Return a short excerpt around a match for debugging."""
    s = max(0, start - ctx)
    e = min(len(text), end + ctx)
    prefix = "..." if s > 0 else ""
    suffix = "..." if e < len(text) else ""
    # Collapse whitespace so the snippet stays on one line in JSON.
    raw = text[s:e].replace("\n", " ").replace("\r", " ")
    return f"{prefix}{raw}{suffix}"[:200]


def _looks_german(text: str) -> bool:
    """Cheap heuristic: does this text look like German prose?"""
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _GERMAN_HINT_WORDS)
    return hits >= 3


def _scan_pattern(
    text: str,
    pattern: re.Pattern[str],
    rule_id: str,
    severity: str,
) -> list[Violation]:
    """Yield violations for every match of ``pattern`` in ``text``."""
    out: list[Violation] = []
    for m in pattern.finditer(text):
        out.append(
            Violation(
                rule_id=rule_id,
                severity=severity,
                snippet=_snippet_around(text, m.start(), m.end()),
                char_offset=m.start(),
            )
        )
    return out


def scan_response(text: str, language_hint: str | None = None) -> list[Violation]:
    """Run all production detectors over ``text``. Pure function.

    Args:
        text: The assistant's final response text.
        language_hint: Optional ISO language code (e.g. ``"de"``).
            When ``"de"`` the German transliteration detector fires
            unconditionally. Otherwise it fires only if the text looks
            German via :func:`_looks_german`.

    Returns:
        A list of :class:`Violation` instances; empty if the response
        is clean.
    """
    if not text:
        return []

    violations: list[Violation] = []
    violations.extend(_scan_pattern(text, _EM_DASH_RE, "no_em_dash", "strict"))
    violations.extend(_scan_pattern(text, _EN_DASH_RE, "no_en_dash", "strict"))
    violations.extend(_scan_pattern(text, _MD_HEADER_RE, "no_markdown_header", "strict"))
    violations.extend(_scan_pattern(text, _MD_BOLD_RE, "no_markdown_bold", "strict"))
    violations.extend(_scan_pattern(text, _MD_ITALIC_RE, "no_markdown_italic", "warn"))
    violations.extend(
        _scan_pattern(text, _DECIMAL_PACE_RE, "decimal_pace_leak", "strict")
    )

    is_german = (language_hint or "").lower().startswith("de") or _looks_german(text)
    if is_german:
        violations.extend(
            _scan_pattern(
                text,
                _GERMAN_TRANSLIT_RE,
                "german_ascii_transliteration",
                "strict",
            )
        )

    return violations


# ---------------------------------------------------------------------------
# Ring buffer + singleton store
# ---------------------------------------------------------------------------


_MAX_RECORDS_DEFAULT = 500
_ALERT_RATE_THRESHOLD = 0.05  # 5% violations per minute trips the warn log
_ALERT_THROTTLE_SECONDS = 60.0
_MIN_SAMPLES_FOR_ALERT = 10   # do not alert on a tiny sample (avoid noise)


class PromptMetrics:
    """Singleton telemetry collector for prompt rule violations.

    Use :func:`get_prompt_metrics` to obtain the process-wide instance.

    Thread safety: a single :class:`threading.Lock` guards the violation
    deque, the per-bucket counters, and the alert throttle timestamp.
    Append is O(1). Summary is O(N) over the buffer.
    """

    def __init__(self, max_records: int = _MAX_RECORDS_DEFAULT):
        self._violations: deque[ViolationRecord] = deque(maxlen=max_records)
        self._response_log: deque[tuple[float, str, str]] = deque(maxlen=max_records)
        # (ts, model, variant) per scanned response, for rate denominators.
        self._lock = threading.Lock()
        self._last_alert_ts: float = 0.0

    # -- Recording -----------------------------------------------------------

    def record_response(
        self,
        model: str,
        variant: str,
        text: str,
        violations: list[Violation],
    ) -> None:
        """Record one scanned response and any violations it produced.

        Always logs the response (even if clean) so the rate denominator
        is correct. Each violation appends one :class:`ViolationRecord`.
        Fires a throttled WARN if the per-minute rate exceeds 5%.
        """
        now = time.time()
        response_length = len(text) if text else 0

        records = [
            ViolationRecord(
                ts=now,
                model=model,
                variant=variant,
                rule_id=v.rule_id,
                severity=v.severity,
                response_length=response_length,
                snippet=v.snippet,
            )
            for v in violations
        ]

        # Snapshot the locked state we will need to evaluate the alert.
        with self._lock:
            self._response_log.append((now, model, variant))
            for rec in records:
                self._violations.append(rec)
            rate_state = self._compute_rate_state_locked(window_seconds=60, now=now)
            should_alert = (
                rate_state["responses"] >= _MIN_SAMPLES_FOR_ALERT
                and rate_state["rate"] > _ALERT_RATE_THRESHOLD
                and (now - self._last_alert_ts) >= _ALERT_THROTTLE_SECONDS
            )
            if should_alert:
                self._last_alert_ts = now

        # Alert log is emitted outside the lock to keep critical section tight.
        if should_alert:
            logger.warning(
                "Prompt violation rate spike: %.1f%% over last 60s "
                "(responses=%d, violations=%d, top_rule=%s)",
                rate_state["rate"] * 100,
                rate_state["responses"],
                rate_state["violations"],
                rate_state["top_rule"],
            )

    # -- Reads ---------------------------------------------------------------

    def violation_rate(self, window_seconds: int = 60) -> float:
        """Fraction of responses in the last ``window_seconds`` with at least one violation."""
        with self._lock:
            return self._compute_rate_state_locked(window_seconds, time.time())["rate"]

    def summary(self, window_seconds: int = 3600) -> dict:
        """Aggregate violations and responses over a time window.

        Args:
            window_seconds: How far back to look. Default 1 hour.

        Returns:
            A dict suitable for the /admin/prompt-metrics endpoint.
            See DESIGN.md for the shape.
        """
        now = time.time()
        cutoff = now - max(0, window_seconds)

        with self._lock:
            recent_violations = [r for r in self._violations if r.ts >= cutoff]
            recent_responses = [r for r in self._response_log if r[0] >= cutoff]

        by_rule: dict[str, dict] = {}
        for r in recent_violations:
            d = by_rule.setdefault(
                r.rule_id,
                {"count": 0, "severity": r.severity, "last_seen_ts": r.ts},
            )
            d["count"] += 1
            if r.ts > d["last_seen_ts"]:
                d["last_seen_ts"] = r.ts

        by_model: dict[str, dict[str, int]] = {}
        for ts, model, _variant in recent_responses:
            by_model.setdefault(model, {"responses": 0, "violations": 0})["responses"] += 1
        for r in recent_violations:
            by_model.setdefault(r.model, {"responses": 0, "violations": 0})["violations"] += 1

        by_variant: dict[str, dict[str, int]] = {}
        for ts, _model, variant in recent_responses:
            by_variant.setdefault(variant, {"responses": 0, "violations": 0})["responses"] += 1
        for r in recent_violations:
            by_variant.setdefault(r.variant, {"responses": 0, "violations": 0})["violations"] += 1

        total_resp = len(recent_responses)
        total_viol = len(recent_violations)
        rate_pct = (total_viol / total_resp * 100.0) if total_resp else 0.0

        # Last 10 violations for inspection
        recent_samples = [
            {
                "rule_id": r.rule_id,
                "severity": r.severity,
                "snippet": r.snippet,
                "model": r.model,
                "variant": r.variant,
                "ts": r.ts,
            }
            for r in list(recent_violations)[-10:]
        ]

        return {
            "window_minutes": round(window_seconds / 60.0, 1),
            "total_responses_scanned": total_resp,
            "total_violations": total_viol,
            "violation_rate_pct": round(rate_pct, 2),
            "by_rule": by_rule,
            "by_model": by_model,
            "by_variant": by_variant,
            "recent_samples": recent_samples,
        }

    # -- Private -------------------------------------------------------------

    def _compute_rate_state_locked(
        self, window_seconds: int, now: float
    ) -> dict:
        """Compute rate + top-rule over the window. Caller MUST hold the lock."""
        cutoff = now - max(0, window_seconds)
        responses = sum(1 for ts, *_ in self._response_log if ts >= cutoff)
        violations = [r for r in self._violations if r.ts >= cutoff]
        rate = (len(violations) / responses) if responses else 0.0
        top_rule = "n/a"
        if violations:
            counts: dict[str, int] = {}
            for r in violations:
                counts[r.rule_id] = counts.get(r.rule_id, 0) + 1
            top_rule = max(counts.items(), key=lambda kv: kv[1])[0]
        return {
            "responses": responses,
            "violations": len(violations),
            "rate": rate,
            "top_rule": top_rule,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_singleton: PromptMetrics | None = None
_singleton_lock = threading.Lock()


def get_prompt_metrics() -> PromptMetrics:
    """Return the process-wide :class:`PromptMetrics` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = PromptMetrics()
    return _singleton


# ---------------------------------------------------------------------------
# Convenience entry point for the agent loop
# ---------------------------------------------------------------------------


def scan_and_record(
    text: str,
    model: str,
    user_id: str | None = None,
    language_hint: str | None = None,
) -> list[Violation]:
    """Scan ``text`` for violations and record them in the singleton.

    Safe to call unconditionally from the agent loop: wraps detection
    and recording. Returns the (possibly empty) violation list so
    callers can log or react if desired.

    Args:
        text: The assistant's final response text.
        model: The model identifier that produced the response.
        user_id: The athlete's user id, for variant routing.
        language_hint: Optional language hint (e.g. ``"de"``). When the
            agent knows which language the athlete is using, pass it.

    Returns:
        The detected :class:`Violation` list (empty on clean response).
    """
    violations = scan_response(text, language_hint=language_hint)
    variant = resolve_variant(user_id).value
    get_prompt_metrics().record_response(
        model=model,
        variant=variant,
        text=text or "",
        violations=violations,
    )
    return violations
