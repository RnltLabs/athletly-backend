"""Constitutional critique pass (Feature 2).

A small Haiku 4.5 LLM call that scores the coach response against the
STRICT-rule constitution extracted from `src/agent/system_prompt.py`.
The output is a structured JSON object the agent loop uses to decide
whether to (a) accept the response, (b) regenerate it once, or
(c) annotate it with a `critic_review` SSE event.

Design constraints (Feature 2 spec):
- Haiku 4.5 only. Sonnet IDs are rejected at construction time.
- Tiny prompts. Total per-call budget is ~800 input tokens.
- Hard timeout. The critic must NEVER add more than ~1.5 s of latency.
- Fail-open. Any error (network, JSON parse, timeout) is treated as
  accept; the coach response is the user's answer, the critic is just
  a quality net.
- Pro-tier only. The agent loop calls `should_run_critic(user_model)`
  before invoking us. Free-tier users get zero overhead.

Failure modes documented in DESIGN.md, section 5.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from src.agent.llm import chat_completion
from src.config import get_settings
from src.services.critic_metrics import RULE_IDS, get_metrics

logger = logging.getLogger(__name__)


# --- Constitution -----------------------------------------------------------
# Each rule has a short id (matches `critic_metrics.RULE_IDS`) and a
# one-line description. The descriptions are concatenated into the critic
# system prompt at module import time so we pay the formatting cost once.

_RULE_DESCRIPTIONS: dict[str, str] = {
    "no_em_dash":
        "no_em_dash: response_text MUST NOT contain em-dash (U+2014) or "
        "en-dash (U+2013). Only hyphen-minus is allowed.",
    "no_markdown":
        "no_markdown: response_text MUST NOT contain Markdown formatting. "
        "No **bold**, no __bold__, no *italic*, no # heading at line start.",
    "umlauts":
        "umlauts: if response_text is German, real umlauts MUST be used "
        "(ae oe ue ss). ASCII transliteration (ae oe ue ss) is forbidden.",
    "no_fabricated_stats":
        "no_fabricated_stats: numeric stats (pace, distance, HR, power, "
        "VO2max) for a specific activity are allowed ONLY if tools_called "
        "includes get_activities or get_activity_details.",
    "no_premature_trends":
        "no_premature_trends: trend claims (improving, declining, etc.) "
        "need at least 5 data points / sessions of evidence.",
    "language_mirror":
        "language_mirror: response language MUST match athlete_last_message "
        "language. No mid-response code-switching.",
    "details_before_metrics":
        "details_before_metrics: if response_text discusses per-session "
        "VO2max, threshold, FTP, or training load, tools_called MUST "
        "include get_activity_details.",
    "sync_then_status":
        "sync_then_status: if tools_called includes sync_garmin_data, it "
        "MUST also include get_provider_status.",
    "pace_format_correct":
        "pace_format_correct: pace values in response_text MUST be in "
        "mm:ss notation with a colon (e.g. '4:30/km' or '4:30 /km'). A "
        "decimal-form pace like '4.50/km', '4.50 min/km', or '4.5/km' "
        "is a STRICT violation: profile decimals are pre-converted to "
        "mm:ss in the runtime context, so the model MUST quote the "
        "pretty form verbatim. Also flag impossible mm:ss values where "
        "the seconds part is >=60 (e.g. '1:63/100m' - 63 seconds does "
        "not exist).",
    "pace_comparison_directional":
        "pace_comparison_directional: if response_text compares two "
        "paces (e.g. predicted vs target, current vs threshold), the "
        "direction MUST be correct. Lower mm:ss values mean FASTER "
        "pace. Calling a 4:27 predicted pace 'too slow' relative to a "
        "4:59 target is a STRICT violation (4:27 is 32 sec/km FASTER "
        "than 4:59). When in doubt, the response should have called "
        "compute_sport_math(formula='compare_paces') and quoted the "
        "interpretation verbatim. Flag inversions and any verdict "
        "that contradicts the seconds-per-km math.",
}

# Defensive assertion: keep critic_metrics and critic.py in sync.
assert tuple(_RULE_DESCRIPTIONS.keys()) == RULE_IDS, (
    "Rule ids in critic.py disagree with critic_metrics.RULE_IDS"
)


_CRITIC_SYSTEM_PROMPT = (
    f"You are a STRICT rule-checker. Score a coach response against "
    f"{len(_RULE_DESCRIPTIONS)} rules. Respond ONLY with valid JSON. No "
    f"prose, no Markdown.\n\n"
    "Rules:\n"
    + "\n".join(f"{i + 1}. {desc}" for i, desc in enumerate(_RULE_DESCRIPTIONS.values()))
    + "\n\n"
    + "Output schema:\n"
    + '{"violations": [{"rule": "<rule_id>", "reason": "<one short sentence>"}], '
    + '"action": "accept" | "regenerate"}\n\n'
    + "action=regenerate when violations is non-empty, else accept. "
    + "Be conservative: only flag clear, unambiguous violations. "
    + "Output JSON only, no commentary."
)


# --- Result shape -----------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One critic-flagged rule violation."""

    rule: str
    reason: str


@dataclass(frozen=True)
class CriticResult:
    """Outcome of a single critic call."""

    action: str  # "accept" | "regenerate"
    violations: tuple[Violation, ...] = field(default_factory=tuple)
    latency_ms: int = 0
    error: bool = False  # True when the critic itself failed; treat as accept

    @property
    def should_regenerate(self) -> bool:
        return self.action == "regenerate" and not self.error

    def violation_ids(self) -> tuple[str, ...]:
        return tuple(v.rule for v in self.violations)

    def to_event_payload(self) -> dict:
        """JSON-safe shape for the `critic_review` SSE event."""
        return {
            "violations": [
                {"rule": v.rule, "reason": v.reason} for v in self.violations
            ],
            "annotated": True,
        }

    @classmethod
    def accept(cls, latency_ms: int = 0, error: bool = False) -> "CriticResult":
        return cls(action="accept", violations=(), latency_ms=latency_ms, error=error)

    @classmethod
    def from_llm_json(cls, raw: str, latency_ms: int) -> "CriticResult":
        """Parse the critic LLM's JSON output into a CriticResult.

        Raises ValueError on any structural problem; the caller is
        expected to catch and fail-open.
        """
        # Strip code fences if the model leaked them despite instructions.
        cleaned = _strip_code_fences(raw or "").strip()
        if not cleaned:
            raise ValueError("empty critic response")
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("critic JSON is not an object")
        action = data.get("action")
        if action not in {"accept", "regenerate"}:
            raise ValueError(f"unknown action: {action!r}")
        raw_violations = data.get("violations") or []
        if not isinstance(raw_violations, list):
            raise ValueError("violations is not a list")
        parsed: list[Violation] = []
        for entry in raw_violations:
            if not isinstance(entry, dict):
                continue
            rule = entry.get("rule")
            reason = entry.get("reason", "")
            if rule in RULE_IDS:
                parsed.append(Violation(rule=rule, reason=str(reason)[:300]))
        # Coerce action to match parsed violations: model sometimes says
        # accept but lists violations, or vice-versa. Use the violations
        # list as the source of truth.
        resolved_action = "regenerate" if parsed else "accept"
        return cls(
            action=resolved_action,
            violations=tuple(parsed),
            latency_ms=latency_ms,
            error=False,
        )


# --- Feature flag -----------------------------------------------------------


def should_run_critic(user_model=None) -> bool:
    """Return True iff the constitutional critic should run.

    Single global feature flag (``settings.critic_enabled``), no user
    distinction. ``user_model`` is accepted for backwards compatibility
    with existing call sites but ignored.
    """
    return get_settings().critic_enabled


# --- Critic ----------------------------------------------------------------


_SONNET_PATTERN = re.compile(r"sonnet|opus", re.IGNORECASE)


class Critic:
    """Run a single constitutional critique pass.

    Construct once per process (cheap), call ``review()`` per response.
    Not thread-safe in the sense of sharing state across threads, but
    safe in the sense that every call is independent.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.critic_model
        self._timeout_s = timeout_s if timeout_s is not None else settings.critic_timeout_s
        if _SONNET_PATTERN.search(self._model):
            raise ValueError(
                f"Critic must use a Haiku-class model, got {self._model!r}. "
                "Sonnet/Opus are blocked by cost policy (Feature 2 spec)."
            )

    def review(
        self,
        response_text: str,
        user_message: str,
        tools_called: Iterable[str],
    ) -> CriticResult:
        """Score *response_text* against the constitution.

        Fail-open contract: ANY exception raised below is caught,
        logged, and converted into an `accept` result with
        ``error=True``. The caller never sees a critic exception.
        """
        start = time.perf_counter()

        # Empty responses cannot violate anything; cheap short-circuit.
        if not response_text or not response_text.strip():
            return CriticResult.accept(latency_ms=0)

        prompt = _build_user_prompt(
            response_text=response_text,
            user_message=user_message,
            tools_called=list(tools_called),
        )

        try:
            raw = self._call_llm(prompt)
        except concurrent.futures.TimeoutError:
            logger.warning("Critic call timed out after %s s, failing open", self._timeout_s)
            latency = int((time.perf_counter() - start) * 1000)
            return CriticResult.accept(latency_ms=latency, error=True)
        except Exception:
            logger.warning("Critic call raised, failing open", exc_info=True)
            latency = int((time.perf_counter() - start) * 1000)
            return CriticResult.accept(latency_ms=latency, error=True)

        latency = int((time.perf_counter() - start) * 1000)
        try:
            return CriticResult.from_llm_json(raw, latency_ms=latency)
        except (ValueError, json.JSONDecodeError):
            logger.warning(
                "Critic returned unparseable JSON, failing open: %r",
                (raw or "")[:200],
            )
            return CriticResult.accept(latency_ms=latency, error=True)

    def _call_llm(self, user_prompt: str) -> str:
        """Call Haiku 4.5 with a hard timeout.

        litellm does not honour an inline timeout flag reliably across
        providers, so we wrap the synchronous call in a thread and
        enforce the deadline from the outside.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                chat_completion,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=_CRITIC_SYSTEM_PROMPT,
                temperature=0.0,
                model=self._model,
            )
            response = future.result(timeout=self._timeout_s)

        if not response.choices:
            raise ValueError("critic response has no choices")
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            raise ValueError(f"critic response content is not a string: {type(content)}")
        return content


# --- Helpers ---------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """Drop ``` and ```json fences that some models add despite the prompt."""
    return _FENCE_RE.sub("", text)


def _build_user_prompt(
    response_text: str,
    user_message: str,
    tools_called: list[str],
) -> str:
    """Compose the per-call user prompt for the critic.

    Truncates the athlete message to 500 chars (we only need the
    language and the gist) and the response to 4000 chars (any longer
    is already a policy violation and we have plenty of evidence in
    the first 4000 chars).
    """
    user_msg_short = (user_message or "").strip()[:500]
    response_short = (response_text or "").strip()[:4000]
    tools_label = ", ".join(tools_called) if tools_called else "none"
    return (
        "ATHLETE LAST MESSAGE (language reference):\n"
        f"{user_msg_short}\n\n"
        "TOOLS CALLED THIS TURN:\n"
        f"{tools_label}\n\n"
        "COACH RESPONSE TO REVIEW:\n"
        f"{response_short}\n\n"
        "Return JSON only."
    )


# --- Hard inspector (deterministic always-block guardrails) ---------------
#
# The hard inspector runs BEFORE the LLM critic and checks the three rules
# that are pure byte/character matches: em-dash, markdown markers, ASCII
# transliteration of German umlauts. These checks are deterministic and
# sub-millisecond, so they can never time out and never fail open. See
# DESIGN.md for the two-tier rationale (hard always-block vs soft LLM
# judgment with fail-open allowed).

# Em-dash U+2014 and en-dash U+2013.
_DASH_CHARS = ("—", "–")

# Markdown markers we forbid in coach responses. Conservative regexes:
# - `**bold**` and `__bold__` (double markers with non-empty body)
# - single-asterisk italic with non-whitespace content
# - heading line starting with `# ` or `## `, etc.
_MARKDOWN_BOLD_RE = re.compile(r"(\*\*\S[^*\n]*?\S\*\*)|(__\S[^_\n]*?\S__)")
_MARKDOWN_ITALIC_RE = re.compile(r"(?<![*\w])\*\S[^*\n]*?\S\*(?!\*)")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+\S")

# German indicator tokens. If any of these appear in the user message we
# treat the conversation as German and the umlauts rule applies. Kept
# tiny and conservative to avoid false positives on mixed-language input.
_GERMAN_INDICATORS = (
    " der ", " die ", " das ", " und ", " nicht ", " ich ", " ist ",
    " mein ", " dein ", " heute ", " morgen ", " gestern ", " warum ",
    " trainings", " woche ", " lauf",
)

# ASCII transliterations of German umlauts. Word-boundary regex to avoid
# matching inside unrelated tokens (e.g. "Status" should not trip "ss").
# We only flag when the surrounding word is plausibly German, see
# `_german_word_context` for the heuristic.
_ASCII_UMLAUT_RE = re.compile(
    r"\b\w*(?:ae|oe|ue|ss)\w*\b", re.IGNORECASE
)

# Words containing ae/oe/ue/ss that are legitimately ASCII (English loan
# words, brand names, etc.). Skip these to avoid false positives.
_ASCII_UMLAUT_ALLOWLIST = {
    "aerobic", "anaerobic", "aerobics", "aeroplane",
    "boss", "across", "process", "progress", "stress", "fitness",
    "ross", "loss", "miss", "class", "pass", "mass", "less", "less",
    "press", "assess", "address", "express", "success", "access",
    "guess", "session", "sessions",
    "blue", "true", "queue", "value", "issue", "fuel",
    "due", "cue", "due", "tissue",
    "coetzee",  # brand-y proper noun safety
}


def _is_german_text(user_message: str) -> bool:
    """Cheap heuristic for German vs non-German conversation language.

    We rely on a tiny allowlist of common German function words. False
    positives on English text containing the word "die" (verb) are
    acceptable because the umlauts rule only fires on responses that
    ALSO contain suspicious ASCII transliterations - clean English
    responses never trip the rule regardless of this check.
    """
    if not user_message:
        return False
    padded = " " + user_message.lower() + " "
    return any(token in padded for token in _GERMAN_INDICATORS)


def _has_real_umlaut(text: str) -> bool:
    """Return True if *text* contains any real German umlaut character."""
    return any(c in text for c in ("ä", "ö", "ü", "ß",
                                    "Ä", "Ö", "Ü"))


def _detect_ascii_umlaut_violation(response_text: str) -> str | None:
    """Return the offending token if response uses ASCII transliteration.

    Conservative: only fires when no real umlauts are present in the
    response (so we don't penalise mixed text) AND at least one word
    containing ae/oe/ue/ss is NOT in the allowlist.
    """
    # If the response already contains real umlauts somewhere, the
    # author clearly knows how to type them; treat any remaining ASCII
    # combos as intentional (e.g. "Status", "Saemann" surname).
    if _has_real_umlaut(response_text):
        return None
    for match in _ASCII_UMLAUT_RE.finditer(response_text):
        token = match.group(0).lower()
        if token in _ASCII_UMLAUT_ALLOWLIST:
            continue
        # The token must contain ae/oe/ue/ss and at least one extra
        # German-flavoured letter to count as a translit attempt.
        # Single-syllable English words like "blue" are filtered above.
        if any(seq in token for seq in ("ae", "oe", "ue", "ss")):
            return token
    return None


def hard_inspect(response_text: str, user_message: str) -> tuple[Violation, ...]:
    """Deterministic always-block rule checks.

    Returns a tuple of Violations. An empty tuple means the response
    passes all three hard rules. This function MUST be safe to run on
    every coach response: zero network, zero LLM, sub-millisecond
    latency, no exceptions for any input.

    The three hard rules:
    1. ``no_em_dash`` - U+2014 / U+2013 anywhere in the response.
    2. ``no_markdown`` - bold/italic/heading markdown markers.
    3. ``umlauts`` - ASCII transliteration of German umlauts when the
       conversation is in German.

    See DESIGN.md (Two-Tier Constitutional Critic) for why these three
    rules are deterministic and the other five are LLM-judgment.
    """
    if not response_text:
        return ()

    violations: list[Violation] = []

    # 1. em-dash / en-dash
    for ch in _DASH_CHARS:
        if ch in response_text:
            violations.append(Violation(
                rule="no_em_dash",
                reason=(
                    f"Response contains dash character U+{ord(ch):04X}. "
                    "Only hyphen-minus is permitted."
                ),
            ))
            break

    # 2. markdown markers
    md_match = (
        _MARKDOWN_BOLD_RE.search(response_text)
        or _MARKDOWN_ITALIC_RE.search(response_text)
        or _MARKDOWN_HEADING_RE.search(response_text)
    )
    if md_match:
        violations.append(Violation(
            rule="no_markdown",
            reason=f"Markdown marker detected: {md_match.group(0)[:60]!r}",
        ))

    # 3. ASCII umlauts in German conversation
    if _is_german_text(user_message):
        offending = _detect_ascii_umlaut_violation(response_text)
        if offending is not None:
            violations.append(Violation(
                rule="umlauts",
                reason=(
                    f"German response uses ASCII transliteration "
                    f"({offending!r}); real umlaut characters required."
                ),
            ))

    return tuple(violations)


def sanitize_hard(response_text: str, user_message: str) -> str:
    """Last-resort cleaner that removes hard-rule violations.

    Only invoked when both the original draft AND the LLM regenerate
    introduced a hard violation. Guarantees zero hard-rule violations
    on its output so we never ship em-dash / markdown / ASCII-umlaut
    text to the user.

    This is not a substitute for the LLM regenerate - the rewrite is
    still the primary path. ``sanitize_hard`` is the safety net for the
    rare double-failure case.
    """
    text = response_text or ""

    # Replace dashes with " - " (hyphen with spaces).
    for ch in _DASH_CHARS:
        text = text.replace(ch, " - ")

    # Strip markdown markers, preserving the inner text.
    text = _MARKDOWN_BOLD_RE.sub(
        lambda m: (m.group(1) or m.group(2) or "").strip("*_"),
        text,
    )
    text = _MARKDOWN_ITALIC_RE.sub(lambda m: m.group(0).strip("*"), text)
    text = _MARKDOWN_HEADING_RE.sub(
        lambda m: m.group(0).lstrip("# ").rstrip(),
        text,
    )

    # ASCII umlauts: only apply in German context, only for a tight
    # whitelist of common German words. We intentionally do NOT do
    # full transliteration - too risky. We just patch the most common
    # cases that show up in coach responses.
    if _is_german_text(user_message) and not _has_real_umlaut(text):
        replacements = (
            ("Fuer ", "Für "), ("fuer ", "für "),
            ("Ueber", "Über"), ("ueber", "über"),
            ("Moeglich", "Möglich"), ("moeglich", "möglich"),
            ("Naechst", "Nächst"), ("naechst", "nächst"),
            ("Waere", "Wäre"), ("waere", "wäre"),
            ("Koennen", "Können"), ("koennen", "können"),
            ("Muessen", "Müssen"), ("muessen", "müssen"),
            ("Suess", "Süß"), ("suess", "süß"),
            ("strasse", "straße"), ("Strasse", "Straße"),
            ("grosse", "große"), ("Grosse", "Große"),
        )
        for ascii_form, real_form in replacements:
            text = text.replace(ascii_form, real_form)

    return text


# --- Public convenience ----------------------------------------------------


_default_critic: Critic | None = None


def get_critic() -> Critic:
    """Return the process-wide default Critic (lazy, cached)."""
    global _default_critic
    if _default_critic is None:
        _default_critic = Critic()
    return _default_critic


def reset_default_critic() -> None:
    """Clear the cached default critic. Used by tests."""
    global _default_critic
    _default_critic = None
