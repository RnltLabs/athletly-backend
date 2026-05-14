"""Constitutional critique pass (Feature 2).

A small Haiku 4.5 LLM call that scores the coach response against an
8-rule constitution extracted from `src/agent/system_prompt.py`. The
output is a structured JSON object the agent loop uses to decide
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
}

# Defensive assertion: keep critic_metrics and critic.py in sync.
assert tuple(_RULE_DESCRIPTIONS.keys()) == RULE_IDS, (
    "Rule ids in critic.py disagree with critic_metrics.RULE_IDS"
)


_CRITIC_SYSTEM_PROMPT = (
    "You are a STRICT rule-checker. Score a coach response against 8 "
    "rules. Respond ONLY with valid JSON. No prose, no Markdown.\n\n"
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


# --- Pro-tier gating --------------------------------------------------------


def should_run_critic(user_model) -> bool:
    """Return True iff the constitutional critic should run for this user.

    Today this is a simple gate. Feature 5/6 will replace the stub with
    real subscription-store lookups.
    """
    settings = get_settings()
    if not settings.critic_enabled:
        return False
    # Dev override: CRITIC_FORCE_PRO=1 or critic_force_pro=true in settings.
    if settings.critic_force_pro or os.environ.get("CRITIC_FORCE_PRO") == "1":
        return True
    return _is_pro_tier(user_model)


def _is_pro_tier(user_model) -> bool:
    """Stub: return whether the user has an active Pro subscription.

    Until Feature 5 wires the subscription store, we look for a
    `tier` attribute on the user model and treat "pro" / "premium" as
    Pro. Anything else (including None) is Free.
    """
    tier = getattr(user_model, "tier", None)
    if isinstance(tier, str):
        return tier.lower() in {"pro", "premium"}
    return False


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
