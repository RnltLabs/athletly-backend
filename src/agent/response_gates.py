"""Sprint D: Deterministic response-gate layer.

A small set of pure-regex policy checks that run after the LLM has
produced a draft response and before the SSE ``message`` event is
emitted. When a gate trips, the agent loop is asked to regenerate the
response once with a concrete instruction string. On second-pass fail,
the rewritten draft is shipped with a ``critic_review`` annotation.

Design contract:
- DETERMINISTIC. Every gate is a pure function over the GateContext.
  No LLM, no I/O, no global state. Mockable, auditable, fast.
- FAIL-OPEN. Any exception is caught, logged, and treated as "pass".
  A broken regex must never break the chat loop.
- METRICS. Every gate fire (pass / fail / error) is counted in the
  in-memory metrics buffer; surfaced via ``/admin/gates-stats``.
- TUNABLE. Each gate has an env feature flag, the whole layer has a
  master flag. Roll back any single gate without a deploy.

The five gates and their triggers are documented in DESIGN.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from src.config import get_settings
from src.services.gates_metrics import GATE_IDS, get_gates_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateContext:
    """Inputs to every gate. Immutable, prebuilt by the agent loop."""

    user_message: str
    response_text: str
    tools_called_this_turn: tuple[str, ...] = ()
    tools_called_recent: tuple[str, ...] = ()
    runtime_context_alerts: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate check."""

    passed: bool
    gate_id: str
    reason: str | None = None
    required_action: str | None = None


@dataclass(frozen=True)
class GateBatchResult:
    """Aggregated outcome of all gates that ran."""

    passed: bool
    failures: tuple[GateResult, ...] = ()
    combined_action: str | None = None
    ran: tuple[str, ...] = field(default_factory=tuple)

    def failure_ids(self) -> tuple[str, ...]:
        return tuple(f.gate_id for f in self.failures)

    def to_event_payload(self) -> dict:
        """Shape for the ``critic_review`` SSE event when gates persist."""
        return {
            "violations": [
                {"rule": f.gate_id, "reason": f.reason or ""}
                for f in self.failures
            ],
            "annotated": True,
            "source": "response_gates",
        }


# ---------------------------------------------------------------------------
# Regex library
# ---------------------------------------------------------------------------


# Gate 1: temporal_freshness ------------------------------------------------

_TEMPORAL_RE = re.compile(
    r"\b("
    r"eben|gerade|heute|gerade fertig|gerade zurueck|gerade zurück|"
    r"war heute|heute morgen|heute frueh|heute früh|heute mittag|"
    r"vorhin|"
    r"just got|just finished|just done|just back|today|earlier today"
    r")\b",
    re.IGNORECASE,
)

_SPORT_NOUN_RE = re.compile(
    r"\b("
    r"laufen|gelaufen|lauf|läufe|"
    r"run|running|ran|"
    r"ride|rode|rad|gefahren|fahrrad|biking|biked|cycling|"
    r"swim|swam|swum|schwimmen|geschwommen|"
    r"workout|session|einheit|training|trainiert|training session|"
    r"race|rennen|wettkampf|"
    r"track|interval|tempo|long run|lange einheit|langer lauf|long ride"
    r")\b",
    re.IGNORECASE,
)


# Gate 2: injury_persistence ------------------------------------------------

_INJURY_RE = re.compile(
    r"\b("
    r"schmerz|schmerzen|schmerzhaft|"
    r"weh|wehtut|wehgetan|tutweh|tut weh|tut\s+\w+\s+weh|"
    r"zwick|zwickt|zwicken|zwicke|"
    r"ziehen|zieht|"
    r"verletz|verletzt|verletzung|"
    r"knie|wadenheber|wade|waden|spannung|sehne|sehnen|gelenk|gelenke|"
    r"muskelkater|muskel|muskeln|"
    r"hueft|hüft|hueftbeuger|hüftbeuger|"
    r"achilles|achillessehne|"
    r"fersen|fuss|fuesse|füsse|fersensporn|"
    r"ruecken|rücken|nacken|schulter|schultern|"
    r"hurt|pain|painful|sore|soreness|tight|tightness|injury|injured|"
    r"hurting|aching|ache|"
    r"strain|strained|sprain|sprained"
    r")\b",
    re.IGNORECASE,
)


# Gate 3: stats_grounding ---------------------------------------------------

_STATS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # HR / HRV
    re.compile(r"\bHR\s*\d{2,3}\b", re.IGNORECASE),
    re.compile(r"\bHerzfrequenz[: ]\s*\d{2,3}\b", re.IGNORECASE),
    re.compile(r"\bHRV\s*\d{1,3}\s*(ms)?\b", re.IGNORECASE),
    re.compile(r"\b\d{2,3}\s*bpm\b", re.IGNORECASE),
    # Pace: "4:30/km" or "4:30 /km"
    re.compile(r"\b\d:\d{2}\s*/\s*km\b"),
    re.compile(r"\b\d:\d{2}\s*/\s*mi\b"),
    re.compile(r"\b\d[.,]\d{1,2}\s*min/km\b", re.IGNORECASE),
    # Distance: "10km", "21,1km", "5 km"
    re.compile(r"\b\d{1,3}(?:[\.,]\d{1,2})?\s*km\b", re.IGNORECASE),
    # Duration in athlete context: "1h 45min", "45 min"
    re.compile(r"\b\d{1,2}h\s*\d{1,2}\s*min\b", re.IGNORECASE),
    # Recovery score / body battery
    re.compile(r"\brecovery\s*(score)?\s*\d{1,3}\b", re.IGNORECASE),
    re.compile(r"\bbody\s*battery\s*(score)?\s*\d{1,3}\b", re.IGNORECASE),
    # TRIMP
    re.compile(r"\btrimp\s*\d{1,4}\b", re.IGNORECASE),
    # VO2max / FTP
    re.compile(r"\bVO2[mM]?ax?\s*\d{2,3}\b"),
    re.compile(r"\bFTP\s*\d{2,4}\b"),
    # Pace alternative "1:45 std", "1h45"
    re.compile(r"\b\d:\d{2}\s*std\b", re.IGNORECASE),
)

_READ_TOOLS_FOR_STATS = frozenset(
    {
        "get_activities",
        "get_activity_details",
        "get_health_summary",
        "get_recovery_alerts",
    }
)


# Gate 4: holistic_alert_acknowledged ---------------------------------------

_RECOVERY_DOMAIN_RE = re.compile(
    r"\b("
    r"schlaf|sleep|"
    r"hrv|"
    r"recovery|erholung|regeneration|regenerieren|regeneriert|"
    r"body\s*battery|"
    r"stress|stresslevel|"
    r"ruhepuls|rhr|resting\s*heart\s*rate|"
    r"müde|muede|tired|fatigue|fatigued|erschoepf|erschöpf"
    r")\b",
    re.IGNORECASE,
)


# Gate 5: language_mirror_hard ---------------------------------------------

_GERMAN_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "ich",
        "und",
        "der",
        "die",
        "das",
        "wie",
        "mit",
        "von",
        "zu",
        "ist",
        "auf",
        "ein",
        "eine",
        "einen",
        "war",
        "habe",
        "hast",
        "hat",
        "hab",
        "gerade",
        "heute",
        "fuer",
        "fur",
        "für",
        "nicht",
        "kein",
        "keine",
        "schon",
        "noch",
        "mal",
        "doch",
        "denn",
        "mein",
        "meine",
        "dein",
        "deine",
        "wir",
        "ihr",
        "sie",
        "es",
        "aber",
        "wenn",
        "auch",
        "soll",
        "muss",
        "kann",
        "wird",
        "werden",
    }
)

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _looks_german(text: str, min_hits: int = 3) -> bool:
    """Return True iff the text contains at least *min_hits* German function words."""
    if not text:
        return False
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _GERMAN_FUNCTION_WORDS)
    return hits >= min_hits


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def _gate_temporal_freshness(ctx: GateContext) -> GateResult:
    """Gate 1: just-finished sport mention must trigger garmin sync + activities."""
    if not ctx.user_message:
        return GateResult(passed=True, gate_id="temporal_freshness")

    if not (_TEMPORAL_RE.search(ctx.user_message) and _SPORT_NOUN_RE.search(ctx.user_message)):
        return GateResult(passed=True, gate_id="temporal_freshness")

    called = set(ctx.tools_called_this_turn)
    if "sync_garmin_data" in called and "get_activities" in called:
        return GateResult(passed=True, gate_id="temporal_freshness")

    return GateResult(
        passed=False,
        gate_id="temporal_freshness",
        reason=(
            "Athlete mentioned a just-completed activity but sync_garmin_data "
            "and/or get_activities was not called this turn."
        ),
        required_action=(
            "SYSTEM CHECK: The athlete mentioned a just-completed activity. "
            "You MUST call sync_garmin_data, get_provider_status, then "
            "get_activities BEFORE responding. Use the real activity data. "
            "Rewrite your response after gathering the latest activity."
        ),
    )


def _gate_injury_persistence(ctx: GateContext) -> GateResult:
    """Gate 2: body-issue mention must trigger persistence to journal or activity."""
    if not ctx.user_message:
        return GateResult(passed=True, gate_id="injury_persistence")

    if not _INJURY_RE.search(ctx.user_message):
        return GateResult(passed=True, gate_id="injury_persistence")

    called = set(ctx.tools_called_this_turn)
    persistence_tools = {"annotate_activity", "append_to_journal", "update_journal_section"}
    if called & persistence_tools:
        return GateResult(passed=True, gate_id="injury_persistence")

    return GateResult(
        passed=False,
        gate_id="injury_persistence",
        reason=(
            "Athlete reported a body issue (pain, tightness, or injury) "
            "but no persistence tool was called this turn."
        ),
        required_action=(
            "SYSTEM CHECK: The athlete reported a body issue (Schmerz, "
            "Zwicken, Knie, Wadenheber, oder ähnlich). You MUST persist "
            "this BEFORE responding via annotate_activity (latest run) AND "
            "append_to_journal(section=\"Open Threads\", ...). The next "
            "session must remember this. Rewrite your response after "
            "persisting."
        ),
    )


def _gate_stats_grounding(ctx: GateContext) -> GateResult:
    """Gate 3: specific numeric stats need a recent read-tool to ground them."""
    if not ctx.response_text:
        return GateResult(passed=True, gate_id="stats_grounding")

    if not _response_contains_stats(ctx.response_text):
        return GateResult(passed=True, gate_id="stats_grounding")

    recent = set(ctx.tools_called_recent)
    if recent & _READ_TOOLS_FOR_STATS:
        return GateResult(passed=True, gate_id="stats_grounding")

    return GateResult(
        passed=False,
        gate_id="stats_grounding",
        reason=(
            "Response references specific numeric stats (HR, pace, "
            "distance, duration, recovery, etc.) but no recent read-tool "
            "call grounds them."
        ),
        required_action=(
            "SYSTEM CHECK: Your response referenced specific numeric stats "
            "but no recent read-tool grounds them. Call get_activities, "
            "get_activity_details, or get_health_summary FIRST, then "
            "respond with real data only. Do not fabricate stats."
        ),
    )


def _response_contains_stats(text: str) -> bool:
    """Cheap multi-pattern scan for athlete-context numeric stats."""
    for pattern in _STATS_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _gate_holistic_alert(ctx: GateContext) -> GateResult:
    """Gate 4: critical / warn alerts in context must be acknowledged in the reply."""
    actionable = _filter_actionable_alerts(ctx.runtime_context_alerts)
    if not actionable:
        return GateResult(passed=True, gate_id="holistic_alert")

    if not ctx.response_text:
        # An empty response cannot acknowledge anything, but failing here
        # gives the model a chance to actually answer; otherwise the user
        # sees silence. We treat empty as a fail.
        return GateResult(
            passed=False,
            gate_id="holistic_alert",
            reason="Active recovery alert exists but response_text is empty.",
            required_action=(
                "SYSTEM CHECK: Active recovery alerts exist in your "
                "runtime context. You MUST acknowledge the relevant one "
                "(Schlaf, HRV, Recovery, Body Battery, Stress, oder "
                "Ruhepuls). Rewrite your response to address the alert."
            ),
        )

    if _RECOVERY_DOMAIN_RE.search(ctx.response_text):
        return GateResult(passed=True, gate_id="holistic_alert")

    return GateResult(
        passed=False,
        gate_id="holistic_alert",
        reason=(
            "Active recovery alert exists but response does not mention "
            "any recovery domain keyword."
        ),
        required_action=(
            "SYSTEM CHECK: Active recovery alerts exist in your runtime "
            "context. You MUST acknowledge the relevant one (Schlaf, HRV, "
            "Recovery, Body Battery, Stress, oder Ruhepuls). Rewrite your "
            "response to address the alert."
        ),
    )


def _filter_actionable_alerts(alerts: tuple[str, ...]) -> tuple[str, ...]:
    """Return only alerts whose severity prefix is critical or warn.

    Alert ids are passed in as either bare pattern ids (e.g.
    ``"sleep_low_3d"``) or as severity-prefixed strings
    (e.g. ``"critical:sleep_critical"``). Bare ids are treated as
    actionable so callers can opt in either format. Info-prefixed ids
    are filtered out.
    """
    result: list[str] = []
    for a in alerts:
        if not a:
            continue
        lower = a.lower()
        if lower.startswith("info:"):
            continue
        result.append(a)
    return tuple(result)


def _gate_language_mirror(ctx: GateContext) -> GateResult:
    """Gate 5: response language must mirror the athlete's language.

    The detector counts German function words in the user message and
    the response. German messages must produce German responses (at
    least one German function word in the reply). English messages
    must NOT produce predominantly German responses.
    """
    if not ctx.user_message or not ctx.response_text:
        return GateResult(passed=True, gate_id="language_mirror")

    user_is_german = _looks_german(ctx.user_message, min_hits=3)
    response_is_german = _looks_german(ctx.response_text, min_hits=2)

    if user_is_german and not response_is_german:
        return GateResult(
            passed=False,
            gate_id="language_mirror",
            reason=(
                "Athlete wrote in German but the response is not in German."
            ),
            required_action=(
                "SYSTEM CHECK: Mirror the athlete's language exactly. The "
                "athlete wrote in German. Reply in German. Use real "
                "umlauts (ä ö ü ß). Rewrite your "
                "response in German."
            ),
        )

    if not user_is_german and response_is_german:
        # English message, German response. Likely code-switch.
        return GateResult(
            passed=False,
            gate_id="language_mirror",
            reason=(
                "Athlete wrote in English but the response is in German."
            ),
            required_action=(
                "SYSTEM CHECK: Mirror the athlete's language exactly. The "
                "athlete wrote in English. Reply in English. Rewrite your "
                "response in English."
            ),
        )

    return GateResult(passed=True, gate_id="language_mirror")


# ---------------------------------------------------------------------------
# Gate registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """A registered gate. Holds id + config flag name + check callable.

    ``needs_tools=True`` marks a gate whose pass condition requires the
    presence of one or more tool calls in the turn. When a tool-required
    gate fails, a plain text rewrite cannot fix it; the agent loop must
    re-enter the tool loop with an explicit "REQUIRED: call X then Y"
    system reminder so the model can gather the missing data. See
    docs/sprint-f/DESIGN.md (Tool-Forcing Regenerate).
    """

    gate_id: str
    setting_attr: str
    check: Callable[[GateContext], GateResult]
    needs_tools: bool = False


GATES: tuple[Gate, ...] = (
    Gate(
        "temporal_freshness",
        "response_gate_temporal_freshness",
        _gate_temporal_freshness,
        needs_tools=True,
    ),
    Gate(
        "injury_persistence",
        "response_gate_injury_persistence",
        _gate_injury_persistence,
        needs_tools=True,
    ),
    Gate(
        "stats_grounding",
        "response_gate_stats_grounding",
        _gate_stats_grounding,
        needs_tools=True,
    ),
    Gate(
        "holistic_alert",
        "response_gate_holistic_alert",
        _gate_holistic_alert,
        needs_tools=False,
    ),
    Gate(
        "language_mirror",
        "response_gate_language_mirror",
        _gate_language_mirror,
        needs_tools=False,
    ),
)


# Map gate_id -> ordered tuple of tool names the regenerate must invoke.
# Used by ``_compose_tool_force_instruction`` to assemble the agent-facing
# system reminder when a tool-required gate fails.
TOOL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "temporal_freshness": (
        "sync_garmin_data",
        "get_provider_status",
        "get_activities",
    ),
    "injury_persistence": (
        "annotate_activity",
        "append_to_journal",
    ),
    "stats_grounding": (
        "get_activities",
        "get_activity_details",
        "get_health_summary",
    ),
}


def gate_by_id(gate_id: str) -> Gate | None:
    """Return the registered :class:`Gate` for *gate_id* or None."""
    for g in GATES:
        if g.gate_id == gate_id:
            return g
    return None


def needs_tool_forcing(failures: tuple[GateResult, ...]) -> bool:
    """Return True iff at least one *failure* belongs to a tool-required gate.

    Pure function over the failure set. Caller (agent loop) routes to
    the tool-forcing regenerate path when this returns True; otherwise
    the existing text-only regenerate path runs.
    """
    for f in failures:
        gate = gate_by_id(f.gate_id)
        if gate is not None and gate.needs_tools:
            return True
    return False


_FORCE_PREAMBLE = (
    "SYSTEM REMINDER: Your previous answer failed deterministic policy "
    "gates that require tool calls before responding. Do not apologise. "
    "Gather the missing data with the tools listed, then re-answer the "
    "athlete using only real results."
)

_FORCE_FRAGMENTS: dict[str, str] = {
    "temporal_freshness": (
        "The athlete just told you about a completed activity. "
        "REQUIRED: call sync_garmin_data, then get_provider_status, "
        "then get_activities, before re-answering. Quote only metrics "
        "that come back from get_activities."
    ),
    "injury_persistence": (
        "The athlete reported a body issue (Schmerz, Zwicken, Knie, "
        "Wadenheber, oder ähnlich). REQUIRED: call annotate_activity "
        "on the latest run AND append_to_journal(section=\"Open "
        "Threads\", ...) so the next session remembers. Then re-answer "
        "with empathy and a concrete adjustment."
    ),
    "stats_grounding": (
        "Your previous answer quoted specific numbers (HR, pace, "
        "distance, duration, recovery) with no read-tool to ground "
        "them. REQUIRED: call get_activities (or get_activity_details "
        "or get_health_summary if the question is about health) before "
        "answering. Quote only numbers that come back from the tool. "
        "If the tool returns nothing relevant, say so plainly instead "
        "of inventing data."
    ),
}

_FORCE_CLOSER = (
    "Call the tools now. The athlete is waiting. Do not skip tools. "
    "Do not invent stats."
)


def _compose_tool_force_instruction(failures: tuple[GateResult, ...]) -> str:
    """Build the synthetic user message that re-enters the tool loop.

    Concatenates the per-gate fragments for every failing tool-required
    gate, wrapped with a common preamble and closer. Idempotent over
    the failure order. Failures that are NOT tool-required are silently
    ignored here; the agent loop routes those through the text-only
    path.
    """
    fragments: list[str] = []
    seen: set[str] = set()
    for f in failures:
        gate = gate_by_id(f.gate_id)
        if gate is None or not gate.needs_tools:
            continue
        if f.gate_id in seen:
            continue
        frag = _FORCE_FRAGMENTS.get(f.gate_id)
        if frag:
            fragments.append(frag)
            seen.add(f.gate_id)

    if not fragments:
        # Defensive: caller only invokes us when ``needs_tool_forcing``
        # returned True, but a registry mismatch should not crash.
        return _FORCE_PREAMBLE + "\n\n" + _FORCE_CLOSER

    body = "\n\n".join(fragments)
    return f"{_FORCE_PREAMBLE}\n\n{body}\n\n{_FORCE_CLOSER}"


# Defensive consistency check: every gate_id is registered in metrics.
assert tuple(g.gate_id for g in GATES) == GATE_IDS, (
    "GATES order disagrees with gates_metrics.GATE_IDS"
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def is_gates_layer_enabled() -> bool:
    """Master switch read at call time so a config change applies immediately."""
    try:
        return bool(get_settings().response_gates_enabled)
    except Exception:
        return False


def _is_gate_enabled(gate: Gate) -> bool:
    """Per-gate switch. Defaults to True if the setting attr is missing."""
    try:
        return bool(getattr(get_settings(), gate.setting_attr, True))
    except Exception:
        return True


def run_gates(context: GateContext) -> GateBatchResult:
    """Run every enabled gate over *context*. Always returns a result.

    Fail-open contract: any single gate that raises is logged, counted
    as ``gate_error``, and skipped. The other gates still run. If
    ``run_gates`` itself raises (shouldn't happen), the caller is
    expected to catch and treat as passed.
    """
    metrics = get_gates_metrics()
    failures: list[GateResult] = []
    ran: list[str] = []

    for gate in GATES:
        if not _is_gate_enabled(gate):
            continue
        try:
            result = gate.check(context)
        except Exception:
            logger.warning(
                "Response gate %s raised; failing open",
                gate.gate_id,
                exc_info=True,
            )
            metrics.record(gate.gate_id, "gate_error")
            continue
        ran.append(gate.gate_id)
        if result.passed:
            metrics.record(gate.gate_id, "pass")
        else:
            metrics.record(gate.gate_id, "fail")
            failures.append(result)

    if not failures:
        return GateBatchResult(passed=True, failures=(), combined_action=None, ran=tuple(ran))

    combined = "\n\n".join(
        f.required_action for f in failures if f.required_action
    ) or None
    return GateBatchResult(
        passed=False,
        failures=tuple(failures),
        combined_action=combined,
        ran=tuple(ran),
    )


def record_regenerate_outcome(failure_ids: tuple[str, ...], success: bool) -> None:
    """Tag the regenerate cycle in metrics so /admin/gates-stats shows it.

    Called by the agent loop after the rewrite. ``failure_ids`` is the
    set of gate ids that fired the FIRST pass. ``success`` is True iff
    the second-pass run_gates returned ``passed=True``.
    """
    metrics = get_gates_metrics()
    outcome = "regenerate_success" if success else "regenerate_failed"
    for gate_id in failure_ids:
        metrics.record(gate_id, outcome)


def record_tool_forced_outcome(failure_ids: tuple[str, ...], success: bool) -> None:
    """Tag a tool-forcing regenerate cycle in metrics.

    Distinct counter from the text-only regenerate path so dashboards
    can show the lift from Sprint F directly. ``failure_ids`` is the
    set of tool-required gate ids that fired the first pass. Recorded
    in addition to (not instead of) ``record_regenerate_outcome`` so
    the existing fail-rate aggregates remain comparable.
    """
    metrics = get_gates_metrics()
    outcome = (
        "tool_forced_regenerate_success"
        if success
        else "tool_forced_regenerate_failed"
    )
    for gate_id in failure_ids:
        metrics.record(gate_id, outcome)
