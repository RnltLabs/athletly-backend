"""Tests for the constitutional critic (Feature 2).

Covers:
- Rule ids stay in sync between critic.py and critic_metrics.py
- JSON parsing accepts both valid shapes and tolerates malformed shapes
- Sonnet / Opus models are blocked at construction time
- Fail-open: chat_completion exceptions, timeouts, and parse errors all
  produce an `accept` result with error=True
- Pro-tier gating: default False, honors CRITIC_FORCE_PRO and the
  user_model.tier attribute
- Metrics: every action increments the right counter
- SSE: _make_sse_event produces a critic_review event
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.critic import (
    Critic,
    CriticResult,
    Violation,
    _build_user_prompt,
    _strip_code_fences,
    get_critic,
    hard_inspect,
    reset_default_critic,
    sanitize_hard,
    should_run_critic,
)
from src.api.routers.chat import _make_sse_event
from src.services.critic_metrics import (
    RULE_IDS,
    CriticMetrics,
    get_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str) -> MagicMock:
    """Build a fake LiteLLM response with the given message content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level singletons / env between tests."""
    reset_default_critic()
    get_metrics().reset()
    os.environ.pop("CRITIC_FORCE_PRO", None)
    yield
    reset_default_critic()
    get_metrics().reset()
    os.environ.pop("CRITIC_FORCE_PRO", None)


def _critic(model: str = "anthropic/claude-haiku-4-5-20251001") -> Critic:
    """Build a critic with a long timeout so tests aren't flaky."""
    return Critic(model=model, timeout_s=5.0)


# ---------------------------------------------------------------------------
# Constitution / rule ids
# ---------------------------------------------------------------------------


def test_rule_ids_count_is_eight():
    assert len(RULE_IDS) == 8


def test_rule_ids_match_design_spec():
    expected = {
        "no_em_dash",
        "no_markdown",
        "umlauts",
        "no_fabricated_stats",
        "no_premature_trends",
        "language_mirror",
        "details_before_metrics",
        "sync_then_status",
    }
    assert set(RULE_IDS) == expected


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_from_llm_json_accept_path():
    raw = '{"violations": [], "action": "accept"}'
    result = CriticResult.from_llm_json(raw, latency_ms=42)
    assert result.action == "accept"
    assert result.violations == ()
    assert result.latency_ms == 42
    assert result.error is False


def test_from_llm_json_regenerate_path():
    raw = json.dumps({
        "violations": [
            {"rule": "no_em_dash", "reason": "Em-dash at position 12."},
            {"rule": "no_markdown", "reason": "**bold** wrapper detected."},
        ],
        "action": "regenerate",
    })
    result = CriticResult.from_llm_json(raw, latency_ms=123)
    assert result.action == "regenerate"
    assert len(result.violations) == 2
    assert result.violations[0].rule == "no_em_dash"
    assert result.violation_ids() == ("no_em_dash", "no_markdown")


def test_from_llm_json_strips_code_fences():
    raw = "```json\n{\"violations\": [], \"action\": \"accept\"}\n```"
    result = CriticResult.from_llm_json(raw, latency_ms=0)
    assert result.action == "accept"


def test_from_llm_json_filters_unknown_rules():
    raw = json.dumps({
        "violations": [
            {"rule": "unknown_rule", "reason": "x"},
            {"rule": "no_em_dash", "reason": "real"},
        ],
        "action": "regenerate",
    })
    result = CriticResult.from_llm_json(raw, latency_ms=0)
    # Only the known rule survives.
    assert result.violation_ids() == ("no_em_dash",)


def test_from_llm_json_coerces_action_to_match_violations():
    # Model says accept but lists violations: trust the violations list.
    raw = json.dumps({
        "violations": [{"rule": "no_em_dash", "reason": "x"}],
        "action": "accept",
    })
    result = CriticResult.from_llm_json(raw, latency_ms=0)
    assert result.action == "regenerate"

    # Model says regenerate but lists no violations: coerce to accept.
    raw = json.dumps({"violations": [], "action": "regenerate"})
    result = CriticResult.from_llm_json(raw, latency_ms=0)
    assert result.action == "accept"


def test_from_llm_json_raises_on_empty():
    with pytest.raises(ValueError):
        CriticResult.from_llm_json("", latency_ms=0)


def test_from_llm_json_raises_on_garbage():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        CriticResult.from_llm_json("not json at all", latency_ms=0)


def test_from_llm_json_raises_on_unknown_action():
    raw = '{"violations": [], "action": "ban"}'
    with pytest.raises(ValueError):
        CriticResult.from_llm_json(raw, latency_ms=0)


def test_strip_code_fences_idempotent():
    assert _strip_code_fences("plain") == "plain"
    assert _strip_code_fences("```\nplain\n```").strip() == "plain"
    assert _strip_code_fences("```json\nplain\n```").strip() == "plain"


# ---------------------------------------------------------------------------
# Critic construction
# ---------------------------------------------------------------------------


def test_critic_blocks_sonnet_model():
    with pytest.raises(ValueError, match="Haiku-class"):
        Critic(model="anthropic/claude-sonnet-4-x")


def test_critic_blocks_opus_model():
    with pytest.raises(ValueError, match="Haiku-class"):
        Critic(model="anthropic/claude-opus-4-x")


def test_critic_accepts_haiku_model():
    c = Critic(model="anthropic/claude-haiku-4-5-20251001")
    assert c is not None


# ---------------------------------------------------------------------------
# Fail-open behaviour
# ---------------------------------------------------------------------------


def test_review_returns_accept_on_empty_response():
    c = _critic()
    result = c.review("", "user msg", [])
    assert result.action == "accept"
    assert result.error is False
    # Empty short-circuit: latency should be ~0.
    assert result.latency_ms == 0


def test_review_fail_open_on_chat_exception():
    c = _critic()
    with patch("src.agent.critic.chat_completion", side_effect=RuntimeError("boom")):
        result = c.review("Some response.", "user msg", [])
    assert result.action == "accept"
    assert result.error is True


def test_review_fail_open_on_unparseable_json():
    c = _critic()
    with patch(
        "src.agent.critic.chat_completion",
        return_value=_mock_llm_response("not json"),
    ):
        result = c.review("Some response.", "user msg", [])
    assert result.action == "accept"
    assert result.error is True


def test_review_fail_open_on_empty_content():
    c = _critic()
    with patch(
        "src.agent.critic.chat_completion",
        return_value=_mock_llm_response(""),
    ):
        result = c.review("Some response.", "user msg", [])
    assert result.action == "accept"
    assert result.error is True


def test_review_fail_open_on_timeout():
    """A critic call exceeding timeout_s falls open silently."""
    import time

    def slow_call(*args, **kwargs):
        time.sleep(2.0)
        return _mock_llm_response('{"violations": [], "action": "accept"}')

    c = Critic(model="anthropic/claude-haiku-4-5-20251001", timeout_s=0.1)
    with patch("src.agent.critic.chat_completion", side_effect=slow_call):
        result = c.review("Some response.", "user msg", [])
    assert result.action == "accept"
    assert result.error is True


# ---------------------------------------------------------------------------
# Happy path: review() flows JSON through correctly
# ---------------------------------------------------------------------------


def test_review_accept_path():
    c = _critic()
    with patch(
        "src.agent.critic.chat_completion",
        return_value=_mock_llm_response('{"violations": [], "action": "accept"}'),
    ):
        result = c.review("Clean response.", "Hello", ["get_activities"])
    assert result.action == "accept"
    assert result.error is False
    assert result.violations == ()


def test_review_regenerate_path():
    c = _critic()
    payload = json.dumps({
        "violations": [{"rule": "no_em_dash", "reason": "em-dash at pos 5"}],
        "action": "regenerate",
    })
    with patch(
        "src.agent.critic.chat_completion",
        return_value=_mock_llm_response(payload),
    ):
        result = c.review("Bad - response.", "Hello", [])
    assert result.should_regenerate
    assert result.violation_ids() == ("no_em_dash",)


# ---------------------------------------------------------------------------
# Pro-tier gating
# ---------------------------------------------------------------------------


def test_should_run_critic_disabled_by_default(monkeypatch):
    # Reset cached settings then ensure critic_enabled defaults to False.
    from src.config import get_settings
    get_settings.cache_clear()
    monkeypatch.delenv("CRITIC_ENABLED", raising=False)
    monkeypatch.delenv("CRITIC_FORCE_PRO", raising=False)

    user = SimpleNamespace(tier=None)
    assert should_run_critic(user) is False
    get_settings.cache_clear()


def test_should_run_critic_no_user_distinction(monkeypatch):
    """Critic runs for every user when the feature flag is on.

    No tier checks, no user identity. Pure global feature flag.
    """
    from src.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CRITIC_ENABLED", "true")

    free_user = SimpleNamespace(tier="free")
    pro_user = SimpleNamespace(tier="pro")
    user_without_tier = SimpleNamespace()

    assert should_run_critic(free_user) is True
    assert should_run_critic(pro_user) is True
    assert should_run_critic(user_without_tier) is True
    assert should_run_critic(None) is True
    get_settings.cache_clear()


def test_should_run_critic_master_switch_off(monkeypatch):
    from src.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CRITIC_ENABLED", "false")

    user = SimpleNamespace(tier="pro")
    assert should_run_critic(user) is False
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_record_accept():
    m = CriticMetrics()
    m.record(action="accept", violations=(), latency_ms=120)
    summary = m.summary()
    assert summary["window_calls"] == 1
    assert summary["accept_rate"] == 1.0
    assert summary["regenerate_rate"] == 0.0
    assert summary["avg_critic_latency_ms"] == 120


def test_metrics_record_regenerate_increments_rule_counters():
    m = CriticMetrics()
    m.record(
        action="regenerate",
        violations=("no_em_dash", "no_markdown"),
        latency_ms=200,
    )
    summary = m.summary()
    assert summary["regenerate_rate"] == 1.0
    assert summary["by_rule"]["no_em_dash"] == 1
    assert summary["by_rule"]["no_markdown"] == 1
    assert summary["by_rule"]["umlauts"] == 0


def test_metrics_rejects_unknown_actions():
    m = CriticMetrics()
    m.record(action="garbage", violations=(), latency_ms=10)
    # Unknown action is silently dropped.
    assert m.summary()["window_calls"] == 0


def test_metrics_summary_structure():
    m = CriticMetrics()
    m.record(action="accept", violations=(), latency_ms=10)
    m.record(action="regenerate", violations=("no_em_dash",), latency_ms=200)
    m.record(action="regenerate_failed", violations=("no_em_dash",), latency_ms=350)
    m.record(action="critic_error", violations=(), latency_ms=5)

    summary = m.summary()
    assert set(summary.keys()) == {
        "window_calls",
        "accept_rate",
        "regenerate_rate",
        "regenerate_failed_rate",
        "critic_error_rate",
        "by_rule",
        "avg_critic_latency_ms",
    }
    assert summary["window_calls"] == 4
    assert summary["accept_rate"] == 0.25
    assert summary["regenerate_rate"] == 0.25
    assert summary["regenerate_failed_rate"] == 0.25
    assert summary["critic_error_rate"] == 0.25
    assert set(summary["by_rule"].keys()) == set(RULE_IDS)


def test_metrics_empty_buffer_zeros():
    m = CriticMetrics()
    summary = m.summary()
    assert summary["window_calls"] == 0
    assert summary["accept_rate"] == 0.0
    assert summary["avg_critic_latency_ms"] == 0


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_build_user_prompt_includes_response_and_tools():
    prompt = _build_user_prompt(
        response_text="hello athlete",
        user_message="hi coach",
        tools_called=["get_activities", "get_provider_status"],
    )
    assert "hi coach" in prompt
    assert "hello athlete" in prompt
    assert "get_activities" in prompt
    assert "get_provider_status" in prompt


def test_build_user_prompt_handles_no_tools():
    prompt = _build_user_prompt(
        response_text="x",
        user_message="y",
        tools_called=[],
    )
    assert "none" in prompt.lower()


def test_build_user_prompt_truncates_long_inputs():
    # Use distinctive characters that do not appear in the prompt frame.
    long_user = "Q" * 10_000
    long_response = "Z" * 10_000
    prompt = _build_user_prompt(long_response, long_user, ["t"])
    # response capped at 4000, user message at 500.
    assert prompt.count("Q") <= 500
    assert prompt.count("Z") <= 4000
    assert prompt.count("Z") >= 1000  # not over-truncated


# ---------------------------------------------------------------------------
# Default critic singleton
# ---------------------------------------------------------------------------


def test_get_critic_returns_same_instance():
    a = get_critic()
    b = get_critic()
    assert a is b


# ---------------------------------------------------------------------------
# SSE event handling
# ---------------------------------------------------------------------------


def test_sse_critic_review_event_shape():
    payload = {
        "violations": [{"rule": "no_em_dash", "reason": "x"}],
        "annotated": True,
    }
    evt = _make_sse_event("critic_review", payload)
    assert evt.event == "critic_review"
    body = json.loads(evt.data)
    assert body["annotated"] is True
    assert body["violations"][0]["rule"] == "no_em_dash"


# ---------------------------------------------------------------------------
# CriticResult helpers
# ---------------------------------------------------------------------------


def test_critic_result_accept_factory():
    r = CriticResult.accept(latency_ms=50)
    assert r.action == "accept"
    assert r.error is False
    assert r.latency_ms == 50

    r_err = CriticResult.accept(latency_ms=10, error=True)
    assert r_err.error is True
    assert r_err.action == "accept"


def test_critic_result_to_event_payload():
    r = CriticResult(
        action="regenerate",
        violations=(
            Violation(rule="no_em_dash", reason="r1"),
            Violation(rule="no_markdown", reason="r2"),
        ),
        latency_ms=100,
    )
    payload = r.to_event_payload()
    assert payload["annotated"] is True
    assert len(payload["violations"]) == 2
    assert payload["violations"][0]["rule"] == "no_em_dash"


def test_critic_result_should_regenerate_excludes_errors():
    err = CriticResult(
        action="regenerate",
        violations=(Violation(rule="no_em_dash", reason="x"),),
        error=True,
    )
    # An error result still wears action="regenerate" only in test
    # setups; the real fail-open path always returns accept. Sanity
    # check anyway: should_regenerate must be False when error=True.
    assert err.should_regenerate is False


# ---------------------------------------------------------------------------
# Hard inspector (deterministic always-block guardrails)
# ---------------------------------------------------------------------------


def test_hard_inspect_clean_response_returns_empty():
    assert hard_inspect("All clean here.", "hello") == ()


def test_hard_inspect_em_dash_blocked():
    """U+2014 em-dash in coach response must always be flagged."""
    out = hard_inspect("Hello world, what a day.".replace(",", "—"), "hi")
    assert any(v.rule == "no_em_dash" for v in out)


def test_hard_inspect_en_dash_blocked():
    """U+2013 en-dash also counts as no_em_dash violation."""
    out = hard_inspect("Pace 4–05 min/km", "hi")
    assert any(v.rule == "no_em_dash" for v in out)


def test_hard_inspect_normal_hyphen_passes():
    """ASCII hyphen-minus (U+002D) is NEVER flagged."""
    out = hard_inspect("Pace 4-05 min/km, all good.", "hi")
    assert all(v.rule != "no_em_dash" for v in out)


def test_hard_inspect_markdown_bold_blocked():
    out = hard_inspect("Hey **athlete**, push it.", "hi")
    assert any(v.rule == "no_markdown" for v in out)


def test_hard_inspect_markdown_heading_blocked():
    out = hard_inspect("## Plan\nDo intervals.", "hi")
    assert any(v.rule == "no_markdown" for v in out)


def test_hard_inspect_markdown_italic_blocked():
    out = hard_inspect("Run *easy* today.", "hi")
    assert any(v.rule == "no_markdown" for v in out)


def test_hard_inspect_no_markdown_on_isolated_asterisk():
    """A single stray asterisk is not markdown italic."""
    out = hard_inspect("Score 5* effort.", "hi")
    assert all(v.rule != "no_markdown" for v in out)


def test_hard_inspect_ascii_umlauts_in_german_flagged():
    """German user message + ASCII translit in response = umlauts violation."""
    out = hard_inspect("Fuer dich ist heute ein Erholungstag.",
                       "Was ist das Training heute?")
    assert any(v.rule == "umlauts" for v in out)


def test_hard_inspect_ascii_umlauts_skipped_in_english_context():
    """Non-German conversation must not trip the umlauts rule."""
    out = hard_inspect("Aerobic threshold today.",
                       "What is my plan today?")
    assert all(v.rule != "umlauts" for v in out)


def test_hard_inspect_real_umlauts_in_german_passes():
    """Real umlaut chars in a German response are clean."""
    out = hard_inspect("Für dich ist heute ein Erholungstag.",
                       "Was ist das Training heute?")
    assert all(v.rule != "umlauts" for v in out)


def test_hard_inspect_allowlisted_words_pass():
    """English loan words like 'aerobic' must not trip umlauts rule."""
    out = hard_inspect("Heute aerobic Training.",
                       "Was ist das Training heute?")
    # The response says "aerobic" which is in the allowlist; should
    # not produce a umlauts violation (other rules irrelevant here).
    assert all(v.rule != "umlauts" for v in out)


def test_hard_inspect_multiple_violations_returned():
    """Multiple hard rules can fire on the same response."""
    out = hard_inspect("Heute **fuer** dich — los.",
                       "Was steht heute an?")
    rules = {v.rule for v in out}
    assert "no_em_dash" in rules
    assert "no_markdown" in rules
    # Note: umlauts may or may not fire depending on detection; the
    # key invariant is the two unambiguous ones fire.


def test_hard_inspect_handles_empty_response():
    assert hard_inspect("", "anything") == ()


def test_hard_inspect_handles_none_user_message():
    # Hard inspector must never raise even on weird inputs.
    out = hard_inspect("clean", "")
    assert out == ()


# ---------------------------------------------------------------------------
# sanitize_hard (last-resort cleaner)
# ---------------------------------------------------------------------------


def test_sanitize_hard_replaces_em_dash():
    cleaned = sanitize_hard("Pace 4—05 min/km.", "hi")
    assert "—" not in cleaned
    assert " - " in cleaned


def test_sanitize_hard_replaces_en_dash():
    cleaned = sanitize_hard("Pace 4–05 min/km.", "hi")
    assert "–" not in cleaned


def test_sanitize_hard_strips_markdown_bold():
    cleaned = sanitize_hard("Hey **athlete**, push.", "hi")
    assert "**" not in cleaned
    assert "athlete" in cleaned


def test_sanitize_hard_strips_markdown_heading():
    cleaned = sanitize_hard("## Plan\nGo.", "hi")
    assert "##" not in cleaned
    assert "Plan" in cleaned


def test_sanitize_hard_patches_german_umlauts_when_in_german_context():
    cleaned = sanitize_hard("Fuer dich heute Ruhe.",
                            "Was ist mein Training heute?")
    # Should patch the common german word "Fuer" -> "Für".
    assert "Für" in cleaned or "fuer" not in cleaned.lower()


def test_sanitize_hard_skips_umlaut_patch_in_english():
    """English context must not get umlaut patches applied."""
    cleaned = sanitize_hard("All good, push it.", "Hello coach")
    assert cleaned == "All good, push it."


def test_sanitize_hard_output_passes_hard_inspect():
    """Critical invariant: sanitize_hard output never has hard violations."""
    bad = "## Today — **fuer** dich"
    user_msg = "Was steht heute an?"
    cleaned = sanitize_hard(bad, user_msg)
    residual = hard_inspect(cleaned, user_msg)
    # After sanitization, the dash and markdown must be gone.
    assert all(v.rule not in {"no_em_dash", "no_markdown"} for v in residual)


# ---------------------------------------------------------------------------
# End-to-end regression: violations MUST block via _run_critique_pass
# ---------------------------------------------------------------------------


class _FakeMetrics:
    """Lightweight stand-in for CriticMetrics that records every call."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, int]] = []

    def record(self, action, violations, latency_ms):
        self.calls.append((action, tuple(violations), latency_ms))


@pytest.fixture
def _emit_collector():
    """Capture events emitted via emit_fn for SSE assertions."""
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    return events, emit


def test_run_critique_pass_blocks_hard_violation_via_regenerate(_emit_collector):
    """Em-dash in coach response triggers regenerate; clean rewrite ships.

    Regression for the production bug: critic detected but response shipped.
    With the two-tier flow, an em-dash MUST not survive.
    """
    import asyncio

    from src.agent.agent_loop import AsyncAgentLoop

    events, emit_fn = _emit_collector
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []

    # Stub the LLM regenerate to return a clean rewrite.
    loop.regenerate_after_critique = MagicMock(
        return_value="Today is an easy run, take it slow.",
    )

    # Stub the soft critic to accept (so we focus on the hard path).
    fake_critic = MagicMock()
    fake_critic.review.return_value = CriticResult.accept(latency_ms=100)

    fake_metrics = _FakeMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch("src.services.critic_metrics.get_metrics",
               return_value=fake_metrics):
        bad_response = "Today is an easy run — take it slow."
        result = asyncio.run(loop._run_critique_pass(
            response_text=bad_response,
            user_message="Was ist heute mein Training?",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # The em-dash must NOT survive to the user.
    assert "—" not in result
    # Regenerate was invoked.
    loop.regenerate_after_critique.assert_called_once()
    # Metrics recorded the regenerate.
    actions = [c[0] for c in fake_metrics.calls]
    assert "regenerate" in actions


def test_run_critique_pass_sanitizes_when_regenerate_still_bad(_emit_collector):
    """If the LLM regenerate ALSO produces an em-dash, sanitize and emit degraded.

    Regression: even if the model can't fix itself, we must NEVER ship a
    hard violation. The sanitizer is the last guarantee.
    """
    import asyncio

    from src.agent.agent_loop import AsyncAgentLoop

    events, emit_fn = _emit_collector
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []

    # The regenerated text STILL contains an em-dash. Sanitizer must
    # remove it before shipping.
    loop.regenerate_after_critique = MagicMock(
        return_value="Still bad — here too.",
    )

    fake_critic = MagicMock()
    fake_critic.review.return_value = CriticResult.accept(latency_ms=50)

    fake_metrics = _FakeMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch("src.services.critic_metrics.get_metrics",
               return_value=fake_metrics):
        result = asyncio.run(loop._run_critique_pass(
            response_text="Original — bad",
            user_message="Hello coach",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # No em-dash survives.
    assert "—" not in result
    # critic_review SSE event was emitted with degraded=True.
    review_events = [e for e in events if e[0] == "critic_review"]
    assert len(review_events) == 1
    assert review_events[0][1].get("degraded") is True
    # Metrics recorded regenerate_failed.
    actions = [c[0] for c in fake_metrics.calls]
    assert "regenerate_failed" in actions


def test_run_critique_pass_soft_timeout_records_critic_error(_emit_collector):
    """Soft critic timeout must record critic_error and ship original.

    Regression: prior code conflated 'error' with 'accept' in metrics
    when first.error was True. Now we record critic_error explicitly
    so the /admin/critic-stats dashboard exposes the fail-open rate.
    """
    import asyncio

    from src.agent.agent_loop import AsyncAgentLoop

    events, emit_fn = _emit_collector
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []
    loop.regenerate_after_critique = MagicMock()

    fake_critic = MagicMock()
    # Simulate a timeout: review returns accept with error=True.
    fake_critic.review.return_value = CriticResult.accept(
        latency_ms=4000, error=True,
    )

    fake_metrics = _FakeMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch("src.services.critic_metrics.get_metrics",
               return_value=fake_metrics):
        original = "Clean response, no hard violations."
        result = asyncio.run(loop._run_critique_pass(
            response_text=original,
            user_message="Hello coach",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # Original ships unchanged (soft-rule fail-open is intentional).
    assert result == original
    # critic_error is recorded, NOT accept.
    actions = [c[0] for c in fake_metrics.calls]
    assert "critic_error" in actions
    assert "accept" not in actions
    # No regenerate attempted.
    loop.regenerate_after_critique.assert_not_called()


def test_run_critique_pass_soft_violation_regenerate_then_clean(_emit_collector):
    """Soft critic flags, regenerate produces clean text, ship rewrite.

    Happy path for the soft tier.
    """
    import asyncio

    from src.agent.agent_loop import AsyncAgentLoop

    events, emit_fn = _emit_collector
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []
    loop.regenerate_after_critique = MagicMock(
        return_value="Rewritten without fabricated stats.",
    )

    fake_critic = MagicMock()
    # First call: flag fabricated_stats. Second call: accept.
    fake_critic.review.side_effect = [
        CriticResult(
            action="regenerate",
            violations=(Violation(rule="no_fabricated_stats",
                                   reason="HRV 67 mentioned without tools"),),
            latency_ms=900,
        ),
        CriticResult.accept(latency_ms=800),
    ]

    fake_metrics = _FakeMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch("src.services.critic_metrics.get_metrics",
               return_value=fake_metrics):
        result = asyncio.run(loop._run_critique_pass(
            response_text="Your HRV is 67, recover today.",
            user_message="How is my recovery?",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # The rewrite ships.
    assert result == "Rewritten without fabricated stats."
    actions = [c[0] for c in fake_metrics.calls]
    assert "regenerate" in actions
    assert "regenerate_failed" not in actions


def test_run_critique_pass_soft_violation_persists_emits_degraded(_emit_collector):
    """Soft violation that persists after regenerate ships with degraded flag.

    Regression: prior behavior was to ship without marking degraded.
    The frontend needs the degraded flag to surface a warning to the user.
    """
    import asyncio

    from src.agent.agent_loop import AsyncAgentLoop

    events, emit_fn = _emit_collector
    loop = AsyncAgentLoop.__new__(AsyncAgentLoop)
    loop._messages = []
    loop.regenerate_after_critique = MagicMock(
        return_value="Still fabricated rewrite.",
    )

    fake_critic = MagicMock()
    fake_critic.review.side_effect = [
        CriticResult(
            action="regenerate",
            violations=(Violation(rule="no_fabricated_stats", reason="x"),),
            latency_ms=900,
        ),
        CriticResult(
            action="regenerate",
            violations=(Violation(rule="no_fabricated_stats", reason="x"),),
            latency_ms=900,
        ),
    ]

    fake_metrics = _FakeMetrics()

    with patch("src.agent.critic.get_critic", return_value=fake_critic), \
         patch("src.services.critic_metrics.get_metrics",
               return_value=fake_metrics):
        result = asyncio.run(loop._run_critique_pass(
            response_text="HRV 67 today.",
            user_message="How is my recovery?",
            tool_names=[],
            emit_fn=emit_fn,
        ))

    # The rewrite ships (soft rules are judgment calls).
    assert result == "Still fabricated rewrite."
    # The critic_review event MUST carry degraded=True.
    review_events = [e for e in events if e[0] == "critic_review"]
    assert len(review_events) == 1
    assert review_events[0][1].get("degraded") is True
    actions = [c[0] for c in fake_metrics.calls]
    assert "regenerate_failed" in actions


# ---------------------------------------------------------------------------
# Config regression: timeout must be wide enough for Haiku p99
# ---------------------------------------------------------------------------


def test_critic_timeout_default_is_at_least_four_seconds():
    """Production regression: 1.5s was below Haiku 4.5 p65 latency.

    Live metrics showed avg_critic_latency_ms=2476 with 42% timeout
    failures. The default MUST be wide enough that fail-open is the
    rare exception, not the norm.
    """
    from src.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.critic_timeout_s >= 4.0
    get_settings.cache_clear()
