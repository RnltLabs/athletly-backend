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
    reset_default_critic,
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


def test_rule_ids_count_is_nine():
    # Sprint C added pace_format_correct (was 8, now 9).
    assert len(RULE_IDS) == 9


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
        "pace_format_correct",
    }
    assert set(RULE_IDS) == expected


def test_pace_format_correct_rule_description_loaded():
    """The Sprint C rule must be in the critic system prompt body."""
    from src.agent.critic import _CRITIC_SYSTEM_PROMPT

    assert "pace_format_correct" in _CRITIC_SYSTEM_PROMPT
    # Sanity: the description names mm:ss as the correct form so the
    # critic LLM can flag decimal-pace responses.
    assert "mm:ss" in _CRITIC_SYSTEM_PROMPT
    assert "4:30" in _CRITIC_SYSTEM_PROMPT or "4.50" in _CRITIC_SYSTEM_PROMPT


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
