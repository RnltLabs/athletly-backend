"""Regression tests for the Anthropic Extended Thinking + temperature contract.

Sprint H production bug:
    agent_loop passes ``temperature=AGENT_TEMPERATURE`` (=0.7) to every
    chat_completion call, including the ``tier="complex"`` ones that
    enable Anthropic Extended Thinking on Sonnet. Anthropic rejects
    any non-1.0 temperature when thinking is enabled with a 400. The
    fallback path in agent_loop then retries the turn on Haiku, which
    is why Lisa's "premium routing" never actually fired - every Sonnet
    attempt failed at the API boundary and silently downgraded.

Fix (src/agent/llm.py): when the resolved model is Anthropic AND the
router set ``thinking_budget > 0``, force ``temperature = 1.0`` in the
outgoing litellm kwargs, regardless of what the caller passed.

These tests pin that behaviour by monkeypatching ``litellm.completion``
and inspecting the kwargs the wrapper hands off.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.agent import llm as llm_module


class _FakeUsage:
    prompt_tokens = 1
    completion_tokens = 1
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _FakeMessage:
    content = "ok"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    """Bare-minimum object that satisfies the post-call telemetry code."""
    choices = [_FakeChoice()]
    usage = _FakeUsage()


def _capture_kwargs() -> tuple[dict[str, Any], Any]:
    """Return a (captured, completion_callable) pair.

    The captured dict accumulates the kwargs that the wrapper passes to
    ``litellm.completion`` so the assertions below can read them back.
    """
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse()

    return captured, fake_completion


def test_complex_call_forces_temperature_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """tier=complex on Anthropic must override caller temperature to 1.0."""
    captured, fake_completion = _capture_kwargs()
    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)

    llm_module.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        tier="complex",
    )

    assert "anthropic" in captured.get("model", "") or "claude" in captured.get("model", ""), \
        f"expected anthropic model for tier=complex, got {captured.get('model')}"
    assert captured.get("thinking", {}).get("type") == "enabled", \
        "thinking must be enabled for tier=complex on Anthropic"
    assert captured["temperature"] == 1.0, (
        f"temperature must be forced to 1.0 when thinking is enabled "
        f"(was {captured['temperature']})"
    )


def test_complex_call_with_existing_temperature_one_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that already passes 1.0 sees no change (still 1.0)."""
    captured, fake_completion = _capture_kwargs()
    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)

    llm_module.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=1.0,
        tier="complex",
    )

    assert captured["temperature"] == 1.0


def test_routine_call_keeps_caller_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """tier=routine on Haiku must NOT touch caller-supplied temperature."""
    captured, fake_completion = _capture_kwargs()
    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)

    llm_module.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        tier="routine",
    )

    assert captured["temperature"] == 0.7, (
        "routine calls must preserve caller temperature; thinking is off"
    )
    # Thinking must NOT be present for routine.
    assert "thinking" not in captured, \
        "thinking must NOT be set for tier=routine"


def test_thinking_block_has_router_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thinking budget on a complex call must match the router decision."""
    from src.config import get_settings

    captured, fake_completion = _capture_kwargs()
    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)

    llm_module.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        tier="complex",
    )

    settings = get_settings()
    assert captured["thinking"] == {
        "type": "enabled",
        "budget_tokens": settings.sonnet_thinking_budget,
    }


def test_explicit_model_haiku_keeps_caller_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """When an explicit model bypasses the router, no thinking is added.

    The override path used by compression/subagent calls must NOT enable
    thinking and must NOT touch the caller temperature.
    """
    captured, fake_completion = _capture_kwargs()
    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)

    llm_module.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.4,
        model="anthropic/claude-haiku-4-5-20251001",
    )

    assert captured["temperature"] == 0.4
    assert "thinking" not in captured
