"""Tests for the simplified model router.

The router routes purely on the caller-declared tier - no user identity,
no subscription lookup. This is the contract:

- tier="routine" / default / unknown -> Haiku, thinking=0
- tier="complex" -> Sonnet, thinking_budget from settings
- tier="compression" -> Haiku, thinking=0
- tier="subagent" -> Haiku, thinking=0
"""

from __future__ import annotations

import pytest

from src.agent.model_router import VALID_TIERS, ModelChoice, resolve_model
from src.config import get_settings


def test_routine_uses_haiku() -> None:
    choice = resolve_model(tier="routine")
    assert "haiku" in choice.model
    assert choice.thinking_budget == 0
    assert choice.is_premium is False


def test_complex_uses_sonnet_with_thinking() -> None:
    choice = resolve_model(tier="complex")
    assert "sonnet" in choice.model
    assert choice.thinking_budget > 0
    assert choice.is_premium is True


def test_compression_always_haiku() -> None:
    choice = resolve_model(tier="compression")
    assert "haiku" in choice.model
    assert choice.thinking_budget == 0
    assert choice.is_premium is False


def test_subagent_always_haiku() -> None:
    choice = resolve_model(tier="subagent")
    assert "haiku" in choice.model
    assert choice.thinking_budget == 0
    assert choice.is_premium is False


def test_unknown_tier_falls_back_to_routine() -> None:
    choice = resolve_model(tier="nonsense_value")
    assert "haiku" in choice.model
    assert choice.tier == "routine"
    assert choice.is_premium is False


def test_none_tier_falls_back_to_routine() -> None:
    choice = resolve_model(tier=None)
    assert "haiku" in choice.model
    assert choice.tier == "routine"


def test_modelchoice_rejects_invalid_tier() -> None:
    with pytest.raises(ValueError, match="invalid tier"):
        ModelChoice(
            model="haiku",
            thinking_budget=0,
            tier="garbage",  # type: ignore[arg-type]
            is_premium=False,
        )


def test_modelchoice_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="thinking_budget"):
        ModelChoice(
            model="haiku",
            thinking_budget=-1,
            tier="routine",
            is_premium=False,
        )


def test_valid_tiers_set() -> None:
    assert "routine" in VALID_TIERS
    assert "complex" in VALID_TIERS
    assert "compression" in VALID_TIERS
    assert "subagent" in VALID_TIERS
    assert len(VALID_TIERS) == 4


def test_complex_thinking_budget_from_settings() -> None:
    choice = resolve_model(tier="complex")
    expected = max(0, int(get_settings().sonnet_thinking_budget))
    assert choice.thinking_budget == expected
