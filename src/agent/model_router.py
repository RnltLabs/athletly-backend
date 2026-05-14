"""Hybrid Haiku / Sonnet model router.

Picks the model and thinking budget for a chat_completion call based on
the CALL SITE's declared complexity, not the user. The router is the
same for everyone.

Routing decision tree:

- tier == "compression" or "subagent" -> Haiku, no thinking
- tier == "complex"                   -> Sonnet, extended thinking
- otherwise (routine)                 -> Haiku, no thinking

Why call-site routing: a call that needs deep reasoning (plan
generation, reflexion synthesis, critic) needs Sonnet regardless of
who triggered it. A routine tool-loop step is fine on Haiku for
everyone. Mixing in a user-based tier check was premature
subscription scaffolding; removed.

Caches on Anthropic are model-bound. The router MUST be called once
at the top of a user turn and the same decision reused for every
chat_completion round of that turn; otherwise the cache prefix is
re-billed as fresh input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from src.config import get_settings

logger = logging.getLogger(__name__)

Tier = Literal["routine", "complex", "compression", "subagent"]
VALID_TIERS: frozenset[str] = frozenset(
    {"routine", "complex", "compression", "subagent"}
)


@dataclass(frozen=True)
class ModelChoice:
    """The resolved routing decision for one chat_completion call.

    ``thinking_budget`` of 0 means extended thinking is disabled. A
    positive value enables Anthropic Extended Thinking for that call.
    ``is_premium`` is set when the call burns Sonnet (used by telemetry
    to attribute cost).
    """
    model: str
    thinking_budget: int
    tier: Tier
    is_premium: bool

    def __post_init__(self) -> None:
        # Frozen dataclasses fall through type checks if someone passes
        # a misspelled tier. Validate at construction so a bad call
        # fails loudly here, not 200ms later inside LiteLLM.
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"invalid tier {self.tier!r}; expected one of {sorted(VALID_TIERS)}"
            )
        if self.thinking_budget < 0:
            raise ValueError("thinking_budget must be >= 0")


def _coerce_tier(value: str | None) -> Tier:
    """Validate the caller-supplied tier. Unknown -> "routine"."""
    if value in VALID_TIERS:
        return value  # type: ignore[return-value]
    return "routine"


def resolve_model(tier: str | None) -> ModelChoice:
    """Return the model + thinking budget for a single chat_completion call.

    Pure function: routing depends only on the caller-declared tier. No
    user lookup, no DB hits, no I/O.
    """
    settings = get_settings()
    coerced_tier = _coerce_tier(tier)

    # Leaf tasks: always cheap. Subagent + compression never escalate.
    if coerced_tier in ("compression", "subagent"):
        return ModelChoice(
            model=settings.haiku_model,
            thinking_budget=0,
            tier=coerced_tier,
            is_premium=False,
        )

    # Complex tier escalates to Sonnet + extended thinking. Reserved for
    # plan generation, reflexion synthesis, constitutional critic, and
    # similar reasoning-heavy operations. Routine tool-loop steps stay
    # on Haiku so the per-turn cost stays predictable.
    if coerced_tier == "complex":
        return ModelChoice(
            model=settings.sonnet_model,
            thinking_budget=max(0, int(settings.sonnet_thinking_budget)),
            tier=coerced_tier,
            is_premium=True,
        )

    # Routine: Haiku. Most calls land here.
    return ModelChoice(
        model=settings.haiku_model,
        thinking_budget=0,
        tier=coerced_tier,
        is_premium=False,
    )


def log_choice(choice: ModelChoice) -> None:
    """Emit a structured log line for the routing decision."""
    logger.info(
        "model_router decision tier=%s model=%s thinking=%d premium=%s",
        choice.tier,
        choice.model,
        choice.thinking_budget,
        choice.is_premium,
    )
