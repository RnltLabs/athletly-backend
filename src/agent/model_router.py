"""Hybrid Haiku / Sonnet model router (Feature 1).

Picks the model and thinking budget for a chat_completion call based on:

1. Caller-declared tier ("routine" | "complex" | "compression" | "subagent")
2. User tier from :mod:`src.services.user_tier` ("free" | "pro")

Routing decision tree (kept tiny on purpose):

- tier == "compression" or "subagent" -> Haiku, no thinking.
- user is Free                          -> Haiku, no thinking.
- user is Pro and tier == "complex"     -> Sonnet, extended thinking.
- otherwise (Pro, routine)              -> Haiku, no thinking.

Why a tiny tree: every escalation rule is another way to accidentally
hit Sonnet from a code path that was budgeted for Haiku. We keep the
caller-declared tier as the only signal that can promote to Sonnet.

Caches on Anthropic are model-bound. The router MUST be called once
at the top of a user turn and the same decision reused for every
chat_completion round of that turn; otherwise the cache prefix is
re-billed as fresh input.

See ``RESEARCH.md`` for the source citations and ``DESIGN.md`` for the
cost model that justifies these defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from src.config import get_settings
from src.services.user_tier import UserTier, get_user_tier

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
    to attribute Pro-tier cost).
    """
    model: str
    thinking_budget: int
    tier: Tier
    user_tier: UserTier
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


def resolve_model(
    tier: str | None,
    user_id: str | None,
) -> ModelChoice:
    """Return the model + thinking budget for a single chat_completion call.

    ``user_id=None`` falls back to ``settings.default_user_tier``
    (cost-safe "free" by default), which keeps CLI / file-based runs
    on Haiku unless ATHLETLY_DEFAULT_TIER is set to "pro".

    The function is pure (no I/O of its own beyond the cached
    ``get_user_tier`` call) so it is safe to call from anywhere in
    the hot path.
    """
    settings = get_settings()
    coerced_tier = _coerce_tier(tier)
    user_tier_value = get_user_tier(user_id)

    # Leaf tasks: always cheap. Subagent + compression never escalate.
    if coerced_tier in ("compression", "subagent"):
        return ModelChoice(
            model=settings.haiku_model,
            thinking_budget=0,
            tier=coerced_tier,
            user_tier=user_tier_value,
            is_premium=False,
        )

    # Free users are pinned to Haiku regardless of declared tier.
    if user_tier_value != "pro":
        return ModelChoice(
            model=settings.haiku_model,
            thinking_budget=0,
            tier=coerced_tier,
            user_tier=user_tier_value,
            is_premium=False,
        )

    # Pro users: complex tier escalates to Sonnet + extended thinking.
    if coerced_tier == "complex":
        return ModelChoice(
            model=settings.sonnet_model,
            thinking_budget=max(0, int(settings.sonnet_thinking_budget)),
            tier=coerced_tier,
            user_tier=user_tier_value,
            is_premium=True,
        )

    # Pro routine: still Haiku. Sonnet is rare on purpose.
    return ModelChoice(
        model=settings.haiku_model,
        thinking_budget=0,
        tier=coerced_tier,
        user_tier=user_tier_value,
        is_premium=False,
    )


def log_choice(choice: ModelChoice, user_id: str | None) -> None:
    """Emit a structured log line for the routing decision.

    Kept separate from ``resolve_model`` so the resolution stays a pure
    function (easy to test) and so logging can be skipped in tight
    inner loops if it ever shows up in profiles.
    """
    logger.info(
        "model_router decision tier=%s user_tier=%s model=%s "
        "thinking=%d premium=%s user_id=%s",
        choice.tier,
        choice.user_tier,
        choice.model,
        choice.thinking_budget,
        choice.is_premium,
        # user_id is logged for premium calls so we can attribute cost
        # in observability. We never log it for free routine calls to
        # keep log volume down.
        user_id if choice.is_premium else "<elided>",
    )
