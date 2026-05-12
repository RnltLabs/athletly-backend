"""Tool hook system: pre/post hooks that fire around tool calls.

Mirrors Claude Code's PreToolUse / PostToolUse hooks. Lets the agent
loop run validation, side-effects, or block actions WITHOUT each tool
having to know about the policy.

Two kinds of hooks:

* **pre hooks** run BEFORE the tool executes. They may return a
  ``HookDecision(action="block", reason="...")`` to abort the tool call.
  The reason is fed back to the model as a tool-result-style message so
  it can react ("I can't do that because X, let me try Y").

* **post hooks** run AFTER the tool result is ready (and after the
  result is recorded). They never block. Used for cascading effects
  (e.g. "goal changed -> archive macrocycle").

Hooks register themselves on import via ``register_hook``. The agent
loop calls ``run_pre_hooks(tool_name, args, ctx)`` and
``run_post_hooks(tool_name, args, result, ctx)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookDecision:
    """Result of a pre hook."""

    action: str = "allow"  # "allow" or "block"
    reason: str = ""
    replace_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class HookContext:
    """Per-turn context passed to every hook."""

    user_id: str
    session_id: str | None
    user_model: Any
    tool_name: str


# Hook callable signatures:
#   pre:  (args, ctx) -> HookDecision | None
#   post: (args, result, ctx) -> None
PreHookFn = Callable[[dict, HookContext], "HookDecision | None"]
PostHookFn = Callable[[dict, Any, HookContext], None]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_PRE_HOOKS: dict[str, list[PreHookFn]] = {}
_POST_HOOKS: dict[str, list[PostHookFn]] = {}


def register_pre_hook(tool_name: str, fn: PreHookFn) -> None:
    """Register *fn* to run before *tool_name* executes."""
    _PRE_HOOKS.setdefault(tool_name, []).append(fn)


def register_post_hook(tool_name: str, fn: PostHookFn) -> None:
    """Register *fn* to run after *tool_name* completes."""
    _POST_HOOKS.setdefault(tool_name, []).append(fn)


def run_pre_hooks(tool_name: str, args: dict, ctx: HookContext) -> HookDecision:
    """Run all pre hooks for *tool_name*. Returns the first block or allow.

    If any hook returns block, that decision wins. Hooks may mutate-by-replace
    args via ``HookDecision(replace_args=...)`` even when action="allow".
    """
    hooks = _PRE_HOOKS.get(tool_name, [])
    current_decision = HookDecision(action="allow")
    for fn in hooks:
        try:
            decision = fn(args, ctx)
        except Exception:
            logger.exception("pre hook %s for %s failed", fn.__name__, tool_name)
            continue
        if decision is None:
            continue
        if decision.action == "block":
            return decision
        # accumulate replacements
        if decision.replace_args:
            current_decision = HookDecision(
                action="allow",
                replace_args={**(current_decision.replace_args or {}), **decision.replace_args},
            )
    return current_decision


def run_post_hooks(tool_name: str, args: dict, result: Any, ctx: HookContext) -> None:
    """Run all post hooks. Never raises - errors are logged."""
    hooks = _POST_HOOKS.get(tool_name, [])
    for fn in hooks:
        try:
            fn(args, result, ctx)
        except Exception:
            logger.exception("post hook %s for %s failed", fn.__name__, tool_name)


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------


def _post_update_profile_cascade(
    args: dict, result: Any, ctx: HookContext
) -> None:
    """When goal.event or goal.target_date changes, archive active macrocycle.

    The athlete is changing their target. The current macrocycle no longer
    matches reality; the coach should rebuild it. We archive (not delete)
    so history is preserved.
    """
    field = (args or {}).get("field", "")
    if field not in {"goal.event", "goal.target_date", "goal.target_time"}:
        return

    try:
        from src.db.client import get_supabase

        client = get_supabase()
        # Archive any active macrocycle - the agent will be prompted to rebuild
        client.table("macrocycle_plans").update({"status": "archived"}).eq(
            "user_id", ctx.user_id
        ).eq("status", "active").execute()
        logger.info(
            "Goal field %s changed for %s - archived active macrocycle",
            field, ctx.user_id,
        )
    except Exception:
        logger.exception("goal-change cascade failed")


def _post_belief_added_dedup(args: dict, result: Any, ctx: HookContext) -> None:
    """Best-effort dedup: invalidate older beliefs that now contradict.

    Stub for future implementation. Currently just logs.
    """
    if not isinstance(result, dict) or not result.get("added"):
        return
    # Future: semantic similarity + invalidate-older
    return


def install_builtin_hooks() -> None:
    """Register all built-in hooks. Called once at agent startup."""
    register_post_hook("update_profile", _post_update_profile_cascade)
    register_post_hook("add_belief", _post_belief_added_dedup)

    # Self-improvement: auto-evaluate every saved plan.
    try:
        from src.agent.self_improvement import post_save_plan_evaluation
        register_post_hook("save_plan", post_save_plan_evaluation)
    except Exception:
        logger.warning("self_improvement post_save_plan hook not installed", exc_info=True)

    # Permission system: pre-hook dangerous tools.
    try:
        from src.agent.permissions import install_permission_hooks
        install_permission_hooks()
    except Exception:
        logger.warning("permission hooks not installed", exc_info=True)
