"""Core Agent Loop -- the heart of AgenticSports.

This is the equivalent of Claude Code's main loop:
    while not done:
        response = LLM(system_prompt, messages, tools)
        if response has tool_calls -> execute tools
        else -> return text response

Design principles (from Claude Code):
    1. Simple loop + disciplined tools = controllable autonomy
    2. The model decides what to do -- no router, no pre-determined pipeline
    3. Tools are atomic and composable -- each does one thing well
    4. Context window IS working memory -- everything stays in messages
    5. Self-correction is natural -- model sees tool results and adjusts

Gaps addressed:
    Gap 1 -- Session Persistence: _save_turn() / _load_session()
    Gap 2 -- Context Compression: _compress_history()
    Gap 3b -- Belief Extraction Safety Net: _post_turn_extraction_check()
    Gap 4b -- Onboarding Completion Check: _check_onboarding_complete()
"""

import asyncio
import json
import logging
import os
import queue
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.complexity_detector import needs_complex_reasoning
from src.agent.hooks import (
    HookContext,
    install_builtin_hooks,
    run_post_hooks,
    run_pre_hooks,
)
from src.agent.llm import MODEL, chat_completion
from src.agent.tools.registry import ToolRegistry, get_default_tools, get_restricted_tools
from src.agent.tools.truncation import execute_with_budget
from src.agent.system_prompt import STATIC_SYSTEM_PROMPT, build_runtime_context
from src.config import get_settings

# Register built-in hooks once at module import.
install_builtin_hooks()

logger = logging.getLogger(__name__)

# -- Configuration -----------------------------------------------------------

MAX_TOOL_ROUNDS = 25          # Safety limit
MAX_CONSECUTIVE_ERRORS = 3    # Stop if tools keep failing
AGENT_TEMPERATURE = 0.7       # Creative for coaching, lower for analysis
# Compression threshold tuned down from 40 to 12: with prompt caching
# active, the cheapest way to keep total_input low is to purge stale
# tool_result payloads aggressively. Keeping the last 4 user-turns of
# full tool-call rounds preserves the model's working memory while
# slashing per-call input tokens for long sessions.
COMPRESSION_THRESHOLD = 12
COMPRESSION_KEEP_ROUNDS = 4   # Keep last N full tool-call rounds verbatim
TOOL_CALL_SUMMARY_THRESHOLD = 8  # Inject summary reminder after N consecutive tool rounds
# Sprint F: Tool-Forcing Regenerate. Bounded forced tool loop that runs
# AFTER a tool-required gate failure. One LLM call per round; at most
# this many rounds per turn. Three is enough for: round 1 model calls
# tools, round 2 model may chain a follow-up tool, round 3 model writes
# the final text. Hard ceiling, never bypassed.
_FORCED_REGEN_MAX_ROUNDS = 3
# When purging stale tool_result payloads inside a single user turn, keep
# this many of the most recent rounds intact. Older tool results get
# replaced with a short outcome string while preserving message shape
# (role, tool_call_id) so the API still validates the conversation.
TOOL_RESULT_PURGE_KEEP_TURNS = 4

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

# JSONL trace files (additive to Supabase / JSONL session storage).
# Used by athctl trace tail/replay/diff.
TRACES_DIR = Path(__file__).parent.parent.parent / "logs" / "traces"


# -- Types -------------------------------------------------------------------

@dataclass
class AgentTurn:
    """A single turn in the agent loop."""
    role: str          # "user", "model", "tool_call", "tool_result"
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    duration_ms: int = 0


@dataclass
class AgentResult:
    """Result of processing one user message."""
    response_text: str
    turns: list[AgentTurn] = field(default_factory=list)
    tool_calls_made: int = 0
    total_duration_ms: int = 0
    onboarding_just_completed: bool = False


# Callback for UI updates: (event_type, detail) -> None
ProgressCallback = Callable[[str, str], None] | None


_MAX_LOADED_PAIRS = 16  # Max user/assistant pairs to keep from session history


def _extract_user_assistant_pairs(rows: list[dict], source: str = "supabase") -> list[dict]:
    """Build a well-formed user/assistant message list from raw DB rows.

    Raw session rows contain user, model, and tool_call entries.  When tool-call
    rounds don't produce a final model response (e.g. the agent hit an error
    loop), the loaded history ends up with consecutive user messages and no
    assistant replies.  This confuses LLMs (especially Gemini) which expect
    alternating user/assistant turns.

    This function:
    1. Collects tool_call results as context between user turns
    2. Ensures every user message is followed by an assistant message
    3. Synthesizes brief assistant placeholders where model responses are missing
    4. Limits output to the last _MAX_LOADED_PAIRS user/assistant pairs
    """
    messages: list[dict] = []
    pending_tool_summaries: list[str] = []

    for row in rows:
        role = row.get("role", "user")
        content = row.get("content", "") or ""

        if role == "tool_call":
            # Collect tool results as context — summarize briefly
            meta = row.get("meta", {}) or {}
            tool_name = meta.get("tool", "tool")
            # Extract a brief summary of the result
            preview = content[:150]
            pending_tool_summaries.append(f"{tool_name}: {preview}")
            continue

        if role not in ("user", "model"):
            continue

        oai_role = "user" if role == "user" else "assistant"

        if oai_role == "user" and messages and messages[-1]["role"] == "user":
            # Consecutive user message — insert a synthetic assistant reply
            # to maintain alternating turn structure
            if pending_tool_summaries:
                summary = "[I processed your request using these tools: " + "; ".join(
                    s[:80] for s in pending_tool_summaries[-5:]
                ) + "]"
                pending_tool_summaries.clear()
            else:
                summary = "[I processed your previous message.]"
            messages.append({"role": "assistant", "content": summary})

        messages.append({"role": oai_role, "content": content})
        if oai_role == "assistant":
            pending_tool_summaries.clear()

    # If the conversation ends with a user message, that's fine —
    # the agent loop will add the new user message after this

    # Limit to last N pairs to prevent context explosion on long sessions
    if len(messages) > _MAX_LOADED_PAIRS * 2:
        messages = messages[-(_MAX_LOADED_PAIRS * 2):]
        # Ensure we start with a user message
        while messages and messages[0]["role"] != "user":
            messages.pop(0)

    return messages


# -- The Loop ----------------------------------------------------------------

class AgentLoop:
    """Core agent loop -- Claude Code architecture for fitness coaching.

    Usage:
        loop = AgentLoop(user_model=model)
        result = loop.process_message("Create a training plan for my marathon")
        print(result.response_text)
    """

    def __init__(
        self,
        user_model,
        tool_registry: ToolRegistry | None = None,
        on_progress: ProgressCallback = None,
        startup_context: str | None = None,
        context: str = "coach",
        max_rounds: int | None = None,
        trace_to_file: bool = False,
    ):
        self.user_model = user_model
        self.tools = tool_registry or get_default_tools(user_model, context=context)
        self.on_progress = on_progress

        # Observability: log resolved tool-loadout + approximate token weight
        # so we can monitor the impact of CORE_TOOL_NAMES / defer_loading.
        # Uses a cheap len(json)/4 heuristic (close enough for tracking).
        try:
            from src.agent.tools.registry import CORE_TOOL_NAMES
            _all_tools = self.tools.get_openai_tools(defer_non_core=False)
            _deferred = self.tools.get_openai_tools(defer_non_core=True)
            _core_count = sum(
                1 for t in _all_tools
                if t.get("function", {}).get("name") in CORE_TOOL_NAMES
            )
            _full_weight = len(json.dumps(_all_tools)) // 4
            # Deferred tools strip descriptions/parameters on the wire; the
            # name-only stub is ~5 tokens each. Core tools keep full schema.
            _slim_weight = sum(
                len(json.dumps(t)) // 4 if not t.get("defer_loading") else 5
                for t in _deferred
            )
            logger.info(
                "agent_tools_loaded context=%s total=%d core=%d deferred=%d "
                "tokens_full~%d tokens_slim~%d savings~%d%%",
                context,
                len(_all_tools),
                _core_count,
                len(_all_tools) - _core_count,
                _full_weight,
                _slim_weight,
                int(round((1 - _slim_weight / _full_weight) * 100)) if _full_weight else 0,
            )
        except Exception as _e:  # pragma: no cover - observability only
            logger.debug("tool_loadout_log_failed: %s", _e)
        self.startup_context = startup_context
        self.context = context
        self._max_rounds = max_rounds or MAX_TOOL_ROUNDS

        # JSONL trace appender (additive to existing session persistence).
        # Enabled via constructor flag OR env var ATHLETLY_TRACE_AGENT=1.
        self._trace_to_file = trace_to_file or (
            os.environ.get("ATHLETLY_TRACE_AGENT", "").strip() in {"1", "true", "TRUE", "yes"}
        )

        # Detect persistence mode
        self._settings = get_settings()
        self._use_supabase = self._settings.use_supabase

        # Resolve user_id: prefer user_model (multi-tenant API), fall back to settings (CLI)
        self._user_id: str = getattr(user_model, "user_id", None) or self._settings.agenticsports_user_id

        # Conversation history (persists across messages within a session)
        # Uses OpenAI-compatible message format: list of dicts
        self._messages: list[dict] = []

        # Active context compression: consecutive tool-call rounds without user response
        self._consecutive_tool_calls: int = 0

        # Session metadata
        self._session_id: str | None = None
        self._session_file: Path | None = None  # file-based mode only
        self._turns_this_session: int = 0

        # Sprint D: deterministic response gates.
        # ``_recent_tools_window`` holds the set of tool names called in
        # each of the last 3 user turns. Used by Gate 3 (stats_grounding)
        # which lets a recent read-tool carry over multiple turns.
        # ``_active_alert_ids`` is set per-turn from the runtime context
        # builder; used by Gate 4 (holistic_alert).
        # ``_gates_regenerated_this_turn`` bounds regenerate to once per
        # turn even if the rewrite still fails gates.
        self._recent_tools_window: deque[frozenset[str]] = deque(maxlen=3)
        self._active_alert_ids: tuple[str, ...] = ()
        self._gates_regenerated_this_turn: bool = False

    # -- JSONL Trace File (athctl trace) -----------------------------------

    def _append_trace_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Append a single JSONL event to logs/traces/{session_id}.jsonl.

        Additive to existing on_progress callbacks and session message
        persistence. Only writes when ``self._trace_to_file`` is True and
        a session has started. Failures are swallowed (logged) so the
        agent loop is never blocked by trace I/O.

        Event schema::

            {"ts": iso8601, "type": "llm_call|tool_call|tool_result|response", "data": {...}}
        """
        if not self._trace_to_file or not self._session_id:
            return
        try:
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "type": event_type,
                "data": data or {},
            }
            path = TRACES_DIR / f"{self._session_id}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.warning("Failed to append trace event %s", event_type, exc_info=True)

    # -- Session Persistence (Gap 1) ----------------------------------------

    def start_session(self, resume_session_id: str | None = None) -> str:
        """Start a new coaching session or resume an existing one."""
        if resume_session_id:
            return self._load_session(resume_session_id)

        self._messages = []
        self._turns_this_session = 0

        if self._use_supabase:
            from src.db import create_session
            self._session_id = create_session(
                self._user_id,
                context=self.context,
            )
        else:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            self._session_id = datetime.now().strftime("session_%Y-%m-%d_%H%M%S")
            self._session_file = SESSIONS_DIR / f"{self._session_id}.jsonl"

        return self._session_id

    def _save_turn(self, role: str, content: str, metadata: dict | None = None):
        """Persist a single turn to the session store (Supabase or JSONL)."""
        if not self._session_id:
            return

        if self._use_supabase:
            from src.db import save_message
            try:
                save_message(
                    session_id=self._session_id,
                    user_id=self._user_id,
                    role=role,
                    content=content[:4000],
                    meta=metadata,
                )
            except Exception:
                logger.warning("Failed to save turn to Supabase", exc_info=True)
        else:
            if not self._session_file:
                return
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "role": role,
                "content": content[:4000],
            }
            if metadata:
                entry["meta"] = metadata
            self._session_file.parent.mkdir(parents=True, exist_ok=True)
            with self._session_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_session(self, session_id: str) -> str:
        """Reload a previous session's history into self._messages."""
        self._session_id = session_id
        self._messages = []
        self._turns_this_session = 0

        if self._use_supabase:
            from src.db import get_session, load_session_messages
            session_row = get_session(session_id)
            if session_row and "context" in session_row:
                self.context = session_row["context"]
            rows = load_session_messages(session_id)
            raw_pairs = _extract_user_assistant_pairs(rows, source="supabase")
        else:
            self._session_file = SESSIONS_DIR / f"{session_id}.jsonl"
            if not self._session_file.exists():
                logger.warning("Session file %s not found, starting fresh", session_id)
                return session_id
            entries = []
            for line in self._session_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            raw_pairs = _extract_user_assistant_pairs(entries, source="jsonl")

        for msg in raw_pairs:
            self._messages.append(msg)
            if msg["role"] == "user":
                self._turns_this_session += 1

        logger.info("Resumed session %s with %d messages", session_id, len(self._messages))
        return session_id

    # -- Context Window Compression (Gap 2) ----------------------------------

    def _compress_history(self):
        """Compress older tool-call rounds into text summaries.

        When self._messages exceeds COMPRESSION_THRESHOLD, older rounds
        are replaced with a single user message summarizing what happened.
        """
        if len(self._messages) <= COMPRESSION_THRESHOLD:
            return

        # Find round boundaries (each user message starts a new round)
        round_starts = []
        for i, m in enumerate(self._messages):
            if m["role"] == "user" and m.get("content"):
                # Skip tool-result "messages" (they have role "tool")
                round_starts.append(i)

        if len(round_starts) <= COMPRESSION_KEEP_ROUNDS + 1:
            return

        keep_from = round_starts[-COMPRESSION_KEEP_ROUNDS]

        # Build summary of old rounds
        summaries = []
        for msg in self._messages[:keep_from]:
            if msg["role"] == "assistant":
                # Check for tool calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args_preview = fn.get("arguments", "")[:80]
                        summaries.append(f"- Called {name}({args_preview})")
                elif msg.get("content"):
                    text = msg["content"].strip()
                    if text:
                        summaries.append(f"- Responded: {text[:120]}...")

        if not summaries:
            return

        summary_text = (
            "[COMPRESSED HISTORY -- earlier conversation rounds]\n"
            + "\n".join(summaries[:30])
        )

        compressed_msg = {"role": "user", "content": summary_text}

        old_count = len(self._messages)
        self._messages = [compressed_msg] + self._messages[keep_from:]
        logger.info(
            "Compressed history: %d messages -> %d (kept last %d rounds)",
            old_count, len(self._messages), COMPRESSION_KEEP_ROUNDS,
        )

    def _purge_old_tool_results(self):
        """Truncate stale tool_result payloads to short outcome strings.

        Different from ``_compress_history`` (which rewrites the whole
        prefix). This runs INSIDE the tool loop of a single user turn:
        after appending fresh tool results we walk back through the
        message list, identify ``role="tool"`` messages whose content is
        full JSON and which sit more than ``TOOL_RESULT_PURGE_KEEP_TURNS``
        user-turns ago, and shrink their content to a short summary while
        keeping ``tool_call_id`` intact (so the conversation still
        validates).

        Idempotent: a message already purged starts with the purge
        marker and is skipped on subsequent passes.
        """
        if not self._messages:
            return

        # Locate user-turn boundaries to count "turns ago".
        user_turn_indices = [
            i for i, m in enumerate(self._messages)
            if m.get("role") == "user"
        ]
        if len(user_turn_indices) <= TOOL_RESULT_PURGE_KEEP_TURNS:
            return

        # Anything before this index is older than the keep-window.
        cutoff_idx = user_turn_indices[-TOOL_RESULT_PURGE_KEEP_TURNS]

        new_messages: list[dict] = []
        purged_count = 0
        for idx, msg in enumerate(self._messages):
            if (
                idx < cutoff_idx
                and msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and not msg["content"].startswith("[purged tool_result")
            ):
                preview = msg["content"][:100]
                purged = {
                    **msg,
                    "content": (
                        f"[purged tool_result - outcome: {preview}]"
                    ),
                }
                new_messages.append(purged)
                purged_count += 1
            else:
                new_messages.append(msg)

        if purged_count > 0:
            self._messages = new_messages
            logger.info(
                "Purged %d old tool_result payloads (kept last %d user turns)",
                purged_count, TOOL_RESULT_PURGE_KEEP_TURNS,
            )

    # -- Post-Turn Safety Nets (Gap 3b, Gap 4b) -----------------------------

    def _post_turn_extraction_check(self, response_text: str):
        """Lightweight safety net for belief extraction (Gap 3b)."""
        profile = self.user_model.project_profile()

        if not profile.get("name") or profile.get("name") == "Athlete":
            import re
            name_patterns = [
                r"(?:Hallo|Hi|Hey|Hello|Servus|Moin)\s+([A-Z][a-zu\u00e4\u00f6\u00fc]{2,})",
            ]
            for pattern in name_patterns:
                match = re.search(pattern, response_text)
                if match:
                    logger.warning(
                        "POST-TURN CHECK: Agent greeted '%s' but profile.name is empty. "
                        "The agent should have called update_profile(field='name').",
                        match.group(1),
                    )
                    break

    def _check_onboarding_complete(self) -> bool:
        """Check if onboarding info is complete (Gap 4b).

        Only runs in onboarding context to avoid spurious events in coach mode.
        """
        if self.context != "onboarding":
            return False

        if self.user_model.meta.get("_onboarding_complete", False):
            return False  # already done, skip

        profile = self.user_model.project_profile()
        constraints = profile.get("constraints", {})

        required = [
            bool(profile.get("name") and profile.get("name") != "Athlete"),
            bool(profile.get("sports")),
            bool(profile.get("goal", {}).get("event")),
            bool(constraints.get("training_days_per_week")),
            bool(constraints.get("max_session_minutes")),
        ]

        if all(required):
            self.user_model.meta = {**self.user_model.meta, "_onboarding_complete": True}
            self.user_model.save()
            logger.info("ONBOARDING COMPLETE: All required profile fields gathered.")
            return True

        return False

    # -- Main Entry Point ----------------------------------------------------

    def process_message(self, user_message: str) -> AgentResult:
        """Process one user message through the agent loop.

        This is the MAIN ENTRY POINT -- equivalent to Claude Code's
        core loop. The model sees all tools and decides what to do.
        """
        start_time = time.time()
        result = AgentResult(response_text="")

        # System prompt is STATIC -- identical for all users/requests.
        # LLM providers cache this automatically when it never changes.
        system_prompt = STATIC_SYSTEM_PROMPT

        # Sprint D: reset the per-turn regenerate guard. Each user turn
        # gets at most ONE regenerate from the response-gate layer.
        self._gates_regenerated_this_turn = False

        # Sprint D: snapshot any active critical/warn alerts so Gate 4
        # (holistic_alert_acknowledged) can check the draft response.
        # Best-effort; failure produces an empty tuple and the gate is a no-op.
        self._active_alert_ids = _detect_active_alert_ids_safe(self.user_model)

        # Compress history before adding new message (Gap 2)
        self._compress_history()

        # Build the per-turn runtime context ONCE per user message.
        # The same string is passed as `runtime_context` to every
        # chat_completion call in the tool loop so the second system
        # block stays byte-stable inside this turn (cache write on
        # round 1, cache read on every subsequent round). Mutating
        # runtime data here would invalidate the breakpoint and re-bill
        # the whole prefix as fresh input.
        runtime_ctx = build_runtime_context(
            user_model=self.user_model,
            date=None,
            startup_context=self.startup_context,
            context=self.context,
            user_message=user_message,
        )

        # If the last loaded message is "user", insert a synthetic assistant
        # reply to maintain alternating turn structure for Gemini.
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.append({
                "role": "assistant",
                "content": "[Continuing from our previous conversation.]",
            })

        # The user message no longer carries the runtime context inline.
        # Anthropic gets runtime_ctx as a dedicated cacheable system
        # block; non-Anthropic providers get it merged in by
        # chat_completion as a prefix on the first user message.
        self._messages.append({
            "role": "user",
            "content": user_message,
        })

        # Persist user turn (Gap 1)
        self._save_turn("user", user_message)

        # -- Complexity routing (Sprint E) -----------------------------------
        # Deterministic detector picks the chat_completion tier for THIS turn
        # at message receipt. Sticky for the entire tool loop so the
        # Anthropic prompt cache prefix is not invalidated mid-turn.
        # Cross-sport, race-strategy, sport-math, and long-horizon plan
        # queries route to Sonnet + extended thinking. Routine turns stay
        # on Haiku. See src/agent/complexity_detector.py for the rule.
        is_complex, matched_keywords = needs_complex_reasoning(user_message)
        selected_tier = "complex" if is_complex else "routine"
        if is_complex:
            logger.info(
                "complexity_detector: complex_triggered keywords=%s user_id=%s",
                matched_keywords, self._user_id,
            )
        else:
            logger.debug(
                "complexity_detector: routine keywords=%s", matched_keywords,
            )
        # Tracks whether we already retried complex -> routine in this turn.
        # When True we prepend a user-visible annotation to the final reply
        # explaining that deep analysis was unavailable.
        complex_fallback_used = False

        # Get tool declarations in OpenAI format for LiteLLM
        # Anthropic native tool_search: only CORE_TOOL_NAMES descriptions
        # are visible in the prompt; everything else is deferred. Saves
        # ~80% of tool-layer tokens. See registry.CORE_TOOL_NAMES.
        openai_tools = self.tools.get_openai_tools(defer_non_core=True)

        consecutive_errors = 0
        last_content = None  # Track last assistant content for MAX_TOOL_ROUNDS fallback

        # -- THE LOOP --
        for round_num in range(self._max_rounds):

            # Trace: LLM call about to start (additive to on_progress)
            self._append_trace_event("llm_call", {
                "round": round_num,
                "message_count": len(self._messages),
                "model": MODEL,
            })

            # Call LLM with conversation history + tools via LiteLLM.
            # selected_tier is pinned at message receipt by the
            # complexity_detector (Sprint E). Cross-sport / sport-math /
            # race-strategy turns route to Sonnet + extended thinking,
            # everything else stays on Haiku. Mid-turn switching is
            # intentionally NOT supported here (it would invalidate the
            # Anthropic prompt cache; see DESIGN.md).
            try:
                response = chat_completion(
                    messages=self._messages,
                    system_prompt=system_prompt,
                    tools=openai_tools if openai_tools else None,
                    temperature=AGENT_TEMPERATURE,
                    runtime_context=runtime_ctx,
                    tier=selected_tier,
                )
            except Exception as exc:
                # Sonnet failure (rate limit exhausted, overload, etc.)
                # falls back to Haiku once. The athlete still gets an
                # answer; the response is annotated so they know deep
                # analysis was unavailable. Non-complex turns re-raise
                # so the existing error-handling path takes over.
                if selected_tier != "complex":
                    raise
                logger.warning(
                    "complex chat_completion failed (%s), retrying as routine",
                    exc,
                    exc_info=True,
                )
                complex_fallback_used = True
                selected_tier = "routine"
                response = chat_completion(
                    messages=self._messages,
                    system_prompt=system_prompt,
                    tools=openai_tools if openai_tools else None,
                    temperature=AGENT_TEMPERATURE,
                    runtime_context=runtime_ctx,
                    tier=selected_tier,
                )

            # Track usage (non-blocking, fire-and-forget)
            try:
                from src.services.usage_tracker import track_usage
                track_usage(self._user_id, response)
            except Exception:
                pass  # Non-critical — never block agent loop

            # Guard against empty choices (Gemini 2.5 can return empty responses)
            if not response.choices:
                consecutive_errors += 1
                logger.warning(
                    "LLM returned empty choices (round %d, errors: %d, msgs: %d), retrying...",
                    round_num, consecutive_errors, len(self._messages),
                )

                # Context may be too large — trim older tool results before retrying
                if consecutive_errors == 2 and len(self._messages) > 20:
                    self._compress_history()
                    logger.info("Compressed history after empty choices (msgs now: %d)", len(self._messages))

                if consecutive_errors >= 5:
                    result.response_text = "Es tut mir leid, ich habe gerade technische Schwierigkeiten. Bitte versuche es nochmal."
                    break  # fall through to _save_turn below
                time.sleep(0.3 * consecutive_errors)  # brief backoff
                continue

            message = response.choices[0].message
            content = message.content
            tool_calls = message.tool_calls

            # Forward reasoning content as thinking event (NanoBot pattern)
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "thinking", None)
            if reasoning and self.on_progress:
                self.on_progress("thinking", str(reasoning)[:500])

            # Handle empty response (edge case -- Gap 9c).
            # Token-burn fix: cap empty-response retries at 1 (was 3) so a
            # broken turn does not triple our per-minute input token spend.
            if not content and not tool_calls:
                consecutive_errors += 1
                logger.warning(
                    "Empty response from LLM (round %d, errors: %d), retrying once with tool-free fallback...",
                    round_num, consecutive_errors,
                )

                # Single retry: drop tools entirely and ask for plain text.
                logger.info("Falling back to tool-free call after empty response...")
                fallback_prompt = (
                    STATIC_SYSTEM_PROMPT
                    + "\n\n# IMPORTANT OVERRIDE\n"
                    "Tools are temporarily unavailable. Respond ONLY with "
                    "natural conversational text. Do NOT list, reference, or "
                    "simulate any tool calls (no update_profile, update_journal_section, "
                    "get_activities, etc.). Just answer the athlete directly "
                    "as a coach would in a normal conversation."
                )
                time.sleep(2.0)  # let any transient provider hiccup clear
                fallback_response = chat_completion(
                    messages=self._messages,
                    system_prompt=fallback_prompt,
                    temperature=AGENT_TEMPERATURE,
                    runtime_context=runtime_ctx,
                    tier=selected_tier,
                )
                fb_message = fallback_response.choices[0].message
                if fb_message.content:
                    result.response_text = fb_message.content.strip()
                    self._messages.append({
                        "role": "assistant",
                        "content": result.response_text,
                    })
                    break

                # Fallback also failed - bail with a friendly message.
                result.response_text = (
                    "I had trouble generating a response. "
                    "Could you rephrase your message?"
                )
                break

            if not tool_calls:
                # -- MODEL DECIDED TO RESPOND --
                result.response_text = (content or "").strip()

                # Reset consecutive tool-call counter (model is responding)
                self._consecutive_tool_calls = 0

                # Append model response to history
                self._messages.append({"role": "assistant", "content": result.response_text})

                if self.on_progress:
                    self.on_progress("responding", result.response_text[:100])

                break
            else:
                # -- MODEL WANTS TO USE TOOLS --
                # Filter out Anthropic SERVER-SIDE tools (web_search, tool_search,
                # code_execution). Their tool_use_id has the ``srvtoolu_`` prefix
                # and Anthropic handles their results internally - we must NOT echo
                # them back as tool_result blocks, or the next request fails with
                # "unexpected tool_use_id". Server tools are billed and run server
                # side; we only manage CLIENT-side function tools.
                client_tool_calls = [
                    tc for tc in tool_calls
                    if not str(getattr(tc, "id", "") or "").startswith("srvtoolu_")
                ]

                # Append the assistant message with tool_calls to history.
                # Only include client-managed function tool_calls.
                assistant_msg: dict = {"role": "assistant", "content": content or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in client_tool_calls
                ]
                # If only server tools fired, the assistant message has no
                # tool_calls. That is valid - the model will continue without
                # client-side tool execution. We still record content.
                if not assistant_msg["tool_calls"]:
                    assistant_msg.pop("tool_calls")
                self._messages.append(assistant_msg)

                # Track last content for MAX_TOOL_ROUNDS fallback
                if content:
                    last_content = content

                # No client tools to execute - server tool result is already in
                # the model's content, loop back for the next LLM call.
                if not client_tool_calls:
                    continue

                # Execute each client tool call
                sent_in_turn = False
                for tc in client_tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        tool_args = {}

                    if self.on_progress:
                        self.on_progress(
                            "tool_call",
                            f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})",
                        )

                    # Trace: tool call about to execute
                    self._append_trace_event("tool_call", {
                        "tool": tool_name,
                        "args": tool_args,
                        "round": round_num,
                    })

                    result.tool_calls_made += 1
                    tool_start = time.time()

                    # Build hook context for this tool invocation.
                    hook_ctx = HookContext(
                        user_id=self._user_id,
                        session_id=self._session_id,
                        user_model=self.user_model,
                        tool_name=tool_name,
                    )

                    # Pre-hook: may block the tool call or rewrite its args.
                    pre_decision = run_pre_hooks(tool_name, tool_args, hook_ctx)
                    if pre_decision.action == "block":
                        tool_result = {
                            "blocked": True,
                            "reason": pre_decision.reason or "Pre-hook blocked this call.",
                        }
                        if self.on_progress:
                            self.on_progress(
                                "tool_error",
                                f"{tool_name} blocked: {pre_decision.reason}",
                            )
                        self._append_trace_event("tool_result", {
                            "tool": tool_name,
                            "ok": False,
                            "blocked": True,
                            "reason": pre_decision.reason,
                        })
                    else:
                        effective_args = (
                            {**tool_args, **pre_decision.replace_args}
                            if pre_decision.replace_args
                            else tool_args
                        )
                        # Execute the tool (with budget-aware truncation).
                        try:
                            tool_result = execute_with_budget(
                                self.tools, tool_name, effective_args,
                            )
                            consecutive_errors = 0

                            if self.on_progress:
                                preview = json.dumps(tool_result, ensure_ascii=False)[:200]
                                self.on_progress("tool_result", f"{tool_name} -> {preview}")

                            self._append_trace_event("tool_result", {
                                "tool": tool_name,
                                "ok": True,
                                "result_preview": json.dumps(tool_result, ensure_ascii=False, default=str)[:500],
                            })

                            # Post-hook: cascading effects after the tool result.
                            run_post_hooks(tool_name, effective_args, tool_result, hook_ctx)

                        except Exception as e:
                            tool_result = {"error": str(e)}
                            consecutive_errors += 1

                            if self.on_progress:
                                self.on_progress("tool_error", f"{tool_name} -> Error: {e}")

                            self._append_trace_event("tool_result", {
                                "tool": tool_name,
                                "ok": False,
                                "error": str(e),
                            })

                    # Check if tool already sent a push notification
                    if isinstance(tool_result, dict) and tool_result.get("_sent_in_turn"):
                        sent_in_turn = True

                    # Forward inline action card requests as SSE events.
                    # Tools opt in by returning ``_action_request`` in their
                    # result dict (see ``action_tools.py``). The marker is
                    # stripped from the dict that the model sees so it does
                    # not pollute the chat history with internal plumbing.
                    if isinstance(tool_result, dict):
                        action_request = tool_result.pop("_action_request", None)
                        if action_request and self.on_progress:
                            try:
                                self.on_progress(
                                    "action_request",
                                    json.dumps(action_request, ensure_ascii=False),
                                )
                            except Exception:
                                logger.warning(
                                    "Failed to forward action_request from %s",
                                    tool_name,
                                    exc_info=True,
                                )

                    # Forward generative UI component renders as SSE events.
                    # Tools opt in by returning ``_ui_component`` in their
                    # result dict (see ``ui_tools.py``). The marker is
                    # stripped from the dict that the model sees so it does
                    # not pollute the chat history with internal plumbing.
                    if isinstance(tool_result, dict):
                        ui_component = tool_result.pop("_ui_component", None)
                        if ui_component and self.on_progress:
                            try:
                                self.on_progress(
                                    "ui_component",
                                    json.dumps(ui_component, ensure_ascii=False),
                                )
                            except Exception:
                                logger.warning(
                                    "Failed to forward ui_component from %s",
                                    tool_name,
                                    exc_info=True,
                                )

                    tool_duration = int((time.time() - tool_start) * 1000)

                    # Record turn
                    result.turns.append(AgentTurn(
                        role="tool_call",
                        content=json.dumps(tool_result, ensure_ascii=False),
                        tool_name=tool_name,
                        tool_args=tool_args,
                        duration_ms=tool_duration,
                    ))

                    # Persist tool call (Gap 1)
                    self._save_turn("tool_call", json.dumps(tool_result, ensure_ascii=False)[:2000], {
                        "tool": tool_name,
                        "args": tool_args,
                        "duration_ms": tool_duration,
                    })

                    # Append tool result to history (OpenAI format)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

                # If a tool already sent a push notification, suppress final LLM response
                if sent_in_turn:
                    result.response_text = ""
                    break

                # Tool-result purge: shrink stale tool_result payloads
                # to short outcome strings before the next LLM call. Runs
                # every round so the per-call total_input stays bounded
                # even during long tool chains.
                self._purge_old_tool_results()

                # Active context compression: track consecutive tool-call rounds
                self._consecutive_tool_calls += 1
                if self._consecutive_tool_calls >= TOOL_CALL_SUMMARY_THRESHOLD:
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "[System: You have made 8+ consecutive tool calls. "
                            "Summarize your findings before continuing.]"
                        ),
                    })
                    self._consecutive_tool_calls = 0
                    logger.info(
                        "Injected active context compression reminder at round %d",
                        round_num,
                    )

                # Safety: too many consecutive errors
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    result.response_text = (
                        "I encountered multiple errors while processing your request. "
                        "Let me try a different approach -- could you rephrase what you need?"
                    )
                    break

        else:
            # Hit MAX_TOOL_ROUNDS -- return partial response
            if last_content:
                result.response_text = last_content.strip()
            else:
                result.response_text = (
                    "I spent a lot of time analyzing your request but need to wrap up. "
                    "Here's what I found so far -- feel free to ask follow-up questions."
                )

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        self._turns_this_session += 1

        # Annotate the response when we fell back from complex -> routine
        # mid-turn. The athlete sees an explicit notice so they know the
        # answer is from simpler reasoning. German UTF-8 with real umlauts.
        if complex_fallback_used and result.response_text:
            annotation = (
                "[Hinweis: Tiefere Analyse war kurz nicht verfügbar, "
                "hier meine beste Antwort mit einfachem Reasoning.]\n\n"
            )
            if not result.response_text.startswith(annotation):
                result.response_text = annotation + result.response_text

        # Sprint D: push this turn's tool names into the rolling window so
        # Gate 3 (stats_grounding) sees the last 3 user-turns of tools.
        turn_tools = frozenset(
            t.tool_name for t in result.turns
            if t.tool_name
        )
        self._recent_tools_window.append(turn_tools)

        # Persist agent response (Gap 1)
        self._save_turn("model", result.response_text, {
            "tool_calls": result.tool_calls_made,
            "duration_ms": result.total_duration_ms,
        })

        # Trace: final response (after the loop completes)
        self._append_trace_event("response", {
            "text_preview": (result.response_text or "")[:500],
            "tool_calls_made": result.tool_calls_made,
            "duration_ms": result.total_duration_ms,
        })

        # Prompt rule-violation telemetry (Feature 6, fire-and-forget).
        # Scans the final response for STRICT rule violations from
        # system_prompt.py (em-dashes, Markdown, ASCII transliteration in
        # German). Pure regex, zero LLM cost. Never blocks the agent loop.
        try:
            from src.services.prompt_metrics import scan_and_record
            scan_and_record(
                text=result.response_text or "",
                model=MODEL,
                user_id=self._user_id,
            )
        except Exception:
            pass  # Non-critical, never block agent loop

        # Post-turn safety nets (Gap 3b, Gap 4b)
        self._post_turn_extraction_check(result.response_text)
        result.onboarding_just_completed = self._check_onboarding_complete()

        return result

    def inject_context(self, role: str, text: str):
        """Inject context into the conversation (e.g., startup greeting)."""
        oai_role = "assistant" if role == "model" else role
        self._messages.append({"role": oai_role, "content": text})
        self._save_turn(role, text, {"injected": True})

    # -- Constitutional Critique Regeneration (Feature 2) -------------------

    def regenerate_after_critique(
        self,
        original_response: str,
        violations: list[dict],
    ) -> str:
        """Rewrite *original_response* so it no longer violates *violations*.

        Used by the constitutional critic (src/agent/critic.py) when the
        first-draft coach response failed the 8-rule check. The retry is
        a SINGLE tool-free LLM call against the existing conversation
        history, so cost is bounded at one extra completion (no extra
        tool rounds).

        The bad first-draft is NOT removed from ``self._messages``; we
        feed it back to the model as context and ask for a fix. This is
        cheaper than rerunning the full tool loop and avoids any risk
        of triggering a different tool path.

        Args:
            original_response: The response text the critic rejected.
                Used in the prompt so the model sees what to fix.
            violations: A list of {"rule": str, "reason": str} dicts
                from the critic. Inlined verbatim into the prompt.

        Returns:
            The rewritten response text. May be empty if the LLM
            returns nothing; the caller should fall back to
            ``original_response`` in that case.
        """
        violation_lines = "\n".join(
            f"- {v.get('rule')}: {v.get('reason', '')}"[:300]
            for v in violations
        )
        retry_prompt = (
            "SYSTEM CHECK: your previous response violated these "
            "constitution rules:\n"
            f"{violation_lines}\n\n"
            "Rewrite your previous response so it fixes ALL listed "
            "violations while keeping the substantive content and "
            "answer to the athlete intact. Mirror the athlete's "
            "language. Do NOT call any tools. Output ONLY the rewritten "
            "response, no preamble, no apology, no meta-comment."
        )
        retry_messages = list(self._messages) + [
            {"role": "user", "content": retry_prompt}
        ]
        runtime_ctx = build_runtime_context(
            user_model=self.user_model,
            date=None,
            startup_context=self.startup_context,
            context=self.context,
        )
        try:
            response = chat_completion(
                messages=retry_messages,
                system_prompt=STATIC_SYSTEM_PROMPT,
                temperature=AGENT_TEMPERATURE,
                runtime_context=runtime_ctx,
            )
        except Exception:
            logger.warning("Critique-driven regenerate raised, keeping original", exc_info=True)
            return original_response

        if not response.choices:
            return original_response
        msg = response.choices[0].message
        new_text = (getattr(msg, "content", None) or "").strip()
        if not new_text:
            return original_response

        # Splice the corrected response into the persisted history so
        # downstream summarization / replay sees the user-facing answer,
        # not the rejected draft. The bad draft itself stays in
        # self._messages for the model's short-term memory of this turn
        # (it informs follow-up turns to avoid the same mistake).
        self._messages.append({"role": "assistant", "content": new_text})
        return new_text


# -- Async Agent Loop (SSE streaming) ----------------------------------------

# Sentinel object placed in the sync queue to signal that the worker thread
# has finished (either successfully or with an error).
_DONE_SENTINEL = object()


class AsyncAgentLoop(AgentLoop):
    """Async wrapper around AgentLoop that streams progress events via SSE.

    The synchronous ``process_message()`` runs in a dedicated thread.
    A thread-safe ``queue.Queue`` bridges progress events from the worker
    thread to an async consumer that calls ``emit_fn``.

    Usage::

        loop = AsyncAgentLoop(user_model=model)
        loop.start_session()

        async def emit(event_type, data):
            ...  # write SSE frame to response

        result = await loop.process_message_sse("I want a marathon plan", emit)
    """

    async def process_message_sse(
        self,
        user_message: str,
        emit_fn: Callable,
    ) -> AgentResult:
        """Process a user message and stream progress events via *emit_fn*.

        Args:
            user_message: The user's chat message.
            emit_fn: An async callable ``(event_type: str, data: dict) -> None``.
                     Called for each agent progress event.

        Returns:
            The ``AgentResult`` from the underlying synchronous loop.

        The following event types are emitted:

        - ``thinking``    — LLM call starting (shows a spinner in the UI).
        - ``tool_hint``   — Tool is about to be executed (name + args).
        - ``tool_result`` — Tool has finished (name + preview of result).
        - ``tool_error``  — Tool execution raised an exception.
        - ``message``     — Final coach reply text.
        - ``done``        — Stream is complete (always emitted last).
        - ``error``       — Unhandled exception in the worker thread.
                            Error responses are NEVER persisted to session
                            history (critical invariant).
        """
        sync_queue: queue.Queue = queue.Queue()
        result_holder: list[AgentResult | BaseException] = []

        # Stable id for grouping all tool calls of this single turn.
        # Frontend uses this to collapse multiple tool_hint events into
        # one collapsible status entry in the chat.
        current_group_id = uuid.uuid4().hex

        # -- Bridge: on_progress callback -> sync queue --
        # The worker thread emits (event_type, detail) pairs; we also
        # carry the current_group_id so the consumer can attach it to
        # the SSE payload without crossing threads.
        def _bridge_progress(event_type: str, detail: str) -> None:
            sync_queue.put((event_type, detail, current_group_id))

        # Swap out on_progress for the duration of this call.
        original_on_progress = self.on_progress
        self.on_progress = _bridge_progress

        # Snapshot the tool registry for display_label lookups inside
        # the consumer (read-only, safe to share across threads).
        tool_registry = self.tools

        # -- Worker: runs synchronous process_message in a thread --
        def _worker() -> None:
            try:
                agent_result = self.process_message(user_message)
                result_holder.append(agent_result)
            except Exception as exc:
                result_holder.append(exc)
            finally:
                sync_queue.put(_DONE_SENTINEL)

        # -- Async consumer: reads from sync queue and calls emit_fn --
        async def _consume() -> None:
            while True:
                try:
                    item = await asyncio.to_thread(
                        sync_queue.get, True, 0.05  # blocking=True, timeout=50ms
                    )
                except queue.Empty:
                    continue

                if item is _DONE_SENTINEL:
                    return

                event_type, detail, group_id = item
                data = _progress_to_sse_data(
                    event_type,
                    detail,
                    group_id=group_id,
                    tool_registry=tool_registry,
                )
                try:
                    await emit_fn(event_type, data)
                except Exception:
                    logger.warning(
                        "emit_fn raised while emitting %s event", event_type, exc_info=True
                    )

        # Emit the group_start sentinel BEFORE the worker begins so the
        # frontend can open a collapsible status panel for this turn.
        try:
            await emit_fn("tool_group_start", {"group_id": current_group_id})
        except Exception:
            logger.warning("emit_fn raised on tool_group_start", exc_info=True)

        # Run worker and consumer concurrently.
        worker_task = asyncio.to_thread(_worker)
        consumer_task = asyncio.create_task(_consume())

        try:
            await asyncio.gather(worker_task, consumer_task)
        finally:
            # Restore the original callback regardless of outcome.
            self.on_progress = original_on_progress

        # Drain any leftover items that arrived after _DONE_SENTINEL
        # (shouldn't happen, but defensive drain is cheap).
        _drain_sync_queue(sync_queue)

        # Unwrap result or re-raise as SSE error (never persist error).
        if not result_holder:
            await emit_fn("error", {"message": "Agent returned no result", "code": "no_result"})
            # Close the group even on error so the UI does not hang.
            try:
                await emit_fn("tool_group_end", {"group_id": current_group_id})
            except Exception:
                logger.warning("emit_fn raised on tool_group_end (no_result)", exc_info=True)
            raise RuntimeError("Agent returned no result")

        outcome = result_holder[0]
        if isinstance(outcome, BaseException):
            logger.exception("AsyncAgentLoop worker raised", exc_info=outcome)
            await emit_fn(
                "error",
                {
                    "message": "An internal error occurred. Please try again.",
                    "code": "internal_error",
                },
            )
            try:
                await emit_fn("tool_group_end", {"group_id": current_group_id})
            except Exception:
                logger.warning("emit_fn raised on tool_group_end (exception)", exc_info=True)
            raise outcome

        final_text = outcome.response_text or ""
        unique_tools = _unique_tool_names(outcome)

        # -- Sprint D: deterministic response-gate layer ----------------
        # Runs BEFORE the Sprint A constitutional critic. Cheap (regex),
        # catches the tool-discipline failure modes that the LLM cannot
        # judge reliably. On gate fail, regenerates ONCE; if the rewrite
        # still fails, annotates with a ``critic_review`` SSE event.
        if final_text and _gates_layer_enabled_safe():
            try:
                final_text = await self._run_gates_pass(
                    response_text=final_text,
                    user_message=user_message,
                    tool_names=unique_tools,
                    emit_fn=emit_fn,
                )
                outcome.response_text = final_text
            except Exception:
                logger.warning("Gates pass raised, accepting original", exc_info=True)

        # -- Constitutional critique (Feature 2) -------------------------
        # Pro-tier only, master-switch gated. Fails open on any error so
        # the user never sees a critic-induced failure. See critic.py for
        # the rule list and decision logic.
        if final_text and _should_run_critic_safe(self.user_model):
            try:
                final_text = await self._run_critique_pass(
                    response_text=final_text,
                    user_message=user_message,
                    tool_names=unique_tools,
                    emit_fn=emit_fn,
                )
                # Mirror any rewrite back into the AgentResult so callers
                # (push notifications, response logging) see the final
                # user-facing text, not the rejected draft.
                outcome.response_text = final_text
            except Exception:
                logger.warning("Critique pass raised, accepting original", exc_info=True)

        # Sprint G: unconditional deterministic finalize step.
        # Every SSE ``message`` payload MUST be free of hard-rule
        # violations regardless of CRITIC_ENABLED / gates state. The
        # finalize step is sub-millisecond, no I/O, no LLM, and
        # idempotent on already-clean text.
        if final_text:
            final_text = _finalize_response(final_text, user_message)
            outcome.response_text = final_text
            await emit_fn("message", {"text": final_text})

        # Close the tool group after the final response so the UI can
        # mark this turn's status panel as complete.
        try:
            await emit_fn("tool_group_end", {"group_id": current_group_id})
        except Exception:
            logger.warning("emit_fn raised on tool_group_end", exc_info=True)

        return outcome

    # -- Constitutional critique hook (Feature 2) ----------------------------

    async def _run_critique_pass(
        self,
        response_text: str,
        user_message: str,
        tool_names: list[str],
        emit_fn: Callable,
    ) -> str:
        """Score *response_text* against the constitution and act.

        Two-tier decision logic (DESIGN.md, Two-Tier Constitutional Critic):

        Stage 1 - Hard inspector (deterministic, always-block).
            Sub-millisecond regex checks for em-dash, markdown, ASCII
            umlauts. If any hard rule fires we MUST block: regenerate
            once, re-inspect, and as last resort sanitize the rewrite
            to guarantee zero hard violations on the shipped text.

        Stage 2 - Soft critic (LLM, fail-open allowed).
            For the five judgment-based rules (fabricated_stats,
            premature_trends, language_mirror, details_before_metrics,
            sync_then_status). A timeout / parse error here records
            ``critic_error`` and ships the original; the soft rules are
            advisory and we keep latency bounded.

        Metrics are recorded for every decision so /admin/critic-stats
        stays accurate even when the critic fails.
        """
        from src.agent.critic import (
            get_critic,
            hard_inspect,
            sanitize_hard,
        )
        from src.services.critic_metrics import get_metrics

        critic = get_critic()
        metrics = get_metrics()

        # -- Stage 1: hard inspector ---------------------------------------
        hard_violations = hard_inspect(response_text, user_message)
        if hard_violations:
            response_text = await self._handle_hard_violations(
                response_text=response_text,
                user_message=user_message,
                hard_violations=hard_violations,
                metrics=metrics,
                emit_fn=emit_fn,
                sanitize=sanitize_hard,
                hard_inspect_fn=hard_inspect,
            )
            # The handler returns text that is guaranteed clean on hard
            # rules. We still run the soft critic on it below so the
            # soft-rule policy is enforced too.

        # -- Stage 2: soft critic -----------------------------------------
        first = await asyncio.to_thread(
            critic.review,
            response_text,
            user_message,
            tool_names,
        )

        if first.error:
            metrics.record(
                action="critic_error",
                violations=(),
                latency_ms=first.latency_ms,
            )
            return response_text
        if first.action == "accept":
            metrics.record(
                action="accept",
                violations=(),
                latency_ms=first.latency_ms,
            )
            return response_text

        # Soft violation: one regenerate attempt.
        rewritten = await asyncio.to_thread(
            self.regenerate_after_critique,
            response_text,
            [{"rule": v.rule, "reason": v.reason} for v in first.violations],
        )

        # Re-inspect hard rules on the rewrite. If the regenerate
        # introduced new hard violations (em-dash, markdown, ascii
        # umlauts) we sanitize before going further. This protects the
        # invariant: NO hard violation ever ships.
        rewrite_hard = hard_inspect(rewritten, user_message)
        if rewrite_hard:
            sanitized = sanitize_hard(rewritten, user_message)
            # Defence in depth: if even the sanitizer somehow left a
            # hard violation, log loudly and ship the sanitized version
            # anyway - it cannot be worse than the input.
            still_hard = hard_inspect(sanitized, user_message)
            if still_hard:
                logger.error(
                    "sanitize_hard left residual hard violations: %s",
                    [v.rule for v in still_hard],
                )
            rewritten = sanitized

        # Second soft pass on the (now hard-clean) rewrite.
        second = await asyncio.to_thread(
            critic.review,
            rewritten,
            user_message,
            tool_names,
        )
        if second.error or second.action == "accept":
            metrics.record(
                action="regenerate",
                violations=first.violation_ids(),
                latency_ms=first.latency_ms + second.latency_ms,
            )
            return rewritten

        # Still flagged on soft rules after regenerate. Soft rules are
        # LLM-judgment so we ship the rewrite but mark it with a
        # critic_review event carrying ``degraded=True`` so the frontend
        # can surface a "this answer may contain inaccuracies" notice
        # rather than silently shipping potentially fabricated content.
        metrics.record(
            action="regenerate_failed",
            violations=second.violation_ids(),
            latency_ms=first.latency_ms + second.latency_ms,
        )
        payload = second.to_event_payload()
        payload["degraded"] = True
        try:
            await emit_fn("critic_review", payload)
        except Exception:
            logger.warning("emit_fn raised on critic_review", exc_info=True)
        return rewritten

    async def _handle_hard_violations(
        self,
        *,
        response_text: str,
        user_message: str,
        hard_violations: tuple,
        metrics,
        emit_fn: Callable,
        sanitize: Callable,
        hard_inspect_fn: Callable,
    ) -> str:
        """Sprint A: Block-and-regenerate for deterministic hard violations.

        Contract: returns text with ZERO hard-rule violations. Never
        fails open. Uses one LLM regenerate, then a deterministic
        sanitizer as last resort.
        """
        rewritten = await asyncio.to_thread(
            self.regenerate_after_critique,
            response_text,
            [{"rule": v.rule, "reason": v.reason} for v in hard_violations],
        )
        rewrite_hard = hard_inspect_fn(rewritten, user_message)
        if not rewrite_hard:
            metrics.record(
                action="regenerate",
                violations=tuple(v.rule for v in hard_violations),
                latency_ms=0,
            )
            return rewritten

        sanitized = sanitize(rewritten, user_message)
        metrics.record(
            action="regenerate_failed",
            violations=tuple(v.rule for v in rewrite_hard),
            latency_ms=0,
        )
        payload = {
            "violations": [
                {"rule": v.rule, "reason": v.reason} for v in rewrite_hard
            ],
            "annotated": True,
            "degraded": True,
        }
        try:
            await emit_fn("critic_review", payload)
        except Exception:
            logger.warning("emit_fn raised on critic_review", exc_info=True)
        return sanitized

    # -- Sprint D: deterministic gate regenerate ---------------------------

    def regenerate_after_gates(
        self,
        original_response: str,
        gate_action: str,
    ) -> str:
        """Rewrite *original_response* with a gate-driven instruction.

        Different from :meth:`regenerate_after_critique` in two ways:

        1. The instruction comes from a deterministic gate (regex), so
           it may explicitly REQUIRE tool calls (e.g. sync_garmin_data,
           annotate_activity). We therefore allow the LLM to call tools
           in this retry, unlike the critic retry which forbids them.
        2. We use the same conversation history but inject the gate
           instruction as a single user message before the retry.

        Tool calls during regenerate are NOT executed here. The caller
        only takes the text answer; if the model wants to call tools,
        we instruct it to do so on the NEXT turn (the regenerate ask
        is text only). This keeps the regenerate cost bounded to one
        completion call.

        Returns the rewritten text, or *original_response* on any
        failure.
        """
        retry_prompt = (
            "SYSTEM CHECK: your previous response failed deterministic "
            "policy gates:\n\n"
            f"{gate_action}\n\n"
            "Rewrite your previous response so it fixes ALL listed "
            "issues. Mirror the athlete's language exactly. Use real "
            "German umlauts (ä ö ü ß) if the athlete wrote German. "
            "Output ONLY the rewritten response text, no preamble, no "
            "apology, no meta-commentary. Do not call tools in this "
            "rewrite step; if the gate requires fresh data, say so "
            "transparently in the response and the next turn will "
            "gather it."
        )
        retry_messages = list(self._messages) + [
            {"role": "user", "content": retry_prompt}
        ]
        runtime_ctx = build_runtime_context(
            user_model=self.user_model,
            date=None,
            startup_context=self.startup_context,
            context=self.context,
        )
        try:
            response = chat_completion(
                messages=retry_messages,
                system_prompt=STATIC_SYSTEM_PROMPT,
                temperature=AGENT_TEMPERATURE,
                runtime_context=runtime_ctx,
            )
        except Exception:
            logger.warning(
                "Gate-driven regenerate raised, keeping original",
                exc_info=True,
            )
            return original_response

        if not response.choices:
            return original_response
        msg = response.choices[0].message
        new_text = (getattr(msg, "content", None) or "").strip()
        if not new_text:
            return original_response

        # Splice the corrected response into the persisted history so
        # downstream summarisation / replay sees the user-facing answer,
        # not the rejected draft.
        self._messages.append({"role": "assistant", "content": new_text})
        return new_text

    # -- Sprint D: deterministic gate pass ---------------------------------

    async def _run_gates_pass(
        self,
        response_text: str,
        user_message: str,
        tool_names: list[str],
        emit_fn: Callable,
    ) -> str:
        """Run the deterministic response gates over *response_text*.

        Decision logic (Sprint D + Sprint F):
        - All gates pass -> return original.
        - One or more gates fail AND no regenerate yet this turn:
          - If at least one failure is a tool-required gate
            (Sprint F): run the tool-forcing regenerate path, which
            re-enters the tool loop with an explicit "REQUIRED: call
            X then Y" system reminder.
          - Otherwise: text-only rewrite (Sprint D path A).
          Re-check gates after either path. If still failing, emit
          ``critic_review`` SSE event for observability but ship the
          new text.
        - Two-pass cap. We never regenerate twice in one turn.

        Fail-open: any exception bubbles up to the caller's try/except.
        Inside this method we catch defensively and treat as accept.
        """
        from src.agent.response_gates import (
            GateContext,
            needs_tool_forcing,
            record_regenerate_outcome,
            record_tool_forced_outcome,
            run_gates,
        )

        if self._gates_regenerated_this_turn:
            # Already regenerated this turn (defensive). Do not enter a
            # loop; ship what we have.
            return response_text

        context = GateContext(
            user_message=user_message or "",
            response_text=response_text or "",
            tools_called_this_turn=tuple(tool_names),
            tools_called_recent=_flatten_recent_tools(self._recent_tools_window, tool_names),
            runtime_context_alerts=tuple(self._active_alert_ids),
        )

        try:
            batch = run_gates(context)
        except Exception:
            logger.warning("run_gates raised, failing open", exc_info=True)
            return response_text

        if batch.passed:
            return response_text

        self._gates_regenerated_this_turn = True

        # -- Sprint F: route tool-required failures to the tool loop --------
        if needs_tool_forcing(batch.failures):
            tool_required_ids = tuple(
                f.gate_id for f in batch.failures
                if _is_tool_required_safe(f.gate_id)
            )
            try:
                rewritten, post_tool_names = await self._handle_tool_required_gate_failure(
                    failing_batch=batch,
                    pre_tool_names=list(tool_names),
                )
            except Exception:
                logger.warning(
                    "Tool-forcing regenerate raised, keeping original",
                    exc_info=True,
                )
                record_regenerate_outcome(batch.failure_ids(), success=False)
                record_tool_forced_outcome(tool_required_ids, success=False)
                return response_text

            if not rewritten:
                # The forced retry produced no text. Ship the original.
                record_regenerate_outcome(batch.failure_ids(), success=False)
                record_tool_forced_outcome(tool_required_ids, success=False)
                return response_text

            second_ctx = replace(
                context,
                response_text=rewritten,
                tools_called_this_turn=tuple(sorted(set(tool_names) | set(post_tool_names))),
                tools_called_recent=_flatten_recent_tools(
                    self._recent_tools_window,
                    list(set(tool_names) | set(post_tool_names)),
                ),
            )

            try:
                second = run_gates(second_ctx)
            except Exception:
                logger.warning(
                    "Second-pass run_gates raised, accepting tool-forced rewrite",
                    exc_info=True,
                )
                record_regenerate_outcome(batch.failure_ids(), success=True)
                record_tool_forced_outcome(tool_required_ids, success=True)
                return rewritten

            if second.passed:
                record_regenerate_outcome(batch.failure_ids(), success=True)
                record_tool_forced_outcome(tool_required_ids, success=True)
                return rewritten

            record_regenerate_outcome(batch.failure_ids(), success=False)
            record_tool_forced_outcome(tool_required_ids, success=False)
            try:
                payload = second.to_event_payload()
                payload["degraded"] = True
                payload["source"] = "response_gates_tool_forced"
                await emit_fn("critic_review", payload)
            except Exception:
                logger.warning(
                    "emit_fn raised on critic_review (tool-forced gates)",
                    exc_info=True,
                )
            return rewritten

        # -- Sprint D: text-only regenerate path (unchanged) ---------------
        action_text = batch.combined_action or "Fix the policy violations."

        try:
            rewritten = await asyncio.to_thread(
                self.regenerate_after_gates,
                response_text,
                action_text,
            )
        except Exception:
            logger.warning(
                "Gate-driven regenerate raised, keeping original",
                exc_info=True,
            )
            record_regenerate_outcome(batch.failure_ids(), success=False)
            return response_text

        # Re-check gates over the rewritten text. If still failing,
        # annotate via SSE but ship the rewrite (one-regenerate cap).
        try:
            second = run_gates(replace(context, response_text=rewritten))
        except Exception:
            logger.warning("Second-pass run_gates raised, accepting rewrite", exc_info=True)
            record_regenerate_outcome(batch.failure_ids(), success=True)
            return rewritten

        if second.passed:
            record_regenerate_outcome(batch.failure_ids(), success=True)
            return rewritten

        record_regenerate_outcome(batch.failure_ids(), success=False)
        try:
            await emit_fn("critic_review", second.to_event_payload())
        except Exception:
            logger.warning(
                "emit_fn raised on critic_review (gates)", exc_info=True
            )
        return rewritten

    # -- Sprint F: Tool-Forcing Regenerate ---------------------------------

    async def _handle_tool_required_gate_failure(
        self,
        failing_batch,
        pre_tool_names: list[str],
    ) -> tuple[str, list[str]]:
        """Re-enter the tool loop to gather data a tool-required gate needs.

        Composes the per-gate "REQUIRED: call X then Y" system reminder
        from ``failing_batch.failures``, appends it as a synthetic user
        turn, then runs a bounded tool loop (at most
        ``_FORCED_REGEN_MAX_ROUNDS`` rounds). Returns the final assistant
        text plus the union of tool names invoked during the forced
        loop. Either return value may be empty on a failure path; the
        caller treats both as "could not satisfy" and ships the
        original response with a degraded annotation.

        Cost: at most THREE chat_completion calls + each tool execution.
        Bounded by ``_FORCED_REGEN_MAX_ROUNDS`` and by
        ``_gates_regenerated_this_turn`` which prevents a re-entry.

        Args:
            failing_batch: The ``GateBatchResult`` returned by the
                first-pass ``run_gates``. We use its ``failures`` to
                compose the instruction.
            pre_tool_names: Tools that ran in this turn BEFORE the
                forced retry. Not used by this method directly; the
                caller unions it with our return value.

        Returns:
            ``(final_text, forced_tool_names)``. ``final_text`` is the
            new candidate response (may be empty). ``forced_tool_names``
            is the set of tool names invoked during the forced rounds.
        """
        from src.agent.response_gates import _compose_tool_force_instruction

        instruction = _compose_tool_force_instruction(failing_batch.failures)
        forced_tool_names: list[str] = []

        # Append the synthetic instruction as a user turn so the tool
        # loop sees it as the next message. The model will respond with
        # either tool_calls (desired) or text (fallback path).
        self._messages.append({"role": "user", "content": instruction})

        runtime_ctx = build_runtime_context(
            user_model=self.user_model,
            date=None,
            startup_context=self.startup_context,
            context=self.context,
        )

        # Run the bounded forced tool loop in a worker thread so we keep
        # the async caller responsive. The loop is small and synchronous
        # to keep the implementation close to the existing one.
        def _run_forced_loop() -> tuple[str, list[str]]:
            return self._forced_tool_loop_sync(
                runtime_ctx=runtime_ctx,
            )

        try:
            text, tools_used = await asyncio.to_thread(_run_forced_loop)
        except Exception:
            logger.warning(
                "Forced tool loop raised, returning empty",
                exc_info=True,
            )
            return "", forced_tool_names

        forced_tool_names = list(tools_used)
        return text, forced_tool_names

    def _forced_tool_loop_sync(
        self,
        runtime_ctx: str,
    ) -> tuple[str, list[str]]:
        """Run up to ``_FORCED_REGEN_MAX_ROUNDS`` rounds of the tool loop.

        Mirrors the body of :meth:`process_message` but limited to a
        small number of rounds and with no telemetry / hook plumbing
        beyond what is required to actually execute the tools. Splices
        the resulting assistant + tool messages into ``self._messages``
        exactly like the main loop so downstream replay / summarisation
        sees a consistent history.

        Returns ``(last_assistant_text, tool_names_invoked)``.
        """
        openai_tools = self.tools.get_openai_tools(defer_non_core=True)
        last_text: str = ""
        tools_used: list[str] = []

        for _round in range(_FORCED_REGEN_MAX_ROUNDS):
            try:
                response = chat_completion(
                    messages=self._messages,
                    system_prompt=STATIC_SYSTEM_PROMPT,
                    tools=openai_tools if openai_tools else None,
                    temperature=AGENT_TEMPERATURE,
                    runtime_context=runtime_ctx,
                )
            except Exception:
                logger.warning(
                    "Forced tool loop: chat_completion raised",
                    exc_info=True,
                )
                break

            if not response.choices:
                break

            message = response.choices[0].message
            content = message.content
            tool_calls = message.tool_calls

            if not tool_calls:
                # Model produced final text. Persist and stop.
                last_text = (content or "").strip()
                if last_text:
                    self._messages.append(
                        {"role": "assistant", "content": last_text}
                    )
                break

            # Filter server-side tools (same logic as main loop).
            client_tool_calls = [
                tc for tc in tool_calls
                if not str(getattr(tc, "id", "") or "").startswith("srvtoolu_")
            ]

            assistant_msg: dict = {"role": "assistant", "content": content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in client_tool_calls
            ]
            if not assistant_msg["tool_calls"]:
                assistant_msg.pop("tool_calls")
            self._messages.append(assistant_msg)

            if not client_tool_calls:
                # Only server tools fired; loop back so the model can
                # finalize on the next round.
                continue

            # Execute the client tools.
            for tc in client_tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}

                hook_ctx = HookContext(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    user_model=self.user_model,
                    tool_name=tool_name,
                )

                pre_decision = run_pre_hooks(tool_name, tool_args, hook_ctx)
                if pre_decision.action == "block":
                    tool_result: Any = {
                        "blocked": True,
                        "reason": pre_decision.reason or "Pre-hook blocked this call.",
                    }
                else:
                    effective_args = (
                        {**tool_args, **pre_decision.replace_args}
                        if pre_decision.replace_args
                        else tool_args
                    )
                    try:
                        tool_result = execute_with_budget(
                            self.tools, tool_name, effective_args,
                        )
                        run_post_hooks(tool_name, effective_args, tool_result, hook_ctx)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                tools_used.append(tool_name)

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                })

            # Continue the loop so the model can produce the final
            # answer or call follow-up tools.

        return last_text, tools_used


# -- Helpers ------------------------------------------------------------------


def _gates_layer_enabled_safe() -> bool:
    """Fail-safe wrapper around the gate-layer master switch."""
    try:
        from src.agent.response_gates import is_gates_layer_enabled
        return is_gates_layer_enabled()
    except Exception:
        logger.warning("is_gates_layer_enabled raised, skipping gates", exc_info=True)
        return False


def _is_tool_required_safe(gate_id: str) -> bool:
    """Return True iff the gate with *gate_id* has needs_tools=True.

    Defensive: a missing or unknown gate id returns False so the
    tool-forced metrics never record a phantom outcome.
    """
    try:
        from src.agent.response_gates import gate_by_id

        gate = gate_by_id(gate_id)
        return bool(gate and gate.needs_tools)
    except Exception:
        return False


def _detect_active_alert_ids_safe(user_model) -> tuple[str, ...]:
    """Best-effort lookup of active critical/warn alert ids for this user.

    Used by ``AgentLoop`` to populate ``GateContext.runtime_context_alerts``
    for Gate 4 (holistic_alert_acknowledged). Returns an empty tuple on
    any error so a missing recovery_alerts table never breaks the loop.

    Logs the detected alert ids at INFO so production traces show the
    gate is receiving fresh data each turn. Helps diagnose 0% fail-rate
    regressions (Sprint J): if this list is consistently empty, either
    the user has no health data or the recovery_alerts thresholds need
    tuning for the persona seed shape.
    """
    try:
        from src.config import get_settings as _gs
        settings = _gs()
        if not settings.use_supabase:
            return ()
        user_id = getattr(user_model, "user_id", None) or settings.agenticsports_user_id
        if not user_id:
            return ()
        from src.services.recovery_alerts import detect_alerts

        alerts = detect_alerts(user_id)
        # Encode as "<severity>:<pattern>" so Gate 4 can filter info-only.
        ids = tuple(f"{a.severity}:{a.pattern}" for a in alerts)
        # Observability: every turn logs the detected alert set so the
        # holistic_alert gate fail-rate is auditable from logs alone.
        logger.info(
            "recovery_alerts.detected user_id=%s count=%d ids=%s",
            user_id,
            len(ids),
            list(ids),
        )
        return ids
    except Exception:
        logger.warning(
            "_detect_active_alert_ids_safe failed; gate sees empty alerts",
            exc_info=True,
        )
        return ()


def _flatten_recent_tools(
    window: "deque[frozenset[str]]",
    current_turn_tools: list[str],
) -> tuple[str, ...]:
    """Union the last-3 user-turns plus the current turn's tools.

    The deque ``window`` is populated AT END of each ``process_message``
    call, so during ``_run_gates_pass`` (which runs after the LLM but
    before the deque is updated) the current turn's tools live in
    ``current_turn_tools``. We union both so Gate 3 sees the freshly
    called tools as well.
    """
    seen: set[str] = set(current_turn_tools)
    for turn_set in window:
        seen.update(turn_set)
    return tuple(sorted(seen))


def _should_run_critic_safe(user_model) -> bool:
    """Fail-safe wrapper around ``critic.should_run_critic``.

    The critic must never break the agent loop, even at the gating
    check. Any exception is logged and treated as "do not run".
    """
    try:
        from src.agent.critic import should_run_critic
        return should_run_critic(user_model)
    except Exception:
        logger.warning("should_run_critic raised, skipping critique", exc_info=True)
        return False


def _finalize_response(text: str, user_message: str) -> str:
    """Unconditional pre-emit sanitization for SSE ``message`` payloads.

    Runs ``sanitize_hard`` exactly once, then ``hard_inspect`` to verify
    no residual violations. Logs an error if anything survives; ships
    the sanitized text either way (it cannot be worse than the input).

    Sprint G guarantees: ANY hard violation in the final response MUST
    be sanitized BEFORE the SSE ``message`` emit. This function is the
    single source of truth for that guarantee. Always-on regardless of
    ``CRITIC_ENABLED`` or gates state. Sub-millisecond, no I/O.
    """
    if not text:
        return text
    try:
        from src.agent.critic import hard_inspect, sanitize_hard
    except Exception:
        logger.warning("_finalize_response import failed, returning original", exc_info=True)
        return text

    try:
        cleaned = sanitize_hard(text, user_message or "")
    except Exception:
        logger.warning("sanitize_hard raised, returning original", exc_info=True)
        return text

    try:
        residual = hard_inspect(cleaned, user_message or "")
    except Exception:
        logger.warning("hard_inspect raised on finalize residual check", exc_info=True)
        return cleaned

    if residual:
        logger.error(
            "Finalize step left residual hard violations: %s",
            [v.rule for v in residual],
        )
    return cleaned


def _unique_tool_names(result: "AgentResult") -> list[str]:
    """Extract the ordered, de-duplicated list of tools called this turn.

    Used by the constitutional critic to validate rules that depend on
    tool-call evidence (no_fabricated_stats, details_before_metrics,
    sync_then_status).
    """
    seen: list[str] = []
    for turn in result.turns:
        name = getattr(turn, "tool_name", None)
        if not name or name in seen:
            continue
        seen.append(name)
    return seen


def _progress_to_sse_data(
    event_type: str,
    detail: str,
    group_id: str = "",
    tool_registry: ToolRegistry | None = None,
) -> dict:
    """Convert an on_progress (event_type, detail) pair into an SSE data dict.

    The ``on_progress`` callback in ``AgentLoop`` passes a plain string as
    ``detail``.  This function translates those strings into structured dicts
    that the SSE router can forward directly to the client.

    For ``tool_call`` / ``tool_hint`` events the dict is enriched with
    ``display_label`` (from the tool registry) and ``group_id`` (stable id
    grouping all tool calls of one agent turn).
    """
    if event_type == "tool_call":
        # detail: "tool_name({"arg": "value"})"
        paren = detail.find("(")
        if paren != -1:
            name = detail[:paren]
            args_str = detail[paren + 1:].rstrip(")")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {"raw": args_str}
        else:
            name = detail
            args = {}
        data: dict = {"name": name, "args": args}
        if tool_registry is not None:
            tool = tool_registry._tools.get(name)
            if tool and tool.display_label:
                data["display_label"] = tool.display_label
        if group_id:
            data["group_id"] = group_id
        return data

    if event_type == "tool_result":
        # detail: "tool_name -> {result preview}"
        arrow = detail.find(" -> ")
        name = detail[:arrow] if arrow != -1 else detail
        preview = detail[arrow + 4:] if arrow != -1 else ""
        return {"name": name, "preview": preview}

    if event_type == "tool_error":
        # detail: "tool_name -> Error: <message>"
        return {"detail": detail}

    if event_type == "action_request":
        # detail: JSON-encoded {"action_type": ..., "label": ..., "payload": ...}
        try:
            return json.loads(detail) if detail else {}
        except json.JSONDecodeError:
            logger.warning("action_request payload was not valid JSON: %r", detail)
            return {"action_type": "", "label": "", "payload": {}}

    if event_type == "ui_component":
        # detail: JSON-encoded {"type": ..., "id": ..., "props": ...}
        try:
            return json.loads(detail) if detail else {}
        except json.JSONDecodeError:
            logger.warning("ui_component payload was not valid JSON: %r", detail)
            return {"type": "", "id": "", "props": {}}

    if event_type == "responding":
        # The loop emits this just before setting response_text; we map it
        # to a thinking event so the UI can clear the spinner.
        return {"text": detail}

    # Default: wrap as text.
    return {"text": detail}


def create_restricted_loop(
    user_model,
    max_tool_rounds: int = 15,
) -> AgentLoop:
    """Create an AgentLoop with restricted tools for background tasks.

    The restricted loop only has access to data, analysis, calc, and health
    tools — no notifications, spawning, or config mutations.

    Args:
        user_model: The user model instance.
        max_tool_rounds: Maximum tool-call rounds (default 15).

    Returns:
        A configured AgentLoop with restricted tools.
    """
    restricted_registry = get_restricted_tools(user_model)
    loop = AgentLoop(
        user_model=user_model,
        tool_registry=restricted_registry,
        context="coach",
        max_rounds=max_tool_rounds,
    )

    return loop


def _drain_sync_queue(sync_queue: queue.Queue) -> None:
    """Empty all remaining items from a sync queue (non-blocking)."""
    while True:
        try:
            sync_queue.get_nowait()
        except queue.Empty:
            break
