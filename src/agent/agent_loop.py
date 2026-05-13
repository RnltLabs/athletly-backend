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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
COMPRESSION_THRESHOLD = 40    # Compress history when messages exceed this
COMPRESSION_KEEP_ROUNDS = 4   # Keep last N full tool-call rounds verbatim
TOOL_CALL_SUMMARY_THRESHOLD = 8  # Inject summary reminder after N consecutive tool rounds

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

        # Compress history before adding new message (Gap 2)
        self._compress_history()

        # Inject runtime context together with the user message to avoid
        # consecutive "user" messages (Gemini requires alternating turns).
        runtime_ctx = build_runtime_context(
            user_model=self.user_model,
            date=None,
            startup_context=self.startup_context,
            context=self.context,
        )

        # If the last loaded message is "user", insert a synthetic assistant
        # reply to maintain alternating turn structure for Gemini.
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.append({
                "role": "assistant",
                "content": "[Continuing from our previous conversation.]",
            })

        # Combine context + user message into a single user turn
        self._messages.append({
            "role": "user",
            "content": f"[CONTEXT]\n{runtime_ctx}\n\n[USER MESSAGE]\n{user_message}",
        })

        # Persist user turn (Gap 1)
        self._save_turn("user", user_message)

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

            # Call LLM with conversation history + tools via LiteLLM
            response = chat_completion(
                messages=self._messages,
                system_prompt=system_prompt,
                tools=openai_tools if openai_tools else None,
                temperature=AGENT_TEMPERATURE,
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

        # Post-turn safety nets (Gap 3b, Gap 4b)
        self._post_turn_extraction_check(result.response_text)
        result.onboarding_just_completed = self._check_onboarding_complete()

        return result

    def inject_context(self, role: str, text: str):
        """Inject context into the conversation (e.g., startup greeting)."""
        oai_role = "assistant" if role == "model" else role
        self._messages.append({"role": oai_role, "content": text})
        self._save_turn(role, text, {"injected": True})


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

        # -- Bridge: on_progress callback → sync queue --
        def _bridge_progress(event_type: str, detail: str) -> None:
            sync_queue.put((event_type, detail))

        # Swap out on_progress for the duration of this call.
        original_on_progress = self.on_progress
        self.on_progress = _bridge_progress

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

                event_type, detail = item
                data = _progress_to_sse_data(event_type, detail)
                try:
                    await emit_fn(event_type, data)
                except Exception:
                    logger.warning(
                        "emit_fn raised while emitting %s event", event_type, exc_info=True
                    )

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
            raise outcome

        # Emit the final message event.
        if outcome.response_text:
            await emit_fn("message", {"text": outcome.response_text})

        return outcome


# -- Helpers ------------------------------------------------------------------

def _progress_to_sse_data(event_type: str, detail: str) -> dict:
    """Convert an on_progress (event_type, detail) pair into an SSE data dict.

    The ``on_progress`` callback in ``AgentLoop`` passes a plain string as
    ``detail``.  This function translates those strings into structured dicts
    that the SSE router can forward directly to the client.
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
            return {"name": name, "args": args}
        return {"name": detail, "args": {}}

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
