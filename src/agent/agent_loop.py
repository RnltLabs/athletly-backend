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

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.agent.llm import MODEL, chat_completion
from src.agent.tools.registry import ToolRegistry, get_default_tools
from src.agent.system_prompt import build_system_prompt
from src.config import get_settings

logger = logging.getLogger(__name__)

# -- Configuration -----------------------------------------------------------

MAX_TOOL_ROUNDS = 25          # Safety limit
MAX_CONSECUTIVE_ERRORS = 3    # Stop if tools keep failing
AGENT_TEMPERATURE = 0.7       # Creative for coaching, lower for analysis
COMPRESSION_THRESHOLD = 40    # Compress history when messages exceed this
COMPRESSION_KEEP_ROUNDS = 4   # Keep last N full tool-call rounds verbatim

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"


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
    ):
        self.user_model = user_model
        self.tools = tool_registry or get_default_tools(user_model)
        self.on_progress = on_progress
        self.startup_context = startup_context

        # Detect persistence mode
        self._settings = get_settings()
        self._use_supabase = self._settings.use_supabase

        # Conversation history (persists across messages within a session)
        # Uses OpenAI-compatible message format: list of dicts
        self._messages: list[dict] = []

        # Session metadata
        self._session_id: str | None = None
        self._session_file: Path | None = None  # file-based mode only
        self._turns_this_session: int = 0

    # -- Session Persistence (Gap 1) ----------------------------------------

    def start_session(self, resume_session_id: str | None = None) -> str:
        """Start a new coaching session or resume an existing one."""
        if resume_session_id:
            return self._load_session(resume_session_id)

        self._messages = []
        self._turns_this_session = 0

        if self._use_supabase:
            from src.db import create_session
            self._session_id = create_session(self._settings.agenticsports_user_id)
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
                    user_id=self._settings.agenticsports_user_id,
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
            with self._session_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_session(self, session_id: str) -> str:
        """Reload a previous session's history into self._messages."""
        self._session_id = session_id
        self._messages = []
        self._turns_this_session = 0

        if self._use_supabase:
            from src.db import load_session_messages
            rows = load_session_messages(session_id)
            for row in rows:
                role = row.get("role", "user")
                content = row.get("content", "")
                if role in ("user", "model"):
                    oai_role = "user" if role == "user" else "assistant"
                    self._messages.append({"role": oai_role, "content": content})
                    self._turns_this_session += 1
        else:
            self._session_file = SESSIONS_DIR / f"{session_id}.jsonl"
            if not self._session_file.exists():
                logger.warning("Session file %s not found, starting fresh", session_id)
                return session_id
            for line in self._session_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = entry.get("role", "user")
                content = entry.get("content", "")
                if role in ("user", "model"):
                    oai_role = "user" if role == "user" else "assistant"
                    self._messages.append({"role": oai_role, "content": content})
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
        """Check if onboarding info is complete (Gap 4b)."""
        profile = self.user_model.project_profile()
        constraints = profile.get("constraints", {})

        required = [
            bool(profile.get("name") and profile.get("name") != "Athlete"),
            bool(profile.get("sports")),
            bool(profile.get("goal", {}).get("event")),
            bool(constraints.get("training_days_per_week")),
            bool(constraints.get("max_session_minutes")),
        ]

        is_complete = all(required)
        was_complete = self.user_model.meta.get("_onboarding_complete", False)

        if is_complete and not was_complete:
            self.user_model.meta["_onboarding_complete"] = True
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

        # Build system prompt with current context
        system_prompt = build_system_prompt(
            self.user_model,
            startup_context=self.startup_context,
        )

        # Compress history before adding new message (Gap 2)
        self._compress_history()

        # Append user message
        self._messages.append({"role": "user", "content": user_message})

        # Persist user turn (Gap 1)
        self._save_turn("user", user_message)

        # Get tool declarations in OpenAI format for LiteLLM
        openai_tools = self.tools.get_openai_tools()

        consecutive_errors = 0
        last_content = None  # Track last assistant content for MAX_TOOL_ROUNDS fallback

        # -- THE LOOP --
        for round_num in range(MAX_TOOL_ROUNDS):

            # Call LLM with conversation history + tools via LiteLLM
            response = chat_completion(
                messages=self._messages,
                system_prompt=system_prompt,
                tools=openai_tools if openai_tools else None,
                temperature=AGENT_TEMPERATURE,
            )

            message = response.choices[0].message
            content = message.content
            tool_calls = message.tool_calls

            # Handle empty response (edge case -- Gap 9c)
            if not content and not tool_calls:
                consecutive_errors += 1
                logger.warning(
                    "Empty response from LLM (round %d, errors: %d), retrying...",
                    round_num, consecutive_errors,
                )

                # After 3 empty retries, try WITHOUT tools as fallback
                if consecutive_errors == 3:
                    logger.info("Falling back to tool-free call...")
                    fallback_prompt = (
                        system_prompt
                        + "\n\n# IMPORTANT OVERRIDE\n"
                        "Tools are temporarily unavailable. Respond ONLY with "
                        "natural conversational text. Do NOT list, reference, or "
                        "simulate any tool calls (no update_profile, add_belief, "
                        "get_activities, etc.). Just answer the athlete directly "
                        "as a coach would in a normal conversation."
                    )
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

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    result.response_text = (
                        "I had trouble generating a response. "
                        "Could you rephrase your message?"
                    )
                    break

                # Brief pause before retry
                time.sleep(0.5)
                continue

            if not tool_calls:
                # -- MODEL DECIDED TO RESPOND --
                result.response_text = (content or "").strip()

                # Append model response to history
                self._messages.append({"role": "assistant", "content": result.response_text})

                if self.on_progress:
                    self.on_progress("responding", result.response_text[:100])

                break
            else:
                # -- MODEL WANTS TO USE TOOLS --
                # Append the assistant message with tool_calls to history
                assistant_msg: dict = {"role": "assistant", "content": content or ""}
                # Serialize tool_calls for the message history
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
                self._messages.append(assistant_msg)

                # Track last content for MAX_TOOL_ROUNDS fallback
                if content:
                    last_content = content

                # Execute each tool call
                for tc in tool_calls:
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

                    result.tool_calls_made += 1
                    tool_start = time.time()

                    # Execute the tool
                    try:
                        tool_result = self.tools.execute(tool_name, tool_args)
                        consecutive_errors = 0

                        if self.on_progress:
                            preview = json.dumps(tool_result, ensure_ascii=False)[:200]
                            self.on_progress("tool_result", f"{tool_name} -> {preview}")

                    except Exception as e:
                        tool_result = {"error": str(e)}
                        consecutive_errors += 1

                        if self.on_progress:
                            self.on_progress("tool_error", f"{tool_name} -> Error: {e}")

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

        # Post-turn safety nets (Gap 3b, Gap 4b)
        self._post_turn_extraction_check(result.response_text)
        result.onboarding_just_completed = self._check_onboarding_complete()

        return result

    def inject_context(self, role: str, text: str):
        """Inject context into the conversation (e.g., startup greeting)."""
        oai_role = "assistant" if role == "model" else role
        self._messages.append({"role": oai_role, "content": text})
        self._save_turn(role, text, {"injected": True})
