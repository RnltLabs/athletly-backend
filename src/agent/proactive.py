"""Proactive communication: triggers, message queue, and engagement tracking."""

from datetime import datetime, timedelta


def check_proactive_triggers(
    athlete_profile: dict,
    activities: list[dict],
    episodes: list[dict],
    trajectory: dict,
) -> list[dict]:
    """Check if any proactive notifications should be sent.

    Returns a list of trigger dicts with type, priority, and data.
    """
    triggers = []

    # 1. Trajectory status check
    traj = trajectory.get("trajectory", {})
    confidence = trajectory.get("confidence", 0)

    if traj.get("on_track") is False:
        triggers.append({
            "type": "goal_at_risk",
            "priority": "high",
            "data": {
                "predicted_time": traj.get("predicted_race_time", "unknown"),
                "target_time": trajectory.get("goal", {}).get("target_time", "unknown"),
            },
        })
    elif traj.get("on_track") is True and confidence >= 0.5:
        triggers.append({
            "type": "on_track",
            "priority": "low",
            "data": {
                "predicted_time": traj.get("predicted_race_time", "unknown"),
                "confidence": confidence,
            },
        })

    # 2. Missed session patterns
    missed_patterns = _detect_missed_patterns(episodes)
    for pattern in missed_patterns:
        triggers.append({
            "type": "missed_session_pattern",
            "priority": "medium",
            "data": pattern,
        })

    # 3. Fitness improvement milestone
    if _detect_fitness_improvement(episodes):
        triggers.append({
            "type": "fitness_improving",
            "priority": "low",
            "data": {"trend": "improving"},
        })

    # 4. Upcoming milestone
    milestones = traj.get("key_milestones", [])
    upcoming = [m for m in milestones if m.get("status") in ("on_track", "at_risk")]
    if upcoming:
        triggers.append({
            "type": "milestone_approaching",
            "priority": "medium",
            "data": {"milestone": upcoming[0]},
        })

    # 5. High fatigue warning
    if _detect_high_fatigue(activities, episodes):
        triggers.append({
            "type": "fatigue_warning",
            "priority": "high",
            "data": {"message": "Recent sessions show signs of accumulated fatigue"},
        })

    return triggers


def format_proactive_message(trigger: dict, context: dict) -> str:
    """Format a proactive message for the user.

    Messages should be conversational, specific, and reference actual data.
    """
    msg_type = trigger.get("type", "")
    data = trigger.get("data", {})
    goal = context.get("goal", {})

    if msg_type == "goal_at_risk":
        return (
            f"Heads up: based on current training data, your predicted finish time "
            f"is {data.get('predicted_time', 'unclear')}, which may not meet your "
            f"target of {data.get('target_time', 'N/A')}. "
            f"Let's look at adjustments to get back on track."
        )

    if msg_type == "on_track":
        confidence_pct = int(data.get("confidence", 0) * 100)
        return (
            f"Looking good! Your current trajectory points to a finish time of "
            f"{data.get('predicted_time', 'TBD')} for your "
            f"{goal.get('event', 'race')} "
            f"(confidence: {confidence_pct}%). Keep up the consistent training."
        )

    if msg_type == "missed_session_pattern":
        day = data.get("day", "a certain day")
        count = data.get("missed_count", "multiple")
        return (
            f"I've noticed you've missed {count} sessions on {day} recently. "
            f"Want me to move that workout to a different day?"
        )

    if msg_type == "fitness_improving":
        return (
            "Great news: your fitness metrics are trending upward. "
            "Heart rate is improving at similar paces, indicating stronger aerobic capacity."
        )

    if msg_type == "milestone_approaching":
        ms = data.get("milestone", {})
        return (
            f"Milestone coming up: \"{ms.get('milestone', 'next goal')}\" "
            f"by {ms.get('date', 'soon')} — status: {ms.get('status', 'unknown')}."
        )

    if msg_type == "fatigue_warning":
        return (
            "Watch out: your recent training data shows signs of accumulated fatigue. "
            "Consider taking an extra rest day or reducing intensity this week."
        )

    return f"[{msg_type}] {data}"


def _detect_missed_patterns(episodes: list[dict]) -> list[dict]:
    """Detect recurring missed session patterns from episodes."""
    patterns = []

    # Look for "Thursday" or specific day mentions in patterns
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    all_patterns_text = " ".join(
        " ".join(ep.get("patterns_detected", []) + ep.get("lessons", []))
        for ep in episodes
    ).lower()

    for day in day_names:
        if day.lower() in all_patterns_text and ("miss" in all_patterns_text or "skip" in all_patterns_text):
            patterns.append({
                "day": day,
                "missed_count": "multiple",
                "source": "episode_patterns",
            })
            break  # One pattern per day check is enough

    return patterns


def _detect_fitness_improvement(episodes: list[dict]) -> bool:
    """Check if episodes indicate fitness improvement."""
    if not episodes:
        return False

    for ep in episodes:
        delta = ep.get("fitness_delta", {})
        trend = delta.get("weekly_volume_trend", "")
        vo2_change = delta.get("estimated_vo2max_change", "")
        if trend == "increasing" or (isinstance(vo2_change, str) and vo2_change.startswith("+")):
            return True

    return False


def _detect_high_fatigue(activities: list[dict], episodes: list[dict]) -> bool:
    """Detect signs of accumulated fatigue in recent data.

    P7 enhancement: Uses LLM to evaluate fatigue context instead of
    fixed HR zone/threshold rules. Falls back to heuristic on failure.
    """
    if not activities:
        return False

    # Primary: LLM-based fatigue evaluation (P7)
    try:
        result = _detect_fatigue_llm(activities[-5:])
        if result is not None:
            return result
    except Exception:
        pass

    # Fallback: heuristic (elevated HR in easy efforts)
    easy_activities = [
        a for a in activities[-5:] if a.get("hr_zone", 0) >= 3
        and a.get("heart_rate", {}).get("avg", 0) > 140
    ]
    return len(easy_activities) >= 2


def _detect_fatigue_llm(recent_activities: list[dict]) -> bool | None:
    """Evaluate fatigue using LLM. Returns None on failure."""
    try:
        from src.agent.json_utils import extract_json
        from src.agent.llm import chat_completion

        # Build a concise activity summary
        summaries = []
        for a in recent_activities:
            hr = a.get("heart_rate", {})
            summaries.append(
                f"- {a.get('sport', '?')}: {a.get('hr_zone', '?')} zone, "
                f"avg HR {hr.get('avg', '?')}, "
                f"{round(a.get('duration_seconds', 0) / 60)}min"
            )

        prompt = (
            "Evaluate if this athlete shows signs of accumulated fatigue "
            "based on their last 5 activities.\n\n"
            f"Activities:\n{chr(10).join(summaries)}\n\n"
            "Consider: elevated HR in easy efforts, HR drift patterns, "
            "activity frequency. Respond with ONLY a JSON object:\n"
            '{"fatigued": true|false, "reasoning": "1 sentence"}'
        )

        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        result = extract_json(response.choices[0].message.content.strip())
        return result.get("fatigued", False)
    except Exception:
        return None


# ── Proactive Message Queue ──────────────────────────────────────


def queue_proactive_message(
    user_id: str,
    trigger: dict,
    priority: float = 0.5,
    context: dict | None = None,
) -> dict:
    """Add a proactive trigger to the message queue.

    Deduplication is handled at the DB level via upsert: if a pending message
    of the same trigger type already exists for the user it is updated in place.

    Args:
        user_id: UUID of the owning user.
        trigger: Trigger dict with type, priority, data fields.
        priority: Numeric priority 0.0-1.0 (higher = more urgent).
        context: Optional context dict for format_proactive_message().

    Returns:
        The queued message dict.
    """
    from src.db.proactive_queue_db import queue_message

    message_text = format_proactive_message(trigger, context or {})
    return queue_message(
        user_id=user_id,
        trigger_type=trigger.get("type", "unknown"),
        priority=priority,
        data=trigger.get("data", {}),
        message_text=message_text,
    )


def get_pending_messages(user_id: str) -> list[dict]:
    """Return pending messages for a user, sorted by priority (highest first)."""
    from src.db.proactive_queue_db import get_pending_messages as _get_pending

    return _get_pending(user_id)


def deliver_message(user_id: str, message_id: str) -> dict | None:
    """Mark a message as delivered and record delivery timestamp.

    Returns the updated message, or None if not found.
    """
    from src.db.proactive_queue_db import deliver_message as _deliver

    return _deliver(user_id, message_id)


def record_engagement(
    user_id: str,
    message_id: str,
    responded: bool = False,
    continued_session: bool = False,
    turns_after: int = 0,
) -> dict | None:
    """Record user engagement with a delivered proactive message.

    Args:
        user_id: UUID of the owning user.
        message_id: UUID of the proactive_queue row.
        responded: Whether the user responded to the message.
        continued_session: Whether the user continued the session after delivery.
        turns_after: Number of conversation turns after delivery.

    Returns the updated message, or None if not found.
    """
    from src.db.proactive_queue_db import record_engagement as _record

    return _record(user_id, message_id, responded, continued_session, turns_after)


def expire_stale_messages(user_id: str, max_age_days: int = 7) -> list[dict]:
    """Expire pending messages older than max_age_days.

    Returns the list of expired messages.
    """
    from src.db.proactive_queue_db import expire_stale_messages as _expire

    return _expire(user_id, max_age_days)


# ── Mid-Conversation Trigger Refresh (P8) ────────────────────────


# Priority mapping from string to numeric
_PRIORITY_MAP = {"high": 0.9, "medium": 0.5, "low": 0.2}

# How often to refresh triggers during conversation (in turns)
PROACTIVE_REFRESH_INTERVAL = 3


def refresh_proactive_triggers(
    user_id: str,
    activities: list[dict],
    episodes: list[dict],
    trajectory: dict,
    athlete_profile: dict,
    context: dict | None = None,
) -> list[dict]:
    """Check for new proactive triggers and queue any that aren't already pending.

    P8 enhancement: called during conversation (not just startup) to keep
    the proactive queue fresh with relevant insights.

    Deduplication is handled at the DB level via upsert, so duplicate trigger
    types for the same user are updated rather than duplicated.

    Args:
        user_id: UUID of the owning user.
        activities: Recent activity dicts.
        episodes: Episode dicts.
        trajectory: Trajectory dict.
        athlete_profile: Athlete profile dict.
        context: Optional context for message formatting.

    Returns:
        List of newly queued message dicts.
    """
    # Detect triggers from current data
    triggers = check_proactive_triggers(
        athlete_profile, activities, episodes, trajectory,
    )

    if not triggers:
        return []

    # Get existing pending types to avoid redundant upserts within the same call
    pending = get_pending_messages(user_id)
    pending_types = {m.get("trigger_type") for m in pending}

    queued = []
    for trigger in triggers:
        trigger_type = trigger.get("type", "unknown")
        if trigger_type in pending_types:
            continue  # Already queued

        priority_str = trigger.get("priority", "medium")
        priority_num = _PRIORITY_MAP.get(priority_str, 0.5)

        msg = queue_proactive_message(
            user_id, trigger, priority=priority_num, context=context,
        )
        queued.append(msg)
        pending_types.add(trigger_type)

    return queued


# ── Silence Decay ────────────────────────────────────────────────


def calculate_silence_decay(
    last_interaction: str | None,
    base_urgency: float = 0.3,
) -> float:
    """Calculate urgency boost based on time since last interaction.

    Longer silence = higher motivation to initiate contact.
    Returns a value 0.0-1.0 that can be added to base urgency.

    Args:
        last_interaction: ISO timestamp of last user interaction, or None.
        base_urgency: Base urgency level before silence decay.
    """
    if not last_interaction:
        return 0.5  # No interaction history — moderate boost

    last_dt = datetime.fromisoformat(last_interaction)
    days_silent = (datetime.now() - last_dt).total_seconds() / 86400

    if days_silent < 1:
        return 0.0  # Active user — no boost
    elif days_silent < 3:
        return 0.1  # Slightly quiet
    elif days_silent < 5:
        return 0.3  # Getting quiet
    elif days_silent < 10:
        return 0.5  # Noticeably absent
    else:
        return 0.7  # Extended silence — strong motivation


def check_conversation_triggers(
    user_model_data: dict,
    last_interaction: str | None = None,
) -> list[dict]:
    """Check for conversation-based proactive triggers.

    These triggers come from conversation patterns rather than FIT data:
    - User hasn't chatted in 5+ days
    - User expressed frustration last session
    - Engagement is declining

    Args:
        user_model_data: User model structured_core or summary data.
        last_interaction: ISO timestamp of last user interaction.

    Returns list of trigger dicts.
    """
    triggers = []

    # Silence-based check-in
    if last_interaction:
        last_dt = datetime.fromisoformat(last_interaction)
        days_silent = (datetime.now() - last_dt).total_seconds() / 86400

        if days_silent >= 5:
            silence_urgency = calculate_silence_decay(last_interaction)
            triggers.append({
                "type": "silence_checkin",
                "priority": "medium",
                "urgency": silence_urgency,
                "data": {
                    "days_since_last_chat": round(days_silent, 1),
                    "message": "It's been a while since we last chatted.",
                },
            })

    return triggers
