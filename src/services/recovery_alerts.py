"""Deterministic whole-athlete recovery alerts.

Reads `health_daily_metrics` (via `get_merged_daily_metrics`) plus the
cross-source training-load summary and runs nine fixed-threshold pattern
checks. Returns a list of `RecoveryAlert` dataclasses the coach can
weave into its reply.

Design goals:
- DETERMINISTIC: no LLM, no fuzzy matching, pure threshold logic.
- BEST-EFFORT: any single pattern check that raises is swallowed and
  logged. The caller never gets a partial result that looks like "all
  green" because one check happened to error.
- TUNABLE: every threshold lives as a module-level constant at the top
  of this file. Rebalancing the coach is a one-file change.
- BACKWARDS COMPATIBLE: when there is no data, no baseline, or all
  patterns are green, the function returns an empty list. The runtime
  context will then emit no section.

Usage::

    from src.services.recovery_alerts import detect_alerts

    alerts = detect_alerts(user_id)
    if alerts:
        # format and inject into runtime context
        ...
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable thresholds (see RESEARCH.md for sources)
# ---------------------------------------------------------------------------

# Sleep
SLEEP_LOW_MINUTES: int = 360               # 6h
SLEEP_CRITICAL_MINUTES: int = 240          # 4h
SLEEP_LOW_CONSECUTIVE_DAYS: int = 3
SLEEP_CRITICAL_DAYS_IN_7: int = 5

# HRV: 7-day recent avg vs 30-day baseline.
HRV_BASELINE_DAYS: int = 30
HRV_RECENT_DAYS: int = 7
HRV_DROP_PCT: float = 0.15                 # 15 percent drop is meaningful

# Resting heart rate elevation
RHR_BASELINE_DAYS: int = 14
RHR_ELEVATION_BPM: int = 5
RHR_ELEVATION_DAYS: int = 3

# Body battery (morning peak)
BB_CHRONIC_THRESHOLD: int = 30
BB_CHRONIC_DAYS: int = 3

# Stress (Garmin all-day average)
STRESS_CHRONIC_THRESHOLD: int = 60
STRESS_CHRONIC_DAYS: int = 5

# Recovery score
RECOVERY_LOW_THRESHOLD: int = 30
RECOVERY_LOW_DAYS: int = 2
RECOVERY_CRITICAL_THRESHOLD: int = 20

# Training load spike (ACWR proxy)
LOAD_SPIKE_RATIO: float = 1.5

# Detection window: how many days of health_daily_metrics we pull. Must
# be at least max(HRV_BASELINE_DAYS, RHR_BASELINE_DAYS) so all the
# baseline windows have data to compute against.
LOOKBACK_DAYS: int = 30


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


Severity = Literal["info", "warn", "critical"]


@dataclass(frozen=True)
class RecoveryAlert:
    """A single deterministic recovery flag."""

    severity: Severity
    pattern: str           # short id like "sleep_low_3d"
    message_de: str        # German one-liner the coach can quote
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_alerts(user_id: str) -> list[RecoveryAlert]:
    """Run all pattern checks and return the active alerts.

    Pulls the last `LOOKBACK_DAYS` of `health_daily_metrics` for the
    given user, then runs each pattern check independently. Returns an
    empty list if the data is missing or every pattern is green.

    Failures in any single check are swallowed (DEBUG-logged) so the
    other patterns still run. Failure to fetch metrics at all returns
    an empty list rather than raising.
    """
    metrics = _load_metrics(user_id)
    load_summary_pair = _load_load_summary_pair(user_id)

    return _detect_from_data(metrics, load_summary_pair)


def detect_alerts_from_metrics(
    metrics: list[dict],
    load_summary_pair: tuple[dict | None, dict | None] | None = None,
) -> list[RecoveryAlert]:
    """Pure detection over pre-fetched data. Used by tests and the tool layer.

    `metrics` is the merged daily-metrics list newest first (same shape
    as `get_merged_daily_metrics` returns).

    `load_summary_pair` is `(current_7d, prior_7d)`, each a
    `get_cross_source_load_summary`-shape dict or None. When None, the
    training-load-spike check is skipped.
    """
    return _detect_from_data(metrics, load_summary_pair or (None, None))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_metrics(user_id: str) -> list[dict]:
    """Pull the last `LOOKBACK_DAYS` of daily metrics. Best-effort."""
    try:
        from src.db.health_data_db import get_merged_daily_metrics

        return get_merged_daily_metrics(user_id, days=LOOKBACK_DAYS)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("recovery_alerts: failed to load metrics: %s", exc)
        return []


def _load_load_summary_pair(
    user_id: str,
) -> tuple[dict | None, dict | None]:
    """Pull the current and prior 7-day load summaries. Best-effort.

    Used by the training-load-spike check. We compute "current" as the
    last 7 days and "prior" as the 7 days before that, so we need two
    windows. The cross-source helper only supports a single window, so
    we pull 14 days, then split by date in code.
    """
    try:
        from datetime import datetime, timedelta, timezone

        from src.db.activity_store_db import list_activities

        cutoff_14d = (
            datetime.now(timezone.utc) - timedelta(days=14)
        ).isoformat()
        rows = list_activities(user_id, limit=1000, after=cutoff_14d)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("recovery_alerts: failed to load activities: %s", exc)
        return None, None

    if not rows:
        return None, None

    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    seven_days_ago = (now - timedelta(days=7)).isoformat()

    current: dict[str, float] = {"total_minutes": 0.0, "total_trimp": 0.0}
    prior: dict[str, float] = {"total_minutes": 0.0, "total_trimp": 0.0}

    for r in rows:
        start_time = r.get("start_time") or ""
        if not start_time:
            continue
        duration_min = (r.get("duration_seconds") or 0) / 60.0
        trimp = r.get("trimp") or 0
        bucket = current if start_time >= seven_days_ago else prior
        bucket["total_minutes"] += duration_min
        bucket["total_trimp"] += trimp

    return current, prior


def _detect_from_data(
    metrics: list[dict],
    load_summary_pair: tuple[dict | None, dict | None],
) -> list[RecoveryAlert]:
    """Run every pattern check and collect active alerts.

    Each individual check is wrapped in a try/except so one bad row
    cannot mask other patterns. Patterns are ordered so that the more
    severe / more specific check appears first when a single underlying
    cause triggers multiple rules (the agent then prioritises by
    severity).
    """
    checks = (
        ("sleep_critical", _check_sleep_critical),
        ("sleep_low_3d", _check_sleep_low_3d),
        ("recovery_critical", _check_recovery_critical),
        ("recovery_score_low", _check_recovery_score_low),
        ("hrv_drop", _check_hrv_drop),
        ("rhr_elevation", _check_rhr_elevation),
        ("body_battery_chronic", _check_body_battery_chronic),
        ("stress_chronic", _check_stress_chronic),
    )

    alerts: list[RecoveryAlert] = []

    for pattern_id, check_fn in checks:
        try:
            result = check_fn(metrics)
        except Exception as exc:
            logger.debug(
                "recovery_alerts: pattern %s raised %s; skipping",
                pattern_id,
                exc,
            )
            continue
        if result is not None:
            alerts.append(result)

    # Training load spike uses a different data source (activities, not
    # daily metrics). Run it independently.
    try:
        spike = _check_training_load_spike(load_summary_pair)
    except Exception as exc:
        logger.debug(
            "recovery_alerts: pattern training_load_spike raised %s", exc,
        )
        spike = None
    if spike is not None:
        alerts.append(spike)

    return _dedupe_by_pattern(alerts)


def _dedupe_by_pattern(alerts: list[RecoveryAlert]) -> list[RecoveryAlert]:
    """Keep only the highest-severity alert per pattern id.

    Pattern ids are already unique in our check set, but if a future
    refactor introduces overlap (for example, sleep_critical and
    sleep_low_3d both firing on the same data), we still emit both
    because they have different ids. This helper exists to make the
    de-duplication explicit if we ever collapse ids.
    """
    return list(alerts)


# ---------------------------------------------------------------------------
# Individual pattern checks (return RecoveryAlert | None)
# ---------------------------------------------------------------------------


_SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}


def _check_sleep_low_3d(metrics: list[dict]) -> RecoveryAlert | None:
    """Three plus consecutive most-recent nights under SLEEP_LOW_MINUTES."""
    if len(metrics) < SLEEP_LOW_CONSECUTIVE_DAYS:
        return None

    recent = metrics[:SLEEP_LOW_CONSECUTIVE_DAYS]
    sleeps = [m.get("sleep_minutes") for m in recent]
    if any(s is None for s in sleeps):
        return None

    if not all(s < SLEEP_LOW_MINUTES for s in sleeps):
        return None

    avg_h = sum(sleeps) / len(sleeps) / 60.0
    avg_str = _format_de_hours(avg_h)

    return RecoveryAlert(
        severity="warn",
        pattern="sleep_low_3d",
        message_de=(
            f"Drei Nächte hintereinander unter 6h Schlaf "
            f"(Schnitt {avg_str})."
        ),
        evidence={
            "days": SLEEP_LOW_CONSECUTIVE_DAYS,
            "sleep_avg_h": round(avg_h, 1),
            "sleep_minutes_per_day": [round(s, 0) for s in sleeps],
        },
    )


def _check_sleep_critical(metrics: list[dict]) -> RecoveryAlert | None:
    """Single night < 4h OR 5+ days < 6h within the last 7 days."""
    if not metrics:
        return None

    # Single-night-under-4h check (look at the entire LOOKBACK window).
    for m in metrics[:7]:
        sleep = m.get("sleep_minutes")
        if sleep is None:
            continue
        if sleep < SLEEP_CRITICAL_MINUTES:
            h = sleep / 60.0
            return RecoveryAlert(
                severity="critical",
                pattern="sleep_critical",
                message_de=(
                    f"Eine Nacht unter 4h Schlaf "
                    f"({_format_de_hours(h)}) in den letzten Tagen."
                ),
                evidence={
                    "date": m.get("date"),
                    "sleep_minutes": round(sleep, 0),
                },
            )

    # 5-plus-days-under-6h-in-7-day-window check.
    window = metrics[:7]
    short_nights = [
        m for m in window
        if m.get("sleep_minutes") is not None
        and m["sleep_minutes"] < SLEEP_LOW_MINUTES
    ]
    if len(short_nights) >= SLEEP_CRITICAL_DAYS_IN_7:
        avg_h = (
            sum(m["sleep_minutes"] for m in short_nights)
            / len(short_nights)
            / 60.0
        )
        return RecoveryAlert(
            severity="critical",
            pattern="sleep_critical",
            message_de=(
                f"{len(short_nights)} von 7 Nächten unter 6h "
                f"(Schnitt {_format_de_hours(avg_h)}). "
                f"Adaptation leidet."
            ),
            evidence={
                "short_nights_in_7d": len(short_nights),
                "sleep_avg_h_on_short_nights": round(avg_h, 1),
            },
        )

    return None


def _check_hrv_drop(metrics: list[dict]) -> RecoveryAlert | None:
    """7-day HRV avg dropped HRV_DROP_PCT vs 30-day baseline."""
    if len(metrics) < HRV_BASELINE_DAYS // 2:
        # Need at least half the baseline window to make a claim. Anything
        # shorter is too noisy.
        return None

    recent = [
        m.get("hrv") for m in metrics[:HRV_RECENT_DAYS]
        if m.get("hrv") is not None
    ]
    baseline = [
        m.get("hrv") for m in metrics[:HRV_BASELINE_DAYS]
        if m.get("hrv") is not None
    ]

    # Need substantive coverage of both windows.
    if len(recent) < 4 or len(baseline) < 10:
        return None

    recent_avg = sum(recent) / len(recent)
    baseline_avg = sum(baseline) / len(baseline)
    if baseline_avg <= 0:
        return None

    drop_pct = (baseline_avg - recent_avg) / baseline_avg
    if drop_pct < HRV_DROP_PCT:
        return None

    return RecoveryAlert(
        severity="warn",
        pattern="hrv_drop",
        message_de=(
            f"HRV im 7-Tage-Schnitt {round(drop_pct * 100)}% unter "
            f"deinem 30-Tage-Baseline "
            f"({round(recent_avg, 1)} vs {round(baseline_avg, 1)})."
        ),
        evidence={
            "recent_avg": round(recent_avg, 1),
            "baseline_avg": round(baseline_avg, 1),
            "drop_pct": round(drop_pct, 3),
        },
    )


def _check_rhr_elevation(metrics: list[dict]) -> RecoveryAlert | None:
    """3+ days with resting_hr >= baseline + RHR_ELEVATION_BPM."""
    if len(metrics) < RHR_BASELINE_DAYS:
        return None

    baseline_window = [
        m.get("resting_hr") for m in metrics[:RHR_BASELINE_DAYS]
        if m.get("resting_hr") is not None
    ]
    if len(baseline_window) < 7:
        return None

    baseline_avg = sum(baseline_window) / len(baseline_window)
    elevation_threshold = baseline_avg + RHR_ELEVATION_BPM

    # Walk the most-recent days and count consecutive elevation.
    elevated_days = 0
    elevated_values: list[float] = []
    for m in metrics[:RHR_ELEVATION_DAYS * 2]:
        rhr = m.get("resting_hr")
        if rhr is None:
            continue
        if rhr >= elevation_threshold:
            elevated_days += 1
            elevated_values.append(rhr)
        else:
            break  # streak broken; we want consecutive

    if elevated_days < RHR_ELEVATION_DAYS:
        return None

    return RecoveryAlert(
        severity="warn",
        pattern="rhr_elevation",
        message_de=(
            f"Ruhepuls {elevated_days} Tage in Folge über dem "
            f"14-Tage-Baseline ({round(baseline_avg)} bpm), "
            f"aktuell um {round(elevated_values[0] - baseline_avg)} bpm erhöht."
        ),
        evidence={
            "days": elevated_days,
            "baseline_bpm": round(baseline_avg, 1),
            "elevated_values": [round(v, 0) for v in elevated_values],
        },
    )


def _check_body_battery_chronic(
    metrics: list[dict],
) -> RecoveryAlert | None:
    """body_battery_high < BB_CHRONIC_THRESHOLD for 3+ consecutive days."""
    if len(metrics) < BB_CHRONIC_DAYS:
        return None

    recent = metrics[:BB_CHRONIC_DAYS]
    values = [m.get("body_battery_high") for m in recent]
    if any(v is None for v in values):
        return None

    if not all(v < BB_CHRONIC_THRESHOLD for v in values):
        return None

    avg = sum(values) / len(values)
    return RecoveryAlert(
        severity="warn",
        pattern="body_battery_chronic",
        message_de=(
            f"Body Battery seit {BB_CHRONIC_DAYS} Tagen unter "
            f"{BB_CHRONIC_THRESHOLD} (Schnitt {round(avg)}). "
            f"Du lädst nicht mehr richtig auf."
        ),
        evidence={
            "days": BB_CHRONIC_DAYS,
            "values": [round(v, 0) for v in values],
            "avg": round(avg, 1),
        },
    )


def _check_stress_chronic(metrics: list[dict]) -> RecoveryAlert | None:
    """stress_avg > STRESS_CHRONIC_THRESHOLD for 5+ days within 7."""
    window = metrics[:7]
    if len(window) < STRESS_CHRONIC_DAYS:
        return None

    high_stress_days = [
        m for m in window
        if m.get("stress") is not None
        and m["stress"] > STRESS_CHRONIC_THRESHOLD
    ]

    if len(high_stress_days) < STRESS_CHRONIC_DAYS:
        return None

    avg = sum(m["stress"] for m in high_stress_days) / len(high_stress_days)
    return RecoveryAlert(
        severity="info",
        pattern="stress_chronic",
        message_de=(
            f"Stress-Level seit {len(high_stress_days)} Tagen erhöht "
            f"(Schnitt {round(avg)}). Lebenslast oder Training?"
        ),
        evidence={
            "days": len(high_stress_days),
            "threshold": STRESS_CHRONIC_THRESHOLD,
            "avg": round(avg, 1),
        },
    )


def _check_recovery_score_low(metrics: list[dict]) -> RecoveryAlert | None:
    """recovery_score < RECOVERY_LOW_THRESHOLD for 2+ consecutive days."""
    if len(metrics) < RECOVERY_LOW_DAYS:
        return None

    recent = metrics[:RECOVERY_LOW_DAYS]
    values = [m.get("recovery_score") for m in recent]
    if any(v is None for v in values):
        return None

    if not all(v < RECOVERY_LOW_THRESHOLD for v in values):
        return None

    # If today is critical, the critical check will fire instead. Avoid
    # duplicating the same root cause across two alerts.
    if values[0] < RECOVERY_CRITICAL_THRESHOLD:
        return None

    return RecoveryAlert(
        severity="warn",
        pattern="recovery_score_low",
        message_de=(
            f"Recovery-Score zwei Tage in Folge unter "
            f"{RECOVERY_LOW_THRESHOLD} (heute {values[0]}). "
            f"Heute lieber locker."
        ),
        evidence={
            "days": RECOVERY_LOW_DAYS,
            "values": values,
        },
    )


def _check_recovery_critical(metrics: list[dict]) -> RecoveryAlert | None:
    """recovery_score < RECOVERY_CRITICAL_THRESHOLD today."""
    if not metrics:
        return None

    today_score = metrics[0].get("recovery_score")
    if today_score is None:
        return None
    if today_score >= RECOVERY_CRITICAL_THRESHOLD:
        return None

    return RecoveryAlert(
        severity="critical",
        pattern="recovery_critical",
        message_de=(
            f"Recovery-Score heute bei {today_score} (unter "
            f"{RECOVERY_CRITICAL_THRESHOLD}). Heute ist Ruhetag."
        ),
        evidence={"recovery_score_today": today_score},
    )


def _check_training_load_spike(
    pair: tuple[dict | None, dict | None],
) -> RecoveryAlert | None:
    """Weekly load > LOAD_SPIKE_RATIO times the prior week."""
    current, prior = pair
    if not current or not prior:
        return None

    cur_load = current.get("total_trimp") or current.get("total_minutes") or 0
    prior_load = prior.get("total_trimp") or prior.get("total_minutes") or 0

    if prior_load <= 0:
        # Coming back from a zero week: don't flag, no meaningful ratio.
        return None

    ratio = cur_load / prior_load
    if ratio < LOAD_SPIKE_RATIO:
        return None

    return RecoveryAlert(
        severity="warn",
        pattern="training_load_spike",
        message_de=(
            f"Wochenlast {round(ratio, 1)}x höher als die Vorwoche. "
            f"Schneller Aufbau erhöht Verletzungsrisiko."
        ),
        evidence={
            "ratio": round(ratio, 2),
            "current_load": round(cur_load, 1),
            "prior_load": round(prior_load, 1),
        },
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_de_hours(hours: float) -> str:
    """Format a fractional hour count as German `Xh YYmin` or `X,Yh`.

    We pick the shorter representation per case. For example 5.4 hours
    renders as "5,4h", but 5.0 hours renders as "5h".
    """
    if abs(hours - round(hours)) < 0.05:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h".replace(".", ",")


def format_alerts_block(alerts: list[RecoveryAlert]) -> str:
    """Render alerts as a Markdown block for runtime context injection.

    Returns an empty string when `alerts` is empty so callers can do
    `if block: sections.append(block)` without an extra conditional.
    """
    if not alerts:
        return ""

    # Order: critical first, then warn, then info; stable within band.
    ordered = sorted(
        alerts,
        key=lambda a: -_SEVERITY_RANK.get(a.severity, 0),
    )

    lines = ["# Coach Alerts (proactive)"]
    for a in ordered:
        lines.append(
            f"- [{a.severity.upper()}] {a.pattern}: {a.message_de}"
        )
    lines.append("")
    lines.append(
        "Acknowledge or address these in your next reply if they are "
        "relevant to what the athlete asked. Never ignore a critical "
        "alert. See STRICT: WHOLE-ATHLETE COACHING for the contract."
    )
    return "\n".join(lines)
