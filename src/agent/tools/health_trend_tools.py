"""Health trend analysis tools -- detect multi-day recovery and fitness trends.

Compares recent 7-day averages against prior 7-day averages to identify
improving, stable, or declining patterns in sleep, HRV, stress, body battery,
recovery score, and steps.
"""

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings


def register_health_trend_tools(registry: ToolRegistry, user_id: str = None) -> None:
    """Register health trend analysis tools."""
    _settings = get_settings()
    _resolved_uid = user_id or _settings.agenticsports_user_id

    def analyze_health_trends(metric: str = "all", days: int = 14) -> dict:
        """Analyze health metric trends over time."""
        from src.db.health_data_db import get_merged_daily_metrics

        user_id = _resolved_uid
        metrics = get_merged_daily_metrics(user_id, days=days)

        if not metrics:
            return {"trends": {}, "data_available": False}

        # Define metrics to analyze
        metric_keys = [
            "sleep_score", "hrv", "resting_hr", "stress",
            "body_battery_high", "recovery_score", "steps",
        ]
        if metric != "all":
            metric_keys = [metric] if metric in metric_keys else []
            if not metric_keys:
                return {
                    "error": f"Unknown metric: {metric}",
                    "available_metrics": [
                        "sleep_score", "hrv", "resting_hr", "stress",
                        "body_battery_high", "recovery_score", "steps",
                    ],
                }

        # Inverted metrics: lower is better
        inverted = {"stress", "resting_hr"}

        trends = {}
        for key in metric_keys:
            trend = _analyze_single_trend(metrics, key, inverted=key in inverted)
            if trend:
                trends[key] = trend

        return {
            "trends": trends,
            "data_available": True,
            "days_analyzed": len(metrics),
        }

    registry.register(Tool(
        name="analyze_health_trends",
        description=(
            "Analyze health metric trends over time. Compares recent 7-day averages "
            "against prior 7-day averages to detect improving, stable, or declining trends. "
            "Use this to identify recovery patterns, fatigue accumulation, or fitness improvements. "
            "Available metrics: sleep_score, hrv, resting_hr, stress, body_battery_high, recovery_score, steps."
        ),
        handler=analyze_health_trends,
        parameters={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Specific metric to analyze, or 'all' for all metrics (default 'all').",
                    "enum": [
                        "all", "sleep_score", "hrv", "resting_hr", "stress",
                        "body_battery_high", "recovery_score", "steps",
                    ],
                },
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days (default 14, minimum 7).",
                },
            },
        },
        category="data",
        display_label="Analysiere Gesundheits-Trends",
    ))


def _analyze_single_trend(
    metrics: list[dict], key: str, inverted: bool = False,
) -> dict | None:
    """Analyze trend for a single metric.

    Compares avg of days 1-7 (recent) vs avg of days 8-14 (prior).
    >5% change = improving/declining, else stable.
    Inverted metrics: stress, resting_hr -- lower is improving.
    Minimum 3 data points per window required.
    """
    recent_window = metrics[:7]  # newest first
    prior_window = metrics[7:14]

    recent_values = [m[key] for m in recent_window if m.get(key) is not None]
    prior_values = [m[key] for m in prior_window if m.get(key) is not None]

    # Current value (latest)
    current = next((m[key] for m in metrics if m.get(key) is not None), None)
    if current is None:
        return None

    result = {
        "current_value": current,
        "avg_7d": round(sum(recent_values) / len(recent_values), 1) if recent_values else None,
    }

    # Need minimum 3 data points in both windows for trend
    if len(recent_values) < 3 or len(prior_values) < 3:
        result["trend_direction"] = "insufficient_data"
        result["trend_pct"] = None
        result["avg_prior_7d"] = (
            round(sum(prior_values) / len(prior_values), 1) if prior_values else None
        )
        return result

    avg_recent = sum(recent_values) / len(recent_values)
    avg_prior = sum(prior_values) / len(prior_values)
    result["avg_prior_7d"] = round(avg_prior, 1)

    if avg_prior == 0:
        result["trend_direction"] = "stable"
        result["trend_pct"] = 0.0
        return result

    pct_change = ((avg_recent - avg_prior) / abs(avg_prior)) * 100
    result["trend_pct"] = round(pct_change, 1)

    # Determine direction
    if abs(pct_change) <= 5:
        result["trend_direction"] = "stable"
    elif pct_change > 5:
        result["trend_direction"] = "declining" if inverted else "improving"
    else:  # pct_change < -5
        result["trend_direction"] = "improving" if inverted else "declining"

    return result
