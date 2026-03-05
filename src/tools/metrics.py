"""Training metrics: DB-driven formula evaluation via CalcEngine.

All formulas (TRIMP, zones, etc.) are stored in the metric_definitions table
and evaluated at runtime. No hardcoded formulas exist in this module.
"""

from __future__ import annotations

import logging

from src.calc.engine import CalcEngine
from src.db.agent_config_db import get_metric_definition, get_metric_definitions

logger = logging.getLogger(__name__)


def compute_metric(
    user_id: str,
    metric_name: str,
    variables: dict,
) -> float | None:
    """Load a single metric definition from DB and evaluate its formula.

    Args:
        user_id: The athlete's user ID.
        metric_name: Name of the metric (e.g. "trimp").
        variables: Variable bindings for formula evaluation.

    Returns:
        Computed float value, or None if the metric is not defined or
        evaluation fails.
    """
    definition = get_metric_definition(user_id, metric_name)
    if definition is None:
        logger.debug("Metric '%s' not defined for user %s", metric_name, user_id)
        return None

    formula = definition.get("formula")
    if not formula:
        logger.warning("Metric '%s' has no formula for user %s", metric_name, user_id)
        return None

    result = CalcEngine.calculate(formula, variables)
    if result is None:
        logger.warning(
            "Formula evaluation returned None for metric '%s' (user %s)",
            metric_name,
            user_id,
        )
    return result


def compute_all_metrics(
    user_id: str,
    variables: dict,
) -> dict[str, float | None]:
    """Load ALL metric definitions for a user and evaluate each.

    Args:
        user_id: The athlete's user ID.
        variables: Variable bindings shared across all formulas.

    Returns:
        Dict mapping metric name to computed value (or None on failure).
    """
    definitions = get_metric_definitions(user_id)
    results: dict[str, float | None] = {}

    for defn in definitions:
        name = defn.get("name", "")
        formula = defn.get("formula")
        if not name or not formula:
            continue
        results[name] = CalcEngine.calculate(formula, variables)

    return results


def get_zone_model(
    user_id: str,
    zone_type: str,
) -> list[dict] | None:
    """Load zone boundaries from a metric_definition named '{zone_type}_zones'.

    The agent stores zone boundaries as JSON in the 'variables' field of the
    metric_definition row. For example, 'hr_zones' or 'pace_zones'.

    Args:
        user_id: The athlete's user ID.
        zone_type: Zone category (e.g. "hr", "pace").

    Returns:
        List of zone boundary dicts from the 'variables' field, or None if
        not defined.
    """
    definition = get_metric_definition(user_id, f"{zone_type}_zones")
    if definition is None:
        return None

    zones = definition.get("variables")
    if not isinstance(zones, list):
        logger.debug(
            "Zone model '%s_zones' variables is not a list for user %s",
            zone_type,
            user_id,
        )
        return None

    return zones
