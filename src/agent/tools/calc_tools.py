"""Calc tools -- agent tools for evaluating metric formulas against data.

The agent calls these tools to compute named metrics (defined via config_tools)
against activity records or ad-hoc variable dicts.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.calc.engine import CalcEngine
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_calc_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register all calc tools into the registry."""
    _settings = get_settings()

    def _get_user_id() -> str:
        if user_model is not None and hasattr(user_model, "user_id"):
            return user_model.user_id
        return _settings.agenticsports_user_id

    # ------------------------------------------------------------------
    # _resolve_formula: shared logic to look up a metric formula from DB
    # ------------------------------------------------------------------

    def _resolve_formula(metric_name: str) -> tuple[str | None, str | None]:
        """Return (formula, error). Looks up metric_name from DB."""
        if not _settings.use_supabase:
            return None, "Supabase not configured"

        from src.db.agent_config_db import get_metric_definition
        defn = get_metric_definition(_get_user_id(), metric_name)
        if defn is None:
            return None, f"Metric '{metric_name}' not found. Use define_metric first."
        return defn["formula"], None

    # ------------------------------------------------------------------
    # calculate_metric
    # ------------------------------------------------------------------

    def calculate_metric(metric_name: str, variables: dict) -> dict:
        formula, error = _resolve_formula(metric_name)
        if error:
            return {"status": "error", "error": error}

        result = CalcEngine.calculate(formula, variables)
        if result is None:
            return {
                "status": "error",
                "error": (
                    f"Calculation failed for metric '{metric_name}'. "
                    "Check that all required variables are present and numeric."
                ),
                "formula": formula,
                "variables": variables,
            }

        return {
            "status": "success",
            "metric": metric_name,
            "result": result,
            "formula": formula,
        }

    # DEPRECATED: calculate_metric - edge case tool. NOT registered.
    # registry.register(Tool(
    #     name="calculate_metric",
    #     description=(
    #         "Evaluate a previously defined metric formula for a single set of variable values."
    #     ),
    #     handler=calculate_metric,
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "metric_name": {"type": "string"},
    #             "variables": {"type": "object"},
    #         },
    #         "required": ["metric_name", "variables"],
    #     },
    #     category="analysis",
    # ))

    # ------------------------------------------------------------------
    # calculate_bulk_metrics
    # ------------------------------------------------------------------

    def calculate_bulk_metrics(metric_name: str, records: list[dict]) -> dict:
        if not isinstance(records, list):
            return {"status": "error", "error": "records must be a list of variable dicts"}

        if not records:
            return {"status": "success", "metric": metric_name, "results": [], "count": 0}

        formula, error = _resolve_formula(metric_name)
        if error:
            return {"status": "error", "error": error}

        results = CalcEngine.calculate_bulk(formula, records)
        success_count = sum(1 for r in results if r is not None)
        failed_count = len(results) - success_count

        return {
            "status": "success",
            "metric": metric_name,
            "formula": formula,
            "results": results,
            "count": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
        }

    # DEPRECATED: calculate_bulk_metrics - edge case tool. NOT registered.
    # registry.register(Tool(
    #     name="calculate_bulk_metrics",
    #     description=(
    #         "Evaluate a previously defined metric formula across multiple records."
    #     ),
    #     handler=calculate_bulk_metrics,
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "metric_name": {"type": "string"},
    #             "records": {"type": "array", "items": {"type": "object"}},
    #         },
    #         "required": ["metric_name", "records"],
    #     },
    #     category="analysis",
    # ))
