"""Self-improvement tools -- agent evaluates its own formula accuracy.

The agent compares its CalcEngine-computed metrics against provider-reported
values (e.g., Garmin Training Effect, VO2max) to identify drift and improve.
"""

from __future__ import annotations

from src.agent.tools.registry import Tool, ToolRegistry


def register_self_improvement_tools(registry: ToolRegistry, user_model) -> None:
    """Register self-improvement tools."""

    def evaluate_formula_accuracy(metric_name: str, days: int = 28) -> dict:
        """Compare agent-computed metric vs provider values over a period.

        Fetches the agent's metric_definition formula, evaluates it against
        recent activities, and compares with provider-reported values if available.
        """
        from datetime import datetime, timedelta, timezone

        from src.calc.engine import CalcEngine
        from src.config import get_settings
        from src.db.agent_config_db import get_metric_definition
        from src.db.health_data_db import list_garmin_activities, list_health_activities

        settings = get_settings()
        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        # Get the agent's formula definition
        metric_def = get_metric_definition(user_id, metric_name)
        if not metric_def:
            return {
                "status": "no_definition",
                "message": f"No metric definition found for '{metric_name}'.",
            }

        formula = metric_def.get("formula", "")
        if not formula:
            return {
                "status": "no_formula",
                "message": f"Metric '{metric_name}' has no formula defined.",
            }

        # Validate the formula
        valid, error = CalcEngine.validate_formula(formula)
        if not valid:
            return {
                "status": "invalid_formula",
                "message": f"Formula validation failed: {error}",
                "formula": formula,
            }

        # Fetch activities for the period
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()

        health_acts = list_health_activities(user_id, limit=200, after=cutoff)
        garmin_acts = list_garmin_activities(user_id, limit=200, after=cutoff)

        if not health_acts and not garmin_acts:
            return {
                "status": "no_data",
                "message": "No activity data available for comparison.",
            }

        # Build variable dicts and evaluate formula for each activity
        comparisons = _evaluate_health_activities(formula, health_acts)
        comparisons = [*comparisons, *_evaluate_garmin_activities(formula, garmin_acts)]

        if not comparisons:
            return {
                "status": "no_computable",
                "message": "Formula could not be evaluated against any activity.",
                "formula": formula,
            }

        # Calculate accuracy stats
        return _build_accuracy_result(metric_name, formula, comparisons)

    def _evaluate_health_activities(
        formula: str, activities: list[dict],
    ) -> list[dict]:
        """Evaluate formula against health activities."""
        from src.calc.engine import CalcEngine

        results: list[dict] = []
        for act in activities:
            variables = {
                "duration_seconds": float(act.get("duration_seconds") or 0),
                "duration_minutes": float(act.get("duration_seconds") or 0) / 60,
                "distance_meters": float(act.get("distance_meters") or 0),
                "avg_heart_rate": float(act.get("avg_heart_rate") or 0),
                "max_heart_rate": float(act.get("max_heart_rate") or 0),
                "calories": float(act.get("calories") or 0),
                "trimp": float(act.get("training_load_trimp") or 0),
            }

            computed = CalcEngine.calculate(formula, variables)
            provider_value = act.get("training_load_trimp")

            if computed is not None:
                results.append({
                    "date": act.get("start_time", ""),
                    "computed": round(computed, 2),
                    "provider": provider_value,
                    "source": "health",
                })
        return results

    def _evaluate_garmin_activities(
        formula: str, activities: list[dict],
    ) -> list[dict]:
        """Evaluate formula against Garmin activities."""
        from src.calc.engine import CalcEngine

        results: list[dict] = []
        for act in activities:
            variables = {
                "duration_seconds": float(act.get("duration") or 0),
                "duration_minutes": float(act.get("duration") or 0) / 60,
                "distance_meters": float(act.get("distance") or 0),
                "avg_heart_rate": float(act.get("avg_hr") or 0),
                "max_heart_rate": float(act.get("max_hr") or 0),
                "calories": float(act.get("calories") or 0),
                "training_effect": float(act.get("training_effect") or 0),
                "vo2max": float(act.get("vo2max_running") or 0),
            }

            computed = CalcEngine.calculate(formula, variables)
            provider_value = act.get("training_effect")

            if computed is not None:
                results.append({
                    "date": act.get("start_time", ""),
                    "computed": round(computed, 2),
                    "provider": provider_value,
                    "source": "garmin",
                })
        return results

    def _build_accuracy_result(
        metric_name: str, formula: str, comparisons: list[dict],
    ) -> dict:
        """Build the accuracy result dict from comparisons."""
        with_provider = [
            c for c in comparisons
            if c["provider"] is not None
        ]

        if with_provider:
            errors = [
                abs(c["computed"] - float(c["provider"]))
                for c in with_provider
            ]
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)

            return {
                "status": "ok",
                "metric_name": metric_name,
                "formula": formula,
                "total_evaluated": len(comparisons),
                "with_provider_comparison": len(with_provider),
                "avg_absolute_error": round(avg_error, 2),
                "max_absolute_error": round(max_error, 2),
                "sample_comparisons": with_provider[:5],
                "recommendation": (
                    "Formula looks accurate" if avg_error < 10
                    else "Consider revising the formula — significant deviation detected"
                ),
            }

        return {
            "status": "no_provider_data",
            "metric_name": metric_name,
            "formula": formula,
            "total_evaluated": len(comparisons),
            "message": (
                "No provider values available for comparison. "
                "Formula computed successfully but accuracy cannot be verified."
            ),
            "sample_computations": comparisons[:5],
        }

    registry.register(Tool(
        name="evaluate_formula_accuracy",
        description=(
            "Evaluate the accuracy of an agent-defined metric formula by comparing "
            "computed values against provider-reported values (Garmin Training Effect, "
            "VO2max, etc.). Use this to self-check whether your formulas produce "
            "reasonable results. Returns accuracy stats and recommendations."
        ),
        handler=evaluate_formula_accuracy,
        parameters={
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "Name of the metric definition to evaluate.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (default 28).",
                },
            },
            "required": ["metric_name"],
        },
        category="analysis",
    ))

    def review_all_formulas() -> dict:
        """Summary review of all agent-defined metric formulas."""
        from src.calc.engine import CalcEngine
        from src.config import get_settings
        from src.db.agent_config_db import get_metric_definitions

        settings = get_settings()
        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"status": "error", "message": "No user_id configured."}

        definitions = get_metric_definitions(user_id)
        if not definitions:
            return {
                "status": "no_definitions",
                "message": "No metric definitions found. Define metrics first.",
            }

        results = []
        for defn in definitions:
            name = defn.get("name", "unknown")
            formula = defn.get("formula", "")
            valid, error = CalcEngine.validate_formula(formula)

            results.append({
                "name": name,
                "formula": formula,
                "valid": valid,
                "error": error,
                "description": defn.get("description", ""),
                "unit": defn.get("unit", ""),
            })

        valid_count = sum(1 for r in results if r["valid"])
        invalid_count = len(results) - valid_count

        return {
            "status": "ok",
            "total_formulas": len(results),
            "valid": valid_count,
            "invalid": invalid_count,
            "formulas": results,
            "recommendation": (
                "All formulas are valid" if invalid_count == 0
                else f"{invalid_count} formula(s) need attention"
            ),
        }

    registry.register(Tool(
        name="review_all_formulas",
        description=(
            "Review all agent-defined metric formulas for validity. "
            "Checks each formula against CalcEngine's whitelist and reports "
            "which are valid and which need fixing. Use this periodically "
            "to ensure all your metric definitions are working correctly."
        ),
        handler=review_all_formulas,
        parameters={},
        category="analysis",
    ))
