"""Config tools -- agent tools for defining and retrieving runtime configuration.

The agent calls these tools to persist metrics, eval criteria, session schemas,
periodization models, and proactive trigger rules. All definitions are stored
per-user in Supabase and survive across sessions.

Every tool validates formulas via CalcEngine before persisting.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.calc.engine import CalcEngine
from src.config import get_settings

logger = logging.getLogger(__name__)

_VALID_CONFIG_TYPES = frozenset({
    "metric_definitions",
    "eval_criteria",
    "session_schemas",
    "periodization_models",
    "proactive_trigger_rules",
})


def register_config_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register all agent config tools into the registry."""
    _settings = get_settings()

    def _get_user_id() -> str:
        """Resolve user_id from user_model (multi-tenant) or settings (CLI)."""
        if user_model is not None and hasattr(user_model, "user_id"):
            return user_model.user_id
        return _settings.agenticsports_user_id

    # ------------------------------------------------------------------
    # define_metric
    # ------------------------------------------------------------------

    def define_metric(
        name: str,
        formula: str,
        description: str = "",
        unit: str = "",
        variables: dict | None = None,
    ) -> dict:
        valid, error = CalcEngine.validate_formula(formula)
        if not valid:
            return {"status": "error", "error": f"Invalid formula: {error}"}

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        from src.db.agent_config_db import upsert_metric_definition
        row = upsert_metric_definition(
            user_id=_get_user_id(),
            name=name,
            formula=formula,
            description=description,
            unit=unit,
            variables=variables or {},
        )
        return {"status": "success", "metric": row}

    registry.register(Tool(
        name="define_metric",
        description=(
            "Define or update a named metric formula that can be calculated against "
            "activity data. The formula is a safe math expression using variable names "
            "(e.g., 'heart_rate * 0.6 + speed * 0.4'). Variables must be numeric. "
            "Formulas support: +, -, *, /, **, abs, min, max, round, sqrt, log, exp, "
            "sum, len, int, float, pow. No imports or attribute access allowed."
        ),
        handler=define_metric,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique metric name (snake_case)"},
                "formula": {"type": "string", "description": "Safe math expression string"},
                "description": {"type": "string", "description": "Human-readable description"},
                "unit": {"type": "string", "description": "Unit of the result (e.g., 'bpm', 'W/kg')"},
                "variables": {
                    "type": "object",
                    "description": "Variable name → description/type hints (for documentation)",
                    "nullable": True,
                },
            },
            "required": ["name", "formula"],
        },
        category="config",
    ))

    # ------------------------------------------------------------------
    # define_eval_criteria
    # ------------------------------------------------------------------

    def define_eval_criteria(
        name: str,
        description: str = "",
        weight: float = 1.0,
        formula: str = "",
    ) -> dict:
        if formula:
            valid, error = CalcEngine.validate_formula(formula)
            if not valid:
                return {"status": "error", "error": f"Invalid formula: {error}"}

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        from src.db.agent_config_db import upsert_eval_criteria
        row = upsert_eval_criteria(
            user_id=_get_user_id(),
            name=name,
            description=description,
            weight=weight,
            formula=formula,
        )
        return {"status": "success", "criteria": row}

    registry.register(Tool(
        name="define_eval_criteria",
        description=(
            "Define or update an evaluation criterion used to score training plans or "
            "sessions. Each criterion has a weight (relative importance) and an optional "
            "formula for numeric scoring. Weights are relative — 2.0 means twice as "
            "important as 1.0."
        ),
        handler=define_eval_criteria,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique criterion name"},
                "description": {"type": "string", "description": "What this criterion measures"},
                "weight": {"type": "number", "description": "Relative weight (default 1.0)"},
                "formula": {
                    "type": "string",
                    "description": "Optional scoring formula",
                    "nullable": True,
                },
            },
            "required": ["name"],
        },
        category="config",
    ))

    # ------------------------------------------------------------------
    # define_session_schema
    # ------------------------------------------------------------------

    def define_session_schema(sport: str, schema: dict) -> dict:
        if not sport or not sport.strip():
            return {"status": "error", "error": "sport must not be empty"}
        if not isinstance(schema, dict):
            return {"status": "error", "error": "schema must be an object"}

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        from src.db.agent_config_db import upsert_session_schema
        row = upsert_session_schema(
            user_id=_get_user_id(),
            sport=sport.lower().strip(),
            schema=schema,
        )
        return {"status": "success", "session_schema": row}

    registry.register(Tool(
        name="define_session_schema",
        description=(
            "Define the structure template for training sessions of a given sport. "
            "The schema describes required and optional fields, session types, intensity "
            "zones, and any sport-specific parameters. This is used to validate and "
            "structure sessions when creating plans."
        ),
        handler=define_session_schema,
        parameters={
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport name (e.g., 'running', 'cycling')"},
                "schema": {
                    "type": "object",
                    "description": "Schema object describing session structure for this sport",
                },
            },
            "required": ["sport", "schema"],
        },
        category="config",
    ))

    # ------------------------------------------------------------------
    # get_config
    # ------------------------------------------------------------------

    def get_config(config_type: str) -> dict:
        if config_type not in _VALID_CONFIG_TYPES:
            return {
                "status": "error",
                "error": f"Unknown config_type '{config_type}'. Valid: {sorted(_VALID_CONFIG_TYPES)}",
            }

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        from src.db import agent_config_db as db
        uid = _get_user_id()

        fetch_fn = {
            "metric_definitions": db.get_metric_definitions,
            "eval_criteria": db.get_eval_criteria,
            "session_schemas": db.get_session_schemas,
            "periodization_models": db.get_periodization_models,
            "proactive_trigger_rules": db.get_proactive_trigger_rules,
        }[config_type]

        items = fetch_fn(uid)
        return {"status": "success", "config_type": config_type, "items": items, "count": len(items)}

    registry.register(Tool(
        name="get_config",
        description=(
            "Retrieve all stored configurations of a given type. Use this to inspect "
            "what metrics, criteria, schemas, or rules are already defined before adding "
            "new ones. config_type must be one of: metric_definitions, eval_criteria, "
            "session_schemas, periodization_models, proactive_trigger_rules."
        ),
        handler=get_config,
        parameters={
            "type": "object",
            "properties": {
                "config_type": {
                    "type": "string",
                    "description": "Type of config to retrieve",
                    "enum": sorted(_VALID_CONFIG_TYPES),
                },
            },
            "required": ["config_type"],
        },
        category="config",
    ))

    # ------------------------------------------------------------------
    # update_config
    # ------------------------------------------------------------------

    def update_config(config_type: str, name: str, updates: dict) -> dict:
        if config_type not in _VALID_CONFIG_TYPES:
            return {
                "status": "error",
                "error": f"Unknown config_type '{config_type}'. Valid: {sorted(_VALID_CONFIG_TYPES)}",
            }

        if not name or not name.strip():
            return {"status": "error", "error": "name must not be empty"}

        if "formula" in updates and updates["formula"]:
            valid, error = CalcEngine.validate_formula(updates["formula"])
            if not valid:
                return {"status": "error", "error": f"Invalid formula: {error}"}

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        from src.db import agent_config_db as db
        uid = _get_user_id()

        update_fn = {
            "metric_definitions": db.update_metric_definition,
            "eval_criteria": db.update_eval_criterion,
            "session_schemas": db.update_session_schema,
            "periodization_models": db.update_periodization_model,
            "proactive_trigger_rules": db.update_proactive_trigger_rule,
        }[config_type]

        # For session_schemas, 'name' IS the sport identifier (same key).
        row = update_fn(uid, name, updates)

        if row is None:
            return {"status": "error", "error": f"No {config_type} named '{name}' found"}

        return {"status": "success", "updated": row}

    registry.register(Tool(
        name="update_config",
        description=(
            "Partially update an existing configuration entry by name. Only the fields "
            "provided in updates will be changed. Formulas are re-validated before saving. "
            "config_type must be one of: metric_definitions, eval_criteria, session_schemas, "
            "periodization_models, proactive_trigger_rules."
        ),
        handler=update_config,
        parameters={
            "type": "object",
            "properties": {
                "config_type": {
                    "type": "string",
                    "description": "Type of config to update",
                    "enum": sorted(_VALID_CONFIG_TYPES),
                },
                "name": {"type": "string", "description": "Name of the config entry to update"},
                "updates": {
                    "type": "object",
                    "description": "Fields to update (only provided fields change)",
                },
            },
            "required": ["config_type", "name", "updates"],
        },
        category="config",
    ))
