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

        uid = _get_user_id()

        # Semantic dedup check (Visionplan 8.12 D): cosine-like similarity
        from src.db.agent_config_db import get_metric_definitions, upsert_metric_definition
        from src.services.config_gc import (
            SIMILARITY_THRESHOLD,
            compute_weighted_config_similarity,
        )
        try:
            existing = get_metric_definitions(uid)
            for m in existing:
                if m.get("name") == name:
                    continue  # Same name = update, not duplicate
                similarity = compute_weighted_config_similarity(
                    formula1=formula,
                    formula2=m.get("formula", ""),
                    name1=name,
                    name2=m.get("name", ""),
                    desc1=description,
                    desc2=m.get("description", ""),
                )
                if similarity > SIMILARITY_THRESHOLD:
                    logger.info(
                        "Semantic dedup: metric '%s' similar to '%s' (score=%.3f)",
                        name, m["name"], similarity,
                    )
                    return {
                        "status": "duplicate",
                        "existing_name": m["name"],
                        "similarity": round(similarity, 3),
                        "message": (
                            f"Formula semantically similar to metric '{m['name']}' "
                            f"(similarity={similarity:.3f}). Use update_config to modify it."
                        ),
                    }
        except Exception:
            logger.debug("Dedup check skipped", exc_info=True)

        row = upsert_metric_definition(
            user_id=uid,
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

        uid = _get_user_id()

        # Light-touch similarity warning (does not block creation)
        from src.db.agent_config_db import get_eval_criteria, upsert_eval_criteria
        from src.services.config_gc import (
            SIMILARITY_THRESHOLD,
            compute_weighted_config_similarity,
        )
        try:
            existing = get_eval_criteria(uid)
            for ec in existing:
                if ec.get("name") == name:
                    continue
                similarity = compute_weighted_config_similarity(
                    formula1=formula,
                    formula2=ec.get("formula", ""),
                    name1=name,
                    name2=ec.get("name", ""),
                    desc1=description,
                    desc2=ec.get("description", ""),
                )
                if similarity > SIMILARITY_THRESHOLD:
                    logger.warning(
                        "Eval criteria '%s' similar to existing '%s' (score=%.3f)",
                        name, ec["name"], similarity,
                    )
        except Exception:
            logger.debug("Eval criteria similarity check skipped", exc_info=True)

        row = upsert_eval_criteria(
            user_id=uid,
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

        uid = _get_user_id()
        normalized_sport = sport.lower().strip()

        # Light-touch similarity warning for sport names (does not block creation)
        from src.db.agent_config_db import get_session_schemas, upsert_session_schema
        from src.services.config_gc import (
            SIMILARITY_THRESHOLD,
            compute_config_similarity,
        )
        try:
            existing = get_session_schemas(uid)
            for ss in existing:
                existing_sport = ss.get("sport", "")
                if existing_sport == normalized_sport:
                    continue
                similarity = compute_config_similarity(normalized_sport, existing_sport)
                if similarity > SIMILARITY_THRESHOLD:
                    logger.warning(
                        "Session schema sport '%s' similar to existing '%s' (score=%.3f)",
                        normalized_sport, existing_sport, similarity,
                    )
        except Exception:
            logger.debug("Session schema similarity check skipped", exc_info=True)

        row = upsert_session_schema(
            user_id=uid,
            sport=normalized_sport,
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
    # define_periodization
    # ------------------------------------------------------------------

    def define_periodization(
        name: str,
        phases: list,
        description: str = "",
    ) -> dict:
        if not name or not name.strip():
            return {"status": "error", "error": "name must not be empty"}
        if not isinstance(phases, list) or len(phases) == 0:
            return {"status": "error", "error": "phases must be a non-empty list"}

        for i, phase in enumerate(phases):
            if not isinstance(phase, dict):
                return {"status": "error", "error": f"phases[{i}] must be an object"}
            if "name" not in phase:
                return {"status": "error", "error": f"phases[{i}] missing required field 'name'"}
            if "weeks" not in phase:
                return {"status": "error", "error": f"phases[{i}] missing required field 'weeks'"}

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        uid = _get_user_id()

        # Light-touch similarity warning (does not block creation)
        from src.db.agent_config_db import get_periodization_models, upsert_periodization_model
        from src.services.config_gc import (
            SIMILARITY_THRESHOLD,
            compute_config_similarity,
        )
        try:
            existing = get_periodization_models(uid)
            for pm in existing:
                if pm.get("name") == name:
                    continue
                similarity = compute_config_similarity(name, pm.get("name", ""))
                if similarity > SIMILARITY_THRESHOLD:
                    logger.warning(
                        "Periodization model '%s' similar to existing '%s' (score=%.3f)",
                        name, pm["name"], similarity,
                    )
        except Exception:
            logger.debug("Periodization similarity check skipped", exc_info=True)

        row = upsert_periodization_model(
            user_id=uid,
            name=name,
            phases=phases,
        )
        return {"status": "success", "periodization_model": row}

    registry.register(Tool(
        name="define_periodization",
        description=(
            "Define or update a periodization model — a multi-phase training structure "
            "that describes how training focus, volume, and intensity change over weeks "
            "or months. Each phase has a name, duration in weeks, focus area, and optional "
            "intensity distribution. Use this to structure macrocycles (e.g., base → build "
            "→ peak → taper)."
        ),
        handler=define_periodization,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique periodization model name (e.g., 'marathon_16_week')"},
                "phases": {
                    "type": "array",
                    "description": (
                        "Ordered list of training phases. Each phase: "
                        "{\"name\": \"Base\", \"weeks\": 4, \"focus\": \"aerobic endurance\", "
                        "\"intensity_distribution\": {\"low\": 80, \"moderate\": 15, \"high\": 5}}"
                    ),
                    "items": {"type": "object"},
                },
                "description": {"type": "string", "description": "Human-readable description of the model"},
            },
            "required": ["name", "phases"],
        },
        category="config",
    ))

    # ------------------------------------------------------------------
    # define_trigger_rule
    # ------------------------------------------------------------------

    def define_trigger_rule(
        name: str,
        condition: str,
        action: str,
        cooldown_hours: int = 24,
    ) -> dict:
        if not name or not name.strip():
            return {"status": "error", "error": "name must not be empty"}
        if not condition or not condition.strip():
            return {"status": "error", "error": "condition must not be empty"}
        if not action or not action.strip():
            return {"status": "error", "error": "action must not be empty"}

        valid, error = CalcEngine.validate_formula(condition)
        if not valid:
            return {"status": "error", "error": f"Invalid condition formula: {error}"}

        if not _settings.use_supabase:
            return {"status": "error", "error": "Supabase not configured"}

        uid = _get_user_id()

        # Light-touch similarity warning (does not block creation)
        from src.db.agent_config_db import get_proactive_trigger_rules, upsert_proactive_trigger_rule
        from src.services.config_gc import (
            SIMILARITY_THRESHOLD,
            compute_config_similarity,
        )
        try:
            existing = get_proactive_trigger_rules(uid)
            for tr in existing:
                if tr.get("name") == name:
                    continue
                similarity = compute_config_similarity(name, tr.get("name", ""))
                if similarity > SIMILARITY_THRESHOLD:
                    logger.warning(
                        "Trigger rule '%s' similar to existing '%s' (score=%.3f)",
                        name, tr["name"], similarity,
                    )
        except Exception:
            logger.debug("Trigger rule similarity check skipped", exc_info=True)

        row = upsert_proactive_trigger_rule(
            user_id=uid,
            name=name,
            condition=condition,
            action=action,
            cooldown_hours=cooldown_hours,
        )
        return {"status": "success", "trigger_rule": row}

    registry.register(Tool(
        name="define_trigger_rule",
        description=(
            "Define or update a proactive trigger rule. When the condition evaluates to "
            "truthy, the agent wakes up and performs the action. The condition is a "
            "CalcEngine formula using context variables: total_sessions_7d, "
            "days_since_last_session, avg_hrv_7d, avg_sleep_score_7d, "
            "body_battery_latest, total_trimp_7d, total_minutes_7d, "
            "avg_resting_hr_7d, stress_avg_latest, recovery_score_latest, "
            "{sport}_sessions_7d, {sport}_trimp_7d. "
            "The action is a natural-language instruction for what to tell the athlete."
        ),
        handler=define_trigger_rule,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique trigger rule name (e.g., 'high_fatigue_alert')"},
                "condition": {
                    "type": "string",
                    "description": (
                        "CalcEngine formula that returns truthy when the trigger should fire "
                        "(e.g., 'total_trimp_7d > 500 and avg_hrv_7d < 40')"
                    ),
                },
                "action": {
                    "type": "string",
                    "description": "What to tell the athlete when the trigger fires (natural language)",
                },
                "cooldown_hours": {
                    "type": "integer",
                    "description": "Hours before the same rule can fire again (default: 24)",
                },
            },
            "required": ["name", "condition", "action"],
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

    _get_config_description = (
        "Retrieve all stored configurations of a given type. Use this to inspect "
        "what metrics, criteria, schemas, or rules are already defined before adding "
        "new ones. config_type must be one of: metric_definitions, eval_criteria, "
        "session_schemas, periodization_models, proactive_trigger_rules."
    )
    _get_config_parameters = {
        "type": "object",
        "properties": {
            "config_type": {
                "type": "string",
                "description": "Type of config to retrieve",
                "enum": sorted(_VALID_CONFIG_TYPES),
            },
        },
        "required": ["config_type"],
    }

    registry.register(Tool(
        name="get_config",
        description=_get_config_description,
        handler=get_config,
        parameters=_get_config_parameters,
        category="config",
    ))

    # Visionplan alias
    registry.register(Tool(
        name="get_agent_config",
        description="Alias for get_config. " + _get_config_description,
        handler=get_config,
        parameters=_get_config_parameters,
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
