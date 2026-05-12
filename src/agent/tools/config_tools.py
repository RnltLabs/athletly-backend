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

    # MERGED into define_config(config_type="metric", ...). NOT registered standalone.
    # define_metric is kept as an internal helper used by define_config.

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

    # MERGED into define_config(config_type="eval_criteria", ...). NOT registered standalone.
    # define_eval_criteria is kept as an internal helper used by define_config.

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

    # MERGED into define_config(config_type="session_schema", ...). NOT registered standalone.
    # define_session_schema is kept as an internal helper used by define_config.

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

    # MERGED into define_config(config_type="periodization", ...). NOT registered standalone.
    # define_periodization is kept as an internal helper used by define_config.

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

    # MERGED into define_config(config_type="trigger_rule", ...). NOT registered standalone.
    # define_trigger_rule is kept as an internal helper used by define_config.

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

    # DEPRECATED: get_agent_config is a duplicate of get_config. NOT registered.
    # registry.register(Tool(
    #     name="get_agent_config",
    #     description="Alias for get_config. " + _get_config_description,
    #     handler=get_config,
    #     parameters=_get_config_parameters,
    #     category="config",
    # ))

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

    # MERGED into define_config: re-calling define_config with same (config_type, name)
    # upserts the entry. NOT registered standalone.
    # update_config is kept as an internal helper for legacy callers.

    # ------------------------------------------------------------------
    # define_config (unified tool replacing 5 define_* + update_config)
    # ------------------------------------------------------------------

    # Map user-facing config_type strings to (internal_handler, db_config_type).
    # The 5 internal handlers above (define_metric, define_eval_criteria,
    # define_session_schema, define_periodization, define_trigger_rule) stay
    # available as Python callables for backwards compatibility.
    _DEFINE_CONFIG_TYPES = frozenset({
        "metric",
        "eval_criteria",
        "periodization",
        "session_schema",
        "trigger_rule",
    })

    def define_config(
        config_type: str,
        name: str,
        definition: dict,
        description: str = "",
    ) -> dict:
        """Define or update an agent-runtime config entry.

        Routes to the correct internal define_* helper based on config_type
        and upserts on (user_id, config_type, name). Same call replaces
        existing entries.
        """
        if config_type not in _DEFINE_CONFIG_TYPES:
            return {
                "status": "error",
                "error": (
                    f"Invalid config_type '{config_type}'. "
                    f"Use one of: {sorted(_DEFINE_CONFIG_TYPES)}"
                ),
            }

        if not name or not name.strip():
            return {"status": "error", "error": "name must not be empty"}

        if not isinstance(definition, dict):
            return {"status": "error", "error": "definition must be an object"}

        try:
            if config_type == "metric":
                return define_metric(
                    name=name,
                    formula=definition.get("formula", ""),
                    description=definition.get("description", description),
                    unit=definition.get("unit", ""),
                    variables=definition.get("variables"),
                )

            if config_type == "eval_criteria":
                return define_eval_criteria(
                    name=name,
                    description=definition.get("description", description),
                    weight=definition.get("weight", 1.0),
                    formula=definition.get("formula", ""),
                )

            if config_type == "session_schema":
                # For session_schema 'name' is the sport identifier.
                schema = definition.get("schema", definition)
                return define_session_schema(sport=name, schema=schema)

            if config_type == "periodization":
                return define_periodization(
                    name=name,
                    phases=definition.get("phases", []),
                    description=definition.get("description", description),
                )

            if config_type == "trigger_rule":
                return define_trigger_rule(
                    name=name,
                    condition=definition.get("condition", ""),
                    action=definition.get("action", ""),
                    cooldown_hours=definition.get("cooldown_hours", 24),
                )
        except Exception as exc:
            logger.exception("define_config dispatch failed")
            return {"status": "error", "error": str(exc)}

        # Should be unreachable (covered by validation above).
        return {"status": "error", "error": f"Unhandled config_type: {config_type}"}

    registry.register(Tool(
        name="define_config",
        description=(
            "Define or update an agent-runtime config entry. Upserts on "
            "(config_type, name): same call replaces an existing entry. "
            "Valid config_type values: 'metric' (custom CalcEngine formula "
            "evaluable via calculate_metric; definition keys: formula, "
            "description, unit, variables), 'eval_criteria' (evaluate_plan "
            "scoring criterion; keys: description, weight, formula), "
            "'periodization' (reusable macrocycle phase sequence; keys: "
            "phases [list of {name, weeks, focus, intensity_distribution}], "
            "description), 'session_schema' (per-sport session template; "
            "name=sport; keys: schema), 'trigger_rule' (proactive condition "
            "+ action; keys: condition CalcEngine formula, action text, "
            "cooldown_hours). The definition object shape varies per type. "
            "Persists per user in Supabase."
        ),
        handler=define_config,
        parameters={
            "type": "object",
            "properties": {
                "config_type": {
                    "type": "string",
                    "description": "Kind of config entry to define.",
                    "enum": sorted(_DEFINE_CONFIG_TYPES),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Unique key within the config_type "
                        "(for session_schema this is the sport name)."
                    ),
                },
                "definition": {
                    "type": "object",
                    "description": (
                        "Type-specific definition body. See description for "
                        "the keys expected per config_type."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Optional human-readable explanation.",
                },
            },
            "required": ["config_type", "name", "definition"],
        },
        category="config",
    ))
