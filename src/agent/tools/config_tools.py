"""Config tools: agent tools for defining and retrieving runtime configuration.

The agent calls these tools to persist metrics and per-sport session schemas.
All definitions are stored per-user in Supabase and survive across sessions.

Formulas are validated via CalcEngine before persisting.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.calc.engine import CalcEngine
from src.config import get_settings

logger = logging.getLogger(__name__)

_VALID_CONFIG_TYPES = frozenset({
    "metric_definitions",
    "session_schemas",
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

        from src.db.agent_config_db import get_metric_definitions, upsert_metric_definition
        from src.services.config_gc import (
            SIMILARITY_THRESHOLD,
            compute_weighted_config_similarity,
        )
        try:
            existing = get_metric_definitions(uid)
            for m in existing:
                if m.get("name") == name:
                    continue
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
            "session_schemas": db.get_session_schemas,
        }[config_type]

        items = fetch_fn(uid)
        return {"status": "success", "config_type": config_type, "items": items, "count": len(items)}

    _get_config_description = (
        "Retrieve all stored configurations of a given type. Use this to inspect "
        "what metrics or session schemas are already defined before adding new ones. "
        "config_type must be one of: metric_definitions, session_schemas."
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

    # ------------------------------------------------------------------
    # define_config (unified entry point)
    # ------------------------------------------------------------------

    _DEFINE_CONFIG_TYPES = frozenset({
        "metric",
        "session_schema",
    })

    def define_config(
        config_type: str,
        name: str,
        definition: dict,
        description: str = "",
    ) -> dict:
        """Define or update an agent-runtime config entry.

        Routes to the correct internal helper based on config_type and
        upserts on (user_id, config_type, name).
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

            if config_type == "session_schema":
                schema = definition.get("schema", definition)
                return define_session_schema(sport=name, schema=schema)
        except Exception as exc:
            logger.exception("define_config dispatch failed")
            return {"status": "error", "error": str(exc)}

        return {"status": "error", "error": f"Unhandled config_type: {config_type}"}

    registry.register(Tool(
        name="define_config",
        description=(
            "Define or update an agent-runtime config entry. Upserts on "
            "(config_type, name): same call replaces an existing entry. "
            "Valid config_type values: 'metric' (custom CalcEngine formula "
            "evaluable via calculate_metric; definition keys: formula, "
            "description, unit, variables), 'session_schema' (per-sport "
            "session template; name=sport; keys: schema). Persists per "
            "user in Supabase."
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
