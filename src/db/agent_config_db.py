"""Supabase CRUD for agent-defined configuration tables.

The agent defines all formulas, metrics, eval criteria, session schemas,
periodization models, and proactive trigger rules at runtime. This module
persists those definitions so they survive across sessions.

Tables (all have user_id, created_at, updated_at managed by Supabase):
  metric_definitions       — formula-based computed metrics
  eval_criteria            — scoring criteria with weights
  session_schemas          — sport-specific session structure templates
  periodization_models     — multi-phase training model definitions
  proactive_trigger_rules  — conditions that wake the agent proactively
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _upsert(table: str, conflict_columns: list[str], row: dict) -> dict:
    """Generic upsert helper. Returns the upserted row as a plain dict."""
    row = {**row, "updated_at": _now_iso()}
    result = (
        get_supabase()
        .table(table)
        .upsert(row, on_conflict=",".join(conflict_columns))
        .execute()
    )
    return result.data[0]


def _select_all(table: str, user_id: str) -> list[dict]:
    result = (
        get_supabase()
        .table(table)
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


def _select_one(table: str, user_id: str, name: str) -> dict | None:
    result = (
        get_supabase()
        .table(table)
        .select("*")
        .eq("user_id", user_id)
        .eq("name", name)
        .maybe_single()
        .execute()
    )
    return result.data if result is not None else None


def _partial_update(table: str, user_id: str, name: str, updates: dict) -> dict | None:
    updates = {**updates, "updated_at": _now_iso()}
    result = (
        get_supabase()
        .table(table)
        .update(updates)
        .eq("user_id", user_id)
        .eq("name", name)
        .execute()
    )
    return result.data[0] if result.data else None


# ---------------------------------------------------------------------------
# metric_definitions
# ---------------------------------------------------------------------------

def upsert_metric_definition(
    user_id: str,
    name: str,
    formula: str,
    description: str = "",
    unit: str = "",
    variables: dict | None = None,
) -> dict:
    """Create or replace a metric definition."""
    row: dict = {
        "user_id": user_id,
        "name": name,
        "formula": formula,
        "description": description,
        "unit": unit,
        "variables": variables or {},
    }
    return _upsert("metric_definitions", ["user_id", "name"], row)


def get_metric_definitions(user_id: str) -> list[dict]:
    return _select_all("metric_definitions", user_id)


def get_metric_definition(user_id: str, name: str) -> dict | None:
    return _select_one("metric_definitions", user_id, name)


def update_metric_definition(user_id: str, name: str, updates: dict) -> dict | None:
    allowed_keys = {"formula", "description", "unit", "variables"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_keys}
    if not safe_updates:
        return None
    return _partial_update("metric_definitions", user_id, name, safe_updates)


# ---------------------------------------------------------------------------
# eval_criteria
# ---------------------------------------------------------------------------

def upsert_eval_criteria(
    user_id: str,
    name: str,
    description: str = "",
    weight: float = 1.0,
    formula: str = "",
) -> dict:
    row: dict = {
        "user_id": user_id,
        "name": name,
        "description": description,
        "weight": weight,
        "formula": formula,
    }
    return _upsert("eval_criteria", ["user_id", "name"], row)


def get_eval_criteria(user_id: str) -> list[dict]:
    return _select_all("eval_criteria", user_id)


def get_eval_criterion(user_id: str, name: str) -> dict | None:
    return _select_one("eval_criteria", user_id, name)


def update_eval_criterion(user_id: str, name: str, updates: dict) -> dict | None:
    allowed_keys = {"description", "weight", "formula"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_keys}
    if not safe_updates:
        return None
    return _partial_update("eval_criteria", user_id, name, safe_updates)


# ---------------------------------------------------------------------------
# session_schemas
# ---------------------------------------------------------------------------

def upsert_session_schema(
    user_id: str,
    sport: str,
    schema: dict,
) -> dict:
    row: dict = {
        "user_id": user_id,
        "name": sport,   # use sport as the unique name for _select_one compatibility
        "sport": sport,
        "schema": schema,
    }
    return _upsert("session_schemas", ["user_id", "sport"], row)


def get_session_schemas(user_id: str) -> list[dict]:
    return _select_all("session_schemas", user_id)


def get_session_schema(user_id: str, sport: str) -> dict | None:
    result = (
        get_supabase()
        .table("session_schemas")
        .select("*")
        .eq("user_id", user_id)
        .eq("sport", sport)
        .maybe_single()
        .execute()
    )
    return result.data if result is not None else None


def update_session_schema(user_id: str, sport: str, updates: dict) -> dict | None:
    allowed_keys = {"schema"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_keys}
    if not safe_updates:
        return None
    safe_updates = {**safe_updates, "updated_at": _now_iso()}
    result = (
        get_supabase()
        .table("session_schemas")
        .update(safe_updates)
        .eq("user_id", user_id)
        .eq("sport", sport)
        .execute()
    )
    return result.data[0] if result.data else None


# ---------------------------------------------------------------------------
# periodization_models
# ---------------------------------------------------------------------------

def upsert_periodization_model(
    user_id: str,
    name: str,
    phases: list,
) -> dict:
    row: dict = {
        "user_id": user_id,
        "name": name,
        "phases": phases,
    }
    return _upsert("periodization_models", ["user_id", "name"], row)


def get_periodization_models(user_id: str) -> list[dict]:
    return _select_all("periodization_models", user_id)


def get_periodization_model(user_id: str, name: str) -> dict | None:
    return _select_one("periodization_models", user_id, name)


def update_periodization_model(user_id: str, name: str, updates: dict) -> dict | None:
    allowed_keys = {"phases"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_keys}
    if not safe_updates:
        return None
    return _partial_update("periodization_models", user_id, name, safe_updates)


# ---------------------------------------------------------------------------
# proactive_trigger_rules
# ---------------------------------------------------------------------------

def upsert_proactive_trigger_rule(
    user_id: str,
    name: str,
    condition: str,
    action: str,
    cooldown_hours: int = 24,
) -> dict:
    row: dict = {
        "user_id": user_id,
        "name": name,
        "condition": condition,
        "action": action,
        "cooldown_hours": cooldown_hours,
    }
    return _upsert("proactive_trigger_rules", ["user_id", "name"], row)


def get_proactive_trigger_rules(user_id: str) -> list[dict]:
    return _select_all("proactive_trigger_rules", user_id)


def get_proactive_trigger_rule(user_id: str, name: str) -> dict | None:
    return _select_one("proactive_trigger_rules", user_id, name)


def update_proactive_trigger_rule(user_id: str, name: str, updates: dict) -> dict | None:
    allowed_keys = {"condition", "action", "cooldown_hours"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_keys}
    if not safe_updates:
        return None
    return _partial_update("proactive_trigger_rules", user_id, name, safe_updates)
