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
            "Persist a named metric formula that the coach can later compute against "
            "activity data, health data, or any numeric variables in context. Metrics "
            "are sandboxed math expressions evaluated by CalcEngine - no imports, no "
            "attribute access, no I/O. This is how the agent INVENTS new measurements "
            "(e.g., a custom load index, a recovery score, a polarization ratio) "
            "without hardcoding them in the backend.\n\n"
            "WHEN TO USE:\n"
            "- The athlete has a sport or goal that needs a measurement we have not "
            "defined yet: invent the formula, save it, then use calc_metric to "
            "evaluate.\n"
            "- You want to track a custom indicator over time (e.g., 'aerobic_index = "
            "avg_hr / avg_pace_min_km' for running efficiency).\n"
            "- A trigger rule needs a derived value: define the metric first, then "
            "reference its name in define_trigger_rule.\n\n"
            "WHEN NOT TO USE:\n"
            "- For one-off calculations you only need now: do the math in your head "
            "or in a single calc_engine call without persisting.\n"
            "- For data the system already provides (avg_hr, distance_km, trimp): "
            "no metric needed, just read the field.\n"
            "- For non-numeric concepts (categories, labels): metrics must produce a "
            "number.\n\n"
            "FORMULA RULES (CalcEngine):\n"
            "- Operators: +, -, *, /, **, %, comparison (<, >, ==, etc.), logical "
            "(and, or, not).\n"
            "- Functions: abs, min, max, round, sqrt, log, exp, sum, len, int, "
            "float, pow.\n"
            "- Variables: any snake_case identifier (you decide the names; pass them "
            "in at calc time).\n"
            "- Forbidden: imports, attribute access, lambdas, comprehensions over "
            "untrusted data, __dunder__ names.\n"
            "- Invalid formulas are rejected at save time - the error is returned, "
            "fix and retry.\n\n"
            "EXAMPLES:\n"
            "- define_metric(name='aerobic_decoupling', formula='(second_half_hr / "
            "first_half_hr) - 1', unit='ratio', description='Cardiac drift over a "
            "long run; >0.05 suggests aerobic insufficiency').\n"
            "- define_metric(name='polarization_ratio', formula='(low_intensity_min "
            "+ high_intensity_min) / max(1, moderate_intensity_min)', "
            "unit='ratio', description='Polarized training adherence; higher is "
            "more polarized').\n"
            "- define_metric(name='vo2max_pace_running', formula='3600 / (vdot * "
            "0.8)', unit='sec/km').\n\n"
            "DEDUP: a semantic similarity check is run against existing metrics. "
            "If a new formula is too similar to an existing one (same name OR same "
            "math + similar description), the tool returns "
            "{'status': 'duplicate', 'existing_name': ..., 'similarity': ...} "
            "instead of saving. Use update_config(config_type='metric_definitions', "
            "name=<existing>, updates=...) to modify the existing metric.\n\n"
            "SIDE EFFECT: persisted per-user in Supabase. Survives sessions. "
            "Requires use_supabase=true; in CLI/local mode returns "
            "{'status': 'error', 'error': 'Supabase not configured'}."
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
            "Persist a named evaluation criterion that evaluate_plan will use to "
            "score training plans. This is how the coach INVENTS quality criteria "
            "specific to this athlete's sport, goal, and context - no hardcoded "
            "list. Without defined criteria, evaluate_plan auto-accepts every plan "
            "(score=100), so define criteria early.\n\n"
            "WHEN TO USE:\n"
            "- During onboarding once you understand the athlete's sport and goal: "
            "define 3-7 criteria that capture what 'a good plan' means for this "
            "athlete (recovery distribution, progressive overload, sport balance, "
            "key-session presence, etc.).\n"
            "- After spotting a recurring plan defect (e.g., 'plans keep stacking "
            "hard days'): codify it as a criterion so evaluator catches it next "
            "time.\n"
            "- When a new training phase or goal shifts what 'good' means: add or "
            "update criteria to match.\n\n"
            "WHEN NOT TO USE:\n"
            "- For one-off plan critique: just reason about it directly. Criteria "
            "should be reusable.\n"
            "- For very general criteria that apply to all athletes ('plan should "
            "have sessions'): these are too trivial to be useful and pollute "
            "scoring.\n\n"
            "ARGS:\n"
            "- name: snake_case identifier ('adequate_recovery_days', "
            "'progressive_overload', 'sport_balance_running_cycling').\n"
            "- description: what this criterion checks, in coach language - the "
            "evaluator LLM uses this prose to interpret the criterion when no "
            "formula is given.\n"
            "- weight: relative importance. 1.0 baseline. 2.0 = twice as important "
            "as a 1.0 criterion. Use 0.5 for nice-to-haves, 3.0+ for safety-"
            "critical criteria like injury prevention.\n"
            "- formula: OPTIONAL CalcEngine expression returning 0-100. If "
            "provided, evaluator uses the numeric score directly. If omitted, "
            "evaluator scores qualitatively from the description.\n\n"
            "EXAMPLES:\n"
            "- define_eval_criteria(name='recovery_adequacy', description='Plan "
            "includes at least one rest or recovery day between two high-intensity "
            "sessions', weight=2.0).\n"
            "- define_eval_criteria(name='weekly_volume_in_target_range', "
            "description='Total weekly volume is within +/-10% of the athlete\\'s "
            "current baseline', weight=1.5, formula='100 - abs(total_minutes - "
            "target_minutes) / target_minutes * 100').\n"
            "- define_eval_criteria(name='youth_safety', description='No session "
            "exceeds 90 minutes for athletes under 18; no double sessions on the "
            "same day', weight=3.0).\n\n"
            "DUPLICATE BEHAVIOR: same name updates the existing criterion. "
            "Similar names trigger a non-blocking warning - the new criterion is "
            "still saved, but consider whether you meant to update_config "
            "instead.\n\n"
            "SIDE EFFECT: persisted per-user in Supabase, survives sessions. "
            "Affects every subsequent evaluate_plan call until explicitly removed "
            "via update_config (or weight set to 0)."
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
            "Define the structural template for training sessions of a given sport. "
            "The schema declares which fields a session of this sport SHOULD have "
            "(duration, intensity zones, target metrics, sport-specific params like "
            "stroke for swimming or cadence for cycling) so create_training_plan "
            "produces well-formed sessions consistently.\n\n"
            "WHEN TO USE:\n"
            "- A new sport enters the athlete's life and no schema exists yet for "
            "it. The coach should define one before the first plan involving that "
            "sport.\n"
            "- Existing schema is missing fields the coach now needs to specify "
            "(e.g., adding 'cadence_target_rpm' to cycling sessions): use "
            "update_config to extend it instead of recreating.\n"
            "- Athlete picks up a niche sport (CrossFit, climbing, rowing, "
            "skimo, kayak) that the system has no prior template for - define it "
            "from scratch.\n\n"
            "WHEN NOT TO USE:\n"
            "- For session-instance values (today's run was 45 min, 8km): those "
            "are activity records, not schemas.\n"
            "- For minor variations of an existing sport (e.g., trail running): "
            "extend the running schema instead of creating 'trail_running'.\n\n"
            "ARGS:\n"
            "- sport: lowercase identifier ('running', 'cycling', 'swimming', "
            "'strength', 'mobility'). Normalized to lowercase + stripped.\n"
            "- schema: free-form dict describing the session structure. Common "
            "keys: 'required_fields' (list), 'optional_fields' (list), "
            "'session_types' (list of allowed type strings), 'intensity_zones' "
            "(dict of zone names -> definitions), 'sport_specific' (nested dict).\n\n"
            "EXAMPLE:\n"
            "  define_session_schema(\n"
            "    sport='swimming',\n"
            "    schema={\n"
            "      'required_fields': ['type', 'duration_min', 'total_meters'],\n"
            "      'optional_fields': ['stroke', 'pool_length_m', 'rpe'],\n"
            "      'session_types': ['endurance', 'threshold', 'sprints', "
            "'technique', 'recovery'],\n"
            "      'intensity_zones': {'z1': 'recovery', 'z2': 'aerobic', "
            "'z3': 'tempo', 'z4': 'threshold', 'z5': 'vo2max'},\n"
            "      'sport_specific': {'stroke_options': ['freestyle', 'back', "
            "'breast', 'fly', 'mixed'], 'pool_length_options': [25, 50]}\n"
            "    }\n"
            "  )\n\n"
            "DUPLICATE BEHAVIOR: same sport name (after normalization) updates the "
            "existing schema. Similar sport names (e.g., 'cycling' vs 'biking') "
            "trigger a warning but still save - prefer using update_config to "
            "extend the existing schema.\n\n"
            "SIDE EFFECT: persisted per-user in Supabase. Used by create_training_plan "
            "to constrain session generation."
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
            "Persist a named periodization model: a reusable phase sequence "
            "(base -> build -> peak -> taper, or polarized blocks, or block "
            "periodization, etc.) that create_macrocycle_plan can use as a "
            "template. Each phase declares duration in weeks, focus area, and "
            "optional intensity distribution.\n\n"
            "WHEN TO USE:\n"
            "- The athlete (or you, the coach) wants a SPECIFIC methodology "
            "applied consistently across macrocycles ('always taper for 14 days', "
            "'use polarized 80/20 in base, threshold focus in build').\n"
            "- After researching a methodology (e.g., 80/20 Lydiard, Jack Daniels, "
            "Maffetone, Friel) and deciding to codify it for reuse.\n"
            "- To capture an existing macrocycle's structure as a reusable model "
            "so the next season can apply the same template.\n\n"
            "WHEN NOT TO USE:\n"
            "- For a one-off macrocycle: just call create_macrocycle_plan without "
            "periodization_model. Only define a model if it will be reused.\n"
            "- For session-level templates: that is define_session_schema.\n"
            "- For a single phase: a model is a SEQUENCE of phases. Define the "
            "full sequence even if currently only the first phase matters.\n\n"
            "ARGS:\n"
            "- name: snake_case identifier ('marathon_16_week_polarized', "
            "'block_periodization_strength_cycle').\n"
            "- phases: ordered list of phase dicts. Each phase MUST have 'name' "
            "(string) and 'weeks' (int). Strongly recommended: 'focus' (string) "
            "and 'intensity_distribution' (dict with low/moderate/high percents "
            "summing to 100).\n"
            "- description: human-readable explanation of when to use this model, "
            "what athlete profile it suits, and key principles.\n\n"
            "EXAMPLE:\n"
            "  define_periodization(\n"
            "    name='marathon_16_week_polarized',\n"
            "    description='80/20 polarized model for marathon build with a "
            "12-day taper. Suits athletes with 3+ years aerobic base.',\n"
            "    phases=[\n"
            "      {'name': 'Base', 'weeks': 6, 'focus': 'aerobic endurance', "
            "'intensity_distribution': {'low': 85, 'moderate': 10, 'high': 5}},\n"
            "      {'name': 'Build', 'weeks': 6, 'focus': 'threshold and tempo', "
            "'intensity_distribution': {'low': 75, 'moderate': 15, 'high': 10}},\n"
            "      {'name': 'Peak', 'weeks': 2, 'focus': 'race-specific "
            "intensity', 'intensity_distribution': {'low': 70, 'moderate': 15, "
            "'high': 15}},\n"
            "      {'name': 'Taper', 'weeks': 2, 'focus': 'volume drop, intensity "
            "retention', 'intensity_distribution': {'low': 75, 'moderate': 10, "
            "'high': 15}}\n"
            "    ]\n"
            "  )\n\n"
            "VALIDATION: phases must be a non-empty list. Each phase missing "
            "'name' or 'weeks' returns an error and the model is not saved. Fix "
            "and retry.\n\n"
            "Similar-name warning is logged but does not block save - prefer "
            "update_config when iterating on an existing model.\n\n"
            "SIDE EFFECT: persisted per-user in Supabase. Referenced by name "
            "when calling create_macrocycle_plan(periodization_model='...')."
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
            "Persist a proactive trigger rule: a condition formula that is "
            "evaluated periodically against the athlete's recent training and "
            "health context, and an action instruction the agent will execute "
            "when the condition fires. This is how the coach proactively "
            "reaches out instead of waiting for the athlete to message first.\n\n"
            "WHEN TO USE:\n"
            "- The athlete has a pattern they want the coach to watch for and "
            "respond to without being asked ('warn me if I'm overtraining', "
            "'nudge me if I miss 3 days', 'celebrate when I PR').\n"
            "- You observe a recurring problem (fatigue spikes, missed sessions, "
            "HRV crashes) and want automatic intervention before the next "
            "occurrence.\n"
            "- After defining a custom metric: hook it up to a trigger so the "
            "metric does something useful.\n\n"
            "WHEN NOT TO USE:\n"
            "- For one-off concerns ('check on me Friday'): use a scheduled "
            "notification, not a permanent rule.\n"
            "- For conditions that cannot be expressed numerically: triggers "
            "evaluate formulas. Subjective conditions need a different mechanism.\n"
            "- When the existing rule does most of what you want: use "
            "update_config to tune the condition or cooldown, do not pile on "
            "near-duplicate rules.\n\n"
            "AVAILABLE CONTEXT VARIABLES (CalcEngine):\n"
            "- total_sessions_7d, total_minutes_7d, total_trimp_7d - 7-day "
            "training volume rollups.\n"
            "- days_since_last_session - integer.\n"
            "- avg_hrv_7d, avg_sleep_score_7d, avg_resting_hr_7d - 7-day health "
            "averages.\n"
            "- body_battery_latest, stress_avg_latest, recovery_score_latest - "
            "most recent reading.\n"
            "- {sport}_sessions_7d, {sport}_trimp_7d - per-sport rollups "
            "(e.g., running_sessions_7d, cycling_trimp_7d).\n"
            "- Plus any custom metric defined via define_metric.\n\n"
            "ARGS:\n"
            "- name: snake_case identifier. Used by update_config and dedup.\n"
            "- condition: CalcEngine formula returning truthy when the rule "
            "should fire. Validated at save time - invalid formulas are "
            "rejected with the error returned.\n"
            "- action: natural-language instruction the agent reads when the "
            "rule fires. Be specific about tone and content.\n"
            "- cooldown_hours: minimum hours between firings of the same rule "
            "(default 24). Use higher values (72-168) for celebratory/positive "
            "rules, lower (12-24) for safety/fatigue rules.\n\n"
            "EXAMPLES:\n"
            "- define_trigger_rule(name='overtraining_warning', "
            "condition='total_trimp_7d > 500 and avg_hrv_7d < 40', "
            "action='Warn the athlete that recent load is high while HRV is "
            "trending low. Suggest 1-2 easy/rest days and ask about sleep and "
            "stress.', cooldown_hours=48).\n"
            "- define_trigger_rule(name='missed_training_check_in', "
            "condition='days_since_last_session >= 3', "
            "action='Check in warmly - ask if everything is okay, offer to "
            "adapt the plan if life is busy. Do not lecture.', "
            "cooldown_hours=72).\n"
            "- define_trigger_rule(name='strong_recovery_celebrate', "
            "condition='avg_sleep_score_7d > 85 and avg_hrv_7d > 70', "
            "action='Celebrate the strong recovery numbers and propose adding "
            "one quality session this week if the athlete feels up for it.', "
            "cooldown_hours=168).\n\n"
            "SIDE EFFECT: persisted per-user in Supabase. Evaluated by the "
            "proactive trigger evaluator on a schedule - rules with truthy "
            "conditions and expired cooldowns cause the agent to send a "
            "message to the athlete.\n\n"
            "Empty/whitespace name, condition, or action are rejected. "
            "Similar-name warning logged but does not block save."
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
