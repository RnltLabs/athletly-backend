"""CalcEngine -- safe formula evaluation using evalidate.

Uses evalidate's AST-whitelist approach: only explicitly allowed nodes and
function names can appear in a formula. This blocks `__import__`, attribute
chains, and all other injection vectors.
"""

from __future__ import annotations

import math

from evalidate import Expr, EvalModel, base_eval_model, EvalException


# ---------------------------------------------------------------------------
# Build a safe evaluation model once at module level (immutable after setup)
# ---------------------------------------------------------------------------

def _build_calc_model() -> EvalModel:
    """Create a locked-down eval model that allows math expressions only."""
    model = base_eval_model.clone()

    # Arithmetic nodes missing from base_eval_model
    for node in ("Mult", "Pow", "FloorDiv", "MatMult"):
        if node not in model.nodes:
            model.nodes.append(node)

    # Allow function calls (required for sqrt, min, max, etc.)
    if "Call" not in model.nodes:
        model.nodes.append("Call")

    # Explicitly whitelist the functions the agent may use
    safe_funcs = [
        "exp", "log", "log2", "log10", "sqrt", "abs",
        "min", "max", "pow", "sum", "len", "round", "int", "float",
    ]
    for fn in safe_funcs:
        if fn not in model.allowed_functions:
            model.allowed_functions.append(fn)

    # Inject math implementations so names resolve correctly
    for fn in safe_funcs:
        if hasattr(math, fn) and fn not in model.imported_functions:
            model.imported_functions[fn] = getattr(math, fn)

    # Builtins that shadow math (min, max, sum, len, round, int, float, abs)
    builtin_overrides = {
        "min": min, "max": max, "sum": sum, "len": len,
        "round": round, "int": int, "float": float, "abs": abs,
        "pow": pow,
    }
    for fn, impl in builtin_overrides.items():
        model.imported_functions[fn] = impl

    return model


_CALC_MODEL: EvalModel = _build_calc_model()


# ---------------------------------------------------------------------------
# CalcEngine
# ---------------------------------------------------------------------------

class CalcEngine:
    """Stateless helper for safe formula validation and evaluation."""

    @staticmethod
    def validate_formula(formula: str) -> tuple[bool, str | None]:
        """Dry-run validation without executing.

        Returns:
            (True, None) on success or (False, error_message) on failure.
        """
        if not formula or not formula.strip():
            return False, "Formula must not be empty"
        try:
            Expr(formula, model=_CALC_MODEL)
            return True, None
        except EvalException as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001 — catch-all for safety
            return False, f"Unexpected validation error: {exc}"

    @staticmethod
    def calculate(formula: str, variables: dict) -> float | None:
        """Evaluate a formula with the given variable bindings.

        Args:
            formula: A safe math expression string.
            variables: Dict of variable names → numeric values.

        Returns:
            float result, or None on any error (bad formula, missing var, etc.).
        """
        if not formula or not formula.strip():
            return None
        try:
            expr = Expr(formula, model=_CALC_MODEL)
            result = expr.eval(variables)
            if result is None:
                return None
            result = float(result)
            if math.isnan(result) or math.isinf(result):
                return None
            return result
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def calculate_bulk(formula: str, records: list[dict]) -> list[float | None]:
        """Evaluate a formula against multiple variable dicts.

        Compiles the expression once and evaluates against each record.
        Records that fail evaluation produce None in the output list.

        Args:
            formula: A safe math expression string.
            records: List of variable dicts, one per data point.

        Returns:
            List of float | None, same length as records.
        """
        if not formula or not formula.strip():
            return [None] * len(records)

        try:
            expr = Expr(formula, model=_CALC_MODEL)
        except Exception:  # noqa: BLE001
            return [None] * len(records)

        results: list[float | None] = []
        for record in records:
            try:
                raw = expr.eval(record)
                if raw is None:
                    results.append(None)
                    continue
                value = float(raw)
                results.append(None if (math.isnan(value) or math.isinf(value)) else value)
            except Exception:  # noqa: BLE001
                results.append(None)

        return results
