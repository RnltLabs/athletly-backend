"""Tests for CalcEngine — safe formula evaluation via evalidate.

Covers:
- Basic arithmetic
- Real-world TRIMP formula
- Division by zero → None
- Missing variable → None
- Security: __import__, open(), attribute access all blocked
- validate_formula (True/False returns)
- calculate_bulk over multiple records
- Math functions: sqrt, abs, min, max, round
"""

import math

import pytest

from src.calc.engine import CalcEngine


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------


def test_simple_formula():
    result = CalcEngine.calculate("a + b", {"a": 3.0, "b": 4.0})
    assert result == pytest.approx(7.0)


def test_subtraction():
    result = CalcEngine.calculate("a - b", {"a": 10.0, "b": 3.5})
    assert result == pytest.approx(6.5)


def test_multiplication():
    result = CalcEngine.calculate("a * b", {"a": 2.5, "b": 4.0})
    assert result == pytest.approx(10.0)


def test_power():
    result = CalcEngine.calculate("a ** 2", {"a": 3.0})
    assert result == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Real-world formula: TRIMP
# ---------------------------------------------------------------------------


def test_trimp_formula():
    """TRIMP = duration * hr_ratio * 0.64 * exp(1.92 * hr_ratio)"""
    formula = "duration * hr_ratio * 0.64 * exp(1.92 * hr_ratio)"
    variables = {"duration": 60.0, "hr_ratio": 0.75}
    expected = 60.0 * 0.75 * 0.64 * math.exp(1.92 * 0.75)
    result = CalcEngine.calculate(formula, variables)
    assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Error / edge cases → None
# ---------------------------------------------------------------------------


def test_division_by_zero_returns_none():
    result = CalcEngine.calculate("a / b", {"a": 1.0, "b": 0.0})
    assert result is None


def test_missing_variable_returns_none():
    result = CalcEngine.calculate("a + b", {"a": 5.0})  # 'b' missing
    assert result is None


def test_empty_formula_returns_none():
    assert CalcEngine.calculate("", {}) is None
    assert CalcEngine.calculate("   ", {}) is None


# ---------------------------------------------------------------------------
# Security: injection attacks must be blocked
# ---------------------------------------------------------------------------


def test_import_attack_blocked():
    """__import__('os') must be rejected at validation/evaluation time."""
    result = CalcEngine.calculate("__import__('os')", {})
    assert result is None


def test_open_attack_blocked():
    """open('/etc/passwd') must be rejected."""
    result = CalcEngine.calculate("open('/etc/passwd')", {})
    assert result is None


def test_attribute_access_blocked():
    """Attribute chains like ''.__class__ must be blocked."""
    result = CalcEngine.calculate("''.__class__", {})
    assert result is None


def test_exec_blocked():
    """exec() must be blocked."""
    result = CalcEngine.calculate("exec('import os')", {})
    assert result is None


def test_eval_blocked():
    """eval() must be blocked."""
    result = CalcEngine.calculate("eval('1+1')", {})
    assert result is None


# ---------------------------------------------------------------------------
# validate_formula
# ---------------------------------------------------------------------------


def test_validate_valid_formula():
    ok, err = CalcEngine.validate_formula("a + b * 2")
    assert ok is True
    assert err is None


def test_validate_invalid_formula_import():
    ok, err = CalcEngine.validate_formula("__import__('os')")
    assert ok is False
    assert err is not None
    assert isinstance(err, str)
    assert len(err) > 0


def test_validate_invalid_formula_syntax():
    ok, err = CalcEngine.validate_formula("a ++ b")  # syntax error
    assert ok is False
    assert err is not None


def test_validate_empty_formula():
    ok, err = CalcEngine.validate_formula("")
    assert ok is False
    assert err is not None


def test_validate_whitespace_only():
    ok, err = CalcEngine.validate_formula("   ")
    assert ok is False
    assert err is not None


# ---------------------------------------------------------------------------
# calculate_bulk
# ---------------------------------------------------------------------------


def test_calculate_bulk():
    formula = "x * 2"
    records = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    results = CalcEngine.calculate_bulk(formula, records)
    assert results == [pytest.approx(2.0), pytest.approx(4.0), pytest.approx(6.0)]


def test_calculate_bulk_with_failures():
    """Records missing a variable produce None; others still succeed."""
    formula = "a + b"
    records = [
        {"a": 1.0, "b": 2.0},   # ok → 3.0
        {"a": 5.0},              # missing b → None
        {"a": 10.0, "b": 5.0},  # ok → 15.0
    ]
    results = CalcEngine.calculate_bulk(formula, records)
    assert results[0] == pytest.approx(3.0)
    assert results[1] is None
    assert results[2] == pytest.approx(15.0)


def test_calculate_bulk_empty_records():
    results = CalcEngine.calculate_bulk("a + b", [])
    assert results == []


def test_calculate_bulk_bad_formula():
    """A syntactically invalid formula → all None."""
    results = CalcEngine.calculate_bulk("__import__('os')", [{"x": 1}])
    assert results == [None]


def test_calculate_bulk_same_length_as_input():
    formula = "x"
    records = [{"x": float(i)} for i in range(10)]
    results = CalcEngine.calculate_bulk(formula, records)
    assert len(results) == 10


# ---------------------------------------------------------------------------
# Math functions
# ---------------------------------------------------------------------------


def test_sqrt_function():
    result = CalcEngine.calculate("sqrt(x)", {"x": 16.0})
    assert result == pytest.approx(4.0)


def test_abs_function():
    result = CalcEngine.calculate("abs(x)", {"x": -7.5})
    assert result == pytest.approx(7.5)


def test_min_function():
    result = CalcEngine.calculate("min(a, b)", {"a": 3.0, "b": 5.0})
    assert result == pytest.approx(3.0)


def test_max_function():
    result = CalcEngine.calculate("max(a, b)", {"a": 3.0, "b": 5.0})
    assert result == pytest.approx(5.0)


def test_round_function():
    result = CalcEngine.calculate("round(x, 2)", {"x": 3.14159})
    assert result == pytest.approx(3.14)


def test_exp_function():
    result = CalcEngine.calculate("exp(x)", {"x": 1.0})
    assert result == pytest.approx(math.e, rel=1e-6)


def test_log_function():
    result = CalcEngine.calculate("log(x)", {"x": math.e})
    assert result == pytest.approx(1.0, rel=1e-6)


def test_pow_function():
    result = CalcEngine.calculate("pow(a, b)", {"a": 2.0, "b": 8.0})
    assert result == pytest.approx(256.0)
