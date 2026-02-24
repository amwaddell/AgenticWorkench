"""
Tests for the CalculatorTool (Day 5).

Validates:
- Basic arithmetic (+, -, *, /, //, %, **)
- Math functions (sqrt, log, sin, cos, abs, etc.)
- Constants (pi, e)
- Comparisons
- Safety: rejects code injection, imports, attribute access
- ToolSpec integration
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from workbench.tools.utility.calculator import (
    CalculatorArgs,
    CalculatorTool,
    safe_eval,
)

# ------------------------------------------------------------------ #
#  safe_eval unit tests                                               #
# ------------------------------------------------------------------ #


class TestSafeEvalArithmetic:
    """Basic arithmetic operations."""

    def test_addition(self) -> None:
        assert safe_eval("2 + 3") == 5

    def test_subtraction(self) -> None:
        assert safe_eval("10 - 4") == 6

    def test_multiplication(self) -> None:
        assert safe_eval("3 * 7") == 21

    def test_division(self) -> None:
        assert safe_eval("15 / 4") == 3.75

    def test_floor_division(self) -> None:
        assert safe_eval("15 // 4") == 3

    def test_modulo(self) -> None:
        assert safe_eval("17 % 5") == 2

    def test_exponentiation(self) -> None:
        assert safe_eval("2 ** 10") == 1024

    def test_unary_negative(self) -> None:
        assert safe_eval("-5") == -5

    def test_unary_positive(self) -> None:
        assert safe_eval("+5") == 5

    def test_operator_precedence(self) -> None:
        assert safe_eval("2 + 3 * 4") == 14

    def test_parentheses(self) -> None:
        assert safe_eval("(2 + 3) * 4") == 20

    def test_nested_parens(self) -> None:
        assert safe_eval("((2 + 3) * (4 - 1))") == 15

    def test_float_literal(self) -> None:
        assert safe_eval("3.14 * 2") == pytest.approx(6.28)

    def test_negative_result(self) -> None:
        assert safe_eval("3 - 10") == -7

    def test_zero_division_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 / 0")


class TestSafeEvalFunctions:
    """Math function calls."""

    def test_sqrt(self) -> None:
        assert safe_eval("sqrt(144)") == 12.0

    def test_sqrt_non_perfect(self) -> None:
        assert safe_eval("sqrt(2)") == pytest.approx(math.sqrt(2))

    def test_log_natural(self) -> None:
        assert safe_eval("log(e)") == pytest.approx(1.0)

    def test_log10(self) -> None:
        assert safe_eval("log10(100)") == pytest.approx(2.0)

    def test_log2(self) -> None:
        assert safe_eval("log2(8)") == pytest.approx(3.0)

    def test_sin(self) -> None:
        assert safe_eval("sin(0)") == pytest.approx(0.0)

    def test_cos(self) -> None:
        assert safe_eval("cos(0)") == pytest.approx(1.0)

    def test_tan(self) -> None:
        assert safe_eval("tan(0)") == pytest.approx(0.0)

    def test_abs_positive(self) -> None:
        assert safe_eval("abs(5)") == 5

    def test_abs_negative(self) -> None:
        assert safe_eval("abs(-5)") == 5

    def test_ceil(self) -> None:
        assert safe_eval("ceil(3.2)") == 4

    def test_floor(self) -> None:
        assert safe_eval("floor(3.8)") == 3

    def test_round(self) -> None:
        assert safe_eval("round(3.7)") == 4

    def test_exp(self) -> None:
        assert safe_eval("exp(1)") == pytest.approx(math.e)

    def test_pow_function(self) -> None:
        assert safe_eval("pow(2, 10)") == 1024

    def test_min_function(self) -> None:
        assert safe_eval("min(3, 1, 2)") == 1

    def test_max_function(self) -> None:
        assert safe_eval("max(3, 1, 2)") == 3

    def test_combined_functions(self) -> None:
        assert safe_eval("sqrt(144) + 3 * 2") == 18.0

    def test_nested_functions(self) -> None:
        assert safe_eval("abs(floor(-3.7))") == 4


class TestSafeEvalConstants:
    """Named constants."""

    def test_pi(self) -> None:
        assert safe_eval("pi") == pytest.approx(math.pi)

    def test_e(self) -> None:
        assert safe_eval("e") == pytest.approx(math.e)

    def test_tau(self) -> None:
        assert safe_eval("tau") == pytest.approx(math.tau)

    def test_pi_in_expression(self) -> None:
        assert safe_eval("2 * pi") == pytest.approx(2 * math.pi)


class TestSafeEvalComparisons:
    """Comparison operators (return 1/0)."""

    def test_greater_true(self) -> None:
        assert safe_eval("5 > 3") == 1

    def test_greater_false(self) -> None:
        assert safe_eval("3 > 5") == 0

    def test_less_than(self) -> None:
        assert safe_eval("3 < 5") == 1

    def test_equal(self) -> None:
        assert safe_eval("5 == 5") == 1

    def test_not_equal(self) -> None:
        assert safe_eval("5 == 3") == 0

    def test_greater_equal(self) -> None:
        assert safe_eval("5 >= 5") == 1

    def test_less_equal(self) -> None:
        assert safe_eval("3 <= 5") == 1


class TestSafeEvalSafety:
    """Verify dangerous constructs are rejected."""

    def test_rejects_import(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval("__import__('os').system('ls')")

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval("().__class__.__bases__")

    def test_rejects_string_literals(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("'hello'")

    def test_rejects_unknown_function(self) -> None:
        with pytest.raises(ValueError, match="Unknown function"):
            safe_eval("eval('1+1')")

    def test_rejects_huge_exponent(self) -> None:
        with pytest.raises(ValueError, match="Exponent too large"):
            safe_eval("2 ** 10000")

    def test_rejects_list_comprehension(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval("[x for x in range(10)]")

    def test_rejects_lambda(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval("lambda: 1")


# ------------------------------------------------------------------ #
#  CalculatorTool tests (via ToolSpec)                                #
# ------------------------------------------------------------------ #


class TestCalculatorTool:
    """Test CalculatorTool via execute()."""

    @pytest.fixture
    def tool(self) -> CalculatorTool:
        return CalculatorTool()

    def test_execute_basic(self, tool: CalculatorTool) -> None:
        result = tool.execute(expression="2 + 3")
        assert result["result"] == 5
        assert result["error"] is None

    def test_execute_returns_expression(self, tool: CalculatorTool) -> None:
        result = tool.execute(expression="sqrt(16)")
        assert result["expression"] == "sqrt(16)"

    def test_execute_error_returns_none_result(self, tool: CalculatorTool) -> None:
        result = tool.execute(expression="1 / 0")
        assert result["result"] is None
        assert result["error"] is not None
        assert "ZeroDivisionError" in result["error"]

    def test_execute_syntax_error(self, tool: CalculatorTool) -> None:
        result = tool.execute(expression="2 +* 3")
        assert result["result"] is None
        assert result["error"] is not None

    def test_schema_rejects_missing_expression(self) -> None:
        with pytest.raises(ValidationError):
            CalculatorArgs()  # type: ignore[call-arg]

    def test_is_toolspec(self) -> None:
        from workbench.tools.base import ToolSpec

        assert issubclass(CalculatorTool, ToolSpec)

    def test_name(self) -> None:
        assert CalculatorTool().name == "calculator"

    def test_repr(self) -> None:
        assert "CalculatorTool" in repr(CalculatorTool())
