"""
Calculator tool: safe AST-based arithmetic evaluation.

Evaluates mathematical expressions using Python's ``ast`` module
for safe parsing — **no** ``eval()`` or ``exec()`` is used.

Supported operations:

- Basic arithmetic: ``+``, ``-``, ``*``, ``/``, ``//``, ``**``, ``%``
- Unary: ``-x``, ``+x``
- Math functions: ``sqrt``, ``log``, ``log10``, ``sin``, ``cos``,
  ``tan``, ``abs``, ``round``, ``ceil``, ``floor``, ``pi``, ``e``
- Comparisons (return 1/0): ``>``, ``<``, ``>=``, ``<=``, ``==``

All inputs and outputs are plain numbers (int or float).

Usage::

    tool = CalculatorTool()
    result = tool.execute(expression="sqrt(144) + 3 * 2")
    # result == {"expression": "sqrt(144) + 3 * 2", "result": 18.0, "error": None}
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

from workbench.observability.tracing import add_span_attributes
from workbench.tools.base import ToolSpec

# ------------------------------------------------------------------ #
#  Safe math evaluation via AST                                       #
# ------------------------------------------------------------------ #

# Allowed binary operators
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed comparison operators
_CMP_OPS: dict[type, Any] = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# Allowed function names → callables
_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "exp": math.exp,
    "pow": pow,
    "min": min,
    "max": max,
}

# Allowed named constants
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Safety limit for exponentiation
_MAX_EXPONENT = 1000


def _safe_eval_node(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node."""

    # Numeric literal
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # Named constant (pi, e, etc.)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    # Unary op: -x, +x
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        return op_fn(_safe_eval_node(node.operand))

    # Binary op: x + y, x ** y, etc.
    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Guard against huge exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
            if abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent too large: {right} (max {_MAX_EXPONENT})")
        return op_fn(left, right)

    # Function call: sqrt(x), log(x, base), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported.")
        func_name = node.func.id
        if func_name not in _FUNCTIONS:
            raise ValueError(
                f"Unknown function: {func_name!r}. Allowed: {sorted(_FUNCTIONS)}"
            )
        args = [_safe_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[func_name](*args)

    # Comparison: x > y  (returns 1 or 0)
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            cmp_fn = _CMP_OPS.get(type(op))
            if cmp_fn is None:
                raise ValueError(f"Unsupported comparison: {ast.dump(op)}")
            right = _safe_eval_node(comparator)
            if not cmp_fn(left, right):
                return 0
            left = right
        return 1

    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}. "
        "Only arithmetic, math functions, and comparisons are allowed."
    )


def safe_eval(expression: str) -> int | float:
    """
    Safely evaluate a mathematical expression string.

    Args:
        expression: A mathematical expression (e.g. ``"sqrt(144) + 3 * 2"``).

    Returns:
        The numeric result.

    Raises:
        ValueError: If the expression contains unsupported constructs.
        SyntaxError: If the expression is not valid Python syntax.
    """
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree.body)


# ------------------------------------------------------------------ #
#  Pydantic args schema                                               #
# ------------------------------------------------------------------ #


class CalculatorArgs(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(
        ...,
        description=(
            "A mathematical expression to evaluate. "
            "Supports: +, -, *, /, //, **, %, sqrt(), log(), sin(), cos(), "
            "tan(), abs(), round(), ceil(), floor(), pi, e."
        ),
    )


# ------------------------------------------------------------------ #
#  Tool implementation                                                #
# ------------------------------------------------------------------ #


class CalculatorTool(ToolSpec):
    """
    Safe arithmetic calculator based on AST evaluation.

    No ``eval()`` or ``exec()`` is used — expressions are parsed
    into an AST and evaluated node-by-node with an explicit allow-list.
    """

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression safely. "
        "Supports arithmetic, sqrt, log, trig, abs, round, ceil, floor, pi, e."
    )
    args_schema = CalculatorArgs

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Evaluate the expression.

        Args (validated by CalculatorArgs):
            expression: The math expression string.

        Returns:
            Dict with keys: expression, result, error.
        """
        expression: str = kwargs["expression"]

        try:
            result = safe_eval(expression)

            add_span_attributes(
                {
                    "expression": expression[:200],
                    "result": str(result)[:50],
                }
            )

            return {
                "expression": expression,
                "result": result,
                "error": None,
            }

        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
            return {
                "expression": expression,
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def __repr__(self) -> str:
        return "CalculatorTool()"
