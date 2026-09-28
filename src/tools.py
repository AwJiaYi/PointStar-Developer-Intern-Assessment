import ast
import logging
import operator
from typing import Union


LOGGER = logging.getLogger("pointstar_agent")
Number = Union[int, float]

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported mathematical operator.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero is not allowed.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(_evaluate(node.operand))

    raise ValueError("Invalid mathematical expression.")


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression.

    Args:
        expression: A basic arithmetic expression such as "14 - 5".

    Returns:
        The numeric result as text, or a clear calculator error message.
    """
    print(f"[Agent Tool Call] Calculator: {expression}")
    LOGGER.info("Calculator tool requested with expression=%r", expression)

    try:
        if not expression or len(expression) > 100:
            raise ValueError("Expression is empty or too long.")

        parsed = ast.parse(expression, mode="eval")
        result = _evaluate(parsed.body)

        if isinstance(result, float) and result.is_integer():
            return str(int(result))

        return str(result)

    except (ValueError, SyntaxError) as exc:
        LOGGER.warning("Calculator error: %s", exc)
        return f"Calculator error: {exc}"
