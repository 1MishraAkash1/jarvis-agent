"""
Phase 1 tools.

Each function here is a "tool" the agent can choose to call. The Gemini SDK's
automatic function calling reads the function's name, its type hints, and its
docstring to figure out (a) what the tool does, (b) what arguments it needs,
and (c) how to describe it to the model — you never write a separate schema
by hand. This is why the docstring below isn't just a comment for humans;
the model reads it too, to decide when this tool is the right one to call.
"""

import ast
import datetime
import operator


def get_current_datetime() -> str:
    """Returns the current date and time.

    Use this whenever the user asks what day, date, or time it is right now.
    """
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")


# Only these operators are allowed in `calculate` — this is the safety
# boundary. We never call Python's built-in eval() on user input, because
# eval() would let a malicious or malformed prompt run arbitrary code
# (e.g. "__import__('os').system('rm -rf /')"). Restricting to a whitelist
# of arithmetic operators makes that class of attack impossible, no matter
# what string ends up in `expression`.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):  # a plain number
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Evaluates a basic arithmetic expression and returns the result.

    Use this for any maths the user asks for, e.g. "what's 42 * 17" or
    "what's 15% of 340". Supports +, -, *, /, ** and parentheses.

    Args:
        expression: A plain arithmetic expression as a string, e.g. "42 * 17".
    """
    try:
        parsed = ast.parse(expression, mode="eval").body
        result = _safe_eval(parsed)
        return str(result)
    except Exception as exc:  # noqa: BLE001 — deliberately broad, we always
        # want to hand a readable error back to the model rather than crash
        # the whole agent loop over one bad expression.
        return f"Could not evaluate '{expression}': {exc}"
