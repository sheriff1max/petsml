from __future__ import annotations

import ast
import operator
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": lambda x: x ** 0.5
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCS or node.keywords:
            raise ValueError(f"function not allowed: {node.func.id}")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])

    raise ValueError(f"expression not allowed: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Вычисление арифметического выражения.
    Допустимые операции: + - * / // % **, abs, round, min, max, sqrt."""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def read_file(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def http_get(url: str, max_chars: int = 4000) -> str:
    """Переход по URL и получение его тела в виде текста"""
    req = urllib.request.Request(url, headers={"User-Agent": "base-agent/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")[:max_chars]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


def _str_param(desc: str) -> dict:
    return {"type": "string", "description": desc}


DEFAULT_TOOLS: list[Tool] = [
    Tool(
        name="calculator",
        description="Evaluate an arithmetic expression, e.g. (23*17)+145 or sqrt(2). Returns the numeric result as a string.",
        parameters={
            "type": "object",
            "properties": {"expression": _str_param("Arithmetic expression to evaluate")},
            "required": ["expression"],
        },
        func=calculator,
    ),
    Tool(
        name="read_file",
        description="Read a text file and return its content.",
        parameters={
            "type": "object",
            "properties": {
                "path": _str_param("Path to the file"),
                "max_chars": {"type": "integer", "description": "Max characters to return"},
            },
            "required": ["path"],
        },
        func=read_file,
    ),
    Tool(
        name="http_get",
        description="Fetch a URL over HTTP(S) and return the response body as text.",
        parameters={
            "type": "object",
            "properties": {
                "url": _str_param("URL to fetch"),
                "max_chars": {"type": "integer", "description": "Max characters to return"},
            },
            "required": ["url"],
        },
        func=http_get,
    ),
]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools = {
            t.name: t
            for t in (tools if tools is not None else DEFAULT_TOOLS)
        }

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """Запуск инструмента.
        Ошибки возвращаются в виде строк, чтобы агент мог их обработать."""
        tool = self._tools.get(name)
        if tool is None:
            return f"[error] unknown tool: {name}"
        try:
            return str(tool.func(**arguments))
        except Exception as e:
            return f"[error] {type(e).__name__}: {e}"

    def __len__(self) -> int:
        return len(self._tools)
