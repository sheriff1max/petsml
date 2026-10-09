"""Unit-тесты. Сервер LLM не требуется — скрипт FakeLLM проверяет цикл агента.

Запуск: python -m unittest discover tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_agent import Agent, ToolRegistry
from mini_agent.llm import LLMResponse, ToolCall
from mini_agent.tools import calculator


class FakeLLM:
    """Возвращает заранее подготовленные ответы, по одному на вызов chat()."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def tool_call(name, **args) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[
            ToolCall(id=f"call_{name}", name=name, arguments=args)
        ]
    )


def text(msg: str) -> LLMResponse:
    return LLMResponse(text=msg)


class TestAgentLoop(unittest.TestCase):
    def test_direct_answer(self):
        agent = Agent(FakeLLM([text("Paris")]), ToolRegistry(), max_steps=4)

        r = agent.run("Capital of France?")
        self.assertEqual(r.answer, "Paris")
        self.assertEqual(r.stopped_reason, "final_answer")
        self.assertEqual(r.n_tool_calls, 0)

    def test_tool_then_answer(self):
        llm = FakeLLM([tool_call("calculator", expression="2+2"), text("4")])

        r = Agent(llm, ToolRegistry(), max_steps=4).run("What is 2+2?")
        self.assertEqual(r.answer, "4")
        self.assertEqual(r.n_tool_calls, 1)
        self.assertEqual(r.steps[0].tool_calls[0]["result"], "4")
        self.assertTrue(any(m["role"] == "tool" and m["content"] == "4"
                            for m in llm.calls[1]))

    def test_max_steps_guard(self):
        llm = FakeLLM([tool_call("calculator", expression="1+1")])
        r = Agent(llm, ToolRegistry(), max_steps=3).run("loop")
        self.assertEqual(r.stopped_reason, "max_steps")
        self.assertEqual(len(r.steps), 3)

    def test_tool_error_is_returned_not_raised(self):
        llm = FakeLLM([
            tool_call("calculator", expression="__import__('os')"),
            text("sorry")
        ])
        r = Agent(llm, ToolRegistry(), max_steps=4).run("hack")
        self.assertIn("[error]", r.steps[0].tool_calls[0]["result"])

    def test_unknown_tool(self):
        llm = FakeLLM([
            tool_call("nope", x=1),
            text("done")
        ])
        r = Agent(llm, ToolRegistry(), max_steps=4).run("x")
        self.assertIn("unknown tool", r.steps[0].tool_calls[0]["result"])


class TestCalculator(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculator("(23*17)+145"), "536")
        self.assertEqual(calculator("2**10"), "1024")
        self.assertEqual(calculator("sqrt(16)"), "4")
        self.assertEqual(calculator("-5+2"), "-3")
        self.assertEqual(calculator("10/4"), "2.5")

    def test_rejects_dangerous(self):
        for bad in ["__import__('os').system('ls')", "open('/etc/passwd').read()",
                    "[x for x in range(3)]", "lambda: 1"]:
            with self.assertRaises(ValueError):
                calculator(bad)

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            calculator("1/0")


class TestRegistry(unittest.TestCase):
    def test_schemas_have_required_fields(self):
        for s in ToolRegistry().schemas():
            self.assertIn("name", s)
            self.assertIn("description", s)
            self.assertEqual(s["parameters"]["type"], "object")


if __name__ == "__main__":
    unittest.main()
