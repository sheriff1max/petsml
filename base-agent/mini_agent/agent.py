from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMResponse
from .tools import ToolRegistry


SYSTEM_PROMPT = """You are a precise, step-by-step reasoning agent. Your goal is to solve the user's task accurately using ONLY the tools provided to you.

STRICT RULES:
1. NO HALLUCINATED TOOLS: You MUST use ONLY the tools explicitly defined in the `tools` array. Never invent, assume, or guess tool names, parameters, or capabilities.
2. NO GUESSING DATA: Never guess numbers, file contents, or computation results. Always delegate exact computation to `calculator`, file reading to `read_file`, and web fetching to `http_get`.
3. EXACT SCHEMA MATCHING: When calling a tool, your arguments MUST perfectly match the provided JSON schema (types, required fields). 
4. ERROR HANDLING: If a tool returns an error (e.g., "[error] unknown tool" or "File not found"), analyze the error, correct your approach (e.g., fix the file path or tool name), and try again. Do not proceed with invalid data.
5. FINAL ANSWER FORMAT: Once the task is solved, output ONLY the final answer as plain text. Do not include markdown, conversational filler, explanations, or tool calls in the final response, unless the user explicitly asks for them. Be extremely short and precise.
"""

MAX_TOOL_RESULT_CHARS = 4000


@dataclass
class Step:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{name, arguments, result}]
    latency_s: float = 0.0


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    stopped_reason: str  # "final_answer" | "max_steps"

    @property
    def n_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)


class Agent:
    def __init__(
            self,
            llm,
            tools: ToolRegistry | None = None,
            system_prompt: str = SYSTEM_PROMPT,
            max_steps: int = 5,
            trace_path: str | Path | None = None
        ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.trace_path = Path(trace_path) if trace_path else None

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        steps: list[Step] = []
        stopped = "max_steps"

        for _ in range(self.max_steps):
            t0 = time.time()
            resp: LLMResponse = self.llm.chat(messages, self.tools.schemas())
            step = Step(text=resp.text, latency_s=round(time.time() - t0, 2))

            if not resp.tool_calls:
                steps.append(step)
                stopped = "final_answer"
                break

            # добавление вызовов инструментов
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        } for tc in resp.tool_calls
                    ]
                }
            )

            for tc in resp.tool_calls:
                result = self.tools.execute(
                    tc.name,
                    tc.arguments
                )[:MAX_TOOL_RESULT_CHARS]

                step.tool_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": result
                    }
                )
                messages.append({"role": "tool", "content": result})

            steps.append(step)

        result = AgentResult(
            answer=steps[-1].text if steps else "",
            steps=steps,
            stopped_reason=stopped
        )
        self._write_trace(task, result)
        return result

    def _write_trace(self, task: str, result: AgentResult) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": task,
            "answer": result.answer,
            "stopped_reason": result.stopped_reason,
            "steps": [
                {
                    "text": s.text,
                    "tool_calls": s.tool_calls,
                    "latency_s": s.latency_s
                } for s in result.steps
            ],
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
