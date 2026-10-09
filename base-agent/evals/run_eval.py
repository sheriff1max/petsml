#!/usr/bin/env python3
"""Запуск тестов агента.
Агент решает задачи, детерминированный оценщик проверяет ответы.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from mini_agent import Agent, LLMError, OllamaLLM, OpenAILLM, ToolRegistry


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_llm(
        backend: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
    ):
    if backend == "ollama":
        return OllamaLLM(model=model, base_url=base_url, api_key=api_key)
    return OpenAILLM(model=model, base_url=base_url, api_key=api_key)


def extract_number(text: str) -> float | None:
    """Получение последнего числа в тексте."""
    matches = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if not matches:
        return None
    return float(matches[-1].replace(",", "."))


def grade(check: dict, answer: str) -> bool:
    if check["type"] == "number":
        got = extract_number(answer)
        tol = check.get("tol", 0.01)
        return got is not None and abs(got - check["value"]) <= tol * max(1.0, abs(check["value"]))
    if check["type"] == "contains":
        return check["value"].lower() in answer.lower()
    raise ValueError(f"unknown check type: {check['type']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai"], required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default="")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "evals" / "results.json"))
    args = ap.parse_args()

    tasks = json.loads((ROOT / "evals" / "tasks.json").read_text())["tasks"]

    llm = make_llm(args.backend, args.model)
    agent = Agent(
        llm,
        ToolRegistry(),
        max_steps=args.max_steps,
        trace_path=ROOT / "evals" / "traces.jsonl"
    )

    results, passed = [], 0
    print(f"backend={args.backend} model={args.model} tasks={len(tasks)}\n")

    for t in tasks:
        t0 = time.time()
        try:
            r = agent.run(t["task"])
        except LLMError as e:
            print(f"{t['id']} ERROR  {e}")
            return
        dt = time.time() - t0

        ok = grade(t["check"], r.answer)
        if t.get("expect_no_tools") and r.n_tool_calls > 0:
            ok = False
        if t.get("requires_network") and r.n_tool_calls and any(
                "[error]" in c["result"] for c in
                (tc for s in r.steps for tc in s.tool_calls)
            ):
            print(f"{t['id']} SKIP   (network unavailable)")
            continue

        passed += ok
        results.append(
            {
                "id": t["id"],
                "passed": ok,
                "answer": r.answer.strip()[:200],
                "tool_calls": r.n_tool_calls,
                "stopped": r.stopped_reason,
                "seconds": round(dt, 1),
            },
        )
        mark = "PASS" if ok else "FAIL"
        print(f"{t['id']} {mark}  {dt:5.1f}s  tools={r.n_tool_calls}  answer={r.answer.strip()[:60]!r}")

    total = len(results)
    score = passed / total if total else 0.0
    print(f"\nscore: {passed}/{total} ({score:.0%})")

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "backend": args.backend,
        "score": score,
        "results": results
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
