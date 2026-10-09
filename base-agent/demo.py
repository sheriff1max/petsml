#!/usr/bin/env python3
"""
Интерактивное демо-приложение.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from mini_agent import Agent, OllamaLLM, ToolRegistry


load_dotenv()
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="https://ollama.com")
    ap.add_argument("--model", default="nemotron-3-nano:30b")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--api_key", default="")
    args = ap.parse_args()

    agent = Agent(
        llm=OllamaLLM(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key if args.api_key else OLLAMA_API_KEY,
        ),
        tools=ToolRegistry(),
        max_steps=args.max_steps,
        trace_path="traces/demo.jsonl"
    )

    print(f"base-agent | model={args.model} | 'exit' to quit")
    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() == 'exit':
            break
        if not prompt:
            continue

        r = agent.run(prompt)
        for s in r.steps:
            for tc in s.tool_calls:
                print(f"  [tool] {tc['name']}({tc['arguments']}) -> {tc['result'][:120]}")
        print(f"agent> {r.answer}  ({r.stopped_reason}, {r.n_tool_calls} tool calls)")


if __name__ == "__main__":
    main()
