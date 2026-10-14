"""
CLI запуск.
Три сценария:
    python main.py chat                — диалог
    python main.py steps "вопрос"      — стриминг шагов агента
    python main.py extract "текст"     — структурированный ответ через pydantic
"""

import sys
from uuid import uuid4

from langchain.messages import HumanMessage
from agent import build_agent, TripRequest

import warnings
warnings.filterwarnings("ignore", message="The v3 streaming protocol")


CHAT_AGENT = build_agent()
EXTRACT_AGENT = build_agent(response_format=TripRequest)
THREAD_CONFIG = {"configurable": {"thread_id": "chat-1"}}  # Ветка диалога для общего thread_id


def chat() -> None:
    """Диалог с агентом."""

    print("Команды консъержа:\n/history — показать память\n/exit — выход")
    while True:
        prompt = input("\nprompt › ").strip()
        if not prompt:
            continue

        if prompt == "/exit":
            break
        if prompt == "/history":
            for m in CHAT_AGENT.get_state(THREAD_CONFIG).values["messages"]:
                print(f"\t{type(m).__name__}: {m.text[:70]!r}")
            continue

        stream = CHAT_AGENT.stream_events(
            {"messages": [HumanMessage(prompt)]},
            config=THREAD_CONFIG,
            version="v3"
        )
        print("Консьерж ›", end=" ", flush=True)
        for kind, item in stream.interleave("messages", "tool_calls"):

            if kind == "messages":  # токены LLM печатаем на лету
                for token in item.text:
                    print(token, end="", flush=True)

            else:  # вызов инструмента
                print(f"\n\t{item.tool_name}({item.input})")
                for _ in item.output_deltas:  # ждём завершения инструмента
                    pass
                result = item.output.content if item.output else f"error: {item.error}"
                print(f"\t{result}")
                print("Консьерж ›", end=" ", flush=True)
        print()


def steps(question: str) -> None:
    """Вывод после каждого шага агента.
    Видно весь цикл «модель -> инструменты -> модель» по узлам графа.
    """
    config = {"configurable": {"thread_id": f"steps-{uuid4()}"}}
    for chunk in CHAT_AGENT.stream(
        {"messages": [HumanMessage(question)]},
        config,
        stream_mode="updates"
    ):
        for node, update in chunk.items():
            msg = update["messages"][-1]
            calls = [tc["name"] for tc in getattr(msg, "tool_calls", [])]
            print(f"[{node:^5}] {type(msg).__name__:<12} tools={calls or '-'}  text={msg.text[:60]!r}")


def extract(text: str) -> None:
    """Вывод структурированного ответа с помощью Pydantic."""

    config = {"configurable": {"thread_id": f"extract-{uuid4()}"}}
    result = EXTRACT_AGENT.invoke({"messages": [HumanMessage(text)]}, config)
    print(result)
    trip: TripRequest = result["structured_response"]
    print("\tTripRequest: ", trip.model_dump_json(indent=2))


def main() -> None:
    cmd, *args = sys.argv[1:]
    arg = " ".join(args)

    if cmd in ("", "chat"):
        chat()
    elif cmd == "steps" and arg:
        steps(arg)
    elif cmd == "extract" and arg:
        extract(arg)


if __name__ == "__main__":
    main()
