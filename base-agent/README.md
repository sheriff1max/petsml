# Базовые агент без зависимостей (локальный и API)

Минималистичный tool-calling агент, написанный с нуля без сторонних зависимостей. Запустить можно как на маленькой локальной модели через OllamaLLM / OpenAILLM, так и через API.

## Запуск

### LLM по API

В файл `.env` сохранить API_KEY под ключом `OLLAMA_API_KEY` (если LLM не локальная).

Запуск интерактивного демо-приложения (можно сохранить API_KEY в `.env` и явно не передавать - сам подгрузится):

```bash
python demo.py --model emotron-3-nano:30b --base_url https://ollama.com --max-steps 5
prompt> Read evals/fixtures/prices.csv and compute the total price. Reply with the number only.
```

Посчитать метрику (доля решённых) на заранее написанных задачах:

```bash
python evals/run_eval.py --backend ollama --base_url https://ollama.com --model nemotron-3-nano:30b
```

### Локальная LLM

Скачать `ollama`, есть не установлена:

```bash
## Linux
curl -fsSL https://ollama.com/install.sh | sh
## Windows
irm https://ollama.com/install.ps1 | iex
```

Скачать локальную [модель](https://ollama.com/search) из ollama:

```bash
ollama pull qwen3:4b
```

Запуск интерактивного демо-приложения:

```bash
python demo.py --model qwen3:4b --base_url http://localhost:11434 --max-steps 5
prompt> Read evals/fixtures/prices.csv and compute the total price. Reply with the number only.
```

Посчитать метрику (доля решённых) на заранее написанных задачах:

```bash
python evals/run_eval.py --backend ollama --base_url http://localhost:11434 --model qwen3:4b
```

## Юнит-тесты

Запуск юнит-тестов (модель не нужна, тк заглушки используются):

```bash
python -m unittest discover tests -v
```

## Структура

```
mini_agent/
  llm.py      # OllamaLLM / OpenAILLM
  tools.py    # набор инструментов
  agent.py    # цикл для агента
evals/
  tasks.json  # задачи с ответами
  run_eval.py # прогон задач
  fixtures/   # файлы для тестов инструментов
tests/        # unittest с FakeLLM
demo.py     # CLI-демо
```

## Тесты бля замера качества

Примеры задач:
- арифметика
- чтение JSON и CSV с файла
- задача "не вызывай инструменты" (проверка, что агент не злоупотребляет ими)
- восстановление после ошибки
- получение информации с сайта.

Отчёт сохраняется в `evals/results.json`.

Это позволяет сравнивать модели на своём железе, например:

```bash
python evals/run_eval.py --model qwen3:4b
python evals/run_eval.py --model llama3.2:3b
```

## Добавление нового инструмента

```python
from mini_agent import Tool, ToolRegistry

registry = ToolRegistry()
registry.add(Tool(
    name="weather",
    description="Current weather in a city.",
    parameters={"type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]},
    func=lambda city: fetch_weather(city),
))
agent = Agent(llm, registry)
```
