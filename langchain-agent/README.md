# Туристический langchain-агент

Реализация travel-консьержа с инструментами, памятью диалога, потоковым выводом и структурированным ответом.

Инструменты работают на локальных заглушках. Нужен только ключ LLM-провайдера (можно использовать любые, но сейчас по умолчанию Ollama).

## Запуск

```bash
pip install -r requirements.txt
cp .env.example .env        # вписать значения
python main.py chat
```

## Примеры запуска

### Диалог

В этом режиме агент помнит прошлые реплики:

```
$ python main.py chat
prompt › Какая погода в Париже и что там посмотреть?

Консьерж ›
  🔧 get_weather({'city': 'Paris'})
  ↳ Paris: 18°C, clear
  🔧 get_attractions({'city': 'Paris'})
  ↳ Eiffel Tower, Louvre, Montmartre
Консьерж › В Париже 18°C, ясно. Стоит увидеть Эйфелеву башню, Лувр и Монмартр.
prompt › /history        # что реально лежит в памяти треда
prompt › /exit
```

### Стриминг шагов

```
$ python main.py steps "What is the weather in Rome?"
[model] AIMessage    tools=['get_weather']  text=''
[tools] ToolMessage  tools=-  text='Rome: 27°C, sunny'
[model] AIMessage    tools=-  text='In Rome it is 27°C and sunny.'
```

### Структурированный выход Pydantic

```
$ python main.py extract "Хочу на 3 дня в Токио, интересуют аниме и еда"
📋 TripRequest:
{
  "city": "Tokyo",
  "days": 3,
  "interests": ["anime", "food"]
}
```
