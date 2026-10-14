"""
Реализация инструментов агента.
Для упрощения используются заглушки.
"""

import hashlib
from random import Random

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather in a city."""
    rnd = Random(hashlib.sha256(city.lower().encode()).hexdigest())
    condition = rnd.choice(["clear", "cloudy", "rain", "snow"])
    return f"{city}: {rnd.randint(5, 32)}°C, {condition}"


@tool
def get_attractions(city: str) -> str:
    """Get top tourist attractions in a city."""
    data = {
        "paris": "Eiffel Tower, Louvre, Montmartre",
        "rome": "Colosseum, Vatican, Trevi Fountain",
        "tokyo": "Shibuya, Senso-ji, Akihabara",
        "london": "Big Ben, British Museum, Tower Bridge",
    }
    return data.get(city.lower(), f"No data for {city}.")


TOOLS = [get_weather, get_attractions]
