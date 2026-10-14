"""
Сборка агента: LLM, tools, prompt, memory, structured output.
"""

import os
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from tools import TOOLS

from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = (
    "You are a travel concierge. Answer concisely, in the user's language. "
    "Use the tools for weather and attractions instead of guessing."
)


model_name = os.getenv("MODEL", "ollama:nemotron-3-nano:30b")
model = init_chat_model(
    model=model_name,
    temperature=0,
    **{"base_url": "https://ollama.com"} if "ollama" in model_name else {}
)

checkpointer = InMemorySaver()


class TripRequest(BaseModel):
    """A tourist's travel request."""

    city: str = Field(description="Destination city")
    days: int = Field(description="Number of trip days")
    interests: list[str] = Field(description="What the tourist is interested in")


def build_agent(response_format: type[BaseModel] | None = None):
    """Одна фабрика create_agent на все сценарии."""
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=response_format,
        checkpointer=checkpointer,
    )
