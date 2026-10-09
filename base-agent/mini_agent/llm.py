from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


def _post_json(
        url: str,
        payload: dict,
        headers: dict | None = None,
        timeout: int = 60 * 2,  # sec
    ) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise LLMError(
            f"Request to {url} failed with status {e.code}: {error_body}"
        ) from e
    except Exception as e:
        raise LLMError(f"Request to {url} failed: {e}") from e


class OllamaLLM:
    """Взаимодействие с LLM через Ollama.
    
    Возможные `base_url` для взаимодействия с LLM:
    - http://localhost:11434
    - https://ollama.com
    """

    def __init__(
            self,
            model: str = "nemotron-3-nano:30b",
            base_url: str = "https://ollama.com",
            api_key: str | None = None,
            temperature: float = 0.0
        ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.auth = {'Authorization': f'Bearer {api_key}'} if api_key else {}

    def chat(
            self,
            messages: list[dict],
            tools: list[dict] | None = None
        ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        data = _post_json(f"{self.base_url}/api/chat", payload, self.auth)

        msg = data.get("message", {})
        calls = [
            ToolCall(id=f"call_{i}", name=tc["function"]["name"],
                     arguments=tc["function"].get("arguments") or {})
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls)


class OpenAILLM:
    """Взаимодействие с LLM через API OpenAI."""

    def __init__(
            self,
            model: str,
            base_url: str = "https://api.openai.com/v1",
            api_key: str | None = None,
            temperature: float = 0.0
        ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.auth = {'Authorization': f'Bearer {api_key}'} if api_key else {}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
        data = _post_json(f"{self.base_url}/chat/completions", payload, self.auth)

        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc["function"]["name"],
                arguments=args
            ))
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls)
