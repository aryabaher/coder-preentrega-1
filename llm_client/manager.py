"""Punto de composición: elige OpenAI o Anthropic según configuración."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv

from llm_client.anthropic_client import AnthropicClient
from llm_client.base import BaseLLMClient
from llm_client.errors import LLMClientError
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse

_ROOT = Path(__file__).resolve().parent.parent
_SUPPORTED = ("openai", "anthropic")


def _load_env() -> None:
    load_dotenv(_ROOT / ".env")
    load_dotenv()


def create_client(provider: str | None = None) -> BaseLLMClient:
    """Instancia el proveedor pedido (o el de `LLM_PROVIDER`) bajo la interfaz común."""

    _load_env()
    name = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if name not in _SUPPORTED:
        raise LLMClientError(f"Proveedor no soportado: {name!r}. Usá openai o anthropic.")

    timeout = float(os.getenv("LLM_TIMEOUT", "30"))
    if name == "openai":
        return OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            default_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout=timeout,
        )
    return AnthropicClient(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        default_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        timeout=timeout,
    )


class AsyncLLMManager:
    """Fachada asíncrona: carga el cliente según `LLM_PROVIDER` y expone generate/stream."""

    def __init__(
        self,
        provider: str | None = None,
        config: ModelConfig | None = None,
        client: BaseLLMClient | None = None,
    ) -> None:
        self._config = config or ModelConfig()
        self._client = client or create_client(provider)

    @property
    def provider(self) -> str:
        return self._client.provider

    @property
    def client(self) -> BaseLLMClient:
        return self._client

    async def generate(
        self,
        messages: list[ChatMessage] | str,
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        return await self._client.generate(_as_messages(messages), config or self._config)

    async def generate_stream(
        self,
        messages: list[ChatMessage] | str,
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        async for token in self._client.generate_stream(
            _as_messages(messages),
            config or self._config,
        ):
            yield token

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        await self._client.aclose()

    async def __aenter__(self) -> AsyncLLMManager:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _as_messages(messages: list[ChatMessage] | str) -> list[ChatMessage]:
    if isinstance(messages, str):
        return [ChatMessage(role="user", content=messages)]
    return messages
