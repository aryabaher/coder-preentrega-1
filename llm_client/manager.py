"""AsyncLLMManager (Factory Pattern): un campo de config, no instanciar SDKs en el negocio."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

from llm_client.anthropic_client import AnthropicClient
from llm_client.base import BaseLLMClient
from llm_client.gemini_client import GeminiClient
from llm_client.openai_client import OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-4-5",
    Provider.GEMINI: "gemini-flash-latest",
}


def _load_env() -> None:
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT.parent / ".env")
    load_dotenv()


def config_from_env(provider: str | Provider | None = None) -> LLMConfig:
    """Arma el LLMConfig leyendo OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY."""

    _load_env()
    name = (provider or os.getenv("LLM_PROVIDER", "openai"))
    if isinstance(name, Provider):
        chosen = name
    else:
        chosen = Provider(str(name).strip().lower())

    def _secret(name: str, *aliases: str) -> SecretStr | None:
        for key in (name, *aliases):
            value = (os.getenv(key) or "").strip()
            if value:
                return SecretStr(value)
        return None

    model = {
        Provider.OPENAI: os.getenv("OPENAI_MODEL", _DEFAULT_MODELS[Provider.OPENAI]),
        Provider.ANTHROPIC: os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODELS[Provider.ANTHROPIC]),
        Provider.GEMINI: os.getenv("GEMINI_MODEL", _DEFAULT_MODELS[Provider.GEMINI]),
    }[chosen]

    return LLMConfig(
        provider=chosen,
        model=model,
        openai_api_key=_secret("OPENAI_API_KEY"),
        anthropic_api_key=_secret("ANTHROPIC_API_KEY"),
        google_api_key=_secret("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )


class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: BaseLLMClient = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        provider = self.config.provider
        if provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("Falta openai_api_key en la configuración")
            return OpenAIClient(
                api_key=self.config.openai_api_key.get_secret_value(),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        if provider == "anthropic":
            if not self.config.anthropic_api_key:
                raise ValueError("Falta anthropic_api_key en la configuración")
            return AnthropicClient(
                api_key=self.config.anthropic_api_key.get_secret_value(),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        if provider == "gemini":
            if not self.config.google_api_key:
                raise ValueError("Falta google_api_key en la configuración")
            return GeminiClient(
                api_key=self.config.google_api_key.get_secret_value(),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        raise ValueError(f"Proveedor no soportado: {provider}")

    @property
    def provider(self) -> Provider:
        return self.config.provider

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        async for chunk in self._client.generate_stream(messages):
            yield chunk

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncLLMManager:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def create_client(provider: str | None = None) -> BaseLLMClient:
    """Atajo: instancia el cliente del proveedor activo vía el factory."""

    return AsyncLLMManager(config_from_env(provider))._client
