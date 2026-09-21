"""Factory AsyncLLMManager / create_client y contrato Pydantic (sin API)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from llm_client.errors import LLMClientError
from llm_client.manager import AsyncLLMManager, create_client
from llm_client.openai_client import OpenAIClient
from schemas import ChatMessage, LLMConfig, Provider


def test_build_model_creates_openai():
    mgr = AsyncLLMManager(
        LLMConfig(
            provider=Provider.OPENAI,
            model="gpt-test",
            openai_api_key=SecretStr("sk-test-fake"),
        )
    )
    assert mgr._client is not None
    assert isinstance(mgr._client, OpenAIClient)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="openai_api_key"):
        create_client("openai")


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bloop")
    with pytest.raises(LLMClientError, match="Proveedor no soportado"):
        create_client()


def test_factory_anthropic():
    mgr = AsyncLLMManager(
        LLMConfig(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4-5",
            anthropic_api_key=SecretStr("sk-ant-test"),
        )
    )
    assert type(mgr._client).__name__ == "AnthropicClient"


def test_temperature_fuera_de_rango() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5)


def test_role_invalido() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="admin", content="hola")
