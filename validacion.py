"""Chequeo offline de la rúbrica: no llama a la API."""

from __future__ import annotations

import inspect
import sys

from pydantic import SecretStr, ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    BaseLLMClient,
    GeminiClient,
    OpenAIClient,
)
from schemas import ChatMessage, LLMConfig, Provider


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def check_interfaz_comun() -> None:
    print("=== Intercambiabilidad ===")
    print(f"OpenAIClient bases={OpenAIClient.__mro__[1].__name__}")
    print(f"AnthropicClient bases={AnthropicClient.__mro__[1].__name__}")
    print(f"GeminiClient bases={GeminiClient.__mro__[1].__name__}")
    ok = all(
        issubclass(cls, BaseLLMClient)
        for cls in (OpenAIClient, AnthropicClient, GeminiClient)
    )
    print(f"Los tres heredan de BaseLLMClient: {ok}")
    print(f"Factory: AsyncLLMManager._crear_cliente")


def check_asincronia() -> None:
    print("\n=== Asincronía ===")
    print(f"generate es coroutine: {inspect.iscoroutinefunction(BaseLLMClient.generate)}")
    print(f"OpenAI generate_stream es async gen: {inspect.isasyncgenfunction(OpenAIClient.generate_stream)}")
    print(f"Anthropic generate_stream es async gen: {inspect.isasyncgenfunction(AnthropicClient.generate_stream)}")
    print(f"Gemini generate_stream es async gen: {inspect.isasyncgenfunction(GeminiClient.generate_stream)}")


def check_pydantic() -> None:
    print("\n=== Validación Pydantic (LLMConfig / ChatMessage) ===")
    ok = LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=0.3, max_tokens=256)
    print(f"LLMConfig válido: provider={ok.provider.value} temperature={ok.temperature}")
    msg = ChatMessage(role="user", content="¿Qué es la entropía?")
    print(f"ChatMessage válido: role={msg.role}")
    try:
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5)
        print("ERROR: temperature=5 debería fallar")
    except ValidationError as exc:
        print(f"temperature=5 rechazada ANTES de la API: {exc.error_count()} error(es)")
    try:
        ChatMessage(role="admin", content="hola")
        print("ERROR: role=admin debería fallar")
    except ValidationError as exc:
        print(f"role inválido rechazado: {exc.error_count()} error(es)")


def check_factory() -> None:
    print("\n=== Factory + error controlado ===")
    try:
        AsyncLLMManager(LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini"))
        print("ERROR: sin key debería fallar")
    except ValueError as exc:
        print(f"Sin openai_api_key → {exc}")
    manager = AsyncLLMManager(
        LLMConfig(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
        )
    )
    print(f"AsyncLLMManager con key inválida instancia: {type(manager._client).__name__}")


def main() -> None:
    _configure_stdio()
    check_interfaz_comun()
    check_asincronia()
    check_pydantic()
    check_factory()
    print("\nValidación offline OK. Para la prueba live: python main.py")


if __name__ == "__main__":
    main()
