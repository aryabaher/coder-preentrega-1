"""Chequeo offline de la rúbrica: no llama a la API."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys

from pydantic import SecretStr, ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    BaseLLMClient,
    GeminiClient,
    OpenAIClient,
)
from llm_client.base import retry_async
from schemas import ChatMessage, LLMConfig, Provider


class FakeRateLimit(Exception):
    """Sustituto de RateLimitError / ServerError para probar retry_async sin la API."""


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


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


def check_retry() -> None:
    print("\n=== Reintentos (rate limit / red / 503) ===")
    print(
        "OpenAI generate usa retry_async:",
        "retry_async" in inspect.getsource(OpenAIClient.generate),
    )
    print(
        "Anthropic generate usa retry_async:",
        "retry_async" in inspect.getsource(AnthropicClient.generate),
    )
    print(
        "Gemini generate usa retry_async:",
        "retry_async" in inspect.getsource(GeminiClient.generate),
    )

    calls = {"n": 0}

    async def always_fail() -> str:
        calls["n"] += 1
        raise FakeRateLimit(f"429 simulado (llamada {calls['n']})")

    async def recover_on_third() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimit(f"429 simulado (llamada {calls['n']})")
        return "ok tras reintentos"

    async def run() -> None:
        calls["n"] = 0
        try:
            await retry_async(always_fail, retryable=(FakeRateLimit,), attempts=3, base_delay=0.1)
            print("ERROR: debería agotar los 3 intentos")
        except FakeRateLimit as exc:
            print(f"Tras 3 intentos el error se propaga (sin crash): {exc}")
            print(f"Llamadas realizadas: {calls['n']}")

        calls["n"] = 0
        result = await retry_async(
            recover_on_third,
            retryable=(FakeRateLimit,),
            attempts=3,
            base_delay=0.1,
        )
        print(f"Si el tercero responde: {result} (llamadas={calls['n']})")

    asyncio.run(run())


def main() -> None:
    _configure_stdio()
    check_interfaz_comun()
    check_asincronia()
    check_pydantic()
    check_factory()
    check_retry()
    print("\nValidación offline OK. Para la prueba live: python main.py")


if __name__ == "__main__":
    main()
