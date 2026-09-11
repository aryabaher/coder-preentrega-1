"""Script de validación: modo normal, streaming y key inválida."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from pydantic import SecretStr, ValidationError

from llm_client import AsyncLLMManager, ChatMessage, LLMConfig, Provider, config_from_env

QUESTION = "¿Qué es la entropía? Respondé en 2 líneas."


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("google.genai").setLevel(logging.ERROR)


def _pregunta() -> list[ChatMessage]:
    return [ChatMessage(role="user", content=QUESTION)]


def demo_pydantic() -> None:
    print("=== Validación Pydantic (antes de llamar a la API) ===")
    try:
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5)
        print("ERROR: temperature=5 debería fallar")
    except ValidationError as exc:
        print("Se detectó ANTES de llamar a la API:")
        print(exc)


async def _run_provider(label: str, config: LLMConfig) -> AsyncLLMManager | None:
    try:
        manager = AsyncLLMManager(config)
    except ValueError as exc:
        print(f"{label}: configuración incompleta ({exc})")
        return None

    print(f"\n=== Modo normal ({label}) ===")
    resultado = await manager.generate(_pregunta())
    if resultado.error:
        print(f"Error controlado: {resultado.error}")
    else:
        print(resultado.content)

    print(f"\n=== Modo streaming ({label}) ===")
    async for chunk in manager.generate_stream(_pregunta()):
        print(chunk, end="", flush=True)
    print()
    return manager


async def demo_resiliencia() -> None:
    print("\n=== Prueba de resiliencia (API key inválida a propósito) ===")
    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )
    manager_roto = AsyncLLMManager(config_rota)
    resultado = await manager_roto.generate(_pregunta())
    print("El programa siguió vivo: sí")
    print("Error capturado (sin crash):", resultado.error)
    await manager_roto.aclose()


async def run_demo(only: str | None) -> None:
    demo_pydantic()
    providers = (only,) if only else ("openai", "anthropic", "gemini")
    managers: list[AsyncLLMManager] = []
    for name in providers:
        config = config_from_env(name)
        manager = await _run_provider(name, config)
        if manager is not None:
            managers.append(manager)
    await demo_resiliencia()
    for manager in managers:
        await manager.aclose()


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Prueba del cliente LLM asíncrono.")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "gemini"),
        default=None,
        help="Corré un solo proveedor. Sin flag: los tres + key inválida.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.provider))


if __name__ == "__main__":
    main()
