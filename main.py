"""Script de validación: modo normal y streaming contra el proveedor configurado."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from llm_client import AsyncLLMManager, ChatMessage, LLMClientError, ModelConfig

_ATHROW = "generator didn't stop after athrow"

QUESTION = "¿Qué es la entropía?"


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def _ignore_httpcore_aclose(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """httpcore2 en Python 3.12 a veces falla al aclose del stream SSE; no es un error de la API."""

    exc = context.get("exception")
    message = str(context.get("message", ""))
    asyncgen = str(context.get("asyncgen", ""))
    if isinstance(exc, RuntimeError) and _ATHROW in str(exc):
        return
    if "PoolByteStream" in message or "PoolByteStream" in asyncgen:
        return
    loop.default_exception_handler(context)


async def run_demo(provider: str | None) -> None:
    asyncio.get_running_loop().set_exception_handler(_ignore_httpcore_aclose)
    messages = [ChatMessage(role="user", content=QUESTION)]
    config = ModelConfig(temperature=0.3, max_tokens=256)

    try:
        manager = AsyncLLMManager(provider=provider, config=config)
    except LLMClientError as exc:
        print(f"Error controlado: {exc}")
        return

    async with manager:
        print(f"Proveedor: {manager.provider}")
        print(f"Pregunta: {QUESTION}\n")

        print("=== Modo normal ===")
        response = await manager.generate(messages)
        if response.ok:
            print(response.content)
            if response.usage:
                print(
                    f"\n[{response.model}] "
                    f"tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens}"
                )
        else:
            print(f"Error controlado: {response.error}")

        print("\n=== Modo streaming ===")
        try:
            async for token in manager.generate_stream(messages):
                print(token, end="", flush=True)
            print()
        except LLMClientError as exc:
            print(f"\nError controlado: {exc}")


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Prueba del cliente LLM asíncrono.")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default=None,
        help="Sobrescribe LLM_PROVIDER del .env",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.provider))


if __name__ == "__main__":
    main()
