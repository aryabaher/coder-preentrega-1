"""Cliente base asíncrono y reintentos ante fallos transitorios."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

from llm_client.errors import LLMClientError
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0


class BaseLLMClient(ABC):
    """Interfaz común: generación completa y streaming de tokens."""

    provider: str
    default_model: str

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Devuelve la respuesta completa. Nunca deja escapar excepciones de la API."""

    @abstractmethod
    def generate_stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generador asíncrono de fragmentos de texto (`async for` + `yield`)."""

    @abstractmethod
    async def aclose(self) -> None:
        """Cierra el HTTP client del SDK."""

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


async def close_quietly(resource: object) -> None:
    """Cierra un cliente/stream HTTP. Traga el RuntimeError de httpcore2 al aclose."""

    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except (RuntimeError, StopAsyncIteration):
        pass
    except Exception:
        logger.debug("Error al cerrar recurso HTTP", exc_info=True)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    retryable: tuple[type[BaseException], ...],
    attempts: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY_SECONDS,
) -> T:
    """Reintenta errores transitorios (rate limit / red) con backoff exponencial."""

    delay = base_delay
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except retryable as exc:
            last_exc = exc
            if attempt == attempts:
                break
            logger.warning(
                "Intento %s/%s falló (%s). Reintento en %.1fs.",
                attempt,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2
    assert last_exc is not None
    raise last_exc


def failed_response(provider: str, model: str, exc: BaseException) -> ModelResponse:
    """Convierte una excepción de SDK en una respuesta estructurada (sin crash)."""

    return ModelResponse(
        provider=provider,
        model=model,
        error=_describe_error(exc),
    )


def _describe_error(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc).strip() or name
    lowered = f"{name} {text}".lower()
    if "authentication" in lowered or "api key" in lowered or "unauthorized" in lowered:
        return f"API key inválida o ausente ({name}): {text}"
    if "rate" in lowered or "429" in lowered:
        return f"Límite de tasa agotado tras reintentos ({name}): {text}"
    if "timeout" in lowered or "connection" in lowered:
        return f"Error de red o timeout tras reintentos ({name}): {text}"
    return f"Error del proveedor ({name}): {text}"


def as_client_error(exc: BaseException) -> LLMClientError:
    message = _describe_error(exc)
    retryable = "rate" in message.lower() or "red" in message.lower() or "timeout" in message.lower()
    return LLMClientError(message, retryable=retryable)
