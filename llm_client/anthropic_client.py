"""AnthropicClient: SDK AsyncAnthropic (messages.create / messages.stream)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
    APIError as AnthropicAPIError,
    APITimeoutError as AnthropicTimeoutError,
    AsyncAnthropic,
    RateLimitError as AnthropicRateLimitError,
)

from llm_client.base import BASE_DELAY_SECONDS, MAX_RETRIES, BaseLLMClient, close_quietly, retry_async
from schemas import ChatMessage, ModelResponse, Provider

logger = logging.getLogger(__name__)

_RETRYABLE = (AnthropicRateLimitError, AnthropicConnectionError, AnthropicTimeoutError)


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _split_messages(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
        """Anthropic recibe el system prompt aparte; el resto va en `messages`."""

        system_parts = [message.content for message in messages if message.role == "system"]
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, payload

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        system, payload = self._split_messages(messages)

        async def _call():
            return await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=payload,
                **({"system": system} if system else {}),
            )

        try:
            try:
                response = await retry_async(_call, retryable=_RETRYABLE)
            except TypeError:
                async def _call_sin_temperature():
                    return await self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=payload,
                        **({"system": system} if system else {}),
                    )

                response = await retry_async(_call_sin_temperature, retryable=_RETRYABLE)
        except AnthropicRateLimitError as e:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=f"Límite de cuota excedido: {e}",
            )
        except (AnthropicConnectionError, AnthropicTimeoutError) as e:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=f"Error de conexión: {e}",
            )
        except AnthropicAPIError as e:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=f"Error de la API de Anthropic: {e}",
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            provider=Provider.ANTHROPIC,
            model=self.model,
            content=text,
        )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        system, payload = self._split_messages(messages)
        delay = BASE_DELAY_SECONDS
        use_temperature = True

        for attempt in range(1, MAX_RETRIES + 1):
            started = False
            try:
                if use_temperature:
                    stream_cm = self._client.messages.stream(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=payload,
                        **({"system": system} if system else {}),
                    )
                else:
                    stream_cm = self._client.messages.stream(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=payload,
                        **({"system": system} if system else {}),
                    )
                async with stream_cm as stream:
                    async for event in stream:
                        if event.type != "content_block_delta":
                            continue
                        if getattr(event.delta, "type", None) != "text_delta":
                            continue
                        text = getattr(event.delta, "text", None)
                        if text:
                            started = True
                            yield text
                return
            except TypeError:
                use_temperature = False
                continue
            except _RETRYABLE as exc:
                if started or attempt == MAX_RETRIES:
                    yield f"\n[Error durante el streaming: {exc}]"
                    return
                logger.warning(
                    "Intento %s/%s falló (%s). Reintento en %.1fs.",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
            except AnthropicAPIError as exc:
                yield f"\n[Error durante el streaming: {exc}]"
                return

    async def aclose(self) -> None:
        await close_quietly(self._client)
