"""OpenAIClient: AsyncOpenAI + await, sin bloquear el event loop."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from llm_client.base import BaseLLMClient, close_quietly, retry_async
from schemas import ChatMessage, ModelResponse, Provider

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _payload(self, messages: list[ChatMessage]) -> list[dict]:
        return [m.model_dump() for m in messages]

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        payload = self._payload(messages)

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        try:
            response = await retry_async(_call, retryable=_RETRYABLE)
        except RateLimitError as e:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=f"Límite de cuota excedido: {e}",
            )
        except (APIConnectionError, APITimeoutError) as e:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=f"Error de conexión: {e}",
            )
        except APIError as e:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=f"Error de la API de OpenAI: {e}",
            )

        return ModelResponse(
            provider=Provider.OPENAI,
            model=self.model,
            content=response.choices[0].message.content or "",
        )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        payload = self._payload(messages)

        async def _open_stream():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

        try:
            stream = await retry_async(_open_stream, retryable=_RETRYABLE)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
            yield f"\n[Error durante el streaming: {e}]"

    async def aclose(self) -> None:
        await close_quietly(self._client)
