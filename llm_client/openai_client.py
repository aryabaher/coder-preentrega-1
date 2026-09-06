"""Cliente OpenAI asíncrono (AsyncOpenAI)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from llm_client.base import BaseLLMClient, as_client_error, close_quietly, failed_response, retry_async
from llm_client.errors import LLMClientError
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse, TokenUsage

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise LLMClientError("Falta OPENAI_API_KEY.")
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        cfg = config or ModelConfig()
        model = cfg.model or self.default_model
        payload = [message.model_dump() for message in messages]

        async def _call():
            return await self._client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

        try:
            response = await retry_async(_call, retryable=_RETRYABLE)
        except (AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
            return failed_response(self.provider, model, exc)
        except Exception as exc:  # noqa: BLE001 — el loop principal no debe caer
            return failed_response(self.provider, model, exc)

        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model or model,
            provider=self.provider,
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            ),
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or ModelConfig()
        model = cfg.model or self.default_model
        payload = [message.model_dump() for message in messages]

        async def _open_stream():
            return await self._client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=True,
            )

        stream = None
        try:
            stream = await retry_async(_open_stream, retryable=_RETRYABLE)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except LLMClientError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise as_client_error(exc) from exc
        finally:
            # Sin este cierre el pool HTTP queda a medio athrow (httpcore2 + 3.12).
            if stream is not None:
                await close_quietly(stream)
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        await close_quietly(self._client)
