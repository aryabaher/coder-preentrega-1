"""Cliente Anthropic asíncrono (AsyncAnthropic)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from llm_client.base import (
    BASE_DELAY_SECONDS,
    MAX_RETRIES,
    BaseLLMClient,
    as_client_error,
    close_quietly,
    failed_response,
    retry_async,
)
from llm_client.errors import LLMClientError
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse, TokenUsage

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "claude-sonnet-4-5",
        timeout: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise LLMClientError("Falta ANTHROPIC_API_KEY.")
        self.default_model = default_model
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    def _split_messages(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
        """Anthropic recibe `system` aparte; el resto va en `messages`."""

        system_parts = [m.content for m in messages if m.role == "system"]
        rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, rest

    def _request_kwargs(self, messages: list[ChatMessage], cfg: ModelConfig) -> dict:
        model = cfg.model or self.default_model
        system, payload = self._split_messages(messages)
        kwargs: dict = {
            "model": model,
            "max_tokens": cfg.max_tokens,
            "messages": payload,
        }
        # Anthropic SDK 1.3+ quitó temperature/top_p de messages.create.
        # ModelConfig.temperature sigue validándose (0–2) y OpenAI lo usa;
        # los modelos actuales de Claude lo ignoran.
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        cfg = config or ModelConfig()
        model = cfg.model or self.default_model
        kwargs = self._request_kwargs(messages, cfg)

        async def _call():
            return await self._client.messages.create(**kwargs)

        try:
            response = await retry_async(_call, retryable=_RETRYABLE)
        except (AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError) as exc:
            return failed_response(self.provider, model, exc)
        except Exception as exc:  # noqa: BLE001 — el loop principal no debe caer
            return failed_response(self.provider, model, exc)

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return ModelResponse(
            content=text,
            model=response.model or model,
            provider=self.provider,
            finish_reason=str(response.stop_reason) if response.stop_reason else None,
            usage=TokenUsage(
                prompt_tokens=usage.input_tokens if usage else None,
                completion_tokens=usage.output_tokens if usage else None,
            ),
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or ModelConfig()
        kwargs = self._request_kwargs(messages, cfg)
        delay = BASE_DELAY_SECONDS
        last_exc: BaseException | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            started = False
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        started = True
                        yield text
                return
            except _RETRYABLE as exc:
                last_exc = exc
                if started or attempt == MAX_RETRIES:
                    raise as_client_error(exc) from exc
                logger.warning(
                    "Intento %s/%s falló (%s). Reintento en %.1fs.",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
            except LLMClientError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise as_client_error(exc) from exc

        if last_exc is not None:
            raise as_client_error(last_exc) from last_exc

    async def aclose(self) -> None:
        await close_quietly(self._client)
