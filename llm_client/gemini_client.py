"""GeminiClient: genai.Client(...).aio. El asistente se llama 'model', no 'assistant'."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from llm_client.base import BaseLLMClient, retry_async
from schemas import ChatMessage, ModelResponse, Provider

_RETRYABLE = (genai_errors.ServerError,)


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _convertir_mensajes(self, messages: list[ChatMessage]):
        """Gemini separa el system prompt del resto, y llama 'model' al rol del asistente."""

        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(role=rol_gemini, parts=[types.Part(text=m.content)]))
        return contents, system_instruction

    def _config(self, system_instruction: str | None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        contents, system_instruction = self._convertir_mensajes(messages)
        config = self._config(system_instruction)

        async def _call():
            return await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        try:
            response = await retry_async(_call, retryable=_RETRYABLE)
            return ModelResponse(provider=Provider.GEMINI, model=self.model, content=response.text or "")
        except Exception as e:
            return ModelResponse(
                provider=Provider.GEMINI,
                model=self.model,
                content="",
                error=f"Error de la API de Gemini: {e}",
            )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        contents, system_instruction = self._convertir_mensajes(messages)
        config = self._config(system_instruction)

        async def _open_stream():
            return await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

        try:
            stream = await retry_async(_open_stream, retryable=_RETRYABLE)
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[Error durante el streaming: {e}]"
