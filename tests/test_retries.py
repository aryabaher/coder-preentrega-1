"""Reintentos: transitorio recupera, 401 no reintenta, 429/red quedan controlados."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from llm_client.base import retry_async
from llm_client.openai_client import OpenAIClient
from schemas import ChatMessage


def _http_error(status: int, url: str = "https://api.openai.com/v1/chat/completions"):
    request = httpx.Request("POST", url)
    return httpx.Response(status, request=request, json={"error": {"message": "mock"}})


@pytest.mark.asyncio
async def test_transient_retries_then_succeeds() -> None:
    calls = {"n": 0}

    async def recover() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("rate limit", response=_http_error(429), body=None)
        return "ok"

    assert await retry_async(recover, retryable=(RateLimitError,), attempts=3, base_delay=0.01) == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_non_transient_does_not_retry() -> None:
    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini", temperature=0.2, max_tokens=32)
    create = AsyncMock(
        side_effect=AuthenticationError("bad key", response=_http_error(401), body=None)
    )
    client._client.chat.completions.create = create
    result = await client.generate([ChatMessage(role="user", content="hola")])
    assert result.error
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_retry_async_agota() -> None:
    class FakeRateLimit(Exception):
        pass

    calls = {"n": 0}

    async def always_fail() -> str:
        calls["n"] += 1
        raise FakeRateLimit(f"429 simulado ({calls['n']})")

    with pytest.raises(FakeRateLimit):
        await retry_async(always_fail, retryable=(FakeRateLimit,), attempts=3, base_delay=0.01)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_openai_429_queda_en_model_response_error() -> None:
    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini", temperature=0.2, max_tokens=32)
    client._client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError("quota", response=_http_error(429), body=None)
    )
    result = await client.generate([ChatMessage(role="user", content="hola")])
    assert result.error
    assert "cuota" in result.error.lower() or "429" in result.error or "quota" in result.error.lower()
    assert result.content == ""


@pytest.mark.asyncio
async def test_openai_red_queda_en_error() -> None:
    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini", temperature=0.2, max_tokens=32)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    client._client.chat.completions.create = AsyncMock(
        side_effect=APIConnectionError(request=request, message="dns")
    )
    result = await client.generate([ChatMessage(role="user", content="hola")])
    assert result.error
    assert (
        "conexión" in result.error.lower()
        or "conexion" in result.error.lower()
        or "connection" in result.error.lower()
    )
