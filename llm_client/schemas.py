"""Esquemas Pydantic para mensajes, configuración del modelo y respuestas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Un turno de la conversación. Evita diccionarios anidados sin validar."""

    role: MessageRole
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de generación compartidos por todos los proveedores."""

    model: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    timeout: float = Field(default=30.0, gt=0)


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelResponse(BaseModel):
    """Respuesta uniforme. Si `error` tiene valor, la llamada no crasheó: falló controlada."""

    content: str = ""
    model: str = ""
    provider: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
