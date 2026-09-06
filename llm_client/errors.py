"""Errores controlados del cliente unificado."""

from __future__ import annotations


class LLMClientError(Exception):
    """Falla de red, autenticación o cuota convertida en error de aplicación."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
