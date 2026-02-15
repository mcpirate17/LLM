"""
Abstract LLM Backend + Factory

Provides a pluggable interface for different LLM providers.
Configuration via environment variables; defaults to None (rule-based fallback).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    text: str
    model: str
    tokens_used: int = 0
    cached: bool = False


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is configured and reachable."""
        ...

    @abstractmethod
    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.7) -> LLMResponse:
        """Generate a completion from the LLM."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} available={self.is_available()}>"


def create_backend() -> Optional[LLMBackend]:
    """Factory: create an LLM backend from environment variables.

    Environment:
        ARIA_LLM_BACKEND: 'ollama', 'anthropic', or 'openai'

    Returns None if no backend is configured (rule-based fallback).
    """
    backend_name = os.environ.get("ARIA_LLM_BACKEND", "").lower().strip()

    if not backend_name:
        logger.info("No ARIA_LLM_BACKEND set — using rule-based fallback")
        return None

    if backend_name == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend()
    elif backend_name == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend()
    elif backend_name == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend()
    else:
        logger.warning(f"Unknown ARIA_LLM_BACKEND={backend_name!r}, using rule-based fallback")
        return None


def create_backend_from_config(
    backend_name: str,
    api_key: str = "",
    model: str = "",
    host: str = "",
) -> Optional[LLMBackend]:
    """Create an LLM backend from explicit configuration (no env vars).

    Args:
        backend_name: 'ollama', 'anthropic', or 'openai'
        api_key: API key for anthropic/openai
        model: Model name override
        host: Host URL override (ollama)

    Returns the backend, or None on failure.
    """
    backend_name = backend_name.lower().strip()

    try:
        if backend_name == "ollama":
            from .ollama import OllamaBackend
            b = OllamaBackend()
            if host:
                b.host = host
            if model:
                b.model = model
            return b
        elif backend_name == "anthropic":
            from .anthropic import AnthropicBackend
            b = AnthropicBackend()
            if api_key:
                b.api_key = api_key
                b._client = None  # force re-init with new key
            if model:
                b.model = model
            return b
        elif backend_name == "openai":
            from .openai_backend import OpenAIBackend
            b = OpenAIBackend()
            if api_key:
                b.api_key = api_key
                b._client = None
            if model:
                b.model = model
            return b
    except Exception as e:
        logger.warning(f"Failed to create {backend_name} backend: {e}")

    return None
