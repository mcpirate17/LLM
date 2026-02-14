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
