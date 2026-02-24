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


def _auto_discover_primary_model(probe) -> Optional[str]:
    """Find the best available Ollama model for primary LLM use.

    Preference order:
    1. Largest instruct/chat model (best reasoning)
    2. Largest general model
    Excludes tiny models (<2GB) that can't handle complex prompts.
    """
    try:
        import requests
        r = requests.get(f"{probe.host}/api/tags", timeout=3)
        if r.status_code != 200:
            return None

        models = r.json().get("models", [])
        if not models:
            return None

        # Score each model: prefer large instruct models
        scored = []
        for m in models:
            name = m.get("name", "")
            size_gb = m.get("size", 0) / (1024**3)
            if size_gb < 2.0:
                continue  # too small for primary reasoning
            # Prefer instruct/chat variants
            is_instruct = any(k in name.lower() for k in ("instruct", "chat"))
            # Prefer coder models (good at structured output)
            is_coder = "coder" in name.lower()
            score = size_gb + (5.0 if is_instruct else 0) + (3.0 if is_coder else 0)
            scored.append((name, score, size_gb))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    except Exception:
        return None


def create_backend(is_analyst: bool = False) -> Optional[LLMBackend]:
    """Factory: create an LLM backend from environment variables.

    Environment:
        ARIA_LLM_BACKEND: 'ollama', 'anthropic', or 'openai' (Primary)
        ARIA_ANALYST_BACKEND: Faster fallback for standard analysis (optional)
        ARIA_ANALYST_MODEL: Model override for analyst mode

    Returns None if no backend is configured (rule-based fallback).
    """
    # Auto-detection: if no explicit backend set, probe local Ollama
    if not os.environ.get("ARIA_LLM_BACKEND") and not (
        is_analyst and os.environ.get("ARIA_ANALYST_BACKEND")
    ):
        from .ollama import OllamaBackend
        probe = OllamaBackend()
        if probe.is_available():
            if is_analyst:
                discovered = probe.auto_discover_analyst_model()
                if discovered:
                    probe.model = discovered
                    probe.keep_alive = 0
                    logger.info(f"Auto-detected analyst model: {discovered} (Ollama)")
                    return probe
            else:
                # For primary LLM, pick the best available model
                discovered = _auto_discover_primary_model(probe)
                if discovered:
                    probe.model = discovered
                    logger.info(f"Auto-detected primary LLM: {discovered} (Ollama)")
                    return probe

    if is_analyst:
        backend_name = os.environ.get("ARIA_ANALYST_BACKEND", "").lower().strip()
        # If no specific analyst backend, fall back to primary
        if not backend_name:
            backend_name = os.environ.get("ARIA_LLM_BACKEND", "").lower().strip()
    else:
        backend_name = os.environ.get("ARIA_LLM_BACKEND", "").lower().strip()

    if not backend_name:
        if not is_analyst:
            logger.info("No ARIA_LLM_BACKEND set and no Ollama detected — using rule-based fallback")
        return None

    if backend_name == "ollama":
        from .ollama import OllamaBackend
        b = OllamaBackend()
        
        # Determine model
        model_override = os.environ.get("ARIA_ANALYST_MODEL") if is_analyst else os.environ.get("ARIA_LLM_MODEL")
        if model_override:
            b.model = model_override
        elif is_analyst:
            # If backend is 'ollama' but no model specified, try auto-discover
            discovered = b.auto_discover_analyst_model()
            if discovered:
                b.model = discovered
                logger.info(f"Analyst backend set to Ollama, auto-discovered model: {discovered}")
        
        if is_analyst:
            b.keep_alive = 0 # Immediate unload for analyst tasks
        return b
    elif backend_name == "anthropic":
        from .anthropic import AnthropicBackend
        b = AnthropicBackend()
        if is_analyst and os.environ.get("ARIA_ANALYST_MODEL"):
            b.model = os.environ.get("ARIA_ANALYST_MODEL")
        return b
    elif backend_name == "openai":
        from .openai_backend import OpenAIBackend
        b = OpenAIBackend()
        if is_analyst and os.environ.get("ARIA_ANALYST_MODEL"):
            b.model = os.environ.get("ARIA_ANALYST_MODEL")
        return b
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
