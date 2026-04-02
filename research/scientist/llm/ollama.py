"""Ollama LLM Backend — local inference via Ollama REST API."""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from .backend import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """LLM backend using a local Ollama instance."""

    name = "ollama"

    def __init__(self):
        from research.defaults import OLLAMA_BASE

        self.host = os.environ.get("OLLAMA_HOST", OLLAMA_BASE)
        self.model = os.environ.get("OLLAMA_MODEL", "llama3")
        self.keep_alive = int(os.environ.get("OLLAMA_KEEP_ALIVE", "300"))

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except (OSError, ValueError):
            return False

    def auto_discover_analyst_model(self) -> Optional[str]:
        """Find a suitable small analyst model (<4GB, non-coder)."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            if r.status_code != 200:
                return None

            data = r.json()
            models = data.get("models", [])

            # Candidates: size < 4GB, name doesn't contain 'coder'
            candidates = []
            for m in models:
                name = m.get("name", "")
                size_bytes = m.get("size", 0)
                size_gb = size_bytes / (1024**3)

                if size_gb < 4.0 and "coder" not in name.lower():
                    # Prefer gemma if available
                    if "gemma" in name.lower():
                        return name
                    candidates.append((name, size_gb))

            if candidates:
                # Sort by size and return smallest
                candidates.sort(key=lambda x: x[1])
                return candidates[0][0]

            return None
        except (OSError, ValueError):
            return None

    def unload_model(self) -> None:
        """Force-unload the current model from GPU memory.

        Sends a dummy request with keep_alive=0 to trigger immediate unload.
        Call this before GPU-intensive work (training, evaluation).
        """
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=10,
            )
            logger.info(f"Unloaded Ollama model {self.model} from GPU")
        except Exception as e:
            logger.debug(f"Ollama unload failed (may not be running): {e}")

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        r = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        return LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            tokens_used=data.get("eval_count", 0),
        )
