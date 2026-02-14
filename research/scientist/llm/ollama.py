"""Ollama LLM Backend — local inference via Ollama REST API."""

from __future__ import annotations

import json
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
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.7) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
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
