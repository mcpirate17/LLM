"""Anthropic (Claude) LLM Backend."""

from __future__ import annotations

import logging
import os

from .backend import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-latest"


class AnthropicBackend(LLMBackend):
    """LLM backend using the Anthropic Claude API."""

    name = "anthropic"

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        if "ANTHROPIC_MODEL" not in os.environ:
            logger.info("Using default Anthropic model alias: %s", self.model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.7) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        tokens = (response.usage.input_tokens + response.usage.output_tokens
                  if response.usage else 0)

        return LLMResponse(
            text=text,
            model=self.model,
            tokens_used=tokens,
        )
