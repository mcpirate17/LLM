"""OpenAI LLM Backend."""

from __future__ import annotations

import logging
import os

from .backend import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """LLM backend using the OpenAI API."""

    name = "openai"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception as exc:
            logger.debug("Returning default due to error: %s", exc)
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return LLMResponse(
            text=text,
            model=self.model,
            tokens_used=tokens,
        )
