"""Anthropic (Claude) LLM Backend."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from .backend import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def _response_attr(response, name: str):
    if isinstance(response, Mapping):
        return response.get(name)
    return getattr(response, name, None)


def _extract_response_text(response) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    block_types: list[str] = []

    content = _response_attr(response, "content")
    if isinstance(content, str):
        if content.strip():
            text_parts.append(content)
        return "".join(text_parts).strip(), ["string"]

    if isinstance(content, Mapping):
        blocks = [content]
    else:
        blocks = content or []

    for block in blocks:
        block_type = _response_attr(block, "type")
        if block_type:
            block_types.append(str(block_type))

        # Native SDK objects: TextBlock, etc.
        block_text = _response_attr(block, "text")
        if isinstance(block_text, str) and block_text.strip():
            text_parts.append(block_text)

    if not text_parts:
        # Some wrappers expose text at the top level rather than as content
        # blocks. Keep this as a last resort so native content typing wins.
        for attr_name in ("text", "output_text"):
            top_level_text = _response_attr(response, attr_name)
            if isinstance(top_level_text, str) and top_level_text.strip():
                text_parts.append(top_level_text)
                block_types.append(attr_name)
                break

    return "".join(text_parts).strip(), block_types


def _usage_tokens(response) -> int:
    usage = _response_attr(response, "usage")
    if not usage:
        return 0
    input_tokens = _response_attr(usage, "input_tokens") or 0
    output_tokens = _response_attr(usage, "output_tokens") or 0
    return int(input_tokens) + int(output_tokens)


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

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        text, block_types = _extract_response_text(response)

        if not text:
            logger.debug(
                "Anthropic response had no text blocks; retrying once "
                "(content types: %s, stop_reason: %s)",
                ",".join(block_types) if block_types else "none",
                _response_attr(response, "stop_reason") or "unknown",
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = max(max_tokens, 128)
            response = client.messages.create(**retry_kwargs)
            text, block_types = _extract_response_text(response)

        if not text:
            logger.warning(
                "Anthropic response had no text blocks "
                "(content types: %s, stop_reason: %s, id: %s)",
                ",".join(block_types) if block_types else "none",
                _response_attr(response, "stop_reason") or "unknown",
                _response_attr(response, "id") or "unknown",
            )

        return LLMResponse(
            text=text,
            model=self.model,
            tokens_used=_usage_tokens(response),
        )
