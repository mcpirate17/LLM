"""
LLM Backend Module for Dr. Aria Nexus

Pluggable LLM backends for intelligent analysis, hypothesis generation,
and experiment summarization. Falls back to rule-based methods when
no backend is configured.
"""

from .backend import LLMBackend, LLMResponse, create_backend

__all__ = ["LLMBackend", "LLMResponse", "create_backend"]
