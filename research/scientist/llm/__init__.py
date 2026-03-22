"""
LLM Backend Module for Dr. Aria Nexus

Pluggable LLM backends for intelligent analysis, hypothesis generation,
and experiment summarization. Falls back to rule-based methods when
no backend is configured.
"""

from . import context, prompts
from .backend import LLMBackend, LLMResponse, create_backend, create_backend_from_config
from .decision import NextExperimentDecisionPlanner, NextExperimentPlannerConfig

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "create_backend",
    "create_backend_from_config",
    "NextExperimentDecisionPlanner",
    "NextExperimentPlannerConfig",
    "context",
    "prompts",
]
