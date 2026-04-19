"""Dr. Aria Nexus — AI Research Scientist."""

from __future__ import annotations

__all__ = [
    "Aria",
    "get_aria",
    "LabNotebook",
    "ExperimentEntry",
    "ExperimentRunner",
]


def __getattr__(name):
    if name in {"Aria", "get_aria"}:
        from .persona import Aria, get_aria

        return Aria if name == "Aria" else get_aria
    if name in {"LabNotebook", "ExperimentEntry"}:
        from .notebook import ExperimentEntry, LabNotebook

        return LabNotebook if name == "LabNotebook" else ExperimentEntry
    if name == "ExperimentRunner":
        from .runner import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(name)
