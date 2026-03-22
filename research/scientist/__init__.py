"""
Dr. Aria Nexus — AI Research Scientist.
"""

from .notebook import LabNotebook, ExperimentEntry

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
    if name == "ExperimentRunner":
        from .runner import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(name)
