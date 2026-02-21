"""
Dr. Aria Nexus — AI Research Scientist

An autonomous AI scientist with personality, an electronic lab notebook,
experiment tracking, and the ability to work continuously on novel
architecture discovery.
"""

from .persona import Aria, get_aria
from .notebook import LabNotebook, ExperimentEntry


def __getattr__(name):
    if name == "ExperimentRunner":
        from .runner import ExperimentRunner
        return ExperimentRunner
    raise AttributeError(name)
