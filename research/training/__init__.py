"""
Training Methodology Synthesis

Generate novel training methods — loss functions, optimizers,
curricula — from the same primitive vocabulary used for architectures.
"""

from .curriculum import CurriculumStrategy, synthesize_curriculum
from .loss_synthesis import SynthesizedLoss, synthesize_loss
from .optimizer_synthesis import (
    MuonOptimizer,
    SynthesizedOptimizer,
    build_optimizer,
    synthesize_optimizer,
)
from .training_program import TrainingProgram, synthesize_training_program

__all__ = [
    "CurriculumStrategy",
    "MuonOptimizer",
    "SynthesizedLoss",
    "SynthesizedOptimizer",
    "TrainingProgram",
    "build_optimizer",
    "synthesize_curriculum",
    "synthesize_loss",
    "synthesize_optimizer",
    "synthesize_training_program",
]
