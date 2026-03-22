"""
Training Methodology Synthesis

Generate novel training methods — loss functions, optimizers,
curricula — from the same primitive vocabulary used for architectures.
"""

from .loss_synthesis import (
    synthesize_loss as synthesize_loss,
    SynthesizedLoss as SynthesizedLoss,
)
from .optimizer_synthesis import (
    synthesize_optimizer as synthesize_optimizer,
    SynthesizedOptimizer as SynthesizedOptimizer,
    build_optimizer as build_optimizer,
    MuonOptimizer as MuonOptimizer,
)
from .curriculum import (
    synthesize_curriculum as synthesize_curriculum,
    CurriculumStrategy as CurriculumStrategy,
)
from .training_program import (
    TrainingProgram as TrainingProgram,
    synthesize_training_program as synthesize_training_program,
)
