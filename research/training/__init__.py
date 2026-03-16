"""
Training Methodology Synthesis

Generate novel training methods — loss functions, optimizers,
curricula — from the same primitive vocabulary used for architectures.
"""

from .loss_synthesis import synthesize_loss, SynthesizedLoss
from .optimizer_synthesis import synthesize_optimizer, SynthesizedOptimizer, build_optimizer, MuonOptimizer
from .curriculum import synthesize_curriculum, CurriculumStrategy
from .training_program import TrainingProgram, synthesize_training_program
