"""Structured digest dataclasses for knowledge distillation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class ConvergenceProfile:
    """Training curve convergence statistics for a category."""
    category: str  # fast_converge, slow_converge, plateau, divergent
    count: int = 0
    avg_final_loss: float = 0.0
    avg_convergence_speed: float = 0.0  # steps to reach 90% of improvement
    avg_variance: float = 0.0
    avg_monotonicity: float = 0.0  # fraction of steps where loss decreases
    s1_pass_rate: float = 0.0


@dataclass
class ArchitectureFamily:
    """Clustered architecture family based on op-set similarity."""
    family_id: int
    representative_ops: List[str] = field(default_factory=list)
    n_members: int = 0
    s1_rate: float = 0.0
    avg_novelty: float = 0.0
    avg_loss_ratio: float = 0.0
    example_fingerprints: List[str] = field(default_factory=list)


@dataclass
class ConfigEffect:
    """Spearman correlation between a config parameter and outcomes."""
    param_name: str
    target: str  # s1_count, best_loss_ratio
    rho: float = 0.0  # Spearman r
    p_value: float = 1.0
    direction: str = "neutral"  # positive, negative, neutral
    n_samples: int = 0


@dataclass
class OpSynergy:
    """Op pair co-occurrence lift in S1 survivors."""
    op_a: str
    op_b: str
    lift: float = 1.0  # observed / expected co-occurrence
    co_occurrences: int = 0
    label: str = "neutral"  # synergistic, anti_synergistic, neutral


@dataclass
class HypothesisOutcome:
    """Hypothesis closure status."""
    hypothesis: str
    experiment_id: str = ""
    outcome: str = "inconclusive"  # confirmed, refuted, inconclusive
    evidence: str = ""
    s1_count: int = 0


@dataclass
class EfficiencyProfile:
    """Per-family FLOP and parameter efficiency metrics."""
    family_id: int
    avg_flops_per_token: float = 0.0
    avg_params: float = 0.0
    loss_per_megaparam: float = 0.0
    pareto_optimal: bool = False


@dataclass
class ExperimentDigest:
    """Top-level knowledge digest container."""
    timestamp: float = 0.0
    cycle_number: int = 0
    n_experiments_analyzed: int = 0
    n_curves_analyzed: int = 0

    convergence_profiles: List[ConvergenceProfile] = field(default_factory=list)
    architecture_families: List[ArchitectureFamily] = field(default_factory=list)
    config_effects: List[ConfigEffect] = field(default_factory=list)
    op_synergies: List[OpSynergy] = field(default_factory=list)
    hypothesis_outcomes: List[HypothesisOutcome] = field(default_factory=list)
    efficiency_profiles: List[EfficiencyProfile] = field(default_factory=list)

    narrative: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentDigest:
        """Reconstruct from dict, handling nested dataclasses."""
        return cls(
            timestamp=d.get("timestamp", 0.0),
            cycle_number=d.get("cycle_number", 0),
            n_experiments_analyzed=d.get("n_experiments_analyzed", 0),
            n_curves_analyzed=d.get("n_curves_analyzed", 0),
            convergence_profiles=[
                ConvergenceProfile(**p)
                for p in d.get("convergence_profiles", [])
            ],
            architecture_families=[
                ArchitectureFamily(**f)
                for f in d.get("architecture_families", [])
            ],
            config_effects=[
                ConfigEffect(**e)
                for e in d.get("config_effects", [])
            ],
            op_synergies=[
                OpSynergy(**s)
                for s in d.get("op_synergies", [])
            ],
            hypothesis_outcomes=[
                HypothesisOutcome(**h)
                for h in d.get("hypothesis_outcomes", [])
            ],
            efficiency_profiles=[
                EfficiencyProfile(**e)
                for e in d.get("efficiency_profiles", [])
            ],
            narrative=d.get("narrative", ""),
            recommendations=d.get("recommendations", []),
        )

    def summary_stats(self) -> str:
        """One-line summary for logging."""
        n_families = len(self.architecture_families)
        n_sig_effects = sum(1 for e in self.config_effects if e.p_value < 0.05)
        n_synergies = sum(1 for s in self.op_synergies if s.label == "synergistic")
        n_anti = sum(1 for s in self.op_synergies if s.label == "anti_synergistic")
        return (
            f"Digest: {self.n_experiments_analyzed} experiments, "
            f"{self.n_curves_analyzed} curves, "
            f"{n_families} families, "
            f"{n_sig_effects} significant config effects, "
            f"{n_synergies} synergies, {n_anti} anti-synergies"
        )
