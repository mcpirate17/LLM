"""
Behavioral Fingerprinting

Answers: "Is this genuinely novel, or just attention with extra steps?"

Computes behavioral fingerprints that characterize HOW a model processes
information, independent of specific weights:
- Token interaction patterns (attention-like? local? hierarchical?)
- Representation geometry (intrinsic dimensionality, isotropy)
- Input sensitivity (Jacobian spectrum)
- CKA similarity vs known architectures
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import aria_core

logger = logging.getLogger(__name__)
NOVELTY_REFERENCE_SCHEME_VERSION = "nv1"

_SENSITIVITY_SKIP_COUNTS: Dict[str, int] = {}
_SENSITIVITY_SKIP_LAST_LOG_TS: float = 0.0
_SENSITIVITY_SKIP_LOG_INTERVAL_S: float = 60.0


def _record_sensitivity_skip(reason: str) -> None:
    """Track skipped sensitivity probes and emit a rate-limited debug summary."""
    global _SENSITIVITY_SKIP_LAST_LOG_TS
    key = str(reason or "unknown")
    _SENSITIVITY_SKIP_COUNTS[key] = _SENSITIVITY_SKIP_COUNTS.get(key, 0) + 1

    now = time.monotonic()
    if (now - _SENSITIVITY_SKIP_LAST_LOG_TS) < _SENSITIVITY_SKIP_LOG_INTERVAL_S:
        return

    _SENSITIVITY_SKIP_LAST_LOG_TS = now
    total = sum(_SENSITIVITY_SKIP_COUNTS.values())
    breakdown = ", ".join(f"{name}={count}" for name, count in sorted(_SENSITIVITY_SKIP_COUNTS.items()))
    logger.debug("Sensitivity probes skipped (%d total): %s", total, breakdown)


def get_sensitivity_skip_stats(reset: bool = False) -> Dict[str, object]:
    """Return aggregated sensitivity-skip counters for diagnostics."""
    global _SENSITIVITY_SKIP_LAST_LOG_TS
    by_reason = dict(_SENSITIVITY_SKIP_COUNTS)
    payload = {
        "total": int(sum(by_reason.values())),
        "by_reason": by_reason,
        "log_interval_seconds": _SENSITIVITY_SKIP_LOG_INTERVAL_S,
        "last_log_monotonic": _SENSITIVITY_SKIP_LAST_LOG_TS,
    }
    if reset:
        _SENSITIVITY_SKIP_COUNTS.clear()
        _SENSITIVITY_SKIP_LAST_LOG_TS = 0.0
    return payload


@dataclass
class BehavioralFingerprint:
    """Characterizes how a model behaves, not what it computes."""
    # Token interaction pattern
    interaction_locality: float = 0.0  # 0=global, 1=purely local
    interaction_sparsity: float = 0.0  # 0=dense, 1=sparse attention
    interaction_symmetry: float = 0.0  # 0=asymmetric, 1=symmetric
    interaction_hierarchy: float = 0.0  # how hierarchical the interaction pattern is

    # Representation geometry
    intrinsic_dim: float = 0.0  # estimated intrinsic dimensionality
    isotropy: float = 0.0      # how uniformly directions are used (0=collapsed, 1=isotropic)
    rank_ratio: float = 0.0    # effective rank / full rank

    # Input sensitivity
    jacobian_spectral_norm: float = 0.0
    jacobian_effective_rank: float = 0.0
    sensitivity_uniformity: float = 0.0  # how uniformly sensitive to each input token

    # Similarity to known architectures (CKA)
    cka_vs_transformer: float = 0.0
    cka_vs_ssm: float = 0.0
    cka_vs_conv: float = 0.0

    # Hierarchy detection (Gromov delta-hyperbolicity)
    hierarchy_fitness: float = 0.0  # 0=flat/Euclidean, 1=very tree-like
    gromov_delta: float = 0.0      # raw Gromov 4-point delta

    # Overall novelty estimate
    novelty_score: float = 0.0

    # CKA provenance
    cka_source: str = "none"  # "artifact", "heuristic_fallback", "none"
    cka_artifact_version: Optional[str] = None
    cka_probe_protocol_hash: Optional[str] = None
    cka_reference_quality: Optional[str] = None
    similarity_path: Optional[str] = None
    novelty_reference_version: Optional[str] = None
    novelty_valid_for_promotion: bool = False
    novelty_validity_reason: str = "missing_reference"

    # Quality tracking: how many of the 4 sub-analyses succeeded
    analyses_succeeded: int = 0  # 0–4
    quality: str = "none"  # "full" (4/4), "partial" (1–3), "none" (0)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    def summary(self) -> str:
        lines = [
            f"Novelty Score: {self.novelty_score:.3f}",
            f"Interaction: locality={self.interaction_locality:.2f}, "
            f"sparsity={self.interaction_sparsity:.2f}, "
            f"hierarchy={self.interaction_hierarchy:.2f}",
            f"Geometry: intrinsic_dim={self.intrinsic_dim:.1f}, "
            f"isotropy={self.isotropy:.3f}, rank_ratio={self.rank_ratio:.3f}",
            f"Sensitivity: jacobian_rank={self.jacobian_effective_rank:.1f}, "
            f"uniformity={self.sensitivity_uniformity:.3f}",
            f"CKA similarity: transformer={self.cka_vs_transformer:.3f}, "
            f"ssm={self.cka_vs_ssm:.3f}, conv={self.cka_vs_conv:.3f}",
        ]
        return "\n".join(lines)


def compute_fingerprint(
    model: nn.Module,
    seq_len: int = 64,
    model_dim: int = 256,
    vocab_size: int = 32000,
    device: str = "cuda",
    n_probes: int = 32,
) -> BehavioralFingerprint:
    """Compute behavioral fingerprint for a model."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    fp = BehavioralFingerprint()

    n_succeeded = 0

    with torch.no_grad():
        # Generate probe inputs
        probe_ids = torch.randint(0, vocab_size, (n_probes, seq_len), device=dev)

        # Get intermediate representations
        reps = _get_representations(model, probe_ids, dev)

        if reps is not None and len(reps) > 0:
            # Token interaction pattern
            interaction = _analyze_interactions(model, probe_ids, dev, seq_len)
            fp.interaction_locality = interaction["locality"]
            fp.interaction_sparsity = interaction["sparsity"]
            fp.interaction_symmetry = interaction["symmetry"]
            fp.interaction_hierarchy = interaction["hierarchy"]
            if interaction.get("_succeeded"):
                n_succeeded += 1

            # Representation geometry
            geometry = _analyze_geometry(reps)
            fp.intrinsic_dim = geometry["intrinsic_dim"]
            fp.isotropy = geometry["isotropy"]
            fp.rank_ratio = geometry["rank_ratio"]
            if geometry.get("_succeeded"):
                n_succeeded += 1

            # Hierarchy detection (Gromov delta-hyperbolicity)
            try:
                from .hierarchy_probe import hierarchy_fitness as _hf
                hf_result = _hf(reps, max_tokens=100)
                fp.hierarchy_fitness = hf_result["hierarchy_fitness"]
                fp.gromov_delta = hf_result["gromov_delta"]
            except Exception:
                pass

        # Input sensitivity (Jacobian analysis)
        sensitivity = _analyze_sensitivity(model, dev, seq_len, vocab_size)
        fp.jacobian_spectral_norm = sensitivity["spectral_norm"]
        fp.jacobian_effective_rank = sensitivity["effective_rank"]
        fp.sensitivity_uniformity = sensitivity["uniformity"]
        if sensitivity.get("_succeeded"):
            n_succeeded += 1

        # CKA similarity to reference architectures
        # Try artifact-backed CKA first, fall back to heuristic
        from .cka_references import get_default_store
        store = get_default_store()
        ref_activations = store.get_references()
        cka_meta = store.get_metadata()

        cka = _compute_reference_cka(reps, ref_activations=ref_activations)
        fp.cka_vs_transformer = cka.get("transformer", 0.0)
        fp.cka_vs_ssm = cka.get("ssm", 0.0)
        fp.cka_vs_conv = cka.get("conv", 0.0)
        fp.cka_source = cka_meta.get("cka_source", "none")
        fp.cka_artifact_version = cka_meta.get("cka_artifact_version")
        fp.cka_probe_protocol_hash = cka_meta.get("cka_probe_protocol_hash")
        fp.cka_reference_quality = cka_meta.get("cka_reference_quality")
        fp.similarity_path = cka_meta.get("cka_similarity_path", "_compute_reference_cka")
        fp.novelty_reference_version = build_novelty_reference_version(
            fp.cka_source,
            fp.cka_artifact_version,
            fp.cka_probe_protocol_hash,
        )
        if fp.cka_source == "artifact":
            fp.novelty_valid_for_promotion = True
            fp.novelty_validity_reason = "artifact_reference"
        elif fp.cka_source == "heuristic_fallback":
            fp.novelty_valid_for_promotion = False
            fp.novelty_validity_reason = "heuristic_fallback_reference"
        else:
            fp.novelty_valid_for_promotion = False
            fp.novelty_validity_reason = "missing_reference"
        if cka.get("_succeeded"):
            n_succeeded += 1

        # Overall novelty: low similarity to all known architectures
        max_cka = max(fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv, 0.01)
        fp.novelty_score = 1.0 - max_cka

    # Record analysis quality
    fp.analyses_succeeded = n_succeeded
    if n_succeeded == 4:
        fp.quality = "full"
    elif n_succeeded > 0:
        fp.quality = "partial"
    else:
        fp.quality = "none"

    model.train()
    return fp


def compute_lightning_fingerprint(
    model: nn.Module,
    seq_len: int = 64,
    model_dim: int = 256,
    device: str = "cpu",
    n_probes: int = 8,
) -> BehavioralFingerprint:
    """
    Lightning-fast behavioral fingerprint for pre-experiment gating.
    Uses fixed-seed initialization and minimal probes on CPU to estimate novelty.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    fp = BehavioralFingerprint()
    
    with torch.no_grad():
        # 1. Minimal probe with fixed seed for reproducibility
        torch.manual_seed(42)
        probe_ids = torch.randint(0, 32000, (n_probes, seq_len), device=dev)
        
        # 2. Forward pass (Lightning reps)
        reps = _get_representations(model, probe_ids, dev)
        
        if reps is not None:
            # 3. CKA vs Reference (The critical novelty gate)
            from .cka_references import get_default_store
            store = get_default_store()
            ref_activations = store.get_references()
            
            # Move to CPU for native aria_core.linear_cka_f32
            reps_cpu = reps.cpu()
            
            cka = _compute_reference_cka(reps_cpu, ref_activations=ref_activations)
            fp.cka_vs_transformer = cka.get("transformer", 0.0)
            fp.cka_vs_ssm = cka.get("ssm", 0.0)
            fp.cka_vs_conv = cka.get("conv", 0.0)
            
            max_cka = max(fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv, 0.01)
            fp.novelty_score = 1.0 - max_cka
            fp.cka_source = "lightning_dry_run"
            fp.quality = "partial"
            fp.analyses_succeeded = 1

    return fp


def build_novelty_reference_version(
    cka_source: Optional[str],
    cka_artifact_version: Optional[str],
    cka_probe_protocol_hash: Optional[str],
) -> str:
    """Stable version id used to compare novelty across time."""
    source = str(cka_source or "none")
    art = str(cka_artifact_version or "none")
    probe = str(cka_probe_protocol_hash or "none")
    return f"{NOVELTY_REFERENCE_SCHEME_VERSION}:{source}:{art}:{probe}"


def _get_representations(model: nn.Module, input_ids: torch.Tensor,
                         dev: torch.device) -> Optional[torch.Tensor]:
    """Get output representations from a model."""
    try:
        logits = model(input_ids)
        return logits
    except Exception as e:
        logger.warning("Failed to get representations: %s", e)
        return None


def _analyze_interactions(
    model: nn.Module, input_ids: torch.Tensor,
    dev: torch.device, seq_len: int,
) -> Dict[str, float]:
    """Analyze token-to-token interaction patterns."""
    result = {"locality": 0.5, "sparsity": 0.5, "symmetry": 0.5, "hierarchy": 0.5,
              "_succeeded": False}

    try:
        B = input_ids.shape[0]
        # Compute per-token influence by masking
        # Use a single sample for efficiency
        ids = input_ids[:1]

        base_out = model(ids)  # (1, S, V)

        # Measure how much each position affects each other position
        # by perturbing one token at a time (sampling a few positions)
        n_positions = min(8, seq_len)
        positions = torch.linspace(0, seq_len - 1, n_positions).long()
        influence_matrix = torch.zeros(n_positions, seq_len, device=dev)

        for i, pos in enumerate(positions):
            perturbed = ids.clone()
            perturbed[0, pos] = (ids[0, pos] + 1) % 32000
            pert_out = model(perturbed)
            diff = (pert_out - base_out).abs().mean(dim=-1).squeeze(0)  # (S,)
            influence_matrix[i] = diff

        # Locality: how concentrated is influence around the perturbed position?
        for i, pos in enumerate(positions):
            distances = (torch.arange(seq_len, device=dev).float() - pos.float()).abs()
            weights = influence_matrix[i]
            if weights.sum() > 1e-8:
                mean_dist = (distances * weights).sum() / weights.sum()
                result["locality"] += (1.0 - mean_dist / seq_len)
        result["locality"] /= (n_positions + 1)

        # Sparsity: how peaked is the influence distribution?
        flat = influence_matrix.flatten()
        if flat.sum() > 1e-8:
            probs = flat / flat.sum()
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            max_entropy = math.log(flat.numel())
            result["sparsity"] = 1.0 - (entropy / max_entropy)

        # Symmetry: is influence_matrix roughly symmetric?
        n = min(n_positions, seq_len)
        if influence_matrix.shape[0] >= 2:
            # Compare influence of A on B vs B on A (approximate)
            upper = influence_matrix[:n, :n].triu(1)
            lower = influence_matrix[:n, :n].tril(-1).t()
            if upper.norm() > 1e-8:
                result["symmetry"] = 1.0 - (upper - lower).norm().item() / upper.norm().item()

        # Hierarchy: multi-scale structure in influence
        # Check if influence has different patterns at different scales
        if influence_matrix.shape[1] >= 4:
            fine = influence_matrix[:, ::1].var().item()
            coarse = F.avg_pool1d(influence_matrix.unsqueeze(0), 4).squeeze(0).var().item()
            if fine > 1e-10:
                result["hierarchy"] = min(1.0, coarse / fine)

        result["_succeeded"] = True

    except Exception as e:
        logger.warning("Interaction analysis failed: %s", e)

    return result


def _analyze_geometry(reps: torch.Tensor) -> Dict[str, float]:
    """Analyze the geometry of representation space."""
    result = {"intrinsic_dim": 0.0, "isotropy": 0.0, "rank_ratio": 0.0,
              "_succeeded": False}

    try:
        # Flatten to (N, D)
        flat = reps.reshape(-1, reps.shape[-1]).float()
        N, D = flat.shape

        if N < 2 or D < 2:
            return result

        # Center
        flat = flat - flat.mean(dim=0, keepdim=True)

        # SVD for spectral analysis (use subset for efficiency)
        n_samples = min(N, 500)
        idx = torch.randperm(N)[:n_samples]
        subset = flat[idx]

        try:
            U, S, V = torch.linalg.svd(subset, full_matrices=False)
        except Exception as e:
            logger.debug("SVD failed in geometry analysis: %s", e)
            return result

        S = S.clamp(min=1e-10)

        # Intrinsic dimensionality (participation ratio)
        S_norm = S / S.sum()
        result["intrinsic_dim"] = (1.0 / (S_norm ** 2).sum()).item()

        # Isotropy: how uniform are the singular values?
        # Perfect isotropy: all singular values equal
        result["isotropy"] = (S.min() / S.max()).item()

        # Effective rank
        S_log = S_norm * torch.log(S_norm)
        entropy = -S_log.sum().item()
        result["rank_ratio"] = math.exp(entropy) / len(S)

        result["_succeeded"] = True

    except Exception as e:
        logger.warning("Geometry analysis failed: %s", e)

    return result


def _analyze_sensitivity(
    model: nn.Module, dev: torch.device,
    seq_len: int, vocab_size: int,
) -> Dict[str, float]:
    """Analyze input sensitivity via approximate Jacobian."""
    result = {"spectral_norm": 0.0, "effective_rank": 0.0, "uniformity": 0.0,
              "_succeeded": False}

    try:
        model.eval()
        with torch.enable_grad():
            # Small batch for Jacobian estimation
            ids = torch.randint(0, vocab_size, (1, seq_len), device=dev)
            ids.requires_grad_(False)

            # Get embedding and make it require grad
            embed = model.embed(ids).detach().requires_grad_(True)

            # Forward through layers only (skip embed/head)
            x = embed
            if hasattr(model, 'pos_enc') and model.pos_enc is not None:
                x = model.pos_enc(x)
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    x = layer(x)
            elif hasattr(model, 'topology'):
                x = model.topology(x)

            if not x.requires_grad:
                _record_sensitivity_skip("output_no_grad")
                return result

            # Compute sensitivity per position
            sensitivities = []
            n_positions = max(1, min(4, seq_len))
            step = max(1, seq_len // n_positions)
            for pos in range(0, seq_len, step):
                if embed.grad is not None:
                    embed.grad.zero_()
                target = x[:, pos, :].sum()
                if not target.requires_grad:
                    _record_sensitivity_skip("target_no_grad")
                    continue
                target.backward(retain_graph=True)
                if embed.grad is not None:
                    grad = embed.grad.clone()
                    sensitivities.append(grad.norm(dim=-1).squeeze(0))  # (S,)

            if not sensitivities:
                _record_sensitivity_skip("no_sensitivity_grads")

            if sensitivities:
                sens_matrix = torch.stack(sensitivities)  # (n_pos, S)
                result["spectral_norm"] = sens_matrix.norm().item()

                # How uniform is sensitivity across positions?
                per_pos_sens = sens_matrix.sum(dim=0)  # (S,)
                if per_pos_sens.sum() > 1e-8:
                    probs = per_pos_sens / per_pos_sens.sum()
                    entropy = -(probs * (probs + 1e-10).log()).sum().item()
                    result["uniformity"] = entropy / math.log(seq_len)
                    result["effective_rank"] = math.exp(entropy)

            result["_succeeded"] = True

    except Exception as e:
        logger.warning("Sensitivity analysis failed: %s", e)

    return result


def _compute_reference_cka(
    reps: Optional[torch.Tensor],
    ref_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute CKA similarity to reference architecture behaviors.

    If ref_activations is provided (from artifact store), computes CKA
    against real pre-trained reference representations. Otherwise falls
    back to heuristic synthetic patterns.

    Args:
        reps: Candidate model representations.
        ref_activations: Optional dict mapping family name -> reference
            activation tensor from artifact store.
    """
    result = {"transformer": 0.0, "ssm": 0.0, "conv": 0.0, "_succeeded": False}

    if reps is None:
        return result

    try:
        # Compute self-similarity matrix of representations
        flat = reps[0].float() if reps.dim() > 2 else reps.float()
        S, D = flat.shape[-2], flat.shape[-1]
        if S < 4:
            return result

        norm = F.normalize(flat, dim=-1)
        sim = torch.mm(norm.reshape(-1, D), norm.reshape(-1, D).t())
        sim = sim[:S, :S]  # (S, S)

        if ref_activations is not None:
            # Artifact-backed CKA: compute against real reference activations
            for family in ("transformer", "ssm", "conv"):
                ref_tensor = ref_activations.get(family)
                if ref_tensor is None:
                    continue
                # Build reference self-similarity matrix, truncating/padding
                # to match candidate sequence length
                ref_flat = ref_tensor.float()
                rS = ref_flat.shape[-2]
                use_S = min(S, rS)
                ref_norm = F.normalize(ref_flat[..., :use_S, :], dim=-1)
                rD = ref_norm.shape[-1]
                ref_sim = torch.mm(
                    ref_norm.reshape(-1, rD), ref_norm.reshape(-1, rD).t()
                )
                ref_sim = ref_sim[:use_S, :use_S]
                result[family] = _linear_cka(
                    sim[:use_S, :use_S], ref_sim
                )
        else:
            # Heuristic fallback: synthetic reference patterns
            # CAVEAT: These are synthetic approximations, not empirical.
            positions = torch.arange(S, device=sim.device).float()
            dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

            # Transformer: soft attention-like (slow decay from diagonal)
            ref_transformer = torch.exp(-dist / (S * 0.3))
            # SSM: recurrent (lower triangular with exponential decay)
            ref_ssm = torch.exp(-dist / (S * 0.15)) * (dist >= 0).float()
            ref_ssm = ref_ssm.tril()
            # Conv: local (sharp banded)
            ref_conv = (dist <= 5).float()

            result["transformer"] = _linear_cka(sim, ref_transformer)
            result["ssm"] = _linear_cka(sim, ref_ssm)
            result["conv"] = _linear_cka(sim, ref_conv)

        result["_succeeded"] = True

    except Exception as e:
        logger.warning("CKA computation failed: %s", e)

    return result


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA similarity."""
    try:
        # Optimization: use native aria_core kernel if on CPU
        if X.device.type == "cpu" and Y.device.type == "cpu":
            return aria_core.linear_cka_f32(X.contiguous(), Y.contiguous())

        X = X - X.mean()
        Y = Y - Y.mean()
        hsic_xy = (X * Y).sum()
        hsic_xx = (X * X).sum()
        hsic_yy = (Y * Y).sum()
        denom = (hsic_xx * hsic_yy).clamp(min=1e-10).sqrt()
        return (hsic_xy / denom).clamp(0, 1).item()
    except Exception as e:
        logger.debug("CKA computation error: %s", e)
        return 0.0
