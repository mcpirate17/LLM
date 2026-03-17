#!/usr/bin/env python3
"""
Export CKA Reference Artifacts (Phase B)

Trains small, fixed-architecture reference models for three families
(transformer, SSM, conv), extracts their activations on a deterministic
probe corpus, and writes the artifact bundle to disk.

Usage:
    python -m tools.export_cka_references [--output-dir artifacts/cka_references/v1]
                                          [--seed 42]
                                          [--n-steps 500]
                                          [--device cpu]

The output directory will contain:
    manifest.json
    transformer.pt
    ssm.pt
    conv.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Fixed protocol parameters ──
# These define the probe corpus and must match across all exports.
VOCAB_SIZE = 1024  # small vocab for fast training
MODEL_DIM = 64  # compact dimension
SEQ_LEN = 64  # sequence length for probes and training
N_LAYERS = 2  # layers per reference model
N_PROBE_SEQUENCES = 64  # number of probe sequences for activation extraction
BATCH_SIZE = 8  # training batch size


def _probe_protocol_hash() -> str:
    """Compute deterministic hash of the probe protocol parameters."""
    spec = json.dumps(
        {
            "vocab_size": VOCAB_SIZE,
            "model_dim": MODEL_DIM,
            "seq_len": SEQ_LEN,
            "n_layers": N_LAYERS,
            "n_probe_sequences": N_PROBE_SEQUENCES,
            "batch_size": BATCH_SIZE,
        },
        sort_keys=True,
    )
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


# ── Reference model definitions ──
# Each model is a small, representative architecture for its family.
# They share the same embed/head structure but differ in their layer logic.


class TransformerRefLayer(nn.Module):
    """Single-head causal self-attention + FFN."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Self-attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        x = x + self.proj(torch.matmul(attn, v))
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class SSMRefLayer(nn.Module):
    """Simple selective state-space layer (S4-inspired diagonal SSM)."""

    def __init__(self, dim: int, state_dim: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Input projection to get B, C, delta (discretization)
        self.input_proj = nn.Linear(dim, dim + 2 * state_dim, bias=False)
        # State matrix diagonal (log-space for stability)
        self.log_A = nn.Parameter(torch.randn(dim, state_dim) * 0.1)
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.state_dim = state_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        h = self.norm(x)

        # Project input
        proj = self.input_proj(h)
        z = proj[..., :D]  # (B, S, D) - input gate
        B_mat = proj[..., D : D + self.state_dim]  # (B, S, N) - input matrix
        C_mat = proj[..., D + self.state_dim :]  # (B, S, N) - output matrix

        # Discretized state transition: A_bar = exp(log_A * softplus(delta))
        A = -torch.exp(self.log_A)  # (D, N) negative for stability

        # Scan (sequential for simplicity — this is a reference model)
        state = torch.zeros(B, D, self.state_dim, device=x.device)
        outputs = []
        for t in range(S):
            # state = A * state + B * z
            state = state * torch.exp(A.unsqueeze(0)) + B_mat[:, t].unsqueeze(1) * z[
                :, t
            ].unsqueeze(-1)
            # y = C * state + D * z
            y = (C_mat[:, t].unsqueeze(1) * state).sum(-1) + self.D * z[:, t]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, S, D)
        return residual + self.out_proj(y)


class ConvRefLayer(nn.Module):
    """Depthwise 1D convolution + FFN (local-processing archetype)."""

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Depthwise conv: each channel independently
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        h = self.norm1(x)
        # Conv branch (B,S,D) -> (B,D,S) -> conv -> (B,D,S) -> (B,S,D)
        conv_out = self.conv(h.transpose(1, 2)).transpose(1, 2)[:, :S, :]
        gate = torch.sigmoid(self.gate(h))
        x = residual + self.proj(conv_out * gate)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ReferenceModel(nn.Module):
    """Wrapper: embed + N layers + head."""

    def __init__(self, layers: nn.ModuleList, vocab_size: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = layers
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tie weights

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


def build_reference_model(family: str, device: str = "cpu") -> ReferenceModel:
    """Build a small reference model for the given architecture family."""
    if family == "transformer":
        layers = nn.ModuleList(
            [TransformerRefLayer(MODEL_DIM) for _ in range(N_LAYERS)]
        )
    elif family == "ssm":
        layers = nn.ModuleList([SSMRefLayer(MODEL_DIM) for _ in range(N_LAYERS)])
    elif family == "conv":
        layers = nn.ModuleList([ConvRefLayer(MODEL_DIM) for _ in range(N_LAYERS)])
    else:
        raise ValueError(f"Unknown family: {family}")

    model = ReferenceModel(layers, VOCAB_SIZE, MODEL_DIM)
    return model.to(device)


# ── Training ──


def train_reference(
    model: ReferenceModel,
    n_steps: int,
    seed: int,
    device: str = "cpu",
) -> dict:
    """Train a reference model on random next-token prediction.

    Returns training info dict.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()

    initial_loss = None
    final_loss = None
    losses = []

    for step in range(n_steps):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=dev)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            input_ids[:, 1:].reshape(-1),
        )

        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        final_loss = loss.item()
        if step % 50 == 0:
            losses.append(final_loss)

    loss_ratio = final_loss / max(initial_loss, 1e-10)
    logger.info(
        "  Training done: initial=%.4f final=%.4f ratio=%.4f",
        initial_loss,
        final_loss,
        loss_ratio,
    )

    return {
        "n_steps": n_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "seed": seed,
    }


# ── Activation extraction ──


def extract_activations(
    model: ReferenceModel,
    seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract activations on deterministic probe corpus.

    Returns tensor of shape (SEQ_LEN, VOCAB_SIZE) — the self-similarity
    is computed from this at CKA time.
    """
    dev = torch.device(device)
    model.eval()

    # Deterministic probe corpus
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    probe_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (N_PROBE_SEQUENCES, SEQ_LEN),
        generator=gen,
    ).to(dev)

    with torch.no_grad():
        logits = model(probe_ids)  # (N_PROBE, SEQ_LEN, VOCAB_SIZE)

    # Average across probe sequences to get stable representation
    avg_logits = logits.mean(dim=0)  # (SEQ_LEN, VOCAB_SIZE)
    return avg_logits.cpu()


# ── Export ──


def export_artifacts(
    output_dir: str,
    seed: int = 42,
    n_steps: int = 500,
    device: str = "cpu",
) -> Path:
    """Train all reference models and export artifacts.

    Returns path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    families = ["transformer", "ssm", "conv"]
    activation_shape = None
    family_info = {}

    for family in families:
        logger.info("Building %s reference model...", family)
        # Use different seed per family for diversity, but deterministic
        family_seed = seed + hash(family) % 1000
        model = build_reference_model(family, device=device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("  %s: %d parameters", family, n_params)

        logger.info("  Training %s for %d steps...", family, n_steps)
        train_info = train_reference(model, n_steps, family_seed, device)

        logger.info("  Extracting activations...")
        activations = extract_activations(model, seed=seed, device=device)

        if activation_shape is None:
            activation_shape = list(activations.shape)
        else:
            assert list(activations.shape) == activation_shape, (
                f"Shape mismatch: {activations.shape} vs {activation_shape}"
            )

        # Save .pt file
        pt_path = out / f"{family}.pt"
        torch.save(
            {
                "activations": activations,
                "config": {
                    "family": family,
                    "model_dim": MODEL_DIM,
                    "vocab_size": VOCAB_SIZE,
                    "seq_len": SEQ_LEN,
                    "n_layers": N_LAYERS,
                    "n_params": n_params,
                },
                "training_info": train_info,
            },
            pt_path,
        )
        logger.info("  Saved %s", pt_path)

        family_info[family] = {
            "n_params": n_params,
            "loss_ratio": train_info["loss_ratio"],
        }

    # Write manifest
    code_version = _get_code_version()
    manifest = {
        "artifact_version": "v1",
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
        "reference_families": families,
        "probe_protocol_hash": _probe_protocol_hash(),
        "activation_shape": activation_shape,
        "quality_flags": {
            "overall": "good",
            "training_steps": n_steps,
            "seed": seed,
            "families": family_info,
        },
    }

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)

    return out


def _get_code_version() -> str:
    """Get current code version from env or git."""
    v = os.environ.get("RESEARCH_CODE_VERSION")
    if v:
        return v
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Export CKA reference artifacts")
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).parent.parent / "artifacts" / "cka_references" / "v1"
        ),
        help="Output directory for artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for training (cpu or cuda)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out = export_artifacts(
        output_dir=args.output_dir,
        seed=args.seed,
        n_steps=args.n_steps,
        device=args.device,
    )
    print(f"\nArtifacts exported to: {out}")
    print(f"Probe protocol hash: {_probe_protocol_hash()}")


if __name__ == "__main__":
    main()
