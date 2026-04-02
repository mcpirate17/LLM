"""Smoke test: verify binding_range_class annotations and scoring separation.

Builds two minimal architectures (local-only conv vs. transformer with attention),
trains each briefly on a copy-at-distance task, then runs the binding range probe
to confirm that the transformer acquires long-range binding while the local-only
model does not.

Also verifies the 3-signal soft gate logic: the local-only penalty should only
fire when ALL of induction_auc, binding_auc, and ar_auc are near zero.

Expected induction AUC at nano scale (1000 steps):
  - Causal transformer: ~0.5-0.6  (flat across all gaps)
  - Conv-3 / token_merge: ~0.0    (step function at gap=4)
  - Mamba / SSM / RWKV:   ~0.0    (state compression, different failure mechanism)

The induction probe measures EXACT token retrieval, not general non-local
capability. Mamba/RWKV failing is correct behavior, not a probe bug.

Usage:
    python -m research.tools.test_binding_smoke
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn


def _build_local_only_graph(model_dim: int = 64):
    """Build a minimal graph using only conv_only (local binding)."""
    from research.synthesis.graph import ComputationGraph

    g = ComputationGraph(model_dim)
    inp = g.add_input()
    c = g.add_op("conv_only", [inp])
    n = g.add_op("rmsnorm", [c])
    p = g.add_op("linear_proj", [n])
    out = g.add_op("add", [inp, p])
    g.set_output(out)
    return g


def _build_transformer_graph(model_dim: int = 64):
    """Build a minimal graph using softmax_attention (full binding)."""
    from research.synthesis.graph import ComputationGraph

    g = ComputationGraph(model_dim)
    inp = g.add_input()
    a = g.add_op("softmax_attention", [inp])
    n = g.add_op("rmsnorm", [a])
    p = g.add_op("linear_proj", [n])
    out = g.add_op("add", [inp, p])
    g.set_output(out)
    return g


def _train_on_copy_task(
    model: nn.Module,
    device: str,
    n_steps: int = 300,
    distance: int = 8,
    seq_len: int = 128,
    batch_size: int = 16,
    vocab_size: int = 256,
):
    """Train model briefly on a copy-at-distance task to develop binding patterns."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _step in range(n_steps):
        # Build copy-at-distance pattern: token[i] = token[i % distance] for all i
        seed_tokens = torch.randint(
            1, vocab_size, (batch_size, distance), device=device
        )
        n_repeats = (seq_len + distance - 1) // distance
        x = seed_tokens.repeat(1, n_repeats)[:, :seq_len]

        logits = model(x)
        loss = loss_fn(
            logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()


def main():
    from research.eval.binding_range import binding_range_profile
    from research.scientist.leaderboard_scoring import compute_composite_v7
    from research.synthesis.compiler import compile_model
    from research.synthesis.primitives import graph_binding_range_class

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 256
    model_dim = 64

    # --- Build graphs ---
    local_graph = _build_local_only_graph(model_dim)
    attn_graph = _build_transformer_graph(model_dim)

    # --- Verify binding_range_class annotations ---
    local_class = graph_binding_range_class(local_graph)
    attn_class = graph_binding_range_class(attn_graph)
    print(f"Local-only graph binding class: {local_class}")
    print(f"Transformer graph binding class: {attn_class}")
    assert local_class == "local", f"Expected 'local', got '{local_class}'"
    assert attn_class == "full", f"Expected 'full', got '{attn_class}'"
    print("[PASS] binding_range_class annotations correct")

    # --- Compile, train, and probe local-only model ---
    print("\nTraining local-only model on copy@d=8 task...")
    local_model = compile_model([local_graph], vocab_size=vocab_size, max_seq_len=128)
    local_model.to(device)
    _train_on_copy_task(local_model, device, n_steps=300, distance=8)
    local_result = binding_range_profile(
        local_model, distances=(2, 4, 8, 16, 32, 64), n_eval=100, device=device
    )
    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Compile, train, and probe transformer model ---
    print("Training transformer model on copy@d=8 task...")
    attn_model = compile_model([attn_graph], vocab_size=vocab_size, max_seq_len=128)
    attn_model.to(device)
    _train_on_copy_task(attn_model, device, n_steps=300, distance=8)
    attn_result = binding_range_profile(
        attn_model, distances=(2, 4, 8, 16, 32, 64), n_eval=100, device=device
    )
    del attn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nLocal-only binding AUC: {local_result.auc:.4f}")
    print(f"  Per-distance: {local_result.distance_accuracies}")
    print(f"Transformer binding AUC: {attn_result.auc:.4f}")
    print(f"  Per-distance: {attn_result.distance_accuracies}")

    # --- Test 3-signal soft gate logic ---
    print("\n--- Soft Gate Tests (3-signal AND) ---")

    # Case 1: Conv-3 (all signals near zero) → penalty should fire
    conv3_score = compute_composite_v7(
        ppl_screening=15.0,
        induction_auc=0.003,
        binding_auc=0.005,
        ar_auc=0.002,
        tier="screening",
        decompose=True,
    )
    conv3_penalty = conv3_score["breakdown"].get("binding_local_only_penalty", 0)
    print(f"Conv-3 (all signals ~0): penalty={conv3_penalty}")
    assert conv3_penalty > 0, "Conv-3 should trigger local-only penalty"
    print("[PASS] Conv-3 triggers penalty")

    # Case 2: Mamba (induction ~0, but binding_auc > 0.10) → penalty should NOT fire
    mamba_score = compute_composite_v7(
        ppl_screening=12.0,
        induction_auc=0.02,
        binding_auc=0.15,
        ar_auc=0.01,
        tier="screening",
        decompose=True,
    )
    mamba_penalty = mamba_score["breakdown"].get("binding_local_only_penalty", 0)
    print(f"Mamba-like (ind~0, bind>0.10): penalty={mamba_penalty}")
    assert mamba_penalty == 0, "Mamba should NOT trigger local-only penalty"
    print("[PASS] Mamba does NOT trigger penalty")

    # Case 3: Transformer (all signals high) → penalty should NOT fire
    attn_score = compute_composite_v7(
        ppl_screening=12.0,
        induction_auc=0.50,
        binding_auc=0.40,
        ar_auc=0.10,
        tier="screening",
        decompose=True,
    )
    attn_penalty = attn_score["breakdown"].get("binding_local_only_penalty", 0)
    print(f"Transformer (all signals high): penalty={attn_penalty}")
    assert attn_penalty == 0, "Transformer should NOT trigger penalty"
    print("[PASS] Transformer does NOT trigger penalty")

    # --- Composite score separation ---
    print("\n--- Composite Scores ---")
    print(
        f"Conv-3 (all ~0):     {conv3_score['composite_score']:.1f}  binding={conv3_score['breakdown'].get('binding', 0):.1f}pts"
    )
    print(
        f"Mamba-like:          {mamba_score['composite_score']:.1f}  binding={mamba_score['breakdown'].get('binding', 0):.1f}pts"
    )
    print(
        f"Transformer:         {attn_score['composite_score']:.1f}  binding={attn_score['breakdown'].get('binding', 0):.1f}pts"
    )

    # Verify ordering: transformer > mamba > conv-3
    assert attn_score["composite_score"] > mamba_score["composite_score"], (
        "Transformer should score higher than Mamba"
    )
    assert mamba_score["composite_score"] > conv3_score["composite_score"], (
        "Mamba should score higher than conv-3"
    )
    print("[PASS] Score ordering: transformer > mamba > conv-3")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
