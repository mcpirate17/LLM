"""
Grammar Pass-Rate Validation Test

Validates that the motif-based compositional grammar achieves the
target pass rates from JUDGMENT_ENGINE_PLAN.md:
  - ≥70% smoke test pass rate over 100 seeds
  - ≥40% compile + train without crash/NaN
  - ≥90 unique fingerprints (no mode collapse)

Run manually: pytest research/tests/test_grammar_pass_rate.py -v --timeout=300
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.unit]

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from research.env import aria_core, HAS_ARIA_CORE


N_SEEDS = 100
SMOKE_PASS_TARGET = 0.70
TRAIN_PASS_TARGET = 0.40
UNIQUE_FP_TARGET = 90

# Tiny dimensions for speed
D_MODEL = 64
SEQ_LEN = 16
BATCH_SIZE = 2
VOCAB_SIZE = 256
TRAIN_STEPS = 10
PER_GRAPH_TIMEOUT = 30.0


def _smoke_test_structural(graph) -> bool:
    """Python structural smoke test — checks gradient path and param presence."""
    from research.synthesis.op_roles import get_role, OpRole

    has_params = False
    has_unsafe_standalone = False
    for node in graph.nodes.values():
        role = get_role(node.op_name)
        if role in (OpRole.PROJECT, OpRole.MIX, OpRole.GATE):
            has_params = True
        if role == OpRole.UNSAFE:
            has_unsafe_standalone = True

    if not has_params:
        return False
    if has_unsafe_standalone:
        return False
    return True


def _smoke_test_native(graph) -> bool:
    """C++ smoke test via aria_core if available."""
    import json
    role_map = {}
    try:
        from research.synthesis.op_roles import get_role
        for node in graph.nodes.values():
            role_map[node.op_name] = get_role(node.op_name).value
    except Exception:
        return _smoke_test_structural(graph)

    graph_json = json.dumps({
        "nodes": {
            str(nid): {"op_name": n.op_name, "role": role_map.get(n.op_name, 0)}
            for nid, n in graph.nodes.items()
        },
        "edges": [
            {"src": str(src), "dst": str(nid)}
            for nid, n in graph.nodes.items()
            for src in n.input_ids
        ],
    })
    try:
        result = aria_core.smoke_test_graph(graph_json, D_MODEL, SEQ_LEN)
        return bool(result.get("ok", False) if isinstance(result, dict) else getattr(result, "ok", False))
    except Exception:
        return _smoke_test_structural(graph)


def _try_compile_and_train(graph) -> bool:
    """Attempt compile + micro-train. Returns True if no crash/NaN."""
    from research.synthesis.compiler import compile_model

    try:
        model = compile_model([graph], vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN)
    except Exception:
        return False

    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        for _ in range(TRAIN_STEPS):
            optimizer.zero_grad()
            logits = model(input_ids)
            if logits.dim() == 3:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                )
            else:
                loss = logits.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                return False

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        return True
    except Exception:
        return False


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_grammar_pass_rates():
    """Generate N_SEEDS graphs, measure smoke/train pass rates and diversity."""
    from research.synthesis.grammar import GrammarConfig, batch_generate

    config = GrammarConfig(model_dim=D_MODEL)
    graphs = batch_generate(N_SEEDS, config, base_seed=12345)

    assert len(graphs) >= N_SEEDS * 0.8, (
        f"batch_generate produced only {len(graphs)}/{N_SEEDS} graphs"
    )

    smoke_fn = _smoke_test_native if HAS_ARIA_CORE else _smoke_test_structural

    smoke_passed = 0
    train_passed = 0
    fingerprints: set = set()

    for graph in graphs:
        fp = graph.fingerprint()
        fingerprints.add(fp)

        if smoke_fn(graph):
            smoke_passed += 1
            if _try_compile_and_train(graph):
                train_passed += 1

    n = len(graphs)
    smoke_rate = smoke_passed / n
    train_rate = train_passed / n
    n_unique = len(fingerprints)

    # Report results before asserting
    print(f"\n--- Grammar Pass-Rate Results ({n} graphs) ---")
    print(f"Smoke test:  {smoke_passed}/{n} = {smoke_rate:.1%} (target ≥{SMOKE_PASS_TARGET:.0%})")
    print(f"Train test:  {train_passed}/{n} = {train_rate:.1%} (target ≥{TRAIN_PASS_TARGET:.0%})")
    print(f"Unique FPs:  {n_unique}/{n} (target ≥{UNIQUE_FP_TARGET})")

    assert smoke_rate >= SMOKE_PASS_TARGET, (
        f"Smoke test pass rate {smoke_rate:.1%} < {SMOKE_PASS_TARGET:.0%} target"
    )
    assert train_rate >= TRAIN_PASS_TARGET, (
        f"Train pass rate {train_rate:.1%} < {TRAIN_PASS_TARGET:.0%} target"
    )
    assert n_unique >= UNIQUE_FP_TARGET, (
        f"Only {n_unique} unique fingerprints < {UNIQUE_FP_TARGET} target (mode collapse)"
    )
