"""Sprint-6 tests: new probe tasks, adaptive anchors, ledger rotation."""

from __future__ import annotations

from pathlib import Path

import torch

from component_fab.improver import adaptive as adaptive_mod
from component_fab.harness.probe_tasks import DEFAULT_PROBE_TASKS
from component_fab.improver.adaptive import (
    adaptive_axis_variants,
    adaptive_cross_anchor_variants,
    build_anchor_pool,
)
from component_fab.state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)


def _seed_ledger(tmp_path: Path, *, n_promoted: int, n_rejected: int = 0) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for i in range(n_promoted):
        for cycle in (1, 2):
            ledger.record_grade(
                proposal_id=f"pro_{i}",
                name="improve_test_anchor_add_state_OL",
                category="lane",
                synthesis_kind="state_kernel_swap",
                cycle=cycle,
                composite_score=0.7,
                smoke_pass=True,
                learned_signal=True,
            )
        ledger.record_promotion(f"pro_{i}", PROMOTION_PROMOTED)
    for i in range(n_rejected):
        for cycle in (1, 2, 3, 4):
            ledger.record_grade(
                proposal_id=f"rej_{i}",
                name="improve_test_anchor_top_k_sparsity",
                category="lane",
                synthesis_kind="projection_swap",
                cycle=cycle,
                composite_score=0.1,
                smoke_pass=False,
                learned_signal=False,
            )
        ledger.record_promotion(f"rej_{i}", PROMOTION_REJECTED)
    return ledger


# ---------- Probe tasks ----------


def test_new_probe_tasks_in_default_suite() -> None:
    names = {t.name for t in DEFAULT_PROBE_TASKS}
    assert "copy_from_uniform_past" in names
    assert "causal_induction" in names


def test_new_probe_tasks_preserve_shape() -> None:
    x = torch.randn(2, 16, 8)
    for task in DEFAULT_PROBE_TASKS:
        if task.name not in ("copy_from_uniform_past", "causal_induction"):
            continue
        y = task.target_fn(x)
        assert y.shape == x.shape, f"{task.name} broke shape: {y.shape} vs {x.shape}"


def test_copy_from_uniform_past_is_genuinely_random() -> None:
    # Two distinct seeds should produce different outputs (random source indices).
    from component_fab.harness.probe_tasks import _copy_from_uniform_past

    x = torch.randn(1, 16, 4)
    torch.manual_seed(0)
    a = _copy_from_uniform_past(x)
    torch.manual_seed(1)
    b = _copy_from_uniform_past(x)
    assert not torch.allclose(a, b)


# ---------- Adaptive anchor expansion ----------


def test_build_anchor_pool_without_promoted_returns_only_corpus(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    pool = build_anchor_pool(
        ["tropical_attention"],
        ledger,
        use_promoted_as_anchors=False,
    )
    assert pool.fab_anchors == ()
    assert len(pool.corpus_anchors) == 1


def test_build_anchor_pool_with_promoted_adds_fab_anchors(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(adaptive_mod, "_load_saved_winners", lambda: [])
    ledger = _seed_ledger(tmp_path, n_promoted=3)
    pool = build_anchor_pool(
        ["tropical_attention"],
        ledger,
        use_promoted_as_anchors=True,
    )
    assert len(pool.fab_anchors) == 3
    assert len(pool.corpus_anchors) == 1
    assert len(pool.all_anchors) == 4


def test_adaptive_axis_variants_deprioritize_failed_deltas(tmp_path: Path) -> None:
    ledger = _seed_ledger(tmp_path, n_promoted=0, n_rejected=4)
    pool = build_anchor_pool(
        ["tropical_attention"],
        ledger,
        use_promoted_as_anchors=False,
    )
    specs = adaptive_axis_variants(pool, ledger)
    delta_names = {s.name.split("_", 2)[-1] for s in specs}
    assert "top_k_sparsity" not in delta_names


def test_adaptive_cross_anchor_caps_pairs(tmp_path: Path) -> None:
    ledger = _seed_ledger(tmp_path, n_promoted=5)
    pool = build_anchor_pool(
        ["tropical_attention", "clifford_attention"],
        ledger,
        use_promoted_as_anchors=True,
    )
    specs = adaptive_cross_anchor_variants(pool, ledger, max_pairs=4)
    # max_pairs=4 → 4 unique pairs → 8 specs (host,donor + donor,host per pair)
    assert len(specs) == 8


# ---------- Ledger rotation ----------


def test_ledger_rotation_creates_indexed_file(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for i in range(100):
        ledger.record_grade(
            proposal_id=f"p{i}",
            name=f"p{i}",
            category="lane",
            synthesis_kind="x",
            cycle=1,
            composite_score=0.5,
            smoke_pass=True,
            learned_signal=False,
        )
    rotated = ledger.rotate_if_oversized(max_bytes=512)
    assert rotated is not None
    assert rotated.exists()
    assert rotated.name == "ledger.jsonl.1"
    assert (tmp_path / "ledger.jsonl").exists()


def test_ledger_rotation_skipped_when_undersized(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="p1",
        name="p1",
        category="lane",
        synthesis_kind="x",
        cycle=1,
        composite_score=0.5,
        smoke_pass=True,
        learned_signal=False,
    )
    rotated = ledger.rotate_if_oversized(max_bytes=1_048_576)
    assert rotated is None


def test_ledger_rotation_increments_index(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for _ in range(2):
        for i in range(60):
            ledger.record_grade(
                proposal_id=f"p_{i}",
                name=f"p_{i}",
                category="lane",
                synthesis_kind="x",
                cycle=1,
                composite_score=0.5,
                smoke_pass=True,
                learned_signal=False,
            )
        ledger.rotate_if_oversized(max_bytes=256)
    assert (tmp_path / "ledger.jsonl.1").exists()
    assert (tmp_path / "ledger.jsonl.2").exists()
