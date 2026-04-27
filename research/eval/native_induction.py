from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from tasks.induction_native_probe.fast_induction_probe import (
        NativeProbeConfig,
        induction_score_fast,
        load_native_induction_probe,
    )
except ModuleNotFoundError:
    from research.eval.induction_probe import induction_score

    @dataclass(slots=True)
    class NativeProbeConfig:
        gaps: tuple[int, ...] = (4, 8, 16, 32, 64)
        n_train_steps: int = 1000
        n_eval: int = 200
        lr: float = 1e-3
        batch_size: int = 32
        device: str = "cuda"
        timeout_s: float = 120.0
        seed: int | None = None
        pool_size: int = 0
        use_native_generator: bool = False

    def load_native_induction_probe():
        return None

    def induction_score_fast(model, *, config: NativeProbeConfig | None = None):
        cfg = config or NativeProbeConfig()
        return induction_score(
            model,
            gaps=cfg.gaps,
            n_train_steps=cfg.n_train_steps,
            n_eval=cfg.n_eval,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            device=cfg.device,
            timeout_s=cfg.timeout_s,
            seed=cfg.seed,
        )

INDUCTION_METRIC_VERSION = "native_pool_64_v1"
INDUCTION_SPEED_MODE = "native_pool_64"
INDUCTION_GAPS = (4, 8, 16, 32, 64)
INDUCTION_TRAIN_STEPS = 500
INDUCTION_EVAL_EXAMPLES = 100
INDUCTION_BATCH_SIZE = 16
INDUCTION_POOL_SIZE = 64


def induction_score_gold(model, *, device: str, seed: int | None = None):
    """Run the canonical induction probe used for all future writes."""
    load_native_induction_probe()
    cfg = NativeProbeConfig(
        gaps=INDUCTION_GAPS,
        n_train_steps=INDUCTION_TRAIN_STEPS,
        n_eval=INDUCTION_EVAL_EXAMPLES,
        batch_size=INDUCTION_BATCH_SIZE,
        device=device,
        seed=seed,
        pool_size=INDUCTION_POOL_SIZE,
        use_native_generator=True,
    )
    return induction_score_fast(model, config=cfg)


def induction_result_metadata(result) -> dict[str, Any]:
    return {
        "induction_auc": result.auc,
        "induction_gap_accuracies": dict(result.gap_accuracies or {}),
        "induction_probe_train_steps": INDUCTION_TRAIN_STEPS,
        "induction_probe_eval_examples": INDUCTION_EVAL_EXAMPLES,
        "induction_probe_batch_size": INDUCTION_BATCH_SIZE,
        "induction_probe_gaps": list(INDUCTION_GAPS),
        "induction_probe_elapsed_ms": result.elapsed_ms,
        "induction_probe_metric_version": INDUCTION_METRIC_VERSION,
        "induction_probe_speed_mode": INDUCTION_SPEED_MODE,
        "induction_probe_pool_size": INDUCTION_POOL_SIZE,
    }
