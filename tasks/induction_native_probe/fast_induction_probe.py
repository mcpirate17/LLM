from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from research.eval._probe_runtime import disable_native_probe_dispatch
from research.eval.induction_probe import InductionResult, _generate_induction_batch
from research.eval.utils import make_adamw

_RESTRICTED_VOCAB = 256


def _amp_context(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@lru_cache(maxsize=1)
def load_native_induction_probe():
    source = Path(__file__).with_name("native_induction_probe.cpp")
    return load(
        name="native_induction_probe_ext",
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )


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
    use_native_generator: bool = True


class NativeInductionBatchSource:
    def __init__(
        self,
        *,
        device: str,
        gap: int,
        batch_size: int,
        pool_size: int,
        use_native: bool,
    ) -> None:
        self.device = device
        self.gap = int(gap)
        self.batch_size = int(batch_size)
        self.pool_size = int(pool_size)
        self.use_native = bool(use_native)
        self._device_ref = torch.empty(0, dtype=torch.long, device=device)
        self._native = load_native_induction_probe() if self.use_native else None
        self._cursor = 0
        self._pool_inputs: torch.Tensor | None = None
        self._pool_targets: torch.Tensor | None = None
        if self.pool_size > 0:
            self._refresh_pool()

    def _refresh_pool(self) -> None:
        if self._native is None:
            inputs = []
            targets = []
            for _ in range(self.pool_size):
                inp, tgt = _generate_induction_batch(
                    self.batch_size,
                    self.gap,
                    self.device,
                )
                inputs.append(inp)
                targets.append(tgt)
            self._pool_inputs = torch.stack(inputs, dim=0)
            self._pool_targets = torch.stack(targets, dim=0)
        else:
            self._pool_inputs, self._pool_targets = (
                self._native.induction_batch_pool_like(
                    self._device_ref,
                    self.pool_size,
                    self.batch_size,
                    self.gap,
                    _RESTRICTED_VOCAB,
                )
            )
        self._cursor = 0

    def next(self, size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if size is None or int(size) == self.batch_size:
            if self._pool_inputs is not None and self._pool_targets is not None:
                idx = self._cursor
                self._cursor += 1
                if self._cursor >= self.pool_size:
                    self._refresh_pool()
                return self._pool_inputs[idx], self._pool_targets[idx]
            if self._native is not None:
                return self._native.induction_batch_like(
                    self._device_ref,
                    self.batch_size,
                    self.gap,
                    _RESTRICTED_VOCAB,
                )

        actual_size = self.batch_size if size is None else int(size)
        if self._native is not None:
            return self._native.induction_batch_like(
                self._device_ref,
                actual_size,
                self.gap,
                _RESTRICTED_VOCAB,
            )
        return _generate_induction_batch(actual_size, self.gap, self.device)


def induction_score_fast(
    model: nn.Module,
    *,
    config: NativeProbeConfig | None = None,
) -> InductionResult:
    cfg = config or NativeProbeConfig()
    t0 = time.perf_counter()
    result = InductionResult(gap_accuracies={})
    train_gap = 8
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
        if str(cfg.device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(cfg.seed))

    try:
        probe_model = copy.deepcopy(model)
        probe_model.to(cfg.device)
        probe_model.train()
    except Exception as exc:
        result.status = f"copy_failed: {exc}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    opt = make_adamw(probe_model.parameters(), lr=cfg.lr)
    train_source = NativeInductionBatchSource(
        device=cfg.device,
        gap=train_gap,
        batch_size=cfg.batch_size,
        pool_size=cfg.pool_size,
        use_native=cfg.use_native_generator,
    )
    eval_sources = {
        int(gap): NativeInductionBatchSource(
            device=cfg.device,
            gap=int(gap),
            batch_size=cfg.batch_size,
            pool_size=cfg.pool_size,
            use_native=cfg.use_native_generator,
        )
        for gap in sorted(cfg.gaps)
    }

    try:
        with disable_native_probe_dispatch(probe_model, device=cfg.device):
            for step in range(1, cfg.n_train_steps + 1):
                if time.perf_counter() - t0 > cfg.timeout_s:
                    result.status = "timeout"
                    break

                input_ids, targets = train_source.next()
                opt.zero_grad(set_to_none=True)
                with _amp_context(cfg.device):
                    logits = probe_model(input_ids)
                    pred_logits = logits[:, input_ids.shape[1] - 1, :_RESTRICTED_VOCAB]
                    loss = F.cross_entropy(pred_logits.float(), targets)
                if not torch.isfinite(loss):
                    result.status = "diverged"
                    break
                loss.backward()
                nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
                opt.step()
                result.steps_trained = step

            probe_model.eval()
            with torch.inference_mode():
                for gap in sorted(cfg.gaps):
                    if time.perf_counter() - t0 > cfg.timeout_s:
                        result.gap_accuracies[int(gap)] = 0.0
                        continue
                    correct = 0
                    total = 0
                    remaining = cfg.n_eval
                    source = eval_sources[int(gap)]
                    while remaining > 0:
                        bs = min(cfg.batch_size, remaining)
                        inp, tgt = source.next(bs)
                        out = probe_model(inp)
                        preds = out[:, inp.shape[1] - 1, :_RESTRICTED_VOCAB].argmax(
                            dim=-1
                        )
                        correct += (preds == tgt).sum().item()
                        total += tgt.numel()
                        remaining -= bs
                    result.gap_accuracies[int(gap)] = round(correct / max(total, 1), 4)

    except Exception as exc:
        result.status = f"train_failed: {exc}"
    finally:
        del probe_model
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    if result.gap_accuracies:
        vals = list(result.gap_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)
    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def benchmark_induction_variants(
    model: nn.Module,
    *,
    device: str,
    n_train_steps: int = 300,
    n_eval: int = 100,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    from research.eval.induction_probe import induction_score

    variants = [
        (
            "baseline_python",
            lambda: induction_score(
                model,
                n_train_steps=n_train_steps,
                n_eval=n_eval,
                batch_size=batch_size,
                device=device,
                seed=123,
            ),
        ),
        (
            "native_generator",
            lambda: induction_score_fast(
                model,
                config=NativeProbeConfig(
                    n_train_steps=n_train_steps,
                    n_eval=n_eval,
                    batch_size=batch_size,
                    device=device,
                    seed=123,
                    pool_size=0,
                    use_native_generator=True,
                ),
            ),
        ),
        (
            "native_pool_64",
            lambda: induction_score_fast(
                model,
                config=NativeProbeConfig(
                    n_train_steps=n_train_steps,
                    n_eval=n_eval,
                    batch_size=batch_size,
                    device=device,
                    seed=123,
                    pool_size=64,
                    use_native_generator=True,
                ),
            ),
        ),
    ]
    out: list[dict[str, Any]] = []
    for label, fn in variants:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        out.append(
            {
                "label": label,
                "wall_ms": round(wall_ms, 1),
                "reported_elapsed_ms": result.elapsed_ms,
                "auc": result.auc,
                "status": result.status,
                "steps_trained": result.steps_trained,
                "gap_accuracies": result.gap_accuracies,
            }
        )
    return out
