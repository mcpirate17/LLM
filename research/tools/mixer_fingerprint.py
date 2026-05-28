# pyright: reportPrivateImportUsage=false
# guardrail: allow-god-file
"""Per-mixer capability fingerprint at nano-scale.

Trains a ~10M-param ``TinyLM`` whose mixer is a single lane primitive,
for 10K steps on wikitext-103, with checkpoints at 500 / 1K / 5K / 10K.
At each checkpoint runs the **cheap** eval suite; at the final
checkpoint additionally runs the **expensive** suite. Output: one JSONL
per checkpoint plus a summary table.

Goal: produce a fingerprint of which capability emerges fastest with
each mixer primitive, holding everything else (data, schedule, scale)
fixed. Output is a `mixer × capability → emergence_signal` matrix that
tells you what each primitive is structurally good at.

See ``research/notes/tropical_gate_120m_pretrain_README.md`` §5.5 for
the motivating context (the 120M tropical+wavelet run halted at step
143K with capability instability; this experiment characterizes mixers
at the regime where the architecture's BLiMP advantage actually shows
up — small scale, short training).

Cost: ~25-35 minutes per nano mixer on a single GPU; larger 60M/100K
runs are long-running training jobs. Training uses CUDA AMP and
``torch.compile`` by default when available. Compile is training-only so
eval probes and saved weights remain eager-compatible.

Usage:
    PYTHONPATH=. python -m research.tools.mixer_fingerprint \
        --mixer softmax_attention \
        --output research/reports/mixer_fingerprint/

Available --mixer values: any name accepted by
``research.tools.scaling_blimp_study._build_lane_factory``, plus a few
single-lane shortcuts added here for the experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from component_fab.harness.nano_bind_probe import nano_bind_gate
from component_fab.harness.nano_induction_probe import (
    nano_induction_gate,
    nano_induction_nearest,
)
from component_fab.harness.standard_block import LaneTestBlock
from research.defaults import VOCAB_SIZE
from research.tools.scaling_blimp_study import (
    _build_lane_factory,
    _build_tinylm,
    _causal_lm_loss,
    _load_wikitext_tokens,
    _RandomWindowBatcher,
)

NANO_SIZING = {"dim": 96, "n_blocks": 12}  # ~10.6M params with vocab=100K
DEFAULT_CHECKPOINT_STEPS = (500, 1_000, 5_000, 10_000)
_REPO = Path(__file__).resolve().parents[2]


def _configure_torch_performance() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.recompile_limit = 32
        # CompiledOp.forward gets traced per block; each block's weight tensor
        # has a different rank (linear_proj 2D, rmsnorm 1D, etc.) so dynamo's
        # default static-shape guard recompiles on every rank change, blowing
        # the cache for graphs with many op kinds. Letting params be
        # shape-dynamic lets the trace be reused across blocks.
        torch._dynamo.config.force_parameter_static_shapes = False
    except Exception:
        pass
    try:
        import torch._inductor.config as inductor_config

        # The lane catalogue includes custom control flow; no-cudagraph compile
        # modes have been the most reliable in the long pretrain runner.
        inductor_config.triton.cudagraphs = False
        if hasattr(inductor_config.triton, "cudagraph_trees"):
            inductor_config.triton.cudagraph_trees = False
    except Exception:
        pass


def _mark_cudagraph_step_begin() -> None:
    if not torch.cuda.is_available():
        return
    try:
        marker = torch.compiler.cudagraph_mark_step_begin
    except AttributeError:
        return
    marker()


def _is_lazy_compile_failure(exc: BaseException) -> bool:
    module = type(exc).__module__
    name = type(exc).__name__
    return module.startswith(("torch._dynamo", "torch._inductor")) or name in {
        "BackendCompilerFailed",
        "InductorError",
        "Unsupported",
    }


def _compact_exception_message(exc: BaseException, *, max_chars: int = 220) -> str:
    message = " ".join(str(exc).split())
    for marker in ("Set TORCHDYNAMO_VERBOSE=", "For even more developer context"):
        pos = message.find(marker)
        if pos >= 0:
            message = message[:pos].strip()
    if len(message) > max_chars:
        message = message[: max_chars - 3].rstrip() + "..."
    return message or type(exc).__name__


def _autocast_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class _TrainLossWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(ids)
        return _causal_lm_loss(logits, ids)


def _maybe_compile_training_model(
    model: nn.Module,
    *,
    enabled: bool,
    required: bool,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    meta: dict[str, Any] = {
        "requested": bool(enabled),
        "compiled": False,
        "mode": mode,
        "fullgraph": bool(fullgraph),
        "dynamic": bool(dynamic),
    }
    if not enabled:
        return model, meta
    if device.type != "cuda":
        meta["error"] = f"compile skipped on device={device.type}"
        if required:
            raise RuntimeError(meta["error"])
        return model, meta
    if not hasattr(torch, "compile"):
        meta["error"] = "torch.compile unavailable"
        if required:
            raise RuntimeError(meta["error"])
        return model, meta
    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=bool(fullgraph),
            dynamic=bool(dynamic),
        )
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {_compact_exception_message(exc)}"
        if required:
            raise
        return model, meta
    meta["compiled"] = True
    return compiled, meta


def _make_optimizer(
    model: nn.Module, *, learning_rate: float, device: torch.device
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    use_fused = device.type == "cuda"
    try:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=0.0,
            fused=use_fused,
        )
        return optim, {"name": "AdamW", "fused": bool(use_fused), "weight_decay": 0.0}
    except TypeError:
        optim = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        return optim, {"name": "AdamW", "fused": False, "weight_decay": 0.0}


class _WarmupCosineSchedule:
    """Linear warmup followed by cosine decay, keyed by completed steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))

    def lr_at(self, completed_steps: int) -> float:
        s = max(0, int(completed_steps))
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return self.learning_rate * float(s + 1) / float(self.warmup_steps)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (s - self.warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.learning_rate - self.min_lr) * cosine

    def apply(self, completed_steps: int) -> float:
        lr = self.lr_at(completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def _scheduled_seq_len(
    *,
    schedule: str,
    step: int,
    max_seq_len: int,
    initial_seq_len: int,
    warmup_steps: int,
) -> int:
    """Return training sequence length for the local 1-indexed step."""
    max_seq = max(1, int(max_seq_len))
    if str(schedule).strip().lower() != "growing":
        return max_seq
    initial = max(1, min(int(initial_seq_len), max_seq))
    warmup = max(1, int(warmup_steps))
    if int(step) >= warmup:
        return max_seq
    progress = max(0.0, float(int(step) - 1) / float(warmup))
    return int(initial + progress * (max_seq - initial))


@dataclass
class _PlateauTracker:
    """Hard early-stop on held-out wikitext PPL.

    A tick where ``ppl >= best_ppl * (1 - min_delta)`` counts as stale. After
    ``patience`` consecutive stale ticks past ``min_steps``, ``update`` returns
    True and the training loop halts. Pre-``min_steps`` ticks still refresh
    ``best_ppl`` so the first qualifying tick has a meaningful baseline.
    """

    patience: int = 3
    min_delta: float = 0.005
    min_steps: int = 20_000
    best_ppl: float = math.inf
    best_step: int | None = None
    stale_ticks: int = 0
    triggered_at_step: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(self, step: int, ppl: float) -> bool:
        if not math.isfinite(ppl):
            self.history.append(
                {"step": int(step), "ppl": None, "stale": self.stale_ticks}
            )
            return False
        improved = ppl < self.best_ppl * (1.0 - float(self.min_delta))
        if improved:
            self.best_ppl = float(ppl)
            self.best_step = int(step)
            if step >= self.min_steps:
                self.stale_ticks = 0
        else:
            if self.best_ppl == math.inf:
                self.best_ppl = float(ppl)
                self.best_step = int(step)
            if step >= self.min_steps:
                self.stale_ticks += 1
        self.history.append(
            {"step": int(step), "ppl": float(ppl), "stale": int(self.stale_ticks)}
        )
        if step >= self.min_steps and self.stale_ticks >= int(self.patience):
            self.triggered_at_step = int(step)
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "patience": int(self.patience),
            "min_delta": float(self.min_delta),
            "min_steps": int(self.min_steps),
            "best_ppl": (None if self.best_ppl == math.inf else float(self.best_ppl)),
            "best_step": self.best_step,
            "stale_ticks": int(self.stale_ticks),
            "triggered_at_step": self.triggered_at_step,
            "history": list(self.history),
        }


@torch.inference_mode()
def _eval_ppl_fast(
    model: nn.Module,
    batches: list[torch.Tensor],
    *,
    amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> float:
    if not batches:
        return float("nan")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    with torch.amp.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=bool(amp and device.type == "cuda"),
    ):
        for batch in batches:
            logits = model(batch)
            total_loss += float(_causal_lm_loss(logits, batch).item())
    if was_training:
        model.train()
    return float(math.exp(min(total_loss / max(1, len(batches)), 30.0)))


def _dataclass_to_metrics(d: Any) -> dict[str, Any]:
    if is_dataclass(d):
        return {k: v for k, v in asdict(d).items()}
    if hasattr(d, "to_dict"):
        return d.to_dict()
    return {"value": str(d)}


def _cheap_evals(
    *,
    model: nn.Module,
    factory: Callable[[int], nn.Module],
    val_batches: list[torch.Tensor],
    device: torch.device,
    seed: int,
    amp: bool,
    amp_dtype: torch.dtype,
    probe_dim: int = 32,
    model_dim: int | None = None,
    run_nano_induction_nearest: bool = True,
) -> dict[str, Any]:
    """Probes under ~10s wall time each (run at every checkpoint)."""
    out: dict[str, Any] = {}

    t = time.monotonic()
    out["wikitext_ppl"] = _eval_ppl_fast(
        model, val_batches, amp=amp, amp_dtype=amp_dtype, device=device
    )
    out["_t_wikitext"] = round(time.monotonic() - t, 2)

    t = time.monotonic()
    from research.eval.blimp_eval import evaluate_blimp

    blimp = evaluate_blimp(
        model,
        vocab_size=VOCAB_SIZE,
        device=str(device),
        n_per_subtask=25,
        max_seq_len=256,
    )
    out["blimp_overall"] = float(blimp.overall_accuracy or 0.0)
    out["_t_blimp"] = round(time.monotonic() - t, 2)

    t = time.monotonic()
    from research.eval.hellaswag_eval import evaluate_hellaswag

    hs = evaluate_hellaswag(model, VOCAB_SIZE, str(device), n_examples=200)
    out["hellaswag_acc"] = hs.get("hellaswag_acc")
    out["_t_hellaswag"] = round(time.monotonic() - t, 2)

    # Production induction screening (deepcopy + frozen-weights eval; ~10s)
    t = time.monotonic()
    from research.eval.induction_probe import induction_score

    ind = induction_score(model, device=str(device), seed=seed)
    out["induction_screening_auc"] = getattr(ind, "auc", None)
    out["_t_induction"] = round(time.monotonic() - t, 2)

    # NB 0.5 + NI 0.5 on a fresh lane (structural probe — doesn't see trained weights).
    # Use a small ``probe_dim`` (not the model's dim) so the probe stays cheap.
    t = time.monotonic()
    dim = probe_dim
    out["nb05"] = _dataclass_to_metrics(
        nano_bind_gate(factory(dim), dim=dim, n_train_steps=60, seed=seed)
    )
    out["_t_nb05"] = round(time.monotonic() - t, 2)

    t = time.monotonic()
    stacked = nn.Sequential(
        LaneTestBlock(factory(dim), dim), LaneTestBlock(factory(dim), dim)
    )
    out["ni05"] = _dataclass_to_metrics(
        nano_induction_gate(stacked, dim=dim, n_train_steps=150, seed=seed)
    )
    out["_t_ni05"] = round(time.monotonic() - t, 2)

    if run_nano_induction_nearest:
        t = time.monotonic()
        nearest_dim = int(model_dim or probe_dim)
        try:
            nearest_stack = nn.Sequential(
                LaneTestBlock(factory(nearest_dim), nearest_dim),
                LaneTestBlock(factory(nearest_dim), nearest_dim),
            )
            out["nano_induction_nearest"] = _dataclass_to_metrics(
                nano_induction_nearest(
                    nearest_stack,
                    dim=nearest_dim,
                    n_train_steps=120,
                    seed=seed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            out["nano_induction_nearest"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        out["_t_nano_induction_nearest"] = round(time.monotonic() - t, 2)

    return out


def _try_probe(out: dict[str, Any], key: str, fn: Callable[[], Any]) -> None:
    """Run a probe, store its dict result + wall time, skip on error.

    Each enrichment probe runs against deepcopies of the model, so a single
    failure shouldn't tank the rest of the suite — record the error and move on.
    """
    t = time.monotonic()
    try:
        result = fn()
        out[key] = result if isinstance(result, dict) else _dataclass_to_metrics(result)
    except Exception as exc:  # noqa: BLE001
        out[key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    out[f"_t_{key}"] = round(time.monotonic() - t, 2)


def _expensive_core_evals(
    *, model: nn.Module, device: torch.device, out: dict[str, Any]
) -> None:
    """The core probes the fingerprint runs: induction_intermediate,
    ar_legacy, binding_intermediate.

    ar_curriculum was removed 2026-05-22: at production scale (76M+ params,
    well-pretrained) its 1200-step probe-train budget can't move the model
    and the metric saturates. It remains a valid screening tool for nano-
    scale architecture differentiation; see ``research/eval/ar_curriculum_probe.py``
    and the synthesis-scientist screening pipeline. The harder
    ``ar_validation`` (v3 stable, 3-seed) in ``_expensive_enrichment_evals``
    is the production-scale AR signal."""
    from research.eval.induction_intermediate_probe import run_induction_intermediate
    from research.eval.associative_recall import associative_recall_score
    from research.eval.binding_intermediate_probe import run_binding_intermediate

    _try_probe(
        out,
        "induction_intermediate",
        lambda: run_induction_intermediate(
            model, n_train_steps=300, n_eval=128, batch_size=8, device=device
        ).to_dict(),
    )
    _try_probe(
        out,
        "ar_legacy",
        lambda: associative_recall_score(
            model, n_train_steps=300, n_eval=128, batch_size=8, device=device
        ),
    )
    _try_probe(
        out,
        "binding_v2",
        lambda: run_binding_intermediate(
            model,
            n_train_steps=300,
            n_eval=128,
            train_batch_size=8,
            eval_batch_size=8,
            device=device,
        ).to_dict(),
    )


def _expensive_enrichment_evals(
    *, model: nn.Module, device: torch.device, out: dict[str, Any]
) -> None:
    """Harder probes added 2026-05-17 because binding_intermediate saturated
    at 0.998 across every 30M/60M run, leaving no resolution to distinguish
    architectures or undertrain-vs-saturation. Includes longer-distance binding,
    curriculum binding, multi-slot binding, deeper induction_validation, and
    deeper ar_validation."""
    from research.eval.binding_range import binding_range_profile
    from research.eval.binding_curriculum import curriculum_binding_range_profile
    from research.eval.binding_multislot_probe import (
        binding_multislot_probe,
        BindingMultislotConfig,
    )
    from research.eval.induction_validation_probe import (
        run_induction_validation_champion,
    )
    from research.eval.ar_validation import (
        run_ar_validation,
        ARValidationConfig,
        STABLE_AR_VALIDATION_PROTOCOL,
    )

    _try_probe(
        out,
        "binding_range",
        lambda: binding_range_profile(
            model,
            distances=(8, 16, 32, 64, 128, 256),
            n_eval=128,
            seq_len=320,
            batch_size=8,
            device=device,
        ).to_dict(),
    )
    _try_probe(
        out,
        "binding_curriculum",
        lambda: curriculum_binding_range_profile(
            model,
            distances=(4, 8, 16, 32, 64),
            n_train_steps=300,
            n_eval=128,
            train_batch_size=8,
            eval_batch_size=8,
            device=device,
        ).to_dict(),
    )
    _try_probe(
        out,
        "binding_multislot",
        lambda: binding_multislot_probe(
            model,
            cfg=BindingMultislotConfig(train_steps=400, batch_size=8, n_eval=128),
            device=device,
        ).to_dict(),
    )
    # induction_validation only accepts {2000, 5000, 10000} as n_train_steps;
    # 2K is the calibrated minimum.
    _try_probe(
        out,
        "induction_validation",
        lambda: run_induction_validation_champion(
            model,
            n_train_steps=2000,
            device=device,
        ).to_dict(),
    )
    _try_probe(
        out,
        "ar_validation",
        lambda: run_ar_validation(
            model,
            cfg=ARValidationConfig(
                protocol=STABLE_AR_VALIDATION_PROTOCOL,
                seed_count=3,
                auto_size_budget=True,
                deterministic_episode_bank=True,
            ),
            device=device,
        ).to_dict(),
    )


def _expensive_evals(
    *, model: nn.Module, device: torch.device, seed: int
) -> dict[str, Any]:
    """Probes >10s each (run only at the final checkpoint)."""
    del seed  # reserved for per-probe seeding if needed; probes self-seed today
    out: dict[str, Any] = {}
    _expensive_core_evals(model=model, device=device, out=out)
    _expensive_enrichment_evals(model=model, device=device, out=out)
    return out


def _mid_tier_evals(
    *,
    model: nn.Module,
    device: torch.device,
    val_batches: list[torch.Tensor] | None = None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Cheap mid-run screening signal; harder train-copy probes stay final-only.

    When ``val_batches`` is provided, also reports held-out ``wikitext_ppl``
    so the plateau gate has a signal to act on.
    """
    out: dict[str, Any] = {}
    from research.eval.binding_pipeline import run_screening_binding_probes

    t = time.monotonic()
    try:
        out.update(run_screening_binding_probes(model, device=str(device)))
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["_t_screening_binding_induction"] = round(time.monotonic() - t, 2)

    if val_batches:
        t = time.monotonic()
        out["wikitext_ppl"] = _eval_ppl_fast(
            model, val_batches, amp=amp, amp_dtype=amp_dtype, device=device
        )
        out["_t_wikitext"] = round(time.monotonic() - t, 2)
    return out


def _forward_loss_with_lazy_fallback(
    model: nn.Module,
    train_model: nn.Module,
    ids,
    *,
    amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    compile_required: bool,
    compile_meta: dict[str, Any],
) -> tuple[Any, nn.Module]:
    """Forward + loss under autocast, falling back to eager on lazy-compile failure."""
    with torch.amp.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=bool(amp and device.type == "cuda"),
    ):
        try:
            if train_model is model:
                logits = train_model(ids)
                loss = _causal_lm_loss(logits, ids)
            else:
                loss = train_model(ids)
            return loss, train_model
        except Exception as exc:  # noqa: BLE001
            if (
                train_model is model
                or compile_required
                or not _is_lazy_compile_failure(exc)
            ):
                raise
            compile_meta["lazy_disabled"] = True
            compile_meta["lazy_error"] = (
                f"{type(exc).__name__}: {_compact_exception_message(exc)}"
            )
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            train_model = model
            logits = train_model(ids)
            loss = _causal_lm_loss(logits, ids)
            return loss, train_model


def _log_train_step(
    *,
    step: int,
    n_steps: int,
    global_step: int,
    step_offset: int,
    loss,
    grad_norm,
    lr: float,
    curr_seq_len: int,
    t0: float,
    last_log_t: float,
    tokens_seen: int,
    last_log_tokens: int,
) -> tuple[float, int]:
    now = time.monotonic()
    elapsed = max(now - t0, 1e-9)
    recent_elapsed = max(now - last_log_t, 1e-9)
    recent_tokens = max(0, tokens_seen - last_log_tokens)
    loss_f = float(loss.detach().item())
    g_val = grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm
    grad_norm_f = float(g_val.item() if torch.is_tensor(g_val) else g_val)
    if not math.isfinite(loss_f):
        raise FloatingPointError(f"nonfinite loss at step={step}")
    ppl_f = float(math.exp(min(loss_f, 30.0)))
    print(
        "train "
        f"step={global_step}/{step_offset + n_steps} "
        f"local={step}/{n_steps} "
        f"loss={loss_f:.4f} "
        f"ppl={ppl_f:.2f} "
        f"lr={lr:.3g} "
        f"seq_len={int(curr_seq_len)} "
        f"grad_norm={grad_norm_f:.3g} "
        f"tok_s={tokens_seen / elapsed:.0f} "
        f"recent_tok_s={recent_tokens / recent_elapsed:.0f}",
        flush=True,
    )
    return now, tokens_seen


def _finalize_history(
    history: list[dict[str, Any]], seq_len: int
) -> list[dict[str, Any]]:
    final_history: list[dict[str, Any]] = []
    for h in history:
        if "__async_tensor__" not in h:
            final_history.append(h)
            continue
        loss_f = float(h["loss_t"].item())
        g_val = h["grad_norm_t"]
        grad_norm_f = float(g_val.item() if torch.is_tensor(g_val) else g_val)
        if not math.isfinite(grad_norm_f):
            final_history.append(
                {
                    "step": h["local_step"],
                    "loss": loss_f,
                    "grad_norm": grad_norm_f,
                    "skipped": "nonfinite_grad",
                }
            )
        else:
            final_history.append(
                {
                    "step": h["step"],
                    "local_step": h["local_step"],
                    "loss": loss_f,
                    "ppl": float(math.exp(min(loss_f, 30.0))),
                    "grad_norm": grad_norm_f,
                    "lr": h["lr"],
                    "seq_len": int(h.get("seq_len", seq_len)),
                }
            )
    return final_history


def _maybe_save(
    *,
    on_save,
    step: int,
    global_step: int,
    n_steps: int,
    save_every_steps: int,
    save_steps: list[dict[str, Any]],
    force: bool = False,
) -> None:
    if on_save is None:
        return
    if not (
        force
        or (
            int(save_every_steps) > 0
            and (step % int(save_every_steps) == 0 or step == n_steps)
        )
    ):
        return
    saved_path = on_save(global_step)
    if saved_path is not None:
        save_steps.append(
            {"step": global_step, "local_step": step, "path": str(saved_path)}
        )


def _build_train_meta(
    *,
    history,
    optim_meta,
    learning_rate,
    min_lr,
    warmup_steps,
    n_steps,
    scheduler,
    seq_len_schedule,
    initial_seq_len,
    seq_len,
    curriculum_warmup_steps,
    step_offset,
    amp,
    amp_dtype,
    device,
    compile_meta,
    save_steps,
    mid_tier_steps,
    t0,
    halt_reason: str,
    halt_step: int,
) -> dict[str, Any]:
    final_history = _finalize_history(history, seq_len)
    return {
        "history": final_history,
        "optimizer": optim_meta,
        "scheduler": {
            "name": "warmup_cosine",
            "learning_rate": float(learning_rate),
            "min_lr": float(min_lr),
            "warmup_steps": int(warmup_steps),
            "total_steps": int(n_steps),
            "final_lr": scheduler.lr_at(int(n_steps)),
        },
        "seq_len_schedule": {
            "name": str(seq_len_schedule),
            "initial_seq_len": int(initial_seq_len),
            "max_seq_len": int(seq_len),
            "warmup_steps": int(curriculum_warmup_steps),
        },
        "step_offset": int(step_offset),
        "amp": {
            "enabled": bool(amp and device.type == "cuda"),
            "dtype": str(amp_dtype).replace("torch.", ""),
        },
        "compile_runtime": compile_meta,
        "saved_weights": save_steps,
        "mid_tier_steps": mid_tier_steps,
        "wall_clock_s": round(time.monotonic() - t0, 1),
        "halt_reason": halt_reason,
        "halt_step": halt_step,
    }


def _record_history(
    history,
    *,
    step: int,
    global_step: int,
    loss,
    grad_norm,
    lr: float,
    curr_seq_len: int,
) -> None:
    if not (step % 100 == 0 or step <= 50):
        return
    history.append(
        {
            "__async_tensor__": True,
            "step": global_step,
            "local_step": step,
            "loss_t": loss.detach(),
            "grad_norm_t": (
                grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm
            ),
            "lr": lr,
            "seq_len": int(curr_seq_len),
        }
    )


@dataclass
class _TrainState:
    optim: Any
    scheduler: Any
    scaler: Any
    compile_meta: dict[str, Any]
    history: list[dict[str, Any]]
    save_steps: list[dict[str, Any]]
    mid_tier_steps: list[int]
    sorted_ckpts: list[int]
    next_idx: int = 0
    tokens_seen: int = 0
    last_log_t: float = 0.0
    last_log_tokens: int = 0


def _do_train_step(
    model,
    train_model,
    train_batcher,
    state: _TrainState,
    *,
    step: int,
    n_steps: int,
    step_offset: int,
    device,
    amp: bool,
    amp_dtype,
    compile_required: bool,
    log_every: int,
    batch_size: int,
    seq_len: int,
    seq_len_schedule: str,
    initial_seq_len: int,
    curriculum_warmup_steps: int,
    t0: float,
) -> Any:
    """Execute a single training step in-place against ``state``; returns updated train_model."""
    global_step = int(step_offset) + step
    lr = state.scheduler.apply(step - 1)
    curr_seq_len = _scheduled_seq_len(
        schedule=seq_len_schedule,
        step=step,
        max_seq_len=seq_len,
        initial_seq_len=initial_seq_len,
        warmup_steps=curriculum_warmup_steps,
    )
    ids = train_batcher.next(curr_seq_len)
    state.optim.zero_grad(set_to_none=True)
    if train_model is not model:
        _mark_cudagraph_step_begin()
    loss, train_model = _forward_loss_with_lazy_fallback(
        model,
        train_model,
        ids,
        amp=amp,
        amp_dtype=amp_dtype,
        device=device,
        compile_required=compile_required,
        compile_meta=state.compile_meta,
    )
    state.scaler.scale(loss).backward()
    state.scaler.unscale_(state.optim)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=False
    )
    state.scaler.step(state.optim)
    state.scaler.update()
    state.tokens_seen += int(batch_size) * int(curr_seq_len)
    _record_history(
        state.history,
        step=step,
        global_step=global_step,
        loss=loss,
        grad_norm=grad_norm,
        lr=lr,
        curr_seq_len=curr_seq_len,
    )
    if log_every > 0 and (step % log_every == 0 or step == 1 or step == n_steps):
        state.last_log_t, state.last_log_tokens = _log_train_step(
            step=step,
            n_steps=n_steps,
            global_step=global_step,
            step_offset=int(step_offset),
            loss=loss,
            grad_norm=grad_norm,
            lr=lr,
            curr_seq_len=int(curr_seq_len),
            t0=t0,
            last_log_t=state.last_log_t,
            tokens_seen=state.tokens_seen,
            last_log_tokens=state.last_log_tokens,
        )
    return train_model


def _post_step_callbacks(
    state: _TrainState,
    *,
    step: int,
    global_step: int,
    n_steps: int,
    step_offset: int,
    save_every_steps: int,
    on_save,
    on_mid_tier,
    on_checkpoint,
    mid_tier_every_steps: int,
) -> tuple[str, int] | None:
    """Run save/mid-tier/checkpoint callbacks; return (halt_reason, halt_step) if training should stop."""
    _maybe_save(
        on_save=on_save,
        step=step,
        global_step=global_step,
        n_steps=n_steps,
        save_every_steps=save_every_steps,
        save_steps=state.save_steps,
    )
    halted = False
    if (
        on_mid_tier is not None
        and int(mid_tier_every_steps) > 0
        and step % int(mid_tier_every_steps) == 0
        and step != n_steps
    ):
        halted = bool(on_mid_tier(global_step))
        state.mid_tier_steps.append(global_step)
    if halted:
        _maybe_save(
            on_save=on_save,
            step=step,
            global_step=global_step,
            n_steps=n_steps,
            save_every_steps=save_every_steps,
            save_steps=state.save_steps,
            force=True,
        )
        on_checkpoint(global_step, force_final=True)
        return ("ppl_plateau", global_step)
    if (
        state.next_idx < len(state.sorted_ckpts)
        and step >= state.sorted_ckpts[state.next_idx]
    ):
        on_checkpoint(int(step_offset) + state.sorted_ckpts[state.next_idx])
        state.next_idx += 1
    return None


@dataclass
class _LoopCfg:
    n_steps: int
    step_offset: int
    learning_rate: float
    min_lr: float
    warmup_steps: int
    device: torch.device
    amp: bool
    amp_dtype: torch.dtype
    compile_required: bool
    log_every_steps: int
    batch_size: int
    seq_len: int
    seq_len_schedule: str
    initial_seq_len: int
    curriculum_warmup_steps: int
    checkpoint_steps: tuple[int, ...]
    save_every_steps: int
    mid_tier_every_steps: int


def _init_training_state(
    model: nn.Module, cfg: _LoopCfg
) -> tuple[_TrainState, dict[str, Any], float]:
    optim, optim_meta = _make_optimizer(
        model, learning_rate=cfg.learning_rate, device=cfg.device
    )
    scheduler = _WarmupCosineSchedule(
        optim,
        learning_rate=float(cfg.learning_rate),
        min_lr=float(cfg.min_lr),
        warmup_steps=int(cfg.warmup_steps),
        total_steps=int(cfg.n_steps),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(
            cfg.amp and cfg.amp_dtype == torch.float16 and cfg.device.type == "cuda"
        ),
    )
    t0 = time.monotonic()
    state = _TrainState(
        optim=optim,
        scheduler=scheduler,
        scaler=scaler,
        compile_meta={"lazy_disabled": False},
        history=[],
        save_steps=[],
        mid_tier_steps=[],
        sorted_ckpts=sorted(set(int(s) for s in cfg.checkpoint_steps)),
        last_log_t=t0,
    )
    return state, optim_meta, t0


def _training_loop(
    model: nn.Module,
    train_model: nn.Module,
    train_batcher: _RandomWindowBatcher,
    *,
    on_checkpoint: Callable[..., None],
    on_save: Callable[[int], Path | None] | None,
    on_mid_tier: Callable[[int], bool] | None,
    **cfg_kwargs,
) -> dict[str, Any]:
    """Run training, calling ``on_checkpoint(step)`` at each ckpt step."""
    cfg = _LoopCfg(**cfg_kwargs)
    state, optim_meta, t0 = _init_training_state(model, cfg)
    model.train()
    train_model.train()
    log_every = int(cfg.log_every_steps)
    halt_reason = "completed"
    halt_step = int(cfg.step_offset) + int(cfg.n_steps)
    for step in range(1, cfg.n_steps + 1):
        global_step = int(cfg.step_offset) + step
        train_model = _do_train_step(
            model,
            train_model,
            train_batcher,
            state,
            step=step,
            n_steps=cfg.n_steps,
            step_offset=cfg.step_offset,
            device=cfg.device,
            amp=cfg.amp,
            amp_dtype=cfg.amp_dtype,
            compile_required=cfg.compile_required,
            log_every=log_every,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            seq_len_schedule=cfg.seq_len_schedule,
            initial_seq_len=cfg.initial_seq_len,
            curriculum_warmup_steps=cfg.curriculum_warmup_steps,
            t0=t0,
        )
        halt_info = _post_step_callbacks(
            state,
            step=step,
            global_step=global_step,
            n_steps=cfg.n_steps,
            step_offset=cfg.step_offset,
            save_every_steps=cfg.save_every_steps,
            on_save=on_save,
            on_mid_tier=on_mid_tier,
            on_checkpoint=on_checkpoint,
            mid_tier_every_steps=cfg.mid_tier_every_steps,
        )
        if halt_info is not None:
            halt_reason, halt_step = halt_info
            break

    return _build_train_meta(
        history=state.history,
        optim_meta=optim_meta,
        learning_rate=cfg.learning_rate,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps,
        n_steps=cfg.n_steps,
        scheduler=state.scheduler,
        seq_len_schedule=cfg.seq_len_schedule,
        initial_seq_len=cfg.initial_seq_len,
        seq_len=cfg.seq_len,
        curriculum_warmup_steps=cfg.curriculum_warmup_steps,
        step_offset=cfg.step_offset,
        amp=cfg.amp,
        amp_dtype=cfg.amp_dtype,
        device=cfg.device,
        compile_meta=state.compile_meta,
        save_steps=state.save_steps,
        mid_tier_steps=state.mid_tier_steps,
        t0=t0,
        halt_reason=halt_reason,
        halt_step=halt_step,
    )


def _parse_pattern(pattern: str) -> list[tuple[str, int]]:
    """``conv:4,three_lane:4,conv:4`` → ``[("conv",4),("three_lane",4),("conv",4)]``."""
    out: list[tuple[str, int]] = []
    for chunk in pattern.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, count = chunk.partition(":")
        out.append((name.strip(), int(count) if count else 1))
    return out


def _resolve_lane_factories(
    mixer: str, pattern: str | None
) -> tuple[Callable[[int], nn.Module], Callable[[int], nn.Module]]:
    """Return (model_factory, probe_factory).

    For ``interleaved`` mode the model_factory is **stateful** — each call returns
    the next lane in the per-block pattern. The probe_factory is stateless and
    always returns the first pattern entry's lane (used by NB 0.5 / NI 0.5).

    For non-interleaved mixers both factories are the same stateless builder.
    """
    if mixer != "interleaved":
        f = _build_lane_factory(mixer)
        return f, f
    if not pattern:
        raise ValueError("--pattern required when --mixer=interleaved")
    aliases = {
        "three_lane": "tropical_sparsemax_wavelet_three_lane",
        "two_lane": "block_gated_parallel",
        "conv": "causal_conv",
        "softmax": "softmax_attention",
        "tropical": "tropical_attention",
        "sparsemax": "sparsemax_attention",
        "wavelet": "multiscale_wavelet",
        "mamba": "simplified_mamba",
    }
    expanded: list[str] = []
    for name, count in _parse_pattern(pattern):
        name = aliases.get(name, name)
        expanded.extend([name] * int(count))
    if not expanded:
        raise ValueError("pattern is empty")
    sub_factories = {n: _build_lane_factory(n) for n in set(expanded)}
    counter = [0]

    def model_factory(dim: int) -> nn.Module:
        if counter[0] >= len(expanded):
            raise RuntimeError(
                f"model_factory called {counter[0] + 1} times but pattern has only {len(expanded)} blocks"
            )
        name = expanded[counter[0]]
        counter[0] += 1
        return sub_factories[name](dim)

    probe_factory = sub_factories[expanded[0]]
    return model_factory, probe_factory


def _build_model_and_batchers(
    *,
    mixer: str,
    pattern: str | None,
    batch_size: int,
    seq_len: int,
    device: str,
    n_eval_batches: int,
    dim: int,
    n_blocks: int,
    use_ffn: bool = True,
) -> tuple[
    nn.Module, Callable[[int], nn.Module], _RandomWindowBatcher, list[torch.Tensor], int
]:
    model_factory, probe_factory = _resolve_lane_factories(mixer, pattern)
    model = _build_tinylm(model_factory, dim=dim, n_blocks=n_blocks, use_ffn=use_ffn)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_tokens, val_tokens, _, _ = _load_wikitext_tokens(
        variant="wikitext-103-raw-v1",
        vocab_size=VOCAB_SIZE,
        max_chars_train=200_000_000,
        max_chars_val=2_000_000,
    )
    train_batcher = _RandomWindowBatcher(
        train_tokens, batch_size=batch_size, seq_len=seq_len, device=device, seed=42
    )
    val_batcher = _RandomWindowBatcher(
        val_tokens, batch_size=batch_size, seq_len=seq_len, device=device, seed=123
    )
    val_batches = val_batcher.fixed_batches(n_eval_batches)
    return model, probe_factory, train_batcher, val_batches, n_params


def _make_writer(jsonl_path: Path) -> Callable[[dict[str, Any]], None]:
    if jsonl_path.exists():
        jsonl_path.unlink()

    def append(row: dict[str, Any]) -> None:
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    return append


def _save_weights(model: nn.Module, output_dir: Path, label: str, step: int) -> Path:
    """Persist model state_dict so future probes can be added without retraining."""
    path = output_dir / f"{label}_step{step:06d}.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": int(step)}, path)
    return path


def _rotate_weight_saves(paths: list[Path], keep_last: int) -> None:
    if keep_last <= 0:
        return
    while len(paths) > keep_last:
        old = paths.pop(0)
        try:
            old.unlink()
        except FileNotFoundError:
            pass


def _make_checkpoint_handler(
    *,
    model: nn.Module,
    factory: Callable[[int], nn.Module],
    val_batches: list[torch.Tensor],
    device: torch.device,
    seed: int,
    final_step: int,
    append: Callable[[dict[str, Any]], None],
    state: dict[str, Any],
    amp: bool,
    amp_dtype: torch.dtype,
    model_dim: int,
) -> Callable[..., None]:
    def on_checkpoint(step: int, *, force_final: bool = False) -> None:
        model.eval()
        t = time.monotonic()
        cheap = _cheap_evals(
            model=model,
            factory=factory,
            val_batches=val_batches,
            device=device,
            seed=seed,
            amp=amp,
            amp_dtype=amp_dtype,
            model_dim=model_dim,
        )
        row: dict[str, Any] = {
            "event": "checkpoint",
            "step": int(step),
            "cheap": cheap,
            "eval_wall_s": round(time.monotonic() - t, 1),
        }
        if force_final or int(step) >= int(final_step):
            t = time.monotonic()
            row["expensive"] = _expensive_evals(model=model, device=device, seed=seed)
            row["expensive_wall_s"] = round(time.monotonic() - t, 1)
        append(row)
        state["last_evals"] = row

    return on_checkpoint


def _make_save_handler(
    *,
    model: nn.Module,
    output_dir: Path,
    label: str,
    append: Callable[[dict[str, Any]], None],
    keep_last: int,
) -> Callable[[int], Path | None]:
    saved: list[Path] = []

    def on_save(step: int) -> Path | None:
        t = time.monotonic()
        path = _save_weights(model, output_dir, label, int(step))
        saved.append(path)
        _rotate_weight_saves(saved, int(keep_last))
        row = {
            "event": "weights_saved",
            "step": int(step),
            "weights_path": str(path),
            "kept_weight_paths": [str(p) for p in saved],
            "wall_s": round(time.monotonic() - t, 2),
        }
        append(row)
        return path

    return on_save


def _make_mid_tier_handler(
    *,
    model: nn.Module,
    device: torch.device,
    append: Callable[[dict[str, Any]], None],
    val_batches: list[torch.Tensor] | None = None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    plateau_tracker: _PlateauTracker | None = None,
) -> Callable[[int], bool]:
    def on_mid_tier(step: int) -> bool:
        model.eval()
        t = time.monotonic()
        mid = _mid_tier_evals(
            model=model,
            device=device,
            val_batches=val_batches,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        row: dict[str, Any] = {
            "event": "mid_tier",
            "step": int(step),
            "mid_tier": mid,
            "wall_s": round(time.monotonic() - t, 1),
        }
        halted = False
        if plateau_tracker is not None:
            ppl = mid.get("wikitext_ppl")
            if ppl is not None:
                halted = plateau_tracker.update(int(step), float(ppl))
                row["plateau"] = {
                    "stale_ticks": int(plateau_tracker.stale_ticks),
                    "best_ppl": (
                        None
                        if plateau_tracker.best_ppl == math.inf
                        else float(plateau_tracker.best_ppl)
                    ),
                    "best_step": plateau_tracker.best_step,
                    "halted": bool(halted),
                }
        append(row)
        return bool(halted)

    return on_mid_tier


def _validate_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    notes_dir = (_REPO / "research" / "notes").resolve()
    if resolved == notes_dir or notes_dir in resolved.parents:
        raise ValueError(
            "mixer_fingerprint writes JSONL/weights; use research/reports/... "
            "or tasks/audit/... instead of research/notes/..."
        )


def _load_resume_weights(
    model: nn.Module, resume_path: Path, device: torch.device
) -> int:
    payload = torch.load(resume_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"resume checkpoint missing model_state_dict: {resume_path}")
    model.load_state_dict(payload["model_state_dict"])
    return int(payload.get("step", 0) or 0)


def _emit_start_event(
    append,
    *,
    mixer,
    pattern,
    n_params,
    dim,
    n_blocks,
    n_steps,
    step_offset,
    resume,
    checkpoint_steps,
    batch_size,
    seq_len,
    seq_len_schedule,
    initial_seq_len,
    curriculum_warmup_steps,
    learning_rate,
    min_lr,
    warmup_steps,
    device_obj,
    amp,
    amp_dtype,
    compile_meta,
    save_weights,
    save_every_steps,
    keep_last_saves,
    mid_tier_every_steps,
    log_every_steps,
    plateau_patience,
    plateau_min_delta,
    plateau_min_steps,
) -> None:
    append(
        {
            "event": "start",
            "mixer": mixer,
            "pattern": pattern,
            "n_params_label": f"{n_params / 1e6:.1f}M",
            "dim": dim,
            "n_blocks": n_blocks,
            "n_params": int(n_params),
            "n_steps": int(n_steps),
            "step_offset": int(step_offset),
            "target_step": int(step_offset) + int(n_steps),
            "resume": str(resume) if resume is not None else None,
            "checkpoint_steps": list(checkpoint_steps),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seq_len_schedule": {
                "name": str(seq_len_schedule),
                "initial_seq_len": int(initial_seq_len),
                "max_seq_len": int(seq_len),
                "warmup_steps": int(curriculum_warmup_steps),
            },
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "warmup_steps": int(warmup_steps),
            "device": str(device_obj),
            "amp": {
                "requested": bool(amp),
                "enabled": bool(amp and device_obj.type == "cuda"),
                "dtype": str(amp_dtype).replace("torch.", ""),
            },
            "compile": compile_meta,
            "save_weights": bool(save_weights),
            "save_every_steps": int(save_every_steps),
            "keep_last_saves": int(keep_last_saves),
            "mid_tier_every_steps": int(mid_tier_every_steps),
            "log_every_steps": int(log_every_steps),
            "plateau": {
                "patience": int(plateau_patience),
                "min_delta": float(plateau_min_delta),
                "min_steps": int(plateau_min_steps),
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )


def _make_callbacks(
    *,
    model,
    factory,
    val_batches,
    device_obj,
    seed,
    checkpoint_steps,
    step_offset,
    append,
    state,
    amp,
    amp_dtype,
    save_weights,
    output_dir,
    label,
    keep_last_saves,
    mid_tier_every_steps,
    plateau_patience,
    plateau_min_delta,
    plateau_min_steps,
    model_dim,
):
    on_checkpoint = _make_checkpoint_handler(
        model=model,
        factory=factory,
        val_batches=val_batches,
        device=device_obj,
        seed=seed,
        final_step=int(step_offset) + sorted(checkpoint_steps)[-1],
        append=append,
        state=state,
        amp=bool(amp),
        amp_dtype=amp_dtype,
        model_dim=int(model_dim),
    )
    on_save = (
        _make_save_handler(
            model=model,
            output_dir=output_dir,
            label=label,
            append=append,
            keep_last=int(keep_last_saves),
        )
        if save_weights
        else None
    )
    plateau_tracker = (
        _PlateauTracker(
            patience=int(plateau_patience),
            min_delta=float(plateau_min_delta),
            min_steps=int(plateau_min_steps),
        )
        if int(mid_tier_every_steps) > 0
        else None
    )
    on_mid_tier = (
        _make_mid_tier_handler(
            model=model,
            device=device_obj,
            append=append,
            val_batches=val_batches,
            amp=bool(amp),
            amp_dtype=amp_dtype,
            plateau_tracker=plateau_tracker,
        )
        if int(mid_tier_every_steps) > 0
        else None
    )
    return on_checkpoint, on_save, on_mid_tier, plateau_tracker


def _setup_run_environment(
    *,
    output_dir: Path,
    seed: int,
    device: str,
    amp_dtype_name: str,
    seq_len_schedule: str,
    compile_model: bool,
    compile_dynamic: bool,
    run_label: str | None,
    mixer: str,
) -> tuple[torch.device, torch.dtype, bool, Path, str, Any]:
    _configure_torch_performance()
    _validate_output_dir(output_dir)
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    amp_dtype = _autocast_dtype(str(amp_dtype_name), device_obj)
    effective_compile_dynamic = bool(compile_dynamic)
    if str(seq_len_schedule).strip().lower() == "growing" and bool(compile_model):
        effective_compile_dynamic = True
    output_dir.mkdir(parents=True, exist_ok=True)
    label = run_label or mixer
    jsonl_path = output_dir / f"{label}.jsonl"
    append = _make_writer(jsonl_path)
    return device_obj, amp_dtype, effective_compile_dynamic, jsonl_path, label, append


def run_fingerprint(
    *,
    mixer: str,
    output_dir: Path,
    n_steps: int = 10_000,
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS,
    batch_size: int = 16,
    seq_len: int = 256,
    learning_rate: float = 3e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 2_000,
    seq_len_schedule: str = "fixed",
    initial_seq_len: int = 16,
    curriculum_warmup_steps: int = 2_000,
    n_eval_batches: int = 32,
    device: str = "cuda",
    seed: int = 0,
    pattern: str | None = None,
    run_label: str | None = None,
    dim: int = 96,
    n_blocks: int = 12,
    save_weights: bool = True,
    save_every_steps: int = 5_000,
    keep_last_saves: int = 3,
    mid_tier_every_steps: int = 10_000,
    amp: bool = True,
    amp_dtype_name: str = "bf16",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    compile_required: bool = False,
    log_every_steps: int = 10,
    resume: Path | None = None,
    plateau_patience: int = 3,
    plateau_min_delta: float = 0.005,
    plateau_min_steps: int = 20_000,
    use_ffn: bool = True,
) -> Path:
    """End-to-end: build → train → checkpoint-eval → final-eval → write JSONL.

    Pure CLI dispatch; the work lives in _setup_run_environment,
    _build_model_and_batchers, _emit_start_event, _make_callbacks,
    _training_loop. The argument list IS the public surface area.
    """
    # guardrail: allow-god-function
    device_obj, amp_dtype, effective_compile_dynamic, jsonl_path, label, append = (
        _setup_run_environment(
            output_dir=output_dir,
            seed=seed,
            device=device,
            amp_dtype_name=amp_dtype_name,
            seq_len_schedule=seq_len_schedule,
            compile_model=compile_model,
            compile_dynamic=compile_dynamic,
            run_label=run_label,
            mixer=mixer,
        )
    )
    model, factory, train_batcher, val_batches, n_params = _build_model_and_batchers(
        mixer=mixer,
        pattern=pattern,
        batch_size=batch_size,
        seq_len=seq_len,
        device=str(device_obj),
        n_eval_batches=n_eval_batches,
        dim=dim,
        n_blocks=n_blocks,
        use_ffn=use_ffn,
    )
    step_offset = 0
    if resume is not None:
        step_offset = _load_resume_weights(model, Path(resume), device_obj)
    loss_wrapper = _TrainLossWrapper(model)
    train_model, compile_meta = _maybe_compile_training_model(
        loss_wrapper,
        enabled=bool(compile_model),
        required=bool(compile_required),
        mode=str(compile_mode),
        fullgraph=bool(compile_fullgraph),
        dynamic=effective_compile_dynamic,
        device=device_obj,
    )
    _emit_start_event(
        append,
        mixer=mixer,
        pattern=pattern,
        n_params=n_params,
        dim=dim,
        n_blocks=n_blocks,
        n_steps=n_steps,
        step_offset=step_offset,
        resume=resume,
        checkpoint_steps=checkpoint_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        seq_len_schedule=seq_len_schedule,
        initial_seq_len=initial_seq_len,
        curriculum_warmup_steps=curriculum_warmup_steps,
        learning_rate=learning_rate,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        device_obj=device_obj,
        amp=amp,
        amp_dtype=amp_dtype,
        compile_meta=compile_meta,
        save_weights=save_weights,
        save_every_steps=save_every_steps,
        keep_last_saves=keep_last_saves,
        mid_tier_every_steps=mid_tier_every_steps,
        log_every_steps=log_every_steps,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        plateau_min_steps=plateau_min_steps,
    )
    state: dict[str, Any] = {"last_evals": {}}
    on_checkpoint, on_save, on_mid_tier, plateau_tracker = _make_callbacks(
        model=model,
        factory=factory,
        val_batches=val_batches,
        device_obj=device_obj,
        seed=seed,
        checkpoint_steps=checkpoint_steps,
        step_offset=step_offset,
        append=append,
        state=state,
        amp=amp,
        amp_dtype=amp_dtype,
        save_weights=save_weights,
        output_dir=output_dir,
        label=label,
        keep_last_saves=keep_last_saves,
        mid_tier_every_steps=mid_tier_every_steps,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        plateau_min_steps=plateau_min_steps,
        model_dim=dim,
    )
    train_meta = _training_loop(
        model,
        train_model,
        train_batcher,
        n_steps=n_steps,
        step_offset=int(step_offset),
        learning_rate=learning_rate,
        min_lr=float(min_lr),
        warmup_steps=int(warmup_steps),
        device=device_obj,
        amp=bool(amp),
        amp_dtype=amp_dtype,
        compile_required=bool(compile_required),
        log_every_steps=int(log_every_steps),
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        seq_len_schedule=str(seq_len_schedule),
        initial_seq_len=int(initial_seq_len),
        curriculum_warmup_steps=int(curriculum_warmup_steps),
        checkpoint_steps=checkpoint_steps,
        on_checkpoint=on_checkpoint,
        save_every_steps=int(save_every_steps),
        on_save=on_save,
        mid_tier_every_steps=int(mid_tier_every_steps),
        on_mid_tier=on_mid_tier,
    )
    train_meta["compile"] = compile_meta
    if plateau_tracker is not None:
        train_meta["plateau_tracker"] = plateau_tracker.to_dict()
    append(
        {"event": "done", "train_meta": train_meta, "last_evals": state["last_evals"]}
    )
    return jsonl_path


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mixer", required=True, type=str)
    p.add_argument(
        "--output",
        default=Path("research/reports/mixer_fingerprint"),
        type=Path,
    )
    p.add_argument("--steps", default=10_000, type=int)
    p.add_argument(
        "--checkpoint-steps",
        default="500,1000,5000,10000",
        type=str,
        help="Comma-separated step counts at which to eval.",
    )
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--seq-len", default=256, type=int)
    p.add_argument(
        "--seq-len-schedule",
        default="fixed",
        choices=["fixed", "growing"],
        help="Training sequence-length schedule. 'growing' ramps from "
        "--initial-seq-len to --seq-len over --curriculum-warmup-steps.",
    )
    p.add_argument(
        "--initial-seq-len",
        default=16,
        type=int,
        help="Initial training sequence length when --seq-len-schedule=growing.",
    )
    p.add_argument(
        "--curriculum-warmup-steps",
        default=2_000,
        type=int,
        help="Number of local optimizer steps for growing seq_len to reach --seq-len.",
    )
    p.add_argument("--learning-rate", default=3e-4, type=float)
    p.add_argument("--min-lr", default=1e-5, type=float)
    p.add_argument("--warmup-steps", default=2_000, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument(
        "--pattern",
        default=None,
        type=str,
        help="Per-block pattern when --mixer=interleaved, e.g. 'conv:6,three_lane:6'.",
    )
    p.add_argument(
        "--run-label",
        default=None,
        type=str,
        help="Override JSONL filename (default: --mixer value).",
    )
    p.add_argument(
        "--resume",
        default=None,
        type=Path,
        help="Load model_state_dict from a prior mixer_fingerprint weight checkpoint.",
    )
    p.add_argument(
        "--dim", default=96, type=int, help="Model dim (default 96 ≈ 10M params)."
    )
    p.add_argument("--n-blocks", default=12, type=int, help="Number of stacked blocks.")


def _add_checkpoint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-save-weights",
        dest="save_weights",
        action="store_false",
        help="Skip saving the final-step model state_dict (default: save to "
        "<output>/<label>_step<N>.pt).",
    )
    p.add_argument(
        "--save-every-steps",
        default=5_000,
        type=int,
        help="Save weights every N training steps, independently of eval checkpoints.",
    )
    p.add_argument(
        "--keep-last-saves",
        default=3,
        type=int,
        help="Rotate weight saves, keeping only the newest N checkpoint files.",
    )
    p.add_argument(
        "--mid-tier-every-steps",
        default=10_000,
        type=int,
        help="Run cheap induction/binding screening every N steps; 0 disables.",
    )


def _add_compile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA autocast during training and lightweight PPL eval.",
    )
    p.add_argument(
        "--amp-dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Autocast dtype. bf16 falls back to fp16 when unsupported.",
    )
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the training forward pass only; evals and saved weights use the eager model.",
    )
    p.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )
    p.add_argument(
        "--compile-fullgraph", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--compile-dynamic", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--compile-required",
        action="store_true",
        help="Exit instead of falling back to eager mode if torch.compile fails.",
    )


def _add_plateau_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--log-every-steps",
        default=10,
        type=int,
        help="Print training progress every N optimizer steps; 0 disables.",
    )
    p.add_argument(
        "--plateau-patience",
        default=3,
        type=int,
        help="Halt training after N consecutive mid-tier ticks without PPL improvement.",
    )
    p.add_argument(
        "--plateau-min-delta",
        default=0.005,
        type=float,
        help="Relative wikitext PPL improvement that resets the plateau counter.",
    )
    p.add_argument(
        "--plateau-min-steps",
        default=20_000,
        type=int,
        help="Suppress plateau halt before this global step (avoid early-warmup false trigger).",
    )
    p.add_argument(
        "--use-ffn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap each lane in TinyLM's outer FFN (norm2+MLP+residual). Set --no-ffn "
        "for lanes that already contain their own FFN internally (e.g. graph-derived "
        "ensemble lanes) to avoid double-FFN, which crashes AR per the 2026-05-19 "
        "sensitivity ablation.",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _add_run_args(p)
    _add_checkpoint_args(p)
    _add_compile_args(p)
    _add_plateau_args(p)
    p.set_defaults(save_weights=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ckpts = tuple(int(s) for s in str(args.checkpoint_steps).split(",") if s.strip())
    path = run_fingerprint(
        mixer=str(args.mixer),
        output_dir=Path(args.output),
        n_steps=int(args.steps),
        checkpoint_steps=ckpts,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        seq_len_schedule=str(args.seq_len_schedule),
        initial_seq_len=int(args.initial_seq_len),
        curriculum_warmup_steps=int(args.curriculum_warmup_steps),
        learning_rate=float(args.learning_rate),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        device=str(args.device),
        seed=int(args.seed),
        pattern=args.pattern,
        run_label=args.run_label,
        resume=args.resume,
        dim=int(args.dim),
        n_blocks=int(args.n_blocks),
        save_weights=bool(args.save_weights),
        save_every_steps=int(args.save_every_steps),
        keep_last_saves=int(args.keep_last_saves),
        mid_tier_every_steps=int(args.mid_tier_every_steps),
        amp=bool(args.amp),
        amp_dtype_name=str(args.amp_dtype),
        compile_model=bool(args.compile),
        compile_mode=str(args.compile_mode),
        compile_fullgraph=bool(args.compile_fullgraph),
        compile_dynamic=bool(args.compile_dynamic),
        compile_required=bool(args.compile_required),
        log_every_steps=int(args.log_every_steps),
        plateau_patience=int(args.plateau_patience),
        plateau_min_delta=float(args.plateau_min_delta),
        plateau_min_steps=int(args.plateau_min_steps),
        use_ffn=bool(args.use_ffn),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
