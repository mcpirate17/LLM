from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm

CURRICULUM_BINDING_PROTOCOL_VERSION = "copy_curriculum_v1"
CURRICULUM_BINDING_DISTANCES = (4, 8, 16, 32)
CURRICULUM_BINDING_STEPS_SCREENING = 400
CURRICULUM_BINDING_STEPS_FULL = 800
CURRICULUM_BINDING_EVAL_SCREENING = 100
CURRICULUM_BINDING_EVAL_FULL = 200
CURRICULUM_BINDING_TRAIN_BATCH_SIZE = 16
CURRICULUM_BINDING_EVAL_BATCH_SIZE = 32


def _module_primary_device(module: nn.Module) -> torch.device | None:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return None


@dataclass(slots=True)
class CurriculumBindingResult:
    auc: float = 0.0
    distance_accuracies: Dict[int, float] | None = None
    status: str = "ok"
    elapsed_ms: float = 0.0
    train_steps: int = 0
    protocol_version: str = CURRICULUM_BINDING_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_auc": self.auc,
            "binding_distance_accuracies": self.distance_accuracies,
            "binding_probe_elapsed_ms": self.elapsed_ms,
            "binding_probe_distances": [4, 8, 16, 32],
            "binding_probe_eval_examples": None,
        }


def _generate_copy_train_batch(
    *,
    batch_size: int,
    seq_len: int,
    distance: int,
    vocab_size: int,
    device: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    seed_tokens = torch.randint(
        1,
        vocab_size,
        (batch_size, distance),
        device=device,
        generator=generator,
    )
    n_repeats = (seq_len + distance - 1) // distance
    return seed_tokens.repeat(1, n_repeats)[:, :seq_len]


@torch.inference_mode()
def _eval_copy_distances(
    model: nn.Module,
    *,
    distances: tuple[int, ...],
    n_eval: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    device: str,
    generator: torch.Generator | None,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for distance in distances:
        if distance <= 0 or distance + 1 >= seq_len:
            out[int(distance)] = 0.0
            continue
        correct = 0
        total = 0
        remaining = n_eval
        while remaining > 0:
            bs = min(batch_size, remaining)
            batch = _generate_copy_train_batch(
                batch_size=bs,
                seq_len=seq_len,
                distance=distance,
                vocab_size=vocab_size,
                device=device,
                generator=generator,
            )
            logits = model(batch)
            preds = logits[:, distance - 1 : seq_len - 1, :vocab_size].argmax(dim=-1)
            targets = batch[:, distance:seq_len]
            correct += preds.eq(targets).sum().item()
            total += targets.numel()
            remaining -= bs
        out[int(distance)] = round(correct / max(total, 1), 4)
    return out


def curriculum_binding_range_profile(
    model: nn.Module,
    *,
    distances: tuple[int, ...] = (4, 8, 16, 32),
    n_train_steps: int = 800,
    n_eval: int = 200,
    train_seq_len: int = 128,
    eval_seq_len: int = 128,
    train_batch_size: int = CURRICULUM_BINDING_TRAIN_BATCH_SIZE,
    eval_batch_size: int = CURRICULUM_BINDING_EVAL_BATCH_SIZE,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int | None = None,
    offload_source_model: bool = False,
) -> CurriculumBindingResult:
    """Train a copy briefly on a distance curriculum, then evaluate exact copy.

    Unlike the zero-shot probe, this is a direct learnability diagnostic.
    Unlike the first adapted probe, it trains only on copied positions and
    cycles through multiple distances so the score is not dominated by a
    single long-distance objective.
    """

    t0 = time.perf_counter()
    result = CurriculumBindingResult(distance_accuracies={})
    valid_distances = tuple(
        int(d) for d in distances if int(d) > 0 and int(d) + 1 < train_seq_len
    )
    if not valid_distances:
        result.status = "no_valid_distances"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    original_device = _module_primary_device(model)
    did_offload_source_model = False
    if offload_source_model:
        if original_device is not None and original_device.type == "cuda":
            model.to("cpu")
            did_offload_source_model = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        probe_model = copy.deepcopy(model)
        probe_device = _module_primary_device(probe_model)
        if probe_device is None or str(probe_device) != device:
            probe_model.to(device)
        probe_model.train()
    except Exception as exc:
        if did_offload_source_model and original_device is not None:
            model.to(original_device)
        result.status = f"copy_failed: {exc}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    vocab_size = int(getattr(probe_model, "vocab_size", 256) or 256)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    optimizer_kwargs = {"lr": lr}
    if device == "cuda" and torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(probe_model.parameters(), **optimizer_kwargs)
    use_autocast = device == "cuda" and torch.cuda.is_available()

    try:
        for step in range(n_train_steps):
            distance = valid_distances[step % len(valid_distances)]
            batch = _generate_copy_train_batch(
                batch_size=train_batch_size,
                seq_len=train_seq_len,
                distance=distance,
                vocab_size=vocab_size,
                device=device,
                generator=generator,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                logits = probe_model(batch)
                pred_logits = logits[:, distance - 1 : train_seq_len - 1, :vocab_size]
                targets = batch[:, distance:train_seq_len]
                loss = F.cross_entropy(
                    pred_logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                )
            if not torch.isfinite(loss):
                result.status = "diverged"
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm(probe_model.parameters(), 1.0)
            optimizer.step()
            result.train_steps = step + 1

        probe_model.eval()
        result.distance_accuracies = _eval_copy_distances(
            probe_model,
            distances=valid_distances,
            n_eval=n_eval,
            seq_len=eval_seq_len,
            batch_size=eval_batch_size,
            vocab_size=vocab_size,
            device=device,
            generator=generator,
        )
        vals = list(result.distance_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4) if vals else 0.0
    except Exception as exc:
        result.status = f"train_failed: {exc}"
    finally:
        del probe_model
        if did_offload_source_model and original_device is not None:
            model.to(original_device)
        if did_offload_source_model and torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
