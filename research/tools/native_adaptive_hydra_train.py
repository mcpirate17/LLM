"""Train native adaptive surprise lanes on HYDRA universal-loader data.

This is intentionally a thin CPU trainer for the component_fab TinyLM stack:
the recurrent surprise-memory math stays in the native C++ extension, and the
HYDRA loader supplies local FineFineWeb / Pleias / small-chat batches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research.defaults import PROJECT_ROOT, VOCAB_SIZE
from research.tools._scaling_lanes import NativeAdaptiveReciprocalSlotDeltaLane
from research.tools.native_gate_floor_utils import (
    DEFAULT_NATIVE_GATE_FLOORS,
    parse_float_csv,
)
from research.tools.native_recip_slot_synthetic_gate_probe import (
    ProbeBatch,
    _make_ar,
    _make_binding,
    _make_induction,
    _make_inline,
)
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm
from research.training._optimizer_muon import MuonOptimizer


LOCAL_MIX_NAME = "codex_ffw60_chat30_pleias10_local"
GATE_AUX_BRANCHES = ("native", "reciprocal", "slot")
GATE_AUX_NATIVE_TARGETS_8 = DEFAULT_NATIVE_GATE_FLOORS


def _load_hydra_loader_module(hydra_root: Path):
    root = str(hydra_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from hydra.data import universal_data_loader as udl  # type: ignore

    return udl


class _TiktokenAdapter:
    """Small callable tokenizer adapter for HYDRA's streaming loader."""

    def __init__(self, name: str = "cl100k_base") -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding(name)
        self.eos_token_id = 100257
        self.eos_token = "<|endoftext|>"
        self.pad_token = self.eos_token

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.enc.encode(str(text), allowed_special="all"))

    def __call__(
        self,
        texts,
        *,
        add_special_tokens: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        padding: bool = False,
        return_attention_mask: bool = False,
    ) -> dict[str, list[list[int]]]:
        del padding, return_attention_mask
        if isinstance(texts, str):
            texts = [texts]
        rows = [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]
        if truncation and max_length is not None:
            rows = [row[: int(max_length)] for row in rows]
        return {"input_ids": rows}


def _ensure_hydra_tokenizer(udl: Any, tokenizer_name: str) -> None:
    tokenizer = udl.get_tokenizer(tokenizer_name)
    if tokenizer is not None:
        return
    adapter = _TiktokenAdapter("cl100k_base")
    udl._TOKENIZER_CACHE[tokenizer_name] = adapter
    udl.get_tokenizer = lambda name="gpt2": udl._TOKENIZER_CACHE.get(name) or adapter


def _register_local_mix(udl: Any) -> None:
    """Register the exact local-data mix requested for this experiment."""
    udl.DATASET_CONFIGS[LOCAL_MIX_NAME] = {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.60},
            {"name": "small_chat_seqaware_flat", "weight": 0.30},
            {"name": "pleias_synth", "weight": 0.10},
        ],
        "description": "Codex local mix: 60% FineFineWeb-local + 30% flat small-chat + 10% Pleias",
    }


def _make_loader(args: argparse.Namespace, *, dataset: str, seed: int):
    udl = _load_hydra_loader_module(args.hydra_root)
    _ensure_hydra_tokenizer(udl, args.tokenizer)
    _register_local_mix(udl)
    loader = udl.create_universal_loader(
        dataset=dataset,
        batch_size=args.batch,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device="cpu",
        tokenizer_name=args.tokenizer,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=seed,
        max_steps=args.steps,
    )
    if args.require_sources and hasattr(loader, "loaders"):
        names = list(getattr(loader, "dataset_names", []))
        loaders = list(getattr(loader, "loaders", []))
        missing = [names[i] for i, child in enumerate(loaders) if child is None]
        if missing:
            if hasattr(loader, "close"):
                loader.close()
            raise RuntimeError(
                f"HYDRA loader missing required sources for {dataset}: {missing}"
            )
    return loader


def _prepare_batch(
    batch: dict[str, torch.Tensor], *, vocab_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = batch["input_ids"].to(dtype=torch.long, device=device)
    labels = batch.get("labels")
    if labels is None:
        labels = ids[:, 1:].clone()
        ids = ids[:, :-1]
    labels = labels.to(dtype=torch.long, device=device)
    if ids.max().item() >= vocab_size:
        ids = torch.remainder(ids, vocab_size)
    valid = labels >= 0
    if bool(valid.any()) and labels[valid].max().item() >= vocab_size:
        labels = labels.clone()
        labels[valid] = torch.remainder(labels[valid], vocab_size)
    return ids.contiguous(), labels.contiguous()


def _lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _parse_gate_aux_probes(raw: str) -> tuple[str, ...]:
    aliases = {"associative": "ar", "associative_recall": "ar", "refinement": "surprise"}
    probes: list[str] = []
    for item in raw.split(","):
        probe = aliases.get(item.strip().lower(), item.strip().lower())
        if not probe:
            continue
        if probe not in {"binding", "ar", "induction", "surprise", "inline"}:
            raise ValueError(f"unsupported --gate-aux-probes entry: {item!r}")
        probes.append("surprise" if probe == "inline" else probe)
    return tuple(dict.fromkeys(probes))


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    return parse_float_csv(raw)


def _gate_aux_native_target(block_idx: int, n_blocks: int) -> float:
    if n_blocks <= 1:
        return GATE_AUX_NATIVE_TARGETS_8[-1]
    table_pos = round(block_idx * (len(GATE_AUX_NATIVE_TARGETS_8) - 1) / (n_blocks - 1))
    return float(GATE_AUX_NATIVE_TARGETS_8[int(table_pos)])


def _gate_aux_target_dist(
    probe: str, block_idx: int, n_blocks: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    native = _gate_aux_native_target(block_idx, n_blocks)
    remaining = max(1.0 - native, 0.0)
    depth = block_idx / max(n_blocks - 1, 1)
    if probe == "induction":
        recip_frac = 0.88 if depth < 0.25 or depth >= 0.75 else 0.78
    elif probe in {"binding", "ar"}:
        recip_frac = 0.45 if 0.25 <= depth < 0.625 else 0.58
    elif probe == "surprise":
        recip_frac = 0.42 if depth < 0.25 else 0.30
    else:
        recip_frac = 0.5
    reciprocal = remaining * recip_frac
    slot = remaining - reciprocal
    return torch.tensor([native, reciprocal, slot], device=device, dtype=dtype)


def _set_native_gate_floors(model: nn.Module, floors: tuple[float, ...]) -> list[float]:
    lanes = [
        module
        for module in model.modules()
        if isinstance(module, NativeAdaptiveReciprocalSlotDeltaLane)
    ]
    if not lanes:
        return []
    assigned: list[float] = []
    for block_idx, lane in enumerate(lanes):
        floor = floors[round(block_idx * (len(floors) - 1) / max(len(lanes) - 1, 1))]
        lane.native_gate_floor = float(floor)
        assigned.append(float(floor))
    return assigned


def _make_gate_aux_probe_batch(
    probe: str, *, batch: int, difficulty: int, device: torch.device
) -> ProbeBatch:
    if probe == "binding":
        return _make_binding(batch, difficulty, device)
    if probe == "ar":
        return _make_ar(batch, difficulty, device)
    if probe == "induction":
        return _make_induction(batch, difficulty, device)
    if probe == "surprise":
        return _make_inline(batch, difficulty, device)
    raise ValueError(f"unsupported gate aux probe: {probe}")


def _mean_gate_aux_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_probe: dict[str, list[float]] = {}
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_probe.setdefault(str(row["probe"]), []).append(float(row["aux_loss"]))
        by_block.setdefault(str(row["block"]), []).append(row)
    return {
        "by_probe": {
            probe: round(sum(vals) / len(vals), 6) for probe, vals in sorted(by_probe.items())
        },
        "by_block": {
            block: {
                "raw_gate_mean": [
                    round(sum(float(r["raw_gate_mean"][i]) for r in block_rows) / len(block_rows), 5)
                    for i in range(3)
                ],
                "effective_gate_mean": [
                    round(
                        sum(float(r["effective_gate_mean"][i]) for r in block_rows)
                        / len(block_rows),
                        5,
                    )
                    for i in range(3)
                ],
                "gate_entropy": round(
                    sum(float(r["gate_entropy"]) for r in block_rows) / len(block_rows), 5
                ),
                "weighted_branch_rms": [
                    round(
                        sum(float(r["weighted_branch_rms"][i]) for r in block_rows)
                        / len(block_rows),
                        5,
                    )
                    for i in range(3)
                ],
            }
            for block, block_rows in sorted(by_block.items(), key=lambda kv: int(kv[0]))
        },
    }


def _gate_aux_loss(
    model: nn.Module, args: argparse.Namespace, step: int
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if (
        args.gate_aux_every <= 0
        or args.gate_aux_weight <= 0.0
        or step < args.gate_aux_start_step
        or step % args.gate_aux_every != 0
    ):
        return None, None
    lanes = [
        module
        for module in model.modules()
        if isinstance(module, NativeAdaptiveReciprocalSlotDeltaLane)
    ]
    probes = _parse_gate_aux_probes(args.gate_aux_probes)
    if not lanes or not probes:
        return None, None

    handles = []
    captured: list[tuple[int, torch.Tensor]] = []
    for block_idx, lane in enumerate(lanes):
        handles.append(
            lane.gate.register_forward_hook(
                lambda _module, _inputs, output, idx=block_idx: captured.append((idx, output))
            )
        )

    device = torch.device(args.device)
    losses: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    try:
        for probe in probes:
            for difficulty in range(1, args.gate_aux_max_batches + 1):
                batch = _make_gate_aux_probe_batch(
                    probe, batch=args.gate_aux_batch, difficulty=difficulty, device=device
                )
                captured.clear()
                model(batch.ids)
                for block_idx, gate_logits in captured:
                    raw_gate = torch.softmax(gate_logits, dim=-1)
                    target = _gate_aux_target_dist(
                        probe,
                        block_idx,
                        len(lanes),
                        device=gate_logits.device,
                        dtype=gate_logits.dtype,
                    )
                    block_loss = -(target * (raw_gate + 1e-8).log()).sum(dim=-1).mean()
                    losses.append(block_loss)
                    metrics = getattr(lanes[block_idx], "last_gate_metrics", {})
                    rows.append(
                        {
                            "probe": probe,
                            "difficulty": difficulty,
                            "block": block_idx,
                            "target_dist": [round(float(v), 5) for v in target.detach().cpu()],
                            "aux_loss": float(block_loss.detach().cpu()),
                            "raw_gate_mean": [
                                round(float(v), 5)
                                for v in metrics.get("raw_gate_mean", torch.zeros(3))
                            ],
                            "effective_gate_mean": [
                                round(float(v), 5)
                                for v in metrics.get("effective_gate_mean", torch.zeros(3))
                            ],
                            "gate_entropy": round(float(metrics.get("gate_entropy", 0.0)), 5),
                            "weighted_branch_rms": [
                                round(float(v), 5)
                                for v in metrics.get("weighted_branch_rms", torch.zeros(3))
                            ],
                        }
                    )
    finally:
        for handle in handles:
            handle.remove()

    if not losses:
        return None, None
    aux = torch.stack(losses).mean()
    return aux, {
        "event": "gate_aux",
        "step": step,
        "weight": args.gate_aux_weight,
        "branches": GATE_AUX_BRANCHES,
        "loss": round(float(aux.detach().cpu()), 6),
        "probes": probes,
        "rows": rows,
        "summary": _mean_gate_aux_rows(rows),
    }


def _adaptive_depth_stats(model: nn.Module) -> dict[str, Any] | None:
    depths: list[torch.Tensor] = []
    for module in model.modules():
        depth = getattr(module, "last_depth_counts", None)
        if depth is not None:
            depths.append(depth.detach().cpu().reshape(-1))
    if not depths:
        mor = [
            float(m.last_mean_depth)
            for m in model.modules()
            if getattr(m, "last_mean_depth", None) is not None
        ]
        if mor:
            hists = [
                m.last_depth_hist
                for m in model.modules()
                if getattr(m, "last_depth_hist", None) is not None
            ]
            stats: dict[str, Any] = {
                "mean_depth": round(sum(mor) / len(mor), 4),
                "skip_fraction": 0.0,
                "max_depth": 0,
                "router": "mor_soft",
            }
            if hists:
                n_d = len(hists[0])
                avg = [sum(h[i] for h in hists) / len(hists) for i in range(n_d)]
                stats["histogram_fraction"] = {
                    str(i + 1): round(avg[i], 4) for i in range(n_d)
                }
            return stats
        return None
    d = torch.cat(depths).float()
    if d.numel() == 0:
        return None
    depth_int = d.to(torch.long)
    max_depth = int(depth_int.max().item())
    counts = torch.bincount(depth_int, minlength=max_depth + 1)
    total = float(depth_int.numel())
    return {
        "mean_depth": round(float(d.mean().item()), 4),
        "skip_fraction": round(float((d == 0).float().mean().item()), 4),
        "max_depth": max_depth,
        "histogram": {str(i): int(counts[i].item()) for i in range(counts.numel())},
        "histogram_fraction": {
            str(i): round(float(counts[i].item()) / total, 4)
            for i in range(counts.numel())
        },
    }


@torch.no_grad()
def _eval_loss(
    model: nn.Module, loader, *, vocab_size: int, n_batches: int, device: str = "cpu"
) -> dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    for _ in range(n_batches):
        batch = next(loader)
        ids, labels = _prepare_batch(batch, vocab_size=vocab_size, device=device)
        logits = model(ids)
        loss = _lm_loss(logits, labels)
        if torch.isfinite(loss):
            total += float(loss.item())
            n += 1
    if n == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    mean = total / n
    return {
        "loss": round(mean, 4),
        "ppl": round(float(torch.exp(torch.tensor(mean)).item()), 4),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _classify_muon_params(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Hybrid-Muon split: large 2D hidden matrices -> Muon; everything else
    (embeddings/tied head, 1D norms/biases, scalars, vector-shaped gate
    projections, AND the MoR halt-router heads) -> AdamW.

    Muon's update is the Newton-Schulz orthogonalization of the gradient — a
    fixed-magnitude (gradient-INDEPENDENT) step calibrated for full-rank hidden
    matrices. On a degenerate, effectively rank-1 matrix (any param with a
    dimension of 1, e.g. a ``Linear(dim, 1)`` gate's ``(1, dim)`` weight) that
    orthogonalization is ill-conditioned and marches the weight norm up
    unboundedly, gradient regardless, until the forward overflows fp32 -> NaN.
    This is what killed two long runs:
      * 2026-06-03: the MoR ``halt_head`` heads (carved out by name below).
      * 2026-06-04: ``lane_b.write_gate`` ``(1, 512)`` reached |w|~108 (vs ~1 for
        its siblings) and overflowed at step ~43645. The by-name halt_head carve
        missed the structurally identical gate vectors -> hence the
        ``min(p.shape) == 1`` rule, which routes ALL vector-shaped matrices
        (write/forget/blend gates) to AdamW. No genuine hidden matrix ever has a
        dimension of 1, so this excludes only degenerate cases.
    Small/special/vector params on AdamW is standard Muon practice."""
    force_adamw_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                force_adamw_ids.add(id(p))
    try:
        from component_fab.generator.mor_bilane import MoRLaneA

        for module in model.modules():
            if isinstance(module, MoRLaneA):
                for p in module.halt_head.parameters():
                    force_adamw_ids.add(id(p))
    except ImportError:
        pass
    muon: list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []
    seen: set[int] = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in force_adamw_ids or p.ndim < 2 or min(p.shape) == 1:
            adamw.append(p)
        else:
            muon.append(p)
    return muon, adamw


def _build_optimizers(
    model: nn.Module, args: argparse.Namespace
) -> list[torch.optim.Optimizer]:
    """AdamW only, or hybrid Muon(2D) + AdamW(embedding/head/1D)."""
    if args.optimizer != "muon":
        return [
            torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        ]
    muon_params, adamw_params = _classify_muon_params(model)
    opts: list[torch.optim.Optimizer] = []
    if muon_params:
        opts.append(
            MuonOptimizer(
                muon_params,
                lr=args.muon_lr,
                weight_decay=args.weight_decay,
                momentum=args.muon_momentum,
                ns_steps=args.ns_steps,
            )
        )
    if adamw_params:
        opts.append(
            torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay)
        )
    return opts


def _lr_multiplier(step: int, *, warmup: int, total: int, min_frac: float) -> float:
    """Linear warmup, then cosine decay from peak (1.0) to ``min_frac`` of peak.

    Defaults (warmup=0, min_frac=1.0) make this a flat 1.0 -> no schedule.
    """
    if warmup > 0 and step <= warmup:
        return step / float(warmup)
    if total <= warmup:
        return 1.0
    progress = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_frac + (1.0 - min_frac) * cosine


def _save_checkpoint(
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    args: argparse.Namespace,
    step: int,
) -> Path:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_lane = args.lane.replace("/", "_")
    path = args.checkpoint_dir / f"{args.run_label}_{safe_lane}_step{step:06d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": int(step),
            "lane": args.lane,
            "dataset": args.dataset,
            "dim": args.dim,
            "n_blocks": args.n_blocks,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "optimizer_state_dicts": [o.state_dict() for o in optimizers],
        },
        path,
    )
    return path


def _track_nonfinite(loss: float, run_count: int, limit: int, step: int) -> int:
    """Update the consecutive non-finite counter; raise on true divergence.

    Returns the new consecutive-count (0 when ``loss`` is finite). A NaN loss
    means ``_train_step`` skipped the update; tolerate transient skips but abort
    loudly once more than ``limit`` occur back-to-back.
    """
    if loss == loss:  # finite
        return 0
    run_count += 1
    if run_count > limit:
        raise RuntimeError(
            f"{run_count} consecutive non-finite losses ending at step {step} — "
            "true divergence, not a transient; aborting."
        )
    return run_count


def _batch_stats(ids: torch.Tensor) -> dict[str, Any]:
    """Token stats of a batch — to test whether a grad spike is the model reacting
    to *bad/degenerate data* (a near-single-token batch, a huge repeat, OOV) vs
    intrinsic instability. High top_tok_frac / low uniq_frac ⇒ degenerate batch."""
    flat = ids.detach().reshape(-1)
    n = int(flat.numel())
    vals, counts = torch.unique(flat, return_counts=True)
    top_i = int(counts.argmax())
    # longest run of an identical token (repeat detection), per row then max
    max_run = 0
    for row in ids.detach():
        run, prev = 1, None
        for t in row.tolist():
            run = run + 1 if t == prev else 1
            prev = t
            max_run = max(max_run, run)
    return {
        "n_tok": n,
        "uniq_frac": round(int(vals.numel()) / max(n, 1), 4),
        "top_tok": int(vals[top_i]),
        "top_tok_frac": round(int(counts[top_i]) / max(n, 1), 4),
        "max_run": max_run,
        "tok_min": int(flat.min()),
        "tok_max": int(flat.max()),
    }


def _grad_component_norms(model: nn.Module) -> dict[str, Any]:
    """L2 grad-norm grouped by component + the single largest-grad parameter, for
    diagnosing gradient spikes (which part of the model produced the big norm)."""
    groups: dict[str, float] = {}
    max_name, max_val = "", 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = float(p.grad.detach().norm())
        if n > max_val:
            max_val, max_name = n, name
        key = (
            "router"
            if "halt_head" in name
            else "lane_a"
            if "lane_a" in name
            else "lane_b"
            if "lane_b" in name
            else "attn"
            if "attn" in name
            else "ffn"
            if ("mlp" in name or "ffn" in name or "swiglu" in name)
            else "embed"
            if ("embed" in name or "wte" in name or "tok" in name)
            else "other"
        )
        groups[key] = math.hypot(groups.get(key, 0.0), n)
    return {
        "by_component": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
        "max_param": [max_name, max_val],
    }


def _cautious_step(optimizers, base_lrs, mult) -> None:
    """Cautious update (Liang et al. 2024, arXiv:2411.16085): run each
    optimizer's step, then keep only the coordinates whose realized update
    descends the loss — i.e. the param delta sign-agrees with -grad — and
    renormalize by the kept fraction to preserve mean update magnitude.

    Masks the *realized* param delta (post-step minus pre-step), so it composes
    with ANY base optimizer (Muon, AdamW) and their decoupled weight decay
    without touching their internals. Opt-in via --cautious; a near-free, widely
    reported small speedup. One transient param-sized snapshot is the only cost.
    """
    snaps = [
        (p, p.detach().clone(), p.grad)
        for opt in optimizers
        for group in opt.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    for opt, bases in zip(optimizers, base_lrs):
        for group, base in zip(opt.param_groups, bases):
            group["lr"] = base * mult
        opt.step()
    for p, p0, g in snaps:
        delta = p.data - p0
        keep = (delta * g < 0).to(p.dtype)  # coords where the step descends
        scale = keep.numel() / keep.sum().clamp_min(1.0)
        p.data.copy_(p0 + delta * keep * scale)


def _train_step(model, optimizers, base_lrs, ids, labels, args, step):
    """One forward/backward/opt step with warmup-cosine LR. Returns (loss, grad, lr).

    On a non-finite loss the optimizer step is SKIPPED (grads zeroed, no update)
    and a NaN loss is returned so the caller can count it. A single transient
    bad microbatch / Muon hiccup must not kill a multi-hour run; the run loop
    fails loud only if too many *consecutive* non-finite steps occur (true
    divergence), per ``--max-consecutive-nonfinite``.
    """
    model.train()
    mult = _lr_multiplier(
        step, warmup=args.warmup_steps, total=args.steps, min_frac=args.min_lr_frac
    )
    lm_loss = _lm_loss(model(ids), labels)
    gate_aux, gate_aux_info = _gate_aux_loss(model, args, step)
    loss = lm_loss if gate_aux is None else lm_loss + args.gate_aux_weight * gate_aux
    args._last_gate_aux = gate_aux_info
    if gate_aux_info is not None:
        gate_aux_info["lm_loss_before_aux"] = round(float(lm_loss.detach().cpu()), 6)
        gate_aux_info["loss_after_aux"] = round(float(loss.detach().cpu()), 6)
        _append_jsonl(args.out, gate_aux_info)
    if not torch.isfinite(loss):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        print(
            f"[WARN] non-finite loss at step {step}: {loss} — skipping optimizer step",
            flush=True,
        )
        return float("nan"), float("nan"), base_lrs[0][0] * mult
    from component_fab.generator.mor_bilane import collect_ponder_cost

    ponder = collect_ponder_cost(model)
    if ponder is not None:
        loss = loss + ponder
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    # Per-OPTIMIZER-GROUP gradient clipping. The AdamW group (token embedding /
    # head — 100k-vocab, sparse, naturally spiky grads) would, under a single
    # GLOBAL clip, scale down the Muon-trained hidden matrices and the MoR router
    # too whenever it spikes (~5% of steps) — starving the rest of the model and
    # plateauing the loss. Clipping each group independently decouples them.
    group_norms = []
    for opt in optimizers:
        ps = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
        if ps:
            gn = torch.nn.utils.clip_grad_norm_(ps, args.grad_clip)
            group_norms.append(float(gn.item() if torch.is_tensor(gn) else gn))
    grad = max(group_norms) if group_norms else 0.0  # worst group, for log/guard
    # Gradient guard: a finite loss can still produce a non-finite *gradient*
    # (numerical edge in the backward on a hard batch); clip_grad_norm_ passes the
    # NaN/Inf through, so stepping would poison the weights and cascade. Skip the
    # step instead — grads zeroed, weights preserved, training continues on the
    # next batch. (Standard practice for unstable large-model training.)
    if not all(math.isfinite(g) for g in group_norms):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        bs = _batch_stats(ids)
        _append_jsonl(
            args.out,
            {
                "event": "grad_nonfinite",
                "step": step,
                "grad_norm": str(grad),
                "loss": round(float(loss.item()), 4),
                "batch": bs,
            },
        )
        print(
            f"[WARN] non-finite grad-norm at step {step}: {grad} — skipping step "
            f"(weights preserved); batch={bs}",
            flush=True,
        )
        return float("nan"), float("nan"), base_lrs[0][0] * mult
    # Observability: dissect a gradient spike — which component spiked, whether the
    # clip engaged, the data batch's token stats (MoR vs bad-data), and MoR depth.
    spike_thr = getattr(args, "grad_spike_threshold", 0.0)
    if spike_thr > 0 and grad > spike_thr:
        det = _grad_component_norms(model)  # grads are post-clip here
        rs = grad / args.grad_clip if grad > args.grad_clip else 1.0  # -> pre-clip
        _append_jsonl(
            args.out,
            {
                "event": "grad_spike",
                "step": step,
                "grad_norm": round(grad, 3),
                "clip_ratio": round(min(1.0, args.grad_clip / (grad + 1e-12)), 4),
                "loss": round(float(loss.item()), 4),
                "by_component": {
                    k: round(v * rs, 3) for k, v in det["by_component"].items()
                },
                "max_param": [det["max_param"][0], round(det["max_param"][1] * rs, 3)],
                "batch": _batch_stats(ids),
                "mor": _adaptive_depth_stats(model),
            },
        )
    if getattr(args, "cautious", False):
        _cautious_step(optimizers, base_lrs, mult)
    else:
        for opt, bases in zip(optimizers, base_lrs):
            for group, base in zip(opt.param_groups, bases):
                group["lr"] = base * mult
            opt.step()
    return float(loss.item()), grad, base_lrs[0][0] * mult


def _eval_only_row(args, model, n_params, loaded_step, started):
    """Build the eval_only result row (loads a fresh val loader, closes it)."""
    val_loader = _make_loader(args, dataset=args.val_dataset, seed=args.seed + 1009)
    row = {
        "event": "eval_only",
        "run_label": args.run_label,
        "lane": args.lane,
        "checkpoint": str(args.load_checkpoint)
        if args.load_checkpoint is not None
        else None,
        "loaded_step": loaded_step,
        "val_dataset": args.val_dataset,
        "dim": args.dim,
        "n_blocks": args.n_blocks,
        "params": n_params,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "eval_batches": args.eval_batches,
        "eval": _eval_loss(
            model,
            val_loader,
            vocab_size=args.vocab_size,
            n_batches=args.eval_batches,
            device=args.device,
        ),
        "depth": _adaptive_depth_stats(model),
        "elapsed_sec": round(time.time() - started, 2),
    }
    if hasattr(val_loader, "close"):
        val_loader.close()
    return row


def _weight_health(model: nn.Module) -> tuple[float, str]:
    """Largest |weight| in the model + its name. Proactive NaN early-warning:
    the 2026-06-04 death was a gate weight marching to |w|~108 before overflow,
    so surfacing max|w| every log step makes that visible long before the NaN."""
    mx, name = 0.0, ""
    for n, p in model.named_parameters():
        a = float(p.detach().abs().max())
        if a > mx:
            mx, name = a, n
    return mx, name


def _fmt_step_line(row: dict[str, Any], grad_comp: dict[str, Any]) -> str:
    """Dense one-line console summary (the jsonl keeps the full structured row)."""
    d = row.get("depth") or {}
    hist = d.get("histogram_fraction") or {}
    hist_s = " ".join(f"{hist[k]:.2f}" for k in sorted(hist)) if hist else "-"
    comp = grad_comp.get("by_component", {})
    comp_s = " ".join(f"{k} {v:.2f}" for k, v in list(comp.items())[:3]) or "-"
    ev = row.get("eval")
    ev_s = f" ppl {ev['ppl']:>7.1f}" if ev else ""
    wmax, wname = row.get("w_max", 0.0), row.get("w_max_param", "")
    wshort = wname.replace(".weight", "").replace("blocks.", "b")
    return (
        f"step {row['step']:>6} | loss {row['loss']:.3f}{ev_s} | "
        f"grad {row['grad_norm']:.2f} [{comp_s}] | lr {row['lr']:.2e} | "
        f"depth {d.get('mean_depth', 0):.2f} [{hist_s}] | "
        f"w|max| {wmax:.1f} {wshort} | {row.get('tok_per_s', 0):.0f} tok/s"
    )


def _record_step(args, model, train_loader, val_loader, started, step, metrics):
    """Append a log/eval row when due. metrics = (loss, grad, lr). Returns the row or None."""
    last_loss, last_grad, cur_lr = metrics
    should_log = step == 1 or step % args.log_every == 0 or step == args.steps
    should_eval = step % args.eval_every == 0 or step == args.steps
    if not (should_log or should_eval):
        return None
    now = time.time()
    # Throughput since the previous log (tokens = batch * seq_len * steps_elapsed).
    t_prev = getattr(args, "_t_prev", started)
    s_prev = getattr(args, "_s_prev", step - 1)
    dt = max(now - t_prev, 1e-6)
    tok_per_s = (step - s_prev) * args.batch * args.seq_len / dt
    args._t_prev, args._s_prev = now, step
    # Grads are still live here (zeroed only at the next step) -> per-component
    # breakdown (post-clip) shows WHERE the gradient mass sits.
    grad_comp = _grad_component_norms(model)
    w_max, w_max_param = _weight_health(model)
    row: dict[str, Any] = {
        "event": "step",
        "step": step,
        "loss": round(last_loss, 4),
        "grad_norm": round(last_grad, 4),
        "grad_by_component": {
            k: round(v, 3) for k, v in grad_comp["by_component"].items()
        },
        "grad_max_param": [
            grad_comp["max_param"][0],
            round(grad_comp["max_param"][1], 3),
        ],
        "w_max": round(w_max, 3),
        "w_max_param": w_max_param,
        "lr": round(cur_lr, 8),
        "tok_per_s": round(tok_per_s, 1),
        "elapsed_sec": round(now - started, 2),
        "depth": _adaptive_depth_stats(model),
        "loader_stats": getattr(train_loader, "stats", lambda: {})(),
    }
    gate_aux = getattr(args, "_last_gate_aux", None)
    if gate_aux is not None and int(gate_aux.get("step", -1)) == step:
        row["gate_aux"] = {
            "weight": gate_aux["weight"],
            "loss": gate_aux["loss"],
            "lm_loss_before_aux": gate_aux["lm_loss_before_aux"],
            "loss_after_aux": gate_aux["loss_after_aux"],
            "summary": gate_aux["summary"],
        }
    if should_eval:
        row["eval"] = _eval_loss(
            model,
            val_loader,
            vocab_size=args.vocab_size,
            n_batches=args.eval_batches,
            device=args.device,
        )
    _append_jsonl(args.out, row)
    print(_fmt_step_line(row, grad_comp), flush=True)
    return row


def _start_row(args, n_params, loaded_step, first_step, train_loader) -> dict[str, Any]:
    """Build the run's 'start' provenance row."""
    return {
        "event": "start",
        "run_label": args.run_label,
        "lane": args.lane,
        "load_checkpoint": str(args.load_checkpoint)
        if args.load_checkpoint is not None
        else None,
        "loaded_step": loaded_step,
        "first_step": first_step,
        "dataset": args.dataset,
        "val_dataset": args.val_dataset,
        "dim": args.dim,
        "n_blocks": args.n_blocks,
        "params": n_params,
        "steps": args.steps,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
        "warmup_steps": args.warmup_steps,
        "min_lr_frac": args.min_lr_frac,
        "vocab_size": args.vocab_size,
        "train_loader_stats": getattr(train_loader, "stats", lambda: {})(),
        "native_gate_floors": getattr(args, "_native_gate_floors", None),
        "gate_aux": {
            "every": args.gate_aux_every,
            "weight": args.gate_aux_weight,
            "probes": _parse_gate_aux_probes(args.gate_aux_probes),
            "start_step": args.gate_aux_start_step,
            "max_batches": args.gate_aux_max_batches,
            "batch": args.gate_aux_batch,
            "native_targets_8": GATE_AUX_NATIVE_TARGETS_8,
        },
    }


def _load_checkpoint(
    model: nn.Module, args: argparse.Namespace
) -> tuple[dict[str, Any] | None, int | None]:
    """Load --load-checkpoint into model. Under --load-nonstrict, tolerate the
    new MoR ``halt_head`` (fail loud on any other mismatch), deep-start re-init
    the router, and reset the optimizer (handled by the caller skipping its load)."""
    if args.load_checkpoint is None:
        return None, None
    payload = torch.load(args.load_checkpoint, map_location="cpu")  # nosec B614 - local experiment checkpoint
    res = model.load_state_dict(
        payload["model_state_dict"], strict=not args.load_nonstrict
    )
    if args.load_nonstrict:
        missing = list(getattr(res, "missing_keys", []))
        unexpected = list(getattr(res, "unexpected_keys", []))
        # Allow any fresh halting param (the MLP halt_head AND the surprise lane's
        # halt_surprise_coupling) to be missing; anything else is a real mismatch.
        if unexpected or any("halt" not in key for key in missing):
            raise RuntimeError(
                "--load-nonstrict resume mismatch beyond halt params: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        from component_fab.generator.mor_bilane import apply_resume_init

        n_lanes = apply_resume_init(model)
        print(
            f"[resume] non-strict: {len(missing)} fresh halt params; deep-start "
            f"re-init on {n_lanes} MoR lanes; optimizer state reset",
            flush=True,
        )
    return payload, int(payload.get("step", 0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    factory = _build_lane_factory(args.lane)
    model = _build_tinylm(
        factory,
        dim=args.dim,
        n_blocks=args.n_blocks,
        vocab_size=args.vocab_size,
        max_seq_len=max(args.seq_len, 1024),
        use_ffn=True,
    ).to(args.device)
    if args.native_gate_floors:
        args._native_gate_floors = _set_native_gate_floors(
            model, _parse_float_csv(args.native_gate_floors)
        )
    else:
        args._native_gate_floors = None
    n_params = sum(p.numel() for p in model.parameters())
    payload, loaded_step = _load_checkpoint(model, args)
    if args.ponder_weight is not None:
        from component_fab.generator.mor_bilane import set_ponder_weight

        set_ponder_weight(model, args.ponder_weight)

    if getattr(args, "freeze_router", False):
        from component_fab.generator.mor_bilane import MoRLaneA

        n_frozen = 0
        for module in model.modules():
            if isinstance(module, MoRLaneA):
                for p in module.halt_head.parameters():
                    p.requires_grad_(False)
                    n_frozen += 1
        print(
            f"[freeze-router] froze {n_frozen} halt-head params -> fixed depth "
            "(no adaptive confound for the capability ablation)",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.append:
        args.out.unlink()

    started = time.time()
    if args.eval_only:
        eval_row = _eval_only_row(args, model, n_params, loaded_step, started)
        _append_jsonl(args.out, eval_row)
        print(json.dumps(eval_row, default=str), flush=True)
        return eval_row

    train_loader = _make_loader(args, dataset=args.dataset, seed=args.seed)
    val_loader = _make_loader(args, dataset=args.val_dataset, seed=args.seed + 1009)
    optimizers = _build_optimizers(model, args)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    if (
        payload is not None
        and payload.get("optimizer_state_dicts") is not None
        and not args.load_nonstrict
    ):
        for opt, sd in zip(optimizers, payload["optimizer_state_dicts"]):
            if sd is not None:
                opt.load_state_dict(sd)
    first_step = 1
    if loaded_step is not None and not args.restart_step:
        if loaded_step >= args.steps:
            raise RuntimeError(
                f"checkpoint step {loaded_step} is already >= requested target --steps {args.steps}"
            )
        first_step = loaded_step + 1

    start_row = _start_row(args, n_params, loaded_step, first_step, train_loader)
    _append_jsonl(args.out, start_row)
    print(json.dumps(start_row, default=str), flush=True)

    last_loss = float("nan")
    last_grad = float("nan")
    nonfinite_run = 0
    checkpoints: list[str] = []
    # Seed the recovery fallback with the resume checkpoint so a divergence before
    # the first in-run save can still reload (and skip the bad window) rather than
    # abort — without this, a fresh resume has nothing to recover to.
    last_ckpt: str | None = (
        str(args.load_checkpoint) if args.load_checkpoint is not None else None
    )
    recoveries = 0
    for step in range(first_step, args.steps + 1):
        try:
            if hasattr(train_loader, "set_step"):
                train_loader.set_step(step)
            batch = next(train_loader)
            ids, labels = _prepare_batch(
                batch, vocab_size=args.vocab_size, device=args.device
            )
            metrics = _train_step(model, optimizers, base_lrs, ids, labels, args, step)
            last_loss, last_grad, _ = metrics
            nonfinite_run = _track_nonfinite(
                last_loss, nonfinite_run, args.max_consecutive_nonfinite, step
            )
            _record_step(args, model, train_loader, val_loader, started, step, metrics)

            if args.save_every and (step % args.save_every == 0 or step == args.steps):
                path = str(_save_checkpoint(model, optimizers, args, step))
                checkpoints.append(path)
                last_ckpt = path
        except RuntimeError as exc:
            # Auto-recover from a rare-transient divergence: reload the last good
            # checkpoint (discards the corrupting step), reset the optimizer, skip
            # the bad data window, and continue. Re-raise if no ckpt / budget out.
            if (
                "non-finite" not in str(exc)
                or last_ckpt is None
                or recoveries >= args.max_recoveries
            ):
                raise
            recoveries += 1
            payload = torch.load(last_ckpt, map_location=args.device)  # nosec B614
            model.load_state_dict(payload["model_state_dict"])
            optimizers = _build_optimizers(model, args)
            base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]
            nonfinite_run = 0
            _append_jsonl(
                args.out,
                {
                    "event": "recover",
                    "step": step,
                    "from_ckpt": Path(last_ckpt).name,
                    "weights_step": int(payload.get("step", -1)),
                    "recovery": recoveries,
                },
            )
            print(
                f"[recover {recoveries}/{args.max_recoveries}] divergence near step "
                f"{step}; reloaded {Path(last_ckpt).name} (weights @ step "
                f"{payload.get('step')}), optimizer reset, skipping bad window.",
                flush=True,
            )

    if hasattr(train_loader, "close"):
        train_loader.close()
    if hasattr(val_loader, "close"):
        val_loader.close()

    done = {
        "event": "done",
        "run_label": args.run_label,
        "lane": args.lane,
        "steps": args.steps,
        "last_loss": round(last_loss, 4),
        "last_grad_norm": round(last_grad, 4),
        "elapsed_sec": round(time.time() - started, 2),
        "checkpoints": checkpoints,
    }
    _append_jsonl(args.out, done)
    print(json.dumps(done, default=str), flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--dataset", default=LOCAL_MIX_NAME)
    ap.add_argument("--val-dataset", default=LOCAL_MIX_NAME)
    ap.add_argument("--hydra-root", type=Path, default=PROJECT_ROOT / "HYDRA")
    ap.add_argument("--run-label", default="native_adaptive_hydra")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument(
        "--cautious",
        action="store_true",
        help="Cautious update (Liang et al. 2024): mask each step to coordinates "
        "that descend the loss, renormalized. Composes with adamw/muon; opt-in, "
        "near-free, a small reported speedup.",
    )
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument(
        "--min-lr-frac",
        type=float,
        default=1.0,
        help="Cosine-decay floor as a fraction of peak LR (1.0 = no decay).",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--native-gate-floors",
        default="",
        help="Comma-separated native branch effective-floor table assigned across "
        "native_adaptive_reciprocal_slot_delta blocks. Empty keeps the lane default.",
    )
    ap.add_argument(
        "--gate-aux-every",
        type=int,
        default=0,
        help="If >0, run the raw branch-gate auxiliary loss every N steps.",
    )
    ap.add_argument(
        "--gate-aux-weight",
        type=float,
        default=0.0,
        help="Weight for the raw pre-floor gate auxiliary loss. Suggested first pass: 0.005-0.02.",
    )
    ap.add_argument(
        "--gate-aux-probes",
        default="binding,induction,surprise",
        help="Comma-separated gate aux probes: binding, ar, induction, surprise.",
    )
    ap.add_argument(
        "--gate-aux-start-step",
        type=int,
        default=0,
        help="Do not apply gate aux before this absolute training step.",
    )
    ap.add_argument(
        "--gate-aux-max-batches",
        type=int,
        default=1,
        help="Number of tiny synthetic difficulty batches per selected aux probe.",
    )
    ap.add_argument(
        "--gate-aux-batch",
        type=int,
        default=1,
        help="Batch size for each synthetic gate aux probe batch.",
    )
    ap.add_argument(
        "--grad-spike-threshold",
        type=float,
        default=0.0,
        help="If >0, log a detailed 'grad_spike' record (per-component grad, clip "
        "ratio, batch token stats, MoR depth) whenever the pre-clip grad-norm "
        "exceeds this. For diagnosing whether spikes are MoR vs bad data.",
    )
    ap.add_argument(
        "--max-consecutive-nonfinite",
        type=int,
        default=8,
        help="Abort only after this many consecutive non-finite (skipped) steps.",
    )
    ap.add_argument(
        "--max-recoveries",
        type=int,
        default=0,
        help="On divergence, reload the last checkpoint and continue, up to this "
        "many times (0 = abort, the old behavior). For long runs prone to rare "
        "transient NaNs (needs --save-every set).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument(
        "--eval-batches",
        type=int,
        default=32,
        help="Batches averaged per eval-ppl estimate. The val stream is a "
        "high-variance 3-way mixture; 4 (the old default) gave ~72<->455 ppl "
        "jitter. 32 averages a ~8x larger sample -> ~3x tighter estimate. Raise "
        "further (e.g. 64) for an even smoother curve at proportional eval cost.",
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("research/reports/native_adaptive_hydra_ckpts"),
    )
    ap.add_argument("--load-checkpoint", type=Path, default=None)
    ap.add_argument(
        "--load-nonstrict",
        action="store_true",
        help="Resume a checkpoint into a MoR-router model: tolerate the fresh "
        "halt_head, deep-start re-init it, reset the optimizer.",
    )
    ap.add_argument(
        "--ponder-weight",
        type=float,
        default=None,
        help="Override the MoR ponder (expected-depth) penalty. 0.0 lets the LM "
        "loss alone decide depth (isolates 'does depth help' from compute cost).",
    )
    ap.add_argument(
        "--freeze-router",
        action="store_true",
        help="Freeze the MoR halt-head router at its init -> fixed recursion "
        "depth (no adaptive routing). For the fixed-depth-vs-shallow capability "
        "ablation.",
    )
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--restart-step", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/native_adaptive_hydra_train.jsonl"),
    )
    ap.add_argument("--append", action="store_true")
    ap.add_argument(
        "--require-sources", action=argparse.BooleanOptionalAction, default=True
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
