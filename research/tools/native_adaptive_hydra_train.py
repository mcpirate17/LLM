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
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm
from research.training._optimizer_muon import MuonOptimizer


LOCAL_MIX_NAME = "codex_ffw60_chat30_pleias10_local"


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


def _adaptive_depth_stats(model: nn.Module) -> dict[str, float] | None:
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
    """Hybrid-Muon split: 2D hidden matrices -> Muon, everything else
    (embeddings/tied head, 1D norms/biases, scalar params) -> AdamW."""
    embed_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embed_ids.add(id(p))
    muon: list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []
    seen: set[int] = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids or p.ndim < 2:
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
    loss = _lm_loss(model(ids), labels)
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
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    for opt, bases in zip(optimizers, base_lrs):
        for group, base in zip(opt.param_groups, bases):
            group["lr"] = base * mult
        opt.step()
    grad = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
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


def _record_step(args, model, train_loader, val_loader, started, step, metrics):
    """Append a log/eval row when due. metrics = (loss, grad, lr). Returns the row or None."""
    last_loss, last_grad, cur_lr = metrics
    should_log = step == 1 or step % args.log_every == 0 or step == args.steps
    should_eval = step % args.eval_every == 0 or step == args.steps
    if not (should_log or should_eval):
        return None
    row: dict[str, Any] = {
        "event": "step",
        "step": step,
        "loss": round(last_loss, 4),
        "grad_norm": round(last_grad, 4),
        "lr": round(cur_lr, 8),
        "elapsed_sec": round(time.time() - started, 2),
        "depth": _adaptive_depth_stats(model),
        "loader_stats": getattr(train_loader, "stats", lambda: {})(),
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
    print(json.dumps(row, default=str), flush=True)
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
        if unexpected or any("halt_head" not in key for key in missing):
            raise RuntimeError(
                "--load-nonstrict resume mismatch beyond halt_head: "
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
    n_params = sum(p.numel() for p in model.parameters())
    payload, loaded_step = _load_checkpoint(model, args)

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
    for step in range(first_step, args.steps + 1):
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
            checkpoints.append(str(_save_checkpoint(model, optimizers, args, step)))

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
        "--max-consecutive-nonfinite",
        type=int,
        default=8,
        help="Abort only after this many consecutive non-finite (skipped) steps.",
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
    ap.add_argument("--eval-batches", type=int, default=4)
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
