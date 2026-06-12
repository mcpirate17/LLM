import argparse
import json
import math
import sys
from pathlib import Path

import torch

from component_fab.generator.memory_primitives import UniversalMasterLane
from component_fab.harness.tiny_lm import TinyLM, TinyLMConfig
from research.defaults import PROJECT_ROOT
from research.training._optimizer_muon import MuonOptimizer

LOCAL_MIX_NAME = "codex_ffw60_chat30_pleias10_local"
_SAFE_VOCAB = 100277


def _build_model_and_opts(
    args: argparse.Namespace,
) -> tuple[TinyLM, MuonOptimizer, torch.optim.AdamW]:
    """Construct the model (compiled) and its Muon+AdamW optimizers."""
    cfg = TinyLMConfig(
        dim=args.dim, n_blocks=args.n_blocks, vocab_size=_SAFE_VOCAB, use_ffn=True
    )
    model = TinyLM(lambda d: UniversalMasterLane(d), cfg).to(args.device)
    model.lm_head.weight = model.embed.weight
    model = torch.compile(model)
    muon_p = [
        p
        for n, p in model.named_parameters()
        if p.ndim >= 2 and "embed" not in n and "lm_head" not in n
    ]
    adam_p = [
        p for n, p in model.named_parameters() if all(p is not mp for mp in muon_p)
    ]
    opt_muon = MuonOptimizer(muon_p, lr=args.muon_lr)
    opt_adam = torch.optim.AdamW(adam_p, lr=args.lr, weight_decay=0.01)
    return model, opt_muon, opt_adam


def _setup_loaders(args: argparse.Namespace) -> tuple[object, object]:
    """Register the local dataset mix and create train/val loaders."""
    if str(PROJECT_ROOT / "HYDRA") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "HYDRA"))
    from hydra.data import universal_data_loader as udl

    udl.DATASET_CONFIGS[LOCAL_MIX_NAME] = {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.60},
            {"name": "small_chat_seqaware_flat", "weight": 0.30},
            {"name": "pleias_synth", "weight": 0.10},
        ],
    }
    train_loader = udl.create_universal_loader(
        dataset=LOCAL_MIX_NAME,
        batch_size=args.batch,
        seq_len=args.seq_len,
        vocab_size=_SAFE_VOCAB,
        tokenizer_name="cl100k_base",
        seed=42,
    )
    val_loader = udl.create_universal_loader(
        dataset=LOCAL_MIX_NAME,
        batch_size=args.batch,
        seq_len=args.seq_len,
        vocab_size=_SAFE_VOCAB,
        tokenizer_name="cl100k_base",
        seed=1009,
    )
    return train_loader, val_loader


def _eval_val_loss(
    model: TinyLM,
    val_loader: object,
    args: argparse.Namespace,
) -> float:
    """Run a validation pass and return the mean cross-entropy loss."""
    model.eval()
    v_total = 0.0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            v_batch = next(val_loader)
            v_ids = v_batch["input_ids"].to(args.device)
            v_labels = v_batch["labels"].to(args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_loss = torch.nn.functional.cross_entropy(
                    model(torch.remainder(v_ids, _SAFE_VOCAB)).view(-1, _SAFE_VOCAB),
                    torch.where(
                        (v_labels >= 0) & (v_labels < _SAFE_VOCAB),
                        v_labels,
                        torch.full_like(v_labels, -100),
                    ),
                    ignore_index=-100,
                )
            v_total += v_loss.item()
    model.train()
    return v_total / args.eval_batches


def _log_step(
    model: TinyLM,
    val_loader: object,
    args: argparse.Namespace,
    step: int,
    loss_accum: float,
    mult: float,
    grad_norm: torch.Tensor,
) -> None:
    """Log step metrics (and optionally val loss) to stdout and the JSONL file."""
    ppl = math.exp(min(loss_accum, 11.5))
    row: dict = {
        "event": "step",
        "step": step,
        "loss": round(loss_accum, 4),
        "lr": round(args.lr * mult, 8),
        "grad_norm": round(float(grad_norm), 4),
        "ppl": round(ppl, 2),
    }
    if step % args.eval_every == 0:
        v_loss = _eval_val_loss(model, val_loader, args)
        row["val"] = {
            "loss": round(v_loss, 4),
            "ppl": round(math.exp(min(v_loss, 11.5)), 2),
        }
        print(
            f"step {step:>6d} lr={row['lr']:.2e} loss={row['loss']:>7.3f} "
            f"grad_norm={row['grad_norm']:>7.3f} ppl≈{row['ppl']:.1f} "
            f"val_ppl={row['val']['ppl']:.1f}",
            flush=True,
        )
    else:
        print(
            f"step {step:>6d} lr={row['lr']:.2e} loss={row['loss']:>7.3f} "
            f"grad_norm={row['grad_norm']:>7.3f} ppl≈{row['ppl']:.1f}",
            flush=True,
        )
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")


def _run_training_loop(
    model: TinyLM,
    opt_muon: MuonOptimizer,
    opt_adam: torch.optim.AdamW,
    train_loader: object,
    val_loader: object,
    args: argparse.Namespace,
) -> None:
    """Execute the full training loop."""
    for step in range(1, args.steps + 1):
        if args.stop_at and step > args.stop_at:
            break
        opt_muon.zero_grad()
        opt_adam.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            batch = next(train_loader)
            ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            ids = torch.remainder(ids, _SAFE_VOCAB)
            labels = torch.where(
                (labels >= 0) & (labels < _SAFE_VOCAB),
                labels,
                torch.full_like(labels, -100),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, _SAFE_VOCAB), labels.view(-1), ignore_index=-100
                )
                loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            loss_accum += loss.item() / args.grad_accum
        mult = (
            (step / args.warmup_steps)
            if step < args.warmup_steps
            else (
                args.min_lr_frac
                + (1.0 - args.min_lr_frac)
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (step - args.warmup_steps)
                        / (args.steps - args.warmup_steps)
                    )
                )
            )
        )
        for g in opt_muon.param_groups:
            g["lr"] = args.muon_lr * mult
        for g in opt_adam.param_groups:
            g["lr"] = args.lr * mult
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adam.step()
        if step % 25 == 0 or step <= 10:
            _log_step(model, val_loader, args, step, loss_accum, mult, grad_norm)
        if step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step},
                Path(args.out).parent / f"{args.run_label}_step_{step}.pt",
            )


def train() -> None:
    """Parse args, build model and data, run training."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", default="gemini_master_135m")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--stop-at", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--warmup-steps", type=int, default=800)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, opt_muon, opt_adam = _build_model_and_opts(args)
    train_loader, val_loader = _setup_loaders(args)
    _run_training_loop(model, opt_muon, opt_adam, train_loader, val_loader, args)


if __name__ == "__main__":
    train()
