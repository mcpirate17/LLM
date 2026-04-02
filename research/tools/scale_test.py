"""Scale test: train the champion architecture at 177M params on FineWeb-Edu.

Usage:
    python -m research.tools.scale_test [--steps 50000] [--dim 1536] [--bs 16]

Logs loss every 10 steps, saves checkpoints every 1000 steps.
Loss curve visible in real-time via stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.tools._data_prefetch import get_data_iterator
from research.tools._lm_benchmarks import run_benchmarks
from research.tools._muon_optimizer import get_muon_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CUDA backend tuning ─────────────────────────────────────────────
def _configure_cuda():
    """Enable hardware-level fast paths before any tensor allocation."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Model builder ────────────────────────────────────────────────────
def _make_champion_graph(d: int, layer_idx: int):
    """Build one champion layer graph with its own parameters."""
    from research.synthesis.graph import ComputationGraph

    g = ComputationGraph(model_dim=d)
    inp = g.add_input()
    n = g.add_op

    # Block 1: sparse bottleneck FFN
    r = n("rmsnorm", [inp])
    d2 = n("linear_proj_down", [r])
    s = n("nm_sparse_linear", [d2], config={"out_dim": d})
    u = n("linear_proj_up", [s])
    a = n("add", [r, u])

    # Block 2: token merge + conv + swiglu + sparse
    r2 = n("rmsnorm", [a])
    t = n("token_merge", [r2])
    c = n("conv1d_seq", [t])
    w = n("swiglu_mlp", [c], config={"mlp_ratio": 2.0})
    s2 = n("nm_sparse_linear", [w], config={"out_dim": d})
    gl = n("gelu", [s2])
    g.set_output(n("add", [r2, gl]))

    g.metadata["mutation_name"] = f"champion_c9c7075e_L{layer_idx}"
    return g


def build_champion(d: int, n_layers: int = 1, vocab_size: int = 100277):
    """Build the champion architecture at dimension d with n_layers."""
    from research.synthesis.compiler import compile_model

    graphs = [_make_champion_graph(d, i) for i in range(n_layers)]
    return compile_model(graphs, vocab_size=vocab_size)


# ── Async checkpoint saving ──────────────────────────────────────────
class _AsyncCheckpointer:
    """Save checkpoints in a background thread to avoid blocking training."""

    __slots__ = ("_thread", "_queue")

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            path, data = item
            tmp = path.with_suffix(".tmp")
            torch.save(data, str(tmp))
            os.replace(str(tmp), str(path))

    def save(self, path: Path, data: dict):
        self._queue.put((path, data))

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=60)


# ── Loss plot (numpy-vectorized smoothing) ───────────────────────────
def _save_loss_plot(
    plot_path: Path,
    loss_history: list,
    start_step: int,
    step: int,
    total_steps: int,
    best_loss: float,
    n_params: int,
    tokens_seen: int,
    eta_str: str,
):
    """Save loss curve PNG with O(n) numpy smoothing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        raw = np.array(loss_history, dtype=np.float32)
        n = len(raw)
        xs = np.arange(start_step + 1, start_step + n + 1)

        fig, ax = plt.subplots(figsize=(12, 5))
        if n > 200:
            stride = 10
            ax.plot(
                xs[::stride], raw[::stride], alpha=0.15, color="blue", linewidth=0.5
            )
            w = 100
            cumsum = np.cumsum(raw)
            cumsum = np.insert(cumsum, 0, 0.0)
            indices = np.arange(0, n, stride)
            starts = np.maximum(indices - w + 1, 0)
            smoothed = (cumsum[indices + 1] - cumsum[starts]) / (indices - starts + 1)
            ax.plot(
                xs[::stride][: len(smoothed)],
                smoothed,
                color="blue",
                linewidth=2,
                label=f"avg100={smoothed[-1]:.4f}",
            )
        else:
            ax.plot(xs, raw, color="blue", linewidth=1, label=f"loss={raw[-1]:.4f}")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(
            f"Champion c9c7075e \u2014 {n_params / 1e6:.0f}M params \u2014 "
            f"step {step}/{total_steps} \u2014 best={best_loss:.4f} \u2014 "
            f"{tokens_seen / 1e6:.0f}M tokens \u2014 ETA {eta_str}"
        )
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=100)
        plt.close(fig)
    except Exception:
        pass


# ── Checkpoint data builder ──────────────────────────────────────────
def _make_ckpt_data(
    model,
    optimizer,
    step,
    loss_val,
    best_loss,
    tokens_seen,
    loss_history,
    args,
    n_params,
):
    """Build checkpoint dict. Single source of truth for checkpoint format."""
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_muon_buffers": [b.cpu().clone() for b in optimizer.muon.buffers],
        "loss": loss_val,
        "best_loss": best_loss,
        "tokens_seen": tokens_seen,
        "loss_history": loss_history,
        "config": {
            "dim": args.dim,
            "layers": args.layers,
            "vocab_size": args.vocab,
            "n_params": n_params,
            "architecture": "champion_c9c7075e",
        },
    }


def _save_loss_json(ckpt_dir: Path, loss_history: list):
    with open(ckpt_dir / "loss_curve.json", "w") as f:
        json.dump(
            {"steps": list(range(1, len(loss_history) + 1)), "loss": loss_history}, f
        )


# ── Main training loop ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Scale test: champion architecture")
    parser.add_argument("--dim", type=int, default=1536, help="Model dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--vocab",
        type=int,
        default=50257,
        help="Vocabulary size (50257=GPT-2, 100277=cl100k)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Training steps (0=auto Chinchilla-optimal)",
    )
    parser.add_argument("--bs", type=int, default=12, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--lr", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--log-every", type=int, default=10, help="Log loss every N steps"
    )
    parser.add_argument(
        "--save-every", type=int, default=2000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep only the last N checkpoints",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument(
        "--plot", action="store_true", default=True, help="Live loss plot"
    )
    parser.add_argument(
        "--eval-every", type=int, default=10000, help="Run benchmarks every N steps"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Just run evals on existing checkpoint"
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps"
    )
    args = parser.parse_args()

    _configure_cuda()

    tokenizer_name = "gpt2" if args.vocab <= 50257 else "cl100k_base"

    if not args.checkpoint_dir:
        args.checkpoint_dir = (
            f"research/artifacts/scale_test_{args.layers}L_{args.vocab}v"
        )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    logger.info(
        f"Building champion: d={args.dim}, layers={args.layers}, vocab={args.vocab}..."
    )
    model = build_champion(args.dim, n_layers=args.layers, vocab_size=args.vocab).to(
        args.device
    )
    n_params = sum(p.numel() for p in model.parameters())
    embed_params = args.vocab * args.dim
    compute_params = n_params - embed_params
    logger.info(
        f"Model: {n_params:,} params ({n_params / 1e6:.1f}M) — "
        f"embed={embed_params / 1e6:.1f}M, compute={compute_params / 1e6:.1f}M"
    )

    # Auto-compute Chinchilla-optimal steps
    tokens_per_step = args.bs * args.seq_len
    if args.steps == 0:
        chinchilla_tokens = n_params * 20
        args.steps = chinchilla_tokens // tokens_per_step
        logger.info(
            f"Auto steps: {args.steps:,} (Chinchilla-optimal for {n_params / 1e6:.0f}M params)"
        )

    # torch.compile
    use_compile = (
        not args.no_compile
        and args.device.startswith("cuda")
        and hasattr(torch, "compile")
    )
    if use_compile:
        logger.info("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    # Optimizer
    optimizer = get_muon_optimizer(model, lr=args.lr)
    logger.info(f"Optimizer: Muon (lr={args.lr}, momentum=0.95)")

    # AMP
    use_amp = args.device.startswith("cuda")
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    if use_amp:
        logger.info(f"AMP enabled: {amp_dtype}")

    # Resume from checkpoint
    start_step = 0
    loss_history = []
    tokens_seen = 0
    best_loss = float("inf")

    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        logger.info(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
        state = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(
                cleaned
            )
        start_step = ckpt["step"]
        loss_history = ckpt.get("loss_history", [])
        tokens_seen = ckpt.get("tokens_seen", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        optimizer = get_muon_optimizer(model, lr=args.lr)
        if "optimizer_muon_buffers" in ckpt:
            for buf, saved in zip(
                optimizer.muon.buffers, ckpt["optimizer_muon_buffers"]
            ):
                buf.copy_(saved)
        logger.info(
            f"Resumed: step {start_step}, best_loss={best_loss:.4f}, "
            f"tokens={tokens_seen:,}"
        )

    # Data
    get_batch, vocab_size = get_data_iterator(
        args.bs, args.seq_len, args.device, tokenizer_name
    )
    tokens_per_step = args.bs * args.seq_len
    total_tokens = args.steps * tokens_per_step
    logger.info(
        f"Training: steps {start_step + 1}\u2192{args.steps} ({args.steps - start_step:,} remaining), "
        f"{total_tokens / 1e9:.1f}B tokens total, bs={args.bs}, seq_len={args.seq_len}"
    )
    logger.info(
        f"Chinchilla optimal: {n_params * 20 / 1e9:.1f}B tokens "
        f"({n_params * 20 // tokens_per_step:,} steps)"
    )

    if args.eval_only:
        logger.info("Eval-only mode \u2014 running benchmarks on current checkpoint")
        run_benchmarks(model, start_step, ckpt_dir, args.device, tokenizer_name)
        return

    if start_step >= args.steps:
        logger.info("Already completed. Use --steps to increase target.")
        return

    # Plot setup
    plot_path = ckpt_dir / "loss_curve.png"
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            logger.info(f"Loss plot: {plot_path} (updates every 50 steps)")
        except Exception as e:
            logger.warning(f"Plot unavailable: {e}")
            args.plot = False

    checkpointer = _AsyncCheckpointer()

    # Pre-cache LR schedule values
    warmup_inv = 1.0 / args.warmup if args.warmup > 0 else 1.0
    decay_denom = max(args.steps - args.warmup, 1)

    # Graceful Ctrl+C
    import signal

    _stop_requested = False

    def _sigint_handler(signum, frame):
        nonlocal _stop_requested
        if _stop_requested:
            logger.info("Second Ctrl+C — forcing exit")
            raise SystemExit(1)
        _stop_requested = True
        logger.info("Ctrl+C caught — will save checkpoint after current step")

    signal.signal(signal.SIGINT, _sigint_handler)

    # Training loop
    model.train()
    t0 = time.time()

    logger.info("=" * 80)
    logger.info(
        f"{'Step':>8} {'Loss':>8} {'LR':>8} {'Tok/s':>8} "
        f"{'Tokens':>12} {'Elapsed':>10} {'ETA':>10}"
    )
    logger.info("=" * 80)

    avg_window: deque = deque(maxlen=100)
    for v in loss_history[-100:]:
        avg_window.append(v)

    grad_accum = args.grad_accum
    loss_val = 0.0

    for step in range(start_step + 1, args.steps + 1):
        # LR schedule: warmup + cosine decay
        if step <= args.warmup:
            lr_mult = step * warmup_inv
        else:
            progress = min((step - args.warmup) / decay_denom, 1.0)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = args.lr * lr_mult

        optimizer.muon.lr = current_lr
        for pg in optimizer.adamw.param_groups:
            pg["lr"] = current_lr * 0.1

        # Forward + backward with AMP and gradient accumulation
        accum_loss = 0.0
        for _micro in range(grad_accum):
            inputs, targets = get_batch()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                if grad_accum > 1:
                    loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * (grad_accum if grad_accum > 1 else 1)

        if scaler.is_enabled():
            scaler.unscale_(optimizer.adamw)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inv_scale = 1.0 / scaler.get_scale()
            for p in optimizer.muon.params:
                if p.grad is not None:
                    p.grad.mul_(inv_scale)
            scaler.step(optimizer.adamw)
            optimizer.muon.step()
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad()

        loss_val = accum_loss / grad_accum if grad_accum > 1 else accum_loss
        loss_history.append(loss_val)
        avg_window.append(loss_val)
        tokens_seen += tokens_per_step * grad_accum
        best_loss = min(best_loss, loss_val)

        # Log
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            eta = (args.steps - step) * elapsed / max(step - start_step, 1)
            eta_str = f"{eta / 3600:.1f}h" if eta > 3600 else f"{eta / 60:.0f}m"
            elapsed_str = (
                f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.0f}m"
            )

            avg = sum(avg_window) / len(avg_window)

            logger.info(
                f"{step:>8} {loss_val:>8.4f} {current_lr:>8.5f} "
                f"{tok_per_sec:>8.0f} {tokens_seen:>12,} "
                f"{elapsed_str:>10} {eta_str:>10}"
                f"  (avg100={avg:.4f} best={best_loss:.4f})"
            )

            if args.plot and step % 50 == 0:
                _save_loss_plot(
                    plot_path,
                    loss_history,
                    start_step,
                    step,
                    args.steps,
                    best_loss,
                    n_params,
                    tokens_seen,
                    eta_str,
                )

        # Checkpoint (async)
        if step % args.save_every == 0:
            ckpt_data = _make_ckpt_data(
                model,
                optimizer,
                step,
                loss_val,
                best_loss,
                tokens_seen,
                loss_history,
                args,
                n_params,
            )
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            checkpointer.save(ckpt_path, ckpt_data)
            checkpointer.save(ckpt_dir / "latest.pt", ckpt_data)
            _save_loss_json(ckpt_dir, loss_history)

            # Prune old checkpoints
            existing = sorted(
                ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])
            )
            while len(existing) > args.keep_checkpoints:
                old = existing.pop(0)
                old.unlink()

            logger.info(
                f"  Checkpoint saved: step {step} (keeping last {args.keep_checkpoints})"
            )

            # Benchmarks
            if args.eval_every > 0 and step % args.eval_every == 0:
                try:
                    run_benchmarks(model, step, ckpt_dir, args.device, tokenizer_name)
                except Exception as e:
                    logger.warning(f"Benchmark eval failed (training continues): {e}")

        # Graceful stop
        if _stop_requested:
            logger.info(f"Graceful stop at step {step} — saving checkpoint...")
            ckpt_data = _make_ckpt_data(
                model,
                optimizer,
                step,
                loss_val,
                best_loss,
                tokens_seen,
                loss_history,
                args,
                n_params,
            )
            torch.save(ckpt_data, ckpt_dir / f"step_{step}.pt")
            torch.save(ckpt_data, ckpt_dir / "latest.pt")
            _save_loss_json(ckpt_dir, loss_history)
            logger.info(f"  Checkpoint saved at step {step}. Exiting.")
            break

    # Cleanup
    get_batch.stop()
    checkpointer.shutdown()

    # Final summary
    elapsed = time.time() - t0
    logger.info("=" * 80)
    logger.info(f"DONE: {args.steps:,} steps in {elapsed / 3600:.1f}h")
    logger.info(f"  Final loss: {loss_val:.4f}")
    logger.info(f"  Best loss: {best_loss:.4f}")
    logger.info(f"  Tokens: {tokens_seen:,} ({tokens_seen / 1e9:.2f}B)")
    logger.info(f"  Throughput: {tokens_seen / elapsed:.0f} tok/s")

    # Final checkpoint
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "step": args.steps,
            "model_state_dict": raw_model.state_dict(),
            "loss": loss_val,
            "best_loss": best_loss,
            "tokens_seen": tokens_seen,
            "loss_history": loss_history,
            "config": {
                "dim": args.dim,
                "layers": args.layers,
                "vocab_size": args.vocab,
                "n_params": n_params,
                "architecture": "champion_c9c7075e",
            },
        },
        ckpt_dir / "final.pt",
    )
    _save_loss_json(ckpt_dir, loss_history)
    logger.info(f"  Saved to {args.checkpoint_dir}/")

    # Final benchmarks
    logger.info("Running final benchmarks...")
    try:
        run_benchmarks(model, args.steps, ckpt_dir, args.device, tokenizer_name)
    except Exception as e:
        logger.warning(f"Final benchmark eval failed: {e}")
        logger.info(
            "Run benchmarks separately with: python -m research.tools.scale_test --eval-only"
        )


if __name__ == "__main__":
    main()
