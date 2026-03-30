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
from torch import nn

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
    # Allow cuDNN to benchmark and cache the fastest convolution algorithms
    torch.backends.cudnn.benchmark = True
    # TF32 on Ampere+ gives ~3x matmul throughput at negligible precision cost
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Reduce CPU-side overhead from CUDA allocator
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Model builder ────────────────────────────────────────────────────
def build_champion(d: int):
    """Build the champion architecture at dimension d."""
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.compiler import compile_model

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

    g.metadata["mutation_name"] = "champion_c9c7075e_scaled"
    return compile_model([g])


# ── Muon optimizer ───────────────────────────────────────────────────
class _MuonGroup:
    """Newton-Schulz orthogonalized momentum SGD for 2D+ parameters.

    Applies NS iteration to approximate the matrix square root of the
    second moment, giving natural-gradient-like updates without Adam's
    memory cost.
    """
    __slots__ = ("params", "lr", "momentum", "wd", "ns_steps", "buffers")

    def __init__(self, params, lr, momentum, wd, ns_steps=5):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.ns_steps = ns_steps
        self.buffers = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self):
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            g = p.grad

            if g.ndim >= 2:
                shape = g.shape
                g2d = g.reshape(shape[0], -1) if g.ndim > 2 else g
                g2d = g2d.float()
                g2d = g2d / (g2d.norm() + 1e-8)

                rows, cols = g2d.shape
                X = g2d
                if rows <= cols:
                    for _ in range(self.ns_steps):
                        A = X @ X.T
                        X = 1.5 * X - 0.5 * A @ X
                else:
                    for _ in range(self.ns_steps):
                        A = X.T @ X
                        X = 1.5 * X - 0.5 * X @ A

                g = X.reshape(shape).to(p.dtype)

            buf.mul_(self.momentum).add_(g)
            p.mul_(1 - self.lr * self.wd)
            p.add_(buf, alpha=-self.lr)


class _CombinedOptimizer:
    __slots__ = ("muon", "adamw", "_all_muon_params")

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        self._all_muon_params = muon.params

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self):
        # set_to_none=True avoids memset-to-zero, lets allocator reuse memory
        for p in self._all_muon_params:
            p.grad = None
        self.adamw.zero_grad(set_to_none=True)


def get_muon_optimizer(model, lr=0.02, momentum=0.95, wd=0.01):
    """Muon optimizer: Newton-Schulz orthogonalized momentum SGD."""
    params_2d = []
    params_other = []

    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            params_2d.append(p)
        else:
            params_other.append(p)

    muon = _MuonGroup(params_2d, lr=lr, momentum=momentum, wd=wd)
    adamw = torch.optim.AdamW(params_other, lr=lr * 0.1, weight_decay=wd)
    return _CombinedOptimizer(muon, adamw)


# ── Async data prefetch ──────────────────────────────────────────────
def get_data_iterator(batch_size: int, seq_len: int, device: str):
    """Streaming FineWeb-Edu + UltraChat with background prefetch.

    Uses a background thread for tokenization + batch assembly,
    pin_memory for async H2D transfer, and a numpy ring buffer
    instead of a Python list.
    """
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = 100277
    tokens_per_batch = batch_size * (seq_len + 1)

    # ── Streaming dataset iterators ──
    logger.info("Loading FineWeb-Edu (streaming)...")
    fw_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    logger.info("Loading UltraChat (streaming)...")
    uc_ds = load_dataset(
        "stingning/ultrachat",
        split="train",
        streaming=True,
    )

    fw_iter = iter(fw_ds)
    uc_iter = iter(uc_ds)
    logger.info(f"Tokenizer: cl100k_base, vocab_size={vocab_size}")

    # ── Ring buffer (numpy for O(1) slicing) ──
    buf_capacity = tokens_per_batch * 16
    ring = np.empty(buf_capacity, dtype=np.int32)
    ring_len = 0
    step_counter = 0

    def _next_text():
        nonlocal fw_iter, uc_iter, step_counter
        step_counter += 1
        use_ultrachat = step_counter % 10 < 3

        if use_ultrachat:
            try:
                example = next(uc_iter)
                messages = example.get("data") or example.get("messages") or []
                if isinstance(messages, list):
                    return "\n".join(str(m) for m in messages)
                return str(messages)
            except StopIteration:
                uc_iter = iter(load_dataset(
                    "stingning/ultrachat", split="train", streaming=True,
                ))
                return _next_text()
        else:
            try:
                return next(fw_iter).get("text", "")
            except StopIteration:
                fw_iter = iter(load_dataset(
                    "HuggingFaceFW/fineweb-edu", name="sample-10BT",
                    split="train", streaming=True,
                ))
                return _next_text()

    def _fill_ring():
        nonlocal ring, ring_len, buf_capacity
        target = tokens_per_batch * 8
        while ring_len < target:
            text = _next_text()
            if len(text) < 50:
                continue
            tokens = enc.encode(text)
            n_tok = len(tokens)
            # Grow ring if needed
            if ring_len + n_tok > buf_capacity:
                buf_capacity = max(buf_capacity * 2, ring_len + n_tok)
                new_ring = np.empty(buf_capacity, dtype=np.int32)
                new_ring[:ring_len] = ring[:ring_len]
                ring = new_ring
            ring[ring_len:ring_len + n_tok] = tokens
            ring_len += n_tok

    def _extract_batch_np():
        """Extract one batch from the ring buffer. Returns numpy array."""
        nonlocal ring_len
        if ring_len < tokens_per_batch:
            _fill_ring()
        batch = ring[:tokens_per_batch].copy()
        # Shift remaining data (memmove — fast C-level copy)
        remaining = ring_len - tokens_per_batch
        if remaining > 0:
            ring[:remaining] = ring[tokens_per_batch:tokens_per_batch + remaining]
        ring_len = remaining
        return batch

    # Initial fill
    _fill_ring()
    logger.info(f"Buffer ready: {ring_len:,} tokens (70% FineWeb-Edu + 30% UltraChat)")

    # ── Prefetch queue: background thread assembles batches ──
    prefetch_q: queue.Queue = queue.Queue(maxsize=4)
    _stop_event = threading.Event()

    def _prefetch_worker():
        """Background thread: tokenize → batch → pin_memory."""
        use_cuda = device.startswith("cuda")
        while not _stop_event.is_set():
            try:
                batch_np = _extract_batch_np()
                t = torch.from_numpy(batch_np).long().reshape(batch_size, seq_len + 1)
                if use_cuda:
                    t = t.pin_memory()
                prefetch_q.put(t, timeout=5.0)
            except queue.Full:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                break

    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    def _get_batch():
        t = prefetch_q.get()
        non_blocking = device.startswith("cuda")
        t = t.to(device, non_blocking=non_blocking)
        return t[:, :seq_len], t[:, 1:seq_len + 1]

    # Attach cleanup so caller can stop the thread
    _get_batch.stop = lambda: _stop_event.set()
    return _get_batch, vocab_size


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
        """Queue a checkpoint save. Blocks only if 2 saves already queued."""
        self._queue.put((path, data))

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=60)


# ── Benchmarks ───────────────────────────────────────────────────────
def run_benchmarks(model, step: int, ckpt_dir: Path, device: str = "cuda"):
    """Run standard LM benchmarks: WikiText-103 PPL, HellaSwag, LAMBADA, ARC-Easy."""
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM
    import tiktoken

    logger.info(f"Running benchmarks at step {step}...")
    model.eval()

    class AriaLM(LM):
        def __init__(self, model, device, max_length=1024):
            super().__init__()
            self._model = model
            self._device = device
            self._max_length = max_length
            self._enc = tiktoken.get_encoding("cl100k_base")
            self._vocab_size = 100277

        @property
        def eot_token_id(self):
            return self._enc.eot_token

        @property
        def max_length(self):
            return self._max_length

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return 4

        @property
        def device(self):
            return self._device

        def tok_encode(self, string, **kwargs):
            return self._enc.encode(string)

        def tok_decode(self, tokens, **kwargs):
            return self._enc.decode(tokens)

        def _model_call(self, inps):
            with torch.no_grad():
                return self._model(inps.to(self._device))

        def _model_generate(self, context, max_length, eos_token_id):
            raise NotImplementedError("Generation not supported")

        def loglikelihood(self, requests):
            results = []
            for ctx, cont in [req.args for req in requests]:
                ctx_ids = self._enc.encode(ctx) if ctx else []
                cont_ids = self._enc.encode(cont)
                all_ids = (ctx_ids + cont_ids)[-self._max_length:]
                input_ids = torch.tensor([all_ids], device=self._device)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = self._model(input_ids)
                log_probs = F.log_softmax(logits[0].float(), dim=-1)
                cont_start = len(all_ids) - len(cont_ids)
                total_ll = 0.0
                greedy_match = True
                for i, tok in enumerate(cont_ids):
                    pos = cont_start + i - 1
                    if 0 <= pos < log_probs.size(0):
                        total_ll += log_probs[pos, tok].item()
                        if log_probs[pos].argmax().item() != tok:
                            greedy_match = False
                results.append((total_ll, greedy_match))
            return results

        def loglikelihood_rolling(self, requests):
            results = []
            for (string,) in [req.args for req in requests]:
                tokens = self._enc.encode(string)
                total_ll = 0.0
                for start in range(0, len(tokens), self._max_length):
                    chunk = tokens[start:start + self._max_length]
                    input_ids = torch.tensor([chunk], device=self._device)
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits = self._model(input_ids)
                    log_probs = F.log_softmax(logits[0].float(), dim=-1)
                    for i in range(1, len(chunk)):
                        total_ll += log_probs[i - 1, chunk[i]].item()
                results.append(total_ll)
            return results

        def generate_until(self, requests):
            return [""] * len(requests)

    lm = AriaLM(model, device)

    benchmarks = ["wikitext", "hellaswag", "lambada_openai"]
    try:
        import lm_eval.evaluator as _ev
        _ev.add_env_info = lambda results: None
        _ev.add_tokenizer_info = lambda results, lm: None

        eval_results = simple_evaluate(
            model=lm,
            tasks=benchmarks,
            batch_size=4,
            device=device,
        )

        results_dict = {}
        logger.info(f"\n{'=' * 60}")
        logger.info(f"BENCHMARKS at step {step}")
        logger.info(f"{'=' * 60}")

        for task_name, task_results in eval_results.get("results", {}).items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and "stderr" not in metric:
                    results_dict[f"{task_name}/{metric}"] = value
                    logger.info(f"  {task_name}/{metric}: {value:.4f}")

        results_path = ckpt_dir / f"benchmarks_step_{step}.json"
        with open(results_path, "w") as f:
            json.dump({"step": step, "results": results_dict}, f, indent=2)
        logger.info(f"  Saved to {results_path}")

        return results_dict

    except Exception as e:
        logger.error(f"Benchmark eval failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        model.train()


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
    """Save loss curve PNG with O(n) numpy smoothing instead of O(n²) list comp."""
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
            ax.plot(xs[::stride], raw[::stride], alpha=0.15, color="blue", linewidth=0.5)
            # O(n) cumsum-based rolling average
            w = 100
            cumsum = np.cumsum(raw)
            cumsum = np.insert(cumsum, 0, 0.0)
            # Smoothed at every stride-th point
            indices = np.arange(0, n, stride)
            starts = np.maximum(indices - w + 1, 0)
            smoothed = (cumsum[indices + 1] - cumsum[starts]) / (indices - starts + 1)
            ax.plot(xs[::stride][:len(smoothed)], smoothed, color="blue", linewidth=2,
                    label=f"avg100={smoothed[-1]:.4f}")
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


# ── Main training loop ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Scale test: champion at 177M")
    parser.add_argument("--dim", type=int, default=1536, help="Model dimension")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--bs", type=int, default=12, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--lr", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log loss every N steps")
    parser.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep only the last N checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="research/artifacts/scale_test")
    parser.add_argument("--plot", action="store_true", default=True, help="Live loss plot")
    parser.add_argument("--eval-every", type=int, default=10000, help="Run benchmarks every N steps")
    parser.add_argument("--eval-only", action="store_true", help="Just run evals on existing checkpoint")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # CUDA tuning — must happen before any tensor allocation
    _configure_cuda()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    logger.info(f"Building champion at d={args.dim}...")
    model = build_champion(args.dim).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params ({n_params / 1e6:.1f}M)")

    # torch.compile — fuses ops, eliminates Python overhead
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

    # AMP scaler for mixed precision
    use_amp = args.device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
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
        # Handle compiled model state dict (keys may have _orig_mod. prefix)
        state = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(cleaned)
        start_step = ckpt["step"]
        loss_history = ckpt.get("loss_history", [])
        tokens_seen = ckpt.get("tokens_seen", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        optimizer = get_muon_optimizer(model, lr=args.lr)
        if "optimizer_muon_buffers" in ckpt:
            for buf, saved in zip(optimizer.muon.buffers, ckpt["optimizer_muon_buffers"]):
                buf.copy_(saved)
        logger.info(
            f"Resumed: step {start_step}, best_loss={best_loss:.4f}, "
            f"tokens={tokens_seen:,}"
        )

    # Data
    get_batch, vocab_size = get_data_iterator(args.bs, args.seq_len, args.device)
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
        run_benchmarks(model, start_step, ckpt_dir, args.device)
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

    # Async checkpointer
    checkpointer = _AsyncCheckpointer()

    # Pre-cache LR schedule values (avoid repeated math in hot loop)
    warmup_inv = 1.0 / args.warmup if args.warmup > 0 else 1.0
    decay_denom = max(args.steps - args.warmup, 1)

    # Training loop
    model.train()
    t0 = time.time()

    logger.info("=" * 80)
    logger.info(
        f"{'Step':>8} {'Loss':>8} {'LR':>8} {'Tok/s':>8} "
        f"{'Tokens':>12} {'Elapsed':>10} {'ETA':>10}"
    )
    logger.info("=" * 80)

    # Rolling average window (O(1) update via deque)
    avg_window: deque = deque(maxlen=100)
    for v in loss_history[-100:]:
        avg_window.append(v)

    for step in range(start_step + 1, args.steps + 1):
        # Warmup + cosine decay
        if step <= args.warmup:
            lr_mult = step * warmup_inv
        else:
            progress = (step - args.warmup) / decay_denom
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = args.lr * lr_mult

        # Update LRs
        optimizer.muon.lr = current_lr
        for pg in optimizer.adamw.param_groups:
            pg["lr"] = current_lr * 0.1

        # Forward + backward with AMP
        inputs, targets = get_batch()
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer.adamw)  # unscale for grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Muon operates on unscaled grads (AdamW already unscaled above)
            # For Muon params, manually unscale
            inv_scale = 1.0 / scaler.get_scale()
            for p in optimizer.muon.params:
                if p.grad is not None:
                    p.grad.mul_(inv_scale)
            scaler.step(optimizer.adamw)
            optimizer.muon.step()
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad()

        loss_val = loss.item()
        loss_history.append(loss_val)
        avg_window.append(loss_val)
        tokens_seen += tokens_per_step
        best_loss = min(best_loss, loss_val)

        # Log
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            eta = (args.steps - step) * elapsed / max(step - start_step, 1)
            eta_str = f"{eta / 3600:.1f}h" if eta > 3600 else f"{eta / 60:.0f}m"
            elapsed_str = f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.0f}m"

            avg = sum(avg_window) / len(avg_window)

            logger.info(
                f"{step:>8} {loss_val:>8.4f} {current_lr:>8.5f} "
                f"{tok_per_sec:>8.0f} {tokens_seen:>12,} "
                f"{elapsed_str:>10} {eta_str:>10}"
                f"  (avg100={avg:.4f} best={best_loss:.4f})"
            )

            if args.plot and step % 50 == 0:
                _save_loss_plot(
                    plot_path, loss_history, start_step, step,
                    args.steps, best_loss, n_params, tokens_seen, eta_str,
                )

        # Checkpoint (async)
        if step % args.save_every == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_data = {
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_muon_buffers": [b.cpu().clone() for b in optimizer.muon.buffers],
                "loss": loss_val,
                "best_loss": best_loss,
                "tokens_seen": tokens_seen,
                "loss_history": loss_history,
                "config": {
                    "dim": args.dim,
                    "n_params": n_params,
                    "architecture": "champion_c9c7075e",
                },
            }
            # Save step checkpoint + latest (async)
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            checkpointer.save(ckpt_path, ckpt_data)
            checkpointer.save(ckpt_dir / "latest.pt", ckpt_data)

            # Save loss curve JSON (small, fine to do sync)
            with open(ckpt_dir / "loss_curve.json", "w") as f:
                json.dump({"steps": list(range(1, len(loss_history) + 1)), "loss": loss_history}, f)

            # Prune old checkpoints
            existing = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            while len(existing) > args.keep_checkpoints:
                old = existing.pop(0)
                old.unlink()

            logger.info(f"  Checkpoint saved: step {step} (keeping last {args.keep_checkpoints})")

            # Benchmarks
            if args.eval_every > 0 and step % args.eval_every == 0:
                try:
                    run_benchmarks(model, step, ckpt_dir, args.device)
                except Exception as e:
                    logger.warning(f"Benchmark eval failed (training continues): {e}")

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

    # Save final checkpoint
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
                "n_params": n_params,
                "architecture": "champion_c9c7075e",
            },
        },
        ckpt_dir / "final.pt",
    )

    with open(ckpt_dir / "loss_curve.json", "w") as f:
        json.dump({"steps": list(range(1, len(loss_history) + 1)), "loss": loss_history}, f)

    logger.info(f"  Saved to {args.checkpoint_dir}/")

    # Final benchmarks
    logger.info("Running final benchmarks...")
    try:
        run_benchmarks(model, args.steps, ckpt_dir, args.device)
    except Exception as e:
        logger.warning(f"Final benchmark eval failed: {e}")
        logger.info("Run benchmarks separately with: python -m research.tools.scale_test --eval-only")


if __name__ == "__main__":
    main()
