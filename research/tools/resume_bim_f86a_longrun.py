"""Resume the f86a BIM scale checkpoint and continue to a target step.

This is a narrow operational runner for the follow-up experiment:
resume ``f86a6903d32c4ab6`` from its 10K ``lm_continue`` checkpoint,
print frequent progress, and run/save evaluation checkpoints at 20K/40K.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.eval.utils import (
    clip_grad_norm,
    compute_perplexity,
    language_model_loss,
    make_adamw,
    make_batches,
)
from research.tools.run_bim_scale_experiment import (
    CANDIDATES,
    DB_PATH,
    RUNTIME_ROOT,
    TRAIN_TOKENS,
    VAL_TOKENS,
    append_row,
    base_row,
    build_model,
    count_parameters,
    fetch_candidate_record,
    historical_metrics,
    load_easy25_history,
    normalize_token_batches,
    run_final_probes,
    run_quick_probes,
    save_checkpoint,
)


F86A_FP = "f86a6903d32c4ab6"
DEFAULT_RESUME = (
    RUNTIME_ROOT
    / "bim_scale_20260508T130328Z"
    / "checkpoints"
    / F86A_FP
    / "lm_continue"
    / "step_010000.pt"
)


PROGRESS_FIELDS = (
    "run_id",
    "timestamp_utc",
    "elapsed_s",
    "step",
    "loss",
    "avg_loss_20",
    "train_ppl_20",
    "lr",
    "grad_norm",
    "tok_s",
    "status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_progress(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PROGRESS_FIELDS, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _tail_mean(values: list[float], n: int) -> float | None:
    if not values:
        return None
    tail = values[-n:]
    return float(sum(tail) / len(tail))


def _train_ppl(loss: float | None) -> float | None:
    if loss is None:
        return None
    return float(math.exp(min(float(loss), 20.0)))


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: str,
    *,
    load_optimizer: bool,
) -> int:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=False)
    opt_state = payload.get("optimizer_state")
    if load_optimizer and opt_state:
        optimizer.load_state_dict(opt_state)
    return int(payload.get("step") or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-checkpoint", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--target-step", type=int, default=40_000)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[20_000, 40_000])
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--save-progress-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-batches", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--quick-probes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--final-probes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--autocast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--mod-token-ids", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--load-optimizer", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.resume_checkpoint.exists():
        raise FileNotFoundError(args.resume_checkpoint)

    run_id = "bim_f86a_long_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNTIME_ROOT / run_id
    csv_path = RUNTIME_ROOT / f"{run_id}.csv"
    jsonl_path = RUNTIME_ROOT / f"{run_id}.jsonl"
    progress_path = RUNTIME_ROOT / f"{run_id}.progress.csv"
    checkpoint_dir = run_dir / "checkpoints"
    start_time = time.perf_counter()

    candidate = next(
        candidate for candidate in CANDIDATES if candidate.fingerprint == F86A_FP
    )
    conn = sqlite3.connect(DB_PATH)
    try:
        db_row = fetch_candidate_record(conn, candidate)
    finally:
        conn.close()
    history = historical_metrics(
        db_row, load_easy25_history().get(candidate.fingerprint)
    )

    train_tokens = np.load(TRAIN_TOKENS, mmap_mode="r")
    val_tokens = np.load(VAL_TOKENS, mmap_mode="r")
    val_batches = make_batches(
        val_tokens,
        args.batch_size,
        args.seq_len,
        args.val_batches,
        args.device,
        seed=args.seed + 999,
    )
    val_batches = normalize_token_batches(
        val_batches,
        100277,
        mod_token_ids=args.mod_token_ids,
    )

    model = build_model(
        db_row["graph_json"],
        candidate.d_model,
        12,
        100277,
        args.seq_len,
        core_dim=candidate.core_dim,
        scaled_shell=True,
    ).to(args.device)
    param_count = count_parameters(model)
    optimizer = make_adamw(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    step = _load_checkpoint(
        model,
        optimizer,
        args.resume_checkpoint,
        args.device,
        load_optimizer=args.load_optimizer,
    )
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    initial = base_row(
        run_id=run_id,
        start_time=start_time,
        candidate=candidate,
        args=argparse.Namespace(
            n_layers=12,
            vocab_size=100277,
            seq_len=args.seq_len,
        ),
        param_count=param_count,
        phase="branch",
        branch="lm_continue_long",
        event="resume",
        history=history,
    )
    initial.update(
        {"step": step, "checkpoint_path": str(args.resume_checkpoint), "status": "ok"}
    )
    append_row(csv_path, jsonl_path, initial)

    batches = make_batches(
        train_tokens,
        args.batch_size,
        args.seq_len,
        max(0, args.target_step - step),
        args.device,
        seed=args.seed + step,
    )
    losses: list[float] = []
    last_log_time = time.perf_counter()
    last_log_step = step
    autocast_enabled = bool(args.autocast and str(args.device).startswith("cuda"))
    eval_steps = set(int(value) for value in args.eval_steps)
    print(
        f"resuming {candidate.fingerprint} from step {step}; "
        f"target={args.target_step}; csv={csv_path}; progress={progress_path}",
        flush=True,
    )

    model.train()
    for batch in batches:
        step += 1
        if args.mod_token_ids:
            batch = torch.remainder(batch, 100277)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(batch)
            loss = language_model_loss(logits, batch, 100277)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.detach().item()}"
            )
        loss.backward()
        grad = clip_grad_norm(model.parameters(), args.clip_grad)
        optimizer.step()
        loss_value = float(loss.detach().item())
        grad_value = (
            float(grad.detach().item()) if torch.is_tensor(grad) else float(grad)
        )
        losses.append(loss_value)

        should_log = (
            step % args.print_every == 0
            or step in eval_steps
            or step == args.target_step
        )
        if should_log:
            now = time.perf_counter()
            delta_steps = max(1, step - last_log_step)
            tok_s = (
                delta_steps
                * args.batch_size
                * args.seq_len
                / max(now - last_log_time, 1e-9)
            )
            avg20 = _tail_mean(losses, 20)
            train_ppl = _train_ppl(avg20)
            line = (
                f"step={step} loss={loss_value:.4f} avg20={avg20:.4f} "
                f"train_ppl20={train_ppl:.1f} grad={grad_value:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} tok/s={tok_s:.0f}"
            )
            print(line, flush=True)
            if (
                step % args.save_progress_every == 0
                or step in eval_steps
                or step == args.target_step
            ):
                _write_progress(
                    progress_path,
                    {
                        "run_id": run_id,
                        "timestamp_utc": _utc_now(),
                        "elapsed_s": round(now - start_time, 3),
                        "step": step,
                        "loss": loss_value,
                        "avg_loss_20": avg20,
                        "train_ppl_20": train_ppl,
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_value,
                        "tok_s": round(tok_s, 1),
                        "status": "ok",
                    },
                )
            last_log_time = now
            last_log_step = step

        if step in eval_steps or step == args.target_step:
            val_ppl = compute_perplexity(model, val_batches, 100277)
            ckpt = checkpoint_dir / F86A_FP / "lm_continue_long" / f"step_{step:06d}.pt"
            save_checkpoint(
                ckpt,
                model,
                optimizer,
                candidate=candidate,
                args=argparse.Namespace(
                    n_layers=12,
                    vocab_size=100277,
                    seq_len=args.seq_len,
                    mod_token_ids=args.mod_token_ids,
                    scaled_shell=True,
                ),
                step=step,
                branch="lm_continue_long",
                param_count=param_count,
            )
            row = base_row(
                run_id=run_id,
                start_time=start_time,
                candidate=candidate,
                args=argparse.Namespace(
                    n_layers=12,
                    vocab_size=100277,
                    seq_len=args.seq_len,
                ),
                param_count=param_count,
                phase="branch",
                branch="lm_continue_long",
                event="checkpoint",
                history=history,
            )
            avg20 = _tail_mean(losses, 20)
            row.update(
                {
                    "step": step,
                    "branch_step": step - 10_000,
                    "train_loss": loss_value,
                    "train_loss_avg100": _tail_mean(losses, 100),
                    "train_loss_avg20": avg20,
                    "train_ppl_avg20": _train_ppl(avg20),
                    "val_ppl": val_ppl,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_value,
                    "checkpoint_path": str(ckpt),
                    "status": "ok",
                }
            )
            if args.quick_probes:
                row.update(
                    run_quick_probes(
                        model,
                        device=args.device,
                        seed=args.seed + step,
                        include_ar_validation=True,
                    )
                )
            if args.final_probes:
                row.update(
                    run_final_probes(
                        model, device=args.device, seed=args.seed, vocab_size=100277
                    )
                )
            append_row(csv_path, jsonl_path, row)
            model.train()
            print(
                f"eval step={step} val_ppl={val_ppl:.3f} checkpoint={ckpt}", flush=True
            )
        if step >= args.target_step:
            break

    summary = {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "progress_path": str(progress_path),
        "checkpoint_dir": str(checkpoint_dir),
        "elapsed_s": round(time.perf_counter() - start_time, 3),
        "status": "complete",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
