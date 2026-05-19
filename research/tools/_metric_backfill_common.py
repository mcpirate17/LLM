"""Shared helpers for metric smoke and backfill tools."""

from __future__ import annotations

import sqlite3
import subprocess
import threading
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
CORPUS_PATH = ROOT / "research" / "corpus" / "wikitext103_train.npy"

TRAJECTORY_COLUMNS = (
    "fp_jacobian_spectral_norm",
    "fp_jacobian_effective_rank",
    "fp_sensitivity_uniformity",
    "fp_spec_norm_status",
    "fp_metric_phase",
    "fp_jacobian_erf_density",
    "fp_jacobian_erf_variance",
    "fp_jacobian_erf_decay_slope",
    "fp_jacobian_erf_last_norm",
    "fp_jacobian_erf_first_norm",
    "fp_jacobian_erf_status",
    "fp_jacobian_erf_elapsed_ms",
    "fp_icld_velocity",
    "fp_icld_early_loss",
    "fp_icld_late_loss",
    "fp_icld_delta_loss",
    "fp_icld_seq_len",
    "fp_icld_status",
    "fp_icld_elapsed_ms",
    "fp_id_pr_early",
    "fp_id_pr_late",
    "fp_id_norm_early",
    "fp_id_norm_late",
    "fp_id_step_early",
    "fp_id_step_late",
    "fp_id_collapse_rate",
    "fp_id_collapse_rate_normalized",
    "fp_id_collapse_status",
    "fp_id_collapse_elapsed_ms",
    "fp_logit_margin_velocity",
    "fp_logit_margin_initial",
    "fp_logit_margin_final",
    "fp_logit_margin_delta",
    "fp_logit_margin_n_steps",
    "fp_logit_margin_status",
    "fp_logit_margin_elapsed_ms",
)


def load_projected_corpus(
    path: Path = CORPUS_PATH,
    *,
    vocab_size: int,
    max_tokens: int | None = 4_000_000,
) -> torch.Tensor:
    """Load a token npy and project tokens into the requested vocab."""
    if not path.exists():
        raise FileNotFoundError(f"corpus npy not found: {path}")
    arr = np.load(path, mmap_mode="r")
    if max_tokens is not None and arr.size > max_tokens:
        arr = arr[:max_tokens]
    tokens = torch.as_tensor(np.asarray(arr), dtype=torch.long)
    return tokens % int(vocab_size)


def sample_token_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    error_detail: str = "corpus too small",
) -> torch.Tensor:
    n = tokens.numel() - int(seq_len) - 1
    if n <= 0:
        raise ValueError(error_detail)
    starts = torch.randint(0, n, (int(batch_size),))
    return torch.stack(
        [tokens[s : s + int(seq_len) + 1] for s in starts.tolist()], dim=0
    ).to(device)


def train_next_token_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> float:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    if not torch.isfinite(loss):
        return float("nan")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def update_graph_runs_columns(
    conn: sqlite3.Connection,
    fingerprint: str,
    payload: dict,
    columns: tuple[str, ...],
    *,
    extra_where: str = "",
) -> int:
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    values = [payload.get(c) for c in columns]
    where = "WHERE graph_fingerprint = ?"
    if extra_where:
        where += f" {extra_where}"
    cursor = conn.execute(
        f"UPDATE graph_runs SET {set_clause} {where}",
        (*values, fingerprint),
    )
    return cursor.rowcount


def add_gpu_safety_args(
    parser: ArgumentParser,
    *,
    default_max_other_gpu_mib: int = 4096,
    default_gpu_memory_fraction: float = 0.5,
) -> None:
    parser.add_argument(
        "--max-other-gpu-mib",
        type=int,
        default=default_max_other_gpu_mib,
        help=(
            "Refuse to start if any other GPU process is using more than this "
            "many MiB of VRAM. Set to a large value to override (e.g. 30000)."
        ),
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=default_gpu_memory_fraction,
        help="Cap our process's CUDA memory at this fraction of the card.",
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="Sleep-poll until the GPU is quiet instead of exiting on busy.",
    )


def prepare_cuda_if_requested(
    *,
    device: str,
    max_other_gpu_mib: int,
    tool_name: str,
    wait_for_gpu: bool,
    gpu_memory_fraction: float | None = None,
) -> None:
    if not device.startswith("cuda"):
        return
    from research.tools._concurrency import assert_gpu_quiet, cap_gpu_memory

    assert_gpu_quiet(
        max_other_used_mib=max_other_gpu_mib,
        tool_name=tool_name,
        sleep_until_quiet=wait_for_gpu,
    )
    if gpu_memory_fraction is not None:
        cap_gpu_memory(fraction=gpu_memory_fraction)


@dataclass(slots=True)
class ParallelBackfillRuntime:
    ctx: Any
    pause_event: threading.Event
    stop_event: threading.Event
    governor: threading.Thread
    conn: sqlite3.Connection
    total: int
    t_start: float


def start_parallel_backfill_runtime(
    *,
    tasks: list,
    n_workers: int,
    gpu_memory_fraction: float,
    high_water_mib: int,
    low_water_mib: int,
    governor_poll_s: float,
    multiprocessing_module: Any,
    db_path: Path = DB_PATH,
) -> ParallelBackfillRuntime:
    print(
        f"[setup] starting {n_workers} workers; per-worker VRAM cap {gpu_memory_fraction:.0%}",
        flush=True,
    )

    ctx = multiprocessing_module.get_context("spawn")
    pause_event = threading.Event()
    stop_event = threading.Event()
    governor = threading.Thread(
        target=vram_governor,
        args=(stop_event, pause_event, high_water_mib, low_water_mib, governor_poll_s),
        daemon=True,
    )
    governor.start()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return ParallelBackfillRuntime(
        ctx=ctx,
        pause_event=pause_event,
        stop_event=stop_event,
        governor=governor,
        conn=conn,
        total=len(tasks),
        t_start=time.time(),
    )


def print_serial_commit_progress(
    conn: sqlite3.Connection,
    *,
    idx: int,
    total: int,
    succeeded: int,
    failed: int,
    propagated: int,
    t_start: float,
    eta_decimals: int = 0,
    flush: bool = True,
) -> None:
    conn.commit()
    elapsed = time.time() - t_start
    rate = succeeded / max(elapsed, 1e-6)
    eta_min = (total - idx) / max(rate, 1e-6) / 60.0
    print(
        f"  [{idx}/{total}] ok={succeeded} fail={failed} rows={propagated} "
        f"rate={rate:.2f}/s eta={eta_min:.{eta_decimals}f}m",
        flush=flush,
    )


def read_total_gpu_mib() -> int:
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            .stdout.strip()
            .splitlines()
        )
        return int(out[0]) if out else 0
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ):
        return 0


def vram_governor(
    stop_event: threading.Event,
    pause_event: threading.Event,
    high_water_mib: int,
    low_water_mib: int,
    poll_interval_s: float,
) -> None:
    """Pause dispatch while total GPU memory is above the high-water mark."""
    while not stop_event.is_set():
        used = read_total_gpu_mib()
        if used >= high_water_mib and not pause_event.is_set():
            print(
                f"[governor] VRAM at {used} MiB >= {high_water_mib} - pausing dispatch",
                flush=True,
            )
            pause_event.set()
        elif used <= low_water_mib and pause_event.is_set():
            print(
                f"[governor] VRAM at {used} MiB <= {low_water_mib} - resuming dispatch",
                flush=True,
            )
            pause_event.clear()
        stop_event.wait(timeout=poll_interval_s)
