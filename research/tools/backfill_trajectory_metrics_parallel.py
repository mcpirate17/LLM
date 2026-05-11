#!/usr/bin/env python
"""Parallel trained backfill of all Gemini trajectory metrics + spec_norm.

Design notes:

* **Multiprocessing, not threading.** PyTorch + CUDA + Python GIL together
  mean threads serialize on kernel launches; processes get independent
  CUDA contexts. Use ``torch.multiprocessing`` with the ``spawn`` start
  method so each worker gets a clean CUDA stack.

* **Sorted by cost, smallest first.** Per-fingerprint cost is dominated
  by graph_n_ops × graph_depth (training-step kernel size). Small
  architectures warm the worker pool quickly and let us measure real
  VRAM/throughput before the heavy fingerprints saturate the card.

* **VRAM-budget governor.** Each worker calls
  ``cap_gpu_memory(fraction=1/N - margin)`` so the allocator can't
  steal more than its share. The master also watches
  ``nvidia-smi --query-gpu=memory.used`` every N seconds; if total VRAM
  blows past a high-water mark, the master stops dispatching new tasks
  until usage drops.

* **Skip-when-complete.** ``_candidate_fingerprints`` filters out rows
  already at ``fp_metric_phase = 'screening_750'`` AND
  ``fp_id_collapse_rate IS NOT NULL`` (the canonical "fully measured"
  pair). Partial-completion fingerprints get re-trained from scratch —
  the dominant cost is the 750-step training, so per-metric skip
  inside a fingerprint isn't worth the complexity.

Usage::

    python -m research.tools.backfill_trajectory_metrics_parallel
    python -m research.tools.backfill_trajectory_metrics_parallel --workers 8
    python -m research.tools.backfill_trajectory_metrics_parallel --workers 12 --vram-cap-mib 28000
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

# Defer heavy imports to worker init so spawn doesn't fork-import them
# at master startup (slow on CUDA platforms).
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
CORPUS_PATH = ROOT / "research" / "corpus" / "wikitext103_train.npy"
TOOL_NAME = "backfill_trajectory_metrics_parallel"

SCREENING_STEPS = 750
ID_COLLAPSE_EARLY_STEP = 150
ID_COLLAPSE_LATE_STEP = 750
DEFAULT_SEQ_LEN = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_VOCAB = 32000
LR = 3e-4

_TRAJECTORY_COLUMNS = (
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

# Subset re-measured by --only-logit-margin mode. Backfill #2 case: ~1366
# rows have ``fp_logit_margin_status LIKE 'deepcopy_failed%'`` from the
# pre-state_dict-fix logit-margin probe. Re-running the full trajectory
# would be wasteful; this set is the minimum we have to write back.
_LOGIT_MARGIN_COLUMNS = (
    "fp_logit_margin_velocity",
    "fp_logit_margin_initial",
    "fp_logit_margin_final",
    "fp_logit_margin_delta",
    "fp_logit_margin_n_steps",
    "fp_logit_margin_status",
    "fp_logit_margin_elapsed_ms",
)


@dataclass(slots=True)
class FingerprintTask:
    fingerprint: str
    graph_json: str
    cost_score: int  # ops * depth — used to sort smallest-first


@dataclass(slots=True)
class TaskResult:
    fingerprint: str
    payload: dict | None
    error: str | None
    elapsed_s: float


# ─── Worker process ────────────────────────────────────────────────────


def _worker_init(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    gpu_memory_fraction: float,
    corpus_path: str,
    skip_training: bool,
    only_logit_margin: bool = False,
) -> None:
    """One-time setup per worker: import heavy stack, load corpus, cap VRAM."""
    global _WORKER_TOKENS, _WORKER_VOCAB, _WORKER_SEQ_LEN, _WORKER_BATCH
    global _WORKER_SKIP_TRAINING, _WORKER_ONLY_LOGIT_MARGIN

    # Heavy imports are deferred to worker init so the master process
    # stays lean and we don't pay 5 s of compiler-import in every
    # multiprocessing handshake.
    import research.synthesis.compiler  # noqa: F401

    if torch.cuda.is_available():
        from research.tools._concurrency import cap_gpu_memory

        cap_gpu_memory(fraction=gpu_memory_fraction)

    arr = np.load(corpus_path, mmap_mode="r")
    arr = arr[:4_000_000] if arr.size > 4_000_000 else arr
    tokens = torch.as_tensor(np.asarray(arr), dtype=torch.long)
    _WORKER_TOKENS = tokens % vocab_size
    _WORKER_VOCAB = vocab_size
    _WORKER_SEQ_LEN = seq_len
    _WORKER_BATCH = batch_size
    _WORKER_SKIP_TRAINING = skip_training
    _WORKER_ONLY_LOGIT_MARGIN = only_logit_margin


def _worker_sample_batch(device: torch.device) -> torch.Tensor:
    n = _WORKER_TOKENS.numel() - _WORKER_SEQ_LEN - 1
    starts = torch.randint(0, n, (_WORKER_BATCH,))
    return torch.stack(
        [_WORKER_TOKENS[s : s + _WORKER_SEQ_LEN + 1] for s in starts.tolist()],
        dim=0,
    ).to(device)


def _worker_train_step(
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


def _worker_measure(task: FingerprintTask) -> TaskResult:
    """Worker entry point — optionally train, then run metrics.

    When ``_WORKER_SKIP_TRAINING`` is set the worker measures the four
    v9-scoring metrics (ERF density, ERF variance, ICLD velocity, logit
    margin velocity) on the freshly-built model at random init. This is
    ~7.6× faster per fingerprint because the 750-step WikiText training
    only exists to enable ID Collapse Rate — and ID Collapse is not
    part of v9 composite scoring. Phase tag becomes ``"init"`` and ID
    Collapse stays NULL on these rows.

    When ``_WORKER_ONLY_LOGIT_MARGIN`` is set, only the transitive
    logit-margin probe runs and only logit-margin columns are
    returned — used by Backfill #2 to fix the ~1366 rows where the
    pre-state_dict-fix probe failed with deepcopy_failed status.
    """
    from research.synthesis import graph_from_json
    from research.synthesis.compiled_model import SynthesizedModel
    from research.eval.trajectory_metrics import (
        capture_hidden_state_snapshot,
        compute_trajectory_metrics,
    )

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        graph = graph_from_json(task.graph_json)
        model = SynthesizedModel(
            [graph], vocab_size=_WORKER_VOCAB, model_dim=graph.model_dim
        ).to(dev)

        # Logit-margin-only path: skip optimizer creation and the full
        # trajectory; just probe the metric we need.
        if _WORKER_ONLY_LOGIT_MARGIN:
            from research.eval.transitive_logit_margin import (
                compute_transitive_logit_margin,
            )

            margin = compute_transitive_logit_margin(model, device=str(dev))
            payload = margin.to_dict()
            return TaskResult(
                fingerprint=task.fingerprint,
                payload={k: payload.get(k) for k in _LOGIT_MARGIN_COLUMNS},
                error=None if margin.status == "ok" else f"status:{margin.status}",
                elapsed_s=time.time() - t0,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        early_snap = None
        late_snap = None
        if _WORKER_SKIP_TRAINING:
            metric_phase = "init"
        else:
            metric_phase = "screening_750"
            probe_ids = _worker_sample_batch(dev)[:, :_WORKER_SEQ_LEN]
            for step in range(1, SCREENING_STEPS + 1):
                batch = _worker_sample_batch(dev)
                _worker_train_step(model, optimizer, batch)
                if step == ID_COLLAPSE_EARLY_STEP:
                    early_snap = capture_hidden_state_snapshot(
                        model, probe_ids, step=step, device=str(dev)
                    )
                    model.train()
                if step == ID_COLLAPSE_LATE_STEP:
                    late_snap = capture_hidden_state_snapshot(
                        model, probe_ids, step=step, device=str(dev)
                    )
                    model.train()

        result = compute_trajectory_metrics(
            model,
            metric_phase=metric_phase,
            device=str(dev),
            spec_norm_vocab_size=_WORKER_VOCAB,
            id_collapse_early=early_snap,
            id_collapse_late=late_snap,
        )
        payload = result.to_column_dict()
        if payload.get("fp_jacobian_erf_status") in (None, "init"):
            return TaskResult(
                fingerprint=task.fingerprint,
                payload=None,
                error="trajectory_metrics_returned_init",
                elapsed_s=time.time() - t0,
            )
        filtered = {k: payload.get(k) for k in _TRAJECTORY_COLUMNS}
        return TaskResult(
            fingerprint=task.fingerprint,
            payload=filtered,
            error=None,
            elapsed_s=time.time() - t0,
        )
    except Exception as exc:
        # Catch broadly — Triton compilation errors, broken graph IR,
        # CUDA OOM, anything. A single bad fingerprint must not kill
        # the worker pool; report it as a fail and move on.
        return TaskResult(
            fingerprint=task.fingerprint,
            payload=None,
            error=f"{type(exc).__name__}: {str(exc)[:160]}",
            elapsed_s=time.time() - t0,
        )
    finally:
        # Best-effort cleanup so the worker doesn't accumulate CUDA cache
        # across hundreds of tasks.
        try:
            del model, optimizer  # type: ignore[name-defined]
        except UnboundLocalError:
            pass
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ─── Master process ────────────────────────────────────────────────────


def _candidate_tasks(
    conn: sqlite3.Connection,
    limit: int | None,
    *,
    skip_training: bool,
    only_logit_margin: bool = False,
    leaderboard_tiers: tuple[str, ...] = (),
    require_missing_id_collapse: bool = False,
) -> list[FingerprintTask]:
    """Pick fingerprints missing the v9-scoring metrics.

    When ``skip_training`` is set we treat any row with
    ``fp_jacobian_erf_density IS NULL`` as needing backfill — the four
    v9-scoring metrics either populate together or not at all in this
    mode.

    When ``skip_training`` is false (full trained backfill) we additionally
    require rows to be at ``fp_metric_phase = 'screening_750'`` AND have
    a populated ``fp_id_collapse_rate``; rows tagged ``init`` get
    re-measured at the proper screening phase.

    When ``only_logit_margin`` is set we filter for rows whose probe
    failed with the pre-state_dict-fix deepcopy bug. Mutually exclusive
    with the trained / skip-training paths.

    When ``leaderboard_tiers`` is non-empty, restrict candidates to
    fingerprints with at least one leaderboard row in those tiers.
    Used to target id-collapse backfill at investigated/validated
    architectures only.

    When ``require_missing_id_collapse`` is set, additionally require
    that no row under the fingerprint already has ``fp_id_collapse_rate``
    populated. Pairs with the trained backfill so we don't re-train
    fingerprints that already have a usable snapshot pair.
    """
    if only_logit_margin:
        filter_clause = "fp_logit_margin_status LIKE 'deepcopy_failed%'"
    elif skip_training:
        # Skip-training mode only requires the 4 v9-scoring metrics.
        # Any row missing erf_density needs the fast backfill.
        filter_clause = "fp_jacobian_erf_density IS NULL"
    else:
        filter_clause = (
            "fp_id_collapse_rate IS NULL "
            "OR fp_metric_phase IS NULL "
            "OR fp_metric_phase != 'screening_750'"
        )

    extra_clauses: list[str] = []
    if leaderboard_tiers:
        # Build a parameterised IN clause; SQLite allows a literal tier
        # list since these are short trusted strings, but we prefer a
        # parameter binding for safety.
        placeholders = ",".join(["?"] * len(leaderboard_tiers))
        extra_clauses.append(
            f"EXISTS (SELECT 1 FROM leaderboard lb_t "
            f"WHERE lb_t.result_id = pr.result_id "
            f"  AND lb_t.tier IN ({placeholders}))"
        )
    if require_missing_id_collapse:
        extra_clauses.append(
            "NOT EXISTS (SELECT 1 FROM program_results_compat pr_id "
            "WHERE pr_id.graph_fingerprint = pr.graph_fingerprint "
            "  AND pr_id.fp_id_collapse_rate IS NOT NULL)"
        )
    # length > 10 excludes the legacy "{}" placeholder graph_json that
    # ~160 of the phase=NULL rows carry — those have no recoverable
    # structure and would crash the worker on graph_from_json. Pick the
    # LONGEST sibling graph_json for the fingerprint, not the latest:
    # the latest under a placeholder-fingerprint may itself be "{}",
    # while an older sibling can carry the full graph.
    extra_sql = ("  AND " + " AND ".join(extra_clauses) + " ") if extra_clauses else ""
    sql = (
        "SELECT graph_fingerprint, "
        "       MAX(timestamp) AS ts, "
        "       MAX(COALESCE(graph_n_ops, 0)) AS n_ops, "
        "       MAX(COALESCE(graph_depth, 0)) AS depth, "
        "       (SELECT graph_json FROM program_results_compat pr2 "
        "         WHERE pr2.graph_fingerprint = pr.graph_fingerprint "
        "           AND pr2.graph_json IS NOT NULL "
        "           AND length(pr2.graph_json) > 10 "
        "         ORDER BY length(pr2.graph_json) DESC, "
        "                  pr2.timestamp DESC LIMIT 1) AS graph_json "
        "FROM program_results_compat pr "
        "WHERE graph_json IS NOT NULL AND length(graph_json) > 10 "
        "  AND graph_fingerprint IS NOT NULL "
        f"  AND ({filter_clause}) "
        f"{extra_sql}"
        "GROUP BY graph_fingerprint "
        "HAVING graph_json IS NOT NULL "
        "ORDER BY n_ops * depth ASC, ts DESC "
    )
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"

    params = list(leaderboard_tiers)
    tasks: list[FingerprintTask] = []
    for row in conn.execute(sql, params):
        fp, _ts, n_ops, depth, graph_json = row
        if not graph_json:
            continue
        graph_json = resolve_graph_json_value(conn, DB_PATH, graph_json)
        cost = max(1, int(n_ops or 1)) * max(1, int(depth or 1))
        tasks.append(
            FingerprintTask(
                fingerprint=fp,
                graph_json=graph_json,
                cost_score=cost,
            )
        )
    return tasks


def _propagate(
    conn: sqlite3.Connection,
    fingerprint: str,
    payload: dict,
    columns: tuple[str, ...] = _TRAJECTORY_COLUMNS,
) -> int:
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    values = [payload.get(c) for c in columns]
    cursor = conn.execute(
        f"UPDATE graph_runs SET {set_clause} WHERE graph_fingerprint = ?",
        (*values, fingerprint),
    )
    return cursor.rowcount


def _read_total_gpu_mib() -> int:
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


def _vram_governor(
    stop_event: threading.Event,
    pause_event: threading.Event,
    high_water_mib: int,
    low_water_mib: int,
    poll_interval_s: float,
) -> None:
    """Background thread: pause task dispatch when VRAM exceeds high water,
    resume when it drops below low water. Both thresholds expressed in MiB.
    """
    while not stop_event.is_set():
        used = _read_total_gpu_mib()
        if used >= high_water_mib and not pause_event.is_set():
            print(
                f"[governor] VRAM at {used} MiB ≥ {high_water_mib} — pausing dispatch",
                flush=True,
            )
            pause_event.set()
        elif used <= low_water_mib and pause_event.is_set():
            print(
                f"[governor] VRAM at {used} MiB ≤ {low_water_mib} — resuming dispatch",
                flush=True,
            )
            pause_event.clear()
        stop_event.wait(timeout=poll_interval_s)


def _run_parallel(
    tasks: list[FingerprintTask],
    *,
    n_workers: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    gpu_memory_fraction: float,
    high_water_mib: int,
    low_water_mib: int,
    governor_poll_s: float,
    commit_every: int,
    skip_training: bool,
    only_logit_margin: bool = False,
) -> None:
    if not tasks:
        return

    print(
        f"[setup] starting {n_workers} workers; per-worker VRAM cap {gpu_memory_fraction:.0%}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    pause_event = threading.Event()
    stop_event = threading.Event()
    governor = threading.Thread(
        target=_vram_governor,
        args=(stop_event, pause_event, high_water_mib, low_water_mib, governor_poll_s),
        daemon=True,
    )
    governor.start()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    succeeded = 0
    failed = 0
    propagated = 0
    total = len(tasks)
    t_start = time.time()

    pool = ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(
            vocab_size,
            seq_len,
            batch_size,
            gpu_memory_fraction,
            str(CORPUS_PATH),
            skip_training,
            only_logit_margin,
        ),
    )

    update_columns = _LOGIT_MARGIN_COLUMNS if only_logit_margin else _TRAJECTORY_COLUMNS

    try:
        # Submit all tasks; imap_unordered yields results as they complete.
        # The governor pauses the master from dispatching MORE tasks if VRAM
        # gets hot — workers in flight finish naturally.
        result_iter = pool.imap_unordered(_worker_measure, tasks, chunksize=1)
        for idx, result in enumerate(result_iter, start=1):
            # If governor signaled pause, sleep before processing more
            # results — gives the workers time to finish in-flight tasks
            # and frees VRAM. Workers themselves keep going since we
            # already submitted everything; the pause is advisory.
            while pause_event.is_set():
                time.sleep(2.0)

            if result.payload is None:
                failed += 1
                if result.error:
                    print(
                        f"  [{idx}/{total}] {result.fingerprint[:12]} fail: "
                        f"{result.error}",
                        flush=True,
                    )
                continue

            rowcount = _propagate(
                conn, result.fingerprint, result.payload, update_columns
            )
            succeeded += 1
            propagated += rowcount

            if succeeded % commit_every == 0:
                conn.commit()
                elapsed = time.time() - t_start
                rate = succeeded / max(elapsed, 1e-6)
                eta_min = (total - idx) / max(rate, 1e-6) / 60.0
                used = _read_total_gpu_mib()
                print(
                    f"  [{idx}/{total}] ok={succeeded} fail={failed} "
                    f"rows={propagated} rate={rate:.2f}/s eta={eta_min:.0f}m "
                    f"vram={used}MiB",
                    flush=True,
                )
    finally:
        pool.close()
        pool.join()
        stop_event.set()
        governor.join(timeout=5.0)
        conn.commit()
        conn.close()

    elapsed = time.time() - t_start
    print(
        f"[done] ok={succeeded} fail={failed} rows_updated={propagated} "
        f"elapsed={elapsed / 60:.1f}m",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commit-every", type=int, default=20)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=None,
        help="Per-worker VRAM cap. Default = 0.85 / workers (leaves 15% headroom).",
    )
    parser.add_argument(
        "--vram-cap-mib",
        type=int,
        default=28000,
        help="High-water mark (MiB). Master pauses dispatch if total VRAM exceeds this.",
    )
    parser.add_argument(
        "--vram-resume-mib",
        type=int,
        default=22000,
        help="Low-water mark (MiB). Master resumes dispatch when total VRAM drops below this.",
    )
    parser.add_argument("--governor-poll-s", type=float, default=5.0)
    parser.add_argument("--max-other-gpu-mib", type=int, default=4096)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip the 750-step WikiText training. Populates ERF density / "
            "variance, ICLD velocity, logit-margin velocity (the 4 metrics "
            "v9 composite uses) at random init weights. ID Collapse Rate "
            "stays NULL on these rows. ~7.6× faster per fingerprint."
        ),
    )
    parser.add_argument(
        "--only-logit-margin",
        action="store_true",
        help=(
            "Only re-measure transitive logit-margin velocity, on rows "
            "whose status starts with 'deepcopy_failed'. Used by "
            "Backfill #2 — repairs the ~1366 rows where the pre-state_dict "
            "fix probe failed. Mutually exclusive with --skip-training "
            "(this flag also implies skipping the training loop)."
        ),
    )
    parser.add_argument(
        "--leaderboard-tiers",
        default="",
        help=(
            "Comma-separated leaderboard tiers — restrict candidates to "
            "fingerprints with at least one row in those tiers. Example: "
            "'investigation,validation' targets id-collapse backfill at "
            "investigated/validated architectures only."
        ),
    )
    parser.add_argument(
        "--require-missing-id-collapse",
        action="store_true",
        help=(
            "Skip fingerprints whose id_collapse_rate is already populated "
            "anywhere in program_results — propagation will copy the value "
            "to siblings. Pairs with the trained backfill so we don't "
            "re-train architectures that already have a usable snapshot pair."
        ),
    )
    args = parser.parse_args()
    if args.only_logit_margin and args.skip_training:
        parser.error("--only-logit-margin already implies skip-training")
    if args.only_logit_margin:
        args.skip_training = True  # downstream paths gate on this too
    leaderboard_tiers = tuple(
        t.strip() for t in args.leaderboard_tiers.split(",") if t.strip()
    )

    # Heavy imports at master level — workers will redo them via spawn but
    # we need them here for type registry warmup (avoids deadlock during
    # first model rebuild in some PyTorch versions).
    import research.synthesis.compiler  # noqa: F401
    from research.tools._concurrency import (
        acquire_gpu_lock,
        acquire_writer_lock,
        assert_gpu_quiet,
    )

    if torch.cuda.is_available():
        assert_gpu_quiet(
            max_other_used_mib=args.max_other_gpu_mib,
            tool_name=TOOL_NAME,
            sleep_until_quiet=args.wait_for_gpu,
        )

    if args.gpu_memory_fraction is None:
        # Leave 15% of the card for system/desktop/headroom; split the rest
        # evenly across workers. This lets us run more workers safely than
        # the naive 1/N split would.
        args.gpu_memory_fraction = max(0.05, min(0.5, 0.85 / args.workers))

    with (
        acquire_gpu_lock(tool_name=TOOL_NAME),
        acquire_writer_lock(tool_name=TOOL_NAME),
    ):
        conn = sqlite3.connect(str(DB_PATH))
        tasks = _candidate_tasks(
            conn,
            args.limit,
            skip_training=args.skip_training,
            only_logit_margin=args.only_logit_margin,
            leaderboard_tiers=leaderboard_tiers,
            require_missing_id_collapse=args.require_missing_id_collapse,
        )
        conn.close()
        total = len(tasks)
        print(
            f"[setup] backfill scope: {total} unique fingerprints "
            f"(sorted ascending by graph cost)",
            flush=True,
        )
        if total == 0:
            return

        # Show cost-distribution histogram so we know what the worker pool
        # is biting off.
        if total >= 10:
            costs = sorted(t.cost_score for t in tasks)
            print(
                f"[setup] cost distribution: "
                f"p10={costs[total // 10]} "
                f"p50={costs[total // 2]} "
                f"p90={costs[(9 * total) // 10]} "
                f"max={costs[-1]}",
                flush=True,
            )

        _run_parallel(
            tasks,
            n_workers=args.workers,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            gpu_memory_fraction=args.gpu_memory_fraction,
            high_water_mib=args.vram_cap_mib,
            low_water_mib=args.vram_resume_mib,
            governor_poll_s=args.governor_poll_s,
            commit_every=args.commit_every,
            skip_training=args.skip_training,
            only_logit_margin=args.only_logit_margin,
        )


if __name__ == "__main__":
    main()
