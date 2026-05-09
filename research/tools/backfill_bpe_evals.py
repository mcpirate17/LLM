"""BPE-eval backfill: train each fingerprint on cl100k_base BPE then re-eval.

The wikitext perplexity, BLiMP accuracy, HellaSwag accuracy, and TinyStories
perplexity columns in program_results were all measured under byte
tokenization (research/eval/utils.py:tokenize_string defaulted to UTF-8
bytes). Training was switched to cl100k_base BPE on 2026-03-22 but the
eval module never followed. Result: every eval column is garbage for
BPE-trained models.

This tool walks fingerprints sorted by composite_score DESC, rebuilds each
model from graph_json with vocab_size=100277, trains it on the BPE corpus
(research/corpus/wikitext103_train.npy, 120M cl100k tokens), then
re-evaluates with BPE tokenization and writes the new numbers back to
program_results.

Usage::

    # Performance sweep
    python -m research.tools.backfill_bpe_evals --workers 2 --limit 6 --tier validation,breakthrough
    python -m research.tools.backfill_bpe_evals --workers 4 --limit 6 --tier validation,breakthrough
    python -m research.tools.backfill_bpe_evals --workers 6 --limit 6 --tier validation,breakthrough

    # Production runs (highest composite first)
    python -m research.tools.backfill_bpe_evals --workers <best> --tier validation,breakthrough
    python -m research.tools.backfill_bpe_evals --workers <best> --tier investigation,investigation_failed
    python -m research.tools.backfill_bpe_evals --workers <best> --tier screening
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
CORPUS_PATH = ROOT / "research" / "corpus" / "wikitext103_train.npy"
TOOL_NAME = "backfill_bpe_evals"

DEFAULT_VOCAB = 100277  # cl100k_base
DEFAULT_TRAIN_STEPS = 750
DEFAULT_SEQ_LEN = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 3e-4
DEFAULT_VAL_TOKENS = 200_000  # ~200K BPE val tokens, last slice of corpus
DEFAULT_BLIMP_PER_SUBTASK = 25
DEFAULT_HELLASWAG_LIMIT = 200

_BPE_EVAL_COLUMNS = (
    "wikitext_perplexity",
    "wikitext_score",
    "wikitext_pre_perplexity",
    "wikitext_ppl_improvement",
    "wikitext_eval_steps",
    "blimp_overall_accuracy",
    "blimp_n_subtasks",
    "blimp_status",
    "hellaswag_acc",
    "hellaswag_n_examples",
    "hellaswag_status",
    "hellaswag_metric_version",
    "hellaswag_tokenizer_mode",
    "hellaswag_tiktoken_encoding",
    "tinystories_perplexity",
    "tinystories_score",
    "screening_wikitext_metric_version",
)
_NEW_METRIC_VERSION = "bpe_eval_v1"


@dataclass(slots=True)
class FingerprintTask:
    fingerprint: str
    graph_json: str
    cost_score: int


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
    train_steps: int,
    lr: float,
    val_tokens: int,
    gpu_memory_fraction: float,
    corpus_path: str,
    blimp_per_subtask: int,
    hellaswag_limit: int,
) -> None:
    """One-time setup per worker: imports, BPE corpus mmap, monkey-patch tokenizer."""
    global _W_VOCAB, _W_SEQ_LEN, _W_BATCH, _W_TRAIN_STEPS, _W_LR, _W_VAL_TOKENS
    global _W_TRAIN_TOKENS, _W_VAL_TOKEN_TENSOR
    global _W_BLIMP_PER_SUBTASK, _W_HELLASWAG_LIMIT

    import research.synthesis.compiler  # noqa: F401  — populates OP_DISPATCH

    if torch.cuda.is_available():
        from research.tools._concurrency import cap_gpu_memory

        cap_gpu_memory(fraction=gpu_memory_fraction)

    arr = np.load(corpus_path, mmap_mode="r")
    n_total = int(arr.size)
    val_n = int(min(val_tokens, max(1024, n_total // 100)))
    # Last `val_n` tokens are val (stable across workers).
    train_arr = arr[: n_total - val_n]
    val_arr = arr[n_total - val_n :]
    _W_TRAIN_TOKENS = torch.as_tensor(np.asarray(train_arr), dtype=torch.long)
    _W_VAL_TOKEN_TENSOR = torch.as_tensor(np.asarray(val_arr), dtype=torch.long)

    _W_VOCAB = vocab_size
    _W_SEQ_LEN = seq_len
    _W_BATCH = batch_size
    _W_TRAIN_STEPS = train_steps
    _W_LR = lr
    _W_VAL_TOKENS = val_n
    _W_BLIMP_PER_SUBTASK = blimp_per_subtask
    _W_HELLASWAG_LIMIT = hellaswag_limit

    # Monkey-patch the byte tokenizer in BLiMP / HellaSwag / TinyStories
    # modules so they pick up the BPE path without a deep refactor of
    # each eval module's call sites.
    _patch_eval_tokenizers()


def _patch_eval_tokenizers() -> None:
    """Replace tokenize_string in eval modules with a BPE-defaulting variant."""
    from research.eval import utils as _eu

    def _bpe_tokenize_string(text, vocab_size):
        return _eu.tokenize_string(text, vocab_size, tokenizer="tiktoken")

    def _bpe_tokenize_file(path, vocab_size):
        return _eu.tokenize_file(path, vocab_size, tokenizer="tiktoken")

    # Patch the bound names in modules that imported tokenize_string / file.
    for mod_name in (
        "research.eval.blimp_eval",
        "research.eval.hellaswag_eval",
        "research.eval.corpus_pipeline",
        "research.eval.tinystories_eval",
    ):
        try:
            mod = __import__(mod_name, fromlist=["tokenize_string", "tokenize_file"])
            if hasattr(mod, "tokenize_string"):
                mod.tokenize_string = _bpe_tokenize_string
            if hasattr(mod, "tokenize_file"):
                mod.tokenize_file = _bpe_tokenize_file
        except ImportError:
            pass


def _sample_train_batch(device: torch.device) -> torch.Tensor:
    n = _W_TRAIN_TOKENS.numel() - _W_SEQ_LEN - 1
    starts = torch.randint(0, n, (_W_BATCH,))
    return torch.stack(
        [_W_TRAIN_TOKENS[s : s + _W_SEQ_LEN + 1] for s in starts.tolist()],
        dim=0,
    ).to(device)


def _train_step(model, optimizer, batch):
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


def _measure_zero_shot_ppl(model, device: torch.device) -> tuple[float | None, int]:
    """Compute next-token CE on the held-out BPE val tokens. No training."""
    model.eval()
    val = _W_VAL_TOKEN_TENSOR
    seq_len = _W_SEQ_LEN
    batch_size = _W_BATCH
    # Make non-overlapping windows over the entire val slice.
    n_windows = max(1, (val.numel() - 1) // seq_len)
    n_windows = min(n_windows, 256)  # cap eval at ~256 windows for speed
    starts = torch.arange(0, n_windows * seq_len, seq_len)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, n_windows, batch_size):
            chunk = starts[i : i + batch_size]
            batch = torch.stack(
                [val[s : s + seq_len + 1] for s in chunk.tolist()],
                dim=0,
            ).to(device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                total_tokens += batch[:, 1:].numel()
    if total_tokens == 0:
        return None, 0
    import math

    mean_nats = total_loss / total_tokens
    return math.exp(min(mean_nats, 20.0)), total_tokens


def _worker_measure(task: FingerprintTask) -> TaskResult:
    """Build → train BPE → eval BPE perplexity + blimp + hellaswag + tinystories."""
    from research.synthesis import graph_from_json
    from research.synthesis.compiled_model import SynthesizedModel
    from research.eval.blimp_eval import evaluate_blimp
    from research.eval.hellaswag_eval import evaluate_hellaswag
    from research.eval.tinystories_eval import evaluate_tinystories

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        graph = graph_from_json(task.graph_json)
        model_dim = getattr(graph, "model_dim", None) or 256
        model = SynthesizedModel([graph], vocab_size=_W_VOCAB, model_dim=model_dim).to(
            dev
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=_W_LR)

        # Pre-train ppl (on val, no training yet)
        pre_ppl, _ = _measure_zero_shot_ppl(model, dev)

        # Train on BPE corpus
        model.train()
        for step in range(1, _W_TRAIN_STEPS + 1):
            batch = _sample_train_batch(dev)
            _train_step(model, optimizer, batch)

        # Zero-shot eval on BPE val
        post_ppl, n_val_tok = _measure_zero_shot_ppl(model, dev)

        # BLiMP (zero-shot, monkey-patched to use BPE)
        try:
            blimp = evaluate_blimp(
                model,
                vocab_size=_W_VOCAB,
                device=str(dev),
                n_per_subtask=_W_BLIMP_PER_SUBTASK,
            )
            blimp_acc = blimp.overall_accuracy
            blimp_n = blimp.n_subtasks
            blimp_status = blimp.status
        except Exception as exc:
            blimp_acc, blimp_n, blimp_status = (
                None,
                None,
                f"failed:{type(exc).__name__}",
            )

        # HellaSwag (zero-shot, monkey-patched to use BPE)
        try:
            hella = evaluate_hellaswag(
                model,
                vocab_size=_W_VOCAB,
                device=str(dev),
                n_examples=_W_HELLASWAG_LIMIT,
            )
            hellaswag_acc = hella.get("hellaswag_acc")
            hellaswag_n = hella.get("hellaswag_n_examples")
            hellaswag_status = hella.get("hellaswag_status")
            hellaswag_metric_version = hella.get("hellaswag_metric_version")
            hellaswag_tokenizer_mode = hella.get("hellaswag_tokenizer_mode")
            hellaswag_tiktoken_encoding = hella.get("hellaswag_tiktoken_encoding")
        except Exception as exc:
            hellaswag_acc = None
            hellaswag_n = None
            hellaswag_status = f"failed:{type(exc).__name__}"
            hellaswag_metric_version = None
            hellaswag_tokenizer_mode = None
            hellaswag_tiktoken_encoding = None

        # HellaSwag's native scorer leaks inference-mode tensors into
        # module buffers. Clone them out before tinystories micro-training
        # so backprop doesn't explode.
        n_decontam = _decontaminate_inference_tensors(model)
        if n_decontam:
            print(
                f"[ts-decontam] {task.fingerprint[:12]} cloned {n_decontam} inference tensors",
                file=sys.stderr,
                flush=True,
            )

        # TinyStories perplexity (uses corpus_pipeline → tokenize_file
        # which we monkey-patched, so it now BPE-tokenizes its own corpus
        # — value will reflect BPE PPL on tinystories text).
        try:
            ts = evaluate_tinystories(
                model,
                _W_VOCAB,
                str(dev),
                n_train_steps=200,
                seq_len=_W_SEQ_LEN,
            )
            ts_ppl = ts.get("tinystories_perplexity")
            ts_score = ts.get("tinystories_score")
            if ts_ppl is None:
                err = ts.get("error") or "post_ppl=None (likely diverged)"
                print(
                    f"[ts-fail] {task.fingerprint[:12]} {err}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            ts_ppl, ts_score = None, None
            print(
                f"[ts-fail] {task.fingerprint[:12]} {type(exc).__name__}: {str(exc)[:200]}",
                file=sys.stderr,
                flush=True,
            )

        improvement = None
        if pre_ppl is not None and post_ppl is not None and pre_ppl > 0:
            improvement = round(post_ppl / pre_ppl, 4)

        payload = {
            "wikitext_perplexity": round(post_ppl, 2) if post_ppl else None,
            "wikitext_pre_perplexity": round(pre_ppl, 2) if pre_ppl else None,
            "wikitext_score": _wikitext_score(post_ppl, _W_VOCAB),
            "wikitext_ppl_improvement": improvement,
            "wikitext_eval_steps": _W_TRAIN_STEPS,
            "blimp_overall_accuracy": blimp_acc,
            "blimp_n_subtasks": blimp_n,
            "blimp_status": blimp_status,
            "hellaswag_acc": hellaswag_acc,
            "hellaswag_n_examples": hellaswag_n,
            "hellaswag_status": hellaswag_status,
            "hellaswag_metric_version": hellaswag_metric_version,
            "hellaswag_tokenizer_mode": hellaswag_tokenizer_mode,
            "hellaswag_tiktoken_encoding": hellaswag_tiktoken_encoding,
            "tinystories_perplexity": ts_ppl,
            "tinystories_score": ts_score,
            "screening_wikitext_metric_version": _NEW_METRIC_VERSION,
        }
        return TaskResult(
            fingerprint=task.fingerprint,
            payload=payload,
            error=None,
            elapsed_s=time.time() - t0,
        )
    except Exception as exc:
        return TaskResult(
            fingerprint=task.fingerprint,
            payload=None,
            error=f"{type(exc).__name__}: {str(exc)[:160]}",
            elapsed_s=time.time() - t0,
        )
    finally:
        if model is not None:
            del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def _decontaminate_inference_tensors(model: torch.nn.Module) -> int:
    """Replace any inference-mode tensors stored in module buffers or plain
    attributes with normal-mode clones. HellaSwag's native scorer
    (research/eval/_eval_native.cpp) wraps its forward pass in
    ``c10::InferenceMode guard(true)``; any state buffer mutated under that
    guard becomes an inference tensor, and a later autograd-tracked op (e.g.
    tinystories micro-training) raises ``RuntimeError: Inference tensors
    cannot be saved for backward``. Returns the number of tensors replaced."""
    n = 0
    skip_attrs = {
        "_parameters",
        "_buffers",
        "_modules",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_non_persistent_buffers_set",
        "training",
    }
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_inference():
                module._buffers[name] = buf.clone()
                n += 1
        for attr_name, val in list(vars(module).items()):
            if attr_name in skip_attrs or attr_name.startswith("__"):
                continue
            if isinstance(val, torch.Tensor) and val.is_inference():
                setattr(module, attr_name, val.clone())
                n += 1
    return n


def _wikitext_score(ppl: float | None, vocab_size: int) -> float | None:
    if ppl is None or ppl <= 0:
        return None
    import math

    # Same shape as eval/wikitext_eval.py:wikitext_score_from_ppl —
    # bounded [0, 1], higher is better.
    return round(max(0.0, min(1.0, 1.0 - math.log(ppl) / math.log(vocab_size))), 4)


# ─── Master process ────────────────────────────────────────────────────


def _candidate_tasks(
    conn: sqlite3.Connection,
    tiers: tuple[str, ...],
    limit: int | None,
    *,
    include_current_bpe: bool,
    scope: str,
    trust_label: str | None,
    comparability_label: str | None,
) -> list[FingerprintTask]:
    """Pick fingerprints in the requested tiers, sorted by composite DESC."""
    params: list[object] = []
    join = "LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id"
    where = ["pr.graph_fingerprint IS NOT NULL"]
    if scope == "leaderboard":
        placeholders = ",".join(["?"] * len(tiers))
        join = "JOIN leaderboard lb ON lb.result_id = pr.result_id"
        where.append(f"lb.tier IN ({placeholders})")
        params.extend(tiers)
    elif scope == "off_leaderboard":
        where.append("lb.entry_id IS NULL")
    elif scope != "all":
        raise ValueError(f"unsupported scope: {scope}")
    if trust_label:
        where.append("COALESCE(pr.trust_label, '') = ?")
        params.append(trust_label)
    if comparability_label:
        where.append("COALESCE(pr.comparability_label, '') = ?")
        params.append(comparability_label)

    sql = (
        "SELECT pr.graph_fingerprint, "
        "       MAX(COALESCE(lb.composite_score, 1.0 / NULLIF(pr.loss_ratio, 0), 0.0)) AS comp, "
        "       MAX(COALESCE(pr.graph_n_ops, 0)) AS n_ops, "
        "       MAX(COALESCE(pr.graph_depth, 0)) AS depth, "
        "       (SELECT graph_json FROM program_results pr2 "
        "         WHERE pr2.graph_fingerprint = pr.graph_fingerprint "
        "           AND pr2.graph_json IS NOT NULL "
        "           AND length(pr2.graph_json) > 10 "
        "         ORDER BY length(pr2.graph_json) DESC, pr2.timestamp DESC LIMIT 1"
        "       ) AS graph_json "
        "FROM program_results pr "
        f"{join} "
    )
    if not include_current_bpe:
        where.append(
            "(COALESCE(pr.screening_wikitext_metric_version, '') <> ? "
            "       OR pr.wikitext_perplexity IS NULL "
            "       OR pr.tinystories_perplexity IS NULL "
            "       OR pr.hellaswag_acc IS NULL "
            "       OR pr.blimp_overall_accuracy IS NULL)"
        )
        params.append(_NEW_METRIC_VERSION)
    sql += "WHERE " + " AND ".join(where) + " "
    sql += (
        "GROUP BY pr.graph_fingerprint "
        "HAVING graph_json IS NOT NULL "
        "ORDER BY comp DESC NULLS LAST"
    )
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"

    tasks: list[FingerprintTask] = []
    for row in conn.execute(sql, params):
        fp, comp, n_ops, depth, gj = row
        if not gj:
            continue
        tasks.append(
            FingerprintTask(
                fingerprint=fp,
                graph_json=resolve_graph_json_value(conn, DB_PATH, gj),
                cost_score=max(1, int(n_ops or 1)) * max(1, int(depth or 1)),
            )
        )
    return tasks


def _propagate(
    conn: sqlite3.Connection,
    fingerprint: str,
    payload: dict,
) -> int:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(program_results)")}
    columns = [c for c in _BPE_EVAL_COLUMNS if c in payload and c in existing]
    if not columns:
        return 0
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    values = [payload.get(c) for c in columns]
    cursor = conn.execute(
        f"UPDATE program_results SET {set_clause} WHERE graph_fingerprint = ?",
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
    train_steps: int,
    lr: float,
    val_tokens: int,
    gpu_memory_fraction: float,
    high_water_mib: int,
    low_water_mib: int,
    governor_poll_s: float,
    commit_every: int,
    blimp_per_subtask: int,
    hellaswag_limit: int,
) -> dict:
    """Returns dict with timing + counts so callers can do perf comparisons."""
    if not tasks:
        return {"ok": 0, "fail": 0, "rows": 0, "elapsed_s": 0, "rate": 0}

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
    peak_vram = 0

    pool = ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(
            vocab_size,
            seq_len,
            batch_size,
            train_steps,
            lr,
            val_tokens,
            gpu_memory_fraction,
            str(CORPUS_PATH),
            blimp_per_subtask,
            hellaswag_limit,
        ),
    )

    try:
        result_iter = pool.imap_unordered(_worker_measure, tasks, chunksize=1)
        for idx, result in enumerate(result_iter, start=1):
            while pause_event.is_set():
                time.sleep(2.0)

            if result.payload is None:
                failed += 1
                if result.error:
                    print(
                        f"  [{idx}/{total}] {result.fingerprint[:12]} fail: {result.error}",
                        flush=True,
                    )
                continue

            rowcount = _propagate(conn, result.fingerprint, result.payload)
            succeeded += 1
            propagated += rowcount

            if succeeded % commit_every == 0 or idx == total:
                conn.commit()
                elapsed = time.time() - t_start
                rate = succeeded / max(elapsed, 1e-6)
                eta_min = (total - idx) / max(rate, 1e-6) / 60.0
                used = _read_total_gpu_mib()
                peak_vram = max(peak_vram, used)
                # Pretty preview of one row
                p = result.payload
                print(
                    f"  [{idx}/{total}] ok={succeeded} fail={failed} rows={propagated} "
                    f"rate={rate:.2f}/s eta={eta_min:.0f}m vram={used}MiB | "
                    f"wt_ppl={p.get('wikitext_perplexity')} blimp={p.get('blimp_overall_accuracy')} "
                    f"hella={p.get('hellaswag_acc')} ts={p.get('tinystories_perplexity')}",
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
    rate = succeeded / max(elapsed, 1e-6)
    print(
        f"[done] ok={succeeded} fail={failed} rows_updated={propagated} "
        f"elapsed={elapsed / 60:.1f}m peak_vram={peak_vram}MiB rate={rate:.2f}/s",
        flush=True,
    )
    return {
        "ok": succeeded,
        "fail": failed,
        "rows": propagated,
        "elapsed_s": elapsed,
        "rate": rate,
        "peak_vram_mib": peak_vram,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    parser.add_argument(
        "--blimp-per-subtask", type=int, default=DEFAULT_BLIMP_PER_SUBTASK
    )
    parser.add_argument("--hellaswag-limit", type=int, default=DEFAULT_HELLASWAG_LIMIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected fingerprints without running evals.",
    )
    parser.add_argument("--commit-every", type=int, default=10)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--vram-cap-mib", type=int, default=28000)
    parser.add_argument("--vram-resume-mib", type=int, default=22000)
    parser.add_argument("--governor-poll-s", type=float, default=5.0)
    parser.add_argument("--max-other-gpu-mib", type=int, default=4096)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument(
        "--include-current-bpe",
        action="store_true",
        help="Also rerun fingerprints that already have complete bpe_eval_v1 metrics.",
    )
    parser.add_argument(
        "--tier",
        default="validation,breakthrough",
        help="Comma-separated leaderboard tiers for --scope leaderboard.",
    )
    parser.add_argument(
        "--scope",
        choices=("leaderboard", "off_leaderboard", "all"),
        default="leaderboard",
        help="Fingerprint pool to scan before applying stale/missing BPE filters.",
    )
    parser.add_argument("--trust-label", default=None)
    parser.add_argument("--comparability-label", default=None)
    args = parser.parse_args()

    import research.synthesis.compiler  # noqa: F401  — warm registry
    from research.tools._concurrency import (
        acquire_gpu_lock,
        acquire_writer_lock,
        assert_gpu_quiet,
    )
    from research.tools.db_health import assert_sqlite_health

    if torch.cuda.is_available() and not args.dry_run:
        assert_gpu_quiet(
            max_other_used_mib=args.max_other_gpu_mib,
            tool_name=TOOL_NAME,
            sleep_until_quiet=args.wait_for_gpu,
        )

    if args.gpu_memory_fraction is None:
        args.gpu_memory_fraction = max(0.05, min(0.5, 0.85 / args.workers))

    tiers = tuple(t.strip() for t in args.tier.split(",") if t.strip())

    gpu_lock = nullcontext() if args.dry_run else acquire_gpu_lock(tool_name=TOOL_NAME)
    writer_lock = (
        nullcontext() if args.dry_run else acquire_writer_lock(tool_name=TOOL_NAME)
    )
    with (
        gpu_lock,
        writer_lock,
    ):
        assert_sqlite_health(DB_PATH, label="pre-bpe-backfill")
        conn = sqlite3.connect(str(DB_PATH))
        tasks = _candidate_tasks(
            conn,
            tiers,
            args.limit,
            include_current_bpe=args.include_current_bpe,
            scope=args.scope,
            trust_label=args.trust_label,
            comparability_label=args.comparability_label,
        )
        conn.close()
        total = len(tasks)
        print(
            f"[setup] BPE eval backfill scope: {total} unique fingerprints "
            f"(scope={args.scope}, tiers={tiers}, sorted by priority DESC)",
            flush=True,
        )
        if args.dry_run:
            for task in tasks[:25]:
                print(
                    f"  dry_run fingerprint={task.fingerprint} cost={task.cost_score}",
                    flush=True,
                )
            if total > 25:
                print(f"  dry_run omitted={total - 25}", flush=True)
            return
        if total == 0:
            return

        if total >= 10:
            costs = sorted(t.cost_score for t in tasks)
            print(
                f"[setup] cost distribution: "
                f"p10={costs[total // 10]} p50={costs[total // 2]} "
                f"p90={costs[(9 * total) // 10]} max={costs[-1]}",
                flush=True,
            )

        _run_parallel(
            tasks,
            n_workers=args.workers,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            train_steps=args.train_steps,
            lr=args.lr,
            val_tokens=args.val_tokens,
            gpu_memory_fraction=args.gpu_memory_fraction,
            high_water_mib=args.vram_cap_mib,
            low_water_mib=args.vram_resume_mib,
            governor_poll_s=args.governor_poll_s,
            commit_every=args.commit_every,
            blimp_per_subtask=args.blimp_per_subtask,
            hellaswag_limit=args.hellaswag_limit,
        )
        assert_sqlite_health(DB_PATH, label="post-bpe-backfill")


if __name__ == "__main__":
    main()
