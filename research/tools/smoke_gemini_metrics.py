#!/usr/bin/env python
"""Smoke test for Gemini trajectory metrics.

Rebuilds a fixed list of known-good architectures from ``graph_json``,
runs each through a tiered training schedule, and measures all four
Gemini metrics + spec_norm at each tier boundary. Output is a single
JSON artifact plus a stdout comparison table.

The smoke test does NOT touch the live database (writes go to JSON
only). It DOES use the GPU; pause the continuous run before starting
or expect both to slow down.

Usage::

    python -m research.tools.smoke_gemini_metrics

Expects ``research/corpus/wikitext103_train.npy`` to be present.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
import torch.nn as nn
import torch.nn.functional as F

# Force compiler import so OP_DISPATCH is populated for CompiledOps.
import research.synthesis.compiler  # noqa: F401
from research.synthesis import graph_from_json
from research.synthesis.compiled_model import SynthesizedModel
from research.eval.trajectory_metrics import (
    capture_hidden_state_snapshot,
    compute_trajectory_metrics,
)
from research.tools._concurrency import (
    acquire_gpu_lock,
    assert_gpu_quiet,
    cap_gpu_memory,
)


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
CORPUS_PATH = ROOT / "research" / "corpus" / "wikitext103_train.npy"
ARTIFACT_DIR = ROOT / "research" / "perf_artifacts"
TOOL_NAME = "smoke_gemini_metrics"


# Step budgets per tier — chosen to match what production screening /
# investigation actually run, capped where validation would be impractical
# in a single smoke session.
SCREENING_STEPS = 750
INVESTIGATION_STEPS = 3000  # production runs longer but 3k is enough to
# see metric evolution past the screening budget
VALIDATION_STEPS = 8000  # cap; full validation is much longer
ID_COLLAPSE_EARLY_STEP = 150
ID_COLLAPSE_LATE_STEP = 750  # snapshot at end of screening

DEFAULT_TARGETS = [
    # (result_id, label, run_investigation, run_validation)
    ("ref_gpt2_c9b6bc42", "GPT-2", True, True),
    ("ref_mamba_76ff10cd", "Mamba", True, True),
    ("ref_rwkv_61754c8e", "RWKV", True, True),
    ("ref_retrieval_augmented_ab5cf5ae", "Retrieval-Augmented", True, True),
    ("ec7025d7-338", "top-ind-1 (breakthrough)", False, False),
    ("7d956ba2-4f2", "top-ind-2 (orphan)", False, False),
    ("13442e9f-aea", "top-ind-3 (validation)", False, False),
]


def _load_corpus_tokens(path: Path, vocab_size: int, max_tokens: int) -> torch.Tensor:
    """Load WikiText tokens, project into the model's vocab via modulo."""
    if not path.exists():
        raise FileNotFoundError(f"corpus npy not found: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.size > max_tokens:
        arr = arr[:max_tokens]
    tokens = torch.as_tensor(np.asarray(arr), dtype=torch.long)
    return tokens % vocab_size


def _sample_batch(
    tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    n = tokens.numel() - seq_len - 1
    if n <= 0:
        raise ValueError("corpus too small for the requested seq_len")
    starts = torch.randint(0, n, (batch_size,))
    chunks = [tokens[s : s + seq_len + 1] for s in starts.tolist()]
    batch = torch.stack(chunks, dim=0).to(device)
    return batch  # (B, S+1)


def _train_step(
    model: nn.Module,
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


def _train_to_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    *,
    target_step: int,
    current_step: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    capture_at: Optional[int] = None,
    probe_ids: Optional[torch.Tensor] = None,
) -> tuple[int, Optional[Any]]:
    """Run optimizer steps until ``current_step`` reaches ``target_step``.

    If ``capture_at`` is set, calls ``capture_hidden_state_snapshot`` once
    when the step counter equals ``capture_at`` and returns the snapshot.
    """
    snapshot = None
    while current_step < target_step:
        batch = _sample_batch(tokens, batch_size, seq_len, device)
        _train_step(model, optimizer, batch)
        current_step += 1
        if (
            capture_at is not None
            and probe_ids is not None
            and current_step == capture_at
        ):
            snapshot = capture_hidden_state_snapshot(
                model, probe_ids, step=current_step, device=str(device)
            )
            model.train()
    return current_step, snapshot


def _measure_phase(
    model: nn.Module,
    *,
    metric_phase: str,
    early_snapshot=None,
    late_snapshot=None,
) -> Dict[str, Any]:
    """Run all trajectory metrics on the current model state."""
    res = compute_trajectory_metrics(
        model,
        metric_phase=metric_phase,
        id_collapse_early=early_snapshot,
        id_collapse_late=late_snapshot,
    )
    return res.to_column_dict()


def _summary_line(label: str, phase: str, m: Dict[str, Any]) -> str:
    sn = m.get("fp_jacobian_spectral_norm") or 0
    erf_d = m.get("fp_jacobian_erf_density") or 0
    erf_v = m.get("fp_jacobian_erf_variance") or 0
    icld = m.get("fp_icld_velocity") or 0
    margin = m.get("fp_logit_margin_velocity") or 0
    pr_e = m.get("fp_id_pr_early")
    pr_l = m.get("fp_id_pr_late")
    cr = m.get("fp_id_collapse_rate")
    pr_str = (
        f"{pr_e:.1f}→{pr_l:.1f} ({cr:+.4f})"
        if pr_e and pr_l and cr is not None
        else "—"
    )
    return (
        f"{label:<26}  {phase:<14}  sn={sn:>8.1f}  erf_d={erf_d:.3f}  "
        f"erf_var={erf_v:>10.0f}  icld={icld:+.4f}  margin={margin:+.4f}  "
        f"id_pr={pr_str}"
    )


def _run_screening_phase(
    model,
    optimizer,
    tokens: torch.Tensor,
    *,
    label: str,
    dev: torch.device,
    seq_len: int,
    batch_size: int,
    probe_ids: torch.Tensor,
) -> tuple[int, Dict[str, Any]]:
    current_step, early_snap = _train_to_step(
        model,
        optimizer,
        tokens,
        target_step=ID_COLLAPSE_EARLY_STEP,
        current_step=0,
        batch_size=batch_size,
        seq_len=seq_len,
        device=dev,
        capture_at=ID_COLLAPSE_EARLY_STEP,
        probe_ids=probe_ids,
    )
    current_step, late_snap = _train_to_step(
        model,
        optimizer,
        tokens,
        target_step=SCREENING_STEPS,
        current_step=current_step,
        batch_size=batch_size,
        seq_len=seq_len,
        device=dev,
        capture_at=ID_COLLAPSE_LATE_STEP,
        probe_ids=probe_ids,
    )
    metrics = _measure_phase(
        model,
        metric_phase="screening_750",
        early_snapshot=early_snap,
        late_snapshot=late_snap,
    )
    print("  " + _summary_line(label, "screening_750", metrics))
    return current_step, metrics


def _run_extended_phase(
    model,
    optimizer,
    tokens: torch.Tensor,
    *,
    label: str,
    phase: str,
    target_step: int,
    current_step: int,
    dev: torch.device,
    seq_len: int,
    batch_size: int,
) -> tuple[int, Dict[str, Any]]:
    current_step, _ = _train_to_step(
        model,
        optimizer,
        tokens,
        target_step=target_step,
        current_step=current_step,
        batch_size=batch_size,
        seq_len=seq_len,
        device=dev,
    )
    metrics = _measure_phase(model, metric_phase=phase)
    print("  " + _summary_line(label, phase, metrics))
    return current_step, metrics


def _run_target(
    *,
    result_id: str,
    label: str,
    run_inv: bool,
    run_val: bool,
    graph,
    tokens: torch.Tensor,
    dev: torch.device,
    device: str,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
) -> Dict[str, Any]:
    print(
        f"[start] {label} ({result_id}) — {graph.n_ops()} ops, "
        f"depth {graph.depth()}, model_dim {graph.model_dim}"
    )

    t_arch = time.time()
    model = SynthesizedModel(
        [graph], vocab_size=vocab_size, model_dim=graph.model_dim
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    probe_ids = _sample_batch(tokens, 8, seq_len, dev)[:, :seq_len]

    init_metrics = _measure_phase(model, metric_phase="init")
    print("  " + _summary_line(label, "init", init_metrics))

    current_step, screening_metrics = _run_screening_phase(
        model,
        optimizer,
        tokens,
        label=label,
        dev=dev,
        seq_len=seq_len,
        batch_size=batch_size,
        probe_ids=probe_ids,
    )

    investigation_metrics = None
    validation_metrics = None
    if run_inv:
        current_step, investigation_metrics = _run_extended_phase(
            model,
            optimizer,
            tokens,
            label=label,
            phase="investigation_3000",
            target_step=INVESTIGATION_STEPS,
            current_step=current_step,
            dev=dev,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    if run_val:
        current_step, validation_metrics = _run_extended_phase(
            model,
            optimizer,
            tokens,
            label=label,
            phase="validation_8000",
            target_step=VALIDATION_STEPS,
            current_step=current_step,
            dev=dev,
            seq_len=seq_len,
            batch_size=batch_size,
        )

    elapsed_arch = time.time() - t_arch
    print(f"  {label} done in {elapsed_arch:.1f}s")
    print()

    row = {
        "result_id": result_id,
        "label": label,
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "model_dim": graph.model_dim,
        "elapsed_s": elapsed_arch,
        "metrics": {
            "init": init_metrics,
            "screening_750": screening_metrics,
            "investigation_3000": investigation_metrics,
            "validation_8000": validation_metrics,
        },
    }

    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def run_smoke(
    targets: List[Tuple[str, str, bool, bool]],
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    max_corpus_tokens: int,
    vocab_size: int,
) -> Dict[str, Any]:
    dev = torch.device(device)
    print(f"[setup] loading corpus: {CORPUS_PATH}")
    tokens = _load_corpus_tokens(CORPUS_PATH, vocab_size, max_corpus_tokens)
    print(f"[setup] {tokens.numel():,} corpus tokens, vocab={vocab_size}")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows: List[Dict[str, Any]] = []
    print()
    for result_id, label, run_inv, run_val in targets:
        record = conn.execute(
            "SELECT graph_json FROM program_results WHERE result_id = ? "
            "AND graph_json IS NOT NULL AND length(graph_json) > 0",
            (result_id,),
        ).fetchone()
        if record is None:
            print(f"[skip] {result_id} not found")
            continue

        graph_json = resolve_graph_json_value(conn, DB_PATH, record["graph_json"])
        graph = graph_from_json(graph_json)
        rows.append(
            _run_target(
                result_id=result_id,
                label=label,
                run_inv=run_inv,
                run_val=run_val,
                graph=graph,
                tokens=tokens,
                dev=dev,
                device=device,
                seq_len=seq_len,
                batch_size=batch_size,
                vocab_size=vocab_size,
            )
        )

    conn.close()
    return {
        "config": {
            "device": device,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "screening_steps": SCREENING_STEPS,
            "investigation_steps": INVESTIGATION_STEPS,
            "validation_steps": VALIDATION_STEPS,
            "id_collapse_early_step": ID_COLLAPSE_EARLY_STEP,
            "id_collapse_late_step": ID_COLLAPSE_LATE_STEP,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size for the rebuilt models (matches production)",
    )
    parser.add_argument(
        "--max-corpus-tokens",
        type=int,
        default=4_000_000,
        help="Cap on corpus tokens loaded into memory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: research/perf_artifacts/gemini_smoke_<ts>.json)",
    )
    parser.add_argument(
        "--max-other-gpu-mib",
        type=int,
        default=4096,
        help=(
            "Refuse to start if any other GPU process holds more than this "
            "many MiB of VRAM. Set to a large value to override (e.g. 30000)."
        ),
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.5,
        help="Cap our process's CUDA memory at this fraction of the card.",
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="Sleep-poll until the GPU is quiet instead of exiting on busy.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        assert_gpu_quiet(
            max_other_used_mib=args.max_other_gpu_mib,
            tool_name=TOOL_NAME,
            sleep_until_quiet=args.wait_for_gpu,
        )
        cap_gpu_memory(fraction=args.gpu_memory_fraction)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else ARTIFACT_DIR / f"gemini_smoke_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )

    with acquire_gpu_lock(tool_name=TOOL_NAME):
        result = run_smoke(
            DEFAULT_TARGETS,
            device=args.device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_corpus_tokens=args.max_corpus_tokens,
            vocab_size=args.vocab_size,
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    print(f"\n[done] artifact written: {out_path}")


if __name__ == "__main__":
    main()
