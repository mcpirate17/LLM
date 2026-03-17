"""Re-run top validated architectures + references with tiktoken cl100k_base.

Usage:
    python -m research.tools.tiktoken_rerun --references-only [--device cuda]
    python -m research.tools.tiktoken_rerun --result-ids ID1,ID2 [--device cuda]
    python -m research.tools.tiktoken_rerun --all-validated [--device cuda]
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from research.scientist.notebook import LabNotebook
from research.synthesis.compiler import compile_model
from research.synthesis.reference_architectures import (
    REFERENCE_ARCHITECTURES,
    build_reference,
)
from research.synthesis.serializer import graph_from_json, graph_to_json
from research.training.data_pipeline import (
    CorpusConfig,
    CorpusTokenBatcher,
    TiktokenAdapter,
)

logger = logging.getLogger(__name__)

ENCODING = "cl100k_base"
N_LAYERS = 4
D_MODEL = 256
TRAIN_STEPS = 500
LR = 3e-4
BATCH_SIZE = 4
SEQ_LEN = 256
GRAD_CLIP = 1.0
CORPUS_PATH = str(Path(__file__).resolve().parent.parent / "micro_corpus.txt")


@dataclass
class RerunResult:
    name: str
    original_result_id: str
    new_result_id: str
    tiktoken_loss: float
    byte_loss: Optional[float]
    delta: Optional[float]
    steps_completed: int
    error: Optional[str] = None


def _get_native_vocab_size() -> int:
    adapter = TiktokenAdapter(ENCODING)
    return adapter.native_vocab_size


def _build_batcher(vocab_size: int) -> CorpusTokenBatcher:
    config = CorpusConfig(
        path=CORPUS_PATH,
        tokenizer="tiktoken",
        tiktoken_encoding=ENCODING,
    )
    return CorpusTokenBatcher(config, vocab_size=vocab_size)


def _train_model(
    model: torch.nn.Module,
    batcher: CorpusTokenBatcher,
    device: torch.device,
    steps: int = TRAIN_STEPS,
    lr: float = LR,
    log_milestones: Optional[List[int]] = None,
    log_every: int = 100,
    cosine_decay_start: Optional[int] = None,
) -> float:
    """Train model and return final loss.

    If log_milestones is set, returns dict {step: loss} at each milestone.
    log_every: record loss curve at this interval (default 100).
    cosine_decay_start: if set, apply cosine LR decay from this step to ``steps``.
    """

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = None
    if cosine_decay_start is not None:
        decay_steps = max(1, steps - cosine_decay_start)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=lr * 0.01,
        )

    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    milestone_set = set(log_milestones) if log_milestones else set()
    milestones: Dict[int, float] = {}
    curve: Dict[int, float] = {}
    loss_val = float("inf")

    for step in range(steps):
        batch = batcher.sample_batch(BATCH_SIZE, SEQ_LEN + 1, gen, device)
        if batch is None:
            logger.warning("Batcher returned None at step %d", step)
            break

        x = batch[:, :SEQ_LEN]
        y = batch[:, 1 : SEQ_LEN + 1]

        logits = model(x)
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("output"))
        if logits is None:
            raise RuntimeError("Model returned no logits")

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if scheduler is not None and step >= cosine_decay_start:
            scheduler.step()

        loss_val = loss.item()

        if step in milestone_set:
            milestones[step] = loss_val
        if step % log_every == 0:
            curve[step] = loss_val
            logger.info("  step %d/%d  loss=%.4f", step, steps, loss_val)

    curve[steps - 1] = loss_val
    milestones[steps] = loss_val

    if log_milestones:
        return milestones  # type: ignore[return-value]
    return curve if log_every < 100 else loss_val  # type: ignore[return-value]


def _cleanup_model(model: Optional[torch.nn.Module]):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_single(
    name: str,
    graph_json: str,
    original_result_id: str,
    byte_loss: Optional[float],
    nb: LabNotebook,
    device: torch.device,
    vocab_size: int,
    batcher: CorpusTokenBatcher,
    steps: int = TRAIN_STEPS,
    lr: float = LR,
) -> RerunResult:
    """Run a single architecture with tiktoken and record results."""
    logger.info("=== Running: %s (original=%s) ===", name, original_result_id)

    model = None
    try:
        graph = graph_from_json(graph_json)
        layer_graphs = [graph] * N_LAYERS
        model = compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=SEQ_LEN)

        final_loss = _train_model(model, batcher, device, steps=steps, lr=lr)

        # Record result — must create experiment first (FK constraint)
        new_result_id = str(uuid.uuid4())[:12]
        experiment_id = nb.start_experiment(
            experiment_type="tiktoken_rerun",
            config={"encoding": ENCODING, "original_result_id": original_result_id},
            hypothesis=f"Tiktoken rerun of {name}",
        )
        fp = graph.fingerprint()

        nb.record_program_result(
            experiment_id=experiment_id,
            graph_fingerprint=fp,
            graph_json=graph_json,
            result_id=new_result_id,
            final_loss=final_loss,
            n_train_steps=steps,
            stage0_passed=True,
            stage1_passed=True,
            model_source="tiktoken_rerun",
        )
        nb.flush_writes()
        nb.upsert_leaderboard(
            result_id=new_result_id,
            model_source="tiktoken_rerun",
            architecture_desc=f"tiktoken rerun: {name}",
            tier="screening",
        )

        delta = final_loss - byte_loss if byte_loss is not None else None
        logger.info(
            "  Done: tiktoken_loss=%.4f byte_loss=%s delta=%s",
            final_loss,
            f"{byte_loss:.4f}" if byte_loss is not None else "N/A",
            f"{delta:+.4f}" if delta is not None else "N/A",
        )

        return RerunResult(
            name=name,
            original_result_id=original_result_id,
            new_result_id=new_result_id,
            tiktoken_loss=final_loss,
            byte_loss=byte_loss,
            delta=delta,
            steps_completed=steps,
        )

    except Exception as e:
        logger.error("  FAILED: %s — %s", name, e)
        return RerunResult(
            name=name,
            original_result_id=original_result_id,
            new_result_id="",
            tiktoken_loss=float("inf"),
            byte_loss=byte_loss,
            delta=None,
            steps_completed=0,
            error=str(e),
        )
    finally:
        _cleanup_model(model)


def _load_reference_targets(vocab_size: int) -> List[Dict]:
    """Build targets from reference architectures."""
    targets = []
    for key in REFERENCE_ARCHITECTURES:
        graph = build_reference(key, d_model=D_MODEL)
        targets.append(
            {
                "name": f"ref:{key}",
                "graph_json": graph_to_json(graph),
                "original_result_id": f"ref_{key}",
                "byte_loss": None,
            }
        )
    return targets


def _load_db_targets(nb: LabNotebook, result_ids: List[str]) -> List[Dict]:
    """Load targets from the database by result_id."""
    targets = []
    for rid in result_ids:
        row = nb.conn.execute(
            "SELECT graph_json, final_loss FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()
        if row is None:
            logger.warning("Result ID %s not found in database, skipping", rid)
            continue
        targets.append(
            {
                "name": rid,
                "graph_json": row[0],
                "original_result_id": rid,
                "byte_loss": float(row[1]) if row[1] is not None else None,
            }
        )
    return targets


def _load_all_validated(nb: LabNotebook) -> List[Dict]:
    """Load all validated-tier architectures from the leaderboard."""
    rows = nb.conn.execute(
        "SELECT l.result_id, p.graph_json, p.final_loss "
        "FROM leaderboard l "
        "JOIN program_results p ON l.result_id = p.result_id "
        "WHERE l.tier = 'validation' AND p.graph_json IS NOT NULL "
        "ORDER BY l.composite_score DESC"
    ).fetchall()
    targets = []
    for row in rows:
        targets.append(
            {
                "name": row[0],
                "graph_json": row[1],
                "original_result_id": row[0],
                "byte_loss": float(row[2]) if row[2] is not None else None,
            }
        )
    return targets


def _get_gpt2_ref_loss(results: List[RerunResult]) -> Optional[float]:
    """Extract GPT-2 reference loss from results for comparison."""
    for r in results:
        if "gpt2" in r.name.lower() and r.tiktoken_loss < float("inf"):
            return r.tiktoken_loss
    return None


def _write_report(
    results: List[RerunResult], output_path: Path, steps: int = TRAIN_STEPS
):
    """Write comparison table to markdown."""
    gpt2_loss = _get_gpt2_ref_loss(results)

    lines = [
        "# Tiktoken Re-run Results",
        "",
        f"Encoding: `{ENCODING}` | Steps: {steps} | "
        f"Batch: {BATCH_SIZE} | Seq: {SEQ_LEN} | LR: {LR}",
        "",
        "| Architecture | Byte-era Loss | Tiktoken Loss | Delta | Still beats GPT-2? |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        byte_str = f"{r.byte_loss:.4f}" if r.byte_loss is not None else "N/A"
        tik_str = (
            f"{r.tiktoken_loss:.4f}" if r.tiktoken_loss < float("inf") else "FAILED"
        )
        delta_str = f"{r.delta:+.4f}" if r.delta is not None else "N/A"

        if r.error:
            beats = f"ERROR: {r.error[:40]}"
        elif gpt2_loss is not None and r.tiktoken_loss < float("inf"):
            beats = "Yes" if r.tiktoken_loss < gpt2_loss else "No"
        else:
            beats = "N/A"

        lines.append(f"| {r.name} | {byte_str} | {tik_str} | {delta_str} | {beats} |")

    lines.append("")
    lines.append(
        f"GPT-2 tiktoken baseline: {gpt2_loss:.4f}"
        if gpt2_loss
        else "GPT-2 tiktoken baseline: N/A"
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", output_path)


def lr_sweep(
    result_id: str,
    lrs: List[float],
    db_path: str = "research/lab_notebook.db",
    device_str: str = "cuda",
    steps: int = 4000,
    corpus_path: Optional[str] = None,
) -> List[Dict]:
    """Run LR sweep for a single architecture. Returns list of {lr, milestones} dicts."""

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    vocab_size = _get_native_vocab_size()
    batcher = (
        _build_batcher(vocab_size)
        if corpus_path is None
        else _build_npy_batcher(corpus_path, vocab_size)
    )
    if not batcher.ready:
        raise RuntimeError("Corpus not available")

    nb = LabNotebook(db_path)
    row = nb.conn.execute(
        "SELECT graph_json, final_loss FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Result ID {result_id} not found")
    graph_json, byte_loss = row[0], row[1]
    nb.close()

    milestones_at = [499, 1999, 3999]  # 0-indexed → report as 500/2000/4000
    results = []

    for lr_val in lrs:
        logger.info("=== LR sweep: lr=%.4f steps=%d ===", lr_val, steps)
        graph = graph_from_json(graph_json)
        model = compile_model(
            [graph] * N_LAYERS, vocab_size=vocab_size, max_seq_len=SEQ_LEN
        )
        try:
            ms = _train_model(
                model,
                batcher,
                device,
                steps=steps,
                lr=lr_val,
                log_milestones=milestones_at,
            )
            results.append({"lr": lr_val, "milestones": ms})
            logger.info(
                "  lr=%.4f  final_loss=%.4f",
                lr_val,
                ms.get(steps, ms.get(steps - 1, float("inf"))),
            )
        except Exception as e:
            logger.error("  lr=%.4f FAILED: %s", lr_val, e)
            results.append({"lr": lr_val, "milestones": {}, "error": str(e)})
        finally:
            _cleanup_model(model)

    return results


def _build_npy_batcher(npy_path: str, vocab_size: int) -> CorpusTokenBatcher:
    """Build batcher from pretokenized .npy file."""
    config = CorpusConfig(
        path=npy_path,
        tokenizer="tiktoken",
        tiktoken_encoding=ENCODING,
    )
    return CorpusTokenBatcher(config, vocab_size=vocab_size)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run architectures with tiktoken cl100k_base"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--references-only",
        action="store_true",
        help="Run only the 4 reference architectures",
    )
    group.add_argument(
        "--result-ids", type=str, help="Comma-separated result IDs to re-run"
    )
    group.add_argument(
        "--all-validated",
        action="store_true",
        help="Re-run all validated-tier architectures",
    )
    group.add_argument(
        "--lr-sweep", type=str, help="LR sweep: --lr-sweep RESULT_ID --lrs 0.001,0.0003"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--notebook", type=str, default="research/lab_notebook.db")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument(
        "--lrs", type=str, default=None, help="Comma-separated LRs for sweep"
    )
    parser.add_argument(
        "--corpus", type=str, default=None, help="Path to pretokenized .npy corpus"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # LR sweep mode
    if args.lr_sweep:
        lrs_str = args.lrs or "0.0001,0.0003,0.001,0.003"
        lrs = [float(x.strip()) for x in lrs_str.split(",")]
        results = lr_sweep(
            result_id=args.lr_sweep,
            lrs=lrs,
            db_path=args.notebook,
            device_str=args.device,
            steps=args.steps,
            corpus_path=args.corpus,
        )
        print(f"\n{'=' * 70}")
        print(f"LR SWEEP — {args.lr_sweep} — {args.steps} steps")
        print(f"{'=' * 70}")
        print(f"{'LR':<10} {'Loss@500':>10} {'Loss@2K':>10} {'Loss@4K':>10}")
        print("-" * 50)
        for r in sorted(results, key=lambda x: x["lr"]):
            ms = r["milestones"]
            l500 = f"{ms.get(499, 0):.4f}" if 499 in ms else "N/A"
            l2k = f"{ms.get(1999, 0):.4f}" if 1999 in ms else "N/A"
            l4k = f"{ms.get(args.steps, ms.get(3999, 0)):.4f}" if ms else "FAILED"
            print(f"{r['lr']:<10.4f} {l500:>10} {l2k:>10} {l4k:>10}")
        return

    device = torch.device(args.device)
    vocab_size = _get_native_vocab_size()
    lr = args.lr
    logger.info(
        "Using encoding=%s, native_vocab_size=%d, device=%s, lr=%s",
        ENCODING,
        vocab_size,
        device,
        lr,
    )

    nb = LabNotebook(args.notebook)
    if args.corpus:
        batcher = _build_npy_batcher(args.corpus, vocab_size)
    else:
        batcher = _build_batcher(vocab_size)
    if not batcher.ready:
        logger.error("Corpus not available")
        sys.exit(1)

    # Build target list
    targets: List[Dict] = []
    if args.references_only:
        targets = _load_reference_targets(vocab_size)
    elif args.result_ids:
        ids = [r.strip() for r in args.result_ids.split(",") if r.strip()]
        targets = _load_db_targets(nb, ids)
    elif args.all_validated:
        targets = _load_all_validated(nb)
        targets = _load_reference_targets(vocab_size) + targets

    if not targets:
        logger.error("No targets found")
        sys.exit(1)

    steps = args.steps
    logger.info("Will re-run %d architectures", len(targets))

    results: List[RerunResult] = []
    for t in targets:
        result = _run_single(
            name=t["name"],
            graph_json=t["graph_json"],
            original_result_id=t["original_result_id"],
            byte_loss=t["byte_loss"],
            nb=nb,
            device=device,
            vocab_size=vocab_size,
            batcher=batcher,
            steps=steps,
            lr=lr,
        )
        results.append(result)

    print(f"\n{'=' * 80}")
    print("TIKTOKEN RE-RUN RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Architecture':<30} {'Byte Loss':>10} {'Tiktoken':>10} {'Delta':>10}")
    print("-" * 70)
    for r in results:
        byte_str = f"{r.byte_loss:.4f}" if r.byte_loss is not None else "N/A"
        tik_str = (
            f"{r.tiktoken_loss:.4f}" if r.tiktoken_loss < float("inf") else "FAILED"
        )
        delta_str = f"{r.delta:+.4f}" if r.delta is not None else "N/A"
        print(f"{r.name:<30} {byte_str:>10} {tik_str:>10} {delta_str:>10}")

    report_path = (
        Path(__file__).resolve().parent.parent.parent
        / "tasks"
        / "audit"
        / "TIKTOKEN_RERUN_RESULTS.md"
    )
    _write_report(results, report_path, steps=steps)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
