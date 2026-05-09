"""Re-evaluate suspect rows on the FULL WikiText-103 val to test for cache truncation.

Diagnosis: ``research/eval/wikitext_eval.py::_download_wikitext`` only checks
file existence, not size. A 200KB-truncated cache from Apr 15 has been
returned for every subsequent eval, regardless of the ``max_chars_val``
argument. The ``-full`` cache (1.1MB val) sits unused.

This tool reconstructs each model from its saved ``graph_json``, runs
``evaluate_wikitext_trajectory`` against the FULL cache directory, and
prints the new perplexity alongside the recorded one.

Usage::

    python -m research.tools.reval_loss_monsters
    python -m research.tools.reval_loss_monsters --result-ids 2855449a-ef9,8ead3a1f-be3
    python -m research.tools.reval_loss_monsters --variant wikitext-103-raw-v1-full --steps 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.defaults import RUNS_DB
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

logger = logging.getLogger(__name__)


DEFAULT_CANDIDATES = (
    "2855449a-ef9",  # comp 362.7, ppl 5.75 — top of leaderboard
    "8ead3a1f-be3",  # comp 349.7, ppl 6.03
    "88766cc1-f5c",  # comp 330.0, ppl 5.05
    "8308d222-8df",  # comp 289.5, ppl 3.99 — lowest ppl in the cohort
)


def _fetch_row(
    conn: sqlite3.Connection,
    result_id: str,
) -> Optional[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM program_results WHERE result_id = ?", (result_id,)
    ).fetchone()
    if not row:
        return None
    payload = dict(row)
    payload["graph_json"] = resolve_graph_json_value(
        conn,
        RUNS_DB,
        payload.get("graph_json"),
    )
    return payload


def _reconstruct_model(graph_json: str, vocab_size: int, device: str):
    """Build a SynthesizedModel from its persisted graph_json.

    ``research.synthesis.compiler`` populates ``OP_DISPATCH`` at import time —
    must be imported BEFORE constructing SynthesizedModel because each
    ``CompiledOp`` caches its dispatch_fn during ``__init__``.
    """
    import research.synthesis.compiler  # noqa: F401  — populates OP_DISPATCH
    from research.synthesis import graph_from_json
    from research.synthesis.compiled_model import SynthesizedModel

    graph = graph_from_json(graph_json)
    model_dim = getattr(graph, "model_dim", None) or 256
    model = SynthesizedModel(
        [graph],
        vocab_size=vocab_size,
        model_dim=model_dim,
    ).to(device)
    return model


def _eval_one(
    result_id: str,
    *,
    pr_row: Dict[str, Any],
    variant: str,
    n_train_steps: int,
    seq_len: int,
    n_eval_batches: int,
    eval_batch_size: int,
    max_chars_train: int,
    max_chars_val: int,
    device: str,
) -> Dict[str, Any]:
    """Run the full trajectory eval against the requested variant."""
    from research.eval.wikitext_eval import evaluate_wikitext_trajectory
    import torch

    graph_json = pr_row.get("graph_json")
    vocab_size = 32000
    t0 = time.time()
    try:
        model = _reconstruct_model(graph_json, vocab_size, device)
    except Exception as exc:
        return {
            "result_id": result_id,
            "error": f"reconstruct_failed: {type(exc).__name__}: {exc}",
            "elapsed_s": time.time() - t0,
        }

    # Probe at three checkpoints so we can also see ppl@200/ppl@500/ppl@1000.
    checkpoints = (200, 500, 1000) if n_train_steps >= 1000 else (n_train_steps,)
    try:
        out = evaluate_wikitext_trajectory(
            model,
            vocab_size=vocab_size,
            device=device,
            checkpoints=checkpoints,
            variant=variant,
            seq_len=seq_len,
            n_train_batches=0,  # auto-size to max(checkpoints)
            n_eval_batches=n_eval_batches,
            train_batch_size=8,
            eval_batch_size=eval_batch_size,
            max_chars_train=max_chars_train,
            max_chars_val=max_chars_val,
        )
    except Exception as exc:
        return {
            "result_id": result_id,
            "error": f"trajectory_eval_failed: {type(exc).__name__}: {exc}",
            "elapsed_s": time.time() - t0,
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ckpts = out.get("checkpoints") or {}
    return {
        "result_id": result_id,
        "ppl_at_steps": {
            str(k): (ckpts.get(k) or ckpts.get(str(k)) or {}).get("ppl")
            for k in checkpoints
        },
        "peak_ppl": out.get("peak_ppl"),
        "peak_step": out.get("peak_step"),
        "improvement_ratio": out.get("improvement_ratio"),
        "steps_to_divergence": out.get("steps_to_divergence"),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=RUNS_DB,
        help=f"Path to runs DB (default: {RUNS_DB}).",
    )
    parser.add_argument(
        "--result-ids",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated result_ids to re-evaluate.",
    )
    parser.add_argument(
        "--variant",
        default="wikitext-103-raw-v1-full",
        help=(
            "Wikitext cache variant to use. The eval reads from "
            "~/.cache/aria/wikitext/<variant>/{train,validation}.txt — "
            "passing '-full' targets the 1.1MB val cache instead of "
            "the 200KB truncated one."
        ),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--n-eval-batches",
        type=int,
        default=128,
        help="Eval batches × eval_batch_size × seq_len = total val tokens.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-chars-train", type=int, default=200_000_000)
    parser.add_argument("--max-chars-val", type=int, default=2_000_000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "device=%s variant=%s steps=%d val_tokens≈%d",
        device,
        args.variant,
        args.steps,
        args.n_eval_batches * args.eval_batch_size * args.seq_len,
    )

    # Verify which cache we're hitting
    cache_dir = Path.home() / ".cache" / "aria" / "wikitext" / args.variant
    if cache_dir.exists():
        train = cache_dir / "train.txt"
        val = cache_dir / "validation.txt"
        if train.exists() and val.exists():
            logger.info(
                "cache: train=%.2fMB val=%.2fMB",
                train.stat().st_size / 1e6,
                val.stat().st_size / 1e6,
            )
    else:
        logger.warning("variant cache dir does not exist: %s", cache_dir)

    conn = sqlite3.connect(args.db)
    rids = [r.strip() for r in args.result_ids.split(",") if r.strip()]
    results: List[Dict[str, Any]] = []
    for rid in rids:
        pr = _fetch_row(conn, rid)
        if not pr:
            logger.warning("missing pr row for %s", rid)
            continue
        recorded = pr.get("wikitext_perplexity")
        recorded_ppl_500 = pr.get("wikitext_ppl_500")
        recorded_ppl_200 = pr.get("wikitext_ppl_200")
        logger.info(
            "%s — recorded ppl_200=%s ppl_500=%s ppl_final=%s",
            rid,
            recorded_ppl_200,
            recorded_ppl_500,
            recorded,
        )
        out = _eval_one(
            rid,
            pr_row=pr,
            variant=args.variant,
            n_train_steps=args.steps,
            seq_len=args.seq_len,
            n_eval_batches=args.n_eval_batches,
            eval_batch_size=args.eval_batch_size,
            max_chars_train=args.max_chars_train,
            max_chars_val=args.max_chars_val,
            device=device,
        )
        out["recorded_ppl_at_1000"] = recorded
        out["recorded_ppl_at_500"] = recorded_ppl_500
        out["recorded_ppl_at_200"] = recorded_ppl_200
        results.append(out)
        if "error" in out:
            logger.warning("  FAILED: %s", out["error"])
        else:
            ratios = []
            for step, ppl in (out.get("ppl_at_steps") or {}).items():
                rec = (
                    recorded_ppl_200
                    if step == "200"
                    else recorded_ppl_500
                    if step == "500"
                    else recorded
                )
                ratio = (ppl / rec) if (ppl and rec) else None
                ppl_str = f"{ppl:.2f}" if ppl else "n/a"
                rec_str = f"{rec:.2f}" if rec else "n/a"
                ratio_str = f"{ratio:.2f}x" if ratio else "n/a"
                ratios.append(
                    f"step{step}: new={ppl_str} rec={rec_str} ratio={ratio_str}"
                )
            logger.info("  %s  elapsed=%.1fs", " | ".join(ratios), out["elapsed_s"])

    print()
    print("─" * 100)
    print("RESULTS")
    print("─" * 100)
    print(
        f"{'result_id':<14} {'rec_ppl1000':>11} {'new_ppl200':>10} {'new_ppl500':>10} {'new_ppl1000':>12} {'inflation×':>12}"
    )
    print("─" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['result_id'][:14]:<14}  ERROR: {r['error']}")
            continue
        ppls = r.get("ppl_at_steps") or {}
        rec = r.get("recorded_ppl_at_1000") or 0
        new1000 = ppls.get("1000") or 0
        new500 = ppls.get("500") or 0
        new200 = ppls.get("200") or 0
        infl = (new1000 / rec) if rec and new1000 else 0
        print(
            f"{r['result_id'][:14]:<14} "
            f"{rec or 0:>11.2f} "
            f"{new200 or 0:>10.2f} "
            f"{new500 or 0:>10.2f} "
            f"{new1000 or 0:>12.2f} "
            f"{infl:>11.2f}x"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info("wrote results to %s", args.out)


if __name__ == "__main__":
    main()
