"""Tuning harness for nano_blimp + synthetic_association probes.

Runs both probes on a hand-picked set of architectures across several
config grids (vocab size × train steps) and prints a discrimination
matrix. Goal: find a config where top architectures cleanly separate
from bottom architectures (no saturation, no noise floor).

Usage:
    python -m research.tools.nano_blimp_tune
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.eval.utils import micro_train_loop
from research.eval.controlled_lang_probe import controlled_lang_probe
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

# Hand-picked test set:
#   top inducers (should ace any reasonable nano probe at champion mode)
#   reference baselines (Mamba/RWKV at base train — likely struggle without training)
#   mid-tier (8e601381 silent-failed earlier — should be a clear "below" signal)
DEFAULT_ARCHS = [
    ("b0c38826-905", "latent_compress_block (top inducer)"),
    ("72769b4f-515", "local_attn_ssm_hybrid (top inducer)"),
    ("127a3f46-ad3", "Mamba reference"),
    ("5b6d6017-0d9", "RWKV reference"),
    ("8e601381-c8d", "adaptive_conv_ffn (mid)"),
]

# Configs to try. Tune by:
#  - if every architecture saturates at 100% → expand vocab or shrink steps
#  - if every architecture stays at chance → increase n_train_steps
#  - if top vs bottom separates → that config is calibrated
#
# Round 1 (vocab=32-64, steps=300-1000): saturated everyone.
# Round 2 (vocab=128-256, steps=100-300): still saturated, surprisingly.
# Round 3 (vocab=80, steps=10-40): match codex's calibrated synthetic_assoc
# regime where GPT2/Mamba/RWKV land at 0.78/0.25/0.80 — real spread.
# nano_blimp's minimal-pair task is easier to saturate than 4-way forced
# choice (chance 0.5 vs 0.25), so we go even shorter on training.
# Final calibration after rounds 1-4 + codex's matrix:
#   - S0.5 floor (every capable model should pass): vocab=80, steps=40
#   - S1 candidate (real discrimination): vocab=120, steps=40 or vocab=80, steps=80
#   - Investigation tier: 160/500 steps saturates capable models — costly,
#     not used for now.
DEFAULT_CONFIGS = [
    {"active_vocab_size": 250, "n_train_steps": 40},  # gap fill v200<>v300
    {"active_vocab_size": 300, "n_train_steps": 80},  # investigation candidate
]


def _load_arch(db: Path, entry_id: str) -> dict:
    conn = connect_readonly(db)
    try:
        r = conn.execute(
            """
            SELECT l.entry_id, l.composite_score, l.tier, pr.result_id,
                   pr.graph_json, pr.graph_fingerprint, pgf.template_name
            FROM leaderboard l JOIN program_results pr ON pr.result_id=l.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id=l.result_id
            WHERE l.entry_id=?
            """,
            (entry_id,),
        ).fetchone()
        return dict(r) if r else {}
    finally:
        conn.close()


def _train_base(
    graph_json_str: str, *, base_steps: int, device: str
) -> torch.nn.Module:
    """Quick base train so the model has SOME language signal before the
    probe's own training kicks in. Match production screening (~750 steps)."""
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph]).to(device)
    batches = [torch.randint(0, 50257, (4, 128), device=device) for _ in range(8)]
    micro_train_loop(model, batches, vocab_size=50257, n_steps=base_steps, lr=3e-4)
    return model


def _run_one(model, *, vocab, n_train, device, seed=42) -> dict:
    """One training pass on the controlled-language corpus, both evals.

    Uses the unified ``controlled_lang_probe`` so we don't duplicate the
    training pass between the synthetic_association and nano_blimp evals.
    """
    res = controlled_lang_probe(
        model,
        active_vocab_size=vocab,
        n_train_steps=n_train,
        batch_size=32,
        lr=1e-3,
        device=device,
        seed=seed,
    )
    return {
        "nano_blimp": res.nano_blimp,
        "synthetic_assoc": res.synthetic_association,
        "_status": res.status,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db", default="research/lab_notebook.db", type=Path, help="lab notebook"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--base-train-steps",
        type=int,
        default=750,
        help="base wikitext-style training before each probe (matches screening)",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="if >0, override DEFAULT_ARCHS with top-N leaderboard entries by composite",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(f"research/reports/nano_blimp_tune_{int(time.time())}.json"),
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix: list[dict] = []

    archs = DEFAULT_ARCHS
    if args.top_n > 0:
        conn = connect_readonly(args.db)
        try:
            rows = conn.execute(
                """
                SELECT l.entry_id, pgf.template_name, l.composite_score
                FROM leaderboard l
                JOIN program_results pr ON pr.result_id=l.result_id
                LEFT JOIN program_graph_features pgf ON pgf.result_id=l.result_id
                WHERE l.composite_score IS NOT NULL
                  AND pr.graph_json IS NOT NULL AND pr.graph_json != '{}'
                ORDER BY l.composite_score DESC LIMIT ?
                """,
                (args.top_n,),
            ).fetchall()
        finally:
            conn.close()
        archs = [
            (
                r["entry_id"],
                f"{r['template_name'] or '?'} (#{i + 1}, comp={r['composite_score']:.0f})",
            )
            for i, r in enumerate(rows)
        ]
        logger.info("loaded %d top-N entries", len(archs))

    for entry_id, label in archs:
        logger.info("=== %s (%s) ===", entry_id, label)
        rec = _load_arch(args.db, entry_id)
        if not rec or not rec.get("graph_json"):
            logger.warning("  no graph_json for %s; skipping", entry_id)
            continue

        model = None
        try:
            t0 = time.perf_counter()
            model = _train_base(
                rec["graph_json"], base_steps=args.base_train_steps, device=args.device
            )
            logger.info("  base trained in %.1fs", time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            logger.error("  base train failed: %s", exc)
            continue

        for cfg in DEFAULT_CONFIGS:
            t0 = time.perf_counter()
            try:
                res = _run_one(
                    model,
                    vocab=cfg["active_vocab_size"],
                    n_train=cfg["n_train_steps"],
                    device=args.device,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("  config %s failed: %s", cfg, exc)
                res = {"_error": f"{type(exc).__name__}: {exc}"}
            elapsed = time.perf_counter() - t0
            matrix.append(
                {
                    "entry_id": entry_id,
                    "label": label,
                    "config": cfg,
                    "elapsed_s": round(elapsed, 1),
                    **res,
                }
            )
            nb_score = (res.get("nano_blimp") or {}).get("nano_blimp_score", 0)
            sa_score = (res.get("synthetic_assoc") or {}).get(
                "synthetic_association_score", 0
            )
            logger.info(
                "  vocab=%d steps=%d -> nano_blimp=%.3f synthetic_assoc=%.3f (%.1fs)",
                cfg["active_vocab_size"],
                cfg["n_train_steps"],
                nb_score,
                sa_score,
                elapsed,
            )

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(matrix, indent=2))
    logger.info("wrote %d rows -> %s", len(matrix), args.out)

    # Print discrimination matrix — auto-pick column headers from configs
    print("\n=== nano_blimp discrimination matrix ===")
    cfg_keys = [
        f"v{cfg['active_vocab_size']}/{cfg['n_train_steps']}" for cfg in DEFAULT_CONFIGS
    ]
    print(f"{'arch':34s} " + " ".join(f"{k:>11s}" for k in cfg_keys))
    by_arch: dict = {}
    for row in matrix:
        ent = row["entry_id"]
        cfg_key = (
            f"v{row['config']['active_vocab_size']}/{row['config']['n_train_steps']}"
        )
        nb = (row.get("nano_blimp") or {}).get("nano_blimp_score", 0)
        sa = (row.get("synthetic_assoc") or {}).get("synthetic_association_score", 0)
        by_arch.setdefault(ent, {"label": row["label"]})[cfg_key] = (nb, sa)
    for ent, scores in by_arch.items():
        label = scores["label"][:34]
        cells = [
            f"{scores.get(k, (0, 0))[0]:.2f}/{scores.get(k, (0, 0))[1]:.2f}"
            for k in cfg_keys
        ]
        print(f"{label:34s} " + " ".join(f"{c:>11s}" for c in cells))
    print("(format: nano_blimp/synthetic_assoc; chance: 0.50/0.25)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
