"""Multi-probe driver for the long-context + reasoning bundle.

For each input fingerprint:
  1. Reconstruct model from graph_json
  2. Micro-train on random tokens to a working state
  3. Run trajectory probe (Jacobian / ICLD / id_collapse / logit_margin)
  4. Run CKA fingerprint
  5. Run v2 induction (default gaps + long gaps 128/256/512)
  6. Run v2 binding
  7. Run passkey, multi_hop, long_range_ar
  8. Run long_context scaling sweep
  9. Run NEW: selective_copy
 10. Run NEW: compositional

Writes per-fingerprint JSON to ``research/reports/long_ctx_bundle_<ts>/``.
Read-only against the leaderboard (does NOT write to program_results) — this is
a calibration sample, not a productized backfill. Intended for top-N curated
fingerprints; the operator reviews results before deciding whether to broaden.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.eval.utils import micro_train_loop
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

VOCAB_SIZE = 50257
DEFAULT_TRAIN_STEPS = 500
DEFAULT_SEQ_LEN = 128
DEFAULT_BATCH = 4


def _select_entries(db: Path, n: int, entry_ids: list[str] | None) -> list[dict]:
    """Pick entries by explicit IDs (preserving order) OR top-N by composite."""
    conn = connect_readonly(db)
    try:
        if entry_ids:
            placeholders = ",".join("?" * len(entry_ids))
            rows = conn.execute(
                f"""
                SELECT l.entry_id, l.composite_score, l.tier, l.result_id,
                       pr.graph_fingerprint, pr.graph_json,
                       pr.induction_v2_investigation_auc AS iv2,
                       pr.binding_v2_investigation_auc AS bv2,
                       pgf.template_name
                FROM leaderboard l
                JOIN program_results pr ON pr.result_id = l.result_id
                LEFT JOIN program_graph_features pgf ON pgf.result_id = l.result_id
                WHERE l.entry_id IN ({placeholders})
                  AND pr.graph_json IS NOT NULL AND pr.graph_json != '{{}}'
                """,
                entry_ids,
            ).fetchall()
            by_id = {r["entry_id"]: dict(r) for r in rows}
            return [by_id[e] for e in entry_ids if e in by_id]

        rows = conn.execute(
            """
            SELECT l.entry_id, l.composite_score, l.tier, l.result_id,
                   pr.graph_fingerprint, pr.graph_json,
                   pr.induction_v2_investigation_auc AS iv2,
                   pr.binding_v2_investigation_auc AS bv2,
                   pgf.template_name
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id = l.result_id
            WHERE l.composite_score IS NOT NULL
              AND pr.graph_json IS NOT NULL AND pr.graph_json != '{}'
            ORDER BY l.composite_score DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _train_model(
    graph_json_str: str,
    *,
    device: str,
    train_steps: int,
    d_model: int | None,
    n_layers: int,
    seq_len: int,
    batch: int,
) -> tuple:
    """Compile + train. Champion mode passes d_model=512, n_layers=12,
    seq_len=512, batch=8, train_steps=10000."""
    graph = graph_from_json(graph_json_str, model_dim=d_model)
    layer_graphs = [graph] * n_layers
    t0 = time.perf_counter()
    model = compile_model(layer_graphs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    compile_s = time.perf_counter() - t0

    n_batches = max(8, min(64, train_steps // 50))
    batches = [
        torch.randint(0, VOCAB_SIZE, (batch, seq_len), device=device)
        for _ in range(n_batches)
    ]
    t0 = time.perf_counter()
    final_loss = micro_train_loop(
        model, batches, vocab_size=VOCAB_SIZE, n_steps=train_steps, lr=3e-4
    )
    train_s = time.perf_counter() - t0
    return model, n_params, float(final_loss or 0.0), compile_s, train_s


def _safe_run(name: str, fn) -> dict:
    """Run a probe and return a dict with status, score, elapsed_s, error."""
    t0 = time.perf_counter()
    try:
        result = fn()
        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif isinstance(result, dict):
            payload = result
        else:
            payload = {"raw": str(result)}
        payload["_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        payload = {"_status": "error", "_error": f"{type(exc).__name__}: {exc}"}
        logger.warning("%s failed: %s", name, exc, exc_info=False)
    payload["_elapsed_s"] = round(time.perf_counter() - t0, 2)
    return payload


def _run_all_probes(model: torch.nn.Module, *, device: str) -> dict:
    """Probe execution order matters. The trajectory probe leaves the model
    with weight_norm parametrizations that break ``copy.deepcopy`` for every
    retrieval-style probe afterwards (silent ``status='copy_failed'`` returning
    score=0). We run trajectory LAST so its mutations don't poison the rest.

    Compositional runs second-to-last because it trains in place (no deepcopy
    needed) so it survives the post-trajectory state too — but its training
    would corrupt the model for any deepcopy-using probe that follows.
    """
    out: dict = {}

    # 1. v2 induction (default gaps 4-64) — pre-trajectory so deepcopy works
    from research.eval.induction_probe_v2_investigation import (
        run_induction_v2_investigation,
    )

    out["induction_v2_default"] = _safe_run(
        "induction_v2_default",
        lambda: run_induction_v2_investigation(model, device=device),
    )

    # 2. v2 binding
    from research.eval.binding_probe_v2_investigation import (
        run_binding_v2_investigation,
    )

    out["binding_v2"] = _safe_run(
        "binding_v2", lambda: run_binding_v2_investigation(model, device=device)
    )

    # 3. passkey
    from research.eval.passkey_retrieval import passkey_retrieval_score

    out["passkey"] = _safe_run(
        "passkey",
        lambda: passkey_retrieval_score(
            model, seq_lens=(256, 512, 1024), n_train_steps=100, device=device
        ),
    )

    # 4. multi-hop
    from research.eval.multi_hop_retrieval import multi_hop_retrieval_score

    out["multi_hop"] = _safe_run(
        "multi_hop",
        lambda: multi_hop_retrieval_score(
            model,
            seq_lens=(256, 512),
            hop_depths=(2, 3),
            n_train_steps=100,
            device=device,
        ),
    )

    # 5. selective copy
    from research.eval.selective_copy_probe import selective_copy_score

    out["selective_copy"] = _safe_run(
        "selective_copy", lambda: selective_copy_score(model, device=device)
    )

    # 6. compositional — trains in place (state_dict snapshot/restore)
    from research.eval.compositional_probe import compositional_score

    out["compositional"] = _safe_run(
        "compositional", lambda: compositional_score(model, device=device)
    )

    # 7. trajectory probe LAST — adds weight_norm parametrizations that break
    # subsequent deepcopy. Running it here means its model-state side effects
    # don't poison earlier probes.
    from research.eval.trajectory_metrics import compute_trajectory_metrics

    out["trajectory"] = _safe_run(
        "trajectory",
        lambda: compute_trajectory_metrics(
            model, metric_phase="bundle_eval", device=device
        ),
    )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        default="research/lab_notebook.db",
        type=Path,
        help="lab notebook path",
    )
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument(
        "--entry-ids",
        default="",
        help="comma-separated leaderboard entry_ids; overrides --top-n",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    ap.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="override model_dim (champion mode uses 512)",
    )
    ap.add_argument(
        "--n-layers", type=int, default=1, help="layer count (champion mode uses 12)"
    )
    ap.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="train batch seq_len"
    )
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="train batch size")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir (default: research/reports/long_ctx_bundle_<ts>)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_dir = args.out_dir or Path(
        f"research/reports/long_ctx_bundle_{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    entry_ids = [e.strip() for e in args.entry_ids.split(",") if e.strip()] or None
    fingerprints = _select_entries(args.db, args.top_n, entry_ids)
    logger.info(
        "selected %d fingerprints (%s) -> %s",
        len(fingerprints),
        "explicit entry-ids" if entry_ids else "top by composite_score",
        out_dir,
    )

    t_start = time.perf_counter()
    with summary_path.open("w") as summary_f:
        for idx, fp in enumerate(fingerprints, 1):
            ent = fp["entry_id"]
            tag = f"[{idx}/{len(fingerprints)} {ent}]"
            logger.info(
                "%s comp=%.1f tpl=%s — compile (d_model=%s n_layers=%d) + train (%d steps)",
                tag,
                fp["composite_score"],
                fp.get("template_name"),
                args.d_model or "default",
                args.n_layers,
                args.train_steps,
            )

            try:
                model, n_params, final_loss, compile_s, train_s = _train_model(
                    fp["graph_json"],
                    device=args.device,
                    train_steps=args.train_steps,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    seq_len=args.seq_len,
                    batch=args.batch,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s compile/train failed: %s", tag, exc)
                summary_f.write(
                    json.dumps(
                        {
                            "entry_id": ent,
                            "result_id": fp["result_id"],
                            "fingerprint": fp["graph_fingerprint"],
                            "template": fp.get("template_name"),
                            "composite_score": fp["composite_score"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    + "\n"
                )
                continue

            logger.info(
                "%s params=%d final_loss=%.3f compile=%.1fs train=%.1fs",
                tag,
                n_params,
                final_loss,
                compile_s,
                train_s,
            )

            # Sanity-check the trained model. NaN / inf loss or extreme loss
            # means the model is corrupted (gradient explosion, dead init,
            # weight_norm failure). Probes against a corrupted model return
            # silent zeros and waste GPU time. Skip them.
            import math

            model_corrupted = (
                not math.isfinite(final_loss) or final_loss <= 0 or final_loss > 50
            )
            if not model_corrupted:
                # Quick forward sanity: any NaN parameters?
                for p in model.parameters():
                    if not torch.isfinite(p).all():
                        model_corrupted = True
                        break

            if model_corrupted:
                logger.warning(
                    "%s MODEL CORRUPTED (final_loss=%s) — skipping probes",
                    tag,
                    final_loss,
                )
                probes = {"_corrupted": True, "_final_loss": final_loss}
                probes_s = 0.0
            else:
                t_probes = time.perf_counter()
                probes = _run_all_probes(model, device=args.device)
                probes_s = time.perf_counter() - t_probes

            row = {
                "entry_id": ent,
                "result_id": fp["result_id"],
                "fingerprint": fp["graph_fingerprint"],
                "template": fp.get("template_name"),
                "composite_score": fp["composite_score"],
                "tier": fp["tier"],
                "stored_iv2": fp.get("iv2"),
                "stored_bv2": fp.get("bv2"),
                "n_params": n_params,
                "final_loss": final_loss,
                "compile_s": round(compile_s, 2),
                "train_s": round(train_s, 2),
                "probes_s": round(probes_s, 2),
                "probes": probes,
            }

            (out_dir / f"{ent}.json").write_text(json.dumps(row, indent=2))
            summary_f.write(json.dumps(row) + "\n")
            summary_f.flush()

            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("%s done — probes_s=%.1f", tag, probes_s)

    total_s = time.perf_counter() - t_start
    logger.info(
        "bundle complete in %.1fs (%.1fmin) -> %s", total_s, total_s / 60, out_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
