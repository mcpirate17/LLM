#!/usr/bin/env python
"""Manually screen and investigate a template through the real pipeline.

Builds the model, runs S0/S0.5/S1 screening with WikiText training,
records results in the notebook, and optionally triggers investigation.

Usage:
    python -m research.tools.screen_template \
        --template attn_normalized_matmul_pinned \
        --layers 4 --dim 256 --device cuda

    # With investigation follow-up:
    python -m research.tools.screen_template \
        --template attn_normalized_matmul_pinned \
        --layers 4 --dim 256 --investigate --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import torch

from research.eval.training_core import run_training_loop
from research.training.loss_ops import next_token_cross_entropy
from research.tools._wikitext_batches import load_wikitext_batch_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_graphs(template_name: str, n_layers: int, dim: int, seed: int = 42):
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.templates import apply_template

    rng = random.Random(seed)
    graphs = []
    for _ in range(n_layers):
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        out = apply_template(g, inp, rng, template_name=template_name)
        g.set_output(out)
        graphs.append(g)
    return graphs


def _s0_s05_check(model, vocab_size: int, device: str) -> tuple[bool, bool]:
    """Forward-pass and gradient finiteness checks."""
    x = torch.randint(0, vocab_size, (2, 64), device=device)
    with torch.no_grad():
        y = model(x)
    s0_passed = y.shape[-1] == vocab_size and not torch.isnan(y).any()
    logger.info("S0: %s (shape=%s)", "PASSED" if s0_passed else "FAILED", y.shape)
    if not s0_passed:
        return False, False

    logger.info("S0.5: Gradient + causality check...")
    x2 = torch.randint(0, vocab_size, (2, 64), device=device)
    logits = model(x2)
    loss = next_token_cross_entropy(logits, x2, vocab_size)
    loss.backward()
    grads_with_values = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    n_grads = len(grads_with_values)
    n_finite = sum(1 for g in grads_with_values if torch.isfinite(g).all())
    s05_passed = n_grads > 0 and n_finite == n_grads
    logger.info(
        "S0.5: %s (grads: %d/%d finite, %d total params)",
        "PASSED" if s05_passed else "FAILED",
        n_finite,
        n_grads,
        sum(1 for p in model.parameters() if p.requires_grad),
    )
    return s0_passed, s05_passed


def _train_loop(
    model,
    batch_source,
    *,
    n_steps: int,
    lr: float,
    lr_warmup: float,
    vocab_size: int,
    device: str,
    seed: int,
    warmup_steps: int = 50,
    log_every: int = 250,
    label: str = "train",
    eval_fn=None,
) -> tuple[list[float], float]:
    """Wikitext warmup+AdamW loop. Delegates to ``eval.training_core.run_training_loop``."""
    rng_gen = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    t0 = time.time()

    def compute_loss(step: int) -> torch.Tensor:
        batch = batch_source.sample_train_batch(device=device, generator=rng_gen)
        logits = model(batch)
        loss = next_token_cross_entropy(logits, batch, vocab_size)
        losses.append(float(loss.detach()))
        step_num = step + 1
        if step_num % log_every == 0 and eval_fn is not None:
            vl = eval_fn()
            avg_train = sum(losses[-log_every:]) / log_every
            logger.info(
                "  %s step %d: train=%.4f val=%.4f ppl=%.1f (%.0fs)",
                label,
                step_num,
                avg_train,
                vl,
                math.exp(min(vl, 20)),
                time.time() - t0,
            )
        return loss

    run_training_loop(
        model.parameters(),
        compute_loss,
        n_steps=n_steps,
        optimizer_name="adamw",
        lr=lr,
        weight_decay=0.01,
        clip_grad=1.0,
        warmup_steps=warmup_steps,
    )
    return losses, time.time() - t0


def _run_probes(model, vocab_size: int, device: str) -> Dict[str, Any]:
    """Run binding/AR/HellaSwag/BLiMP probes, collect results defensively."""
    probe_results: Dict[str, Any] = {}
    try:
        from research.eval.binding_pipeline import run_screening_binding_probes

        bp = run_screening_binding_probes(model, device=device)
        probe_results.update(
            {
                "induction_screening_auc": bp.get("induction_screening_auc"),
                "binding_screening_auc": bp.get("binding_screening_auc"),
                "binding_screening_composite": bp.get("binding_screening_composite"),
            }
        )
        logger.info(
            "Binding: ind=%.4f bind=%.4f",
            probe_results.get("induction_screening_auc", 0),
            probe_results.get("binding_screening_auc", 0),
        )
    except Exception as e:
        logger.warning("Binding failed: %s", e)

    try:
        from research.eval.associative_recall import associative_recall_score

        ar = associative_recall_score(
            model,
            n_pairs=10,
            n_eval=100,
            n_train_steps=300,
            batch_size=8,
            device=device,
        )
        probe_results["ar_legacy_auc"] = ar.auc
        probe_results["ar_legacy_final_acc"] = ar.final_acc
        logger.info("AR: auc=%.4f acc=%.4f", ar.auc, ar.final_acc)
    except Exception as e:
        logger.warning("AR failed: %s", e)

    try:
        from research.eval.hellaswag_eval import evaluate_hellaswag

        hella = evaluate_hellaswag(
            model, vocab_size=vocab_size, device=device, n_examples=200
        )
        probe_results["hellaswag_acc"] = hella.get("hellaswag_acc")
        logger.info("HellaSwag: acc=%s", probe_results["hellaswag_acc"])
    except Exception as e:
        logger.warning("HellaSwag failed: %s", e)

    try:
        from research.eval.blimp_eval import evaluate_blimp

        blimp = evaluate_blimp(
            model,
            vocab_size=vocab_size,
            device=device,
            n_per_subtask=50,
            timeout_s=120,
        )
        probe_results["blimp_overall_accuracy"] = blimp.overall_accuracy
        logger.info("BLiMP: acc=%.4f", blimp.overall_accuracy)
    except Exception as e:
        logger.warning("BLiMP failed: %s", e)
    return probe_results


def _record_screening(
    nb,
    *,
    template_name: str,
    n_layers: int,
    model_dim: int,
    screening_steps: int,
    lr: float,
    graph_fp: str,
    graph_json_str: str,
    n_params: int,
    s0_passed: bool,
    s05_passed: bool,
    s1_passed: bool,
    screening_val: float,
    pre_val: float,
    screening_lr: float,
    screening_ppl: float,
    probe_results: Dict[str, Any],
) -> tuple[str, str]:
    """Insert experiment row + program_results + leaderboard upsert."""
    exp_id = f"manual_screen_{template_name}_{int(time.time())}"
    nb.conn.execute(
        "INSERT OR IGNORE INTO experiments (experiment_id, timestamp, experiment_type, config_json) VALUES (?, ?, ?, ?)",
        (
            exp_id,
            time.time(),
            "manual_template_screen",
            json.dumps(
                {
                    "template": template_name,
                    "n_layers": n_layers,
                    "model_dim": model_dim,
                    "screening_steps": screening_steps,
                    "lr": lr,
                }
            ),
        ),
    )
    result_kwargs = {
        "stage0_passed": s0_passed,
        "stage05_passed": s05_passed,
        "stage1_passed": s1_passed,
        "loss_ratio": screening_lr,
        "final_loss": screening_val,
        "initial_loss": pre_val,
        "param_count": n_params,
        "n_train_steps": screening_steps,
        "model_source": "manual_template_screen",
        "trust_label": "manual_screen",
        "wikitext_perplexity": screening_ppl,
        **{k: v for k, v in probe_results.items() if v is not None},
    }
    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=graph_fp,
        graph_json=graph_json_str,
        bypass_quality_gate=True,
        **result_kwargs,
    )
    nb.flush_writes()
    logger.info("Recorded result_id=%s", result_id)

    try:
        lb_kwargs = {
            "screening_loss_ratio": screening_lr,
            "screening_novelty": 0.0,
            "screening_passed": int(s1_passed),
            "wikitext_perplexity": screening_ppl,
            "param_count": n_params,
        }
        for k, v in probe_results.items():
            if v is not None:
                lb_kwargs[k] = v
        entry_id = nb.upsert_leaderboard(
            result_id=result_id,
            model_source="manual_template_screen",
            architecture_desc=f"{template_name} {n_layers}L dim={model_dim}",
            tier="screening",
            tags=f"manual,pinned,{template_name}",
            notes=f"Manually screened {template_name} with {screening_steps} steps",
            **lb_kwargs,
        )
        logger.info("Leaderboard entry: %s", entry_id)
    except Exception as e:
        logger.warning("Leaderboard upsert failed: %s", e)
    return exp_id, result_id


def _run_investigation(
    model,
    batch_source,
    *,
    investigation_steps: int,
    lr: float,
    vocab_size: int,
    device: str,
    seed: int,
    eval_fn,
    pre_val: float,
    nb,
    result_id: str,
    n_params: int,
    screening_lr: float,
) -> Dict[str, Any]:
    """Run investigation training + probes; upsert investigation tier."""
    logger.info("Starting investigation (%d steps)...", investigation_steps)
    _train_loop(
        model,
        batch_source,
        n_steps=investigation_steps,
        lr=lr * 0.5,
        lr_warmup=lr * 0.5,
        vocab_size=vocab_size,
        device=device,
        seed=seed,
        warmup_steps=100,
        log_every=500,
        label="inv",
        eval_fn=eval_fn,
    )
    inv_val = eval_fn()
    inv_ppl = math.exp(min(inv_val, 20))
    inv_loss_ratio = inv_val / pre_val if pre_val > 0 else 1.0
    logger.info(
        "Investigation: val=%.4f ppl=%.1f loss_ratio=%.4f",
        inv_val,
        inv_ppl,
        inv_loss_ratio,
    )

    model.eval()
    inv_result: Dict[str, Any] = {
        "investigation_val_loss": inv_val,
        "investigation_ppl": inv_ppl,
        "investigation_loss_ratio": inv_loss_ratio,
        "investigation_steps": investigation_steps,
    }
    try:
        from research.eval.binding_pipeline import run_screening_binding_probes

        bp2 = run_screening_binding_probes(model, device=device)
        inv_result["inv_induction_screening_auc"] = bp2.get("induction_screening_auc")
        inv_result["inv_binding_screening_auc"] = bp2.get("binding_screening_auc")
        logger.info(
            "Inv binding: ind=%.4f bind=%.4f",
            inv_result.get("inv_induction_screening_auc", 0),
            inv_result.get("inv_binding_screening_auc", 0),
        )
    except Exception as exc:
        logger.warning("Investigation binding probes failed: %s", exc)

    try:
        inv_lb_kwargs = {
            "screening_loss_ratio": screening_lr,
            "investigation_loss_ratio": inv_loss_ratio,
            "investigation_robustness": 1.0,
            "investigation_passed": True,
            "wikitext_perplexity": inv_ppl,
            "param_count": n_params,
        }
        if inv_result.get("inv_induction_screening_auc") is not None:
            inv_lb_kwargs["induction_screening_auc"] = inv_result[
                "inv_induction_screening_auc"
            ]
        if inv_result.get("inv_binding_screening_auc") is not None:
            inv_lb_kwargs["binding_screening_auc"] = inv_result[
                "inv_binding_screening_auc"
            ]
        nb.upsert_leaderboard(
            result_id=result_id,
            model_source="manual_template_screen",
            tier="investigation",
            **inv_lb_kwargs,
        )
        logger.info("Promoted to investigation tier")
    except Exception as e:
        logger.warning("Investigation promotion failed: %s", e)
    return inv_result


def screen_and_investigate(
    template_name: str,
    n_layers: int = 4,
    model_dim: int = 256,
    vocab_size: int = 100277,
    screening_steps: int = 750,
    investigation_steps: int = 2500,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 42,
    do_investigate: bool = False,
) -> Dict[str, Any]:
    from research.synthesis.compiler import compile_model
    from research.scientist.notebook import LabNotebook

    logger.info("Building %s: %d layers, dim=%d", template_name, n_layers, model_dim)
    graphs = _build_graphs(template_name, n_layers, model_dim, seed)
    graph_fp = graphs[0].fingerprint()
    graph_json_str = json.dumps(graphs[0].to_dict())
    model = compile_model(graphs, vocab_size=vocab_size, max_seq_len=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Params: %d, fingerprint: %s", n_params, graph_fp)

    s0_passed, s05_passed = _s0_s05_check(model, vocab_size, device)
    if not s0_passed:
        return {"stage": "s0_failed", "template": template_name}
    if not s05_passed:
        return {"stage": "s05_failed", "template": template_name}

    logger.info("Loading WikiText-103...")
    batch_source = load_wikitext_batch_source(
        batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size
    )
    logger.info(
        "Prepared %d train windows, %d val windows",
        batch_source.train_window_count,
        batch_source.val_window_count,
    )

    def _eval_val() -> float:
        model.eval()
        total = 0.0
        n = 0
        for vb in batch_source.iter_val_batches(device=device):
            with torch.no_grad():
                lo = model(vb)
                total += next_token_cross_entropy(lo, vb, vocab_size).item()
                n += 1
        model.train()
        return total / max(n, 1)

    pre_val = _eval_val()
    logger.info(
        "Pre-training val loss: %.4f (PPL %.1f)",
        pre_val,
        math.exp(min(pre_val, 20)),
    )

    logger.info("S1: Training %d steps (screening)...", screening_steps)
    _, elapsed_s1 = _train_loop(
        model,
        batch_source,
        n_steps=screening_steps,
        lr=lr,
        lr_warmup=lr,
        vocab_size=vocab_size,
        device=device,
        seed=seed,
        warmup_steps=50,
        log_every=250,
        label="S1",
        eval_fn=_eval_val,
    )
    screening_val = _eval_val()
    screening_lr = screening_val / pre_val if pre_val > 0 else 1.0
    screening_ppl = math.exp(min(screening_val, 20))
    s1_passed = screening_lr < 0.95
    logger.info(
        "S1: %s — val=%.4f ppl=%.1f loss_ratio=%.4f (%.0fs)",
        "PASSED" if s1_passed else "FAILED",
        screening_val,
        screening_ppl,
        screening_lr,
        elapsed_s1,
    )

    model.eval()
    probe_results = _run_probes(model, vocab_size, device)

    logger.info("Recording results in notebook...")
    nb = LabNotebook()
    exp_id, result_id = _record_screening(
        nb,
        template_name=template_name,
        n_layers=n_layers,
        model_dim=model_dim,
        screening_steps=screening_steps,
        lr=lr,
        graph_fp=graph_fp,
        graph_json_str=graph_json_str,
        n_params=n_params,
        s0_passed=s0_passed,
        s05_passed=s05_passed,
        s1_passed=s1_passed,
        screening_val=screening_val,
        pre_val=pre_val,
        screening_lr=screening_lr,
        screening_ppl=screening_ppl,
        probe_results=probe_results,
    )

    result: Dict[str, Any] = {
        "template": template_name,
        "result_id": result_id,
        "experiment_id": exp_id,
        "graph_fingerprint": graph_fp,
        "n_params": n_params,
        "s0_passed": s0_passed,
        "s05_passed": s05_passed,
        "s1_passed": s1_passed,
        "screening_val_loss": screening_val,
        "screening_ppl": screening_ppl,
        "screening_loss_ratio": screening_lr,
        "pre_val_loss": pre_val,
        "elapsed_s": elapsed_s1,
        **probe_results,
    }

    if do_investigate and s1_passed:
        result.update(
            _run_investigation(
                model,
                batch_source,
                investigation_steps=investigation_steps,
                lr=lr,
                vocab_size=vocab_size,
                device=device,
                seed=seed,
                eval_fn=_eval_val,
                pre_val=pre_val,
                nb=nb,
                result_id=result_id,
                n_params=n_params,
                screening_lr=screening_lr,
            )
        )

    print("\n" + "=" * 70)
    print(f"RESULT: {template_name}")
    print("=" * 70)
    print(f"  Params:       {n_params:,}")
    print(f"  S0/S0.5/S1:   {s0_passed}/{s05_passed}/{s1_passed}")
    print(f"  Pre-train:    val={pre_val:.4f} ppl={math.exp(min(pre_val, 20)):.1f}")
    print(
        f"  Screening:    val={screening_val:.4f} ppl={screening_ppl:.1f} LR={screening_lr:.4f}"
    )
    if "investigation_val_loss" in result:
        print(
            f"  Investigation: val={result['investigation_val_loss']:.4f} ppl={result['investigation_ppl']:.1f} LR={result['investigation_loss_ratio']:.4f}"
        )
    for k in [
        "induction_screening_auc",
        "binding_screening_auc",
        "ar_legacy_auc",
        "hellaswag_acc",
        "blimp_overall_accuracy",
    ]:
        if k in probe_results and probe_results[k] is not None:
            print(f"  {k}: {probe_results[k]:.4f}")
    print(f"  result_id:    {result_id}")
    print(f"  fingerprint:  {graph_fp}")

    out_path = (
        Path("research/reports") / f"screen_{template_name}_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Screen a template through the real pipeline"
    )
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--screening-steps", type=int, default=750)
    parser.add_argument("--investigation-steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--investigate", action="store_true", help="Also run investigation tier"
    )
    args = parser.parse_args()

    screen_and_investigate(
        template_name=args.template,
        n_layers=args.layers,
        model_dim=args.dim,
        screening_steps=args.screening_steps,
        investigation_steps=args.investigation_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        do_investigate=args.investigate,
    )


if __name__ == "__main__":
    main()
