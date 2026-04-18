#!/usr/bin/env python
"""Chinchilla-optimal head-to-head: GLA vs GPT-2 on full WikiText-103.

Trains both architectures to chinchilla-optimal token count (~6.6 epochs
of WikiText-103 for 39M params) with cosine LR decay, checkpointed
metrics throughout.

Usage:
    python -m research.tools.chinchilla_headtohead --device cuda
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VOCAB = 50257
DEVICE = "cuda"


def _build_model_bundle(template_name, n_layers, dim, seed):
    from research.synthesis.compiler import compile_model
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.reference_architectures import build_gpt2_layer
    from research.synthesis.templates import apply_template

    if template_name == "gpt2_reference":
        graphs = [build_gpt2_layer(dim) for _ in range(n_layers)]
        model = compile_model(graphs, vocab_size=VOCAB, max_seq_len=512)
        return model, graphs

    rng = random.Random(seed)
    graphs = []
    for _ in range(n_layers):
        g = ComputationGraph(model_dim=dim)
        inp = g.add_input()
        out = apply_template(g, inp, rng, template_name=template_name)
        g.set_output(out)
        graphs.append(g)
    model = compile_model(graphs, vocab_size=VOCAB, max_seq_len=512)
    return model, graphs


def _load_full_wikitext(batch_size, seq_len, device):
    import tiktoken
    from research.eval.wikitext_eval import _WIKITEXT_CACHE_DIR

    full_cache = _WIKITEXT_CACHE_DIR / "wikitext-103-raw-v1-full"
    train_path = full_cache / "train.txt"
    val_path = full_cache / "validation.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Full WikiText not cached. Run download first: {train_path}")

    enc = tiktoken.get_encoding("gpt2")
    logger.info("Tokenizing full WikiText-103 (may take a minute)...")

    train_text = train_path.read_text(encoding="utf-8", errors="replace")
    train_tokens = torch.tensor(enc.encode(train_text, allowed_special=set()), dtype=torch.long)
    del train_text

    val_text = val_path.read_text(encoding="utf-8", errors="replace")
    val_tokens = torch.tensor(enc.encode(val_text, allowed_special=set()), dtype=torch.long)
    del val_text

    stride = batch_size * seq_len
    # Keep data on CPU — move to GPU per-batch to avoid OOM
    train_w = [
        train_tokens[i * stride : (i + 1) * stride].reshape(batch_size, seq_len)
        for i in range(len(train_tokens) // stride)
    ]
    val_w = [
        val_tokens[i * stride : (i + 1) * stride].reshape(batch_size, seq_len)
        for i in range(min(128, len(val_tokens) // stride))
    ]
    logger.info(
        "Tokens: %dM train, %dK val. Windows: %d train, %d val",
        len(train_tokens) // 1_000_000,
        len(val_tokens) // 1000,
        len(train_w),
        len(val_w),
    )
    corpus_info = {
        "variant": "wikitext-103-raw-v1-full",
        "cache_dir": str(full_cache),
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "train_tokens": int(len(train_tokens)),
        "validation_tokens": int(len(val_tokens)),
        "train_windows": int(len(train_w)),
        "validation_windows": int(len(val_w)),
        "tokenizer": "tiktoken:gpt2",
        "vocab_size": VOCAB,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
    }
    return train_w, val_w, len(train_tokens), corpus_info


def _eval_loss(model, val_w):
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for b in val_w:
            b = b.to(device)
            total += nn.functional.cross_entropy(
                model(b)[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1)
            ).item()
    model.train()
    return total / max(len(val_w), 1)


def _run_probes(model, device):
    model.eval()
    probes = {}
    try:
        from research.eval.binding_pipeline import run_screening_binding_probes
        bp = run_screening_binding_probes(model, device=device)
        probes.update({
            "induction_auc": bp.get("induction_auc", 0),
            "binding_auc": bp.get("binding_auc", 0),
        })
    except Exception as e:
        logger.warning("Binding: %s", e)
    try:
        from research.eval.associative_recall import associative_recall_score
        ar = associative_recall_score(model, n_pairs=10, n_eval=100, n_train_steps=300, batch_size=8, device=device)
        probes["ar_auc"] = ar.auc
    except Exception as e:
        logger.warning("AR: %s", e)
    try:
        from research.eval.hellaswag_eval import evaluate_hellaswag
        h = evaluate_hellaswag(model, vocab_size=VOCAB, device=device, n_examples=200)
        probes["hellaswag_acc"] = h.get("hellaswag_acc")
    except Exception as e:
        logger.warning("HellaSwag: %s", e)
    try:
        from research.eval.blimp_eval import evaluate_blimp
        bl = evaluate_blimp(model, vocab_size=VOCAB, device=device, n_per_subtask=50, timeout_s=120)
        probes["blimp_accuracy"] = bl.overall_accuracy
    except Exception as e:
        logger.warning("BLiMP: %s", e)
    model.train()
    return probes


def train_chinchilla(
    template_name: str,
    n_steps: int,
    train_w,
    val_w,
    corpus_info,
    n_layers: int = 4,
    model_dim: int = 256,
    lr: float = 3e-4,
    checkpoint_every: int = 5000,
    probe_every: int = 20000,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 16,
    seq_len: int = 256,
):
    logger.info("=" * 70)
    logger.info("  %s  %dL dim=%d  %d steps  cosine LR", template_name, n_layers, model_dim, n_steps)
    logger.info("=" * 70)

    model, graphs = _build_model_bundle(template_name, n_layers, model_dim, seed)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Params: %d", n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Cosine LR schedule with warmup
    warmup_steps = min(500, n_steps // 20)

    def _lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_schedule)

    pre_val = _eval_loss(model, val_w)
    logger.info("Pre-train val: %.4f (PPL %.1f)", pre_val, math.exp(min(pre_val, 20)))

    rg = torch.Generator().manual_seed(seed)
    checkpoints = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        batch = train_w[torch.randint(len(train_w), (1,), generator=rg).item()].to(device)
        logits = model(batch)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), batch[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if not math.isfinite(loss.item()):
            logger.error("DIVERGED at step %d", step)
            break

        if step % checkpoint_every == 0 or step == n_steps:
            val_loss = _eval_loss(model, val_w)
            ppl = math.exp(min(val_loss, 20))
            current_lr = scheduler.get_last_lr()[0] * lr
            elapsed = time.time() - t0
            tokens_seen = step * batch_size * seq_len

            cp = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "ppl": round(ppl, 2),
                "lr": round(current_lr, 6),
                "tokens_M": round(tokens_seen / 1e6, 1),
                "elapsed_s": round(elapsed, 0),
            }

            if step % probe_every == 0 or step == n_steps:
                logger.info("  Running probes at step %d...", step)
                probes = _run_probes(model, device)
                cp["probes"] = probes
                probe_str = " ".join(f"{k}={v:.4f}" for k, v in probes.items() if v is not None)
                logger.info(
                    "  [%s] step=%d val=%.4f ppl=%.1f tok=%.0fM | %s (%.0fs)",
                    template_name, step, val_loss, ppl, tokens_seen / 1e6, probe_str, elapsed,
                )
            else:
                logger.info(
                    "  [%s] step=%d val=%.4f ppl=%.1f lr=%.1e tok=%.0fM (%.0fs)",
                    template_name, step, val_loss, ppl, current_lr, tokens_seen / 1e6, elapsed,
                )

            checkpoints.append(cp)

    result = {
        "template": template_name,
        "baseline_kind": (
            "canonical_reference_architecture"
            if template_name == "gpt2_reference"
            else "template_stack"
        ),
        "n_params": n_params,
        "n_steps": n_steps,
        "seed": seed,
        "n_layers": n_layers,
        "model_dim": model_dim,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "checkpoint_every": checkpoint_every,
        "probe_every": probe_every,
        "tokenizer": corpus_info["tokenizer"],
        "vocab_size": VOCAB,
        "corpus": corpus_info,
        "layer_fingerprints": [g.fingerprint() for g in graphs],
        "layer_graphs": [g.to_dict() for g in graphs],
        "pre_val": pre_val,
        "checkpoints": checkpoints,
        "final_val": checkpoints[-1]["val_loss"] if checkpoints else None,
        "final_ppl": checkpoints[-1]["ppl"] if checkpoints else None,
    }

    del model, opt
    torch.cuda.empty_cache()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Override step count (default: chinchilla-optimal)")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--probe-every", type=int, default=20000)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    train_w, val_w, n_train_tokens, corpus_info = _load_full_wikitext(
        args.batch_size, args.seq_len, args.device
    )

    # Chinchilla-optimal: ~20 × params tokens
    tokens_per_step = args.batch_size * args.seq_len
    if args.steps:
        n_steps = args.steps
    else:
        # Build a quick model to count params
        model_tmp, _ = _build_model_bundle("gated_linear_attention_block", args.layers, args.dim, 42)
        n_params = sum(p.numel() for p in model_tmp.parameters())
        chinchilla_tokens = 20 * n_params
        n_steps = chinchilla_tokens // tokens_per_step
        del model_tmp
        logger.info(
            "Chinchilla optimal: %dM tokens, %d steps (%.1f epochs of %dM)",
            chinchilla_tokens // 1_000_000, n_steps,
            chinchilla_tokens / n_train_tokens, n_train_tokens // 1_000_000,
        )

    templates = ["gated_linear_attention_block", "gpt2_reference"]
    results = []

    for name in templates:
        r = train_chinchilla(
            name, n_steps, train_w, val_w,
            corpus_info,
            n_layers=args.layers, model_dim=args.dim,
            lr=args.lr, checkpoint_every=args.checkpoint_every,
            probe_every=args.probe_every, device=args.device,
            batch_size=args.batch_size, seq_len=args.seq_len,
        )
        results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("CHINCHILLA HEAD-TO-HEAD: GLA vs GPT-2")
    print("=" * 80)
    for r in results:
        last = r["checkpoints"][-1] if r["checkpoints"] else {}
        probes = last.get("probes", {})
        probe_str = " ".join(f"{k}={v:.4f}" for k, v in probes.items() if v is not None)
        print(f"  {r['template']:40s} PPL={last.get('ppl', '?'):>8} val={last.get('val_loss', '?'):>7} params={r['n_params']:>10,}")
        if probe_str:
            print(f"  {'':40s} {probe_str}")

    if len(results) == 2:
        gla_ppl = results[0]["final_ppl"]
        gpt_ppl = results[1]["final_ppl"]
        if gla_ppl and gpt_ppl:
            ratio = gpt_ppl / gla_ppl
            print(f"\n  GLA vs GPT-2 perplexity ratio: {ratio:.1f}x")
            print(f"  {'GLA WINS' if ratio > 1 else 'GPT-2 WINS'}")

    out = Path("research/reports") / f"chinchilla_headtohead_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
