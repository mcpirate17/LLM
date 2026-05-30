"""Full-BLiMP evaluation of the HYDRA mod_mor/ccgqa checkpoint.

HYDRA is a separate architecture (Mixture-of-Depths + Mixture-of-Recursions,
CCGQA attention) trained with the GPT-2 tokenizer (vocab 50257). It is NOT a
TinyLM, so this bespoke scorer: rebuilds HydraModel from the checkpoint's saved
config (mirroring trainer_startup._build_hydra_model), strict=False loads (as
HYDRA's own resume does) and reports key coverage, then scores BLiMP minimal
pairs with the GPT-2 BPE (tiktoken 'gpt2') by mean per-token log-prob — a wrong
load would show low coverage rather than a silently-bad score.

Usage:
    python -m research.tools.hydra_blimp_eval --ckpt HYDRA/checkpoints/hydra_700m_final.pt \
        --n-per-subtask 1000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "HYDRA"))

from research.eval.blimp_eval import _download_blimp  # noqa: E402
from research.tools.eval_checkpoints_blimp import _rollup  # noqa: E402


def _build_hydra_from_config(cfg: dict):
    from hydra.model.framework.model import HydraModel

    return HydraModel(
        vocab_size=cfg["vocab_size"],
        dim=cfg["mod_mor_dim"],
        n_mor_blocks=cfg["n_mor_blocks"],
        recursions_per_block=cfg["mor_recursions"],
        n_heads=cfg["mod_mor_n_heads"],
        n_kv_heads=cfg["mod_mor_n_kv_heads"],
        compression_factor=4,
        mlp_ratio=3.6,
        max_seq_len=cfg.get("max_seq_len", 1024),
        mod_capacity=cfg["mod_capacity"],
        aux_loss_weight=cfg.get("aux_scale", 0.03),
        adaptive=cfg["mor_adaptive"],
        tie_weights=True,
        attention_backend=cfg.get("attention_backend", "ccgqa"),
        mor_min_depth=cfg.get("mor_min_depth", 0),
        moe_enabled=cfg["moe_enabled"],
        moe_num_experts=cfg["moe_num_experts"],
        moe_num_layers=cfg["moe_num_layers"],
        moe_top_k=cfg["moe_top_k"],
        moe_aux_weight=cfg.get("moe_aux_weight", 0.01),
        moe_router_jitter=cfg.get("moe_router_jitter", 0.0),
        moe_expert_diversity_noise=cfg.get("moe_expert_diversity_noise", 0.0),
        moe_warmup_steps=cfg.get("moe_warmup_steps", 1000),
        moe_identity_init=cfg.get("moe_identity_init", True),
        manifold_enabled=cfg.get("manifold_enabled", False),
    )


@torch.no_grad()
def _mean_logprobs(model, batch_ids: list[list[int]], device: str) -> list[float]:
    """Mean per-token causal log-prob for each (unpadded) id sequence.

    Right-pads the batch; causal model => real (leading) positions are unaffected
    by trailing pad tokens, so no attention mask is needed for correctness."""
    lens = [len(s) for s in batch_ids]
    maxlen = max(lens)
    pad = 0
    x = torch.full((len(batch_ids), maxlen), pad, dtype=torch.long, device=device)
    for i, s in enumerate(batch_ids):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    out = model(x, return_losses=False)
    logits = out[0] if isinstance(out, tuple) else out  # [B,L,V]
    logp = F.log_softmax(logits.float(), dim=-1)
    results = []
    for i, L in enumerate(lens):
        if L < 2:
            results.append(0.0)
            continue
        # predict token t from position t-1, for t in 1..L-1
        tgt = x[i, 1:L]
        lp = logp[i, : L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        results.append(float(lp.mean()))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--n-per-subtask", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # First-party HYDRA checkpoint (own training output); needs the config dict.
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # nosec B614
    cfg = payload["config"]
    sd = payload["model"]
    model = _build_hydra_from_config(cfg)
    incompat = model.load_state_dict(sd, strict=False)
    n_model = len(list(model.state_dict().keys()))
    missing, unexpected = len(incompat.missing_keys), len(incompat.unexpected_keys)
    coverage = round((n_model - missing) / n_model, 4)
    print(
        json.dumps(
            {
                "load": "ok",
                "step": int(payload.get("step", 0) or 0),
                "n_params_m": round(
                    sum(p.numel() for p in model.parameters()) / 1e6, 1
                ),
                "key_coverage": coverage,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "missing_sample": incompat.missing_keys[:6],
            }
        ),
        flush=True,
    )
    dtype = getattr(torch, args.dtype)
    model = model.to(args.device).to(dtype).eval()
    if hasattr(model, "set_mor_curriculum") and cfg.get("mor_adaptive"):
        try:
            model.set_mor_curriculum(enable_step=0, rampup_steps=0)
        except Exception as e:
            print(json.dumps({"warn": f"set_mor_curriculum failed: {e}"}), flush=True)

    enc = tiktoken.get_encoding("gpt2")
    subtasks = _download_blimp()
    n = args.n_per_subtask
    per_sub = {}
    t0 = time.monotonic()
    for si, (sub, examples) in enumerate(sorted(subtasks.items())):
        ex = examples[:n]
        correct = 0
        bs = args.batch_size
        for j in range(0, len(ex), bs):
            chunk = ex[j : j + bs]
            goods = [enc.encode(e["good"]) for e in chunk]
            bads = [enc.encode(e["bad"]) for e in chunk]
            gl = _mean_logprobs(model, goods, args.device)
            bl = _mean_logprobs(model, bads, args.device)
            correct += sum(1 for g, b in zip(gl, bl) if g > b)
        per_sub[sub] = correct / len(ex)
        if si % 10 == 0:
            print(
                json.dumps(
                    {
                        "progress": f"{si + 1}/{len(subtasks)}",
                        "last": sub,
                        "acc": round(per_sub[sub], 3),
                    }
                ),
                flush=True,
            )
    overall = sum(per_sub.values()) / len(per_sub)
    result = {
        "ckpt": args.ckpt.name,
        "arch": "hydra_mod_mor_ccgqa",
        "step": int(payload.get("step", 0) or 0),
        "key_coverage": coverage,
        "blimp_overall": round(overall, 4),
        "n_subtasks": len(per_sub),
        "n_per_subtask": n,
        "by_category": _rollup(per_sub),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    print(json.dumps(result), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
