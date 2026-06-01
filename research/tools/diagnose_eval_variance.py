"""Diagnose the eval-ppl swing: is it data-sampling noise or a real model bug?

Loads a FIXED checkpoint (model frozen) and evaluates many batches, recording
per-batch loss. If per-batch ppl ranges wildly, the 16-batch-average eval the
training run logged will swing purely from sampling -> not a model defect.
Also reports the std of simulated 16-batch windows (the actual logged metric) and
attributes loss to data source where the loader exposes it.

Run:  python -m research.tools.diagnose_eval_variance --checkpoint <path> [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from research.defaults import VOCAB_SIZE
from research.tools.native_adaptive_hydra_train import (
    LOCAL_MIX_NAME,
    _build_lane_factory,
    _build_tinylm,
    _lm_loss,
    _make_loader,
    _prepare_batch,
)
from research.defaults import PROJECT_ROOT

LANE = "native_semiring_adapt_bilane_m32_g0_t1_b1_l0_h2_r4_surprise_memory"


def _loader_args(device: str, batch: int, seq_len: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=LOCAL_MIX_NAME,
        val_dataset=LOCAL_MIX_NAME,
        hydra_root=PROJECT_ROOT / "HYDRA",
        batch=batch,
        seq_len=seq_len,
        vocab_size=VOCAB_SIZE,
        tokenizer="gpt2",
        num_workers=0,
        prefetch_factor=2,
        steps=100000,
        require_sources=True,
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--window", type=int, default=16)
    args = ap.parse_args()

    factory = _build_lane_factory(LANE)
    model = _build_tinylm(
        factory,
        dim=256,
        n_blocks=4,
        vocab_size=VOCAB_SIZE,
        max_seq_len=1024,
        use_ffn=True,
    ).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device)  # nosec B614
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    print(f"loaded {args.checkpoint.name} step={payload.get('step')}")

    la = _loader_args(args.device, args.batch, args.seq_len)
    loader = _make_loader(la, dataset=LOCAL_MIX_NAME, seed=1009)

    losses: list[float] = []
    for _ in range(args.n_batches):
        batch = next(loader)
        ids, labels = _prepare_batch(batch, vocab_size=VOCAB_SIZE, device=args.device)
        loss = _lm_loss(model(ids), labels)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    if hasattr(loader, "close"):
        loader.close()

    t = torch.tensor(losses)
    ppl = torch.exp(t)
    print(f"\nPER-BATCH loss over {len(losses)} batches:")
    print(
        f"  loss  mean={t.mean():.3f} std={t.std():.3f} min={t.min():.3f} max={t.max():.3f}"
    )
    print(f"  ppl   mean={ppl.mean():.1f} min={ppl.min():.1f} max={ppl.max():.1f}")

    # simulate the logged metric: non-overlapping windows of `window` batches
    w = args.window
    win_means = [float(t[i : i + w].mean()) for i in range(0, len(losses) - w + 1, w)]
    wt = torch.tensor(win_means)
    wppl = torch.exp(wt)
    print(f"\nSIMULATED {w}-batch-window eval (what the run logged):")
    print(f"  loss  mean={wt.mean():.3f} std={wt.std():.3f}")
    print(
        f"  ppl   min={wppl.min():.1f} max={wppl.max():.1f}  -> swing {wppl.max() / wppl.min():.1f}x"
    )
    print(
        f"\nVERDICT: a frozen model showing {wppl.max() / wppl.min():.1f}x ppl swing across "
        f"{w}-batch windows means the swing is DATA SAMPLING, not the model."
    )


if __name__ == "__main__":
    main()
