"""End-to-end CPU-vs-CUDA check + step-time benchmark for the native bilane TinyLM.

Builds the exact lane used by the 104M run, copies identical weights to CPU and
CUDA, runs forward+backward on the same input, confirms loss and a gradient match,
then times a full train step (fwd+bwd+opt) on each device.

Run:  python -m research.tools.bench_native_model_cuda
"""

from __future__ import annotations

import time

import torch
from torch import nn

from research.defaults import VOCAB_SIZE
from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

LANE = "native_semiring_adapt_bilane_m32_g0_t1_b1_l0_h2_r4_surprise_memory"


def _build(device: str):
    torch.manual_seed(0)
    factory = _build_lane_factory(LANE)
    return _build_tinylm(
        factory,
        dim=256,
        n_blocks=4,
        vocab_size=VOCAB_SIZE,
        max_seq_len=1024,
        use_ffn=True,
    ).to(device)


def _loss(model, ids, labels):
    logits = model(ids)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )


def main() -> None:
    B, L = 16, 256
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB_SIZE, (B, L), generator=g)
    labels = torch.randint(0, VOCAB_SIZE, (B, L), generator=g)

    cpu = _build("cpu")
    cuda = _build("cuda")
    cuda.load_state_dict(cpu.state_dict())  # identical weights

    # correctness: same loss, same embedding grad
    lc = _loss(cpu, ids, labels)
    lc.backward()
    gc = cpu.embed.weight.grad.detach().clone()
    lg = _loss(cuda, ids.cuda(), labels.cuda())
    lg.backward()
    gg = cuda.embed.weight.grad.detach().cpu()
    print(
        f"loss cpu={lc.item():.6f} cuda={lg.item():.6f} diff={abs(lc.item() - lg.item()):.3e}"
    )
    rel = (gc - gg).abs().max().item() / (gc.abs().max().item() + 1e-9)
    print(f"embed grad rel diff={rel:.3e}  {'OK' if rel < 1e-2 else 'FAIL'}")

    # step-time benchmark
    for name, model, dev in [("CPU", cpu, "cpu"), ("CUDA", cuda, "cuda")]:
        ii = ids.to(dev)
        ll = labels.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):  # warm
            opt.zero_grad(set_to_none=True)
            _loss(model, ii, ll).backward()
            opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            _loss(model, ii, ll).backward()
            opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        print(f"{name:5s} step: {(time.perf_counter() - t0) / n * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
