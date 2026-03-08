"""Benchmark: aria_core cumsum vs PyTorch cumsum."""
import torch
import time
import aria_core._C as _C

def bench(label, fn, warmup=20, repeats=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = (time.perf_counter() - t0) / repeats
    return elapsed

print(f"{'Shape':>24s}  {'PyTorch':>12s}  {'aria_core':>12s}  {'Speedup':>8s}")
print("-" * 62)

shapes = [
    (1, 64),
    (1, 256),
    (1, 1024),
    (1, 4096),
    (1, 16384),
    (32, 256),
    (32, 1024),
    (32, 4096),
    (128, 256),
    (128, 1024),
    (128, 4096),
    (512, 1024),
    (512, 4096),
]

for batch, dim in shapes:
    x = torch.randn(batch, dim)

    # Correctness check
    ref = torch.cumsum(x, dim=-1)
    out = _C.cumsum_f32(x)
    maxerr = (ref - out).abs().max().item()
    assert maxerr < 1e-3, f"Correctness failed: max error {maxerr}"

    t_pt = bench(f"pt  {batch}x{dim}", lambda: torch.cumsum(x, dim=-1))
    t_ac = bench(f"ac  {batch}x{dim}", lambda: _C.cumsum_f32(x))

    speedup = t_pt / t_ac
    tag = f"{batch}x{dim}"
    print(f"{tag:>24s}  {t_pt*1e6:>10.1f}us  {t_ac*1e6:>10.1f}us  {speedup:>7.2f}x")
