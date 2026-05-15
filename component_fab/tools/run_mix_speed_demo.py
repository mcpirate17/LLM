"""CLI: demo the mix_speed metric on a few reference forward fns.

Demonstrates the spectrum from "doesn't mix" (rmsnorm-style) through
"local mixer" (boxcar conv) through "global mixer" (mean across positions).

Usage:
    python -m component_fab.tools.run_mix_speed_demo
"""

from __future__ import annotations

import torch

from component_fab.metrics.mix_speed import MixSpeedScorecard, measure_mix_speed


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _rmsnorm_like(x: torch.Tensor) -> torch.Tensor:
    norm = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
    return x / norm


def _local_boxcar_conv(x: torch.Tensor) -> torch.Tensor:
    pad = torch.nn.functional.pad(x, (0, 0, 1, 1))
    return (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0


def _global_mean_broadcast(x: torch.Tensor) -> torch.Tensor:
    return x + x.mean(dim=1, keepdim=True)


def _causal_running_mean(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    weights = torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(-1, 1)
    return x.cumsum(dim=1) / weights


def _print_card(name: str, card: MixSpeedScorecard) -> None:
    head = (
        f"{name:<24}  half_life={card.mix_half_life:>5}  "
        f"peak@offset={card.peak_response_at_offset:>2}  "
        f"global={'Y' if card.mixes_globally else 'N'}  "
        f"local_only={'Y' if card.is_pure_local else 'N'}"
    )
    print(head)
    decay = card.response_decay
    sample = [f"{decay[i]:.2e}" for i in range(min(10, len(decay)))]
    print(f"    decay[0:10] = [{', '.join(sample)}]")


def main() -> int:
    cases = {
        "identity": _identity,
        "rmsnorm_like": _rmsnorm_like,
        "local_boxcar_conv": _local_boxcar_conv,
        "causal_running_mean": _causal_running_mean,
        "global_mean_broadcast": _global_mean_broadcast,
    }
    print("mix_speed scorecard — synthetic reference ops")
    print("=" * 78)
    for name, fn in cases.items():
        card = measure_mix_speed(fn, seq_len=64, feature_dim=16, n_trials=4)
        _print_card(name, card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
