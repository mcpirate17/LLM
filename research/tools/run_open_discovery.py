"""Run the open-ended, name-free mechanism discovery loop.

Samples programs (parametric atoms + mixer), fingerprints each by its physics
invariants, grades it on the label-free induction/binding capability probe, and
keeps the most capable program per physics niche (MAP-Elites). Prints the niche
map + a capability leaderboard. No mechanism catalog is consulted — every winner
is named after the fact by the niche it fell into.

    python -m research.tools.run_open_discovery --iters 120 --dim 32
"""

from __future__ import annotations

import argparse

from research.synthesis.open_discovery import OpenDiscovery


def main() -> None:
    p = argparse.ArgumentParser(description="open-ended name-free mechanism discovery")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top", type=int, default=15)
    p.add_argument(
        "--no-novelty-aware",
        dest="novelty_aware",
        action="store_false",
        default=True,
        help="disable the geometric-novelty MAP-Elites axis (NM-10); on by default",
    )
    args = p.parse_args()

    disc = OpenDiscovery(
        dim=args.dim,
        n_seeds=args.n_seeds,
        device=args.device,
        novelty_aware=args.novelty_aware,
    )
    res = disc.run(iters=args.iters, seed=args.seed)
    elites = res.archive.elites
    cov = len(elites) / res.archive.total_cells

    print(
        f"evaluated={res.evaluated}  inserted={res.inserted}  "
        f"niches={len(elites)}/{res.archive.total_cells}  coverage={cov:.2f}"
    )
    print()
    print(f"{'fit':>6}  perm shift scale egain specr  niche          program")
    for e in res.leaderboard(top=args.top):
        d = e.descriptors
        print(
            f"{e.fitness:6.3f}  "
            f"{d['perm_equivariance']:.2f} {d['shift_equivariance']:>4.2f} "
            f"{d['scale_homogeneity']:>5.2f} {d['energy_gain']:>5.2f} "
            f"{d['spectral_radius']:>5.2f}  {str(e.niche):13s}  {e.payload.key}"
        )


if __name__ == "__main__":
    main()
