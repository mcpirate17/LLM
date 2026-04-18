from __future__ import annotations

import argparse

import torch

from research.tools.backfill import run_backfill


def add_common_backfill_args(
    parser: argparse.ArgumentParser,
    *,
    default_top: int,
    default_tier: str,
    allow_device_auto: bool = False,
    include_timeout: bool = False,
) -> None:
    parser.add_argument("--top", type=int, default=default_top)
    parser.add_argument("--tier", default=default_tier)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device",
        default="auto"
        if allow_device_auto
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if include_timeout:
        parser.add_argument("--timeout", type=int, default=30)


def resolve_device(raw: str) -> str:
    device = str(raw)
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_legacy_backfill(
    *,
    probes: tuple[str, ...],
    tier_csv: str,
    top_per_tier: int,
    device: str,
    force: bool,
    dry_run: bool,
    fp_timeout: int = 30,
) -> None:
    run_backfill(
        probes=probes,
        tiers=tuple(t.strip() for t in tier_csv.split(",") if t.strip()),
        top_per_tier=int(top_per_tier),
        device=resolve_device(device),
        train_steps=500,
        n_passes=1,
        force=bool(force),
        dry_run=bool(dry_run),
        fp_timeout=int(fp_timeout),
    )
