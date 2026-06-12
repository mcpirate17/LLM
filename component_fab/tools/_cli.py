"""Shared CLI plumbing for ``component_fab.tools`` runners.

Three things every runner used to hand-roll:
- the common ``--ledger/--output/--dry-run/--quiet`` argument block,
- opening the ledger (with a CONSISTENT ``include_rotated`` policy), and
- the mkdir + timestamp + ``json.dumps(indent=2)`` report-write block.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from component_fab.state.ledger import (
    DEFAULT_LEDGER_PATH,
    Ledger,
    write_json_report,
)


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    ledger_default: Path | str = DEFAULT_LEDGER_PATH,
    output_default: Path | str | None = None,
    output_help: str = "write the JSON report to this path",
    dry_run: bool = False,
    quiet: bool = False,
) -> argparse.ArgumentParser:
    """Register the genuinely shared runner args (``--ledger``/``--output``
    always; ``--dry-run``/``--quiet`` opt-in so runners that ignore them do
    not grow dead CLI surface)."""
    parser.add_argument("--ledger", default=str(ledger_default))
    parser.add_argument(
        "--output",
        type=Path,
        default=None if output_default is None else Path(output_default),
        help=output_help,
    )
    if dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="print JSON to stdout and do not write an artifact",
        )
    if quiet:
        parser.add_argument("--quiet", action="store_true")
    return parser


def open_ledger(
    args_or_path: argparse.Namespace | Path | str, *, rotated: bool = True
) -> Ledger:
    """Open the ledger with the standard full-history policy.

    Every CLI runner wants the rotated audit trail replayed (promoted
    entries from prior days live in rotations); ``rotated=True`` is the
    single shared default that used to drift per-runner.
    """
    if isinstance(args_or_path, argparse.Namespace):
        path = getattr(args_or_path, "ledger", None) or args_or_path.ledger_path
    else:
        path = args_or_path
    return Ledger(path, include_rotated=rotated)


def write_report(
    payload: dict[str, Any] | list[Any],
    *,
    default_dir: Path,
    prefix: str,
    output: Path | str | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> Path | None:
    """Write a timestamped JSON report (or print it for ``--dry-run``).

    Resolves ``output or default_dir/<prefix>_<YYYYmmdd_HHMMSS>.json``, then
    delegates mkdir + stable-dump to ``state.ledger.write_json_report``.
    Returns the written path, or ``None`` when ``dry_run`` printed instead.
    """
    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output) if output else default_dir / f"{prefix}_{stamp}.json"
    write_json_report(payload, out, default=str)
    if not quiet:
        print(f"wrote: {out}")
    return out
