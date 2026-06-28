"""Shared CLI plumbing for ``component_fab.tools`` runners."""

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
from component_fab.state.provenance import build_run_provenance
from component_fab.state.schema_versions import RUN_REPORT_SCHEMA_VERSION

_REPO = Path(__file__).resolve().parents[2]


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    ledger_default: Path | str = DEFAULT_LEDGER_PATH,
    output_default: Path | str | None = None,
    output_help: str = "write the JSON report to this path",
    dry_run: bool = False,
    quiet: bool = False,
) -> argparse.ArgumentParser:
    """Register shared runner args."""

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
    """Open the ledger with the standard full-history policy."""

    if isinstance(args_or_path, argparse.Namespace):
        path = getattr(args_or_path, "ledger", None) or args_or_path.ledger_path
    else:
        path = args_or_path
    return Ledger(path, include_rotated=rotated)


def _with_report_metadata(payload: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    """Add schema/provenance metadata to dict reports exactly once."""

    if not isinstance(payload, dict):
        return payload
    if "run_metadata" in payload or "schema_version" in payload:
        return payload
    return {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "run_metadata": build_run_provenance(repo_root=_REPO),
        **payload,
    }


def write_report(
    payload: dict[str, Any] | list[Any],
    *,
    default_dir: Path,
    prefix: str,
    output: Path | str | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> Path | None:
    """Write a timestamped JSON report, or print it for ``--dry-run``."""

    payload = _with_report_metadata(payload)
    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output) if output else default_dir / f"{prefix}_{stamp}.json"
    write_json_report(payload, out, default=str)
    if not quiet:
        print(f"wrote: {out}")
    return out
