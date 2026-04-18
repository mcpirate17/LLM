#!/usr/bin/env python3
"""Validate and evaluate generated adaptive lane workflows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aria_designer.runtime.bridge import (
    evaluate_workflow as direct_bridge_evaluate,
    validate_workflow_graph as direct_bridge_validate,
)

DEFAULT_WORKFLOW_DIR = ROOT / "workflows" / "generated"
DEFAULT_API_BASE = "http://127.0.0.1:8091/api/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", type=Path, default=DEFAULT_WORKFLOW_DIR)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def _post(
    session: requests.Session,
    api_base: str,
    timeout: float,
    path: str,
    payload: dict[str, Any],
) -> requests.Response:
    return session.post(f"{api_base}{path}", json=payload, timeout=timeout)


def _load_workflows(workflow_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (workflow_dir / "adaptive_lane_manifest.json").read_text(encoding="utf-8")
    )
    items = []
    for item in manifest:
        workflow = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        items.append({"meta": item, "workflow": workflow})
    return items


def main() -> None:
    args = _parse_args()
    workflow_dir = args.workflow_dir.resolve()
    report: dict[str, Any] = {"api_base": args.api_base, "workflows": []}
    budget = {
        "model_dim": 256,
        "vocab_size": 32000,
        "device": "cpu",
        "batch_size": 2,
        "seq_len": 64,
        "run_fingerprint": True,
        "run_novelty": True,
    }
    with requests.Session() as session:
        for item in _load_workflows(workflow_dir):
            workflow = item["workflow"]
            entry: dict[str, Any] = {
                "workflow_id": workflow["workflow_id"],
                "name": workflow["name"],
                "variant": item["meta"]["variant"],
            }
            try:
                validate_resp = _post(
                    session,
                    args.api_base,
                    args.timeout,
                    "/workflows/validate",
                    {"workflow": workflow},
                )
                entry["validate_status"] = validate_resp.status_code
                entry["validate"] = validate_resp.json()
            except Exception as exc:
                entry["validate_status"] = None
                entry["validate"] = {"error": str(exc)}

            try:
                entry["direct_validate"] = direct_bridge_validate(
                    workflow,
                    model_dim=budget["model_dim"],
                )
            except Exception as exc:
                entry["direct_validate"] = {"error": str(exc)}

            try:
                t0 = time.perf_counter()
                preview_resp = _post(
                    session,
                    args.api_base,
                    args.timeout,
                    "/workflows/preview",
                    {"workflow": workflow},
                )
                entry["preview_elapsed_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 3
                )
                entry["preview_status"] = preview_resp.status_code
                entry["preview"] = preview_resp.json()
            except Exception as exc:
                entry["preview_elapsed_ms"] = None
                entry["preview_status"] = None
                entry["preview"] = {"error": str(exc)}

            try:
                eval_resp = _post(
                    session,
                    args.api_base,
                    args.timeout,
                    "/workflows/evaluate",
                    {"workflow": workflow, "budget": budget},
                )
                entry["evaluate_status"] = eval_resp.status_code
                entry["evaluate"] = eval_resp.json()
            except Exception as exc:
                entry["evaluate_status"] = None
                entry["evaluate"] = {"error": str(exc)}

            try:
                t0 = time.perf_counter()
                direct = direct_bridge_evaluate(
                    workflow,
                    model_dim=budget["model_dim"],
                    vocab_size=budget["vocab_size"],
                    device=budget["device"],
                    run_fingerprint=budget["run_fingerprint"],
                    run_novelty=budget["run_novelty"],
                    batch_size=budget["batch_size"],
                    seq_len=budget["seq_len"],
                )
                entry["direct_bridge_elapsed_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 3
                )
                entry["direct_bridge"] = (
                    direct.to_dict() if hasattr(direct, "to_dict") else dict(direct)
                )
            except Exception as exc:
                entry["direct_bridge_elapsed_ms"] = None
                entry["direct_bridge"] = {"error": str(exc)}
            report["workflows"].append(entry)

    out_path = workflow_dir / "adaptive_lane_benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
