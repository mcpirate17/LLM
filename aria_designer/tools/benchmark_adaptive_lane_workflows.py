#!/usr/bin/env python3
"""Validate and evaluate generated adaptive lane workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
import sys

sys.path.insert(0, str(ROOT := Path("/home/tim/Projects/LLM/aria_designer")))

from runtime.bridge import (
    evaluate_workflow as direct_bridge_evaluate,
    validate_workflow_graph as direct_bridge_validate,
)
WORKFLOW_DIR = ROOT / "workflows" / "generated"
API_BASE = "http://127.0.0.1:8091/api/v1"


def _post(path: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(f"{API_BASE}{path}", json=payload, timeout=180)


def _load_workflows() -> list[dict[str, Any]]:
    manifest = json.loads((WORKFLOW_DIR / "adaptive_lane_manifest.json").read_text(encoding="utf-8"))
    items = []
    for item in manifest:
        workflow = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        items.append({"meta": item, "workflow": workflow})
    return items


def main() -> None:
    report: dict[str, Any] = {"api_base": API_BASE, "workflows": []}
    budget = {
        "model_dim": 256,
        "vocab_size": 32000,
        "device": "cpu",
        "batch_size": 2,
        "seq_len": 64,
        "run_fingerprint": True,
        "run_novelty": True,
    }
    for item in _load_workflows():
        workflow = item["workflow"]
        entry: dict[str, Any] = {
            "workflow_id": workflow["workflow_id"],
            "name": workflow["name"],
            "variant": item["meta"]["variant"],
        }
        try:
            validate_resp = _post("/workflows/validate", {"workflow": workflow})
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
            preview_resp = _post("/workflows/preview", {"workflow": workflow})
            entry["preview_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            entry["preview_status"] = preview_resp.status_code
            entry["preview"] = preview_resp.json()
        except Exception as exc:
            entry["preview_elapsed_ms"] = None
            entry["preview_status"] = None
            entry["preview"] = {"error": str(exc)}

        try:
            eval_resp = _post("/workflows/evaluate", {"workflow": workflow, "budget": budget})
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
            entry["direct_bridge_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            entry["direct_bridge"] = direct.to_dict() if hasattr(direct, "to_dict") else dict(direct)
        except Exception as exc:
            entry["direct_bridge_elapsed_ms"] = None
            entry["direct_bridge"] = {"error": str(exc)}
        report["workflows"].append(entry)

    out_path = WORKFLOW_DIR / "adaptive_lane_benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
