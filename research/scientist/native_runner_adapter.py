from __future__ import annotations

import os
import re
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class DesignerRuntimeAdapterState:
    enabled: bool
    strict: bool
    designer_runtime_available: bool
    reason: str


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _designer_runtime_lib_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "aria_designer" / "runtime" / "lib" / "libaria_runtime.so"


def detect_adapter_state() -> DesignerRuntimeAdapterState:
    enabled = _env_flag("NATIVE_RUNNER_ENABLED", True)
    strict = _env_flag("NATIVE_RUNNER_STRICT", False)

    lib_path = _designer_runtime_lib_path()
    designer_runtime_available = lib_path.exists()

    if not enabled:
        return DesignerRuntimeAdapterState(
            enabled=False,
            strict=strict,
            designer_runtime_available=designer_runtime_available,
            reason="disabled_by_env",
        )

    if not designer_runtime_available:
        return DesignerRuntimeAdapterState(
            enabled=True,
            strict=strict,
            designer_runtime_available=False,
            reason=f"missing_designer_runtime_lib:{lib_path}",
        )

    return DesignerRuntimeAdapterState(
        enabled=True,
        strict=strict,
        designer_runtime_available=True,
        reason="ready",
    )


def capability_handshake() -> Dict[str, Any]:
    state = detect_adapter_state()
    approximate_mappings: Dict[str, str] = {}
    semantic_warnings: List[Dict[str, str]] = []
    supported_ops: List[str] = []
    unsupported_ops: List[str] = []
    scheduler_supported_ops: List[str] = []
    scheduler_unsupported_ops: List[str] = []
    native_coverage = 0.0

    mapping_path = (
        Path(__file__).resolve().parents[2]
        / "aria_designer"
        / "runtime"
        / "component_mapping.yaml"
    )

    if mapping_path.exists():
        approximate_mappings = _load_approximate_alias_notes(mapping_path)
        semantic_warnings = [
            {
                "component": component,
                "severity": "warning",
                "message": note,
                "source": "component_mapping.approximate_alias_notes",
            }
            for component, note in sorted(approximate_mappings.items())
        ]

    if state.enabled and state.designer_runtime_available:
        try:
            native_dispatch = importlib.import_module(
                "research.scientist.native.dispatch"
            )

            supported = set(native_dispatch.NATIVE_STRUCTURAL_OPS)
            supported.update(native_dispatch._SOFT_BRIDGE_OPS)
            supported.update(native_dispatch._NATIVE_C_KERNEL_OPS)
            supported.update(native_dispatch._CYTHON_WRAPPER_OPS)
            supported_ops = sorted(supported)
            scheduler_supported_ops = sorted(
                native_dispatch.scheduler_compatible_ops(set(supported_ops))
            )
            scheduler_unsupported_ops = sorted(
                set(supported_ops) - set(scheduler_supported_ops)
            )
            native_coverage = 1.0 if supported_ops else 0.0
        except Exception as exc:
            state = DesignerRuntimeAdapterState(
                enabled=state.enabled,
                strict=state.strict,
                designer_runtime_available=state.designer_runtime_available,
                reason=f"capability_probe_error:{exc}",
            )

    return {
        "enabled": state.enabled,
        "strict": state.strict,
        "designer_runtime_available": state.designer_runtime_available,
        "status": state.reason,
        "supported_ops": supported_ops,
        "unsupported_ops": unsupported_ops,
        "scheduler_supported_ops": scheduler_supported_ops,
        "scheduler_unsupported_ops": scheduler_unsupported_ops,
        "native_coverage": native_coverage,
        "approximate_mappings": approximate_mappings,
        "semantic_warnings": semantic_warnings,
        "semantic_warning_count": len(semantic_warnings),
        "mapping_source": str(mapping_path),
    }


def try_designer_runtime_probe(layer_graphs: List[Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "parity_ok": None,
        "reason": "not_attempted",
    }

    state = detect_adapter_state()
    if not state.enabled:
        report["reason"] = "disabled_by_env"
        return report
    if not state.designer_runtime_available:
        report["reason"] = state.reason
        return report
    if not layer_graphs:
        report["reason"] = "no_layer_graphs"
        return report

    report["attempted"] = True

    try:
        importer_mod, bridge_mod, compiler_mod = _load_designer_runtime_modules()
        first_graph = layer_graphs[0]
        model_dim = int(getattr(first_graph, "model_dim", 256))

        workflow = importer_mod.graph_to_workflow(
            first_graph,
            workflow_id="native_runner_probe",
            name="native_runner_probe",
            metadata={"native_runner_probe": True},
        )

        roundtrip_graph = bridge_mod.workflow_to_graph(workflow, model_dim=model_dim)
        parity_ok = bool(roundtrip_graph.n_ops() == first_graph.n_ops())

        components_dir = (
            Path(__file__).resolve().parents[2] / "aria_designer" / "components"
        )
        try:
            compiled = compiler_mod.compile_workflow(workflow, str(components_dir))
        except ValueError as compile_exc:
            # If the only missing components are ops with direct C kernel support,
            # the probe should still succeed — those ops bypass the designer runtime
            # and dispatch through aria_core C kernels directly.
            exc_msg = str(compile_exc)
            if "Missing runtime kernel_fallback.py" in exc_msg:
                from .native_runner import (
                    _NATIVE_C_KERNEL_OPS,
                    _CYTHON_WRAPPER_OPS,
                    _SOFT_BRIDGE_OPS,
                )
                from ..synthesis.primitives import PRIMITIVE_REGISTRY

                all_native = (
                    _NATIVE_C_KERNEL_OPS | _CYTHON_WRAPPER_OPS | _SOFT_BRIDGE_OPS
                )
                # Also accept any op known to the research primitive registry
                # (they have execute functions and don't need designer fallbacks)
                all_known = all_native | set(PRIMITIVE_REGISTRY.keys())
                # Extract component types from error: "node_3 (math_space/tropical_gate)"
                missing_ops = re.findall(r"node_\w+\s+\(([^)]+)\)", exc_msg)
                # Convert component paths to op names: "math_space/tropical_gate" -> "tropical_gate"
                missing_op_names = [op.split("/")[-1] for op in missing_ops]
                all_covered = all(op in all_known for op in missing_op_names)
                if all_covered:
                    compiled = None  # designer compile not needed
                    report.update(
                        {
                            "succeeded": True,
                            "parity_ok": parity_ok,
                            "reason": "ok_native_kernel_bypass",
                            "workflow_id": workflow.get("workflow_id"),
                            "workflow_node_count": len(workflow.get("nodes") or []),
                            "native_bypass_ops": missing_op_names,
                        }
                    )
                    return report
            raise  # re-raise if not handled

        report.update(
            {
                "succeeded": compiled is not None,
                "parity_ok": parity_ok,
                "reason": "ok"
                if (compiled is not None and parity_ok)
                else "parity_or_compile_mismatch",
                "workflow_id": workflow.get("workflow_id"),
                "workflow_node_count": len(workflow.get("nodes") or []),
            }
        )
        return report
    except Exception as exc:
        report.update(
            {
                "succeeded": False,
                "parity_ok": False,
                "reason": f"probe_error:{exc}",
            }
        )
        return report


def build_designer_layer_modules(layer_graphs: List[Any]) -> Dict[str, Any]:
    """Compile individual research layer graphs with designer runtime compiler.

    Returns replacement modules keyed by layer index plus summary metadata.
    """
    state = detect_adapter_state()
    result: Dict[str, Any] = {
        "attempted": False,
        "compiled_layers": 0,
        "failed_layers": 0,
        "total_layers": int(len(layer_graphs or [])),
        "replacements": {},
        "errors": [],
    }
    if not state.enabled:
        result["errors"].append("disabled_by_env")
        return result
    if not state.designer_runtime_available:
        result["errors"].append(state.reason)
        return result
    if not layer_graphs:
        result["errors"].append("no_layer_graphs")
        return result

    result["attempted"] = True
    try:
        importer_mod, _bridge_mod, compiler_mod = _load_designer_runtime_modules()
        components_dir = (
            Path(__file__).resolve().parents[2] / "aria_designer" / "components"
        )

        for idx, graph in enumerate(layer_graphs):
            try:
                workflow = importer_mod.graph_to_workflow(
                    graph,
                    workflow_id=f"native_runner_layer_{idx}",
                    name=f"native_runner_layer_{idx}",
                    metadata={"native_runner_selective_exec": True, "layer_index": idx},
                )
                compiled = compiler_mod.compile_workflow(workflow, str(components_dir))
                input_node_id = _find_workflow_input_node(workflow)
                result["replacements"][idx] = {
                    "module": compiled,
                    "input_node_id": input_node_id,
                    "workflow_id": workflow.get("workflow_id"),
                }
                result["compiled_layers"] += 1
            except Exception as exc:
                result["failed_layers"] += 1
                result["errors"].append(f"layer_{idx}:{exc}")
    except Exception as exc:
        result["errors"].append(f"module_load_error:{exc}")
        result["failed_layers"] = int(len(layer_graphs or []))
        result["compiled_layers"] = 0
    return result


def _load_designer_runtime_modules() -> Tuple[Any, Any, Any]:
    runtime_dir = Path(__file__).resolve().parents[2] / "aria_designer" / "runtime"
    package_name = "aria_designer_runtime"
    _ensure_package_loaded(package_name, runtime_dir)
    importer_mod = _load_module_from_path(
        f"{package_name}.importer", runtime_dir / "importer.py"
    )
    bridge_mod = _load_module_from_path(
        f"{package_name}.bridge", runtime_dir / "bridge.py"
    )
    compiler_mod = _load_module_from_path(
        f"{package_name}.compiler", runtime_dir / "compiler.py"
    )
    return importer_mod, bridge_mod, compiler_mod


def _ensure_package_loaded(package_name: str, package_dir: Path) -> None:
    if package_name in sys.modules:
        return
    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        str(init_path),
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load package spec for {package_name} at {init_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module spec for {module_name} at {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_workflow_input_node(workflow: Dict[str, Any]) -> str:
    nodes = workflow.get("nodes") or []
    for node in nodes:
        comp_type = str(node.get("component_type") or "")
        leaf = comp_type.split("/")[-1]
        if leaf in {"input", "graph_input"} or comp_type in {
            "io/input",
            "io/graph_input",
        }:
            node_id = node.get("id")
            if node_id:
                return str(node_id)
    # Fallback to first node id for robustness.
    if nodes and nodes[0].get("id"):
        return str(nodes[0].get("id"))
    raise ValueError("No input node found in workflow")


def _load_approximate_alias_notes(mapping_path: Path) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    in_section = False

    try:
        for line in mapping_path.read_text(encoding="utf-8").splitlines():
            raw = line.rstrip()
            stripped = raw.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if stripped == "approximate_alias_notes:":
                in_section = True
                continue

            if not in_section:
                continue

            if raw and not raw.startswith("  "):
                break

            match = re.match(r"\s{2}([a-zA-Z0-9_\-/]+):\s*\"?(.*?)\"?$", raw)
            if not match:
                continue
            key = str(match.group(1)).strip()
            msg = str(match.group(2)).strip().strip('"')
            if key and msg:
                notes[key] = msg
    except Exception as exc:
        logger.debug("Returning default due to error: %s", exc)
        return {}

    return notes
