from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .abi import (
    _build_native_abi_only_model,
    _maybe_prepare_runner_abi_session,
    _try_load_native_lib,
)
from .autograd import NativeForwardWrapper, SubgraphDispatcher
from .core import (
    PARTIAL_NATIVE_COVERAGE_THRESHOLD,
    _FALLBACK_METRICS,
    _SELECTIVE_GUARDRAIL,
    _SELECTIVE_GUARDRAIL_HISTORY,
    _env_flag,
    detect_native_state,
)
from .dispatch import (
    _activate_selective_native_dispatch,
    _check_native_op_support,
    _requested_execution_mode,
)
from .designer import (
    DesignerWorkflowLayerAdapter,
    _summarize_layer_build,
    _validate_designer_layer_adapter_contract,
)
from .guardrails import (
    _maybe_fail_on_fallback_rate,
    _maybe_fail_on_legacy_compile_usage,
    _maybe_warn_deprecated_legacy_only_flag,
    _record_guardrail_event,
)
from .telemetry import (
    _log_native_fallback_coverage,
    _record_legacy_compile_invocation,
    native_runner_capability_report,
)
from ..native_runner_adapter import try_designer_runtime_probe
from ..native_runner_adapter import build_designer_layer_modules
from ...defaults import VOCAB_SIZE

logger = logging.getLogger(__name__)


def _finalize_native_abi_model(
    *,
    abi_session: Any,
    capability: Dict[str, Any],
    vocab_size: int,
    kwargs: Dict[str, Any],
):
    model = _build_native_abi_only_model(
        abi_session=abi_session,
        vocab_size=int(vocab_size),
        model_dim=int(kwargs.get("model_dim", 0) or 0),
    )
    setattr(model, "_native_runner_abi_session", abi_session)
    capability["runner_abi"]["session_attached"] = True
    capability["execution_path"] = "native_abi_model_only"
    capability["legacy_compile_used"] = False
    _maybe_fail_on_fallback_rate()
    _maybe_fail_on_legacy_compile_usage()
    capability["fallback_metrics"] = native_runner_capability_report().get(
        "fallback_metrics", {}
    )
    try:
        setattr(model, "_native_runner_report", capability)
    except (AttributeError, TypeError):
        pass
    logger.info(
        "Native runner ABI-only model active: execution_path=%s enabled=%s strict=%s",
        capability.get("execution_path"),
        capability.get("enabled"),
        capability.get("strict"),
    )
    return model


def _native_coverage_ready(
    *,
    state: Any,
    op_support: Optional[Dict[str, Any]],
) -> bool:
    return bool(
        state.enabled
        and op_support is not None
        and op_support["native_coverage"] >= PARTIAL_NATIVE_COVERAGE_THRESHOLD
    )


def _attach_native_forward_wrapper(
    *,
    model: Any,
    capability: Dict[str, Any],
    supported_ops: set[str],
) -> None:
    try:
        wrapper = NativeForwardWrapper(model, supported_ops)
        setattr(model, "_native_forward_wrapper", wrapper)
        wrapper_report = {
            "attached": True,
            "supported_ops": len(supported_ops),
        }
        try:
            for layer in getattr(model, "layers", []):
                ops = getattr(layer, "ops", None)
                if ops is None:
                    continue
                for op in ops.values():
                    if hasattr(op, "forward"):
                        op._native_wrapper = wrapper
            wrapper_report["propagated"] = True
        except Exception as exc:
            logger.debug("Failed to propagate native wrapper to ops: %s", exc)
            wrapper_report["propagated"] = False
        capability["native_forward_wrapper"] = wrapper_report
    except Exception as exc:
        logger.debug("Failed to attach native forward wrapper: %s", exc)
        capability["native_forward_wrapper"] = {
            "attached": False,
            "error": str(exc),
        }


def _attach_subgraph_dispatchers(
    *,
    model: Any,
    layer_graphs: List[Any],
    capability: Dict[str, Any],
    supported_ops: set[str],
) -> None:
    try:
        dispatchers_attached = 0
        dispatchers_skipped = 0
        layers = getattr(model, "layers", [])
        layer_graph_list = getattr(model, "_layer_graphs", layer_graphs)
        for i, layer in enumerate(layers):
            if i >= len(layer_graph_list):
                continue
            graph = layer_graph_list[i]
            if not hasattr(graph, "nodes"):
                continue
            dispatcher = _build_layer_subgraph_dispatcher(
                layer=layer,
                graph=graph,
                supported_ops=supported_ops,
            )
            if dispatcher.all_native:
                layer._subgraph_dispatcher = dispatcher
                dispatchers_attached += 1
            else:
                dispatchers_skipped += 1
        capability["subgraph_dispatch"] = {
            "attached": dispatchers_attached,
            "skipped": dispatchers_skipped,
            "total_layers": len(layers),
        }
        if dispatchers_attached > 0:
            logger.info(
                "Subgraph dispatch: attached %d/%d layer dispatchers",
                dispatchers_attached,
                len(layers),
            )
    except Exception as exc:
        logger.debug("Failed to attach subgraph dispatchers: %s", exc)
        capability["subgraph_dispatch"] = {
            "attached": 0,
            "error": str(exc),
        }


def _build_layer_subgraph_dispatcher(
    *,
    layer: Any,
    graph: Any,
    supported_ops: set[str],
) -> Any:
    try:
        from ...synthesis.native_bound_graph import BoundNativeSubgraphDispatcher
        from ...synthesis.native_compile import _bound_dispatcher_inputs_from_layer
        from ...synthesis.native_support import graph_has_bound_params

        if graph_has_bound_params(graph):
            flat_ops, ir_node_ids = _bound_dispatcher_inputs_from_layer(layer, graph)
            return BoundNativeSubgraphDispatcher(
                graph,
                flat_ops=flat_ops,
                ir_node_ids=ir_node_ids,
                supported_ops=supported_ops,
            )
    except Exception as exc:
        logger.debug("Failed to build bound subgraph dispatcher: %s", exc)

    return SubgraphDispatcher(graph, supported_ops)


def _init_layer_build_report(layer_build: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "attempted": bool(layer_build.get("attempted")),
        "compiled_layers": int(layer_build.get("compiled_layers") or 0),
        "failed_layers": int(layer_build.get("failed_layers") or 0),
        "total_layers": int(layer_build.get("total_layers") or 0),
        "errors": list(layer_build.get("errors") or []),
        "skip_reasons": [],
        "skipped_layers": 0,
        "applied_layers": 0,
        "layer_results": [],
    }


def _append_layer_skip(
    layer_build_report: Dict[str, Any],
    layer_result: Dict[str, Any],
    *,
    layer_idx: Any,
    reason: str,
) -> None:
    layer_build_report["skipped_layers"] += 1
    layer_build_report["skip_reasons"].append(f"skip_layer_{layer_idx}:{reason}")
    layer_result["skip_reason"] = reason
    layer_build_report["layer_results"].append(layer_result)


def _apply_selective_designer_layers(
    *,
    model: Any,
    layer_graphs: List[Any],
    capability: Dict[str, Any],
    max_seq_len: Optional[int],
    selective_layer_strict: bool,
) -> None:
    layer_build = build_designer_layer_modules(layer_graphs)
    layer_build_report = _init_layer_build_report(layer_build)
    capability["selective_execution"]["layer_build"] = layer_build_report
    replacements = layer_build.get("replacements") or {}
    model_dim = int(getattr(model, "model_dim", 0) or 0)
    if not hasattr(model, "layers"):
        layer_build_report["errors"].append("model_has_no_layers")
        layer_build_report["summary"] = _summarize_layer_build(layer_build_report)
        return

    total_layers = len(model.layers)
    for idx, payload in replacements.items():
        layer_result: Dict[str, Any] = {
            "layer_index": idx,
            "workflow_id": payload.get("workflow_id"),
            "input_node_id": payload.get("input_node_id"),
            "applied": False,
            "skip_reason": None,
            "error": None,
        }
        try:
            try:
                layer_idx = int(idx)
                layer_result["layer_index"] = layer_idx
            except (TypeError, ValueError):
                _append_layer_skip(
                    layer_build_report,
                    layer_result,
                    layer_idx=idx,
                    reason="invalid_layer_index",
                )
                continue
            if layer_idx < 0 or layer_idx >= total_layers:
                _append_layer_skip(
                    layer_build_report,
                    layer_result,
                    layer_idx=layer_idx,
                    reason="layer_index_out_of_range",
                )
                continue
            wm = payload.get("module")
            in_id = str(payload.get("input_node_id") or "")
            if wm is None:
                _append_layer_skip(
                    layer_build_report,
                    layer_result,
                    layer_idx=layer_idx,
                    reason="missing_workflow_module",
                )
                continue
            if not in_id:
                _append_layer_skip(
                    layer_build_report,
                    layer_result,
                    layer_idx=layer_idx,
                    reason="missing_input_node_id",
                )
                continue
            adapter = DesignerWorkflowLayerAdapter(wm, in_id).as_module()
            contract_error = _validate_designer_layer_adapter_contract(
                adapter,
                model_dim=model_dim,
                max_seq_len=max_seq_len,
            )
            if contract_error:
                _append_layer_skip(
                    layer_build_report,
                    layer_result,
                    layer_idx=layer_idx,
                    reason=contract_error,
                )
                continue
            model.layers[layer_idx] = adapter
            layer_build_report["applied_layers"] += 1
            layer_result["applied"] = True
            layer_build_report["layer_results"].append(layer_result)
        except Exception as exc:
            layer_result["error"] = str(exc)
            layer_build_report["errors"].append(f"apply_layer_{idx}:{exc}")
            layer_build_report["layer_results"].append(layer_result)

    if layer_build_report["applied_layers"] > 0:
        capability["execution_path"] = "selective_designer_layers_active"
    layer_build_report["summary"] = _summarize_layer_build(layer_build_report)
    if not selective_layer_strict:
        return
    failed_layers = [
        item
        for item in layer_build_report["layer_results"]
        if not bool(item.get("applied"))
    ]
    if failed_layers:
        layer_build_report["summary"]["strict_failed"] = True
        details = ", ".join(
            f"layer={item.get('layer_index')} reason={item.get('skip_reason') or item.get('error') or 'unknown'}"
            for item in failed_layers[:3]
        )
        raise RuntimeError(
            f"Selective layer strict mode rejected incompatible replacements: {details}"
        )


def _legacy_compile_model(
    layer_graphs: List[Any],
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: Optional[int] = None,
    **kwargs: Any,
):
    # Lazy import keeps adapter unit tests independent of heavyweight runtime deps.
    from ...synthesis.compiler import compile_model as _compile_model

    return _compile_model(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )


def _check_legacy_only_rollback(
    layer_graphs: List[Any],
    vocab_size: int,
    max_seq_len: Optional[int],
    disable_legacy_compile: bool,
    **kwargs: Any,
) -> Optional[Any]:
    """Check NATIVE_RUNNER_LEGACY_ONLY env flag and return a legacy model or None."""
    if not _env_flag("NATIVE_RUNNER_LEGACY_ONLY", False):
        return None
    _maybe_warn_deprecated_legacy_only_flag()
    state_check = detect_native_state()
    if state_check.enabled:
        raise RuntimeError(
            "Invalid native-runner config: "
            "NATIVE_RUNNER_LEGACY_ONLY=1 cannot be used when NATIVE_RUNNER_ENABLED=1. "
            "Disable native mode first (NATIVE_RUNNER_ENABLED=0) to use legacy-only path."
        )
    if disable_legacy_compile:
        raise RuntimeError(
            "Invalid native-runner config: "
            "NATIVE_RUNNER_LEGACY_ONLY=1 conflicts with NATIVE_RUNNER_DISABLE_LEGACY_COMPILE=1"
        )
    _FALLBACK_METRICS["total_compiles"] += 1
    _record_legacy_compile_invocation()
    return _legacy_compile_model(
        layer_graphs, vocab_size=vocab_size, max_seq_len=max_seq_len, **kwargs
    )


def _prepare_native_dispatch(
    *,
    layer_graphs: List[Any],
    state: Any,
    capability: Dict[str, Any],
    vocab_size: int,
    max_seq_len: Optional[int],
    kwargs: Dict[str, Any],
) -> tuple[
    Optional[Dict[str, Any]],  # op_support
    Any,  # native_lib
    bool,  # full_native_coverage
    bool,  # partial_native_coverage
    Dict[str, Any],  # abi_report
    Optional[Any],  # early-return model (if ABI-only)
]:
    """Load native lib, prep ABI session, check op support, classify coverage."""
    native_lib = _try_load_native_lib()
    _FALLBACK_METRICS["total_compiles"] += 1
    abi_report = _maybe_prepare_runner_abi_session(
        layer_graphs=layer_graphs,
        native_lib=native_lib,
        state=state,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    capability["runner_abi"] = {
        "requested": bool(abi_report.get("requested")),
        "attempted": bool(abi_report.get("attempted")),
        "succeeded": bool(abi_report.get("succeeded")),
        "reason": abi_report.get("reason"),
        "model_handle": abi_report.get("model_handle"),
    }
    if (
        state.strict
        and bool(abi_report.get("requested"))
        and not bool(abi_report.get("succeeded"))
    ):
        raise RuntimeError(
            f"NATIVE_RUNNER_STRICT=1 and runner ABI prepare failed: {abi_report.get('reason')}"
        )
    capability["runner_abi"]["model_only_requested"] = True
    capability["execution_mode_classification"] = "native_abi_model_only"
    if bool(abi_report.get("succeeded")) and abi_report.get("session") is not None:
        model = _finalize_native_abi_model(
            abi_session=abi_report["session"],
            capability=capability,
            vocab_size=vocab_size,
            kwargs=kwargs,
        )
        return None, native_lib, False, False, abi_report, model

    op_support: Optional[Dict[str, Any]] = None
    full_native_coverage = False
    partial_native_coverage = False
    if layer_graphs:
        op_support = _check_native_op_support(layer_graphs, native_lib)
        capability["native_op_support"] = op_support

        coverage = op_support["native_coverage"]
        if coverage >= 1.0:
            full_native_coverage = True
            logger.debug(
                "Full native path available: all %d ops supported by kernel library",
                len(op_support["all_ops"]),
            )
            _FALLBACK_METRICS["native_dispatch_compiles"] += 1
        elif state.strict:
            raise RuntimeError(
                f"NATIVE_RUNNER_STRICT=1 but {len(op_support['unsupported'])} ops lack "
                f"native kernel support: {op_support['unsupported']}"
            )
        elif coverage >= PARTIAL_NATIVE_COVERAGE_THRESHOLD:
            partial_native_coverage = True
            logger.debug(
                "Partial native path: %.1f%% coverage (%d/%d ops). Unsupported: %s",
                coverage * 100,
                len(op_support["supported"]),
                len(op_support["kernel_relevant_ops"]),
                op_support["unsupported"],
            )
            _FALLBACK_METRICS["native_dispatch_compiles"] += 1
        else:
            _log_native_fallback_coverage(op_support)
    elif state.strict:
        # Strict with empty graphs: nothing to check, allow through.
        pass

    return (
        op_support,
        native_lib,
        full_native_coverage,
        partial_native_coverage,
        abi_report,
        None,
    )


def _validate_ir_observational(
    layer_graphs: List[Any],
    capability: Dict[str, Any],
) -> None:
    """Convert graph to native IR and validate (informational only)."""
    try:
        from ..runtime.native.ir_validator import validate_ir
        from ..synthesis.native_ir_converter import graph_to_native_ir

        ir_errors: List = []
        for i, g in enumerate(layer_graphs):
            if not hasattr(g, "nodes"):
                continue  # skip non-ComputationGraph objects
            ir_doc = graph_to_native_ir(g)
            errs = validate_ir(ir_doc)
            if errs:
                ir_errors.extend([(i, e) for e in errs])
        if ir_errors:
            logger.warning("IR validation errors: %s", ir_errors[:5])
            capability["ir_validation"] = {
                "valid": False,
                "errors": [str(e) for e in ir_errors[:10]],
            }
        else:
            capability["ir_validation"] = {"valid": True, "errors": []}
    except Exception as exc:
        logger.debug("IR validation skipped: %s", exc)
        capability["ir_validation"] = {
            "valid": None,
            "errors": [f"validation_unavailable:{exc}"],
        }


def _compute_selective_candidate(
    *,
    requested_mode: str,
    state: Any,
    full_native_coverage: bool,
    probe: Dict[str, Any],
) -> tuple[bool, str]:
    """Determine if selective execution is a candidate."""
    selective_candidate = False
    selective_reason = "mode_not_requested"
    if requested_mode == "selective":
        if not state.enabled:
            selective_reason = "native_runner_disabled"
        elif not full_native_coverage:
            selective_reason = "incomplete_native_coverage"
        elif state.designer_runtime_available and not (
            bool(probe.get("succeeded")) and bool(probe.get("parity_ok"))
        ):
            selective_reason = "probe_not_green"
        else:
            selective_candidate = True
            selective_reason = "candidate_ready"
            _FALLBACK_METRICS["selective_mode_candidates"] += 1
    return selective_candidate, selective_reason


def _update_selective_guardrail(
    *,
    requested_mode: str,
    selective_candidate: bool,
    selective_reason: str,
    capability: Dict[str, Any],
) -> None:
    """Threshold-based guardrail state machine for selective execution."""
    try:
        guardrail_threshold = int(
            str(os.environ.get("NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW", "5"))
        )
    except (TypeError, ValueError):
        guardrail_threshold = 5
    guardrail_threshold = max(1, guardrail_threshold)

    if requested_mode == "selective" and not selective_candidate:
        _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = (
            int(_SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0)
            + 1
        )
        _SELECTIVE_GUARDRAIL["last_reason"] = selective_reason
        if (
            _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"]
            >= guardrail_threshold
        ):
            if not bool(_SELECTIVE_GUARDRAIL.get("triggered")):
                _SELECTIVE_GUARDRAIL["trigger_count"] = (
                    int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0) + 1
                )
                _record_guardrail_event(
                    "triggered",
                    reason=selective_reason,
                    threshold=guardrail_threshold,
                    source="compile_model_native_first",
                )
            _SELECTIVE_GUARDRAIL["triggered"] = True
    else:
        was_triggered = bool(_SELECTIVE_GUARDRAIL.get("triggered"))
        _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = 0
        _SELECTIVE_GUARDRAIL["triggered"] = False
        if was_triggered:
            cleared_reason = (
                "candidate_ready"
                if requested_mode == "selective"
                else "mode_not_selective"
            )
            _record_guardrail_event(
                "cleared",
                reason=cleared_reason,
                threshold=guardrail_threshold,
                source="compile_model_native_first",
            )
        if requested_mode != "selective":
            _SELECTIVE_GUARDRAIL["last_reason"] = None

    capability["selective_guardrail"] = {
        "consecutive_requested_not_candidate": int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ),
        "threshold": guardrail_threshold,
        "triggered": bool(_SELECTIVE_GUARDRAIL.get("triggered")),
        "trigger_count": int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0),
        "last_reason": _SELECTIVE_GUARDRAIL.get("last_reason"),
        "history": [dict(item) for item in _SELECTIVE_GUARDRAIL_HISTORY],
    }


def _run_designer_runtime_probe(
    state: Any,
    layer_graphs: List[Any],
    capability: Dict[str, Any],
) -> Dict[str, Any]:
    """Designer runtime probe (orthogonal to native kernel dispatch)."""
    probe: Dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "parity_ok": None,
        "reason": "not_attempted",
    }
    if state.enabled and state.designer_runtime_available:
        capability["designer_runtime_probe"] = try_designer_runtime_probe(layer_graphs)
        probe = capability["designer_runtime_probe"]
        if bool(probe.get("succeeded")) and bool(probe.get("parity_ok")):
            _FALLBACK_METRICS["probe_successes"] += 1
        else:
            _FALLBACK_METRICS["probe_failures"] += 1
    return probe


def _activate_selective_dispatch(
    selective_candidate: bool,
    native_lib: Any,
    capability: Dict[str, Any],
) -> Dict[str, Any]:
    """Activate selective native dispatch if candidate, update metrics."""
    selective_activation: Dict[str, Any] = {
        "activated": False,
        "ops": ["relu", "add"],
        "reason": "not_candidate",
    }
    if selective_candidate:
        selective_activation = _activate_selective_native_dispatch(native_lib)
        if bool(selective_activation.get("activated")):
            _FALLBACK_METRICS["selective_mode_activations"] += 1
        else:
            _FALLBACK_METRICS["selective_mode_activation_failures"] += 1
    capability["selective_execution"]["activation"] = selective_activation
    return selective_activation


def _classify_execution_path(
    *,
    state: Any,
    capability: Dict[str, Any],
    op_support: Optional[Dict[str, Any]],
    selective_candidate: bool,
    selective_activation: Dict[str, Any],
    full_native_coverage: bool,
    partial_native_coverage: bool,
) -> None:
    """Set final execution path based on coverage + selective status, record fallback metrics."""
    if selective_candidate:
        if bool(selective_activation.get("activated")):
            capability["execution_path"] = "selective_native_active_legacy_compile"
        else:
            capability["execution_path"] = "selective_candidate_legacy_compile"
    elif full_native_coverage:
        capability["execution_path"] = "full_native_legacy_compile"
    elif partial_native_coverage:
        capability["execution_path"] = "hybrid_native_legacy_compile"
    elif state.enabled:
        capability["execution_path"] = "legacy_fallback"

    # --- Compile via legacy path (actual native execution dispatch is a future phase) ---
    if not state.enabled:
        _FALLBACK_METRICS["total_compiles"] += 1
    if state.enabled and (
        op_support is None
        or op_support["native_coverage"] < PARTIAL_NATIVE_COVERAGE_THRESHOLD
    ):
        _FALLBACK_METRICS["fallback_compiles"] += 1
    if state.enabled and partial_native_coverage:
        _FALLBACK_METRICS["hybrid_compiles"] += 1


def _attach_abi_session_to_model(
    *,
    model: Any,
    abi_report: Dict[str, Any],
    capability: Dict[str, Any],
) -> None:
    """Attach ABI session to legacy-compiled model if available."""
    abi_session = abi_report.get("session")
    if abi_session is None:
        return
    try:
        setattr(model, "_native_runner_abi_session", abi_session)
        capability["runner_abi"]["session_attached"] = True
        if capability.get("execution_path") == "selective_native_active_legacy_compile":
            capability["execution_path"] = "selective_native_abi_ready_legacy_compile"
    except Exception as exc:
        logger.debug("Failed to attach runner ABI session: %s", exc)
        capability["runner_abi"]["session_attached"] = False
        capability["runner_abi"]["attach_error"] = str(exc)
        try:
            abi_session.close()
        except (OSError, RuntimeError) as exc:
            logger.debug("Suppressed error: %s", exc)


def _apply_selective_layers_if_enabled(
    *,
    model: Any,
    layer_graphs: List[Any],
    capability: Dict[str, Any],
    max_seq_len: Optional[int],
    selective_candidate: bool,
    selective_activation: Dict[str, Any],
) -> None:
    """Apply selective Designer layer replacements if all prerequisites met."""
    selective_layer_exec_enabled = _env_flag(
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC", False
    )
    selective_layer_strict = _env_flag("NATIVE_RUNNER_SELECTIVE_LAYER_STRICT", False)
    capability["selective_execution"]["layer_exec_enabled"] = (
        selective_layer_exec_enabled
    )
    capability["selective_execution"]["layer_exec_strict"] = selective_layer_strict
    if (
        selective_candidate
        and bool(selective_activation.get("activated"))
        and selective_layer_exec_enabled
    ):
        _apply_selective_designer_layers(
            model=model,
            layer_graphs=layer_graphs,
            capability=capability,
            max_seq_len=max_seq_len,
            selective_layer_strict=selective_layer_strict,
        )
    elif selective_candidate and bool(selective_activation.get("activated")):
        capability["selective_execution"]["layer_build"] = {
            "attempted": False,
            "compiled_layers": 0,
            "failed_layers": 0,
            "total_layers": int(len(layer_graphs or [])),
            "errors": ["layer_exec_disabled_by_env"],
            "skip_reasons": [],
            "skipped_layers": 0,
            "applied_layers": 0,
            "layer_results": [],
        }
        capability["selective_execution"]["layer_build"]["summary"] = (
            _summarize_layer_build(capability["selective_execution"]["layer_build"])
        )


def _compile_legacy_and_attach(
    *,
    state: Any,
    capability: Dict[str, Any],
    layer_graphs: List[Any],
    vocab_size: int,
    max_seq_len: Optional[int],
    kwargs: Dict[str, Any],
    disable_legacy_compile: bool,
    abi_report: Dict[str, Any],
    op_support: Optional[Dict[str, Any]],
    selective_candidate: bool,
    selective_activation: Dict[str, Any],
) -> Any:
    """Legacy compile fallback, attach native wrappers/dispatchers, ABI session, selective layers."""
    if not state.enabled:
        capability["runner_abi"] = {
            "requested": False,
            "attempted": False,
            "succeeded": False,
            "reason": "disabled",
            "model_handle": None,
            "model_only_requested": False,
        }
        capability["execution_mode_classification"] = "legacy_only"

    if state.enabled:
        abi_session = abi_report.get("session")
        if abi_session is None or not bool(abi_report.get("succeeded")):
            # ABI session not available. During the transition, keep the
            # legacy compile path as the safety net unless explicitly disabled.
            if not disable_legacy_compile:
                logger.debug(
                    "ABI session unavailable (reason=%s), falling back to legacy compile",
                    abi_report.get("reason"),
                )
            else:
                raise RuntimeError(
                    "Native mode requires successful ABI session preparation. "
                    f"reason={abi_report.get('reason')}"
                )
        else:
            return _finalize_native_abi_model(
                abi_session=abi_session,
                capability=capability,
                vocab_size=vocab_size,
                kwargs=kwargs,
            )

    if disable_legacy_compile:
        raise RuntimeError(
            "Legacy compile path disabled by NATIVE_RUNNER_DISABLE_LEGACY_COMPILE=1 "
            "or NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1; "
            "remove remaining legacy fallback usage before enabling this gate."
        )

    model = _legacy_compile_model(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )
    _record_legacy_compile_invocation()
    capability["legacy_compile_used"] = True

    if _native_coverage_ready(state=state, op_support=op_support):
        supported_set = set(op_support["supported"])
        _attach_native_forward_wrapper(
            model=model,
            capability=capability,
            supported_ops=supported_set,
        )
        _attach_subgraph_dispatchers(
            model=model,
            layer_graphs=layer_graphs,
            capability=capability,
            supported_ops=supported_set,
        )

    _attach_abi_session_to_model(
        model=model,
        abi_report=abi_report,
        capability=capability,
    )
    _apply_selective_layers_if_enabled(
        model=model,
        layer_graphs=layer_graphs,
        capability=capability,
        max_seq_len=max_seq_len,
        selective_candidate=selective_candidate,
        selective_activation=selective_activation,
    )
    return model


def _finalize_capability_report(
    model: Any,
    capability: Dict[str, Any],
    native_lib: Any,
) -> None:
    """Guardrail checks, fallback rate validation, attach report."""
    _maybe_fail_on_fallback_rate()
    _maybe_fail_on_legacy_compile_usage()
    # Refresh fallback telemetry after current compile increments.
    capability["fallback_metrics"] = native_runner_capability_report().get(
        "fallback_metrics", {}
    )

    # Attach diagnostic metadata for observability.
    try:
        setattr(model, "_native_runner_report", capability)
    except (AttributeError, TypeError):
        pass

    warning_count = int(capability.get("semantic_warning_count") or 0)
    if warning_count > 0:
        logger.debug(
            "Native runner capability reports %d semantic mapping warnings (status=%s)",
            warning_count,
            capability.get("status"),
        )
    else:
        logger.debug(
            "Native runner capability status=%s enabled=%s strict=%s native_lib=%s",
            capability.get("status"),
            capability.get("enabled"),
            capability.get("strict"),
            "loaded" if native_lib is not None else "unavailable",
        )


def compile_model_native_first(
    layer_graphs: List[Any],
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: Optional[int] = None,
    **kwargs: Any,
):
    """Compile model using native-first policy (Phase-D transition state).

    Attaches ``_native_runner_report`` capability metadata to the compiled model.
    """
    disable_legacy = _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE", False)
    if _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED", False):
        disable_legacy_native_gate = True
    else:
        disable_legacy_native_gate = False

    legacy_model = _check_legacy_only_rollback(
        layer_graphs, vocab_size, max_seq_len, disable_legacy, **kwargs
    )
    if legacy_model is not None:
        return legacy_model

    state = detect_native_state()
    capability = native_runner_capability_report()
    requested_mode = _requested_execution_mode()
    capability["execution_mode_requested"] = requested_mode
    capability["execution_path"] = "legacy_disabled"

    if state.enabled and disable_legacy_native_gate:
        disable_legacy = True
        capability["legacy_compile_disabled_reason"] = "native_enabled_gate"
    if state.enabled:
        _FALLBACK_METRICS["native_enabled_compiles"] += 1

    # --- Phase 3: native kernel dispatch checking ---
    op_support: Optional[Dict[str, Any]] = None
    native_lib = None
    full_native = False
    partial_native = False
    abi_report: Dict[str, Any] = {
        "requested": False,
        "attempted": False,
        "succeeded": False,
        "reason": "disabled",
        "model_handle": None,
        "session": None,
    }
    if state.enabled:
        (
            op_support,
            native_lib,
            full_native,
            partial_native,
            abi_report,
            early_model,
        ) = _prepare_native_dispatch(
            layer_graphs=layer_graphs,
            state=state,
            capability=capability,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            kwargs=kwargs,
        )
        if early_model is not None:
            return early_model

    if state.enabled and layer_graphs:
        _validate_ir_observational(layer_graphs, capability)

    probe = _run_designer_runtime_probe(state, layer_graphs, capability)
    selective_candidate, selective_reason = _compute_selective_candidate(
        requested_mode=requested_mode,
        state=state,
        full_native_coverage=full_native,
        probe=probe,
    )
    capability["selective_execution"] = {
        "requested": requested_mode == "selective",
        "candidate": selective_candidate,
        "reason": selective_reason,
    }
    _update_selective_guardrail(
        requested_mode=requested_mode,
        selective_candidate=selective_candidate,
        selective_reason=selective_reason,
        capability=capability,
    )
    selective_activation = _activate_selective_dispatch(
        selective_candidate, native_lib, capability
    )
    _classify_execution_path(
        state=state,
        capability=capability,
        op_support=op_support,
        selective_candidate=selective_candidate,
        selective_activation=selective_activation,
        full_native_coverage=full_native,
        partial_native_coverage=partial_native,
    )

    model = _compile_legacy_and_attach(
        state=state,
        capability=capability,
        layer_graphs=layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        kwargs=kwargs,
        disable_legacy_compile=disable_legacy,
        abi_report=abi_report,
        op_support=op_support,
        selective_candidate=selective_candidate,
        selective_activation=selective_activation,
    )

    _finalize_capability_report(model, capability, native_lib)
    return model
