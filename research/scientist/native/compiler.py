from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .abi import (
    _build_native_abi_only_model,
    _maybe_prepare_runner_abi_session,
    _try_load_native_lib,
)
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
    _maybe_enforce_fallback_guardrails,
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

logger = logging.getLogger(__name__)

def _legacy_compile_model(
    layer_graphs: List[Any],
    vocab_size: int = 32000,
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

def compile_model_native_first(
    layer_graphs: List[Any],
    vocab_size: int = 32000,
    max_seq_len: Optional[int] = None,
    **kwargs: Any,
):
    """Compile model using native-first policy.

    Phase-D behavior (post-cutover):
    - If disabled (NATIVE_RUNNER_ENABLED=0): legacy compile via ``_legacy_compile_model``.
    - If enabled: ABI model-only path — builds model backed by native runner ABI session.
      Legacy compile is unreachable from native-enabled flow.
    - Always attaches an ``_native_runner_report`` to the compiled model.
    """
    disable_legacy_compile = _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE", False)
    disable_legacy_compile_native_enabled = _env_flag(
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED",
        False,
    )

    # Emergency rollback: skip all native logic (only valid when native is NOT enabled).
    # Phase D: NATIVE_RUNNER_LEGACY_ONLY conflicts with NATIVE_RUNNER_ENABLED.
    if _env_flag("NATIVE_RUNNER_LEGACY_ONLY", False):
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
        return _legacy_compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=max_seq_len, **kwargs)

    state = detect_native_state()
    capability = native_runner_capability_report()
    requested_mode = _requested_execution_mode()
    capability["execution_mode_requested"] = requested_mode
    capability["execution_path"] = "legacy_disabled"

    if state.enabled and disable_legacy_compile_native_enabled:
        disable_legacy_compile = True
        capability["legacy_compile_disabled_reason"] = "native_enabled_gate"

    if state.enabled:
        _FALLBACK_METRICS["native_enabled_compiles"] += 1

    # --- Phase 3: native kernel dispatch checking ---
    op_support: Optional[Dict[str, Any]] = None
    native_lib = None
    full_native_coverage = False
    partial_native_coverage = False
    if state.enabled:
        native_lib = _try_load_native_lib()
        if layer_graphs:
            op_support = _check_native_op_support(layer_graphs, native_lib)
            capability["native_op_support"] = op_support

            coverage = op_support["native_coverage"]
            if coverage >= 1.0:
                full_native_coverage = True
                logger.info(
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
                    "Partial native path: %.1f%% coverage (%d/%d ops). "
                    "Unsupported: %s",
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

    # --- IR validation (observational — log warnings but don't block) ---
    if state.enabled and layer_graphs:
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

    # --- Designer runtime probe (orthogonal to native kernel dispatch) ---
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

    capability["selective_execution"] = {
        "requested": requested_mode == "selective",
        "candidate": selective_candidate,
        "reason": selective_reason,
    }

    try:
        guardrail_threshold = int(str(os.environ.get("NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW", "5")))
    except Exception:
        guardrail_threshold = 5
    guardrail_threshold = max(1, guardrail_threshold)

    if requested_mode == "selective" and not selective_candidate:
        _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ) + 1
        _SELECTIVE_GUARDRAIL["last_reason"] = selective_reason
        if _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] >= guardrail_threshold:
            if not bool(_SELECTIVE_GUARDRAIL.get("triggered")):
                _SELECTIVE_GUARDRAIL["trigger_count"] = int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0) + 1
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
            cleared_reason = "candidate_ready" if requested_mode == "selective" else "mode_not_selective"
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
    _FALLBACK_METRICS["total_compiles"] += 1
    if state.enabled and (op_support is None or op_support["native_coverage"] < PARTIAL_NATIVE_COVERAGE_THRESHOLD):
        _FALLBACK_METRICS["fallback_compiles"] += 1
    if state.enabled and partial_native_coverage:
        _FALLBACK_METRICS["hybrid_compiles"] += 1

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
    if state.strict and bool(abi_report.get("requested")) and not bool(abi_report.get("succeeded")):
        raise RuntimeError(
            f"NATIVE_RUNNER_STRICT=1 and runner ABI prepare failed: {abi_report.get('reason')}"
        )
    # Phase D: ABI model-only is always active when native mode is enabled.
    # NATIVE_RUNNER_ABI_MODEL_ONLY and NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK are
    # no longer supported; legacy compile is unreachable from native-enabled flow.
    abi_model_only = state.enabled  # always True when native is enabled
    capability["runner_abi"]["model_only_requested"] = bool(abi_model_only)
    capability["runner_abi"]["allow_legacy_fallback"] = False

    # Classify the execution mode for observability
    if state.enabled:
        capability["execution_mode_classification"] = "native_abi_model_only"
    else:
        capability["execution_mode_classification"] = "legacy_only"

    if state.enabled:
        abi_session = abi_report.get("session")
        if abi_session is None or not bool(abi_report.get("succeeded")):
            # ABI session not available — fall back to legacy compile unless
            # explicitly forbidden.  This happens when NATIVE_RUNNER_ABI_EXEC
            # is not set (the default) or when the native lib lacks symbols.
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
            capability["fallback_metrics"] = native_runner_capability_report().get("fallback_metrics", {})
            try:
                setattr(model, "_native_runner_report", capability)
            except Exception:
                pass
            logger.info(
                "Native runner ABI-only model active: execution_path=%s enabled=%s strict=%s",
                capability.get("execution_path"),
                capability.get("enabled"),
                capability.get("strict"),
            )
            return model

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

    # --- Attach native forward wrapper when coverage meets threshold ---
    if (
        state.enabled
        and op_support is not None
        and op_support["native_coverage"] >= PARTIAL_NATIVE_COVERAGE_THRESHOLD
    ):
        try:
            wrapper = NativeForwardWrapper(model, set(op_support["supported"]))
            setattr(model, "_native_forward_wrapper", wrapper)
            capability["native_forward_wrapper"] = {
                "attached": True,
                "supported_ops": len(op_support["supported"]),
            }
            # Propagate wrapper to individual CompiledOp instances
            try:
                for layer in getattr(model, 'layers', []):
                    ops = getattr(layer, 'ops', None)
                    if ops is not None:
                        for op in ops.values():
                            if hasattr(op, 'forward'):
                                op._native_wrapper = wrapper
                capability["native_forward_wrapper"]["propagated"] = True
            except Exception as exc:
                logger.debug("Failed to propagate native wrapper to ops: %s", exc)
                capability["native_forward_wrapper"]["propagated"] = False
        except Exception as exc:
            logger.debug("Failed to attach native forward wrapper: %s", exc)
            capability["native_forward_wrapper"] = {
                "attached": False,
                "error": str(exc),
            }
    # --- Attach SubgraphDispatchers to compiled layers for batch Rust dispatch ---
    if (
        state.enabled
        and op_support is not None
        and op_support["native_coverage"] >= PARTIAL_NATIVE_COVERAGE_THRESHOLD
    ):
        try:
            supported_set = set(op_support["supported"])
            dispatchers_attached = 0
            dispatchers_skipped = 0
            layers = getattr(model, 'layers', [])
            layer_graph_list = getattr(model, '_layer_graphs', layer_graphs)
            for i, layer in enumerate(layers):
                if i < len(layer_graph_list):
                    graph = layer_graph_list[i]
                    if hasattr(graph, 'nodes'):
                        dispatcher = SubgraphDispatcher(graph, supported_set)
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
                    dispatchers_attached, len(layers),
                )
        except Exception as exc:
            logger.debug("Failed to attach subgraph dispatchers: %s", exc)
            capability["subgraph_dispatch"] = {
                "attached": 0,
                "error": str(exc),
            }

    abi_session = abi_report.get("session")
    if abi_session is not None:
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
            except Exception:
                pass

    selective_layer_exec_enabled = _env_flag("NATIVE_RUNNER_SELECTIVE_LAYER_EXEC", False)
    selective_layer_strict = _env_flag("NATIVE_RUNNER_SELECTIVE_LAYER_STRICT", False)
    capability["selective_execution"]["layer_exec_enabled"] = selective_layer_exec_enabled
    capability["selective_execution"]["layer_exec_strict"] = selective_layer_strict
    if selective_candidate and bool(selective_activation.get("activated")) and selective_layer_exec_enabled:
        layer_build = build_designer_layer_modules(layer_graphs)
        capability["selective_execution"]["layer_build"] = {
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
        replacements = layer_build.get("replacements") or {}
        model_dim = int(getattr(model, "model_dim", 0) or 0)
        if hasattr(model, "layers"):
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
                    except Exception:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "invalid_layer_index"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    if layer_idx < 0 or layer_idx >= total_layers:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "layer_index_out_of_range"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    wm = payload.get("module")
                    in_id = str(payload.get("input_node_id") or "")
                    if wm is None:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "missing_workflow_module"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    if not in_id:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "missing_input_node_id"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    adapter = DesignerWorkflowLayerAdapter(wm, in_id).as_module()
                    contract_error = _validate_designer_layer_adapter_contract(
                        adapter,
                        model_dim=model_dim,
                        max_seq_len=max_seq_len,
                    )
                    if contract_error:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{contract_error}"
                        )
                        layer_result["skip_reason"] = contract_error
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    model.layers[layer_idx] = adapter
                    capability["selective_execution"]["layer_build"]["applied_layers"] += 1
                    layer_result["applied"] = True
                    capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                except Exception as exc:
                    layer_result["error"] = str(exc)
                    capability["selective_execution"]["layer_build"]["errors"].append(
                        f"apply_layer_{idx}:{exc}"
                    )
                    capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
            if capability["selective_execution"]["layer_build"]["applied_layers"] > 0:
                capability["execution_path"] = "selective_designer_layers_active"
            capability["selective_execution"]["layer_build"]["summary"] = _summarize_layer_build(
                capability["selective_execution"]["layer_build"]
            )
            if selective_layer_strict:
                failed_layers = [
                    item
                    for item in capability["selective_execution"]["layer_build"]["layer_results"]
                    if not bool(item.get("applied"))
                ]
                if failed_layers:
                    capability["selective_execution"]["layer_build"]["summary"]["strict_failed"] = True
                    details = ", ".join(
                        f"layer={item.get('layer_index')} reason={item.get('skip_reason') or item.get('error') or 'unknown'}"
                        for item in failed_layers[:3]
                    )
                    raise RuntimeError(
                        "Selective layer strict mode rejected incompatible replacements: "
                        f"{details}"
                    )
        else:
            capability["selective_execution"]["layer_build"]["errors"].append("model_has_no_layers")
            capability["selective_execution"]["layer_build"]["summary"] = _summarize_layer_build(
                capability["selective_execution"]["layer_build"]
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
        capability["selective_execution"]["layer_build"]["summary"] = _summarize_layer_build(
            capability["selective_execution"]["layer_build"]
        )

    _maybe_fail_on_fallback_rate()
    _maybe_fail_on_legacy_compile_usage()
    # Refresh fallback telemetry after current compile increments.
    capability["fallback_metrics"] = native_runner_capability_report().get("fallback_metrics", {})

    # Attach diagnostic metadata for observability.
    try:
        setattr(model, "_native_runner_report", capability)
    except Exception:
        pass

    warning_count = int(capability.get("semantic_warning_count") or 0)
    if warning_count > 0:
        logger.debug(
            "Native runner capability reports %d semantic mapping warnings (status=%s)",
            warning_count,
            capability.get("status"),
        )
    else:
        logger.info(
            "Native runner capability status=%s enabled=%s strict=%s native_lib=%s",
            capability.get("status"),
            capability.get("enabled"),
            capability.get("strict"),
            "loaded" if native_lib is not None else "unavailable",
        )

    return model
