"""Data-driven eval registry for validation external evals.

Replaces the 410-line procedural _run_external_evals with a declarative
spec table + loop.  Each eval is a small function; the loop handles
status emission, robustness counting, model lifecycle, and exception handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ._types import ExternalEvalResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvalContext:
    """Everything an eval function needs — passed as a single arg."""

    config: Any  # RunConfig (Any to avoid import cycle at type-check time)
    dev: Any  # torch.device
    dev_str: str
    model: Any  # nn.Module | None — set by loop when first model-requiring eval runs
    model_factory: Callable
    input_batches: list
    best_seed: dict | None
    base_final_loss: float | None
    val_loss_ratio: float | None
    val_baseline_ratio: float | None
    val_normalized_ratio: float | None
    source: dict | None
    source_params: int
    source_result_id: str
    scaling_enabled: bool = True
    # Runner methods injected by caller
    ood_check: Callable | None = None
    sensitivity_check: Callable | None = None
    scaling_compare: Callable | None = None
    # Threading — set by the validation runner so eval steps can check for
    # cancellation between (not during) long-running steps like d512 scaling.
    stop_event: Any = None  # threading.Event | None


@dataclass(slots=True)
class EvalSpec:
    """Declarative specification for one eval check."""

    name: str
    result_keys: tuple[str, ...]
    requires_model: bool = False
    requires_loss_ratio: bool = False
    requires_scaling: bool = False
    is_robustness_check: bool = False
    run: Callable[[EvalContext], dict[str, Any]] = field(default=lambda ctx: {})


# ── Individual eval runners (pure functions, 5-20 lines each) ──


def _run_ood(ctx: EvalContext) -> dict[str, Any]:
    n_steps = min(100, max(20, int(ctx.config.validation_steps) // 50))
    return {
        "ood_result": ctx.ood_check(
            ctx.model_factory, ctx.config, ctx.dev, n_steps=n_steps
        )
    }


def _run_sensitivity(ctx: EvalContext) -> dict[str, Any]:
    n_steps = min(100, max(20, int(ctx.config.validation_steps) // 50))
    return {
        "sensitivity_result": ctx.sensitivity_check(
            ctx.model_factory,
            ctx.config,
            ctx.dev,
            base_loss_ratio=float(ctx.val_loss_ratio),
            n_steps=n_steps,
        )
    }


def _run_wikitext(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.wikitext_eval import evaluate_wikitext_perplexity

    seq_len = min(128, ctx.config.validation_seq_len)
    wt = evaluate_wikitext_perplexity(
        ctx.model,
        ctx.config.vocab_size,
        ctx.dev_str,
        n_train_steps=200,
        seq_len=seq_len,
    )
    return {
        "wikitext_perplexity": wt.get("wikitext_perplexity"),
        "wikitext_score": wt.get("wikitext_score"),
    }


def _run_tinystories(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.tinystories_eval import evaluate_tinystories

    seq_len = min(128, ctx.config.validation_seq_len)
    ts = evaluate_tinystories(
        ctx.model,
        ctx.config.vocab_size,
        ctx.dev_str,
        n_train_steps=200,
        seq_len=seq_len,
    )
    return {
        "tinystories_perplexity": ts.get("tinystories_perplexity"),
        "tinystories_score": ts.get("tinystories_score"),
    }


def _run_long_context(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.long_context import run_long_context_sweep

    base_loss = ctx.base_final_loss or max(float(ctx.val_loss_ratio or 1.0), 1e-6)
    lr = (
        float(ctx.best_seed.get("optimizer_lr") or ctx.config.stage1_lr)
        if ctx.best_seed
        else float(ctx.config.stage1_lr)
    )
    lc = run_long_context_sweep(
        ctx.model_factory,
        ctx.config.vocab_size,
        ctx.dev,
        base_loss=base_loss,
        seq_lens=(512, 1024),
        n_steps=min(60, max(20, int(ctx.config.validation_steps) // 100)),
        batch_size=max(1, min(2, ctx.config.validation_batch_size)),
        lr=lr,
    )
    return {
        "long_context_score": lc.get("long_context_score"),
        "long_context_details": lc,
        "max_viable_seq_len": lc.get("max_viable_len"),
    }


def _run_noise(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.noise_sensitivity import evaluate_noise_sensitivity

    nr = evaluate_noise_sensitivity(
        ctx.model,
        ctx.input_batches,
        ctx.dev,
        vocab_size=int(ctx.config.vocab_size),
    )
    return {"noise_score": nr.get("noise_sensitivity_score")}


def _run_sparsity(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.sparsity import evaluate_activation_sparsity

    sr = evaluate_activation_sparsity(ctx.model, ctx.input_batches, ctx.dev)
    return {
        "activation_sparsity_score": sr.get("activation_sparsity_score"),
        "dead_neuron_ratio": sr.get("dead_neuron_ratio"),
    }


def _run_routing(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.routing_heatmap import evaluate_routing_heatmap

    rr = evaluate_routing_heatmap(ctx.model, ctx.input_batches, ctx.dev)
    return {"routing_collapse_score": rr.get("routing_collapse_score")}


def _run_quant(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.quantization import evaluate_sparse_quant_quality

    qr = evaluate_sparse_quant_quality(ctx.model, ctx.input_batches, ctx.dev)
    if qr:
        return {
            "quant_int8_retention": qr.get("full_retention"),
            "quant_quality_per_byte": qr.get("quality_per_byte"),
        }
    return {}


def _run_efficiency_wall(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.efficiency_wall import evaluate_efficiency_wall

    wr = evaluate_efficiency_wall(ctx.model, int(ctx.config.vocab_size), ctx.dev)
    out: dict[str, Any] = {
        "efficiency_wall_score": wr.get("efficiency_wall_score"),
        "scaling_regime": wr.get("scaling_regime"),
        "scaling_flop_efficiency": wr.get("time_scaling_factor"),
    }
    viable = int(wr.get("max_viable_seq_len") or 0)
    if viable > 0:
        out["max_viable_seq_len"] = viable
    return out


def _run_cross_task(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.cross_task_eval import evaluate_cross_task_robustness

    ct = evaluate_cross_task_robustness(
        ctx.model_factory,
        vocab_size=int(ctx.config.vocab_size),
        device=ctx.dev,
        n_train_steps=min(80, max(20, int(ctx.config.validation_steps) // 100)),
        batch_size=max(1, min(4, ctx.config.validation_batch_size)),
        seq_len=min(128, ctx.config.validation_seq_len),
    )
    return {"cross_task_score": ct.get("cross_task_score")}


def _run_long_range_ar(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.long_range_ar import long_range_ar_score

    ar = long_range_ar_score(
        ctx.model,
        seq_lens=(128, 256, 512, 1024),
        n_train_steps=min(300, max(100, int(ctx.config.validation_steps) // 50)),
        batch_size=max(1, min(16, ctx.config.validation_batch_size)),
        device=ctx.dev_str,
    )
    return {"long_ctx_assoc_score": ar.score}


def _run_induction_v2(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.induction_probe_v2_investigation import run_induction_v2_investigation

    r = run_induction_v2_investigation(ctx.model, device=ctx.dev_str)
    return {
        "induction_v2_investigation_auc": r.auc,
        "induction_v2_investigation_max_gap_acc": r.max_gap_acc,
        "induction_v2_investigation_protocol_version": r.protocol_version,
    }


def _run_binding_v2(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.binding_probe_v2_investigation import run_binding_v2_investigation

    r = run_binding_v2_investigation(ctx.model, device=ctx.dev_str)
    return {
        "binding_v2_investigation_auc": r.auc,
        "binding_v2_investigation_max_distance_acc": r.max_distance_acc,
        "binding_v2_investigation_protocol_version": r.protocol_version,
    }


def _run_passkey(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.passkey_retrieval import passkey_retrieval_score

    pk = passkey_retrieval_score(
        ctx.model,
        seq_lens=(256, 512, 1024, 2048),
        n_train_steps=min(300, max(100, int(ctx.config.validation_steps) // 50)),
        batch_size=max(1, min(16, ctx.config.validation_batch_size)),
        device=ctx.dev_str,
    )
    return {"long_ctx_passkey_score": pk.score}


def _run_multi_hop(ctx: EvalContext) -> dict[str, Any]:
    from ...eval.multi_hop_retrieval import multi_hop_retrieval_score

    mh = multi_hop_retrieval_score(
        ctx.model,
        seq_lens=(256, 512, 1024),
        hop_depths=(2, 3),
        n_train_steps=min(300, max(100, int(ctx.config.validation_steps) // 50)),
        batch_size=max(1, min(16, ctx.config.validation_batch_size)),
        device=ctx.dev_str,
    )
    return {"long_ctx_multi_hop_score": mh.score}


def _run_hierarchy(ctx: EvalContext) -> dict[str, Any]:
    import torch
    from ...eval.hierarchy_probe import hierarchy_fitness

    model = ctx.model
    if model is None:
        return {}

    # Generate random input and extract hidden states via forward hook
    vocab_size = int(ctx.config.vocab_size)
    seq_len = min(64, ctx.config.validation_seq_len)
    input_ids = torch.randint(0, vocab_size, (2, seq_len), device=ctx.dev)

    hidden_states = []

    def _hook(module, _input, output):
        # Capture output of the last layer before the head
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            hidden_states.append(out.detach())

    # Register hook on the last child module (before output projection)
    hooks = []
    children = list(model.children())
    if children:
        hooks.append(
            children[-2].register_forward_hook(_hook)
            if len(children) >= 2
            else children[-1].register_forward_hook(_hook)
        )

    try:
        with torch.no_grad():
            model(input_ids)
    except Exception as exc:
        logger.debug("Suppressed error: %s", exc)
    finally:
        for h in hooks:
            h.remove()

    if not hidden_states:
        return {}

    result = hierarchy_fitness(hidden_states[-1], max_tokens=100)
    return {
        "fp_gromov_delta": result.get("gromov_delta"),
        "fp_hierarchy_fitness": result.get("hierarchy_fitness"),
    }


def _run_scaling_d256(ctx: EvalContext) -> dict[str, Any]:
    payload = ctx.scaling_compare(
        config=ctx.config,
        dev_str=ctx.dev_str,
        best_seed=ctx.best_seed,
        val_loss_ratio=ctx.val_loss_ratio,
        source_params=ctx.source_params,
        source=ctx.source,
        d_model=int(ctx.config.model_dim),
    )
    if payload is None:
        return {}
    return {
        "scaling_result": payload,
        "scaling_param_efficiency": payload.get("best_param_efficiency"),
        "scaling_flop_efficiency": payload.get("flop_efficiency"),
        "scaling_best_family": payload.get("best_param_efficiency_family"),
        "scaling_gate_passed_val": bool(payload.get("scaling_gate_passed")),
        "scaling_confidence": str(payload.get("confidence") or "local_reference"),
    }


def _run_scaling_d512(ctx: EvalContext) -> dict[str, Any]:
    if not bool(getattr(ctx.config, "scaling_d512_enabled", True)):
        return {}
    # Run d512 scaling comparison with a timeout guard. This step trains
    # a full 10K-step model at d=512 and previously hung indefinitely,
    # blocking the entire continuous loop. The timeout (20 min) lets the
    # validation pipeline proceed even if the comparison stalls.
    import concurrent.futures

    _D512_TIMEOUT_SEC = 20 * 60  # 20 minutes — generous but finite

    def _run():
        return ctx.scaling_compare(
            config=ctx.config,
            dev_str=ctx.dev_str,
            best_seed=ctx.best_seed,
            val_loss_ratio=ctx.val_loss_ratio,
            source_params=ctx.source_params,
            source=ctx.source,
            d_model=512,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            payload = future.result(timeout=_D512_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "d512 scaling comparison timed out after %ds — skipping",
            _D512_TIMEOUT_SEC,
        )
        return {}
    except Exception as exc:
        logger.warning("d512 scaling comparison failed: %s", exc)
        return {}
    if payload is None:
        return {}
    out: dict[str, Any] = {
        "scaling_d512_param_efficiency": payload.get("best_param_efficiency"),
    }
    # Nest full d512 payload into existing scaling_result
    out["_d512_payload"] = payload
    return out


# ── Spec table ──

EVAL_SPECS: tuple[EvalSpec, ...] = (
    EvalSpec(
        name="OOD robustness check",
        result_keys=("ood_result",),
        requires_loss_ratio=True,
        is_robustness_check=True,
        run=_run_ood,
    ),
    EvalSpec(
        name="sensitivity check",
        result_keys=("sensitivity_result",),
        requires_loss_ratio=True,
        is_robustness_check=True,
        run=_run_sensitivity,
    ),
    EvalSpec(
        name="reconstructing eval model",
        result_keys=(),
        requires_model=True,
        run=lambda ctx: {},  # model reconstruction is handled by the loop
    ),
    EvalSpec(
        name="WikiText perplexity",
        result_keys=("wikitext_perplexity", "wikitext_score"),
        requires_model=True,
        run=_run_wikitext,
    ),
    EvalSpec(
        name="TinyStories perplexity",
        result_keys=("tinystories_perplexity", "tinystories_score"),
        requires_model=True,
        run=_run_tinystories,
    ),
    EvalSpec(
        name="long-context sweep",
        result_keys=(
            "long_context_score",
            "long_context_details",
            "max_viable_seq_len",
        ),
        requires_model=True,
        run=_run_long_context,
    ),
    EvalSpec(
        name="long-range associative recall",
        result_keys=("long_ctx_assoc_score",),
        requires_model=True,
        run=_run_long_range_ar,
    ),
    EvalSpec(
        name="passkey retrieval",
        result_keys=("long_ctx_passkey_score",),
        requires_model=True,
        run=_run_passkey,
    ),
    EvalSpec(
        name="multi-hop retrieval",
        result_keys=("long_ctx_multi_hop_score",),
        requires_model=True,
        run=_run_multi_hop,
    ),
    EvalSpec(
        name="noise sensitivity",
        result_keys=("noise_score",),
        requires_model=True,
        is_robustness_check=True,
        run=_run_noise,
    ),
    EvalSpec(
        name="activation sparsity",
        result_keys=("activation_sparsity_score", "dead_neuron_ratio"),
        requires_model=True,
        is_robustness_check=True,
        run=_run_sparsity,
    ),
    EvalSpec(
        name="routing heatmap",
        result_keys=("routing_collapse_score",),
        requires_model=True,
        is_robustness_check=True,
        run=_run_routing,
    ),
    EvalSpec(
        name="quantization quality",
        result_keys=("quant_int8_retention", "quant_quality_per_byte"),
        requires_model=True,
        is_robustness_check=True,
        run=_run_quant,
    ),
    EvalSpec(
        name="efficiency wall",
        result_keys=(
            "efficiency_wall_score",
            "scaling_regime",
            "scaling_flop_efficiency",
            "max_viable_seq_len",
        ),
        requires_model=True,
        requires_scaling=True,
        is_robustness_check=True,
        run=_run_efficiency_wall,
    ),
    EvalSpec(
        name="cross-task robustness",
        result_keys=("cross_task_score",),
        is_robustness_check=True,
        run=_run_cross_task,
    ),
    EvalSpec(
        name="hierarchy probe",
        result_keys=("fp_gromov_delta", "fp_hierarchy_fitness"),
        requires_model=True,
        run=_run_hierarchy,
    ),
    EvalSpec(
        name="scaling reference comparison (d256)",
        result_keys=(
            "scaling_result",
            "scaling_param_efficiency",
            "scaling_flop_efficiency",
            "scaling_best_family",
            "scaling_gate_passed_val",
            "scaling_confidence",
        ),
        requires_scaling=True,
        run=_run_scaling_d256,
    ),
    EvalSpec(
        name="scaling reference comparison (d512)",
        result_keys=("scaling_d512_param_efficiency",),
        requires_scaling=True,
        run=_run_scaling_d512,
    ),
    EvalSpec(
        name="induction v2 (investigation)",
        result_keys=(
            "induction_v2_investigation_auc",
            "induction_v2_investigation_max_gap_acc",
            "induction_v2_investigation_protocol_version",
        ),
        requires_model=True,
        run=_run_induction_v2,
    ),
    EvalSpec(
        name="binding v2 (investigation)",
        result_keys=(
            "binding_v2_investigation_auc",
            "binding_v2_investigation_max_distance_acc",
            "binding_v2_investigation_protocol_version",
        ),
        requires_model=True,
        run=_run_binding_v2,
    ),
)


def apply_breakthrough_logic(
    result: ExternalEvalResult,
    config: Any,
    val_loss_ratio: float | None,
    val_baseline_ratio: float | None,
    val_normalized_ratio: float | None,
    passed_seeds: list,
    source: dict | None,
    scaling_enabled: bool,
    source_result_id: str,
) -> None:
    """Post-eval: determine scaling gate, breakthrough status, confidence fallbacks."""
    # Scaling gate
    scaling_gate_passed = not scaling_enabled or (
        result.scaling_param_efficiency is not None
        and float(result.scaling_param_efficiency)
        >= float(config.scaling_param_efficiency_target)
        and (
            result.scaling_flop_efficiency is None
            or float(result.scaling_flop_efficiency)
            <= float(config.scaling_flop_ceiling)
        )
    )
    # Override with scaling result if present
    if result.scaling_gate_passed_val is not None:
        scaling_gate_passed = result.scaling_gate_passed_val

    result.scaling_gate_passed_val = scaling_gate_passed

    # Confidence fallbacks
    if result.scaling_confidence is None:
        result.scaling_confidence = (
            "disabled"
            if not scaling_enabled
            else "high"
            if scaling_gate_passed
            else "low"
        )
    if result.scaling_best_family is None:
        result.scaling_best_family = str(
            (source or {}).get("most_similar_to") or "reference"
        )

    # Scaling result fallback
    if result.scaling_result is None:
        result.scaling_result = {
            "param_efficiency": result.scaling_param_efficiency,
            "flop_efficiency": result.scaling_flop_efficiency,
            "gate_passed": scaling_gate_passed,
            "confidence": result.scaling_confidence,
            "enabled": scaling_enabled,
        }
    else:
        result.scaling_result["enabled"] = scaling_enabled
        result.scaling_result["gate_passed"] = scaling_gate_passed

    # Breakthrough determination
    raw_threshold = float(getattr(config, "breakthrough_raw_threshold", 0.70) or 0.70)
    norm_threshold = float(
        getattr(config, "breakthrough_normalized_threshold", 0.85) or 0.85
    )
    raw_passed = val_loss_ratio is not None and float(val_loss_ratio) <= raw_threshold
    norm_passed = (
        val_normalized_ratio is not None
        and float(val_normalized_ratio) >= norm_threshold
    )

    seeds_count = len(passed_seeds) if passed_seeds else 0
    seeds_total = int(getattr(config, "validation_n_seeds", 5) or 5)
    seeds_can_promote = (seeds_total < 3) or (seeds_count >= 1)

    if (
        raw_passed
        and norm_passed
        and scaling_gate_passed
        and (val_baseline_ratio is None or float(val_baseline_ratio) < 1.0)
        and seeds_can_promote
    ):
        result.is_breakthrough = True
    elif seeds_count == 0 and seeds_total >= 3:
        result.is_breakthrough = False
        logger.info(
            "breakthrough_blocked_seeds_passed_zero: result_id=%s "
            "val_loss_ratio=%s seeds_passed=%d seeds_total=%d",
            source_result_id[:12],
            val_loss_ratio,
            seeds_count,
            seeds_total,
        )
    else:
        result.is_breakthrough = False

    result.flop_gated = bool(
        not scaling_gate_passed and result.scaling_flop_efficiency is not None
    )

    if result.robustness_checks_failed > 0:
        logger.warning(
            "validation[%s]: %d/%d robustness checks failed",
            source_result_id[:8],
            result.robustness_checks_failed,
            result.robustness_checks_attempted,
        )


def _aggregate_long_ctx_scores(result: ExternalEvalResult) -> None:
    """Compute long-context aggregate and combined scores from sub-scores."""
    # Copy scaling sweep score
    result.long_ctx_scaling_score = result.long_context_score

    # Retrieval aggregate = mean of available retrieval sub-scores
    retrieval_scores = [
        s
        for s in (
            result.long_ctx_assoc_score,
            result.long_ctx_passkey_score,
            result.long_ctx_multi_hop_score,
        )
        if s is not None
    ]
    if retrieval_scores:
        result.long_ctx_retrieval_aggregate = round(
            sum(retrieval_scores) / len(retrieval_scores), 4
        )

    # Combined = 0.4 * scaling + 0.6 * retrieval_aggregate
    scaling = result.long_ctx_scaling_score or 0.0
    retrieval = result.long_ctx_retrieval_aggregate or 0.0
    has_scaling = result.long_ctx_scaling_score is not None
    has_retrieval = result.long_ctx_retrieval_aggregate is not None

    if has_scaling and has_retrieval:
        result.long_ctx_combined_score = round(0.4 * scaling + 0.6 * retrieval, 4)
    elif has_scaling:
        result.long_ctx_combined_score = round(scaling, 4)
    elif has_retrieval:
        result.long_ctx_combined_score = round(retrieval, 4)


def run_eval_suite(
    *,
    ctx: EvalContext,
    result: ExternalEvalResult,
    vstatus: Callable[[str], None],
) -> None:
    """Execute all eval specs, mutating *result* in place.

    Handles model lifecycle (lazy construction + cleanup), robustness
    counting, and per-eval exception handling.
    """
    from ._helpers import clear_gpu_memory

    model = None
    try:
        for spec in EVAL_SPECS:
            # Check stop event before each eval step — the d512 scaling
            # comparison trains a full model (~20 min) and previously
            # hung because cancel couldn't reach the inner training loop.
            # This at least lets us abort between eval steps so the
            # continuous loop can resume instead of being wedged.
            if hasattr(ctx, "stop_event") and ctx.stop_event is not None:
                if ctx.stop_event.is_set():
                    logger.info(
                        "Eval suite aborted at %s — stop event set",
                        spec.name,
                    )
                    break
            if spec.requires_loss_ratio and ctx.val_loss_ratio is None:
                continue
            if spec.requires_scaling and not ctx.scaling_enabled:
                continue

            vstatus(spec.name)

            if spec.is_robustness_check:
                result.robustness_checks_attempted += 1

            # Lazy model construction on first model-requiring eval
            if spec.requires_model and model is None:
                model = ctx.model_factory()
                if model is None:
                    logger.warning(
                        "Model reconstruction returned None for %s",
                        ctx.source_result_id[:8],
                    )
                    return
                model = model.to(ctx.dev)
                ctx.model = model

            try:
                values = spec.run(ctx)
                d512_payload = values.pop("_d512_payload", None)
                for key, val in values.items():
                    # max_viable_seq_len uses max() semantics across evals
                    if key == "max_viable_seq_len" and val is not None:
                        cur = getattr(result, key) or 0
                        setattr(result, key, max(cur, int(val)))
                    else:
                        setattr(result, key, val)
                # Nest d512 payload into scaling_result if present
                if d512_payload is not None and isinstance(result.scaling_result, dict):
                    result.scaling_result["d512_result"] = d512_payload
            except Exception as exc:
                if spec.is_robustness_check:
                    result.robustness_checks_failed += 1
                logger.warning(
                    "%s FAILED for %s: %s",
                    spec.name,
                    ctx.source_result_id[:8],
                    exc,
                )
    finally:
        if model is not None:
            del model
            ctx.model = None
        clear_gpu_memory()

    # Aggregate long-context sub-scores
    _aggregate_long_ctx_scores(result)
