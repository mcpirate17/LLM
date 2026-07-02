"""Trained-instance NM-11 softmax-twin audit (O2 lane, 2026-07-02).

Design-time NM-11 scores (`component_fab/proposer/algebraic_properties.py`)
have only ever been measured on freshly-initialized catalog variants. This
tool answers the open question in
`research/notes/nm_verification_split_plan_2026-07-02.md` (S:O2): does a
mechanism's softmax-twin score drift after gradient descent, for the ops that
actually cleared the production S1 loss gate?

Pipeline, per in-scope op:
  1. Inventory: find S1-passing `graph_runs` rows (production 400-campaign
     `14b8f2c7-c66` + batch-4 `fbbf2566-227`) whose `graph_json` contains the
     op, and check the filesystem/graph for a persisted trained checkpoint.
  2. If no checkpoint is persisted (expected — verified live 2026-07-02: only
     compressed loss-curve JSON is kept under
     `research/artifacts/notebook/training_curves/<result_id>/`, no `.pt`
     weights), compile the exact graph at the ORIGINAL experiment's production
     config (model_dim, n_layers, vocab_size, corpus, lr, batch, steps — read
     straight from the `experiments.config_json` row, not re-typed), extract
     the target op's compiled submodule, score it at init, calibrate CPU
     wall-clock cost on a throwaway instance, and only run the full
     production-length retrain if the calibrated projection clears
     `--cpu-budget-seconds` (default 600s). Otherwise the op is left QUEUED
     with the exact GPU command to finish it later — this script never
     silently shrinks the step budget to force a fit.

Methodology disclosure (kept honest, not buried): the retrain reuses
`ExperimentRunner._micro_train` (the real production loop: Muon/AdamW
optimizer-group split, real wikitext103 corpus batches, real LR schedule) with
`profile_disable_post_eval=True` so the expensive post-S1 probe suite
(wikitext/hellaswag/BLiMP/AR-gate/binding) is skipped — those probes are O1's
job, not O2's, and would blow the CPU budget for no benefit to a twin-score
read. Nothing is written back to `runs.db`; this tool only reads it.

Detector: `component_fab/proposer/algebraic_properties.py:measure_algebraic_properties`
matches the ``lane: nn.Module`` contract exercised by
`component_fab/validator/mechanism.py:_check_softmax_twin` at design time.
Cross-token ops get NO pointwise waiver (the C12 rule) — this tool does not
special-case any op.

This tool FLAGS (score >= 0.55) and reports; it never promotes or demotes an
op. That call is Fable's (QC2).

Usage:
    python -m research.tools.audit_trained_twin_scores \\
        --db research/runs.db \\
        --out research/reports/trained_twin_audit_2026-07-02.json \\
        --cpu-budget-seconds 600
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import signal
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from component_fab.proposer.algebraic_properties import AlgebraicPropertyProbe
from research.defaults import RUNS_DB
from research.synthesis.compiler import compile_model
from research.synthesis.compiled_op import CompiledOp
from research.synthesis.graph import ComputationGraph
from research.synthesis.serializer import graph_from_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = Path(RUNS_DB)
DEFAULT_OUT = REPO_ROOT / "research" / "reports" / "trained_twin_audit_2026-07-02.json"

#: The 8 ops O2 is scoped to (research/notes/nm_verification_split_plan_2026-07-02.md S:O2).
IN_SCOPE_OPS: tuple[str, ...] = (
    "token_merge_mix",
    "weight_dictionary_mix",
    "integral_control_mixer",
    "butterfly_mix",
    "idempotent_oblique_memory",
    "cdma_slot_binding",
    "persistent_memory_refine",
    "block_sparse_mix",
)

#: The two experiments this plan anchors to: the 400-program production
#: campaign and campaign batch 4 (research/notes/autonomous_run_findings_2026-07-02.md).
IN_SCOPE_EXPERIMENT_IDS: tuple[str, ...] = ("14b8f2c7-c66", "fbbf2566-227")

FLAG_THRESHOLD = 0.55
TWIN_THRESHOLD = 0.6

_INVENTORY_SQL = """
WITH nm_ops(op) AS (
    VALUES {op_values}
)
SELECT
    gr.result_id,
    gr.experiment_id,
    gr.stage1_passed,
    gr.loss_ratio,
    gr.graph_fingerprint,
    (SELECT GROUP_CONCAT(op) FROM nm_ops WHERE g.graph_json LIKE '%' || nm_ops.op || '%') AS nm_ops_present
FROM graph_runs AS gr
JOIN graphs AS g ON g.graph_fingerprint = gr.graph_fingerprint
WHERE gr.experiment_id IN ({exp_ids})
  AND gr.stage1_passed = 1
  AND EXISTS (SELECT 1 FROM nm_ops WHERE g.graph_json LIKE '%' || nm_ops.op || '%')
ORDER BY gr.experiment_id, gr.result_id
"""


class TimeoutGuard:
    """Hard SIGALRM wall-clock cutoff for the full retrain (Linux, main thread only)."""

    def __init__(self, seconds: int):
        self.seconds = int(seconds)

    def __enter__(self) -> "TimeoutGuard":
        def _handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"CPU retrain exceeded hard cutoff of {self.seconds}s")

        self._prev = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._prev)


def _inventory(
    db_path: Path, ops: tuple[str, ...], experiment_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    op_values = ", ".join(f"('{op}')" for op in ops)
    exp_ids = ", ".join(f"'{eid}'" for eid in experiment_ids)
    sql = _INVENTORY_SQL.format(op_values=op_values, exp_ids=exp_ids)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()
    for row in rows:
        present = str(row.get("nm_ops_present") or "")
        row["nm_ops_present"] = [op for op in ops if op in present.split(",")]
    return rows


def _select_representative_rows(
    rows: list[dict[str, Any]], ops: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """ALL S1-passer rows bearing each op, best (lowest loss_ratio) first.

    Returning the full ordered list lets ``audit_op`` fall through to a
    cheaper S1-passing graph when the best one's projected CPU retrain
    exceeds the budget (e.g. integral_control_mixer's best row shares a
    heavy graph, but the op also S1-passed in a much lighter one). Every
    candidate is a real S1 passer, so any of them is a valid trained
    instance to audit.
    """
    by_op: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for op in row["nm_ops_present"]:
            if op in ops:
                by_op.setdefault(op, []).append(row)
    for op_rows in by_op.values():
        op_rows.sort(
            key=lambda r: (r.get("loss_ratio") is None, r.get("loss_ratio") or 0.0)
        )
    return by_op


def _find_persisted_checkpoint(result_id: str, graph_fingerprint: str) -> str | None:
    """Search for an actual trained-weight artifact (.pt), not a loss-curve JSON."""
    search_roots = [
        REPO_ROOT / "research" / "checkpoints",
        REPO_ROOT / "research" / "artifacts",
    ]
    needles = [result_id, graph_fingerprint]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pt"):
            name = str(path)
            if any(needle and needle in name for needle in needles):
                return str(path)
    return None


def _experiment_config(db_path: Path, experiment_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT config_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"No experiments row for experiment_id={experiment_id!r}")
    return json.loads(row["config_json"])


def _graph_json(db_path: Path, graph_fingerprint: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT graph_json FROM graphs WHERE graph_fingerprint = ?",
            (graph_fingerprint,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"No graphs row for graph_fingerprint={graph_fingerprint!r}")
    return str(row["graph_json"])


def _build_run_config(
    raw_config: dict[str, Any], *, stage1_steps: int, device: str = "cpu"
):
    from research.scientist.runner import RunConfig

    known = {f.name for f in dataclasses.fields(RunConfig)}
    kwargs = {k: v for k, v in raw_config.items() if k in known}
    config = RunConfig(**kwargs)
    config.device = device
    config.stage1_steps = int(stage1_steps)
    config.profile_disable_post_eval = (
        True  # skip wikitext/hellaswag/blimp/ar-gate/binding — O1's job
    )
    config.enable_cuda_graphs = False
    config.enable_perf_tracing = False
    config.enable_kernel_profiling = False
    config.enable_starvation_monitoring = False
    config.collect_training_curve = False
    # The production inflight-no-progress kill switch reads loss ratio against
    # a threshold tuned for the full stage1_steps budget; it false-positives
    # on this tool's short calibration windows (and is redundant here — these
    # rows are already confirmed S1 passers, so a diagnostic replay isn't
    # re-deciding pass/fail, only producing trained weights to score).
    config.profile_disable_inflight_checks = True
    if device == "cpu":
        # The production config's optimizer_fused/foreach knobs target CUDA;
        # on CPU, fused Adam is unsupported and foreach hits dtype edge cases
        # on the 0-d ReZero gate scalars these NM ops use (measured live
        # 2026-07-02: "b must be float32" from a 0-d param under foreach).
        # On CUDA the production flags are kept as-is.
        config.optimizer_fused = False
        config.optimizer_foreach = False
    # Discovery/validation loss are optional diagnostic side-evals, not part
    # of the gradient-update path this tool cares about; discovery-loss eval
    # crashes outright on CPU for these graphs (measured live 2026-07-02:
    # aria_core.add_f32 dtype mismatch in _micro_train_discovery_eval) and
    # neither is needed to produce trained weights. Kept off on CUDA too so
    # all 8 ops in the report share one training recipe.
    config.stage1_compute_discovery_loss = False
    config.stage1_compute_val_loss = False
    return config


def _op_node_name(graph: ComputationGraph, op_name: str) -> str:
    """Find the node id (as CompiledLayer.ops key) for the first node with this op_name."""
    for nid, node in graph.nodes.items():
        if node.is_input or node.op_name != op_name:
            continue
        if len(node.input_ids) != 1:
            raise ValueError(
                f"{op_name} node {nid} takes {len(node.input_ids)} inputs; "
                "the [B, L, D] -> [B, L, D] twin-score contract requires exactly 1"
            )
        return str(nid)
    raise ValueError(f"op {op_name!r} not found as a non-input node in this graph")


def _compile_and_locate_multi(
    graph_json: str,
    op_names: list[str],
    *,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    seed: int,
) -> tuple[nn.Module, dict[str, CompiledOp], int]:
    """Compile once, extract the layer-0 module for EVERY requested op.

    Multi-op extraction exists because several in-scope ops co-occur in one
    S1-passing graph (e.g. 3a643630-aa5 carries both integral_control_mixer
    and idempotent_oblique_memory) — one retrain then serves all of them.
    """
    torch.manual_seed(seed)
    graph = graph_from_json(graph_json)
    node_ids = {op: _op_node_name(graph, op) for op in op_names}
    layer_graphs = [graph] * int(n_layers)
    model = compile_model(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        executor="compiled",
    )
    layer0 = model.layers[0]
    op_modules: dict[str, CompiledOp] = {}
    for op_name, node_id in node_ids.items():
        op_module = layer0.ops[node_id]
        if op_module.op_name != op_name:
            raise ValueError(
                f"extracted module op_name={op_module.op_name!r} != expected {op_name!r}"
            )
        op_modules[op_name] = op_module
    return model, op_modules, int(graph.model_dim)


def _compile_and_locate(
    graph_json: str,
    op_name: str,
    *,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    seed: int,
) -> tuple[nn.Module, CompiledOp, int]:
    model, op_modules, dim = _compile_and_locate_multi(
        graph_json,
        [op_name],
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    return model, op_modules[op_name], dim


def _score_multi_seed(
    op_module: CompiledOp, *, dim: int, n_probes: int = 5
) -> dict[str, Any]:
    """Mean/max softmax_twin_score over `n_probes` distinct-stimulus probes.

    Uses ``AlgebraicPropertyProbe._measure_from_stimuli`` directly with
    explicit per-probe seeds — calling the ``measure_algebraic_properties``
    wrapper with ``n_seeds=1`` in a loop re-probes the identical seed-0
    stimuli every time (its internal loop is ``range(n_seeds)``), which is how
    the first draft of this tool produced five bit-identical "probes".
    Per-probe stats over distinct stimuli match the "max over N seeds"
    methodology already cited for token_merge_mix (0.647 max over 5 init
    seeds, commit fa3adfd9) so the trained-instance number is directly
    comparable.
    """
    op_module.eval()
    rows: list[dict[str, float]] = []
    for seed in range(n_probes):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(4, 16, dim, generator=gen)
        y = torch.randn(4, 16, dim, generator=gen)
        rows.append(AlgebraicPropertyProbe._measure_from_stimuli(op_module, x, y, gen))
    twin_scores = [row["softmax_twin_score"] for row in rows]
    return {
        "n_probes": n_probes,
        "twin_score_mean": sum(twin_scores) / len(twin_scores),
        "twin_score_max": max(twin_scores),
        "twin_score_all": [round(s, 5) for s in twin_scores],
        "constant_token_preservation_mean": sum(
            row["constant_token_preservation"] for row in rows
        )
        / len(rows),
        "convex_range_fraction_mean": sum(row["convex_range_fraction"] for row in rows)
        / len(rows),
        "cross_token_mixing_mean": sum(row["cross_token_mixing"] for row in rows)
        / len(rows),
    }


def _assert_training_actually_ran(
    result: dict[str, Any], target_steps: int, op_name: str
) -> None:
    """Fail loud rather than silently score an untrained model as 'trained'.

    ``ExperimentRunner._micro_train`` catches its own internal exceptions into
    ``result['error']`` instead of raising (production behavior: a failed
    candidate should not crash a screening batch). That is exactly the
    silent-fallback shape this project's standing rule forbids for a tool
    whose entire job is to report a trained-vs-init delta, so this audit tool
    re-raises here IF the step budget was not actually completed (a real
    crash — e.g. the mid-loop exception this check was written to catch, a
    ``program_results_compat`` lookup or a native-kernel fault). A regression
    S1-pass/fail *verdict* (``error_type`` in {failed_convergence,
    insufficient_learning, inflight_no_progress, ...} with the full step
    count completed) is expected and uninformative noise here — a short
    calibration/full-audit run legitimately won't always hit the same loss
    ratio as the original production run, and this tool cares about trained
    WEIGHTS, not re-deciding a pass/fail verdict already settled in runs.db.
    """
    n_ran = int(result.get("n_train_steps") or 0)
    if n_ran < target_steps:
        raise RuntimeError(
            f"{op_name}: _micro_train did not complete the {target_steps}-step production "
            f"budget (ran {n_ran} steps, error={result.get('error')!r}, "
            f"error_type={result.get('error_type')!r}) — refusing to report a 'trained' score "
            "for a model that was not actually trained"
        )


def _calibrate_seconds_per_step(
    graph_json: str,
    op_name: str,
    raw_config: dict[str, Any],
    db_path: Path,
    *,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    calibration_steps: int,
) -> float:
    """Throwaway compile+train to measure CPU seconds/step; does not touch the scored instance.

    Uses the real ``db_path`` (read via ``ExperimentRunner``'s normal
    connection, never through this tool's own read-only handles) because
    ``_micro_train`` reads the ``program_results_compat`` VIEW (dedup/adaptive
    budget lookups). It is a VIEW with no INSTEAD OF triggers — SQLite refuses
    any write to it outright — and nothing in the ``_micro_train`` call chain
    issues one (verified: no ``self.nb``/notebook writes in
    ``execution_training_micro.py`` or ``execution_training_post.py``).
    """
    from research.scientist.runner import ExperimentRunner

    model, _op_module, _dim = _compile_and_locate(
        graph_json,
        op_name,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        seed=999_001,
    )
    config = _build_run_config(raw_config, stage1_steps=calibration_steps)
    runner = ExperimentRunner(str(db_path))
    dev = torch.device("cpu")
    t0 = time.perf_counter()
    result = runner._micro_train(model, config, dev, seed=999_001)
    elapsed = time.perf_counter() - t0
    _assert_training_actually_ran(result, calibration_steps, op_name)
    del model
    return elapsed / max(1, calibration_steps)


def _train_full(
    graph_json: str,
    op_name: str,
    raw_config: dict[str, Any],
    db_path: Path,
    *,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    target_steps: int,
    seed: int,
) -> tuple[dict[str, Any], CompiledOp, int]:
    from research.scientist.runner import ExperimentRunner

    model, op_module, dim = _compile_and_locate(
        graph_json,
        op_name,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    config = _build_run_config(raw_config, stage1_steps=target_steps)
    runner = ExperimentRunner(str(db_path))
    dev = torch.device("cpu")
    result = runner._micro_train(model, config, dev, seed=seed)
    _assert_training_actually_ran(result, target_steps, op_name)
    return result, op_module, dim


def _gpu_replay_command(result_id: str, target_steps: int) -> str:
    return (
        "python -m research.tools.exact_graph_replay "
        f"--result-id {result_id} --device cuda --stage1-steps {target_steps} --verbose"
    )


@dataclasses.dataclass(slots=True)
class _OpContext:
    op_name: str
    result_id: str
    experiment_id: str
    graph_fingerprint: str
    loss_ratio: float | None
    raw_config: dict[str, Any]
    graph_json: str
    n_layers: int
    vocab_size: int
    max_seq_len: int
    target_steps: int
    checkpoint: str | None
    db_path: Path


def _load_op_context(op_name: str, row: dict[str, Any], db_path: Path) -> _OpContext:
    result_id = str(row["result_id"])
    graph_fingerprint = str(row["graph_fingerprint"])
    experiment_id = str(row["experiment_id"])
    raw_config = _experiment_config(db_path, experiment_id)
    return _OpContext(
        op_name=op_name,
        result_id=result_id,
        experiment_id=experiment_id,
        graph_fingerprint=graph_fingerprint,
        loss_ratio=row.get("loss_ratio"),
        raw_config=raw_config,
        graph_json=_graph_json(db_path, graph_fingerprint),
        n_layers=int(raw_config["n_layers"]),
        vocab_size=int(raw_config["vocab_size"]),
        max_seq_len=int(raw_config["max_seq_len"]),
        target_steps=int(
            row.get("n_train_steps") or raw_config.get("stage1_steps") or 750
        ),
        checkpoint=_find_persisted_checkpoint(result_id, graph_fingerprint),
        db_path=db_path,
    )


def _base_record(ctx: _OpContext) -> dict[str, Any]:
    return {
        "op": ctx.op_name,
        "source_result_id": ctx.result_id,
        "source_experiment_id": ctx.experiment_id,
        "source_graph_fingerprint": ctx.graph_fingerprint,
        "source_loss_ratio": ctx.loss_ratio,
        "target_train_steps": ctx.target_steps,
        "model_dim": None,
        "checkpoint_persisted": ctx.checkpoint is not None,
        "checkpoint_path": ctx.checkpoint,
    }


def _mark_queued(
    record: dict[str, Any], ctx: _OpContext, init_max: float
) -> dict[str, Any]:
    record["trained_status"] = "queued"
    record["trained"] = None
    record["queued_gpu_command"] = _gpu_replay_command(ctx.result_id, ctx.target_steps)
    record["flag_ge_0_55"] = init_max >= FLAG_THRESHOLD
    record["twin_ge_0_6"] = init_max >= TWIN_THRESHOLD
    return record


def _measure_init(
    ctx: _OpContext, n_probes: int, record: dict[str, Any]
) -> dict[str, Any]:
    model, op_module, dim = _compile_and_locate(
        ctx.graph_json,
        ctx.op_name,
        n_layers=ctx.n_layers,
        vocab_size=ctx.vocab_size,
        max_seq_len=ctx.max_seq_len,
        seed=42,
    )
    record["model_dim"] = dim
    init_scores = _score_multi_seed(op_module, dim=dim, n_probes=n_probes)
    record["init"] = init_scores
    logger.info(
        "  init twin_score mean=%.4f max=%.4f",
        init_scores["twin_score_mean"],
        init_scores["twin_score_max"],
    )
    del model, op_module
    return init_scores


def _calibrate(
    ctx: _OpContext,
    calibration_steps: int,
    cpu_budget_seconds: int,
    record: dict[str, Any],
) -> float:
    t_calib = time.perf_counter()
    sec_per_step = _calibrate_seconds_per_step(
        ctx.graph_json,
        ctx.op_name,
        ctx.raw_config,
        ctx.db_path,
        n_layers=ctx.n_layers,
        vocab_size=ctx.vocab_size,
        max_seq_len=ctx.max_seq_len,
        calibration_steps=calibration_steps,
    )
    projected_seconds = sec_per_step * ctx.target_steps
    record["calibration"] = {
        "calibration_steps": calibration_steps,
        "calibration_wall_seconds": round(time.perf_counter() - t_calib, 2),
        "seconds_per_step": round(sec_per_step, 4),
        "projected_full_train_seconds": round(projected_seconds, 1),
        "cpu_budget_seconds": cpu_budget_seconds,
    }
    logger.info(
        "  calibration: %.3fs/step -> projected %.1fs for %d steps (budget %ds)",
        sec_per_step,
        projected_seconds,
        ctx.target_steps,
        cpu_budget_seconds,
    )
    return projected_seconds


def _train_and_score(
    ctx: _OpContext,
    cpu_budget_seconds: int,
    n_probes: int,
    record: dict[str, Any],
    init_max: float,
) -> dict[str, Any]:
    try:
        with TimeoutGuard(cpu_budget_seconds + 30):
            train_result, op_module, dim = _train_full(
                ctx.graph_json,
                ctx.op_name,
                ctx.raw_config,
                ctx.db_path,
                n_layers=ctx.n_layers,
                vocab_size=ctx.vocab_size,
                max_seq_len=ctx.max_seq_len,
                target_steps=ctx.target_steps,
                seed=42,
            )
    except TimeoutError as exc:
        logger.warning(
            "  %s: full retrain hit the hard cutoff (%s); queuing for GPU",
            ctx.op_name,
            exc,
        )
        return _mark_queued(record, ctx, init_max)

    trained_scores = _score_multi_seed(op_module, dim=dim, n_probes=n_probes)
    record["trained_status"] = "measured"
    record["trained"] = trained_scores
    record["trained_run_passed"] = bool(train_result.get("passed"))
    record["trained_run_loss_ratio"] = train_result.get("loss_ratio")
    logger.info(
        "  trained twin_score mean=%.4f max=%.4f",
        trained_scores["twin_score_mean"],
        trained_scores["twin_score_max"],
    )
    max_seen = max(init_max, trained_scores["twin_score_max"])
    record["flag_ge_0_55"] = max_seen >= FLAG_THRESHOLD
    record["twin_ge_0_6"] = max_seen >= TWIN_THRESHOLD
    return record


def audit_op(
    op_name: str,
    candidate_rows: list[dict[str, Any]],
    db_path: Path,
    *,
    cpu_budget_seconds: int,
    calibration_steps: int,
    n_probes: int,
) -> dict[str, Any]:
    """Audit one op, trying candidate S1-passer rows best-first.

    If the best row's projected CPU retrain exceeds the budget, fall through
    to the next S1-passing graph bearing the op (all candidates are real S1
    passers, so any is a valid trained instance). Only when EVERY candidate
    exceeds the budget is the op queued for GPU — with the best row's exact
    replay command.
    """
    first_record: dict[str, Any] | None = None
    first_ctx: _OpContext | None = None
    first_init_max = 0.0
    for row in candidate_rows:
        ctx = _load_op_context(op_name, row, db_path)
        logger.info(
            "=== %s (source %s / %s) ===",
            ctx.op_name,
            ctx.result_id,
            ctx.experiment_id,
        )
        record = _base_record(ctx)

        if ctx.checkpoint is not None:
            # Deliberately not implemented as generic weight-format loading —
            # no persisted checkpoint exists for any in-scope op as of this
            # audit (verified live via _find_persisted_checkpoint); fail loud
            # rather than guess a loader here.
            raise NotImplementedError(
                f"found a persisted checkpoint for {op_name} at {ctx.checkpoint} but this "
                "tool has no loader wired for it yet — extend _find_persisted_checkpoint's "
                "caller before trusting this path"
            )

        init_scores = _measure_init(ctx, n_probes, record)
        projected_seconds = _calibrate(
            ctx, calibration_steps, cpu_budget_seconds, record
        )
        if first_record is None:
            first_record, first_ctx = record, ctx
            first_init_max = init_scores["twin_score_max"]

        if projected_seconds > cpu_budget_seconds:
            logger.info(
                "  %s: row %s over CPU budget; trying next candidate row",
                op_name,
                ctx.result_id,
            )
            continue

        return _train_and_score(
            ctx, cpu_budget_seconds, n_probes, record, init_scores["twin_score_max"]
        )

    assert first_record is not None and first_ctx is not None  # candidate_rows nonempty
    logger.info("  %s: all candidate rows over CPU budget; queued for GPU", op_name)
    return _mark_queued(first_record, first_ctx, first_init_max)


def audit_ops_gpu(
    ops: tuple[str, ...],
    representative: dict[str, list[dict[str, Any]]],
    db_path: Path,
    *,
    n_probes: int,
) -> list[dict[str, Any]]:
    """GPU audit: one production-config retrain per unique source graph.

    Ops whose best rows share a graph (e.g. integral_control_mixer +
    idempotent_oblique_memory in 3a643630-aa5) are scored from the SAME
    trained model — one GPU job serves all of them. Jobs run strictly
    sequentially (ONE GPU run at a time, per standing rules). No CPU budget
    gate applies; twin scoring always happens on CPU after moving the
    trained model back, so every score in the report uses identical
    detector stimuli.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "audit_ops_gpu requires CUDA; unset CUDA_VISIBLE_DEVICES-empty and retry"
        )
    groups: dict[str, list[str]] = {}
    for op in ops:
        best_row = representative[op][0]
        groups.setdefault(str(best_row["result_id"]), []).append(op)

    records: list[dict[str, Any]] = []
    for group_ops in groups.values():
        records.extend(
            _audit_gpu_group(
                group_ops,
                representative[group_ops[0]][0],
                db_path,
                n_probes=n_probes,
            )
        )
    return records


def _audit_gpu_group(
    group_ops: list[str],
    row: dict[str, Any],
    db_path: Path,
    *,
    n_probes: int,
) -> list[dict[str, Any]]:
    """Train one graph on GPU at production config; score every listed op from it."""
    ctx = _load_op_context(group_ops[0], row, db_path)
    logger.info(
        "=== GPU group: %s (source %s / %s) ===",
        ",".join(group_ops),
        ctx.result_id,
        ctx.experiment_id,
    )
    if ctx.checkpoint is not None:
        raise NotImplementedError(
            f"found a persisted checkpoint at {ctx.checkpoint} but this tool has no "
            "loader wired for it yet"
        )
    model, op_modules, dim = _compile_and_locate_multi(
        ctx.graph_json,
        group_ops,
        n_layers=ctx.n_layers,
        vocab_size=ctx.vocab_size,
        max_seq_len=ctx.max_seq_len,
        seed=42,
    )
    group_records: dict[str, dict[str, Any]] = {}
    for op in group_ops:
        record = _base_record(_load_op_context(op, row, db_path))
        record["model_dim"] = dim
        record["init"] = _score_multi_seed(op_modules[op], dim=dim, n_probes=n_probes)
        logger.info(
            "  %s init twin_score mean=%.4f max=%.4f",
            op,
            record["init"]["twin_score_mean"],
            record["init"]["twin_score_max"],
        )
        group_records[op] = record

    from research.scientist.runner import ExperimentRunner

    config = _build_run_config(
        ctx.raw_config, stage1_steps=ctx.target_steps, device="cuda"
    )
    runner = ExperimentRunner(str(db_path))
    train_result = runner._micro_train(model, config, torch.device("cuda"), seed=42)
    _assert_training_actually_ran(train_result, ctx.target_steps, ",".join(group_ops))
    model.cpu()
    torch.cuda.empty_cache()

    for op in group_ops:
        record = group_records[op]
        trained_scores = _score_multi_seed(op_modules[op], dim=dim, n_probes=n_probes)
        record["trained_status"] = "measured"
        record["trained_device"] = "cuda"
        record["trained"] = trained_scores
        record["trained_run_passed"] = bool(train_result.get("passed"))
        record["trained_run_loss_ratio"] = train_result.get("loss_ratio")
        logger.info(
            "  %s trained twin_score mean=%.4f max=%.4f",
            op,
            trained_scores["twin_score_mean"],
            trained_scores["twin_score_max"],
        )
        max_seen = max(
            record["init"]["twin_score_max"], trained_scores["twin_score_max"]
        )
        record["flag_ge_0_55"] = max_seen >= FLAG_THRESHOLD
        record["twin_ge_0_6"] = max_seen >= TWIN_THRESHOLD
    return [group_records[op] for op in group_ops]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cpu-budget-seconds", type=int, default=600)
    parser.add_argument("--calibration-steps", type=int, default=3)
    parser.add_argument("--n-probes", type=int, default=5)
    parser.add_argument(
        "--op",
        action="append",
        default=None,
        help="Restrict to specific op(s); default: all 8 in scope",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "cpu (default): calibrated budget-gated retrains. cuda: retrain each "
            "op's best S1 row at production config, one GPU job at a time, no "
            "budget gate (use only when the GPU is free — O1 owns it)."
        ),
    )
    args = parser.parse_args()

    ops = tuple(args.op) if args.op else IN_SCOPE_OPS
    for op in ops:
        if op not in IN_SCOPE_OPS:
            raise ValueError(f"{op!r} is not in the O2 scope: {IN_SCOPE_OPS}")

    rows = _inventory(args.db, IN_SCOPE_OPS, IN_SCOPE_EXPERIMENT_IDS)
    logger.info("inventory: %d S1-passing rows bear an in-scope NM op", len(rows))
    representative = _select_representative_rows(rows, ops)
    missing = [op for op in ops if op not in representative]
    if missing:
        raise ValueError(f"no S1-passing row found bearing op(s): {missing}")

    if args.device == "cuda":
        records = audit_ops_gpu(ops, representative, args.db, n_probes=args.n_probes)
    else:
        records = [
            audit_op(
                op,
                representative[op],
                args.db,
                cpu_budget_seconds=args.cpu_budget_seconds,
                calibration_steps=args.calibration_steps,
                n_probes=args.n_probes,
            )
            for op in ops
        ]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector": "component_fab/proposer/algebraic_properties.py:measure_algebraic_properties",
        "flag_threshold": FLAG_THRESHOLD,
        "twin_threshold": TWIN_THRESHOLD,
        "cpu_budget_seconds": args.cpu_budget_seconds,
        "inventory_row_count": len(rows),
        "inventory_rows": [
            {
                "result_id": r["result_id"],
                "experiment_id": r["experiment_id"],
                "graph_fingerprint": r["graph_fingerprint"],
                "loss_ratio": r["loss_ratio"],
                "nm_ops_present": r["nm_ops_present"],
            }
            for r in rows
        ],
        "ops": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("wrote %s", args.out)

    print(
        f"\n{'op':<28}{'init_max':>10}{'trained_max':>13}{'status':>10}{'flag':>7}{'twin':>7}"
    )
    for record in records:
        trained = record.get("trained") or {}
        trained_max = trained.get("twin_score_max")
        print(
            f"{record['op']:<28}"
            f"{record['init']['twin_score_max']:>10.4f}"
            f"{(trained_max if trained_max is not None else float('nan')):>13.4f}"
            f"{record['trained_status']:>10}"
            f"{str(record['flag_ge_0_55']):>7}"
            f"{str(record['twin_ge_0_6']):>7}"
        )


if __name__ == "__main__":
    main()
