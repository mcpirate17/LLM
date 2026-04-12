#!/usr/bin/env python
"""Managed backfill runner for attention template campaigns.

Runs attention templates tier-by-tier, persists progress to JSON, captures
stdout/stderr to a log file, retries incomplete templates, and refreshes
stats/ML models once at the end.

Examples:
    python -m research.tools.attention_template_backfill --tier 1 --device cuda
    python -m research.tools.attention_template_backfill --tier 1 2 --target 20 --min-s1 2
    python -m research.tools.attention_template_backfill --all --device cuda --max-rounds 4
    python -m research.tools.attention_template_backfill --resume
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from research.tools.backfill_templates import DB_PATH, get_template_stats
from research.tools._script_audit import (
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)


_TIER_TEMPLATES: "OrderedDict[str, list[str]]" = OrderedDict(
    {
        "1": [
            "latent_attn_ffn_block",
            "local_attn_ffn_block",
            "latent_attn_sparse_ffn",
            "local_attn_swiglu",
            "attn_three_way_split",
            "latent_attn_moe",
            "latent_attn_conv_hybrid",
            "latent_attn_ssm_hybrid",
        ],
        "2": [
            "diff_attn_ffn_block",
            "diff_attn_conv_hybrid",
            "diff_attn_routing",
            "local_attn_routing",
            "local_attn_moe",
            "local_attn_ssm_hybrid",
            "graph_attn_ffn_block",
            "diff_attn_moe",
            "graph_attn_moe",
            "attn_sparse_moe",
            "attn_spectral_filter",
        ],
        "3": [
            "attn_residual_block",
            "attn_gated_residual",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_state_space_hybrid",
            "dual_attn_block",
            "cascaded_attn_ffn",
            "attn_conditional_compute",
            "diff_attn_gated_ffn",
            "attn_multi_head_mix",
            "attn_cross_dim",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "attn_exp_gated",
            "attn_reciprocal_gated",
            "attn_decay_sequence",
            "attn_gated_product",
            "attn_chebyshev_hybrid",
            "attn_kronecker_hybrid",
            "attn_log_gated",
            "attn_gated_maximum",
            "attn_hyperbolic",
            "attn_normalized_matmul",
            "attn_softmax_normalized_matmul",
            "attn_softmax_normalized_matmul_compact_ffn",
            "attn_softmax_normalized_matmul_fixed_tail_norm",
            "attn_linear_no_matmul_ffn",
            "attn_linear_no_matmul_ffn_dense_tail",
            "attn_linear_no_matmul_ffn_direct_recovery",
            "attn_safe_division",
            "attn_spiking_hybrid",
            "linear_attn_ffn_block",
            "linear_attn_sparse_ffn",
            "graph_attn_sparse_ffn",
        ],
    }
)


def _default_runtime_dir() -> Path:
    return Path("research/runtime/backfill")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _checkpoint_active(
    campaign: dict[str, Any],
    *,
    template: str | None,
    round_idx: int | None,
    command: list[str] | None,
    state_path: Path,
    status: str,
    last_line: str | None = None,
) -> None:
    campaign["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    campaign["active_template"] = template
    campaign["active_round"] = round_idx
    campaign["active_command"] = command or []
    campaign["active_status"] = status
    if last_line is not None:
        campaign["last_subprocess_line"] = last_line
    _write_json(state_path, campaign)


def _template_needs_more(
    stats: dict[str, dict[str, int]],
    template: str,
    target_metric: str,
    target: int,
    min_s1: int,
) -> bool:
    bucket = stats.get(template, {})
    return (
        int(bucket.get(target_metric, 0)) < target or int(bucket.get("s1", 0)) < min_s1
    )


class _InterruptState:
    __slots__ = ("soft_requested", "hard_requested")

    def __init__(self) -> None:
        self.soft_requested = False
        self.hard_requested = False


def _run_one_template(
    *,
    template: str,
    target: int,
    target_metric: str,
    min_s1: int,
    batch_size: int,
    phase: str,
    device: str,
    db: Path,
    weights: str,
    log_path: Path,
    interrupt_state: _InterruptState | None = None,
    on_proc_started: Any = None,
    on_output_line: Any = None,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "research.tools.backfill_templates",
        "--templates",
        template,
        "--target",
        str(target),
        "--target-metric",
        target_metric,
        "--min-s1",
        str(min_s1),
        "--batch-size",
        str(batch_size),
        "--phase",
        phase,
        "--device",
        device,
        "--db",
        str(db),
        "--weights",
        weights,
        "--no-refresh",
    ]
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started}] RUN {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path.cwd(),
            start_new_session=True,
        )
        if on_proc_started is not None:
            on_proc_started(proc.pid)
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            print(f"[sub:{template}] {line.rstrip()}")
            if on_output_line is not None:
                on_output_line(line.rstrip())
            if interrupt_state is not None and interrupt_state.hard_requested:
                break
        proc.wait()
        if on_proc_started is not None:
            on_proc_started(None)
        output = "".join(lines)
        if output and not output.endswith("\n"):
            log.write("\n")
        log.write(f"[exit_code={proc.returncode}] template={template}\n")
        log.flush()
    return proc.returncode, output


def _refresh_models(db: Path, mode: str, log_path: Path) -> None:
    if mode == "none":
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] REFRESH mode={mode}\n")
        log.flush()

        from research.tools.backfill_stats import backfill

        backfill(str(db))
        log.write("Stats backfill complete\n")
        log.flush()

        from research.tools.train_predictors import (
            train_bayesian,
            train_ensemble_full,
            train_graph_predictor,
        )

        train_bayesian(save=True)
        log.write("Bayesian tracker refreshed\n")
        train_graph_predictor(save=True)
        log.write("Graph predictor refreshed\n")
        train_ensemble_full(save=True)
        log.write("Ensemble predictor refreshed\n")
        log.flush()

        if mode == "all":
            from research.tools.train_predictors import (
                train_embeddings,
                train_interaction,
            )

            train_embeddings(save=True)
            log.write("Op embeddings refreshed\n")
            train_interaction(save=True)
            log.write("Interaction model refreshed\n")
            log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Managed backfill runner for attention templates"
    )
    parser.add_argument("--tier", nargs="*", choices=["1", "2", "3"], default=None)
    parser.add_argument("--all", action="store_true", help="Run all tiers")
    parser.add_argument("--resume", action="store_true", help="Reuse state/log paths")
    parser.add_argument("--list", action="store_true", help="List tier contents")
    parser.add_argument("--target", type=int, default=3)
    parser.add_argument(
        "--target-metric",
        choices=["eval", "s0", "s1"],
        default="s1",
        help="Coverage target for each template",
    )
    parser.add_argument(
        "--min-s1",
        type=int,
        default=3,
        help="Optional minimum S1 survivors required for each template",
    )
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--phase",
        choices=["isolation", "stack"],
        default="isolation",
        help="Backfill phase: isolation for clean rehab evidence, stack for survivability validation",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--weights",
        choices=["uniform", "random", "default", "scaffold_guided"],
        default="uniform",
    )
    parser.add_argument(
        "--refresh",
        choices=["none", "light", "all"],
        default="light",
        help="Refresh stats/models after the campaign",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the campaign on the first non-zero subprocess exit",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    args = parser.parse_args()

    if args.list:
        for tier, templates in _TIER_TEMPLATES.items():
            print(f"Tier {tier} ({len(templates)} templates)")
            for name in templates:
                print(f"  {name}")
        return

    tiers = list(_TIER_TEMPLATES.keys()) if args.all or not args.tier else args.tier
    templates = [tpl for tier in tiers for tpl in _TIER_TEMPLATES[tier]]

    runtime_dir = _default_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        state_path = args.state_file or (runtime_dir / "attention_backfill_state.json")
        state = _read_json(state_path)
        log_path = args.log_file or Path(
            state.get("log_file") or (runtime_dir / "attention_backfill.log")
        )
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = args.log_file or (runtime_dir / f"attention_backfill_{ts}.log")
        state_path = args.state_file or (runtime_dir / "attention_backfill_state.json")
        state = {}

    campaign = {
        "started_at": state.get("started_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "tiers": tiers,
        "target": args.target,
        "target_metric": args.target_metric,
        "min_s1": args.min_s1,
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "phase": args.phase,
        "weights": args.weights,
        "db": str(args.db),
        "log_file": str(log_path),
        "templates": state.get("templates") or {},
    }
    interrupt_state = _InterruptState()
    active_proc: dict[str, Any] = {"pid": None}
    active_context: dict[str, Any] = {
        "template": None,
        "round_idx": None,
        "command": [],
    }
    audit_nb, exp_id = start_script_experiment(
        db_path=args.db,
        experiment_type="attention_template_backfill",
        config={
            "tiers": tiers,
            "target": args.target,
            "target_metric": args.target_metric,
            "min_s1": args.min_s1,
            "batch_size": args.batch_size,
            "max_rounds": args.max_rounds,
            "phase": args.phase,
            "device": args.device,
            "weights": args.weights,
            "refresh": args.refresh,
            "stop_on_error": bool(args.stop_on_error),
            "resume": bool(args.resume),
            "log_file": log_path,
            "state_file": state_path,
        },
        source_script="attention_template_backfill",
        hypothesis="Managed attention template backfill campaign",
    )

    def _handle_sigint(_signum: int, _frame: Any) -> None:
        if not interrupt_state.soft_requested:
            interrupt_state.soft_requested = True
            note = (
                "Soft stop requested by operator; finishing current template before stopping. "
                "Press Ctrl-C again for immediate interrupt."
            )
            print(note)
            _checkpoint_active(
                campaign,
                template=active_context["template"],
                round_idx=active_context["round_idx"],
                command=active_context["command"],
                state_path=state_path,
                status="stopping_after_current",
                last_line=note,
            )
            return

        interrupt_state.hard_requested = True
        note = (
            "Hard stop requested by operator; interrupting current template subprocess."
        )
        print(note)
        _checkpoint_active(
            campaign,
            template=active_context["template"],
            round_idx=active_context["round_idx"],
            command=active_context["command"],
            state_path=state_path,
            status="interrupting",
            last_line=note,
        )
        pid = active_proc.get("pid")
        if pid:
            try:
                os.killpg(pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    print(
        f"Running {len(templates)} attention templates "
        f"(target {args.target_metric}>={args.target}, min_s1={args.min_s1}, phase={args.phase}, device={args.device})"
    )
    print(f"Log:   {log_path}")
    print(f"State: {state_path}")

    stats = get_template_stats(args.db)
    for template in templates:
        tpl_state = campaign["templates"].setdefault(
            template,
            {
                "status": "pending",
                "rounds": 0,
                "history": [],
            },
        )
        if not _template_needs_more(
            stats, template, args.target_metric, args.target, args.min_s1
        ):
            tpl_state["status"] = "complete"
            tpl_state["last_stats"] = stats.get(template, {"eval": 0, "s0": 0, "s1": 0})

    campaign["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(state_path, campaign)

    try:
        for template in templates:
            tpl_state = campaign["templates"][template]
            if tpl_state.get("status") == "complete":
                print(f"[skip] {template}: already complete")
                continue

            while tpl_state["rounds"] < args.max_rounds:
                pre_stats = get_template_stats(args.db)
                if not _template_needs_more(
                    pre_stats, template, args.target_metric, args.target, args.min_s1
                ):
                    tpl_state["status"] = "complete"
                    tpl_state["last_stats"] = pre_stats.get(
                        template, {"eval": 0, "s0": 0, "s1": 0}
                    )
                    break

                round_idx = tpl_state["rounds"] + 1
                print(
                    f"[run] {template} round {round_idx}/{args.max_rounds} "
                    f"pre={pre_stats.get(template, {'eval': 0, 's0': 0, 's1': 0})}"
                )
                cmd_preview = [
                    sys.executable,
                    "-u",
                    "-m",
                    "research.tools.backfill_templates",
                    "--templates",
                    template,
                    "--target",
                    str(args.target),
                    "--target-metric",
                    args.target_metric,
                    "--min-s1",
                    str(args.min_s1),
                    "--batch-size",
                    str(args.batch_size),
                    "--phase",
                    args.phase,
                    "--device",
                    args.device,
                    "--db",
                    str(args.db),
                    "--weights",
                    args.weights,
                    "--no-refresh",
                ]
                tpl_state["status"] = "running"
                active_context["template"] = template
                active_context["round_idx"] = round_idx
                active_context["command"] = cmd_preview
                _checkpoint_active(
                    campaign,
                    template=template,
                    round_idx=round_idx,
                    command=cmd_preview,
                    state_path=state_path,
                    status="running",
                )

                def _on_output_line(line: str) -> None:
                    if line:
                        _checkpoint_active(
                            campaign,
                            template=template,
                            round_idx=round_idx,
                            command=cmd_preview,
                            state_path=state_path,
                            status="running",
                            last_line=line,
                        )

                rc, output = _run_one_template(
                    template=template,
                    target=args.target,
                    target_metric=args.target_metric,
                    min_s1=args.min_s1,
                    batch_size=args.batch_size,
                    phase=args.phase,
                    device=args.device,
                    db=args.db,
                    weights=args.weights,
                    log_path=log_path,
                    interrupt_state=interrupt_state,
                    on_proc_started=lambda pid: active_proc.__setitem__("pid", pid),
                    on_output_line=_on_output_line,
                )
                active_proc["pid"] = None
                post_stats = get_template_stats(args.db)
                tpl_state["rounds"] = round_idx
                tpl_state["last_stats"] = post_stats.get(
                    template, {"eval": 0, "s0": 0, "s1": 0}
                )
                tpl_state["history"].append(
                    {
                        "round": round_idx,
                        "returncode": rc,
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "pre_stats": pre_stats.get(
                            template, {"eval": 0, "s0": 0, "s1": 0}
                        ),
                        "post_stats": tpl_state["last_stats"],
                        "saw_traceback": "Traceback" in output,
                        "saw_zero_graphs": "generated 0 graphs" in output,
                        "saw_deleted_zero_value": "Deleted zero-value failed experiment"
                        in output,
                    }
                )

                if rc != 0:
                    interrupted = rc < 0 or "KeyboardInterrupt" in output
                    tpl_state["status"] = "incomplete" if interrupted else "error"
                    tpl_state["last_error"] = f"subprocess exit code {rc}"
                    _checkpoint_active(
                        campaign,
                        template=None,
                        round_idx=None,
                        command=None,
                        state_path=state_path,
                        status="interrupted" if interrupted else "idle",
                    )
                    print(
                        f"[{'interrupt' if interrupted else 'error'}] {template}: exit code {rc}"
                    )
                    if interrupted:
                        raise KeyboardInterrupt
                    if args.stop_on_error:
                        raise SystemExit(rc)
                    break

                if interrupt_state.soft_requested:
                    tpl_state["status"] = (
                        "complete"
                        if not _template_needs_more(
                            post_stats,
                            template,
                            args.target_metric,
                            args.target,
                            args.min_s1,
                        )
                        else "incomplete"
                    )
                    _checkpoint_active(
                        campaign,
                        template=None,
                        round_idx=None,
                        command=None,
                        state_path=state_path,
                        status="interrupted",
                        last_line="Campaign soft-stopped after current subprocess",
                    )
                    break

                if _template_needs_more(
                    post_stats, template, args.target_metric, args.target, args.min_s1
                ):
                    tpl_state["status"] = "incomplete"
                    print(f"[retry] {template}: post={tpl_state['last_stats']}")
                    _checkpoint_active(
                        campaign,
                        template=None,
                        round_idx=None,
                        command=None,
                        state_path=state_path,
                        status="idle",
                    )
                    continue

                tpl_state["status"] = "complete"
                print(f"[done] {template}: post={tpl_state['last_stats']}")
                _checkpoint_active(
                    campaign,
                    template=None,
                    round_idx=None,
                    command=None,
                    state_path=state_path,
                    status="idle",
                )
                break

            if interrupt_state.soft_requested:
                tpl_state["status"] = (
                    "complete"
                    if tpl_state.get("status") == "complete"
                    else "incomplete"
                )
                campaign["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _checkpoint_active(
                    campaign,
                    template=None,
                    round_idx=None,
                    command=None,
                    state_path=state_path,
                    status="interrupted",
                    last_line="Campaign soft-stopped after current template",
                )
                _write_json(state_path, campaign)
                break

            if tpl_state.get("status") not in {"complete", "error"}:
                tpl_state["status"] = "incomplete"
            campaign["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_json(state_path, campaign)
    except KeyboardInterrupt:
        fail_script_experiment(
            audit_nb,
            exp_id,
            error="KeyboardInterrupt",
            results={"templates": len(templates)},
        )
        _checkpoint_active(
            campaign,
            template=None,
            round_idx=None,
            command=None,
            state_path=state_path,
            status="interrupted",
            last_line="Campaign interrupted by operator",
        )
        raise
    except Exception as exc:
        fail_script_experiment(
            audit_nb,
            exp_id,
            error=str(exc),
            results={"templates": len(templates)},
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    if interrupt_state.soft_requested:
        print("Campaign stopped after current template. Refresh skipped.")
        fail_script_experiment(
            audit_nb,
            exp_id,
            error="Soft stop requested by operator",
            results={"templates": len(templates)},
        )
        audit_nb.close()
        return

    _refresh_models(args.db, args.refresh, log_path)
    campaign["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(state_path, campaign)

    n_complete = sum(
        1 for data in campaign["templates"].values() if data.get("status") == "complete"
    )
    n_incomplete = sum(
        1
        for data in campaign["templates"].values()
        if data.get("status") == "incomplete"
    )
    n_error = sum(
        1 for data in campaign["templates"].values() if data.get("status") == "error"
    )
    print(
        f"Campaign complete: complete={n_complete} incomplete={n_incomplete} error={n_error}"
    )
    complete_script_experiment(
        audit_nb,
        exp_id,
        results={
            "templates": len(templates),
            "complete": n_complete,
            "incomplete": n_incomplete,
            "error": n_error,
            "refresh": args.refresh,
        },
        summary=(
            f"Attention backfill campaign complete: complete={n_complete} "
            f"incomplete={n_incomplete} error={n_error}"
        ),
    )
    audit_nb.close()


if __name__ == "__main__":
    main()
