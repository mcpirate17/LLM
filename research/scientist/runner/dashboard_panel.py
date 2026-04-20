"""Dashboard mixin: read-side data feed for the React dashboard and the
fixed-budget routing benchmark."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from ._helpers import clear_gpu_memory
from ._types import RunConfig

logger = logging.getLogger(__name__)


_ROUTING_BENCH_FIXED_BASE: Dict[str, str] = {
    "token_representation": "dense_float",
    "weight_storage": "dense_matrix",
    "token_mixing": "softmax_attention",
    "channel_mixing": "swiglu_mlp",
    "topology": "sequential",
    "normalization": "rmsnorm_pre",
    "positional_encoding": "rope",
}


class _DashboardPanelMixin:
    """Dashboard data reads + routing benchmark."""

    def get_dashboard_data(self) -> Dict:
        """Get all data needed for the React dashboard."""
        nb = self._make_notebook()
        try:
            return {
                "aria": self.aria.get_status(),
                "summary": nb.get_dashboard_summary(),
                "recent_experiments": nb.get_recent_experiments(20),
                "top_programs": nb.get_top_programs(20),
                "insights": nb.get_insights(limit=20),
                "recent_entries": nb.get_entries(limit=30),
                "is_running": self.is_running,
                "progress": self.progress.to_dict(),
            }
        finally:
            nb.close()

    def get_live_loss_curve(self) -> List[Dict]:
        """Return the in-memory training loss curve for the current/last training run."""
        return list(self._live_loss_curve)

    # ── Routing benchmark ────────────────────────────────────────────────

    def _routing_benchmark_config(self, config: RunConfig) -> RunConfig:
        bench_config = config.copy()
        if bench_config.stage1_steps <= 0:
            bench_config.stage1_steps = 1
        bench_config.profile_disable_post_eval = True
        bench_config.stage1_compute_val_loss = False
        bench_config.stage1_compute_discovery_loss = False
        return bench_config

    def _run_routing_benchmark_iteration(
        self,
        routing_mode: str,
        seed: int,
        bench_config: RunConfig,
        dev: torch.device,
    ) -> Dict[str, Any]:
        """Run one (routing_mode, seed) trial and return the raw run record."""
        from ...morphological_box import roll
        from ...arch_builder import build_model, BuildConfig

        run_data: Dict[str, Any] = {
            "routing_mode": routing_mode,
            "seed": int(seed),
            "status": "ok",
        }
        try:
            fixed = {**_ROUTING_BENCH_FIXED_BASE, "compute_routing": routing_mode}
            spec = roll(seed=int(seed), fixed=fixed)
            model = build_model(
                spec,
                BuildConfig(
                    dim=int(bench_config.model_dim),
                    n_layers=int(bench_config.n_layers),
                    vocab_size=int(bench_config.vocab_size),
                    max_seq_len=int(bench_config.max_seq_len),
                ),
            )
            train_result = self._micro_train(
                model=model,
                config=bench_config,
                dev=dev,
                seed=int(seed),
            )

            seq_len = min(128, int(bench_config.max_seq_len))
            n_steps = int(
                train_result.get("n_train_steps") or bench_config.stage1_steps
            )
            tokens_total = int(bench_config.stage1_batch_size) * seq_len * n_steps
            eff_factor = float(self._ROUTING_EFFICIENCY_FACTOR.get(routing_mode, 1.0))

            run_data.update(
                {
                    "validation_loss": train_result.get("final_loss"),
                    "tokens_per_sec": train_result.get("throughput"),
                    "routing_stability": self._routing_stability_from_curve(
                        train_result.get("training_curve") or []
                    ),
                    "tokens_total": tokens_total,
                    "effective_token_compute": tokens_total * eff_factor,
                    "loss_ratio": train_result.get("loss_ratio"),
                }
            )
            del model
            clear_gpu_memory()
        except (
            Exception
        ) as exc:  # top-level error boundary: benchmark run must not crash loop
            logger.debug("Routing benchmark run failed: %s", exc)
            run_data["status"] = "error"
            run_data["error"] = str(exc)
        return run_data

    @staticmethod
    def _summarize_routing_points(
        raw_runs: List[Dict[str, Any]],
        modes: List[str],
    ) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for routing_mode in modes:
            mode_runs = [
                row
                for row in raw_runs
                if row.get("routing_mode") == routing_mode and row.get("status") == "ok"
            ]
            if not mode_runs:
                continue

            def _mean(key: str) -> Optional[float]:
                vals = [float(r[key]) for r in mode_runs if r.get(key) is not None]
                return (sum(vals) / len(vals)) if vals else None

            points.append(
                {
                    "routing_mode": routing_mode,
                    "n_runs": len(mode_runs),
                    "validation_loss": _mean("validation_loss"),
                    "tokens_per_sec": _mean("tokens_per_sec"),
                    "effective_token_compute": _mean("effective_token_compute"),
                    "routing_stability": _mean("routing_stability"),
                }
            )
        return points

    def run_routing_benchmark(
        self,
        config: RunConfig,
        seed_set: Optional[List[int]] = None,
        modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare routing modes on identical skeleton, seed set, and budget."""
        requested_modes = modes or list(self._ROUTING_BENCHMARK_MODES)
        supported_modes = [
            m for m in requested_modes if m in self._ROUTING_BENCHMARK_MODES
        ]
        seeds = seed_set or [101, 202, 303]
        if not supported_modes:
            return {
                "available": False,
                "reason": "No supported routing modes requested",
                "modes_requested": requested_modes,
                "seed_set": seeds,
                "points": [],
                "raw_runs": [],
            }

        dev_str = config.device
        if dev_str == "cuda" and not torch.cuda.is_available():
            dev_str = "cpu"
        dev = torch.device(dev_str)
        bench_config = self._routing_benchmark_config(config)

        raw_runs: List[Dict[str, Any]] = []
        for routing_mode in supported_modes:
            for seed in seeds:
                if self._stop_event.is_set():
                    break
                raw_runs.append(
                    self._run_routing_benchmark_iteration(
                        routing_mode, int(seed), bench_config, dev
                    )
                )

        points = self._summarize_routing_points(raw_runs, supported_modes)
        return {
            "available": len(points) > 0,
            "seed_set": seeds,
            "modes_requested": requested_modes,
            "modes_evaluated": [p["routing_mode"] for p in points],
            "points": points,
            "raw_runs": raw_runs,
            "benchmark_config": {
                "stage1_steps": int(bench_config.stage1_steps),
                "stage1_batch_size": int(bench_config.stage1_batch_size),
                "max_seq_len": int(bench_config.max_seq_len),
                "data_mode": str(bench_config.data_mode),
            },
        }
