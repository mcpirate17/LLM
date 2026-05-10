"""Branching binding/induction/mixing scale experiment.

This runner is intentionally append-only: it reads the lab notebook and prior
AR validation CSVs, trains selected graph fingerprints at ~100M scale, checkpoints
every N steps, branches the strongest 5K trunks, and records checkpoint probes
to CSV/JSONL under ``research/runtime/bim_scale_experiment``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE
from research.eval.blimp_eval import evaluate_blimp
from research.eval.binding_intermediate_probe import run_binding_intermediate
from research.eval.language_control_probe import language_control_probe
from research.eval.hellaswag_eval import screening_hellaswag_eval
from research.eval.induction_intermediate_probe import run_induction_intermediate
from research.eval.induction_validation_probe import run_induction_validation_champion
from research.eval.ar_gate import ARGateConfig, ar_gate
from research.eval.native_induction import (
    induction_result_metadata,
    induction_score_gold,
)
from research.eval.ar_validation import (
    ARValidationConfig,
    run_ar_validation,
)
from research.eval.utils import (
    clip_grad_norm,
    compute_perplexity,
    language_model_loss,
    make_adamw,
    make_batches,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.shared_utils import coerce_finite_float as _safe_float
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.synthesis.compiled_model import CompiledLayer
from research.synthesis.serializer import graph_from_json


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
RUNTIME_ROOT = ROOT / "research" / "runtime" / "bim_scale_experiment"
TRAIN_TOKENS = ROOT / "research" / "corpus" / "wikitext103_train.npy"
VAL_TOKENS = ROOT / "research" / "corpus" / "wikitext103_val.npy"
EASY25_CSVS = (
    ROOT
    / "research"
    / "runtime"
    / "ar_validation_fingerprint_sweep"
    / "top_continuing_cuda_5k_pretrain_5k_ar_validation.csv",
    ROOT
    / "research"
    / "runtime"
    / "ar_validation_fingerprint_sweep"
    / "top50_cuda_5k_pretrain_5k_ar_validation.csv",
    ROOT
    / "research"
    / "runtime"
    / "ar_validation_fingerprint_sweep"
    / "ar_validation_fp_sweep_20260507T221610_offset0000_limit0050.csv",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    fingerprint: str
    result_id: str
    label: str
    d_model: int
    core_dim: int
    rationale: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "0cffa5cff90c3bc5",
        "80805639-fc0",
        "official_easy25_winner",
        1024,
        256,
        "Best official easy25 score in the 5K-pretrain continuing sweep.",
    ),
    Candidate(
        "ce47c80b4d581606",
        "10fc2e87-301",
        "raw_easy25_curve5k_winner",
        1024,
        256,
        "Best raw easy25 curve at 5K and the model currently being revalidated.",
    ),
    Candidate(
        "f86a6903d32c4ab6",
        "ec7025d7-338",
        "old_induction_binding_aggregate",
        1024,
        256,
        "Strong prior induction/binding aggregate with good loss ratio.",
    ),
    Candidate(
        "5d7c2086c4a4e06f",
        "0aabe135-e1a",
        "prior_champion_family",
        1024,
        256,
        "Prior champion family with broad nano/control-language performance.",
    ),
    Candidate(
        "c26893ddbd228ddd",
        "e197f295-d8b",
        "top_continuing_curve5k_held_pair",
        1024,
        256,
        "Best raw held-pair result at 5K inside the continuing sweep.",
    ),
)


CSV_FIELDS = (
    "run_id",
    "timestamp_utc",
    "elapsed_s",
    "event",
    "phase",
    "branch",
    "fingerprint",
    "result_id",
    "label",
    "rationale",
    "d_model",
    "core_model_dim",
    "n_layers",
    "vocab_size",
    "max_seq_len",
    "param_count",
    "step",
    "branch_step",
    "train_loss",
    "train_loss_avg100",
    "val_ppl",
    "lr",
    "grad_norm",
    "checkpoint_path",
    "status",
    "error",
    "selection_score",
    "historical_easy25_score",
    "historical_easy25_final_acc",
    "historical_easy25_held_pair_acc",
    "historical_easy25_held_class_acc",
    "historical_easy25_curve5000_held_pair_acc",
    "historical_validation_loss_ratio",
    "historical_wikitext_perplexity",
    "historical_hellaswag_acc",
    "historical_blimp_overall_accuracy",
    "historical_induction_screening_auc",
    "historical_binding_screening_auc",
    "historical_induction_intermediate_auc",
    "historical_binding_intermediate_auc",
    "historical_induction_validation_auc",
    "historical_ar_validation_rank_score",
    "historical_ar_gate_score",
    "historical_language_control_s10_sentence_assoc_score",
    "induction_screening_auc",
    "induction_gap_accuracies",
    "induction_screening_elapsed_ms",
    "ar_gate_in_dist_pair_acc",
    "ar_gate_in_dist_class_acc",
    "ar_gate_held_pair_acc",
    "ar_gate_held_class_acc",
    "ar_gate_status",
    "ar_gate_elapsed_ms",
    "language_control_s10_sentence_assoc_score",
    "language_control_s10_binding_order_acc",
    "language_control_s10_binding_score",
    "language_control_status",
    "language_control_elapsed_ms",
    "ar_validation_rank_score",
    "ar_validation_final_acc",
    "ar_validation_held_pair_acc",
    "ar_validation_held_class_acc",
    "ar_validation_steps_to_floor",
    "ar_validation_status",
    "ar_validation_elapsed_ms",
    "hellaswag_acc",
    "hellaswag_status",
    "hellaswag_n_examples",
    "blimp_overall_accuracy",
    "blimp_status",
    "blimp_n_examples",
    "induction_intermediate_auc",
    "induction_intermediate_status",
    "binding_intermediate_auc",
    "binding_intermediate_status",
    "induction_validation_auc",
    "induction_validation_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _csv_cell(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def append_row(csv_path: Path, jsonl_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: _csv_cell(row.get(key)) for key in CSV_FIELDS})
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def load_easy25_history() -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for path in EASY25_CSVS:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                fp = row.get("fingerprint")
                if not fp:
                    continue
                score = _safe_float(row.get("score"))
                existing = best.get(fp)
                existing_score = (
                    _safe_float(existing.get("score")) if existing else None
                )
                if existing is None or (
                    score is not None
                    and (existing_score is None or score > existing_score)
                ):
                    row = dict(row)
                    row["_source_csv"] = str(path)
                    best[fp] = row
    return best


def fetch_candidate_record(
    conn: sqlite3.Connection, candidate: Candidate
) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT *
        FROM program_results
        WHERE graph_fingerprint = ?
        ORDER BY
            CASE WHEN result_id = ? THEN 0 ELSE 1 END,
            CASE WHEN validation_loss_ratio IS NULL THEN 1 ELSE 0 END,
            validation_loss_ratio ASC,
            timestamp DESC
        LIMIT 1
        """,
        (candidate.fingerprint, candidate.result_id),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"no program_results row for {candidate.fingerprint}")
    payload = dict(row)
    payload["graph_json"] = resolve_graph_json_value(
        conn, DB_PATH, payload["graph_json"]
    )
    return payload


def historical_metrics(
    db_row: dict[str, Any],
    easy25_row: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = {
        "historical_validation_loss_ratio": db_row.get("validation_loss_ratio"),
        "historical_wikitext_perplexity": db_row.get("wikitext_perplexity"),
        "historical_hellaswag_acc": db_row.get("hellaswag_acc"),
        "historical_blimp_overall_accuracy": db_row.get("blimp_overall_accuracy"),
        "historical_induction_screening_auc": db_row.get("induction_screening_auc"),
        "historical_binding_screening_auc": db_row.get("binding_screening_auc"),
        "historical_induction_intermediate_auc": db_row.get(
            "induction_intermediate_auc"
        ),
        "historical_binding_intermediate_auc": db_row.get("binding_intermediate_auc"),
        "historical_induction_validation_auc": db_row.get("induction_validation_auc"),
        "historical_ar_validation_rank_score": db_row.get("ar_validation_rank_score"),
        "historical_ar_gate_score": db_row.get("ar_gate_score"),
        "historical_language_control_s10_sentence_assoc_score": db_row.get(
            "language_control_s10_sentence_assoc_score"
        ),
    }
    if easy25_row:
        metrics.update(
            {
                "historical_easy25_score": _safe_float(easy25_row.get("score")),
                "historical_easy25_final_acc": _safe_float(easy25_row.get("final_acc")),
                "historical_easy25_held_pair_acc": _safe_float(
                    easy25_row.get("held_pair_acc")
                ),
                "historical_easy25_held_class_acc": _safe_float(
                    easy25_row.get("held_class_acc")
                ),
                "historical_easy25_curve5000_held_pair_acc": _safe_float(
                    easy25_row.get("curve5000_held_pair_acc")
                ),
            }
        )
    return metrics


class ScaledGraphCoreLM(nn.Module):
    """Large LM shell around a 256-wide fingerprint graph core.

    These fingerprints were discovered with graph-internal 256-wide tensor
    shapes. Forcing their internal width larger breaks ops that intentionally
    produce seq-by-seq tensors. This wrapper scales the surrounding language
    model while preserving the graph core where it is valid.
    """

    def __init__(
        self,
        graph_json: str,
        *,
        core_dim: int,
        outer_dim: int,
        n_layers: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.model_dim = int(outer_dim)
        self.core_dim = int(core_dim)
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, outer_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=outer_dim**-0.5)
        graphs = [
            graph_from_json(graph_json, model_dim=core_dim) for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList([CompiledLayer(graph) for graph in graphs])
        self.layer_needs_residual = [not graph.has_residual_path() for graph in graphs]
        self.in_proj = nn.Linear(outer_dim, core_dim, bias=False)
        self.core_norm = nn.LayerNorm(core_dim)
        self.out_proj = nn.Linear(core_dim, outer_dim, bias=False)
        self.norm = nn.LayerNorm(outer_dim)
        self.lm_head = nn.Linear(outer_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._routing_progress = 1.0
        self.set_routing_progress(1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outer = self.embed(input_ids)
        x = self.in_proj(outer)
        for i, layer in enumerate(self.layers):
            if self.layer_needs_residual[i]:
                out = layer(x)
                x = x + out if out.shape == x.shape else out
            else:
                x = layer(x)
        mixed = self.out_proj(self.core_norm(x))
        return self.lm_head(self.norm(outer + mixed))

    def set_routing_progress(self, progress: float) -> None:
        clipped = float(max(0.0, min(1.0, progress)))
        self._routing_progress = clipped
        for layer in self.layers:
            if hasattr(layer, "set_routing_progress"):
                layer.set_routing_progress(clipped)

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        for layer in self.layers:
            if hasattr(layer, "set_capture_heatmap"):
                layer.set_capture_heatmap(enabled)


def build_model(
    graph_json: str,
    d_model: int,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    *,
    core_dim: int | None = None,
    scaled_shell: bool = True,
) -> nn.Module:
    del max_seq_len
    if scaled_shell:
        return ScaledGraphCoreLM(
            graph_json,
            core_dim=int(core_dim or 256),
            outer_dim=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
        )
    graph = graph_from_json(graph_json, model_dim=d_model)
    model = compile_model(
        [graph_from_json(graph_json, model_dim=d_model) for _ in range(n_layers)],
        vocab_size=vocab_size,
        max_seq_len=256,
    )
    if graph.model_dim != d_model:
        raise RuntimeError("graph deserialization ignored requested model_dim")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def make_binding_mix_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    lo = 1000
    hi = min(int(vocab_size), 30000)
    x = torch.randint(lo, hi, (batch_size, seq_len), device=device, generator=generator)
    gaps = (8, 16, 32, 64)
    for row in range(batch_size):
        gap = gaps[
            int(
                torch.randint(
                    0, len(gaps), (), device=device, generator=generator
                ).item()
            )
        ]
        stride = gap + 6
        for pos in range(0, max(1, seq_len - stride), stride):
            key = int(
                torch.randint(lo, hi, (), device=device, generator=generator).item()
            )
            val = int(
                torch.randint(lo, hi, (), device=device, generator=generator).item()
            )
            x[row, pos] = key
            x[row, pos + 1] = val
            x[row, pos + gap + 2] = key
            x[row, pos + gap + 3] = val
    return x


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    train_tokens: np.ndarray,
    device: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_steps: int,
    start_step: int,
    lr: float,
    warmup_steps: int,
    clip_grad: float,
    seed: int,
    branch: str,
    autocast: bool,
    mix_every: int,
    mod_token_ids: bool,
) -> tuple[int, float | None, float | None, float | None, float, str | None]:
    model.train()
    loss_window: list[float] = []
    last_loss: float | None = None
    last_grad: float | None = None
    batches = make_batches(
        train_tokens,
        batch_size,
        seq_len,
        n_steps,
        device,
        seed=seed + start_step,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 100_000 + start_step)
    t0 = time.perf_counter()
    autocast_enabled = bool(autocast and str(device).startswith("cuda"))
    for offset, batch in enumerate(batches, start=1):
        step = start_step + offset
        for group in optimizer.param_groups:
            scale = min(1.0, step / max(1, warmup_steps))
            group["lr"] = lr * scale
        if mix_every > 0 and branch == "capability_mix" and step % mix_every == 0:
            batch = make_binding_mix_batch(
                batch_size, seq_len, vocab_size, device, generator
            )
        elif mod_token_ids:
            batch = torch.remainder(batch, vocab_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(batch)
            loss = language_model_loss(logits, batch, vocab_size)
        if not torch.isfinite(loss):
            return (
                step,
                last_loss,
                _mean_tail(loss_window),
                last_grad,
                time.perf_counter() - t0,
                "non_finite_loss",
            )
        loss.backward()
        grad = clip_grad_norm(model.parameters(), clip_grad)
        optimizer.step()
        last_loss = float(loss.detach().item())
        last_grad = (
            float(grad.detach().item()) if torch.is_tensor(grad) else float(grad)
        )
        loss_window.append(last_loss)
        if len(loss_window) > 100:
            loss_window.pop(0)
    return (
        start_step + n_steps,
        last_loss,
        _mean_tail(loss_window),
        last_grad,
        time.perf_counter() - t0,
        None,
    )


def _mean_tail(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def normalize_token_batches(
    batches: list[torch.Tensor],
    vocab_size: int,
    *,
    mod_token_ids: bool,
) -> list[torch.Tensor]:
    if not mod_token_ids:
        return batches
    return [torch.remainder(batch, vocab_size) for batch in batches]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    candidate: Candidate,
    args: argparse.Namespace,
    step: int,
    branch: str,
    param_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "candidate": asdict(candidate),
        "args": vars(args),
        "step": step,
        "branch": branch,
        "param_count": param_count,
        "saved_at_utc": _utc_now(),
    }
    torch.save(payload, path)


def load_checkpoint_weights(
    model: nn.Module, checkpoint_path: Path, device: str
) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=False)


def run_quick_probes(
    model: nn.Module,
    *,
    device: str,
    seed: int,
    include_ar_validation: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        result = induction_score_gold(model, device=device, seed=seed)
        out.update(induction_result_metadata(result))
    except Exception as exc:  # pragma: no cover - defensive experiment logging
        out["induction_status"] = f"failed:{type(exc).__name__}"
        out["induction_error"] = str(exc)
    try:
        result = ar_gate(
            model=model,
            cfg=ARGateConfig(
                seed=seed,
                finetune_steps=200,
                batch_size=24,
                timeout_s=45,
                from_s1=True,
            ),
            device=device,
        )
        out.update(result.to_dict())
    except Exception as exc:  # pragma: no cover
        out["ar_gate_status"] = f"failed:{type(exc).__name__}"
        out["ar_gate_error"] = str(exc)
    try:
        result = language_control_probe(
            model,
            active_vocab_size=80,
            n_train_steps=20,
            eval_repeats=8,
            batch_size=32,
            device=device,
            seed=seed,
            timeout_s=45,
            preserve_state=True,
        )
        out.update(result.to_dict())
    except Exception as exc:  # pragma: no cover
        out["language_control_status"] = f"failed:{type(exc).__name__}"
        out["language_control_error"] = str(exc)
    if include_ar_validation:
        try:
            result = run_ar_validation(
                model,
                cfg=ARValidationConfig(
                    seed=seed,
                    train_steps=1000,
                    eval_every=500,
                    batch_size=12,
                    n_eval=128,
                    timeout_s=180,
                    copy_model=True,
                ),
                device=device,
            )
            out.update(result.to_dict())
        except Exception as exc:  # pragma: no cover
            out["ar_validation_status"] = f"failed:{type(exc).__name__}"
            out["ar_validation_error"] = str(exc)
    out["quick_probes_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run_final_probes(
    model: nn.Module, *, device: str, seed: int, vocab_size: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out.update(screening_hellaswag_eval(model, vocab_size, device, n_examples=50))
    except Exception as exc:  # pragma: no cover
        out["hellaswag_status"] = f"failed:{type(exc).__name__}"
        out["hellaswag_error"] = str(exc)
    try:
        out.update(
            evaluate_blimp(
                model,
                vocab_size=vocab_size,
                device=device,
                n_per_subtask=10,
                timeout_s=90,
            ).to_dict()
        )
    except Exception as exc:  # pragma: no cover
        out["blimp_status"] = f"failed:{type(exc).__name__}"
        out["blimp_error"] = str(exc)
    try:
        out.update(
            run_induction_intermediate(
                model,
                seeds=(seed,),
                n_train_steps=500,
                n_eval=100,
                batch_size=16,
                device=device,
                timeout_s=90,
            ).to_dict()
        )
    except Exception as exc:  # pragma: no cover
        out["induction_intermediate_status"] = f"failed:{type(exc).__name__}"
        out["induction_intermediate_error"] = str(exc)
    try:
        out.update(
            run_binding_intermediate(
                model,
                seeds=(seed,),
                n_train_steps=600,
                n_eval=100,
                train_batch_size=16,
                eval_batch_size=32,
                device=device,
                timeout_s=120,
            ).to_dict()
        )
    except Exception as exc:  # pragma: no cover
        out["binding_intermediate_status"] = f"failed:{type(exc).__name__}"
        out["binding_intermediate_error"] = str(exc)
    try:
        out.update(
            run_induction_validation_champion(
                model,
                n_train_steps=1000,
                seeds=(seed,),
                n_eval=100,
                batch_size=16,
                device=device,
                timeout_s=150,
            ).to_dict()
        )
    except Exception as exc:  # pragma: no cover
        out["induction_validation_status"] = f"failed:{type(exc).__name__}"
        out["induction_validation_error"] = str(exc)
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def checkpoint_score(row: dict[str, Any]) -> float:
    score = 0.0
    val_ppl = _safe_float(row.get("val_ppl"))
    if val_ppl:
        score += max(0.0, 8.0 - math.log(max(val_ppl, 1.0)))
    for key, weight in (
        ("induction_screening_auc", 2.0),
        ("ar_gate_in_dist_pair_acc", 1.5),
        ("language_control_s10_sentence_assoc_score", 1.0),
        ("ar_validation_rank_score", 0.75),
    ):
        value = _safe_float(row.get(key))
        if value is not None:
            score += weight * value
    return round(score, 6)


def base_row(
    *,
    run_id: str,
    start_time: float,
    candidate: Candidate,
    args: argparse.Namespace,
    param_count: int | None,
    phase: str,
    branch: str,
    event: str,
    history: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "elapsed_s": round(time.perf_counter() - start_time, 3),
        "event": event,
        "phase": phase,
        "branch": branch,
        "fingerprint": candidate.fingerprint,
        "result_id": candidate.result_id,
        "label": candidate.label,
        "rationale": candidate.rationale,
        "d_model": candidate.d_model,
        "core_model_dim": candidate.core_dim,
        "n_layers": args.n_layers,
        "vocab_size": args.vocab_size,
        "max_seq_len": args.seq_len,
        "param_count": param_count,
        **history,
    }


def train_candidate_trunk(
    *,
    run_id: str,
    start_time: float,
    candidate: Candidate,
    db_row: dict[str, Any],
    history: dict[str, Any],
    train_tokens: np.ndarray,
    val_batches: list[torch.Tensor],
    args: argparse.Namespace,
    csv_path: Path,
    jsonl_path: Path,
    checkpoint_dir: Path,
    device: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    model = build_model(
        db_row["graph_json"],
        candidate.d_model,
        args.n_layers,
        args.vocab_size,
        args.seq_len,
        core_dim=candidate.core_dim,
        scaled_shell=args.scaled_shell,
    ).to(device)
    param_count = count_parameters(model)
    optimizer = make_adamw(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_row: dict[str, Any] | None = None
    best_checkpoint: Path | None = None
    step = 0
    try:
        while step < args.trunk_steps:
            if time.perf_counter() - start_time > args.time_budget_hours * 3600:
                break
            segment = min(args.checkpoint_every, args.trunk_steps - step)
            step, train_loss, train_loss_avg100, grad_norm, train_elapsed, error = (
                train_steps(
                    model,
                    optimizer,
                    train_tokens=train_tokens,
                    device=device,
                    vocab_size=args.vocab_size,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    n_steps=segment,
                    start_step=step,
                    lr=args.lr,
                    warmup_steps=args.warmup_steps,
                    clip_grad=args.clip_grad,
                    seed=args.seed,
                    branch="trunk",
                    autocast=args.autocast,
                    mix_every=0,
                    mod_token_ids=args.mod_token_ids,
                )
            )
            val_ppl = compute_perplexity(model, val_batches, args.vocab_size)
            ckpt = (
                checkpoint_dir / candidate.fingerprint / "trunk" / f"step_{step:06d}.pt"
            )
            save_checkpoint(
                ckpt,
                model,
                optimizer,
                candidate=candidate,
                args=args,
                step=step,
                branch="trunk",
                param_count=param_count,
            )
            row = base_row(
                run_id=run_id,
                start_time=start_time,
                candidate=candidate,
                args=args,
                param_count=param_count,
                phase="trunk",
                branch="trunk",
                event="checkpoint",
                history=history,
            )
            row.update(
                {
                    "step": step,
                    "branch_step": step,
                    "train_loss": train_loss,
                    "train_loss_avg100": train_loss_avg100,
                    "val_ppl": val_ppl,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "checkpoint_path": str(ckpt),
                    "status": error or "ok",
                    "train_segment_elapsed_s": round(train_elapsed, 3),
                }
            )
            if args.quick_probes and step % args.probe_every == 0:
                row.update(
                    run_quick_probes(
                        model,
                        device=device,
                        seed=args.seed + step,
                        include_ar_validation=args.ar_validation_at_checkpoints,
                    )
                )
            row["selection_score"] = checkpoint_score(row)
            append_row(csv_path, jsonl_path, row)
            best_row = row
            best_checkpoint = ckpt
            if error:
                break
    finally:
        del optimizer, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return best_checkpoint, best_row


def train_branch(
    *,
    run_id: str,
    start_time: float,
    candidate: Candidate,
    db_row: dict[str, Any],
    history: dict[str, Any],
    trunk_checkpoint: Path,
    train_tokens: np.ndarray,
    val_batches: list[torch.Tensor],
    args: argparse.Namespace,
    csv_path: Path,
    jsonl_path: Path,
    checkpoint_dir: Path,
    device: str,
    branch: str,
) -> dict[str, Any] | None:
    model = build_model(
        db_row["graph_json"],
        candidate.d_model,
        args.n_layers,
        args.vocab_size,
        args.seq_len,
        core_dim=candidate.core_dim,
        scaled_shell=args.scaled_shell,
    ).to(device)
    load_checkpoint_weights(model, trunk_checkpoint, device)
    param_count = count_parameters(model)
    optimizer = make_adamw(
        model.parameters(), lr=args.branch_lr, weight_decay=args.weight_decay
    )
    best_row: dict[str, Any] | None = None
    branch_step = 0
    total_start = args.trunk_steps
    try:
        while branch_step < args.branch_steps:
            if time.perf_counter() - start_time > args.time_budget_hours * 3600:
                break
            segment = min(args.checkpoint_every, args.branch_steps - branch_step)
            (
                next_step,
                train_loss,
                train_loss_avg100,
                grad_norm,
                train_elapsed,
                error,
            ) = train_steps(
                model,
                optimizer,
                train_tokens=train_tokens,
                device=device,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                n_steps=segment,
                start_step=total_start + branch_step,
                lr=args.branch_lr,
                warmup_steps=args.branch_warmup_steps,
                clip_grad=args.clip_grad,
                seed=args.seed + 10_000,
                branch=branch,
                autocast=args.autocast,
                mix_every=args.mix_every,
                mod_token_ids=args.mod_token_ids,
            )
            branch_step = next_step - total_start
            total_step = total_start + branch_step
            val_ppl = compute_perplexity(model, val_batches, args.vocab_size)
            ckpt = (
                checkpoint_dir
                / candidate.fingerprint
                / branch
                / f"step_{total_step:06d}.pt"
            )
            save_checkpoint(
                ckpt,
                model,
                optimizer,
                candidate=candidate,
                args=args,
                step=total_step,
                branch=branch,
                param_count=param_count,
            )
            row = base_row(
                run_id=run_id,
                start_time=start_time,
                candidate=candidate,
                args=args,
                param_count=param_count,
                phase="branch",
                branch=branch,
                event="checkpoint",
                history=history,
            )
            row.update(
                {
                    "step": total_step,
                    "branch_step": branch_step,
                    "train_loss": train_loss,
                    "train_loss_avg100": train_loss_avg100,
                    "val_ppl": val_ppl,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "checkpoint_path": str(ckpt),
                    "status": error or "ok",
                    "train_segment_elapsed_s": round(train_elapsed, 3),
                }
            )
            final_checkpoint = branch_step >= args.branch_steps
            if args.quick_probes and (
                branch_step % args.probe_every == 0 or final_checkpoint
            ):
                row.update(
                    run_quick_probes(
                        model,
                        device=device,
                        seed=args.seed + total_step,
                        include_ar_validation=True,
                    )
                )
            if args.final_probes and final_checkpoint:
                row.update(
                    run_final_probes(
                        model, device=device, seed=args.seed, vocab_size=args.vocab_size
                    )
                )
            row["selection_score"] = checkpoint_score(row)
            append_row(csv_path, jsonl_path, row)
            best_row = row
            if error:
                break
    finally:
        del optimizer, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return best_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--time-budget-hours", type=float, default=4.0)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=24)
    parser.add_argument("--trunk-steps", type=int, default=5000)
    parser.add_argument("--branch-steps", type=int, default=15000)
    parser.add_argument("--branch-top-k", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--probe-every", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--branch-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--branch-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--mix-every", type=int, default=5)
    parser.add_argument(
        "--quick-probes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--small-ar-at-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--final-probes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--autocast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--mod-token-ids", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--scaled-shell", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_id = "bim_scale_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNTIME_ROOT / run_id
    csv_path = RUNTIME_ROOT / f"{run_id}.csv"
    jsonl_path = RUNTIME_ROOT / f"{run_id}.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    start_time = time.perf_counter()

    train_tokens = np.load(TRAIN_TOKENS, mmap_mode="r")
    val_tokens = np.load(VAL_TOKENS, mmap_mode="r")
    val_batches = make_batches(
        val_tokens,
        args.batch_size,
        args.seq_len,
        args.val_batches,
        args.device,
        seed=args.seed + 999,
    )
    val_batches = normalize_token_batches(
        val_batches,
        args.vocab_size,
        mod_token_ids=args.mod_token_ids,
    )
    easy25 = load_easy25_history()
    conn = sqlite3.connect(DB_PATH)
    candidate_records: dict[str, dict[str, Any]] = {}
    histories: dict[str, dict[str, Any]] = {}
    trunk_results: list[tuple[Candidate, Path, dict[str, Any]]] = []

    try:
        candidates = (
            CANDIDATES[: args.candidate_limit]
            if args.candidate_limit > 0
            else CANDIDATES
        )
        for candidate in candidates:
            db_row = fetch_candidate_record(conn, candidate)
            candidate_records[candidate.fingerprint] = db_row
            history = historical_metrics(db_row, easy25.get(candidate.fingerprint))
            histories[candidate.fingerprint] = history
            start_row = base_row(
                run_id=run_id,
                start_time=start_time,
                candidate=candidate,
                args=args,
                param_count=None,
                phase="trunk",
                branch="trunk",
                event="start",
                history=history,
            )
            start_row["status"] = "ok"
            append_row(csv_path, jsonl_path, start_row)
            checkpoint, row = train_candidate_trunk(
                run_id=run_id,
                start_time=start_time,
                candidate=candidate,
                db_row=db_row,
                history=history,
                train_tokens=train_tokens,
                val_batches=val_batches,
                args=args,
                csv_path=csv_path,
                jsonl_path=jsonl_path,
                checkpoint_dir=checkpoint_dir,
                device=args.device,
            )
            if checkpoint is not None and row is not None and row.get("status") == "ok":
                trunk_results.append((candidate, checkpoint, row))
            if time.perf_counter() - start_time > args.time_budget_hours * 3600:
                break

        trunk_results.sort(
            key=lambda item: _safe_float(item[2].get("selection_score")) or -1.0,
            reverse=True,
        )
        selected = trunk_results[: max(0, args.branch_top_k)]
        for candidate, checkpoint, trunk_row in selected:
            row = base_row(
                run_id=run_id,
                start_time=start_time,
                candidate=candidate,
                args=args,
                param_count=trunk_row.get("param_count"),
                phase="branch",
                branch="selection",
                event="selected_for_branch",
                history=histories[candidate.fingerprint],
            )
            row.update(
                {
                    "selection_score": trunk_row.get("selection_score"),
                    "checkpoint_path": str(checkpoint),
                    "status": "ok",
                }
            )
            append_row(csv_path, jsonl_path, row)
            for branch in ("lm_continue", "capability_mix"):
                if time.perf_counter() - start_time > args.time_budget_hours * 3600:
                    break
                train_branch(
                    run_id=run_id,
                    start_time=start_time,
                    candidate=candidate,
                    db_row=candidate_records[candidate.fingerprint],
                    history=histories[candidate.fingerprint],
                    trunk_checkpoint=checkpoint,
                    train_tokens=train_tokens,
                    val_batches=val_batches,
                    args=args,
                    csv_path=csv_path,
                    jsonl_path=jsonl_path,
                    checkpoint_dir=checkpoint_dir,
                    device=args.device,
                    branch=branch,
                )
    finally:
        conn.close()

    summary = {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "checkpoint_dir": str(checkpoint_dir),
        "elapsed_s": round(time.perf_counter() - start_time, 3),
        "status": "complete",
    }
    (run_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
