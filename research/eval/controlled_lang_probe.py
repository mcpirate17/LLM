"""Unified controlled-language probe — one training pass, two evaluations.

Both ``synthetic_association_score`` (codex, HellaSwag-style 4-way forced
choice) and ``nano_blimp_score`` (claude, BLiMP-style minimal-pair
log-prob) train on the SAME (noun, query, target) distribution from
``synthetic_association_eval._make_train_batch``. Running them
back-to-back via the public APIs duplicates training (~30s+ per call on
nano models). This module trains once and runs both evals on the trained
state — same data, half the wall time.

Use this when you want both signals; use the individual probes when you
want to call only one.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nano_blimp_eval import NANO_BLIMP_METRIC_VERSION, nano_blimp_eval_only
from .synthetic_association_eval import (
    SYNTHETIC_ASSOCIATION_METRIC_VERSION,
    SyntheticAssociationResult,
    _eval_forced_choice_accuracy,
    _make_layout,
    _make_train_batch,
    _ADJ_QUERY,
    _VERB_QUERY,
)
from .utils import clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

CONTROLLED_LANG_METRIC_VERSION = "controlled_lang_v2"

_DEFAULT_ACTIVE_VOCAB = 80  # codex's calibrated default
_DEFAULT_TRAIN_STEPS = 20  # codex's calibrated default
_DEFAULT_BATCH = 32
_DEFAULT_LR = 1e-3
_DEFAULT_EVAL_REPEATS = 8
_TIMEOUT_S = 60.0


@dataclass(slots=True)
class ControlledLangResult:
    nano_blimp: Dict[str, Any]
    synthetic_association: Dict[str, Any]
    n_train_steps: int
    active_vocab_size: int
    elapsed_ms: float
    status: str
    checkpoints: tuple[Dict[str, Any], ...] = ()
    metric_version: str = CONTROLLED_LANG_METRIC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "controlled_lang_metric_version": self.metric_version,
            "controlled_lang_train_steps": self.n_train_steps,
            "controlled_lang_active_vocab_size": self.active_vocab_size,
            "controlled_lang_elapsed_ms": self.elapsed_ms,
            "controlled_lang_status": self.status,
        }
        if self.checkpoints:
            out["controlled_lang_checkpoints"] = list(self.checkpoints)
        out.update(self.nano_blimp)
        out.update(self.synthetic_association)
        return out


def controlled_lang_probe(
    model: nn.Module,
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    eval_repeats: int = _DEFAULT_EVAL_REPEATS,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _TIMEOUT_S,
    checkpoint_steps: tuple[int, ...] | None = None,
    preserve_state: bool = True,
) -> ControlledLangResult:
    """Train once on the controlled-language association corpus, then
    evaluate both synthetic_association (4-way forced choice) and
    nano_blimp (minimal-pair log-prob) on the trained state.

    Caller's model state is preserved (state_dict snapshot/restore).
    """
    t0 = time.perf_counter()
    deadline = t0 + float(timeout_s)
    layout = _make_layout(active_vocab_size)
    if layout.adjective_hi > int(getattr(model, "vocab_size", layout.adjective_hi)):
        return ControlledLangResult(
            nano_blimp={},
            synthetic_association={},
            n_train_steps=0,
            active_vocab_size=layout.active_vocab_size,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
        )

    # state_dict snapshot — survives weight_norm parametrize where deepcopy
    # fails (the silent-fail bug we hit on adaptive_conv_ffn earlier).
    saved_state = (
        {k: v.detach().clone() for k, v in model.state_dict().items()}
        if preserve_state
        else None
    )
    was_training = model.training

    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    steps = 0
    train_status = "ok"
    checkpoints = tuple(
        sorted(
            {
                int(step)
                for step in (checkpoint_steps or ())
                if int(step) > 0 and int(step) <= int(n_train_steps)
            }
        )
    )
    checkpoint_set = set(checkpoints)
    checkpoint_payloads: list[Dict[str, Any]] = []
    checkpoint_results: dict[int, tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def _eval_current(current_steps: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # Eval 1: synthetic_association (4-way forced choice). Reuse codex's
        # internal eval helper directly so we don't re-train.
        model.eval()
        verb_acc = _eval_forced_choice_accuracy(
            model,
            layout,
            relation_id=_VERB_QUERY,
            eval_repeats=eval_repeats,
            batch_size=batch_size,
            device=device,
        )
        adj_acc = _eval_forced_choice_accuracy(
            model,
            layout,
            relation_id=_ADJ_QUERY,
            eval_repeats=eval_repeats,
            batch_size=batch_size,
            device=device,
        )
        sa_score = (verb_acc + adj_acc) / 2.0
        sa = SyntheticAssociationResult(
            score=round(float(sa_score), 4),
            verb_accuracy=round(float(verb_acc), 4),
            adjective_accuracy=round(float(adj_acc), 4),
            n_words=layout.n_per_type * 3,
            n_pairs=layout.n_per_type * 2,
            n_train_steps=current_steps,
            active_vocab_size=layout.active_vocab_size,
            chance=layout.chance,
            elapsed_ms=0.0,
            status="ok",
            metric_version=SYNTHETIC_ASSOCIATION_METRIC_VERSION,
        ).to_dict()

        # Eval 2: nano_blimp on the same trained state (minimal-pair log-prob).
        nb = nano_blimp_eval_only(model, layout, device=device).to_dict()
        nb.setdefault("nano_blimp_metric_version", NANO_BLIMP_METRIC_VERSION)
        return sa, nb

    def _checkpoint_payload(
        current_steps: int, sa: Dict[str, Any], nb: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "steps": int(current_steps),
            "active_vocab_size": int(layout.active_vocab_size),
            "synthetic_association_score": sa.get("synthetic_association_score"),
            "synthetic_association_verb_accuracy": sa.get(
                "synthetic_association_verb_accuracy"
            ),
            "synthetic_association_adjective_accuracy": sa.get(
                "synthetic_association_adjective_accuracy"
            ),
            "nano_blimp_score": nb.get("nano_blimp_score"),
            "nano_blimp_order_grammaticality_acc": nb.get(
                "nano_blimp_order_grammaticality_acc"
            ),
            "nano_blimp_binding_fidelity_acc": nb.get(
                "nano_blimp_binding_fidelity_acc"
            ),
            "nano_blimp_binding_fidelity_held_out_acc": nb.get(
                "nano_blimp_binding_fidelity_held_out_acc"
            ),
            "nano_blimp_held_out_score": nb.get("nano_blimp_held_out_score"),
        }

    try:
        model.train()
        opt = make_adamw(model.parameters(), lr=lr)
        for step in range(int(n_train_steps)):
            if time.perf_counter() > deadline:
                train_status = "timeout"
                break
            input_ids, targets = _make_train_batch(layout, batch_size, device, rng)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids)
            pred_logits = logits[:, 1, layout.answer_lo : layout.answer_hi]
            loss = F.cross_entropy(pred_logits, targets - layout.answer_lo)
            if not torch.isfinite(loss):
                train_status = "non_finite_loss"
                break
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()
            steps = step + 1
            if steps in checkpoint_set:
                sa_cp, nb_cp = _eval_current(steps)
                checkpoint_results[steps] = (sa_cp, nb_cp)
                checkpoint_payloads.append(_checkpoint_payload(steps, sa_cp, nb_cp))

        if train_status not in ("ok", "timeout"):
            return ControlledLangResult(
                nano_blimp={},
                synthetic_association={},
                n_train_steps=steps,
                active_vocab_size=layout.active_vocab_size,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                status=train_status,
            )

        if checkpoint_payloads and checkpoint_payloads[-1]["steps"] == int(steps):
            # The final checkpoint was just evaluated; reuse it instead of
            # running the same forced-choice/minimal-pair eval a second time.
            sa, nb = checkpoint_results[int(steps)]
        else:
            sa, nb = _eval_current(steps)
            if checkpoints:
                checkpoint_payloads.append(_checkpoint_payload(steps, sa, nb))

        return ControlledLangResult(
            nano_blimp=nb,
            synthetic_association=sa,
            n_train_steps=steps,
            active_vocab_size=layout.active_vocab_size,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status=train_status,
            checkpoints=tuple(checkpoint_payloads),
        )
    finally:
        if saved_state is not None:
            model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
