"""Sensitivity probing and Jacobian-derived metrics."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ._probe_runtime import disable_native_probe_dispatch
from ._sensitivity_skip_stats import (
    record_sensitivity_skip,
)
from .fingerprint_native import collect_sensitivity_rows, sensitivity_metrics

logger = logging.getLogger(__name__)


def forward_model_from_embed(model: nn.Module, embed_in: torch.Tensor) -> torch.Tensor:
    native_forward = getattr(model, "_fingerprint_forward_from_embed", None)
    if callable(native_forward):
        return native_forward(embed_in)

    x_local = embed_in
    if hasattr(model, "pos_enc") and model.pos_enc is not None:
        x_local = model.pos_enc(x_local)
    if hasattr(model, "layers"):
        for layer in model.layers:
            x_local = layer(x_local)
        return x_local
    if hasattr(model, "topology"):
        return model.topology(x_local)
    return x_local


def analyze_sensitivity(
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int,
) -> Dict[str, float]:
    result = {
        "spectral_norm": 0.0,
        "effective_rank": 0.0,
        "uniformity": 0.0,
        "_succeeded": False,
    }
    try:
        model.eval()
        device_str = str(device) if not isinstance(device, str) else device
        with (
            disable_native_probe_dispatch(model, device=device_str),
            torch.enable_grad(),
        ):
            ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
            embed = model.embed(ids).detach().requires_grad_(True)

            def forward_from_embed(embed_in: torch.Tensor) -> torch.Tensor:
                return forward_model_from_embed(model, embed_in)

            x = forward_from_embed(embed)
            if not x.requires_grad:
                record_sensitivity_skip("output_no_grad")
                return result

            n_positions = max(1, min(4, seq_len))
            step = max(1, seq_len // n_positions)
            positions = torch.arange(
                0, seq_len, step, device=device, dtype=torch.int64
            )[:n_positions]
            if (
                getattr(collect_position_sensitivities, "__module__", __name__)
                != __name__
            ):
                sens_matrix = collect_position_sensitivities(
                    forward_from_embed, embed, positions
                )
            else:
                sens_matrix = collect_sensitivity_rows(x, embed, positions)
                if sens_matrix is None:
                    sens_matrix = collect_position_sensitivities(
                        forward_from_embed, embed, positions
                    )
            if sens_matrix is None:
                record_sensitivity_skip("no_sensitivity_grads")
                return result

            result.update(sensitivity_metrics(sens_matrix))
            result["_succeeded"] = True
    except Exception as exc:
        logger.warning("Sensitivity analysis failed: %s", exc)
    return result


def collect_position_sensitivities(
    x_or_forward: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    embed: torch.Tensor,
    positions: torch.Tensor,
) -> Optional[torch.Tensor]:
    n_pos = positions.numel()
    if n_pos == 0 or not embed.requires_grad:
        return None

    if callable(x_or_forward):
        forward_from_embed = x_or_forward
    else:
        x = x_or_forward
        native_rows = collect_sensitivity_rows(x, embed, positions)
        if native_rows is not None:
            return native_rows
        try:
            grad_outputs = torch.zeros(n_pos, *x.shape, device=x.device, dtype=x.dtype)
            grad_outputs[
                torch.arange(n_pos, device=positions.device), :, positions, :
            ] = 1.0
            batched = torch.autograd.grad(
                x,
                embed,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                is_grads_batched=True,
            )[0]
            return batched.norm(dim=-1).squeeze(1)
        except RuntimeError:

            def forward_from_embed(_embed_in: torch.Tensor) -> torch.Tensor:
                return x

    try:
        embed_expanded = embed.expand(n_pos, *embed.shape[1:]).contiguous()
        embed_expanded.requires_grad_(True)
        out = forward_from_embed(embed_expanded)
        selected = out[torch.arange(n_pos, device=positions.device), positions, :]
        grad_out = torch.autograd.grad(
            selected.sum(),
            embed_expanded,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad_out is not None:
            return grad_out.norm(dim=-1).squeeze(1)
    except RuntimeError as exc:
        logger.debug("expanded-batch sensitivity path unavailable: %s", exc)

    try:
        from torch.func import grad, vmap

        def probe_loss(embed_in: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
            out = forward_from_embed(embed_in)
            return torch.index_select(out, 1, pos_idx.reshape(1)).sum()

        batched = vmap(lambda pos_idx: grad(probe_loss, argnums=0)(embed, pos_idx))(
            positions
        )
        return batched.norm(dim=-1).squeeze(1)
    except (ImportError, RuntimeError) as exc:
        logger.debug("vmap sensitivity path unavailable: %s", exc)
    return None
