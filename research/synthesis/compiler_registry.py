from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

OP_DISPATCH: Dict[
    str, Callable[[nn.Module, Tuple[torch.Tensor, ...], Dict], torch.Tensor]
] = {}


def load_split_op_modules() -> None:
    split_modules = {
        "compiler_ops_math": ".compiler_ops_math",
        "compiler_ops_attention": ".compiler_ops_attention",
        "compiler_ops_sequence": ".compiler_ops_sequence",
        "compiler_ops_sparse": ".compiler_ops_sparse",
        "compiler_ops_mathspaces": ".compiler_ops_mathspaces",
        "compiler_ops_routing": ".compiler_ops_routing",
        "true_routing_ops": ".true_routing_ops",
    }
    for label in split_modules:
        try:
            mod = __import__(f"research.synthesis.{label}", fromlist=["OP_IMPLS"])
            OP_DISPATCH.update(mod.OP_IMPLS)
        except ImportError as exc:
            raise ImportError(
                f"Failed to load compiler op module '{label}': {exc}\n"
                f"Compiler handlers for ops in that module are missing. "
                f"Fix the import error before continuing."
            ) from exc
