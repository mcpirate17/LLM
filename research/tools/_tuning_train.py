from __future__ import annotations

import torch

from research.eval.utils import micro_train_loop
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.synthesis.serializer import graph_from_json


def train_compiled_graph_base(
    graph_json_str: str,
    *,
    base_steps: int,
    device: str,
    vocab_size: int,
) -> torch.nn.Module:
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph]).to(device)
    batches = [torch.randint(0, vocab_size, (4, 128), device=device) for _ in range(8)]
    micro_train_loop(model, batches, vocab_size=vocab_size, n_steps=base_steps, lr=3e-4)
    return model
