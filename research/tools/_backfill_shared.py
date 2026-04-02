"""Shared utilities for backfill scripts."""

from __future__ import annotations

from research.defaults import VOCAB_SIZE
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json

DB_PATH = "research/lab_notebook.db"


def reconstruct_model(graph_json_str: str, device: str):
    """Deserialize graph JSON → compiled model on device, eval mode."""
    graph = graph_from_json(graph_json_str)
    return compile_model([graph], vocab_size=VOCAB_SIZE).to(device).eval()
