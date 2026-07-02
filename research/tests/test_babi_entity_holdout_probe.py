import polars as pl
import torch

from research.tools.babi_entity_holdout_probe import (
    _binding_split,
    _target_binding,
)
from research.tools.babi_twoarg_cpu_probe import _mtp_loss


def _row(answer: str, relation: str, anchor: str, idx: int) -> dict:
    passage = f"The {answer} is {relation} of the {anchor}."
    return {
        "query": f"q-{idx}",
        "answer": answer,
        "answer_passage": repr([passage]),
    }


def test_target_binding_parses_supporting_relation_entity() -> None:
    row = _row("kitchen", "south", "bedroom", 0)

    assert _target_binding(row) == ("south", "kitchen")


def test_binding_split_keeps_held_entities_trainable() -> None:
    rooms = ["bathroom", "bedroom", "garden", "hallway"]
    relations = ["north", "south", "east", "west"]
    df = pl.DataFrame(
        [
            _row(room, relation, rooms[(i + j + 1) % len(rooms)], i * 10 + j)
            for i, room in enumerate(rooms)
            for j, relation in enumerate(relations)
        ]
    )

    train, test, held = _binding_split(df, n_holdout=2, seed=3)
    held_pairs = {(h["relation"], h["entity"]) for h in held}
    train_pairs = set(zip(train["_target_relation"], train["_target_entity"]))
    test_pairs = set(zip(test["_target_relation"], test["_target_entity"]))

    assert held_pairs == test_pairs
    assert held_pairs.isdisjoint(train_pairs)
    for entity in {h["entity"] for h in held}:
        assert entity in set(train["answer"])


def test_mtp_loss_ignores_padding_and_is_finite() -> None:
    torch.manual_seed(0)
    ids = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]])
    logits = torch.randn(2, 4, 16, requires_grad=True)
    loss = _mtp_loss(logits, ids, depth=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
