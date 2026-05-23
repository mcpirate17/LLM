import torch
import pytest
from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_to_json, graph_from_json

pytestmark = pytest.mark.unit


def test_ir_serialization_roundtrip():
    """Test that Graph -> JSON -> Graph -> IR produces correct results."""
    dim = 32
    g = ComputationGraph(dim)
    i1 = g.add_input()
    n1 = g.add_op("gelu", [i1])
    n2 = g.add_op("linear_proj", [n1], config={"out_dim": dim})
    g.set_output(n2)

    js = graph_to_json(g)
    g2 = graph_from_json(js)

    model = compile_model([g2])
    assert model is not None

    from research.defaults import VOCAB_SIZE

    input_ids = torch.randint(0, VOCAB_SIZE, (1, 16))
    out = model(input_ids)
    assert out.shape == (1, 16, VOCAB_SIZE)


if __name__ == "__main__":
    test_ir_serialization_roundtrip()
