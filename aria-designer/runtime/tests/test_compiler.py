import sys
import os
import torch
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from runtime.compiler import compile_workflow

def test_compile_and_run():
    workflow_json = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "test-wf",
        "name": "Test Workflow",
        "nodes": [
            {
                "id": "input1",
                "component_type": "io/input",
                "params": {},
                "ui_meta": {}
            },
            {
                "id": "relu1",
                "component_type": "math/relu",
                "params": {},
                "ui_meta": {}
            }
        ],
        "edges": [
            {
                "id": "e1",
                "source": "input1",
                "source_port": "y",
                "target": "relu1",
                "target_port": "x"
            }
        ]
    }

    components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "components"))
    model = compile_workflow(workflow_json, components_dir)

    assert isinstance(model, torch.nn.Module)
    assert "relu1" in model.submodules

    # Test forward pass
    x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    # For source nodes, we provide initial values in the 'inputs' dict
    outputs = model({"input1": x})

    assert "relu1" in outputs
    expected = torch.tensor([0.0, 2.0, 0.0, 4.0])
    torch.testing.assert_close(outputs["relu1"], expected)


def test_multi_input_identity_falls_back_to_handler_forward():
    workflow_json = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "test-multi-input-fallback",
        "name": "Test Multi Input Fallback",
        "nodes": [
            {
                "id": "input_a",
                "component_type": "io/input",
                "params": {},
                "ui_meta": {}
            },
            {
                "id": "input_b",
                "component_type": "io/input",
                "params": {},
                "ui_meta": {}
            },
            {
                "id": "add1",
                "component_type": "math/add",
                "params": {},
                "ui_meta": {}
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "input_a",
                "source_port": "y",
                "target": "add1",
                "target_port": "a"
            },
            {
                "id": "e2",
                "source": "input_b",
                "source_port": "y",
                "target": "add1",
                "target_port": "b"
            },
        ]
    }

    components_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "components"))
    model = compile_workflow(workflow_json, components_dir)

    assert "add1" in model.submodules

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    outputs = model({"input_a": a, "input_b": b})

    assert "add1" in outputs
    torch.testing.assert_close(outputs["add1"], a)

if __name__ == "__main__":
    test_compile_and_run()
    print("Compiler tests passed!")
