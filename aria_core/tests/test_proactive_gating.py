
import torch
import aria_core
from research.synthesis.primitives import OPCODE_MAP

def test_proactive_gating():
    print("Testing Project Hephaestus: Proactive Gating...")
    
    # OPCODES for testing (based on graph_validator.cpp heuristics)
    # Norm is 60-64, Parameterized is 40-55, Linear is 15
    LIN = 15
    PROJ = 40
    NORM = 60
    IN = 0
    
    # 1. Deep graph with NO normalization (should fail)
    # Depth: input -> p1 -> p2 -> p3 -> p4 -> p5 -> p6 -> p7 -> p8 -> p9
    n_nodes = 10
    op_codes = [IN, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ]
    edges = [[i, i+1] for i in range(n_nodes - 1)]
    
    res = aria_core.proactive_gating(n_nodes, edges, op_codes)
    print(f"Deep (No Norm) Result: passed={res['passed']}, reason='{res['reason']}', depth={res['max_depth']}")
    assert res['passed'] is False
    assert "Normalization Gap" in res['reason']
    
    # 2. Deep graph WITH normalization (should pass)
    op_codes_norm = [IN, PROJ, PROJ, PROJ, NORM, PROJ, PROJ, PROJ, PROJ, PROJ]
    res_norm = aria_core.proactive_gating(n_nodes, edges, op_codes_norm)
    print(f"Deep (With Norm) Result: passed={res_norm['passed']}, reason='{res_norm['reason']}', depth={res_norm['max_depth']}")
    assert res_norm['passed'] is True

    # 3. Toxic Motif Detection (A -> B -> C: Param -> Linear -> Param)
    # input(0) -> p1(40) -> lin(15) -> p2(40)
    op_codes_toxic = [IN, PROJ, LIN, PROJ]
    edges_toxic = [[0, 1], [1, 2], [2, 3]]
    # We need > 5 for a fail in my current impl, so let's check count
    res_toxic = aria_core.proactive_gating(4, edges_toxic, op_codes_toxic)
    print(f"Toxic Motif Count: {res_toxic['n_toxic_motifs']}")
    assert res_toxic['n_toxic_motifs'] == 1

    print("Proactive Gating Logic: ALL TESTS PASSED")

if __name__ == "__main__":
    test_proactive_gating()
