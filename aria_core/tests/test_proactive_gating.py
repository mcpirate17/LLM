import aria_core


def test_proactive_gating():
    # OPCODES for testing (based on graph_validator.cpp heuristics)
    LIN = 15
    PROJ = 40
    NORM = 60
    IN = 0

    # The C++ binding requires explicit opcode classification lists:
    #   norm_opcodes, param_opcodes, linear_opcodes
    norm_opcodes = list(range(60, 65))  # 60-64
    param_opcodes = list(range(40, 56))  # 40-55
    linear_opcodes = [15]

    # 1. Deep graph with NO normalization (should fail)
    n_nodes = 10
    op_codes = [IN, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ, PROJ]
    edges = [[i, i + 1] for i in range(n_nodes - 1)]

    res = aria_core.proactive_gating(
        n_nodes, edges, op_codes, norm_opcodes, param_opcodes, linear_opcodes
    )
    assert res["passed"] is False
    assert "Normalization Gap" in res["reason"]

    # 2. Deep graph WITH normalization (should pass)
    op_codes_norm = [IN, PROJ, PROJ, PROJ, NORM, PROJ, PROJ, PROJ, PROJ, PROJ]
    res_norm = aria_core.proactive_gating(
        n_nodes, edges, op_codes_norm, norm_opcodes, param_opcodes, linear_opcodes
    )
    assert res_norm["passed"] is True

    # 3. Toxic Motif Detection (Param -> Linear -> Param)
    op_codes_toxic = [IN, PROJ, LIN, PROJ]
    edges_toxic = [[0, 1], [1, 2], [2, 3]]
    res_toxic = aria_core.proactive_gating(
        4, edges_toxic, op_codes_toxic, norm_opcodes, param_opcodes, linear_opcodes
    )
    assert res_toxic["n_toxic_motifs"] == 1
