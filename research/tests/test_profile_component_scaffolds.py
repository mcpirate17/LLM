from research.tools.profile_component_scaffolds import (
    _append_log,
    build_gpt2_attn_scaffold,
    build_gpt2_ffn_scaffold,
    build_gpt2_replace_scaffold,
    build_mamba_mixer_scaffold,
    build_pair_residual_scaffold,
    generate_cases,
)


def _assert_graph_ok(graph, family: str, expected_ops: set[str]) -> None:
    assert graph.output_node is not None
    assert graph.output_node.output_shape.dim == graph.model_dim
    assert graph.metadata.get("scaffold_family") == family
    ops = {node.op_name for node in graph.nodes.values() if not node.is_input}
    assert expected_ops.issubset(ops)


def test_build_gpt2_attn_scaffold():
    graph = build_gpt2_attn_scaffold("linear_attention", model_dim=96)
    _assert_graph_ok(
        graph,
        "gpt2_attn",
        {"rmsnorm", "linear_attention", "linear_proj", "swiglu_mlp", "add"},
    )


def test_build_gpt2_ffn_scaffold():
    graph = build_gpt2_ffn_scaffold("conv1d_seq", model_dim=96)
    _assert_graph_ok(
        graph,
        "gpt2_ffn",
        {"rmsnorm", "softmax_attention", "linear_proj", "conv1d_seq", "add"},
    )


def test_build_gpt2_replace_scaffold():
    graph = build_gpt2_replace_scaffold("linear_attention", model_dim=96)
    _assert_graph_ok(
        graph,
        "gpt2_replace",
        {"rmsnorm", "softmax_attention", "linear_proj", "linear_attention", "add"},
    )


def test_build_mamba_mixer_scaffold():
    graph = build_mamba_mixer_scaffold("rwkv_channel", model_dim=96)
    _assert_graph_ok(
        graph,
        "mamba_mixer",
        {"rmsnorm", "conv1d_seq", "rwkv_channel", "linear_proj", "swiglu_mlp", "add"},
    )


def test_build_pair_residual_scaffold():
    graph = build_pair_residual_scaffold("conv1d_seq", "swiglu_mlp", model_dim=96)
    _assert_graph_ok(
        graph,
        "pair_residual",
        {"rmsnorm", "conv1d_seq", "swiglu_mlp", "add"},
    )


def test_generate_cases_includes_family_controls():
    cases = generate_cases(
        ["gpt2_attn", "pair_residual"], ["linear_attention", "conv1d_seq"], max_pairs=3
    )
    names = [case.name for case in cases]
    assert "gpt2_attn:control" in names
    assert "pair_residual:control" in names
    assert any(name.startswith("pair_residual:") and "+" in name for name in names)


def test_generate_cases_keeps_gpt2_ffn_clean():
    cases = generate_cases(
        ["gpt2_ffn"], ["linear_attention", "conv1d_seq"], max_pairs=3
    )
    names = [case.name for case in cases]
    assert "gpt2_ffn:conv1d_seq" in names
    assert "gpt2_ffn:linear_attention" not in names


def test_generate_cases_routes_replacements_to_replacement_family():
    cases = generate_cases(
        ["gpt2_replace"], ["linear_attention", "conv1d_seq"], max_pairs=3
    )
    names = [case.name for case in cases]
    assert "gpt2_replace:control" in names
    assert "gpt2_replace:linear_attention" in names
    assert "gpt2_replace:conv1d_seq" in names


def test_generate_cases_can_allow_arbitrary_ops():
    cases = generate_cases(
        ["gpt2_ffn"],
        ["linear_attention", "block_sparse_linear"],
        max_pairs=3,
        allow_arbitrary_ops=True,
    )
    names = [case.name for case in cases]
    assert "gpt2_ffn:linear_attention" in names
    assert "gpt2_ffn:block_sparse_linear" in names


def test_append_log_writes_lines(tmp_path):
    log_path = tmp_path / "scaffold.log"
    _append_log(log_path, "line one")
    _append_log(log_path, "line two\n")
    assert log_path.read_text(encoding="utf-8") == "line one\nline two\n"
