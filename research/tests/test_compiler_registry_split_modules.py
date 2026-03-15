from research.synthesis.compiler import _OP_DISPATCH


def test_split_compiler_modules_register_expected_ops():
    expected = {
        "identity",
        "softmax_attention",
        "moe_topk",
        "tropical_attention",
    }

    missing = expected.difference(_OP_DISPATCH)
    assert not missing
