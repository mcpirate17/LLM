from __future__ import annotations

import ctypes

from research.synthesis.native_param_formula import evaluate_param_formula_natively
from research.synthesis.primitives import PrimitiveOp, OpCategory, estimate_op_params


def test_evaluate_param_formula_natively_prefers_runtime(monkeypatch):
    calls = {}

    class FakeLib:
        def aria_eval_param_formula(self, formula_ptr, out_ptr):
            formula = ctypes.cast(formula_ptr, ctypes.c_char_p).value.decode("ascii")
            calls["formula"] = formula
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int64))[0] = 123
            return 0

    monkeypatch.setattr(
        "research.synthesis.native_param_formula.load_native_graph_analysis_lib",
        lambda: FakeLib(),
    )

    assert evaluate_param_formula_natively("64*8") == 123
    assert calls["formula"] == "64*8"


def test_evaluate_param_formula_natively_returns_none_on_failure(monkeypatch):
    class FakeLib:
        def aria_eval_param_formula(self, formula_ptr, out_ptr):
            return -1

    monkeypatch.setattr(
        "research.synthesis.native_param_formula.load_native_graph_analysis_lib",
        lambda: FakeLib(),
    )

    assert evaluate_param_formula_natively("bad") is None


def test_estimate_op_params_falls_back_when_native_formula_unavailable(monkeypatch):
    op = PrimitiveOp(
        name="test_linear",
        category=OpCategory.PARAMETERIZED,
        n_inputs=1,
        shape_rule="linear",
        has_params=True,
        param_formula="D*D//2",
    )
    monkeypatch.setattr(
        "research.synthesis.primitives.evaluate_param_formula_natively",
        lambda formula: None,
    )

    assert estimate_op_params(op, 16) == 128


def test_estimate_op_params_uses_native_formula_result(monkeypatch):
    op = PrimitiveOp(
        name="test_bias",
        category=OpCategory.PARAMETERIZED,
        n_inputs=1,
        shape_rule="identity",
        has_params=True,
        param_formula="D*4",
    )
    monkeypatch.setattr(
        "research.synthesis.primitives.evaluate_param_formula_natively",
        lambda formula: 77,
    )

    assert estimate_op_params(op, 16) == 77
