from __future__ import annotations

import numpy as np

from research.scientist.native.dispatch import (
    _check_native_op_support,
    dispatch_op_native,
)


class _FakeBridge:
    def dispatch_rmsnorm(self, x, weight, eps=1e-5):
        return {
            "x_shape": tuple(x.shape),
            "weight_shape": tuple(weight.shape),
            "eps": float(eps),
        }

    def dispatch_gated_delta_compiled(
        self,
        x,
        q_weight,
        k_weight,
        v_weight,
        alpha_weight,
        beta_weight,
        o_weight,
        n_heads,
    ):
        return {
            "x_shape": tuple(x.shape),
            "q_weight_shape": tuple(q_weight.shape),
            "k_weight_shape": tuple(k_weight.shape),
            "v_weight_shape": tuple(v_weight.shape),
            "alpha_weight_shape": tuple(alpha_weight.shape),
            "beta_weight_shape": tuple(beta_weight.shape),
            "o_weight_shape": tuple(o_weight.shape),
            "n_heads": int(n_heads),
        }

    def dispatch_state_space_compiled(
        self,
        x,
        ssm_a,
        ssm_b_weight,
        ssm_c_weight,
        ssm_d,
        ssm_dt_weight,
        ssm_dt_bias,
    ):
        return {
            "x_shape": tuple(x.shape),
            "ssm_a_shape": tuple(ssm_a.shape),
            "ssm_b_weight_shape": tuple(ssm_b_weight.shape),
            "ssm_c_weight_shape": tuple(ssm_c_weight.shape),
            "ssm_d_shape": tuple(ssm_d.shape),
            "ssm_dt_weight_shape": tuple(ssm_dt_weight.shape),
            "ssm_dt_bias_shape": tuple(ssm_dt_bias.shape),
        }

    def dispatch_selective_scan_compiled(self, x, a_log, dt_proj, b_weight, c_weight):
        return {
            "x_shape": tuple(x.shape),
            "a_log_shape": tuple(a_log.shape),
            "dt_proj_shape": tuple(dt_proj.shape),
            "b_weight_shape": tuple(b_weight.shape),
            "c_weight_shape": tuple(c_weight.shape),
        }

    def dispatch_softmax_attention(self, x, wq, wk, wv, wo, n_heads):
        return {
            "x_shape": tuple(x.shape),
            "wq_shape": tuple(wq.shape),
            "wk_shape": tuple(wk.shape),
            "wv_shape": tuple(wv.shape),
            "wo_shape": tuple(wo.shape),
            "n_heads": int(n_heads),
        }

    def dispatch_gated_linear(self, x, w, w_gate, bias=None, bias_gate=None):
        return {
            "x_shape": tuple(x.shape),
            "w_shape": tuple(w.shape),
            "w_gate_shape": tuple(w_gate.shape),
            "bias_shape": None if bias is None else tuple(np.asarray(bias).shape),
            "bias_gate_shape": None
            if bias_gate is None
            else tuple(np.asarray(bias_gate).shape),
        }

    def dispatch_rwkv_time_mixing(self, x, w_decay, u_bonus, w_k, w_v, w_r):
        return {
            "x_shape": tuple(x.shape),
            "w_decay_shape": tuple(w_decay.shape),
            "u_bonus_shape": tuple(u_bonus.shape),
            "w_k_shape": tuple(w_k.shape),
            "w_v_shape": tuple(w_v.shape),
            "w_r_shape": tuple(w_r.shape),
        }


def test_dispatch_op_native_routes_gated_linear(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((4, 8), dtype=np.float32)
    w = np.zeros((6, 8), dtype=np.float32)
    w_gate = np.zeros((6, 8), dtype=np.float32)
    bias = np.zeros((6,), dtype=np.float32)
    bias_gate = np.zeros((6,), dtype=np.float32)

    result = dispatch_op_native(
        "gated_linear",
        x,
        w,
        w_gate,
        bias=bias,
        bias_gate=bias_gate,
    )

    assert result == {
        "x_shape": (4, 8),
        "w_shape": (6, 8),
        "w_gate_shape": (6, 8),
        "bias_shape": (6,),
        "bias_gate_shape": (6,),
    }


def test_dispatch_op_native_routes_softmax_attention(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((2, 5, 8), dtype=np.float32)
    wq = np.zeros((8, 8), dtype=np.float32)
    wk = np.zeros((8, 8), dtype=np.float32)
    wv = np.zeros((8, 8), dtype=np.float32)
    wo = np.zeros((8, 8), dtype=np.float32)

    result = dispatch_op_native(
        "softmax_attention",
        x,
        wq,
        wk,
        wv,
        wo,
        n_heads=2,
    )

    assert result == {
        "x_shape": (2, 5, 8),
        "wq_shape": (8, 8),
        "wk_shape": (8, 8),
        "wv_shape": (8, 8),
        "wo_shape": (8, 8),
        "n_heads": 2,
    }


def test_dispatch_op_native_routes_selective_scan(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((2, 5, 8), dtype=np.float32)
    a_log = np.zeros((8,), dtype=np.float32)
    dt_proj = np.zeros((8,), dtype=np.float32)
    b_weight = np.zeros((8, 8), dtype=np.float32)
    c_weight = np.zeros((8, 8), dtype=np.float32)

    result = dispatch_op_native(
        "selective_scan",
        x,
        a_log,
        dt_proj,
        b_weight,
        c_weight,
    )

    assert result == {
        "x_shape": (2, 5, 8),
        "a_log_shape": (8,),
        "dt_proj_shape": (8,),
        "b_weight_shape": (8, 8),
        "c_weight_shape": (8, 8),
    }


def test_dispatch_op_native_routes_state_space(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((2, 5, 8), dtype=np.float32)
    ssm_a = np.zeros((8, 16), dtype=np.float32)
    ssm_b_weight = np.zeros((128, 8), dtype=np.float32)
    ssm_c_weight = np.zeros((8, 128), dtype=np.float32)
    ssm_d = np.zeros((8,), dtype=np.float32)
    ssm_dt_weight = np.zeros((8, 8), dtype=np.float32)
    ssm_dt_bias = np.zeros((8,), dtype=np.float32)

    result = dispatch_op_native(
        "state_space",
        x,
        ssm_a,
        ssm_b_weight,
        ssm_c_weight,
        ssm_d,
        ssm_dt_weight,
        ssm_dt_bias,
    )

    assert result == {
        "x_shape": (2, 5, 8),
        "ssm_a_shape": (8, 16),
        "ssm_b_weight_shape": (128, 8),
        "ssm_c_weight_shape": (8, 128),
        "ssm_d_shape": (8,),
        "ssm_dt_weight_shape": (8, 8),
        "ssm_dt_bias_shape": (8,),
    }


def test_dispatch_op_native_routes_gated_delta(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((2, 5, 8), dtype=np.float32)
    w = np.zeros((8, 8), dtype=np.float32)

    result = dispatch_op_native(
        "gated_delta",
        x,
        w,
        w,
        w,
        w,
        w,
        w,
        n_heads=4,
    )

    assert result == {
        "x_shape": (2, 5, 8),
        "q_weight_shape": (8, 8),
        "k_weight_shape": (8, 8),
        "v_weight_shape": (8, 8),
        "alpha_weight_shape": (8, 8),
        "beta_weight_shape": (8, 8),
        "o_weight_shape": (8, 8),
        "n_heads": 4,
    }


def test_check_native_op_support_keeps_known_ops_without_bridge(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch
    from research.synthesis.graph import ComputationGraph

    monkeypatch.setattr(native_dispatch, "_try_import_cython_bridge", lambda: None)

    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("state_space", [inp])
    graph.set_output(out)

    support = _check_native_op_support([graph], native_lib=None)

    assert "state_space" in support["supported"]
    assert "state_space" not in support["unsupported"]
    assert support["native_coverage"] == 1.0


def test_dispatch_op_native_routes_rwkv_time_mixing(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((2, 5, 8), dtype=np.float32)
    w_decay = np.zeros((8,), dtype=np.float32)
    u_bonus = np.zeros((8,), dtype=np.float32)
    w_k = np.zeros((8, 8), dtype=np.float32)
    w_v = np.zeros((8, 8), dtype=np.float32)
    w_r = np.zeros((8, 8), dtype=np.float32)

    result = dispatch_op_native(
        "rwkv_time_mixing",
        x,
        w_decay,
        u_bonus,
        w_k,
        w_v,
        w_r,
    )

    assert result == {
        "x_shape": (2, 5, 8),
        "w_decay_shape": (8,),
        "u_bonus_shape": (8,),
        "w_k_shape": (8, 8),
        "w_v_shape": (8, 8),
        "w_r_shape": (8, 8),
    }


def test_dispatch_op_native_routes_rmsnorm_3d(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch, "_try_import_cython_bridge", lambda: _FakeBridge()
    )

    x = np.zeros((2, 5, 8), dtype=np.float32)
    weight = np.zeros((8,), dtype=np.float32)

    result = dispatch_op_native("rmsnorm", x, weight, eps=1e-6)

    assert result == {
        "x_shape": (2, 5, 8),
        "weight_shape": (8,),
        "eps": 1e-6,
    }
