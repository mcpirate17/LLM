from __future__ import annotations

import ctypes
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.native


class NrCompileRequest(ctypes.Structure):
    _fields_ = [
        ("ir_json", ctypes.c_char_p),
        ("ir_json_len", ctypes.c_int64),
        ("vocab_size", ctypes.c_int32),
        ("max_seq_len", ctypes.c_int32),
    ]


class NrCompileResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("model_handle", ctypes.c_int64),
        ("message", ctypes.c_char_p),
    ]


class NrExecuteRequest(ctypes.Structure):
    _fields_ = [
        ("model_handle", ctypes.c_int64),
        ("token_ids", ctypes.POINTER(ctypes.c_int32)),
        ("batch", ctypes.c_int32),
        ("seq_len", ctypes.c_int32),
    ]


class NrExecuteResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int32),
        ("message", ctypes.c_char_p),
    ]


class NrExecuteBatchResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("batch", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("message", ctypes.c_char_p),
    ]


class NrCapability(ctypes.Structure):
    _fields_ = [
        ("supported_ops", ctypes.POINTER(ctypes.c_char_p)),
        ("n_supported", ctypes.c_int32),
        ("unsupported_ops", ctypes.POINTER(ctypes.c_char_p)),
        ("n_unsupported", ctypes.c_int32),
    ]


class NrTrainTensorF32(ctypes.Structure):
    _fields_ = [
        ("param", ctypes.POINTER(ctypes.c_float)),
        ("grad", ctypes.POINTER(ctypes.c_float)),
        ("momentum", ctypes.POINTER(ctypes.c_float)),
        ("exp_avg", ctypes.POINTER(ctypes.c_float)),
        ("exp_avg_sq", ctypes.POINTER(ctypes.c_float)),
        ("numel", ctypes.c_int64),
    ]


class NrOptimizerStepRequest(ctypes.Structure):
    _fields_ = [
        ("optimizer", ctypes.c_int32),
        ("tensors", ctypes.POINTER(NrTrainTensorF32)),
        ("n_tensors", ctypes.c_int32),
        ("learning_rate", ctypes.c_double),
        ("momentum", ctypes.c_double),
        ("beta1", ctypes.c_double),
        ("beta2", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("weight_decay", ctypes.c_double),
        ("max_grad_norm", ctypes.c_double),
        ("nesterov", ctypes.c_int32),
        ("step", ctypes.c_int64),
    ]


class NrOptimizerStepResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("grad_norm", ctypes.c_double),
        ("elements", ctypes.c_int64),
        ("message", ctypes.c_char_p),
    ]


def _load_native_lib():
    lib_path = (
        Path(__file__).resolve().parents[1]
        / "runtime"
        / "native"
        / "build"
        / "libaria_native_runtime.so"
    )
    if not lib_path.exists():
        pytest.skip(f"native runtime library not built: {lib_path}")
    return ctypes.CDLL(str(lib_path))


def test_runner_abi_optimizer_clip_step_sgd_updates_flat_spans():
    lib = _load_native_lib()
    lib.nr_optimizer_clip_step_f32.argtypes = [ctypes.POINTER(NrOptimizerStepRequest)]
    lib.nr_optimizer_clip_step_f32.restype = NrOptimizerStepResponse

    param_a = (ctypes.c_float * 3)(1.0, -2.0, 3.0)
    grad_a = (ctypes.c_float * 3)(3.0, 4.0, 0.0)
    param_b = (ctypes.c_float * 1)(0.5)
    grad_b = (ctypes.c_float * 1)(0.0)
    tensors = (NrTrainTensorF32 * 2)(
        NrTrainTensorF32(param_a, grad_a, None, None, None, 3),
        NrTrainTensorF32(param_b, grad_b, None, None, None, 1),
    )
    req = NrOptimizerStepRequest(
        optimizer=1,
        tensors=tensors,
        n_tensors=2,
        learning_rate=0.1,
        momentum=0.0,
        beta1=0.0,
        beta2=0.0,
        eps=1.0e-8,
        weight_decay=0.0,
        max_grad_norm=2.0,
        nesterov=0,
        step=0,
    )

    res = lib.nr_optimizer_clip_step_f32(ctypes.byref(req))

    scale = 2.0 / (5.0 + 1.0e-6)
    assert int(res.status) == 0
    assert int(res.elements) == 4
    assert float(res.grad_norm) == pytest.approx(5.0, rel=1e-6)
    assert list(param_a) == pytest.approx(
        [1.0 - 0.1 * 3.0 * scale, -2.0 - 0.1 * 4.0 * scale, 3.0],
        rel=1e-6,
    )
    assert list(param_b) == pytest.approx([0.5], rel=1e-6)


def test_runner_abi_optimizer_clip_step_adamw_updates_state_and_params():
    lib = _load_native_lib()
    lib.nr_optimizer_clip_step_f32.argtypes = [ctypes.POINTER(NrOptimizerStepRequest)]
    lib.nr_optimizer_clip_step_f32.restype = NrOptimizerStepResponse

    param = (ctypes.c_float * 2)(1.0, -2.0)
    grad = (ctypes.c_float * 2)(0.5, -0.25)
    exp_avg = (ctypes.c_float * 2)(0.0, 0.0)
    exp_avg_sq = (ctypes.c_float * 2)(0.0, 0.0)
    tensors = (NrTrainTensorF32 * 1)(
        NrTrainTensorF32(param, grad, None, exp_avg, exp_avg_sq, 2),
    )
    req = NrOptimizerStepRequest(
        optimizer=2,
        tensors=tensors,
        n_tensors=1,
        learning_rate=0.01,
        momentum=0.0,
        beta1=0.9,
        beta2=0.999,
        eps=1.0e-8,
        weight_decay=0.1,
        max_grad_norm=10.0,
        nesterov=0,
        step=1,
    )

    res = lib.nr_optimizer_clip_step_f32(ctypes.byref(req))

    expected_params = [1.0, -2.0]
    expected_exp_avg = []
    expected_exp_avg_sq = []
    for idx, g in enumerate([0.5, -0.25]):
        p = expected_params[idx] * (1.0 - 0.01 * 0.1)
        m = 0.9 * 0.0 + 0.1 * g
        v = 0.999 * 0.0 + 0.001 * g * g
        denom = math.sqrt(v) / math.sqrt(1.0 - 0.999**1) + 1.0e-8
        p -= (0.01 / (1.0 - 0.9**1)) * m / denom
        expected_params[idx] = p
        expected_exp_avg.append(m)
        expected_exp_avg_sq.append(v)

    assert int(res.status) == 0
    assert int(res.elements) == 2
    assert float(res.grad_norm) == pytest.approx(math.sqrt(0.5**2 + 0.25**2), rel=1e-6)
    assert list(param) == pytest.approx(expected_params, rel=1e-6)
    assert list(exp_avg) == pytest.approx(expected_exp_avg, rel=1e-6)
    assert list(exp_avg_sq) == pytest.approx(expected_exp_avg_sq, rel=1e-6, abs=1e-8)


def test_runner_abi_lifecycle_compile_execute_smoke():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_set_strict_mode.argtypes = [ctypes.c_int32]
    lib.nr_set_strict_mode.restype = ctypes.c_int32
    assert lib.nr_set_strict_mode(0) == 0

    lib.nr_query_capabilities.argtypes = [ctypes.POINTER(NrCapability)]
    lib.nr_query_capabilities.restype = ctypes.c_int32
    cap = NrCapability()
    assert lib.nr_query_capabilities(ctypes.byref(cap)) == 0
    assert int(cap.n_supported) > 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n0","target":"n2"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":32,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=128,
        max_seq_len=16,
    )
    compile_res = lib.nr_compile(ctypes.byref(req))
    assert int(compile_res.status) == 0
    assert int(compile_res.model_handle) > 0

    lib.nr_execute.argtypes = [ctypes.POINTER(NrExecuteRequest)]
    lib.nr_execute.restype = NrExecuteResponse

    token_buf = (ctypes.c_int32 * 4)(1, 2, 3, 4)
    exec_req = NrExecuteRequest(
        model_handle=compile_res.model_handle,
        token_ids=token_buf,
        batch=1,
        seq_len=4,
    )
    exec_res = lib.nr_execute(ctypes.byref(exec_req))
    assert int(exec_res.status) == 0
    assert int(exec_res.vocab_size) == 128
    assert bool(exec_res.logits)
    logits = [float(exec_res.logits[i]) for i in range(exec_res.vocab_size)]
    non_target_max = max(
        logits[i] for i in range(exec_res.vocab_size) if i not in {1, 2, 3, 4}
    )
    for idx in [1, 2, 3, 4]:
        assert logits[idx] > non_target_max

    # Deterministic replay for same handle + tokens.
    exec_res_2 = lib.nr_execute(ctypes.byref(exec_req))
    assert int(exec_res_2.status) == 0
    logits_2 = [float(exec_res_2.logits[i]) for i in range(exec_res_2.vocab_size)]
    assert logits == logits_2

    lib.nr_execute_batch.argtypes = [ctypes.POINTER(NrExecuteRequest)]
    lib.nr_execute_batch.restype = NrExecuteBatchResponse
    batch_token_buf = (ctypes.c_int32 * 8)(1, 2, 3, 4, 4, 3, 2, 1)
    batch_req = NrExecuteRequest(
        model_handle=compile_res.model_handle,
        token_ids=batch_token_buf,
        batch=2,
        seq_len=4,
    )
    batch_res = lib.nr_execute_batch(ctypes.byref(batch_req))
    assert int(batch_res.status) == 0
    assert int(batch_res.batch) == 2
    assert int(batch_res.vocab_size) == 128
    assert bool(batch_res.logits)
    row0 = [float(batch_res.logits[i]) for i in range(128)]
    row1 = [float(batch_res.logits[128 + i]) for i in range(128)]
    assert row0 == logits
    assert row1 != row0

    lib.nr_release_model.argtypes = [ctypes.c_int64]
    lib.nr_release_model(compile_res.model_handle)

    released_res = lib.nr_execute(ctypes.byref(exec_req))
    assert int(released_res.status) == -4

    lib.nr_runtime_shutdown()


def test_runner_abi_strict_mode_rejects_marked_ir():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_set_strict_mode.argtypes = [ctypes.c_int32]
    lib.nr_set_strict_mode.restype = ctypes.c_int32
    assert lib.nr_set_strict_mode(1) == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    # Minimal marker-based strict-mode rejection path in smoke implementation.
    ir = b'{"schema_version":"native_ir.v1","unsupported":true}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    compile_res = lib.nr_compile(ctypes.byref(req))
    assert int(compile_res.status) == -6

    lib.nr_get_fallback_count.restype = ctypes.c_int64
    assert int(lib.nr_get_fallback_count()) >= 1

    lib.nr_set_strict_mode(0)
    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_unsupported_schema_and_seq_overflow():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse
    bad_ir = b'{"schema_version":"native_ir.v0","nodes":[]}'
    bad_req = NrCompileRequest(
        ir_json=ctypes.c_char_p(bad_ir),
        ir_json_len=len(bad_ir),
        vocab_size=16,
        max_seq_len=4,
    )
    bad_res = lib.nr_compile(ctypes.byref(bad_req))
    assert int(bad_res.status) == -2

    good_ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n0","target":"n2"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":8,"output_node_id":"n7"}'
    good_req = NrCompileRequest(
        ir_json=ctypes.c_char_p(good_ir),
        ir_json_len=len(good_ir),
        vocab_size=32,
        max_seq_len=2,
    )
    good_res = lib.nr_compile(ctypes.byref(good_req))
    assert int(good_res.status) == 0

    lib.nr_execute.argtypes = [ctypes.POINTER(NrExecuteRequest)]
    lib.nr_execute.restype = NrExecuteResponse
    token_buf = (ctypes.c_int32 * 3)(4, 5, 6)
    overflow_req = NrExecuteRequest(
        model_handle=good_res.model_handle,
        token_ids=token_buf,
        batch=1,
        seq_len=3,
    )
    overflow_res = lib.nr_execute(ctypes.byref(overflow_req))
    assert int(overflow_res.status) == -1

    lib.nr_release_model.argtypes = [ctypes.c_int64]
    lib.nr_release_model(good_res.model_handle)
    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_unsupported_graph_family():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    # First-family ABI supports exp+add+mul+matmul+linear+softmax+rmsnorm+sub graph nodes only for now.
    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"matmul","input_ids":[],"config":{},"is_input":true,"is_output":true}],"edges":[],"model_dim":16,"output_node_id":"n0"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_add_mul_matmul_linear_softmax_rmsnorm_sub_without_exp_family():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"relu","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n0","target":"n2"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_out_of_order_exp_add_mul_matmul_linear_softmax_rmsnorm_sub_family():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"add","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"exp","input_ids":["n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n0","target":"n2"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_unlinked_exp_add_mul_matmul_linear_softmax_rmsnorm_sub_family():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n0"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n0","target":"n2"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n0","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_transitively_linked_exp_add_mul_matmul_linear_softmax_rmsnorm_sub_family():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"b1","op_name":"relu","input_ids":["n0"],"config":{},"is_input":false,"is_output":false},{"id":"n1","op_name":"add","input_ids":["b1","b1"],"config":{},"is_input":false,"is_output":false},{"id":"b2","op_name":"relu","input_ids":["n1"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["b2","n0"],"config":{},"is_input":false,"is_output":false},{"id":"b3","op_name":"relu","input_ids":["n2"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["b3","n1"],"config":{},"is_input":false,"is_output":false},{"id":"b4","op_name":"relu","input_ids":["n3"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["b4"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"b5","op_name":"relu","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["b5"],"config":{},"is_input":false,"is_output":false},{"id":"b6","op_name":"relu","input_ids":["n5"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["b6"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"b7","op_name":"relu","input_ids":["n6"],"config":{},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["b7","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"b1"},{"source":"b1","target":"n1"},{"source":"n1","target":"b2"},{"source":"b2","target":"n2"},{"source":"n2","target":"b3"},{"source":"b3","target":"n3"},{"source":"n3","target":"b4"},{"source":"b4","target":"n4"},{"source":"n4","target":"b5"},{"source":"b5","target":"n5"},{"source":"n5","target":"b6"},{"source":"b6","target":"n6"},{"source":"n6","target":"b7"},{"source":"b7","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_edges_break_family_ancestry_even_if_input_ids_linked():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n2","target":"n3"},{"source":"n0","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_edges_declared_empty_even_if_input_ids_linked():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_required_family_marker_is_duplicated():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true},{"id":"n8","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_required_link_exists_only_in_edges_not_input_ids():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n0"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n2","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_required_link_is_duplicated_in_input_ids():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n0","target":"n2"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_required_link_is_duplicated_in_explicit_edges():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n2","target":"n3"},{"source":"n2","target":"n3"},{"source":"n1","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"},{"source":"n1","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()


def test_runner_abi_rejects_when_required_chain_input_refs_missing_node_id():
    lib = _load_native_lib()

    lib.nr_runtime_init.restype = ctypes.c_int32
    assert lib.nr_runtime_init() == 0

    lib.nr_compile.argtypes = [ctypes.POINTER(NrCompileRequest)]
    lib.nr_compile.restype = NrCompileResponse

    ir = b'{"schema_version":"native_ir.v1","nodes":[{"id":"n0","op_name":"exp","input_ids":[],"config":{},"is_input":true,"is_output":false},{"id":"n1","op_name":"add","input_ids":["n0"],"config":{},"is_input":false,"is_output":false},{"id":"n2","op_name":"mul","input_ids":["n1","n0"],"config":{},"is_input":false,"is_output":false},{"id":"n3","op_name":"matmul","input_ids":["n2","n1"],"config":{},"is_input":false,"is_output":false},{"id":"n4","op_name":"linear","input_ids":["n3","n_missing"],"config":{"in_dim":2,"out_dim":1},"is_input":false,"is_output":false},{"id":"n5","op_name":"softmax","input_ids":["n4"],"config":{},"is_input":false,"is_output":false},{"id":"n6","op_name":"rmsnorm","input_ids":["n5"],"config":{"eps":1.0e-5},"is_input":false,"is_output":false},{"id":"n7","op_name":"sub","input_ids":["n6","n1"],"config":{},"is_input":false,"is_output":true}],"edges":[{"source":"n0","target":"n1"},{"source":"n1","target":"n2"},{"source":"n2","target":"n3"},{"source":"n3","target":"n4"},{"source":"n4","target":"n5"},{"source":"n5","target":"n6"},{"source":"n6","target":"n7"}],"model_dim":16,"output_node_id":"n7"}'
    req = NrCompileRequest(
        ir_json=ctypes.c_char_p(ir),
        ir_json_len=len(ir),
        vocab_size=64,
        max_seq_len=8,
    )
    res = lib.nr_compile(ctypes.byref(req))
    assert int(res.status) == -3

    lib.nr_runtime_shutdown()
