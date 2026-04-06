from __future__ import annotations

import os
import ctypes
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from research.eval.sandbox import safe_eval
from research.scientist.native.abi import (
    _NrExecuteBatchResponse,
    NativeRunnerAbiSession,
    _NrExecuteResponse,
    _build_native_abi_only_model,
)

pytestmark = pytest.mark.native


@contextmanager
def _env(**values):
    prev = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


class _GoodModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, dim: int = 32):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.linear = torch.nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.linear(self.embed(x))


def test_safe_eval_native_abi_probe_success():
    vocab_size = 128
    model = _GoodModel(vocab_size=vocab_size)

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            assert isinstance(token_ids, list)
            assert batch == 2
            out = [0.0] * vocab_size
            for i in token_ids:
                out[int(i) % vocab_size] += 1.0
            return out

    model._native_runner_abi_session = _Session()

    with _env(NATIVE_RUNNER_ABI_INFER_PROBE="1"):
        result = safe_eval(
            model,
            batch_size=2,
            seq_len=8,
            vocab_size=vocab_size,
            device="cpu",
            run_stability_probe=False,
        )

    assert result.passed is True
    probe = result.native_abi_probe or {}
    assert probe.get("attempted") is True
    assert probe.get("succeeded") is True
    assert probe.get("reason") == "ok"
    assert int(probe.get("vocab_size")) == vocab_size


def test_safe_eval_native_abi_probe_failure_is_non_fatal():
    vocab_size = 64
    model = _GoodModel(vocab_size=vocab_size)

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            raise RuntimeError("boom")

    model._native_runner_abi_session = _Session()

    with _env(NATIVE_RUNNER_ABI_INFER_PROBE="1"):
        result = safe_eval(
            model,
            batch_size=2,
            seq_len=6,
            vocab_size=vocab_size,
            device="cpu",
            run_stability_probe=False,
        )

    assert result.passed is True
    probe = result.native_abi_probe or {}
    assert probe.get("attempted") is True
    assert probe.get("succeeded") is False
    assert str(probe.get("reason", "")).startswith("execute_error:")


def test_safe_eval_native_abi_probe_prefers_tensor_execute_path():
    vocab_size = 32
    model = _GoodModel(vocab_size=vocab_size)

    class _Session:
        def __init__(self):
            self.tensor_calls = 0
            self.list_calls = 0

        def execute_tokens_tensor(self, token_ids, batch=1):
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.device.type == "cpu"
            self.tensor_calls += 1
            return [0.0] * vocab_size

        def execute_tokens(self, token_ids, batch=1):
            self.list_calls += 1
            return [0.0] * vocab_size

    session = _Session()
    model._native_runner_abi_session = session

    with _env(NATIVE_RUNNER_ABI_INFER_PROBE="1"):
        result = safe_eval(
            model,
            batch_size=2,
            seq_len=8,
            vocab_size=vocab_size,
            device="cpu",
            run_stability_probe=False,
        )

    assert result.passed is True
    assert session.tensor_calls == 1
    assert session.list_calls == 0


def test_safe_eval_native_abi_primary_forward_only_uses_session_logits():
    vocab_size = 32

    class _BadShapeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # Intentionally wrong output rank; primary ABI mode should bypass this.
            return torch.zeros((x.shape[0], vocab_size), dtype=torch.float32)

    model = _BadShapeModel()

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [float(i % 7) for i in range(vocab_size)]

    model._native_runner_abi_session = _Session()

    with _env(NATIVE_RUNNER_ABI_INFER_PROBE="1", NATIVE_RUNNER_ABI_INFER_PRIMARY="1"):
        result = safe_eval(
            model,
            batch_size=2,
            seq_len=5,
            vocab_size=vocab_size,
            device="cpu",
            run_stability_probe=False,
        )

    assert result.passed is True
    probe = result.native_abi_probe or {}
    assert probe.get("primary_requested") is True
    assert probe.get("primary_used") is True
    assert probe.get("mode") == "primary_forward_only"


def test_safe_eval_native_abi_primary_parity_strict_passes_on_match():
    vocab_size = 16
    base = torch.arange(vocab_size, dtype=torch.float32) * 0.1

    class _MatchModel(torch.nn.Module):
        def forward(self, x):
            return base.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1).contiguous()

    model = _MatchModel()

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [float(v) for v in base.tolist()]

    model._native_runner_abi_session = _Session()

    with _env(
        NATIVE_RUNNER_ABI_INFER_PROBE="1",
        NATIVE_RUNNER_ABI_INFER_PRIMARY="1",
        NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE="1.0",
        NATIVE_RUNNER_ABI_PARITY_MAX_ABS="1e-6",
        NATIVE_RUNNER_ABI_PARITY_STRICT="1",
    ):
        result = safe_eval(
            model,
            batch_size=2,
            seq_len=4,
            vocab_size=vocab_size,
            device="cpu",
            run_stability_probe=False,
        )

    assert result.passed is True
    probe = result.native_abi_probe or {}
    assert probe.get("parity_attempted") is True
    assert probe.get("parity_pass") is True


def test_safe_eval_native_abi_primary_parity_strict_fails_on_drift():
    vocab_size = 16

    class _DriftModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(
                (x.shape[0], x.shape[1], vocab_size), dtype=torch.float32
            )

    model = _DriftModel()

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [1.0] * vocab_size

    model._native_runner_abi_session = _Session()

    with _env(
        NATIVE_RUNNER_ABI_INFER_PROBE="1",
        NATIVE_RUNNER_ABI_INFER_PRIMARY="1",
        NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE="1.0",
        NATIVE_RUNNER_ABI_PARITY_MAX_ABS="1e-4",
        NATIVE_RUNNER_ABI_PARITY_STRICT="1",
    ):
        result = safe_eval(
            model,
            batch_size=2,
            seq_len=4,
            vocab_size=vocab_size,
            device="cpu",
            run_stability_probe=False,
        )

    assert result.passed is False
    assert result.error_type == "abi_parity_regression"
    probe = result.native_abi_probe or {}
    assert probe.get("parity_attempted") is True
    assert probe.get("parity_pass") is False


def test_native_runner_abi_session_respects_batch_seq_len_and_caches():
    class _FakeLib:
        def __init__(self):
            self.calls = []
            self._buffers = []

        def nr_execute(self, req_ptr):
            req = req_ptr._obj
            token_count = int(req.batch) * int(req.seq_len)
            tokens = [int(req.token_ids[i]) for i in range(token_count)]
            self.calls.append(
                {
                    "batch": int(req.batch),
                    "seq_len": int(req.seq_len),
                    "tokens": tokens,
                }
            )
            buf = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
            self._buffers.append(buf)
            return _NrExecuteResponse(
                status=0,
                logits=buf,
                vocab_size=4,
                message=None,
            )

        def nr_release_model(self, handle):
            return None

    lib = _FakeLib()
    session = NativeRunnerAbiSession(
        native_lib=lib,
        model_handle=1,
        vocab_size=4,
        max_seq_len=8,
    )

    logits = session.execute_tokens([5, 6, 7, 8], batch=2)
    assert logits == [1.0, 2.0, 3.0, 4.0]
    assert lib.calls == [{"batch": 2, "seq_len": 2, "tokens": [5, 6, 7, 8]}]

    logits_cached = session.execute_tokens([5, 6, 7, 8], batch=2)
    assert logits_cached == logits
    assert len(lib.calls) == 1


def test_native_runner_abi_session_execute_tokens_tensor_uses_buffer_pointer():
    class _FakeLib:
        def __init__(self):
            self.calls = []
            self._buffers = []

        def nr_execute(self, req_ptr):
            req = req_ptr._obj
            token_count = int(req.batch) * int(req.seq_len)
            tokens = [int(req.token_ids[i]) for i in range(token_count)]
            self.calls.append(
                {
                    "batch": int(req.batch),
                    "seq_len": int(req.seq_len),
                    "tokens": tokens,
                }
            )
            buf = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
            self._buffers.append(buf)
            return _NrExecuteResponse(
                status=0,
                logits=buf,
                vocab_size=4,
                message=None,
            )

        def nr_release_model(self, handle):
            return None

    lib = _FakeLib()
    session = NativeRunnerAbiSession(
        native_lib=lib,
        model_handle=1,
        vocab_size=4,
        max_seq_len=8,
    )

    logits = session.execute_tokens_tensor(torch.tensor([5, 6, 7, 8]), batch=2)
    assert logits == [1.0, 2.0, 3.0, 4.0]
    assert lib.calls == [{"batch": 2, "seq_len": 2, "tokens": [5, 6, 7, 8]}]


def test_build_native_abi_only_model_uses_row_execution_path():
    class _Session:
        def __init__(self):
            self.rows = None
            self.tensor_rows = None

        def execute_tokens(self, token_ids, batch=1):
            raise AssertionError("per-row execute_tokens path should not be used")

        def execute_token_rows_tensor(self, token_rows):
            self.tensor_rows = token_rows.clone()
            return [
                (10.0, 11.0, 12.0),
                (20.0, 21.0, 22.0),
            ]

        def execute_token_rows(self, token_rows):
            self.rows = [tuple(int(v) for v in row) for row in token_rows]
            return [
                (10.0, 11.0, 12.0),
                (20.0, 21.0, 22.0),
            ]

    session = _Session()
    model = _build_native_abi_only_model(session, vocab_size=3)
    input_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    out = model(input_ids)

    assert session.rows is None
    assert torch.equal(session.tensor_rows, input_ids)
    assert tuple(out.shape) == (2, 2, 3)
    assert torch.allclose(out[0, 0], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.allclose(out[1, 1], torch.tensor([20.0, 21.0, 22.0]))


def test_native_runner_abi_session_uses_native_batch_execute_for_rows():
    class _FakeLib:
        def __init__(self):
            self.batch_calls = []
            self._buffers = []

        def nr_execute_batch(self, req_ptr):
            req = req_ptr._obj
            token_count = int(req.batch) * int(req.seq_len)
            tokens = [int(req.token_ids[i]) for i in range(token_count)]
            self.batch_calls.append(
                {
                    "batch": int(req.batch),
                    "seq_len": int(req.seq_len),
                    "tokens": tokens,
                }
            )
            buf = (ctypes.c_float * 12)(
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            )
            self._buffers.append(buf)
            return _NrExecuteBatchResponse(
                status=0,
                logits=buf,
                batch=2,
                vocab_size=4,
                message=None,
            )

        def nr_execute(self, req_ptr):
            raise AssertionError("single execute path should not be used")

        def nr_release_model(self, handle):
            return None

    lib = _FakeLib()
    session = NativeRunnerAbiSession(
        native_lib=lib,
        model_handle=1,
        vocab_size=4,
        max_seq_len=8,
    )

    rows = session.execute_token_rows([(1, 2), (3, 4), (1, 2)])
    assert lib.batch_calls == [{"batch": 2, "seq_len": 2, "tokens": [1, 2, 3, 4]}]
    assert rows == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
        (1.0, 2.0, 3.0, 4.0),
    ]


def test_native_runner_abi_session_uses_native_batch_execute_for_row_tensor():
    class _FakeLib:
        def __init__(self):
            self.batch_calls = []
            self._buffers = []

        def nr_execute_batch(self, req_ptr):
            req = req_ptr._obj
            token_count = int(req.batch) * int(req.seq_len)
            tokens = [int(req.token_ids[i]) for i in range(token_count)]
            self.batch_calls.append(
                {
                    "batch": int(req.batch),
                    "seq_len": int(req.seq_len),
                    "tokens": tokens,
                }
            )
            buf = (ctypes.c_float * 8)(
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            )
            self._buffers.append(buf)
            return _NrExecuteBatchResponse(
                status=0,
                logits=buf,
                batch=2,
                vocab_size=4,
                message=None,
            )

        def nr_release_model(self, handle):
            return None

    lib = _FakeLib()
    session = NativeRunnerAbiSession(
        native_lib=lib,
        model_handle=1,
        vocab_size=4,
        max_seq_len=8,
    )

    rows = session.execute_token_rows_tensor(torch.tensor([[1, 2], [3, 4]]))
    assert lib.batch_calls == [{"batch": 2, "seq_len": 2, "tokens": [1, 2, 3, 4]}]
    assert rows == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ]
