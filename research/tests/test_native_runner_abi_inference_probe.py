from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from research.eval.sandbox import safe_eval

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
