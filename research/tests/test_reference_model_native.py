from __future__ import annotations

import os

import pytest
import torch

from research.eval.reference_training import BaselineTransformer


pytestmark = pytest.mark.unit


def test_native_reference_forward_matches_legacy_path():
    torch.manual_seed(0)
    model = BaselineTransformer(vocab_size=128, d_model=32, n_layers=2).eval()
    input_ids = torch.randint(0, 128, (2, 16))

    os.environ["ARIA_DISABLE_REFERENCE_MODEL_NATIVE"] = "1"
    legacy = model(input_ids)
    os.environ.pop("ARIA_DISABLE_REFERENCE_MODEL_NATIVE", None)
    native = model(input_ids)

    torch.testing.assert_close(native, legacy, atol=1e-5, rtol=1e-5)
