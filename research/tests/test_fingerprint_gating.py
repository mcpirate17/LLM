from unittest.mock import patch

import pytest

from research.eval.fingerprint import BehavioralFingerprint, compute_gated_fingerprint

pytestmark = pytest.mark.unit


def test_compute_gated_fingerprint_skips_full_below_threshold():
    light = BehavioralFingerprint(novelty_score=0.2, quality="partial", analyses_succeeded=1)

    with patch("research.eval.fingerprint.compute_lightning_fingerprint", return_value=light) as light_mock:
        with patch("research.eval.fingerprint.compute_fingerprint") as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=True,
                lightning_novelty_threshold=0.4,
            )

    assert fp is light
    assert full_ran is False
    light_mock.assert_called_once()
    full_mock.assert_not_called()


def test_compute_gated_fingerprint_runs_full_above_threshold():
    light = BehavioralFingerprint(novelty_score=0.8, quality="partial", analyses_succeeded=1)
    full = BehavioralFingerprint(novelty_score=0.9, quality="full", analyses_succeeded=4)

    with patch("research.eval.fingerprint.compute_lightning_fingerprint", return_value=light) as light_mock:
        with patch("research.eval.fingerprint.compute_fingerprint", return_value=full) as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=True,
                lightning_novelty_threshold=0.4,
            )

    assert fp is full
    assert full_ran is True
    light_mock.assert_called_once()
    full_mock.assert_called_once()


def test_compute_gated_fingerprint_bypass_runs_full_only():
    full = BehavioralFingerprint(novelty_score=0.7, quality="full", analyses_succeeded=4)

    with patch("research.eval.fingerprint.compute_lightning_fingerprint") as light_mock:
        with patch("research.eval.fingerprint.compute_fingerprint", return_value=full) as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=False,
            )

    assert fp is full
    assert full_ran is True
    light_mock.assert_not_called()
    full_mock.assert_called_once()
