from unittest.mock import patch

import pytest

from research.eval.fingerprint import BehavioralFingerprint, compute_gated_fingerprint

pytestmark = pytest.mark.unit


def test_compute_gated_fingerprint_skips_full_below_structural_floor():
    """Below structural floor → return lightning fp, no full fingerprint."""
    light = BehavioralFingerprint(
        novelty_score=0.05, quality="partial", analyses_succeeded=0
    )

    with patch(
        "research.eval.fingerprint.compute_lightning_fingerprint", return_value=light
    ) as light_mock:
        with patch("research.eval.fingerprint.compute_fingerprint") as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=True,
                structural_floor=0.10,
            )

    assert fp is light
    assert full_ran is False
    light_mock.assert_called_once()
    full_mock.assert_not_called()


def test_compute_gated_fingerprint_runs_full_above_structural_floor():
    """Above structural floor → run full (deferred) fingerprint."""
    light = BehavioralFingerprint(
        novelty_score=0.4, quality="partial", analyses_succeeded=0
    )
    full = BehavioralFingerprint(
        novelty_score=0.3,
        quality="partial",
        analyses_succeeded=0,
        cka_source="deferred",
        novelty_valid_for_promotion=False,
    )

    with patch(
        "research.eval.fingerprint.compute_lightning_fingerprint", return_value=light
    ) as light_mock:
        with patch(
            "research.eval.fingerprint.compute_fingerprint", return_value=full
        ) as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=True,
                structural_floor=0.10,
            )

    assert fp is full
    assert full_ran is True
    light_mock.assert_called_once()
    full_mock.assert_called_once()
    # Verify deferred flags passed through
    call_kwargs = full_mock.call_args
    assert call_kwargs.kwargs.get("include_cka") is False
    assert call_kwargs.kwargs.get("include_behavioral_probes") is False


def test_compute_gated_fingerprint_bypass_runs_full_only():
    """Gate disabled → run full fingerprint directly (still deferred)."""
    full = BehavioralFingerprint(
        novelty_score=0.3,
        quality="partial",
        analyses_succeeded=0,
        cka_source="deferred",
    )

    with patch("research.eval.fingerprint.compute_lightning_fingerprint") as light_mock:
        with patch(
            "research.eval.fingerprint.compute_fingerprint", return_value=full
        ) as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=False,
            )

    assert fp is full
    assert full_ran is True
    light_mock.assert_not_called()
    full_mock.assert_called_once()
    # Verify deferred flags
    call_kwargs = full_mock.call_args
    assert call_kwargs.kwargs.get("include_cka") is False
    assert call_kwargs.kwargs.get("include_behavioral_probes") is False


def test_compute_gated_fingerprint_force_lightning_only():
    """force_lightning_only=True → return lightning fp regardless of score."""
    light = BehavioralFingerprint(
        novelty_score=0.8, quality="partial", analyses_succeeded=0
    )

    with patch(
        "research.eval.fingerprint.compute_lightning_fingerprint", return_value=light
    ) as light_mock:
        with patch("research.eval.fingerprint.compute_fingerprint") as full_mock:
            fp, full_ran = compute_gated_fingerprint(
                model=object(),
                device="cpu",
                full_gate_enabled=True,
                force_lightning_only=True,
                structural_floor=0.10,
            )

    assert fp is light
    assert full_ran is False
    light_mock.assert_called_once()
    full_mock.assert_not_called()
