"""Unit tests for the fab_daily_loop pre-run learning-state refresh."""

from __future__ import annotations

import pytest

import research.tools.fab_daily_loop as L


def test_format_refresh_section_skipped():
    assert L._format_refresh_section(None).startswith("_(")


def test_format_refresh_section_full_renders_signals():
    summary = {
        "axis_lift": {
            "global_pass_rate": 0.12,
            "top_knob_lifts": [
                {"value": "tensor_tucker", "lift": 1.8, "n": 9},
                {"value": "spectral_chebyshev", "lift": 1.2, "n": 5},
            ],
        },
        "failure_attribution": {
            "total_graded": 100,
            "total_promoted": 3,
            "total_rejected": 40,
            "over_eager_gates": ["nano_bind"],
            "gate_kill_rates": [
                {
                    "gate": "nano_bind",
                    "kill_rate": 0.9,
                    "killed": 36,
                    "reached": 40,
                    "over_eager": True,
                },
                {
                    "gate": "smoke",
                    "kill_rate": 0.0,
                    "killed": 0,
                    "reached": 100,
                    "over_eager": False,
                },
            ],
            "anchor_pool_size": 7,
        },
        "tier2_predictor": {"labels": 12, "labels_required": 60, "ready": False},
    }
    out = L._format_refresh_section(summary)
    assert "tensor_tucker=1.80" in out
    assert "global_pass=0.120" in out
    # over-eager gate surfaced as a promotion-blocker warning
    assert "over-eager gates" in out and "nano_bind" in out
    # only gates that were reached appear in the kill-rate line
    assert "smoke 0%" in out
    assert "12/60 labels" in out and "not ready" in out


def test_format_refresh_section_handles_failed_compute():
    summary = {
        "axis_lift": {"status": "failed: boom"},
        "failure_attribution": {"status": "failed: kaboom"},
        "tier2_predictor": {"labels": 0, "labels_required": 60, "ready": False},
    }
    out = L._format_refresh_section(summary)
    assert "failed: boom" in out
    assert "failed: kaboom" in out


def test_tier2_label_count(tmp_path, monkeypatch):
    p = tmp_path / "labels.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    monkeypatch.setattr(L, "_TIER2_LABELS_PATH", p)
    assert L._tier2_label_count() == 2  # blank line ignored


def test_tier2_label_count_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_TIER2_LABELS_PATH", tmp_path / "nope.jsonl")
    assert L._tier2_label_count() == 0


def test_refresh_returns_three_loops(monkeypatch):
    """_phase_refresh_learning_state always reports all three loops, even when
    the heavy compute paths are stubbed out."""

    monkeypatch.setattr(L, "_tier2_label_count", lambda: 5)

    def _boom(*_a, **_k):
        raise RuntimeError("stubbed")

    # Force the lazy-imported compute funcs to fail → exercises the survive path.
    import component_fab.state.axis_lift as ax
    import component_fab.state.failure_attribution as fa

    monkeypatch.setattr(ax, "compute_axis_lift", _boom)
    monkeypatch.setattr(fa, "compute_failure_attribution", _boom)

    summary = L._phase_refresh_learning_state(quiet=True)
    assert set(summary) == {"axis_lift", "failure_attribution", "tier2_predictor"}
    assert summary["axis_lift"]["status"].startswith("failed:")
    assert summary["tier2_predictor"]["labels"] == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
