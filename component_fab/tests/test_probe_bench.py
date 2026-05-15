"""Tests for the component_fab probe timing benchmark CLI."""

from __future__ import annotations

import json

import pytest

from component_fab.tools.run_probe_bench import ProbeSize, run_probe_bench


def test_probe_bench_writes_selected_probe_costs(tmp_path) -> None:
    out = tmp_path / "probe_costs.json"

    report = run_probe_bench(
        sizes=(ProbeSize(dim=8, seq_len=8, batch_size=2),),
        probes=("mix_speed", "s05_gate"),
        out=out,
        repeats=1,
        warmups=0,
        metric_trials=1,
        train_steps=2,
    )

    loaded = json.loads(out.read_text())
    assert loaded == report
    assert loaded["sizes"] == [{"dim": 8, "seq_len": 8, "batch_size": 2}]
    assert {row["probe"] for row in loaded["benchmarks"]} == {
        "mix_speed",
        "s05_gate",
    }
    for row in loaded["benchmarks"]:
        assert row["wall_ms_mean"] >= 0.0
        assert row["python_peak_bytes_max"] >= 0
        assert row["rss_delta_bytes_max"] >= 0
        assert row["output"]


def test_probe_bench_can_time_training_probe(tmp_path) -> None:
    out = tmp_path / "probe_costs.json"

    report = run_probe_bench(
        sizes=(ProbeSize(dim=8, seq_len=8, batch_size=2),),
        probes=("ar_easy",),
        out=out,
        repeats=1,
        warmups=0,
        metric_trials=1,
        train_steps=1,
    )

    row = report["benchmarks"][0]
    assert row["probe"] == "ar_easy"
    assert row["output"]["probe_name"] == "ar_easy"
    assert row["output"]["trained_successfully"] is True


def test_probe_bench_rejects_unknown_probe(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown probe"):
        run_probe_bench(
            sizes=(ProbeSize(dim=8, seq_len=8, batch_size=2),),
            probes=("not_a_probe",),
            out=tmp_path / "probe_costs.json",
        )
