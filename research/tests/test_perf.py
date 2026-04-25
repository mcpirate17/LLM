from __future__ import annotations

from research.scientist import perf as perf_mod
from research.scientist.perf import GPUStarvationDetector


def test_gpu_starvation_detector_uses_wall_clock_waits(monkeypatch):
    clock = iter([10.0, 10.002])
    monkeypatch.setattr(perf_mod.time, "perf_counter", lambda: next(clock))

    detector = GPUStarvationDetector(threshold_ms=0.1)
    detector.start_wait()
    duration_ms = detector.end_wait()

    assert duration_ms is not None
    assert duration_ms >= 0.1
    summary = detector.get_summary()
    assert summary["starvation_detected"] is True
    assert summary["count"] == 1
