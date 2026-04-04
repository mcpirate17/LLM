from __future__ import annotations

import time

from research.scientist.perf import GPUStarvationDetector


def test_gpu_starvation_detector_uses_wall_clock_waits():
    detector = GPUStarvationDetector(threshold_ms=0.1)
    detector.start_wait()
    time.sleep(0.002)
    duration_ms = detector.end_wait()

    assert duration_ms is not None
    assert duration_ms >= 0.1
    summary = detector.get_summary()
    assert summary["starvation_detected"] is True
    assert summary["count"] == 1
