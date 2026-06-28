from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(slots=True)
class TrainingProfileArtifacts:
    output_dir: str
    summary_json: str
    trace_json: Optional[str]


class _Trace:
    """Reusable timing context for ``TrainingRunProfiler.trace(name)``.

    Module-level class so each ``trace()`` call avoids the per-call class-body
    allocation cost of the original closure-based implementation.
    """

    __slots__ = ("_profiler", "_name", "_t0")

    def __init__(self, profiler: "TrainingRunProfiler", name: str):
        self._profiler = profiler
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_Trace":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        if self._profiler.enabled:
            self._profiler.record_timing(
                self._name, (time.perf_counter() - self._t0) * 1000.0
            )
        return False


class TrainingRunProfiler:
    """Guarded short-run profiler for the training loop."""

    def __init__(self, config: Any, device: torch.device):
        self.enabled = bool(getattr(config, "profile_enabled", False))
        self.device = device
        self.output_dir: Optional[Path] = None
        self.summary_path: Optional[Path] = None
        self.trace_path: Optional[Path] = None
        self._timings_ms: dict[str, list[float]] = defaultdict(list)
        self._step_rows: list[dict[str, Any]] = []
        self._events: dict[str, Any] = {}
        self._torch_prof = None
        self._torch_ctx = nullcontext()
        self._activities: list[Any] = []
        self._torch_enabled = False
        if not self.enabled:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_dir = Path(str(getattr(config, "profile_dir", "profiles"))).expanduser()
        self.output_dir = base_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "profile_summary.json"
        self.trace_path = self.output_dir / "torch_trace.json"

        self._activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            self._activities.append(torch.profiler.ProfilerActivity.CUDA)

        if self._activities:
            self._torch_prof = torch.profiler.profile(
                activities=self._activities,
                record_shapes=bool(getattr(config, "profile_record_shapes", True)),
                profile_memory=bool(getattr(config, "profile_memory", True)),
                with_stack=bool(getattr(config, "profile_with_stack", False)),
                acc_events=True,
            )
            self._torch_ctx = self._torch_prof
            self._torch_enabled = True

    def __enter__(self) -> "TrainingRunProfiler":
        if self.enabled:
            self._torch_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.enabled:
            self._torch_ctx.__exit__(exc_type, exc, tb)
            if self._torch_prof is not None and self.trace_path is not None:
                self._torch_prof.export_chrome_trace(str(self.trace_path))
            self._write_summary()
        return False

    def trace(self, name: str) -> "_Trace":
        return _Trace(self, name)

    def record_timing(self, name: str, duration_ms: float) -> None:
        if self.enabled:
            self._timings_ms[str(name)].append(float(duration_ms))

    def record_step(self, step: int, loss: float, grad_norm: float) -> None:
        if not self.enabled:
            return
        row: Dict[str, Any] = {
            "step": int(step),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
        }
        if self.device.type == "cuda":
            row["cuda_memory_allocated_mb"] = round(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024),
                3,
            )
            row["cuda_memory_reserved_mb"] = round(
                torch.cuda.memory_reserved(self.device) / (1024 * 1024),
                3,
            )
        self._step_rows.append(row)

    def step(self) -> None:
        if self._torch_prof is not None:
            self._torch_prof.step()

    def event(self, name: str, value: Any) -> None:
        if self.enabled:
            self._events[str(name)] = value

    def artifacts(self) -> Optional[TrainingProfileArtifacts]:
        if not self.enabled or self.output_dir is None or self.summary_path is None:
            return None
        return TrainingProfileArtifacts(
            output_dir=str(self.output_dir),
            summary_json=str(self.summary_path),
            trace_json=str(self.trace_path) if self.trace_path is not None else None,
        )

    def _timing_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for name, values in self._timings_ms.items():
            if not values:
                continue
            total = sum(values)
            summary[name] = {
                "count": float(len(values)),
                "total_ms": round(total, 4),
                "avg_ms": round(total / len(values), 4),
                "max_ms": round(max(values), 4),
            }
        return summary

    def _torch_summary(self) -> Optional[Dict[str, Any]]:
        if self._torch_prof is None:
            return None
        rows = []
        total_cpu_ms = 0.0
        total_cuda_ms = 0.0
        for evt in self._torch_prof.key_averages():
            cpu_ms = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0) / 1000.0
            cuda_us = float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0)
            if cuda_us <= 0.0:
                cuda_us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
            cuda_ms = cuda_us / 1000.0
            if cpu_ms <= 0.0 and cuda_ms <= 0.0:
                continue
            total_cpu_ms += cpu_ms
            total_cuda_ms += cuda_ms
            rows.append(
                {
                    "op": str(getattr(evt, "key", "unknown")),
                    "cpu_ms": round(cpu_ms, 4),
                    "cuda_ms": round(cuda_ms, 4),
                    "calls": int(getattr(evt, "count", 0) or 0),
                }
            )
        rows.sort(key=lambda row: max(row["cpu_ms"], row["cuda_ms"]), reverse=True)
        return {
            "total_self_cpu_ms": round(total_cpu_ms, 4),
            "total_self_cuda_ms": round(total_cuda_ms, 4),
            "top_ops": rows[:20],
        }

    def _write_summary(self) -> None:
        if self.summary_path is None:
            return
        payload: Dict[str, Any] = {
            "timings_ms": self._timing_summary(),
            "steps": self._step_rows,
            "events": self._events,
            "environment": {
                "python": os.sys.version.split()[0],
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
            },
        }
        if self.device.type == "cuda":
            payload["environment"]["device_name"] = torch.cuda.get_device_name(
                self.device
            )
            payload["environment"]["max_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024),
                3,
            )
            payload["environment"]["max_memory_reserved_mb"] = round(
                torch.cuda.max_memory_reserved(self.device) / (1024 * 1024),
                3,
            )
        torch_summary = self._torch_summary()
        if torch_summary is not None:
            payload["torch_profiler"] = torch_summary
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
