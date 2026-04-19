"""
Performance Tracking and Profiling Utilities

Tools for high-precision timing, GPU utilization tracking,
and bottleneck detection in the AI Scientist pipeline.
"""

from __future__ import annotations

import logging
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PerfTrace:
    """A single performance trace for an operation."""

    name: str
    cpu_start: float
    cpu_end: Optional[float] = None
    gpu_start: Optional[torch.cuda.Event] = None
    gpu_end: Optional[torch.cuda.Event] = None

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if self.cpu_end is None:
            return 0.0

        # If GPU events are available and on CUDA, use them for more precision
        if self.gpu_start and self.gpu_end and torch.cuda.is_available():
            try:
                # Need to synchronize to ensure timing is accurate
                torch.cuda.synchronize()
                return self.gpu_start.elapsed_time(self.gpu_end)
            except RuntimeError:
                pass

        return (self.cpu_end - self.cpu_start) * 1000


class PerfTracer:
    """Context manager and aggregator for performance traces."""

    def __init__(self):
        self.traces: List[PerfTrace] = []
        self._active: Dict[str, PerfTrace] = {}

    def start(self, name: str, use_gpu: bool = True):
        """Start a named trace."""
        gpu_start = None
        if use_gpu and torch.cuda.is_available():
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_start.record()

        trace = PerfTrace(name=name, cpu_start=time.perf_counter(), gpu_start=gpu_start)
        self._active[name] = trace
        return trace

    def stop(self, name: str):
        """Stop a named trace and record it."""
        if name not in self._active:
            return

        trace = self._active.pop(name)
        trace.cpu_end = time.perf_counter()

        if trace.gpu_start and torch.cuda.is_available():
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_end.record()
            trace.gpu_end = gpu_end

        self.traces.append(trace)
        return trace

    def get_summary(self) -> Dict[str, float]:
        """Return a summary of all traces."""
        summary = {}
        for trace in self.traces:
            summary[trace.name] = summary.get(trace.name, 0.0) + trace.duration_ms
        return summary

    def get_report(self) -> Dict[str, Any]:
        """Return a structured report with per-trace timings."""
        return {
            "summary_ms": self.get_summary(),
            "traces": [
                {
                    "name": trace.name,
                    "duration_ms": trace.duration_ms,
                }
                for trace in self.traces
            ],
        }

    def trace(self, name: str, use_gpu: bool = True):
        """Context manager wrapper for start/stop."""
        tracer = self

        class _TraceContext:
            def __enter__(self_inner):
                tracer.start(name, use_gpu=use_gpu)
                return self_inner

            def __exit__(self_inner, _exc_type, exc, tb):
                tracer.stop(name)
                return False

        return _TraceContext()

    def clear(self):
        """Clear all recorded traces."""
        self.traces = []
        self._active = {}


class OpKernelProfiler:
    """Thin wrapper around torch.profiler for per-op kernel timing samples."""

    def __init__(self, enabled: bool = True, top_k: int = 20):
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k))

    def _activities(self) -> List[Any]:
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            return activities
        except Exception as exc:
            logger.debug("Returning default due to error: %s", exc)
            return []

    def profile_callable(self, fn) -> Optional[Dict[str, Any]]:
        """Profile a callable once and return summarized op timings."""
        if not self.enabled:
            return None
        activities = self._activities()
        if not activities:
            return None

        try:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                fn()
            return self.summarize(prof, top_k=self.top_k)
        except Exception as exc:
            logger.debug("Returning default due to error: %s", exc)
            return None

    @staticmethod
    def summarize(prof: Any, top_k: int = 20) -> Dict[str, Any]:
        """Convert profiler events into a compact, JSON-safe summary."""
        rows: List[Dict[str, Any]] = []
        total_cpu_ms = 0.0
        total_cuda_ms = 0.0

        for evt in prof.key_averages():
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

        rows.sort(key=lambda r: max(r["cpu_ms"], r["cuda_ms"]), reverse=True)
        rows = rows[: max(1, int(top_k))]

        return {
            "top_ops": rows,
            "n_profiled_ops": len(rows),
            "total_self_cpu_ms": round(total_cpu_ms, 4),
            "total_self_cuda_ms": round(total_cuda_ms, 4),
        }


class GPUStarvationDetector:
    """Detects if the GPU is starving due to CPU or data-loader bottlenecks."""

    def __init__(self, threshold_ms: float = 5.0):
        self.threshold_ms = threshold_ms
        self.last_wait_start: Optional[float] = None
        self.starvation_events = []

    def start_wait(self):
        """Mark the start of a potential stall (e.g. before data loading)."""
        self.last_wait_start = time.perf_counter()

    def end_wait(self) -> Optional[float]:
        """Mark the end of a potential stall and return stall duration in ms."""
        if self.last_wait_start is None:
            return None

        duration = (time.perf_counter() - self.last_wait_start) * 1000.0
        if duration > self.threshold_ms:
            self.starvation_events.append(duration)

        self.last_wait_start = None
        return duration

    def get_summary(self) -> Dict[str, Any]:
        if not self.starvation_events:
            return {"starvation_detected": False, "count": 0}

        return {
            "starvation_detected": True,
            "count": len(self.starvation_events),
            "max_stall_ms": max(self.starvation_events),
            "total_stall_ms": sum(self.starvation_events),
            "avg_stall_ms": sum(self.starvation_events) / len(self.starvation_events),
        }


class KernelTimer:
    """Hooks into PyTorch models to time individual kernel/op executions."""

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.timings: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._hooks = []
        self._active_events: Dict[int, torch.cuda.Event] = {}

        if enabled:
            self._attach_hooks()

    def _attach_hooks(self):
        """Attach forward hooks to all sub-modules."""
        for name, module in self.model.named_modules():
            # Skip the root model itself to avoid double counting
            if name == "":
                continue

            # Use closure to capture name
            def _get_hooks(mod_name):
                def pre_hook(mod, input):
                    if torch.cuda.is_available() and self.enabled:
                        start_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        # Store start event using id of the current thread/call context if possible
                        # For simplicity in single-threaded training, we use a stack-like approach per module
                        self._active_events.setdefault(id(mod), []).append(start_event)

                def post_hook(mod, input, output):
                    if torch.cuda.is_available() and self.enabled:
                        end_event = torch.cuda.Event(enable_timing=True)
                        end_event.record()

                        starts = self._active_events.get(id(mod), [])
                        if starts:
                            start_event = starts.pop()
                            self.timings.setdefault(mod_name, []).append(
                                (start_event, end_event)
                            )

                return pre_hook, post_hook

            pre, post = _get_hooks(name)
            self._hooks.append(module.register_forward_pre_hook(pre))
            self._hooks.append(module.register_forward_hook(post))

    def synchronize_and_get_timings(self) -> Dict[str, float]:
        """Synchronize GPU and calculate average durations in ms."""
        if not torch.cuda.is_available() or not self.enabled:
            return {}

        torch.cuda.synchronize()
        report = {}
        for name, events in self.timings.items():
            durations = []
            for start, end in events:
                try:
                    durations.append(start.elapsed_time(end))
                except Exception as exc:
                    logger.debug("Skipping due to error: %s", exc)
                    continue
            if durations:
                report[name] = sum(durations) / len(durations)
        return report

    def remove_hooks(self):
        """Remove all attached hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class QueueTelemetry:
    """Tracks latency and throughput for task queues (batching, scheduling)."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}

    def record_wait(self, queue_name: str, wait_ms: float):
        """Record how long a task spent in the queue."""
        q_stats = self.stats.setdefault(
            queue_name,
            {"wait_times_ms": [], "task_count": 0, "start_time": time.time()},
        )
        q_stats["wait_times_ms"].append(wait_ms)
        q_stats["task_count"] += 1

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for name, q_stats in self.stats.items():
            waits = q_stats["wait_times_ms"]
            if not waits:
                continue

            elapsed = time.time() - q_stats["start_time"]
            summary[name] = {
                "avg_wait_ms": sum(waits) / len(waits),
                "max_wait_ms": max(waits),
                "tasks_per_sec": q_stats["task_count"] / max(elapsed, 1.0),
                "total_tasks": q_stats["task_count"],
            }
        return summary
