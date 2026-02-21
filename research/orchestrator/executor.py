"""
Asynchronous Program Orchestrator

Manages a worker pool for concurrent evaluation of computation graphs.
Decouples graph generation/validation (CPU) from micro-training (GPU).
"""

from __future__ import annotations

import queue
import threading
import logging
import time
import torch
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from ..scientist.perf import QueueTelemetry

logger = logging.getLogger(__name__)

@dataclass
class Job:
    """A single evaluation job."""
    index: int
    graph: Any
    config: Any
    seed: int
    payload: Dict[str, Any] = field(default_factory=dict)
    enqueue_time: float = field(default_factory=time.time)
    
@dataclass
class JobResult:
    """Result of an evaluation job."""
    index: int
    s1_result: Dict[str, Any]
    payload: Dict[str, Any]
    telemetry: Dict[str, Any] = field(default_factory=dict)

class WorkerPoolOrchestrator:
    """Orchestrates concurrent Stage 1 micro-training across workers and GPUs."""
    
    def __init__(self, 
                 train_fn: Callable,
                 num_workers: int = 1,
                 max_queue_size: int = 10,
                 devices: List[str] = None):
        self.train_fn = train_fn
        self.num_workers = num_workers
        self.devices = devices or ["cuda:0" if torch.cuda.is_available() else "cpu"]
        
        # If we have more workers than devices, we round-robin
        self.job_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.telemetry = QueueTelemetry()
        self._telemetry_lock = threading.Lock()
        self._submit_wait_times_ms: List[float] = []
        self._execution_times_ms: List[float] = []
        self._failed_jobs = 0
        self._submitted_jobs = 0
        self._completed_jobs = 0
        self._queue_depth_peak = 0
        self._candidate_batches: Dict[str, int] = {}
        self._training_program_queue: Dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
        }
        
        self.workers = []
        self.stop_event = threading.Event()
        
        self._start_workers()

    def _start_workers(self):
        for i in range(self.num_workers):
            device = self.devices[i % len(self.devices)]
            t = threading.Thread(target=self._worker_loop, args=(i, device), daemon=True)
            t.start()
            self.workers.append(t)
            
    def _worker_loop(self, worker_id: int, device: str):
        logger.debug("Worker %d started on device %s", worker_id, device)
        dev = torch.device(device)
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
                if job is None:
                    break
                
                # Record queue wait time
                wait_ms = (time.time() - job.enqueue_time) * 1000
                self.telemetry.record_wait("job_queue", wait_ms)
                start_exec = time.perf_counter()
                
                # Execute micro-train on assigned device
                try:
                    res = self.train_fn(job.graph, job.config, job.seed, dev)
                    exec_ms = (time.perf_counter() - start_exec) * 1000.0
                    with self._telemetry_lock:
                        self._execution_times_ms.append(exec_ms)
                        self._completed_jobs += 1
                        if job.payload.get("queue_kind") == "training_program":
                            self._training_program_queue["completed"] += 1
                    
                    # Attach queue telemetry to result
                    res_telemetry = {
                        "job_queue_wait_ms": wait_ms,
                        "job_execution_ms": exec_ms,
                        "worker_id": worker_id,
                        "device": device
                    }
                    self.result_queue.put(JobResult(job.index, res, job.payload, res_telemetry))
                except Exception as e:
                    logger.error("Worker %d failed job %d on %s: %s", worker_id, job.index, device, e)
                    exec_ms = (time.perf_counter() - start_exec) * 1000.0
                    with self._telemetry_lock:
                        self._execution_times_ms.append(exec_ms)
                        self._completed_jobs += 1
                        self._failed_jobs += 1
                        if job.payload.get("queue_kind") == "training_program":
                            self._training_program_queue["completed"] += 1
                            self._training_program_queue["failed"] += 1
                    self.result_queue.put(JobResult(job.index, {"error": str(e), "passed": False}, job.payload))
                finally:
                    self.job_queue.task_done()
                    
            except queue.Empty:
                continue
                
    def submit(self, index: int, graph: Any, config: Any, seed: int, payload: Dict[str, Any] = None):
        """Submit a job to the queue. Blocks if queue is full (backpressure)."""
        payload_data = payload or {}
        t0 = time.perf_counter()
        job = Job(index, graph, config, seed, payload_data, enqueue_time=time.time())
        self.job_queue.put(job)
        submit_wait_ms = (time.perf_counter() - t0) * 1000.0
        with self._telemetry_lock:
            self._submit_wait_times_ms.append(submit_wait_ms)
            self._submitted_jobs += 1
            self._queue_depth_peak = max(self._queue_depth_peak, self.job_queue.qsize())
            batch_id = payload_data.get("batch_id")
            if batch_id is not None:
                key = str(batch_id)
                self._candidate_batches[key] = int(self._candidate_batches.get(key, 0)) + 1
            if payload_data.get("queue_kind") == "training_program":
                self._training_program_queue["submitted"] += 1

    def get_results(self) -> List[JobResult]:
        """Collect all currently available results."""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

    def shutdown(self):
        self.stop_event.set()
        for _ in self.workers:
            self.job_queue.put(None)
        for t in self.workers:
            t.join(timeout=2.0)

    def get_telemetry(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            submit_waits = list(self._submit_wait_times_ms)
            exec_times = list(self._execution_times_ms)
            submitted_jobs = int(self._submitted_jobs)
            completed_jobs = int(self._completed_jobs)
            failed_jobs = int(self._failed_jobs)
            queue_depth_peak = int(self._queue_depth_peak)
            candidate_batches = dict(self._candidate_batches)
            training_program_queue = dict(self._training_program_queue)

        queue_summary = self.telemetry.get_summary()
        job_queue_summary = queue_summary.get("job_queue", {})
        return {
            "submitted_jobs": submitted_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "queue_depth_peak": queue_depth_peak,
            "queue_depth_current": self.job_queue.qsize(),
            "result_queue_depth_current": self.result_queue.qsize(),
            "submit_wait_avg_ms": (sum(submit_waits) / len(submit_waits)) if submit_waits else 0.0,
            "submit_wait_max_ms": max(submit_waits) if submit_waits else 0.0,
            "job_execution_avg_ms": (sum(exec_times) / len(exec_times)) if exec_times else 0.0,
            "job_execution_max_ms": max(exec_times) if exec_times else 0.0,
            "scheduling_wait_avg_ms": float(job_queue_summary.get("avg_wait_ms", 0.0) or 0.0),
            "scheduling_wait_max_ms": float(job_queue_summary.get("max_wait_ms", 0.0) or 0.0),
            "candidate_batches": candidate_batches,
            "training_program_queue": training_program_queue,
        }
