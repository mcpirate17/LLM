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
    model: Optional[torch.nn.Module] = None  # Z6: Pre-compiled model
    payload: Dict[str, Any] = field(default_factory=dict)
    enqueue_time: float = field(default_factory=time.perf_counter)
    prep_enqueue_time: float = field(default_factory=time.perf_counter)
    worker_enqueue_time: float = 0.0
    preprocessing_start_time: float = 0.0
    preprocessing_end_time: float = 0.0


@dataclass
class JobResult:
    """Result of an evaluation job."""

    index: int
    s1_result: Dict[str, Any]
    payload: Dict[str, Any]
    telemetry: Dict[str, Any] = field(default_factory=dict)


class WorkerPoolOrchestrator:
    """Orchestrates concurrent Stage 1 micro-training across workers and GPUs."""

    def __init__(
        self,
        train_fn: Callable,
        num_workers: int = 1,
        max_queue_size: int = 10,
        devices: List[str] = None,
        remote_workers: List[str] = None,
    ):
        self.train_fn = train_fn
        self.num_workers = num_workers
        self.devices = devices or ["cuda:0" if torch.cuda.is_available() else "cpu"]
        self.remote_workers = remote_workers or []

        # Z6: Preprocessing queue for double buffering
        self.prep_queue = queue.Queue(maxsize=max_queue_size)
        self.job_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.telemetry = QueueTelemetry()
        self._telemetry_lock = threading.Lock()
        self._submit_wait_times_ms: List[float] = []
        self._execution_times_ms: List[float] = []
        self._preprocessing_times_ms: List[float] = []
        self._remote_execution_times_ms: List[float] = []
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
        self.remote_threads = []
        self.preprocessors = []
        self.stop_event = threading.Event()

        self._start_preprocessors()
        self._start_workers()
        self._start_remote_workers()

    def _start_remote_workers(self):
        """Start threads for dispatching jobs to remote nodes (Z12)."""
        for i, url in enumerate(self.remote_workers):
            t = threading.Thread(
                target=self._remote_worker_loop, args=(i, url), daemon=True
            )
            t.start()
            self.remote_threads.append(t)

    def _remote_worker_loop(self, worker_id: int, url: str):
        """Dispatches jobs to remote REST API worker nodes."""
        import requests
        from ..synthesis.graph import graph_to_json

        logger.info("Remote worker %d starting for endpoint %s", worker_id, url)
        while not self.stop_event.is_set():
            try:
                # Remote workers pull from the prep_queue because they handle their own compilation
                job = self.prep_queue.get(timeout=0.5)
                if job is None:
                    break

                start_remote = time.perf_counter()
                try:
                    # Prepare payload (Z12 protocol)
                    payload = {
                        "index": job.index,
                        "graph_json": graph_to_json(job.graph),
                        "config": job.config.to_dict()
                        if hasattr(job.config, "to_dict")
                        else job.config,
                        "seed": job.seed,
                        "payload": job.payload,
                    }

                    # POST to remote worker
                    resp = requests.post(
                        f"{url.rstrip('/')}/api/worker/evaluate",
                        json=payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    res_data = resp.json()

                    remote_ms = (time.perf_counter() - start_remote) * 1000.0
                    with self._telemetry_lock:
                        self._remote_execution_times_ms.append(remote_ms)
                        self._completed_jobs += 1
                        if job.payload.get("queue_kind") == "training_program":
                            self._training_program_queue["completed"] += 1

                    # Wrap result
                    res_telemetry = {
                        "job_execution_ms": remote_ms,
                        "worker_id": f"remote_{worker_id}",
                        "device": res_data.get("device", "remote"),
                        "remote_url": url,
                    }
                    self.result_queue.put(
                        JobResult(
                            job.index, res_data["result"], job.payload, res_telemetry
                        )
                    )

                except Exception as e:
                    logger.error(
                        "Remote worker %d failed job %d on %s: %s",
                        worker_id,
                        job.index,
                        url,
                        e,
                    )
                    # Re-queue for local worker or another remote?
                    # For now, just mark as failed to avoid infinite loops
                    self.result_queue.put(
                        JobResult(
                            job.index,
                            {"error": f"Remote failure: {e}", "passed": False},
                            job.payload,
                        )
                    )
                finally:
                    self.prep_queue.task_done()
            except queue.Empty:
                continue

    def _start_preprocessors(self):
        # 2 preprocessor threads per GPU worker generally enough
        num_preps = max(2, self.num_workers)
        for i in range(num_preps):
            t = threading.Thread(target=self._preprocessor_loop, args=(i,), daemon=True)
            t.start()
            self.preprocessors.append(t)

    def _start_workers(self):
        for i in range(self.num_workers):
            device = self.devices[i % len(self.devices)]
            t = threading.Thread(
                target=self._worker_loop, args=(i, device), daemon=True
            )
            t.start()
            self.workers.append(t)

    def _preprocessor_loop(self, prep_id: int):
        """Background thread for graph-to-model compilation (CPU-heavy)."""
        from ..synthesis.compiler import compile_model

        while not self.stop_event.is_set():
            try:
                job = self.prep_queue.get(timeout=0.5)
                if job is None:
                    break

                prep_wait_ms = (time.perf_counter() - job.prep_enqueue_time) * 1000.0
                self.telemetry.record_wait("prep_queue", prep_wait_ms)
                job.preprocessing_start_time = time.perf_counter()
                try:
                    # CPU-intensive compilation
                    layer_graphs = [job.graph] * job.config.n_layers
                    job.model = compile_model(
                        layer_graphs,
                        vocab_size=job.config.vocab_size,
                        max_seq_len=job.config.max_seq_len,
                    )
                    job.preprocessing_end_time = time.perf_counter()
                    prep_ms = (
                        job.preprocessing_end_time - job.preprocessing_start_time
                    ) * 1000.0
                    with self._telemetry_lock:
                        self._preprocessing_times_ms.append(prep_ms)

                    # Submit to GPU worker queue
                    job.worker_enqueue_time = time.perf_counter()
                    self.job_queue.put(job)
                except Exception as e:
                    logger.error("Preprocessor %d failed compilation: %s", prep_id, e)
                    self.result_queue.put(
                        JobResult(
                            job.index,
                            {"error": f"Compilation failed: {e}", "passed": False},
                            job.payload,
                        )
                    )
                finally:
                    self.prep_queue.task_done()
            except queue.Empty:
                continue

    def _worker_loop(self, worker_id: int, device: str):
        logger.debug("Worker %d started on device %s", worker_id, device)
        dev = torch.device(device)
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
                if job is None:
                    break

                # Record queue wait time
                worker_enqueue_time = (
                    job.worker_enqueue_time
                    if job.worker_enqueue_time > 0.0
                    else job.enqueue_time
                )
                wait_ms = (time.perf_counter() - worker_enqueue_time) * 1000.0
                self.telemetry.record_wait("job_queue", wait_ms)
                start_exec = time.perf_counter()

                # Execute micro-train on assigned device
                try:
                    # Z6: train_fn now expects pre-compiled model
                    # If train_fn still takes graph, we might need a wrapper or refactor
                    # but ExperimentRunner._micro_train takes a model.
                    res = self.train_fn(job.model, job.config, job.seed, dev)
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
                        "job_preprocessing_ms": (
                            job.preprocessing_end_time - job.preprocessing_start_time
                        )
                        * 1000.0,
                        "worker_id": worker_id,
                        "device": device,
                    }
                    self.result_queue.put(
                        JobResult(job.index, res, job.payload, res_telemetry)
                    )
                except Exception as e:
                    logger.error(
                        "Worker %d failed job %d on %s: %s",
                        worker_id,
                        job.index,
                        device,
                        e,
                    )
                    exec_ms = (time.perf_counter() - start_exec) * 1000.0
                    with self._telemetry_lock:
                        self._execution_times_ms.append(exec_ms)
                        self._completed_jobs += 1
                        self._failed_jobs += 1
                        if job.payload.get("queue_kind") == "training_program":
                            self._training_program_queue["completed"] += 1
                            self._training_program_queue["failed"] += 1
                    self.result_queue.put(
                        JobResult(
                            job.index, {"error": str(e), "passed": False}, job.payload
                        )
                    )
                finally:
                    # Clean up model after GPU work to avoid HBM leak
                    if job.model:
                        del job.model
                    self.job_queue.task_done()

            except queue.Empty:
                continue

    def submit(
        self,
        index: int,
        graph: Any,
        config: Any,
        seed: int,
        payload: Dict[str, Any] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        """Submit a job to the queue. Blocks if queue is full (backpressure)."""
        payload_data = payload or {}
        t0 = time.perf_counter()
        job = Job(
            index,
            graph,
            config,
            seed,
            model=model,
            payload=payload_data,
            enqueue_time=time.perf_counter(),
            prep_enqueue_time=time.perf_counter(),
        )
        if model is not None:
            # Skip preprocessing if model is already compiled
            job.preprocessing_start_time = t0
            job.preprocessing_end_time = t0
            job.worker_enqueue_time = time.perf_counter()
            self.job_queue.put(job)
        else:
            # Z6: Submit to preprocessor
            self.prep_queue.put(job)
        submit_wait_ms = (time.perf_counter() - t0) * 1000.0
        with self._telemetry_lock:
            self._submit_wait_times_ms.append(submit_wait_ms)
            self._submitted_jobs += 1
            self._queue_depth_peak = max(
                self._queue_depth_peak, self.prep_queue.qsize() + self.job_queue.qsize()
            )
            batch_id = payload_data.get("batch_id")
            if batch_id is not None:
                key = str(batch_id)
                self._candidate_batches[key] = (
                    int(self._candidate_batches.get(key, 0)) + 1
                )
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
        for _ in self.preprocessors:
            self.prep_queue.put(None)
        for _ in self.workers:
            self.job_queue.put(None)
        for t in self.preprocessors:
            t.join(timeout=2.0)
        for t in self.workers:
            t.join(timeout=2.0)

    def get_telemetry(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            submit_waits = list(self._submit_wait_times_ms)
            exec_times = list(self._execution_times_ms)
            prep_times = list(self._preprocessing_times_ms)
            submitted_jobs = int(self._submitted_jobs)
            completed_jobs = int(self._completed_jobs)
            failed_jobs = int(self._failed_jobs)
            queue_depth_peak = int(self._queue_depth_peak)
            candidate_batches = dict(self._candidate_batches)
            training_program_queue = dict(self._training_program_queue)

        queue_summary = self.telemetry.get_summary()
        prep_queue_summary = queue_summary.get("prep_queue", {})
        job_queue_summary = queue_summary.get("job_queue", {})
        return {
            "submitted_jobs": submitted_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "queue_depth_peak": queue_depth_peak,
            "queue_depth_current": self.prep_queue.qsize() + self.job_queue.qsize(),
            "result_queue_depth_current": self.result_queue.qsize(),
            "submit_wait_avg_ms": (sum(submit_waits) / len(submit_waits))
            if submit_waits
            else 0.0,
            "submit_wait_max_ms": max(submit_waits) if submit_waits else 0.0,
            "job_execution_avg_ms": (sum(exec_times) / len(exec_times))
            if exec_times
            else 0.0,
            "job_execution_max_ms": max(exec_times) if exec_times else 0.0,
            "preprocessing_avg_ms": (sum(prep_times) / len(prep_times))
            if prep_times
            else 0.0,
            "prep_queue_wait_avg_ms": float(
                prep_queue_summary.get("avg_wait_ms", 0.0) or 0.0
            ),
            "prep_queue_wait_max_ms": float(
                prep_queue_summary.get("max_wait_ms", 0.0) or 0.0
            ),
            "scheduling_wait_avg_ms": float(
                job_queue_summary.get("avg_wait_ms", 0.0) or 0.0
            ),
            "scheduling_wait_max_ms": float(
                job_queue_summary.get("max_wait_ms", 0.0) or 0.0
            ),
            "candidate_batches": candidate_batches,
            "training_program_queue": training_program_queue,
        }
