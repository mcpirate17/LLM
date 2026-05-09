"""Shared coordination primitives for GPU + DB writers.

Auxiliary tools (backfill, smoke tests, batch jobs) share two resources
with the long-running dashboard process and with each other: the SQLite
notebook and the single CUDA device. Without coordination, parallel
writers corrupted the b-tree on 2026-04-25 and parallel GPU workloads
hit OOM races / context-switch thrash.

This module provides four primitives, all designed to fail loudly and
early rather than degrade silently:

* :func:`acquire_writer_lock` — fcntl ``LOCK_EX`` on
  the active runs DB writer-lock path. The aria-db Rust path holds this for
  the dashboard's lifetime; aux tools that write must serialize through
  it. Default behavior exits with code 2 when held; pass
  ``blocking=True`` to wait.

* :func:`acquire_gpu_lock` — fcntl ``LOCK_EX`` on
  ``research/.gpu-lock``. Serializes *our own* heavy GPU tools so
  ``smoke_gemini_metrics`` and ``backfill_spec_norm`` don't run at the
  same time. Independent of the dashboard.

* :func:`assert_gpu_quiet` — reads ``nvidia-smi`` and refuses to start
  if any process other than the current one is holding more than
  ``max_other_used_mib`` of VRAM. This is the dashboard / other-agent
  detector: the dashboard does not acquire the GPU flock (it'd hold
  forever) so the only way for aux tools to politely wait is to inspect
  actual usage.

* :func:`cap_gpu_memory` — ``set_per_process_memory_fraction`` cap so
  even if we miscount, our process can't OOM the dashboard.

Usage from a tool's entry point::

    from research.tools._concurrency import (
        acquire_gpu_lock,
        acquire_writer_lock,
        assert_gpu_quiet,
        cap_gpu_memory,
    )

    def main() -> None:
        assert_gpu_quiet(max_other_used_mib=4096, tool_name="backfill_spec_norm")
        cap_gpu_memory(fraction=0.5)
        with acquire_gpu_lock(tool_name="backfill_spec_norm"), \\
             acquire_writer_lock(tool_name="backfill_spec_norm"):
            do_work()
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import torch

from research.defaults import RUNS_DB

ROOT = Path(__file__).resolve().parents[2]
DB_WRITER_LOCK_PATH = ROOT / f"{RUNS_DB}.writer-lock"
GPU_LOCK_PATH = ROOT / "research" / ".gpu-lock"

_LOCK_HELD_EXIT_CODE = 2


def _flock_acquire(
    path: Path,
    *,
    blocking: bool,
    poll_interval_s: float = 1.0,
) -> "Optional[int]":
    """Open ``path`` and acquire LOCK_EX. Returns fd or raises SystemExit.

    ``blocking=False``: tries once, raises ``SystemExit(2)`` on contention.
    ``blocking=True``: polls every ``poll_interval_s`` until acquired.
    """
    path.touch(exist_ok=True)
    fd = os.open(str(path), os.O_RDWR)
    if blocking:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                time.sleep(poll_interval_s)
    else:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
            raise


def _flock_release(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def acquire_writer_lock(
    *,
    lock_path: Path = DB_WRITER_LOCK_PATH,
    blocking: bool = False,
    tool_name: str = "",
) -> Iterator[None]:
    """Acquire the SQLite notebook writer flock.

    Concurrent writers without this lock corrupted the b-tree on
    2026-04-25; the lock is the project's mutual-exclusion contract for
    write access. The dashboard's aria-db Rust path holds this for the
    process lifetime, so aux tools running while the dashboard is up
    will hit the contention path.
    """
    label = tool_name or "writer"
    try:
        fd = _flock_acquire(lock_path, blocking=blocking)
    except BlockingIOError:
        print(
            f"writer lock held — another writer (likely the dashboard) is running. "
            f"Stop it before running {label}, or pass blocking=True to wait.",
            file=sys.stderr,
        )
        raise SystemExit(_LOCK_HELD_EXIT_CODE)
    try:
        yield
    finally:
        _flock_release(fd)


@contextmanager
def acquire_gpu_lock(
    *,
    lock_path: Path = GPU_LOCK_PATH,
    blocking: bool = True,
    tool_name: str = "",
    poll_interval_s: float = 5.0,
) -> Iterator[None]:
    """Acquire the cooperative GPU flock used by aux tools.

    Serializes our own heavy GPU tools (smoke runner, backfill,
    bulk-eval scripts) against each other. The dashboard does *not*
    acquire this lock — see :func:`assert_gpu_quiet` for dashboard
    detection.

    Default ``blocking=True`` because a typical aux tool wants to
    wait for the previous one to finish, not exit immediately. Set
    ``blocking=False`` for one-shot scripts that should refuse rather
    than queue.
    """
    label = tool_name or "gpu_consumer"
    try:
        fd = _flock_acquire(
            lock_path,
            blocking=blocking,
            poll_interval_s=poll_interval_s,
        )
    except BlockingIOError:
        print(
            f"gpu lock held by another aux tool — refusing to start {label}",
            file=sys.stderr,
        )
        raise SystemExit(_LOCK_HELD_EXIT_CODE)
    if blocking:
        # Quick info line so a long-blocking tool prints something visible.
        print(f"[concurrency] gpu lock acquired by {label} (pid={os.getpid()})")
    try:
        yield
    finally:
        _flock_release(fd)


def _query_nvidia_smi_apps() -> list[tuple[int, int]]:
    """Return ``[(pid, used_memory_mib), ...]`` for current GPU compute apps.

    Empty list on failure — caller decides whether to treat that as
    quiet or noisy. We log to stderr but never raise from this helper.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"[concurrency] nvidia-smi unavailable: {exc}", file=sys.stderr)
        return []

    apps: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            apps.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return apps


def assert_gpu_quiet(
    *,
    max_other_used_mib: int = 4096,
    tool_name: str = "",
    sleep_until_quiet: bool = False,
    poll_interval_s: float = 30.0,
    max_wait_s: float = 3600.0,
) -> None:
    """Refuse to start if other GPU consumers are using significant memory.

    ``max_other_used_mib`` defaults to 4 GB — a typical idle dashboard
    holds well over 5 GB just from cached allocator state, so anything
    above this threshold suggests an active heavy workload (training,
    long-running probe, another agent's smoke test).

    When ``sleep_until_quiet=True``, the function polls until the heavy
    consumers go away (up to ``max_wait_s``) instead of exiting.
    """
    label = tool_name or "tool"
    deadline = time.time() + max_wait_s
    me = os.getpid()

    while True:
        apps = _query_nvidia_smi_apps()
        others = [(pid, mib) for pid, mib in apps if pid != me]
        max_other = max((mib for _, mib in others), default=0)
        if max_other <= max_other_used_mib:
            return

        offenders = ", ".join(
            f"pid={pid} ({mib} MiB)" for pid, mib in others if mib > max_other_used_mib
        )
        if not sleep_until_quiet:
            print(
                f"[concurrency] gpu busy: {offenders} — refusing to start "
                f"{label} (max_other_used_mib={max_other_used_mib})",
                file=sys.stderr,
            )
            raise SystemExit(_LOCK_HELD_EXIT_CODE)

        if time.time() > deadline:
            print(
                f"[concurrency] gpu still busy after {max_wait_s:.0f}s wait — "
                f"giving up on {label}: {offenders}",
                file=sys.stderr,
            )
            raise SystemExit(_LOCK_HELD_EXIT_CODE)

        print(
            f"[concurrency] gpu busy: {offenders} — {label} waiting "
            f"{poll_interval_s:.0f}s",
            file=sys.stderr,
        )
        time.sleep(poll_interval_s)


def cap_gpu_memory(*, fraction: float = 0.5, device: int = 0) -> None:
    """Cap this process's CUDA memory at ``fraction`` of total VRAM.

    Belt-and-suspenders backstop: even if :func:`assert_gpu_quiet`
    misses a concurrent consumer, the allocator can't grow past the
    cap and we OOM ourselves before stealing from someone else.
    """
    if not torch.cuda.is_available():
        return
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
