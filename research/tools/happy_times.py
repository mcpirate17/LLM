#!/usr/bin/env python3
"""happy_times.py — Safe shutdown for Aria and the host system.

Usage:
    python happy_times.py              # Shut down Aria, then the computer
    python happy_times.py --aria-only  # Just shut down Aria (dashboard + runner)
    python happy_times.py --dry-run    # Show what would happen without doing it

From Python:
    from happy_times import shutdown
    shutdown()                         # Aria + computer
    shutdown(aria_only=True)           # Just Aria
"""

from __future__ import annotations

import os
import signal
import subprocess
import time


def _find_aria_processes() -> list[dict]:
    """Find all running Aria processes (dashboard, continuous runner, etc.)."""
    targets = [
        "python -m research --mode=continuous",
        "python -m research --mode=dashboard",
        "python -m research --mode=evolve",
        "python -m research --mode=synthesize",
    ]
    found = []
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            for target in targets:
                if target in line and "grep" not in line:
                    parts = line.split()
                    pid = int(parts[1])
                    if pid != os.getpid():
                        found.append({"pid": pid, "cmd": target, "line": line.strip()})
    except Exception:
        pass

    # Also check PID files
    for pid_file in ["/tmp/aria_pid.txt", "/tmp/aria_dashboard_pid.txt"]:
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            # Check if process is alive
            os.kill(pid, 0)
            if not any(p["pid"] == pid for p in found):
                found.append(
                    {"pid": pid, "cmd": pid_file, "line": f"PID {pid} from {pid_file}"}
                )
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            pass

    return found


def _stop_process(pid: int, name: str, dry_run: bool = False) -> bool:
    """Gracefully stop a process: SIGTERM, wait, then SIGKILL if needed."""
    if dry_run:
        print(f"  [dry-run] Would send SIGTERM to PID {pid} ({name})")
        return True

    print(f"  Sending SIGTERM to PID {pid} ({name})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"  PID {pid} already gone.")
        return True

    # Wait up to 10 seconds for graceful shutdown
    for _ in range(20):
        try:
            os.kill(pid, 0)  # Check if alive
            time.sleep(0.5)
        except ProcessLookupError:
            print(f"  PID {pid} stopped gracefully.")
            return True

    # Force kill
    print(f"  PID {pid} didn't stop, sending SIGKILL...")
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        print(f"  PID {pid} killed.")
    except ProcessLookupError:
        print(f"  PID {pid} stopped.")
    return True


def shutdown(aria_only: bool = False, dry_run: bool = False) -> None:
    """Safely shut down Aria processes and optionally the computer.

    1. Finds all running Aria processes (dashboard, runner, etc.)
    2. Sends SIGTERM for graceful shutdown (flushes DB writes)
    3. Waits up to 10s per process, then SIGKILL if needed
    4. Cleans up PID files
    5. If not aria_only: shuts down the computer via systemctl

    Args:
        aria_only: If True, only stop Aria processes (don't shut down computer).
        dry_run: If True, just print what would happen.
    """
    print("=" * 50)
    print("  happy_times.py — Safe Shutdown")
    print("=" * 50)
    print()

    # Step 1: Find Aria processes
    procs = _find_aria_processes()
    if procs:
        print(f"Found {len(procs)} Aria process(es):")
        for p in procs:
            print(f"  PID {p['pid']:>7d}  {p['cmd']}")
        print()

        # Step 2: Stop them gracefully
        print("Stopping Aria processes...")
        for p in procs:
            _stop_process(p["pid"], p["cmd"], dry_run=dry_run)
        print()
    else:
        print("No running Aria processes found.")
        print()

    # Step 3: Clean up PID files
    for pid_file in ["/tmp/aria_pid.txt", "/tmp/aria_dashboard_pid.txt"]:
        if os.path.exists(pid_file):
            if dry_run:
                print(f"[dry-run] Would remove {pid_file}")
            else:
                os.remove(pid_file)

    if aria_only:
        print("Aria shut down. Computer stays on.")
        return

    # Step 4: Shut down the computer
    print("Shutting down the computer in 5 seconds...")
    print("(Ctrl+C to cancel)")
    if dry_run:
        print("[dry-run] Would run: systemctl poweroff")
        return

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutdown cancelled.")
        return

    print("Goodnight!")
    subprocess.run(["systemctl", "poweroff"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Safe shutdown for Aria + computer")
    parser.add_argument(
        "--aria-only",
        action="store_true",
        help="Only shut down Aria (don't power off computer)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without doing it"
    )
    args = parser.parse_args()
    shutdown(aria_only=args.aria_only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
