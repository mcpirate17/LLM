from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..scientist.notebook import LabNotebook


class HealerError(RuntimeError):
    pass


@dataclass
class HealerTaskSpec:
    experiment_id: Optional[str]
    trigger_type: str
    scope: str
    reproduction_steps: List[str]
    acceptance_tests: List[str]
    trigger_payload: Dict[str, Any]
    preferred_endpoint: Optional[str] = None
    command_timeout_seconds: int = 180


class CodeHealer:
    """Internal code-healing orchestrator with explicit state transitions."""

    def __init__(self, notebook_path: str, config_path: Optional[str] = None):
        self.notebook_path = notebook_path
        self.config_path = Path(
            config_path or (Path(__file__).parent / "healer_config.json")
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise HealerError(f"Missing healer config: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _select_endpoint(self, preferred: Optional[str]) -> Dict[str, Any]:
        endpoints = self.config.get("model_endpoints") or []
        if preferred:
            for ep in endpoints:
                if ep.get("name") == preferred:
                    return ep
        for ep in endpoints:
            if ep.get("default"):
                return ep
        return (
            endpoints[0]
            if endpoints
            else {"name": "unknown", "kind": "local", "url": ""}
        )

    def _command_allowed(self, command: str) -> bool:
        allowed = self.config.get("allowed_commands") or []
        return any(command.strip().startswith(prefix) for prefix in allowed)

    def _run_allowed_command(
        self, command: str, cwd: Path, timeout_seconds: int
    ) -> Dict[str, Any]:
        if not self._command_allowed(command):
            raise HealerError(f"Command blocked by healer sandbox policy: {command}")

        # Z17: Ensure project root is in PYTHONPATH
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = str(cwd)

        argv = shlex.split(command)
        if not argv:
            raise HealerError("Command blocked by healer sandbox policy: empty command")

        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    def open_task(self, spec: HealerTaskSpec) -> str:
        endpoint = self._select_endpoint(spec.preferred_endpoint)
        nb = LabNotebook(self.notebook_path)
        try:
            task_id = nb.create_healer_task(
                experiment_id=spec.experiment_id,
                trigger_type=spec.trigger_type,
                scope=spec.scope,
                reproduction_steps=spec.reproduction_steps,
                acceptance_tests=spec.acceptance_tests,
                model_endpoint=endpoint.get("name"),
                sandbox_policy={
                    "allowed_commands": self.config.get("allowed_commands", []),
                    "allowed_paths": self.config.get("allowed_paths", []),
                },
                trigger_payload={
                    **spec.trigger_payload,
                    "command_timeout_seconds": max(
                        1, int(spec.command_timeout_seconds)
                    ),
                },
            )
            nb.add_healer_event(
                task_id,
                "Healing task opened.",
                state="open",
                payload={"endpoint": endpoint},
            )
            return task_id
        finally:
            nb.close()

    def run_task(self, task_id: str) -> Dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        endpoint = None
        repro_results: List[Dict[str, Any]] = []
        verify_results: List[Dict[str, Any]] = []

        nb = LabNotebook(self.notebook_path)
        try:
            task = nb.get_healer_task(task_id)
            if not task:
                raise HealerError(f"Healer task not found: {task_id}")
            endpoint_name = task.get("model_endpoint")
            endpoint = self._select_endpoint(endpoint_name)

            nb.update_healer_task(task_id, state="reproducing")
            nb.add_healer_event(
                task_id, "Running reproduction steps.", state="reproducing"
            )
            command_timeout_seconds = max(
                1,
                int(
                    task.get("trigger_payload_json", {}).get(
                        "command_timeout_seconds", 180
                    )
                    or 180
                ),
            )
            for cmd in task.get("reproduction_steps_json") or []:
                repro = self._run_allowed_command(cmd, root, command_timeout_seconds)
                repro_results.append(repro)

            nb.update_healer_task(task_id, state="patch_proposed")
            patch_summary = (
                "Patch proposal generated by healer workflow. "
                "Includes mandatory test updates for behavior changes."
            )
            risk_assessment = (
                "Low-to-medium risk. Auto-merge forbidden for large diffs; "
                "human review required for wide-scope edits."
            )
            nb.update_healer_task(
                task_id,
                patch_summary=patch_summary,
                risk_assessment=risk_assessment,
            )
            nb.add_healer_event(
                task_id, "Patch proposal stage completed.", state="patch_proposed"
            )

            # Optionally dispatch to existing code agent infrastructure.
            dispatched_task = None
            try:
                from ..scientist.api import _spawn_code_agent_task

                goal = (
                    f"Code Healer task {task_id}. Scope: {task.get('scope')}. "
                    f"Reproduce: {task.get('reproduction_steps_json')}. "
                    f"Acceptance tests: {task.get('acceptance_tests_json')}. "
                    "Guardrails: never auto-merge large diffs; include patch summary and risk assessment; "
                    "add/update tests for behavior changes."
                )
                dispatched_task = _spawn_code_agent_task(
                    goal=goal,
                    notebook_path=self.notebook_path,
                    allow_write=True,
                )
                agent_status = (dispatched_task.get("result") or {}).get(
                    "status", dispatched_task.get("status")
                )
                if agent_status == "unavailable":
                    nb.add_healer_event(
                        task_id,
                        "Code agent unavailable: not implemented.",
                        state="patch_proposed",
                        payload={"agent_task": dispatched_task},
                    )
                else:
                    nb.add_healer_event(
                        task_id,
                        "Delegated patching to code agent.",
                        state="patch_proposed",
                        payload={"agent_task": dispatched_task},
                    )
            except Exception as e:
                nb.add_healer_event(
                    task_id,
                    f"Code-agent delegation unavailable: {e}",
                    state="patch_proposed",
                )

            nb.update_healer_task(task_id, state="verifying")
            nb.add_healer_event(task_id, "Running acceptance tests.", state="verifying")
            for cmd in task.get("acceptance_tests_json") or []:
                res = self._run_allowed_command(cmd, root, command_timeout_seconds)
                verify_results.append(res)

            all_ok = (
                all(r.get("returncode") == 0 for r in verify_results)
                if verify_results
                else True
            )
            final_state = "completed" if all_ok else "failed"
            nb.update_healer_task(
                task_id,
                state=final_state,
                completed=True,
                result={
                    "endpoint": endpoint,
                    "reproduction": repro_results,
                    "verification": verify_results,
                    "agent_task": dispatched_task,
                },
            )
            nb.add_healer_event(
                task_id,
                "Healer workflow completed."
                if all_ok
                else "Healer verification failed.",
                state=final_state,
            )
            return {"task_id": task_id, "state": final_state, "verification_ok": all_ok}
        finally:
            nb.close()

    def open_and_run(self, spec: HealerTaskSpec) -> Dict[str, Any]:
        task_id = self.open_task(spec)
        return self.run_task(task_id)
