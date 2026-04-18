#!/usr/bin/env python3
"""Qwen-powered monitor agent — analyzes log_monitor alerts and recommends actions.

Reads monitor_alerts.json (from log_monitor.py), sends a structured summary
to a local Qwen model via Ollama, and writes recommendations to
monitor_actions.json for Claude or the user to act on.

Usage:
    python -m research.tools.monitor_agent                     # default: qwen3.5:2b
    python -m research.tools.monitor_agent --model qwen3.5:0.8b  # lighter model
    python -m research.tools.monitor_agent --interval 60       # check every 60s
    python -m research.tools.monitor_agent --once              # single check, then exit

Requires: ollama running locally (default http://localhost:11434)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _call_ollama(
    prompt: str,
    model: str = "qwen3.5:2b",
    host: str = "http://localhost:11434",
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> Optional[str]:
    """Call Ollama API and return response text."""
    import urllib.request
    import urllib.error

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Qwen 3.5 may put content in "response" or "thinking"
            response = data.get("response", "")
            if not response.strip() and data.get("thinking"):
                response = data["thinking"]
            return response
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return None


def _build_prompt(alerts: Dict[str, Any]) -> str:
    """Build a structured prompt for Qwen from the alert data."""
    health = alerts.get("health", "unknown")
    totals = alerts.get("totals", {})
    exp = alerts.get("current_experiment", {})
    errors = alerts.get("recent_errors", [])
    actions = alerts.get("action_needed", [])
    transitions = alerts.get("experiment_transitions", [])

    # Summarize errors by type
    error_summary = {}
    for e in errors:
        t = e.get("type", "unknown")
        error_summary[t] = error_summary.get(t, 0) + 1

    prompt = f"""You are a pipeline monitor for an AI architecture search system. Analyze this status and respond ONLY with a JSON object containing "status", "issues", and "recommendations". No explanation, just JSON.

CURRENT STATE:
- Health: {health}
- Uptime: {alerts.get("uptime_seconds", 0)}s
- Programs generated: {totals.get("programs", 0)}
- S1 passers: {totals.get("s1_passers", 0)} ({totals.get("s1_rate", 0)}%)
- Errors: {totals.get("errors", 0)}
- DB locks: {totals.get("db_locks", 0)}
- Investigation failures: {totals.get("investigation_failures", 0)}
- Triage runs: {totals.get("triage_runs", 0)}
- Consecutive failures: {totals.get("consecutive_failures", 0)}

CURRENT EXPERIMENT:
- Mode: {exp.get("mode", "?")}
- Programs: {exp.get("programs", 0)}
- S1: {exp.get("s1", 0)}
- Idle: {exp.get("idle_seconds", 0)}s

ERROR SUMMARY: {json.dumps(error_summary)}

ACTIONS FLAGGED: {json.dumps(actions[:5])}

RECENT TRANSITIONS: {json.dumps(list(transitions)[-5:])}

Respond ONLY with a JSON object:
{{"status": "ok|warning|critical", "issues": ["list of issues"], "recommendations": ["list of actions"], "wake_claude": true/false}}

Set wake_claude=true ONLY if there's a critical issue that needs the main AI agent (crashes, data loss, pipeline stuck >10min, investigation loop).
"""
    return prompt


def _analyze(alerts: Dict[str, Any], model: str, host: str) -> Dict[str, Any]:
    """Analyze alerts using Qwen and return structured recommendations."""
    prompt = _build_prompt(alerts)
    response = _call_ollama(prompt, model=model, host=host)

    if not response:
        return {
            "status": "error",
            "issues": ["Failed to reach Qwen model"],
            "recommendations": ["Check if ollama is running"],
            "wake_claude": False,
            "raw_response": None,
        }

    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        # Qwen might wrap it in markdown or add text
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            result["raw_response"] = response
            return result
    except json.JSONDecodeError:
        pass

    return {
        "status": "unknown",
        "issues": ["Could not parse Qwen response"],
        "recommendations": [],
        "wake_claude": False,
        "raw_response": response,
    }


def _wake_claude(issue: str, analysis: Dict[str, Any], actions_file: Path) -> None:
    """Wake Claude Code to handle a critical issue.

    Spawns a Claude Code instance with the issue context and the
    monitor's analysis. Claude runs with --dangerously-skip-permissions
    since the monitor is a trusted automated system.
    """
    import subprocess

    prompt = (
        f"The Aria pipeline monitor detected a critical issue and woke you up.\n\n"
        f"Issue: {issue}\n\n"
        f"Monitor analysis: {json.dumps(analysis.get('issues', []))}\n"
        f"Recommendations: {json.dumps(analysis.get('recommendations', []))}\n\n"
        f"Full monitor state is in {actions_file}\n\n"
        f"Please investigate and fix the issue. Check research/monitor_alerts.json "
        f"for the raw log monitor data. The pipeline is running at "
        f"http://localhost:5000. The database is at research/lab_notebook.db."
    )

    try:
        # Write the prompt to a file for Claude to read
        wake_file = actions_file.parent / "monitor_wake_prompt.md"
        with open(wake_file, "w") as f:
            f.write(f"# Monitor Wake-Up Alert\n\n{prompt}\n")

        print(f"  Wake prompt written to {wake_file}", file=sys.stderr)

        # Spawn Claude Code in the background
        subprocess.Popen(
            [
                "claude",
                "--dangerously-skip-permissions",
                "-p",
                prompt,
            ],
            cwd=str(actions_file.parent.parent),  # project root
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("  Claude Code spawned", file=sys.stderr)
    except FileNotFoundError:
        print("  claude command not found — write wake file only", file=sys.stderr)
    except Exception as e:
        print(f"  Failed to wake Claude: {e}", file=sys.stderr)


def monitor_loop(
    alert_path: str,
    action_path: str,
    model: str,
    host: str,
    interval: float,
    once: bool = False,
) -> None:
    """Main loop: read alerts, analyze, write actions."""
    alerts_file = Path(alert_path)
    actions_file = Path(action_path)
    last_mtime = 0.0

    print(f"Monitor agent: {model} @ {host}", file=sys.stderr)
    print(f"Reading: {alerts_file}", file=sys.stderr)
    print(f"Writing: {actions_file}", file=sys.stderr)

    while True:
        try:
            if not alerts_file.exists():
                if once:
                    print("No alerts file found", file=sys.stderr)
                    return
                time.sleep(interval)
                continue

            # Only re-analyze if alerts file changed
            mtime = alerts_file.stat().st_mtime
            if mtime <= last_mtime and not once:
                time.sleep(interval)
                continue
            last_mtime = mtime

            with open(alerts_file) as f:
                alerts = json.load(f)

            print(f"[{time.strftime('%H:%M:%S')}] Analyzing alerts...", file=sys.stderr)
            result = _analyze(alerts, model, host)

            # Add metadata
            result["analyzed_at"] = time.time()
            result["model"] = model
            result["alert_health"] = alerts.get("health", "unknown")
            result["alert_totals"] = alerts.get("totals", {})

            # Write actions
            tmp = actions_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(result, f, indent=2)
            tmp.rename(actions_file)

            status = result.get("status", "?")
            wake = result.get("wake_claude", False)
            issues = result.get("issues", [])
            recs = result.get("recommendations", [])

            print(
                f"  Status: {status} | Issues: {len(issues)} | Wake Claude: {wake}",
                file=sys.stderr,
            )
            if wake:
                issue_summary = issues[0] if issues else "critical pipeline issue"
                print("  *** WAKING CLAUDE ***", file=sys.stderr)
                _wake_claude(issue_summary, result, actions_file)
            for r in recs[:3]:
                print(f"  → {r}", file=sys.stderr)

            if once:
                return

        except KeyboardInterrupt:
            print("\nMonitor agent stopped", file=sys.stderr)
            return
        except Exception as e:
            print(f"Monitor agent error: {e}", file=sys.stderr)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Qwen-powered pipeline monitor agent")
    parser.add_argument("--model", default="qwen3.5:2b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--alert-file", default="research/monitor_alerts.json")
    parser.add_argument("--action-file", default="research/monitor_actions.json")
    parser.add_argument(
        "--interval", type=float, default=30.0, help="Check interval (seconds)"
    )
    parser.add_argument("--once", action="store_true", help="Single check then exit")
    args = parser.parse_args()

    monitor_loop(
        args.alert_file,
        args.action_file,
        args.model,
        args.host,
        args.interval,
        args.once,
    )


if __name__ == "__main__":
    main()
