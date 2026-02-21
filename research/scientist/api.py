"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import ast
import json
import csv
import io
from collections import deque
from datetime import datetime
import hashlib
import logging
import math
import os
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from .notebook import LabNotebook
from .evidence import build_evidence_pack
from .persona import get_aria
from .runner import ExperimentRunner, RunConfig
from .llm.context import build_program_context

logger = logging.getLogger(__name__)

# Singleton runner shared across requests
_runner: Optional[ExperimentRunner] = None
_CODE_AGENT_TASKS: Dict[str, Dict[str, Any]] = {}
_CODE_AGENT_TASKS_LOCK = threading.Lock()
_WORKSPACE_FILE_INDEX: Dict[str, Dict[str, Any]] = {}
_WORKSPACE_FILE_INDEX_LOCK = threading.Lock()
_WORKSPACE_FILE_INDEX_BUILT_AT: float = 0.0
_RUN_TRIGGER_LOCK = threading.Lock()
_CHAT_GUARDRAIL_LOCK = threading.Lock()
_CHAT_GUARDRAIL_EVENTS = deque(maxlen=500)
_LAST_RUN_TRIGGER: Dict[str, Any] = {
    "experiment_id": None,
    "source": "unknown",
    "mode": None,
    "timestamp": None,
    "details": {},
}

_ALLOWED_CHAT_ACTION_TYPES = {
    "adjust_config",
    "adjust_grammar",
    "start_experiment",
    "edit_file",
    "spawn_agent",
}


def _record_run_trigger(
    experiment_id: str,
    source: str,
    mode: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "experiment_id": str(experiment_id or "").strip() or None,
        "source": str(source or "unknown").strip() or "unknown",
        "mode": (str(mode).strip() if mode else None),
        "timestamp": time.time(),
        "details": details if isinstance(details, dict) else {},
    }
    payload["details"] = {
        key: value for key, value in payload["details"].items() if value is not None
    }
    with _RUN_TRIGGER_LOCK:
        _LAST_RUN_TRIGGER.update(payload)
        return dict(_LAST_RUN_TRIGGER)


def _get_run_trigger_snapshot(active_experiment_id: Optional[str] = None) -> Dict[str, Any]:
    active_id = str(active_experiment_id or "").strip() or None
    with _RUN_TRIGGER_LOCK:
        snap = dict(_LAST_RUN_TRIGGER)
    if active_id and snap.get("experiment_id") and snap.get("experiment_id") != active_id:
        return {
            "experiment_id": active_id,
            "source": "unknown",
            "mode": None,
            "timestamp": None,
            "details": {},
            "matched": False,
        }
    snap["experiment_id"] = active_id or snap.get("experiment_id")
    snap["matched"] = bool(active_id) and bool(snap.get("timestamp")) and snap.get("experiment_id") == active_id
    return snap


def _insight_dedup_key(content: str) -> str:
    """Normalize numeric values to create a stable dedup key for insights.

    Replaces decimals/percentages and multi-digit integers so that
    'appears in 144 survivors' matches 'appears in 145 survivors'.
    Preserves single-digit suffixes in op names like 'split2'.
    """
    import re
    s = re.sub(r'\d+\.\d+%?', '#', content)   # decimals / pcts
    s = re.sub(r'\b\d{2,}\b', '#', s)           # multi-digit ints
    return s


def _deduplicate_insights(insights: list) -> list:
    """Keep only the most recent insight per semantic dedup key."""
    seen: dict = {}
    for ins in insights:
        key = _insight_dedup_key(ins.get("content", ""))
        if key not in seen:
            seen[key] = ins
    return list(seen.values())


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable primitives for API/SSE payloads."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    # Torch-like tensors
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        try:
            tensor_like = value.detach()
            if hasattr(tensor_like, "cpu") and callable(getattr(tensor_like, "cpu")):
                tensor_like = tensor_like.cpu()
            if hasattr(tensor_like, "tolist") and callable(getattr(tensor_like, "tolist")):
                return _json_safe(tensor_like.tolist())
            if hasattr(tensor_like, "item") and callable(getattr(tensor_like, "item")):
                return _json_safe(tensor_like.item())
            return str(tensor_like)
        except Exception:
            return str(value)

    # NumPy-like arrays/scalars
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    return str(value)


def _to_safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float parse with NaN/inf guard."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    return parsed


def _extract_hypothesis_missing_fields(critique: Any) -> List[str]:
    """Derive a stable checklist of missing hypothesis fields for UI display."""
    if not isinstance(critique, dict):
        return []
    out: List[str] = []

    def _add(field: str) -> None:
        f = str(field or "").strip()
        if f and f not in out:
            out.append(f)

    explicit = critique.get("missing_fields")
    if isinstance(explicit, list):
        for field in explicit:
            _add(str(field))
    if out:
        return out

    check_map = {
        "testability": "success_criteria",
        "measurable_metric": "primary_metric",
        "confound_risk": "confounders_checklist",
        "fallback_plan": "fallback_plan",
    }
    for check in critique.get("checks") or []:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "").lower()
        if status not in {"warn", "fail"}:
            continue
        key = str(check.get("key") or "").lower()
        mapped = check_map.get(key)
        if mapped:
            _add(mapped)

    concern_text = " ".join(str(c).lower() for c in (critique.get("concerns") or []))
    if "source-selection" in concern_text:
        _add("source_selection_rule")
    if "mutation" in concern_text and ("operator" in concern_text or "underspecified" in concern_text or "radius" in concern_text):
        _add("mutation_mechanism")
    if "intent" in concern_text and "undefined" in concern_text:
        _add("intent_weights")
    if "success criteria" in concern_text:
        _add("success_criteria")

    return out


def _normalize_entry(entry: dict) -> dict:
    """Normalize notebook entry shape for UI consumers.

    Ensures ``metadata`` is available as a parsed dict while preserving
    original ``metadata_json`` for compatibility.
    """
    normalized = dict(entry)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        return normalized

    raw = normalized.get("metadata_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            normalized["metadata"] = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            normalized["metadata"] = {}
    else:
        normalized["metadata"] = {}
    return normalized


def _normalize_entries(entries: list) -> list:
    return [_normalize_entry(entry) for entry in entries]


def _program_lineage_chain(nb: LabNotebook, result_id: str, max_depth: int = 16) -> List[Dict[str, Any]]:
    """Resolve refinement lineage by walking source_result_id links."""
    chain: List[Dict[str, Any]] = []
    seen: set[str] = set()
    current = str(result_id or "").strip()

    while current and current not in seen and len(chain) < max_depth:
        seen.add(current)
        row = nb.get_program_detail(current)
        if not row:
            break

        graph_parsed = row.get("graph_json_parsed") if isinstance(row, dict) else None
        metadata = graph_parsed.get("metadata") if isinstance(graph_parsed, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        refinement = metadata.get("refinement") if isinstance(metadata.get("refinement"), dict) else {}
        lineage = metadata.get("lineage") if isinstance(metadata.get("lineage"), dict) else {}

        source_result_id = str(refinement.get("source_result_id") or "").strip() or None
        chain.append({
            "result_id": row.get("result_id"),
            "graph_fingerprint": row.get("graph_fingerprint"),
            "experiment_id": row.get("experiment_id"),
            "stage1_passed": bool(row.get("stage1_passed")),
            "loss_ratio": row.get("loss_ratio"),
            "novelty_score": row.get("novelty_score"),
            "refinement": {
                "intent": refinement.get("intent"),
                "intent_score": refinement.get("intent_score"),
                "source_result_id": source_result_id,
                "seed_fingerprint": refinement.get("seed_fingerprint"),
                "fallback": bool(refinement.get("fallback")),
            },
            "lineage": {
                "type": lineage.get("type"),
                "parent": lineage.get("parent"),
            },
        })

        current = source_result_id if source_result_id and source_result_id not in seen else ""

    return chain


def _enrich_program_detail(nb: LabNotebook, program: Dict[str, Any]) -> Dict[str, Any]:
    """Attach derived fields for program detail responses."""
    if program.get("external_benchmarks_json_parsed") is not None:
        program["external_benchmarks"] = program.get("external_benchmarks_json_parsed")
    try:
        from .analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics(nb)
        qkv_usage = analytics.qkv_usage_enum(program)
        program["qkv_usage"] = qkv_usage
        program["uses_qkv"] = qkv_usage != "qkv_free"
        program["compression_metrics"] = analytics.canonical_compression_metrics(program)
        program["reproducibility_packet"] = analytics.reproducibility_packet_status(program)
    except Exception as e:
        logger.debug("Program enrichment failed for %s: %s", program.get("result_id"), e)
    return program


def _entry_to_live_feed_event(entry: dict) -> Optional[dict]:
    """Convert a persisted notebook live-feed entry into UI event shape."""
    normalized = _normalize_entry(entry)
    metadata = normalized.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None

    live_type = metadata.get("live_feed_type")
    payload = metadata.get("payload")
    if not live_type or not isinstance(payload, dict):
        return None

    event = {"type": live_type, **payload}
    ts = normalized.get("timestamp")
    if isinstance(ts, (int, float)):
        event["ts"] = int(ts * 1000)
    return event


def _normalize_hypothesis(hypothesis: dict) -> dict:
    """Normalize campaign hypothesis shape for UI consumers."""
    normalized = dict(hypothesis)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        return normalized

    raw = normalized.get("metadata_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            normalized["metadata"] = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            normalized["metadata"] = {}
    else:
        normalized["metadata"] = {}
    return normalized


_CHAT_TOOL_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "about",
    "what", "when", "where", "which", "does", "have", "into", "your",
    "they", "them", "there", "their", "should", "would", "could",
    "while", "after", "before", "being", "been", "just", "then",
}


def _chat_extract_terms(question: str, max_terms: int = 8) -> List[str]:
    import re

    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question.lower())
    terms: List[str] = []
    seen = set()
    for token in words:
        if token in _CHAT_TOOL_STOPWORDS:
            continue
        if len(token) < 4 and token not in {"api", "llm"}:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _chat_workspace_root(notebook_path: str) -> Path:
    # notebook_path is typically "research/lab_notebook.db" when launched from repo root.
    resolved = Path(notebook_path).resolve()
    if resolved.parent.exists():
        return resolved.parent
    return Path(__file__).resolve().parent.parent


def _chat_should_use_code_tools(question: str) -> bool:
    # Always use local tools — context is cheap, ignorance is expensive.
    return True


def _chat_search_workspace(
    question: str,
    workspace_root: Path,
    max_hits: int = 6,
    max_files: int = 1200,
) -> List[Dict[str, Any]]:
    import re

    terms = _chat_extract_terms(question)
    if not terms:
        return []

    include_ext = {".py", ".js", ".ts", ".tsx", ".md", ".json"}
    skip_dirs = {
        ".git", "node_modules", "__pycache__", "build", "dist",
        ".venv", "venv", ".mypy_cache", ".pytest_cache",
    }

    results: List[Dict[str, Any]] = []
    inspected = 0
    for path in workspace_root.rglob("*"):
        if inspected >= max_files or len(results) >= max_hits:
            break
        if not path.is_file() or path.suffix.lower() not in include_ext:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            if path.stat().st_size > 350_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        inspected += 1

        lowered = text.lower()
        score = sum(lowered.count(t) for t in terms)
        if score <= 0:
            continue

        match_pos = None
        for term in terms:
            idx = lowered.find(term)
            if idx >= 0:
                match_pos = idx
                break
        if match_pos is None:
            continue

        line_no = lowered[:match_pos].count("\n") + 1
        lines = text.splitlines()
        start = max(0, line_no - 2)
        end = min(len(lines), line_no + 1)
        snippet = "\n".join(lines[start:end]).strip()
        rel = str(path.relative_to(workspace_root))
        results.append({
            "path": rel,
            "abs_path": str(path),
            "line": line_no,
            "score": score,
            "snippet": snippet[:320],
        })

    results.sort(key=lambda item: (-int(item.get("score") or 0), item.get("path", "")))
    return results[:max_hits]


def _run_local_chat_agent(
    question: str,
    runner: ExperimentRunner,
    nb: LabNotebook,
    notebook_path: str,
    enable_code_tools: bool = True,
) -> Dict[str, Any]:
    """Collect local runtime and codebase evidence for Aria chat responses."""
    findings: List[str] = []
    tools_used: List[str] = []
    code_hits: List[Dict[str, Any]] = []

    try:
        progress = runner.progress.to_dict()
        tools_used.append("runner.progress")
        findings.append(
            "Runtime: "
            f"status={progress.get('status')}, "
            f"exp={progress.get('experiment_id')}, "
            f"generation={progress.get('current_generation')}/{progress.get('total_generations')}, "
            f"stage1={progress.get('stage1_passed')}"
        )
    except Exception as exc:
        findings.append(f"Runtime probe unavailable: {exc}")

    try:
        recent = nb.get_recent_experiments(5)
        tools_used.append("notebook.get_recent_experiments")
        if recent:
            top = recent[0]
            findings.append(
                "Recent experiment: "
                f"{top.get('experiment_id')} ({top.get('experiment_type')}, {top.get('status')})"
            )
    except Exception as exc:
        findings.append(f"Recent experiment lookup unavailable: {exc}")

    if enable_code_tools and _chat_should_use_code_tools(question):
        try:
            workspace_root = _chat_workspace_root(notebook_path)
            code_hits = _chat_search_workspace(question, workspace_root=workspace_root)
            tools_used.append("workspace.search")
            if code_hits:
                findings.append(f"Code matches found: {len(code_hits)}")
        except Exception as exc:
            findings.append(f"Workspace search unavailable: {exc}")

    summary_lines: List[str] = []
    if findings:
        summary_lines.append("Local agent findings:")
        for item in findings[:6]:
            summary_lines.append(f"- {item}")
    if code_hits:
        summary_lines.append("Code evidence:")
        for hit in code_hits[:6]:
            summary_lines.append(
                f"- {hit.get('path')}:{hit.get('line')} | {hit.get('snippet')}"
            )

    return {
        "tools_used": tools_used,
        "summary": "\n".join(summary_lines)[:3200],
        "code_hits": code_hits,
    }


def _chat_requests_codebase_fix(question: str) -> bool:
    lowered = (question or "").lower()
    triggers = (
        "fix code", "fix codebase", "self repair", "self-repair", "refactor",
        "architecture", "design", "repair", "autofix", "patch",
        "need to fix", "needed to fix", "needs fixing", "anything to fix",
        "what needs fixing", "what do you need to fix", "what should you fix",
        "underlying", "javascript", "python", "js and python",
        "fix issues", "fix issue", "spin off agent", "spin-off agent",
        "spawn agent", "spawn sub-agent", "autonomous fix", "fix yourself",
        "do this yourself", "check and fix", "investigate and fix",
    )
    return any(trigger in lowered for trigger in triggers)


def _chat_requests_summary_response(question: str) -> bool:
    lowered = (question or "").lower()
    triggers = (
        "summary", "summarize", "summarise", "recap",
        "tl;dr", "tldr", "high-level overview", "give me an overview",
    )
    return any(trigger in lowered for trigger in triggers)


def _should_autospawn_self_repair(error_message: str) -> bool:
    lowered = str(error_message or "").lower()
    triggers = (
        "unexpected keyword argument",
        "attributeerror",
        "nameerror",
        "importerror",
        "modulenotfounderror",
        "syntaxerror",
        "typeerror",
        "keyerror",
        "valueerror",
        "traceback",
    )
    return any(token in lowered for token in triggers)


def _chat_requests_brief_response(question: str) -> bool:
    lowered = (question or "").lower()
    triggers = (
        "not babble", "don't babble", "dont babble", "be concise",
        "keep it short", "short answer", "brief response", "be brief",
        "too verbose", "too much text", "tldr", "tl;dr",
    )
    return any(trigger in lowered for trigger in triggers)


def _chat_requests_self_fix_now(question: str) -> bool:
    lowered = (question or "").lower()
    triggers = (
        "fix yourself", "fix itself", "fix this yourself", "fix this itself",
        "spin off agents and fix", "spawn agents and fix", "check and fix yourself",
        "can you fix yourself", "fix your own codebase",
        "fix what's wrong", "fix what is wrong", "fix whats wrong",
        "what's wrong with you", "what is wrong with you",
        "diagnose yourself", "diagnose and fix", "diagnose & fix",
    )
    return any(trigger in lowered for trigger in triggers)


def _chat_question_is_actionable(question: str) -> bool:
    """Detect if the user's question implies they want something DONE, not just explained."""
    lowered = (question or "").lower()
    # Non-actionable: greetings, pure questions about concepts, status checks
    non_actionable = (
        "what is", "what are", "hello", "hi aria", "thanks", "thank you",
        "how are you", "who are you", "what do you think", "status",
    )
    if any(lowered.startswith(na) for na in non_actionable):
        return False
    # Actionable: anything involving problems, improvements, changes, or exploration
    actionable = (
        "fix", "improve", "why is", "why are", "wrong", "broken", "failing",
        "stagnant", "stuck", "bad", "slow", "issue", "problem", "help",
        "optimize", "increase", "decrease", "change", "adjust", "try",
        "run", "start", "stop", "investigate", "look into", "what should",
        "what next", "next step", "recommend", "suggest", "do something",
    )
    return any(trigger in lowered for trigger in actionable)


def _chat_requests_detailed_response(question: str) -> bool:
    lowered = (question or "").lower()
    triggers = (
        "explain in detail", "detailed explanation", "full analysis",
        "deep dive", "step by step why", "verbose", "long form",
        "why this happened", "show reasoning",
    )
    return any(trigger in lowered for trigger in triggers)


def _diagnose_research_issues(analytics_data: Dict, nb) -> List[Dict[str, Any]]:
    """Rule-based diagnosis using analytics data. Returns list of issues with fix actions."""
    issues: List[Dict[str, Any]] = []
    if not analytics_data:
        return issues

    # 1. Sparsity coverage < 15%
    sparse_cov = analytics_data.get("sparse_coverage")
    if isinstance(sparse_cov, (int, float)) and sparse_cov < 15:
        issues.append({
            "issue": f"Sparsity coverage at {sparse_cov:.1f}% (under 15% target)",
            "action_type": "config_fix",
            "config_fix": {"type": "adjust_config", "changes": {
                "model_source": "morphological_box",
                "use_synthesized_training": True,
            }},
        })

    # 2. Compression coverage < 20%
    comp_cov = analytics_data.get("compression_coverage")
    if isinstance(comp_cov, (int, float)) and comp_cov < 20:
        issues.append({
            "issue": f"Compression coverage at {comp_cov:.1f}% (under 20% target)",
            "action_type": "config_fix",
            "config_fix": {"type": "adjust_config", "changes": {
                "morph_ratio": 0.85,
                "math_space_weight": 2.5,
            }},
        })

    # 3. Zero S1 survivors in last 5 experiments
    try:
        recent_exps = nb.get_recent_experiments(5)
        if recent_exps and all(
            (exp.get("n_stage1_passed") or 0) == 0 for exp in recent_exps
        ):
            issues.append({
                "issue": f"Zero S1 survivors in last {len(recent_exps)} experiments",
                "action_type": "config_fix",
                "config_fix": {"type": "adjust_config", "changes": {
                    "max_depth": 2,
                    "max_ops": 5,
                    "residual_prob": 0.85,
                }},
            })
    except Exception:
        pass

    # 4. Dominant failure pattern (zero_grad > 20 occurrences)
    failure_patterns = analytics_data.get("failure_patterns")
    if isinstance(failure_patterns, dict):
        zero_grad_count = failure_patterns.get("zero_grad", 0)
        if isinstance(zero_grad_count, (int, float)) and zero_grad_count > 20:
            issues.append({
                "issue": f"Dominant zero_grad failures ({int(zero_grad_count)} occurrences)",
                "action_type": "config_fix",
                "config_fix": {"type": "adjust_config", "changes": {
                    "residual_prob": 0.9,
                    "max_depth": 3,
                }},
            })

    # 5. Grammar weight drift (any category > 3x default)
    grammar_weights = analytics_data.get("grammar_weights")
    default_weights = analytics_data.get("default_weights")
    if isinstance(grammar_weights, dict) and isinstance(default_weights, dict):
        drifted = {}
        for cat, w in grammar_weights.items():
            default_w = default_weights.get(cat)
            if isinstance(w, (int, float)) and isinstance(default_w, (int, float)) and default_w > 0:
                if w > 3 * default_w:
                    drifted[cat] = round(default_w * 1.5, 2)
        if drifted:
            issues.append({
                "issue": f"Grammar weight drift in {len(drifted)} categories (>3x default)",
                "action_type": "grammar_fix",
                "config_fix": {"type": "adjust_grammar", "weights": drifted},
            })

    # 6. Anti-patterns from negative results (info only)
    neg_results = analytics_data.get("negative_results")
    if isinstance(neg_results, list) and neg_results:
        patterns = [str(r.get("pattern") or r) for r in neg_results[:3] if r]
        if patterns:
            issues.append({
                "issue": f"Anti-patterns detected: {'; '.join(patterns)[:120]}",
                "action_type": "info",
            })

    return issues


def _format_simple_answer_action_plan(text: str) -> str:
    """Compress LLM response to max 2 sentences, 200 chars. Strip fluff ruthlessly."""
    import re as _re

    raw = str(text or "").strip()
    if not raw:
        return raw

    # Strip code blocks (action blocks handled separately)
    raw = _re.sub(r"```[\s\S]*?```", "", raw).strip()
    # Collapse whitespace
    cleaned = _re.sub(r"\s+", " ", raw)
    sentences = [
        s.strip()
        for s in _re.split(r"(?<=[.!?])\s+", cleaned)
        if s.strip() and len(s.strip()) > 10
    ]
    if sentences:
        result = " ".join(sentences[:2])
        if len(result) > 200:
            result = result[:197].rsplit(" ", 1)[0] + "..."
        return result
    return cleaned[:200]


def _truncate_summary(text: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _record_chat_guardrail_event(
    *,
    actionable: bool,
    advice_only: bool,
    summary_text: str,
) -> None:
    event = {
        "timestamp": time.time(),
        "actionable": bool(actionable),
        "advice_only": bool(advice_only),
        "summary_length": len(str(summary_text or "")),
    }
    with _CHAT_GUARDRAIL_LOCK:
        _CHAT_GUARDRAIL_EVENTS.append(event)


def _chat_guardrail_snapshot(window: int = 200) -> Dict[str, Any]:
    window_size = max(1, min(int(window or 200), 500))
    with _CHAT_GUARDRAIL_LOCK:
        events = list(_CHAT_GUARDRAIL_EVENTS)[-window_size:]
    n = len(events)
    if n == 0:
        return {
            "window": window_size,
            "n_events": 0,
            "actionable_response_rate": 0.0,
            "advice_only_rate": 0.0,
            "summary_length": {"avg": 0.0, "p95": 0.0, "max": 0},
        }

    actionable_count = sum(1 for e in events if e.get("actionable"))
    advice_only_count = sum(1 for e in events if e.get("advice_only"))
    lengths = sorted(int(e.get("summary_length") or 0) for e in events)
    p95_idx = max(0, min(len(lengths) - 1, int(math.ceil(0.95 * len(lengths))) - 1))
    return {
        "window": window_size,
        "n_events": n,
        "actionable_response_rate": round(actionable_count / n, 4),
        "advice_only_rate": round(advice_only_count / n, 4),
        "summary_length": {
            "avg": round(sum(lengths) / n, 2),
            "p95": lengths[p95_idx],
            "max": lengths[-1],
        },
    }


def _summarize_agent_task(task: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = dict(task or {})
    status = str(row.get("status") or "queued").strip().lower()
    phase = str(row.get("phase") or "").strip().lower()
    applied = len(row.get("applied_edits") or [])
    proposed = len(row.get("proposed_edits") or [])
    skipped = len(row.get("skipped_edits") or [])
    summary = str(row.get("summary") or "").strip()
    timings = row.get("timings") if isinstance(row.get("timings"), dict) else {}

    if status in {"queued", "running"}:
        phase_label = phase.replace("_", " ") if phase else "working"
        headline = f"{status.upper()} · {phase_label}"
    elif status == "completed":
        headline = f"COMPLETED · applied {applied}, proposed {proposed}, skipped {skipped}"
    else:
        headline = f"{status.upper()} · review required"

    if summary:
        headline = f"{headline} · {_truncate_summary(summary, max_chars=140)}"
    if timings:
        total_s = sum(float(v or 0.0) for v in timings.values())
        if total_s > 0:
            headline = f"{headline} · {total_s:.1f}s"

    return {
        "task_id": row.get("task_id"),
        "status": status,
        "phase": phase,
        "updated_at": row.get("updated_at"),
        "allow_write": bool(row.get("allow_write")),
        "milestone_summary": _truncate_summary(headline, max_chars=220),
        "full_status_url": f"/api/aria/agent/status/{row.get('task_id')}?detail=full",
    }


def _parse_action_contract_response(text: str) -> Dict[str, Any]:
    """Parse LLM output under strict action contract, returning summary + typed actions."""
    raw = str(text or "").strip()
    if not raw:
        return {"summary": "", "actions": [], "advice_only": True}

    pattern = re.compile(r"```(\w+)?\s*\n(.*?)\n```", re.DOTALL)
    action_blocks: List[Dict[str, Any]] = []
    retained_text = raw
    for match in pattern.finditer(raw):
        block_lang = str(match.group(1) or "").strip().lower()
        block_body = str(match.group(2) or "").strip()
        if block_lang != "action":
            continue
        try:
            parsed = json.loads(block_body)
        except Exception:
            continue
        if isinstance(parsed, dict) and str(parsed.get("type") or "") in _ALLOWED_CHAT_ACTION_TYPES:
            action_blocks.append(parsed)

    retained_text = pattern.sub("", retained_text).strip()
    retained_text = re.sub(r"`{3}[\s\S]*?`{3}", "", retained_text).strip()
    summary = _truncate_summary(_format_simple_answer_action_plan(retained_text), max_chars=220)
    advice_only = len(action_blocks) == 0
    return {
        "summary": summary,
        "actions": action_blocks,
        "advice_only": advice_only,
    }

def _code_agent_task_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    with _CODE_AGENT_TASKS_LOCK:
        task = _CODE_AGENT_TASKS.get(task_id)
        return dict(task) if isinstance(task, dict) else None


def _code_agent_task_update(task_id: str, **fields: Any) -> None:
    with _CODE_AGENT_TASKS_LOCK:
        task = _CODE_AGENT_TASKS.get(task_id)
        if not isinstance(task, dict):
            return
        task.update(fields)
        task["updated_at"] = time.time()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _is_remote_primary_llm(llm: Any) -> bool:
    name = str(getattr(llm, "name", "") or "").strip().lower()
    return name in {"anthropic", "openai"}


def _get_local_ollama_settings() -> Dict[str, Any]:
    model = str(os.environ.get("ARIA_LOCAL_OLLAMA_MODEL", "qwen2.5-coder:7b-instruct") or "").strip()
    small_model = str(os.environ.get("ARIA_LOCAL_OLLAMA_SMALL_MODEL", "qwen2.5-coder:3b") or "").strip()
    host = str(os.environ.get("OLLAMA_HOST", "http://localhost:11434") or "").strip()
    try:
        max_vram_gb = float(os.environ.get("ARIA_LOCAL_OLLAMA_MAX_VRAM_GB", "10") or "10")
    except Exception:
        max_vram_gb = 10.0
    try:
        max_small_workers = int(os.environ.get("ARIA_LOCAL_OLLAMA_3B_MAX_WORKERS", "3") or "3")
    except Exception:
        max_small_workers = 3
    if max_vram_gb <= 0:
        max_vram_gb = 10.0
    if max_small_workers < 1:
        max_small_workers = 1
    if max_small_workers > 3:
        max_small_workers = 3
    return {
        "model": model,
        "small_model": small_model,
        "host": host,
        "max_vram_gb": max_vram_gb,
        "max_small_workers": max_small_workers,
    }


def _build_workspace_file_index(workspace_root: Path, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Walk workspace and build a lightweight file index with AST/regex metadata."""
    global _WORKSPACE_FILE_INDEX, _WORKSPACE_FILE_INDEX_BUILT_AT

    with _WORKSPACE_FILE_INDEX_LOCK:
        if not force and _WORKSPACE_FILE_INDEX and (time.time() - _WORKSPACE_FILE_INDEX_BUILT_AT) < 300:
            return dict(_WORKSPACE_FILE_INDEX)

    include_ext = {".py", ".js", ".ts", ".tsx", ".md", ".json"}
    skip_dirs = {
        ".git", "node_modules", "__pycache__", "build", "dist",
        ".venv", "venv", ".mypy_cache", ".pytest_cache",
    }

    index: Dict[str, Dict[str, Any]] = {}
    for path in workspace_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in include_ext:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            stat = path.stat()
            if stat.st_size > 500_000:
                continue
        except Exception:
            continue

        rel = str(path.relative_to(workspace_root))
        entry: Dict[str, Any] = {
            "rel_path": rel,
            "abs_path": str(path),
            "size_bytes": stat.st_size,
            "suffix": path.suffix.lower(),
            "mtime": stat.st_mtime,
            "top_level_names": [],
            "docstring": "",
            "imports": [],
            "line_count": 0,
        }

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            entry["line_count"] = len(lines)
        except Exception:
            index[rel] = entry
            continue

        suffix = path.suffix.lower()
        if suffix == ".py":
            try:
                tree = ast.parse(text, filename=rel)
                names = []
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        names.append(node.name)
                entry["top_level_names"] = names
                docstr = ast.get_docstring(tree)
                if docstr:
                    entry["docstring"] = docstr[:200]
                imps = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imps.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imps.append(node.module)
                entry["imports"] = imps
            except Exception:
                pass
        elif suffix in (".js", ".ts", ".tsx"):
            names = re.findall(r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)", text)
            entry["top_level_names"] = names[:30]
            first_comment = re.search(r"(?://|/\*)\s*(.{1,120})", text)
            if first_comment:
                entry["docstring"] = first_comment.group(1).strip()
        elif suffix in (".md", ".json"):
            entry["docstring"] = text[:100].strip()

        index[rel] = entry

    with _WORKSPACE_FILE_INDEX_LOCK:
        _WORKSPACE_FILE_INDEX = index
        _WORKSPACE_FILE_INDEX_BUILT_AT = time.time()

    return dict(index)


def _query_file_index(
    goal: str,
    workspace_root: Path,
    max_results: int = 12,
) -> List[Dict[str, Any]]:
    """Score index entries by term relevance and return top matches."""
    index = _build_workspace_file_index(workspace_root)
    terms = _chat_extract_terms(goal)
    if not terms:
        return list(index.values())[:max_results]

    scored: List[tuple] = []
    for rel, entry in index.items():
        score = 0.0
        rel_lower = rel.lower()
        names_lower = " ".join(entry.get("top_level_names", [])).lower()
        doc_lower = (entry.get("docstring") or "").lower()
        imports_lower = " ".join(entry.get("imports", [])).lower()
        for term in terms:
            t = term.lower()
            if t in rel_lower:
                score += 3.0
            if t in names_lower:
                score += 2.0
            if t in doc_lower:
                score += 1.0
            if t in imports_lower:
                score += 1.0
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:max_results]]


def _ollama_model_estimated_vram_gb(model: str, host: str) -> Optional[float]:
    model_name = str(model or "").strip()
    if not model_name:
        return None

    try:
        import requests

        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code != 200:
            return None
        payload = resp.json() if resp.content else {}
    except Exception:
        return None

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None

    for item in models:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("name") or item.get("model") or "").strip()
        if candidate != model_name:
            continue
        size_bytes = item.get("size")
        try:
            size_gb = float(size_bytes) / (1024 ** 3)
        except Exception:
            return None
        return round(size_gb, 3)

    return None


def _ollama_offload_model(model: str, host: str) -> None:
    """Offload a model from Ollama's VRAM to free GPU memory for training.

    Sends a generate request with keep_alive=0 which tells Ollama to
    immediately unload the model from memory after responding.
    """
    try:
        import requests
        requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=5,
        )
    except Exception:
        # Best-effort — don't block the caller if offload fails
        pass


def _local_ollama_helper_status(main_llm: Any) -> Dict[str, Any]:
    import shutil

    settings = _get_local_ollama_settings()
    model = settings["model"]
    small_model = settings["small_model"]
    host = settings["host"]
    max_vram_gb = settings["max_vram_gb"]
    max_small_workers = settings["max_small_workers"]
    remote_primary = _is_remote_primary_llm(main_llm)

    status: Dict[str, Any] = {
        "enabled": False,
        "reason": "primary_llm_not_remote",
        "model": model,
        "small_model": small_model,
        "host": host,
        "max_vram_gb": max_vram_gb,
        "estimated_vram_gb": None,
        "small_model_estimated_vram_gb": None,
        "small_model_max_workers": max_small_workers,
        "small_model_workers_available": 0,
    }

    if not remote_primary:
        return status
    if not model:
        status["reason"] = "model_not_configured"
        return status
    if not shutil.which("ollama"):
        status["reason"] = "ollama_cli_missing"
        return status

    small_est = _ollama_model_estimated_vram_gb(model=small_model, host=host)
    status["small_model_estimated_vram_gb"] = small_est
    status["small_model_workers_available"] = 1 if small_est and small_est < max_vram_gb else 0

    estimated = _ollama_model_estimated_vram_gb(model=model, host=host)
    status["estimated_vram_gb"] = estimated
    if estimated is None:
        status["reason"] = "model_not_found_or_unreachable"
        return status
    if estimated >= max_vram_gb:
        status["reason"] = "vram_limit_exceeded"
        return status

    status["enabled"] = True
    status["reason"] = "ok"
    return status


def _run_understanding_agent(
    file_entry: Dict[str, Any],
    goal: str,
    host: str,
    small_model: str,
    workspace_root: Path,
) -> Optional[Dict[str, Any]]:
    """Read one file, send to 3b model for understanding, return structured analysis."""
    import requests as _requests

    rel_path = file_entry.get("rel_path", "")
    abs_path = file_entry.get("abs_path", "")
    line_count = int(file_entry.get("line_count") or 0)
    target = Path(abs_path) if abs_path else _resolve_code_agent_target(workspace_root, rel_path)
    if target is None or not target.exists():
        return None

    try:
        text = target.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
    except Exception:
        return None

    # Build focused excerpt: small files whole, large files keyword-windowed
    if len(lines) <= 120:
        excerpt = text
    else:
        terms = _chat_extract_terms(goal)
        sections = lines[:30]  # header
        # Find keyword-matched windows
        for i, line in enumerate(lines):
            low = line.lower()
            if any(t in low for t in terms):
                start = max(0, i - 5)
                end = min(len(lines), i + 15)
                sections.append(f"\n... (line {start + 1}) ...")
                sections.extend(lines[start:end])
                break
        sections.append(f"\n... (line {max(0, len(lines) - 10) + 1}) ...")
        sections.extend(lines[-10:])
        excerpt = "\n".join(sections)

    if len(excerpt) > 3000:
        excerpt = excerpt[:3000] + "\n..."

    prompt = (
        "You are a code reader. Read this file and answer about it.\n"
        f"FILE: {rel_path} ({line_count} lines, showing relevant sections)\n"
        "---\n"
        f"{excerpt}\n"
        "---\n"
        f'PROBLEM: "{goal}"\n'
        "Answer JSON only:\n"
        '{"purpose":"1-2 sentences","relevant_sections":[{"lines":"145-162","what":"description","code_snippet":"exact 3 lines"}],'
        '"key_observations":["..."],"suggested_fix_location":"Line 152: ..."}'
    )

    for attempt in range(2):
        if attempt == 1:
            # Simplified fallback prompt
            prompt = (
                f"Read this file and list relevant function names and line numbers for: {goal}\n"
                f"FILE: {rel_path}\n---\n{excerpt[:1500]}\n---\n"
                'Return JSON: {"purpose":"...","relevant_sections":[],"key_observations":["..."],"suggested_fix_location":"..."}'
            )
        try:
            resp = _requests.post(
                f"{host.rstrip('/')}/api/generate",
                json={
                    "model": small_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 512, "temperature": 0.1},
                },
                timeout=60,
            )
            if resp.status_code != 200:
                continue
            body = resp.json()
            raw = body.get("response", "")
            parsed = _extract_json_object(raw)
            if isinstance(parsed, dict):
                parsed["file"] = rel_path
                _ollama_offload_model(small_model, host)
                return parsed
        except Exception:
            pass

    _ollama_offload_model(small_model, host)
    return None


def _run_understanding_phase(
    goal: str,
    relevant_files: List[Dict[str, Any]],
    host: str,
    small_model: str,
    workspace_root: Path,
    max_files: int = 4,
) -> List[Dict[str, Any]]:
    """Run understanding agents sequentially on top files, return list of analyses."""
    results: List[Dict[str, Any]] = []
    for entry in relevant_files[:max_files]:
        understanding = _run_understanding_agent(
            file_entry=entry,
            goal=goal,
            host=host,
            small_model=small_model,
            workspace_root=workspace_root,
        )
        if understanding:
            results.append(understanding)
    return results


def _build_agent_plan(
    goal: str,
    file_understandings: List[Dict[str, Any]],
    file_index_hits: List[Dict[str, Any]],
    workspace_root: Path,
    llm: Any,
) -> Dict[str, Any]:
    """Synthesize understanding into an exact edit plan using primary LLM or 7b fallback."""
    understanding_text = json.dumps(file_understandings, indent=1, default=str)[:2500]
    index_summary_lines = []
    for entry in file_index_hits[:12]:
        names = ", ".join(entry.get("top_level_names", [])[:8])
        purpose = (entry.get("docstring") or "")[:80]
        index_summary_lines.append(f"  {entry.get('rel_path')} — names: [{names}] — {purpose}")
    index_summary = "\n".join(index_summary_lines)

    prompt = (
        "You are Aria planning a code fix.\n"
        f"PROBLEM: {goal}\n\n"
        "FILE UNDERSTANDING (from reader agents):\n"
        f"{understanding_text}\n\n"
        "AVAILABLE FILES (from index):\n"
        f"{index_summary}\n\n"
        "Create a precise fix plan with EXACT find/replace text from the understanding snippets.\n"
        "Return JSON: {\"complexity\":\"simple|moderate|complex\",\"summary\":\"...\","
        "\"steps\":[{\"file\":\"...\",\"find\":\"exact old text\",\"replace\":\"new text\","
        "\"line_range\":\"150-152\",\"reason\":\"...\",\"confidence\":\"high|medium|low\"}],"
        "\"notes\":[]}\n"
        "Max 6 steps. find text must be EXACT from understanding snippets.\n"
    )

    # Try primary LLM first
    plan_text = None
    planner_backend = "none"
    if llm:
        try:
            planning_payload: Dict[str, Any] = {"text": None, "error": None}

            def _llm_call() -> None:
                try:
                    from .llm.prompts import SYSTEM_PROMPT
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1400)
                    planning_payload["text"] = (getattr(resp, "text", "") or "").strip()
                except Exception:
                    try:
                        resp = llm.generate(prompt, max_tokens=1400)
                        planning_payload["text"] = (getattr(resp, "text", "") or "").strip()
                    except Exception as exc:
                        planning_payload["error"] = str(exc)

            t = threading.Thread(target=_llm_call, daemon=True)
            t.start()
            t.join(timeout=45.0)

            if not t.is_alive() and not planning_payload.get("error"):
                plan_text = planning_payload.get("text")
                planner_backend = "primary_llm"
        except Exception:
            pass

    # Fallback to 7b via Ollama
    if not plan_text:
        import requests as _requests

        settings = _get_local_ollama_settings()
        model_7b = settings["model"]
        ollama_host = settings["host"]
        try:
            resp = _requests.post(
                f"{ollama_host.rstrip('/')}/api/generate",
                json={
                    "model": model_7b,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 1024, "temperature": 0.2},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                body = resp.json()
                plan_text = body.get("response", "")
                planner_backend = "local_ollama_7b"
            _ollama_offload_model(model_7b, ollama_host)
        except Exception:
            _ollama_offload_model(settings["model"], settings["host"])

    if not plan_text:
        return {
            "ok": False,
            "reason": "planning_failed",
            "plan": {"complexity": "unknown", "summary": "No planner available.", "steps": [], "notes": ["Both primary LLM and local 7b failed."]},
            "planner_backend": planner_backend,
            "execution_strategy": "none",
        }

    parsed = _extract_json_object(plan_text)
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "reason": "invalid_plan_json",
            "plan": {"complexity": "unknown", "summary": "Plan was not valid JSON.", "steps": [], "notes": [plan_text[:300]]},
            "planner_backend": planner_backend,
            "execution_strategy": "none",
        }

    steps = parsed.get("steps") if isinstance(parsed.get("steps"), list) else []
    all_high = all(str(s.get("confidence", "")).lower() == "high" for s in steps if isinstance(s, dict))
    strategy = "direct" if (all_high and len(steps) <= 2 and steps) else "delegate_7b"

    return {
        "ok": True,
        "reason": "ok",
        "plan": parsed,
        "planner_backend": planner_backend,
        "execution_strategy": strategy,
    }


def _run_execution_agent(
    step: Dict[str, Any],
    host: str,
    model: str,
    workspace_root: Path,
) -> Optional[Dict[str, Any]]:
    """Execute a single plan step. Use plan directly if find text matches, else delegate to 7b."""
    import requests as _requests

    rel_path = str(step.get("file") or "").strip()
    find_text = str(step.get("find") or "")
    replace_text = str(step.get("replace") or "")
    reason = str(step.get("reason") or "")

    if not rel_path or not find_text:
        return None

    target = _resolve_code_agent_target(workspace_root, rel_path)
    if target is None or not target.exists():
        return None

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # If plan's find text exists verbatim — use it directly (no model call)
    if find_text in content:
        return {"path": rel_path, "find": find_text, "replace": replace_text, "reason": reason}

    # Find text didn't match — delegate to 7b for correction
    line_range = str(step.get("line_range") or "")
    lines = content.splitlines()
    start_line, end_line = 0, len(lines)
    if line_range:
        parts = line_range.split("-")
        try:
            start_line = max(0, int(parts[0]) - 1)
            end_line = min(len(lines), int(parts[-1]))
        except Exception:
            pass
    # Expand context window around target lines
    ctx_start = max(0, start_line - 5)
    ctx_end = min(len(lines), end_line + 5)
    context_lines = lines[ctx_start:ctx_end]
    context_text = "\n".join(f"{ctx_start + i + 1}: {l}" for i, l in enumerate(context_lines))

    prompt = (
        "Apply ONE change to this file.\n"
        f"FILE: {rel_path}\n"
        f"CURRENT (lines {ctx_start + 1}-{ctx_end}):\n"
        f"{context_text}\n\n"
        f'CHANGE: Replace "{find_text[:200]}" with "{replace_text[:200]}"\n'
        f"REASON: {reason}\n"
        'Return JSON: {"path":"...","find":"exact old text from CURRENT","replace":"new text","reason":"..."}'
    )

    try:
        resp = _requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 512, "temperature": 0.1},
            },
            timeout=60,
        )
        _ollama_offload_model(model, host)
        if resp.status_code != 200:
            return None
        body = resp.json()
        raw = body.get("response", "")
        parsed = _extract_json_object(raw)
        if isinstance(parsed, dict) and parsed.get("find"):
            return parsed
    except Exception:
        _ollama_offload_model(model, host)

    return None


def _resolve_code_agent_target(workspace_root: Path, rel_path: str) -> Optional[Path]:
    safe_rel = str(rel_path or "").strip().lstrip("/")
    if not safe_rel:
        return None
    if safe_rel.startswith(".."):
        return None
    target = (workspace_root / safe_rel).resolve()
    try:
        target.relative_to(workspace_root)
    except Exception:
        return None
    return target


def _validate_changed_file(path: Path) -> Optional[str]:
    if path.suffix.lower() == ".py":
        import py_compile

        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            return f"python syntax check failed: {exc}"

    if path.suffix.lower() == ".js":
        import shutil
        import subprocess

        node_bin = shutil.which("node")
        if node_bin:
            proc = subprocess.run(
                [node_bin, "--check", str(path)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()
                return f"javascript syntax check failed: {stderr[:240]}"
    return None


def _apply_code_agent_edits(
    edits: List[Dict[str, Any]],
    workspace_root: Path,
    allow_write: bool,
    max_edits: int = 8,
) -> Dict[str, Any]:
    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    proposed: List[Dict[str, Any]] = []

    for edit in (edits or [])[:max_edits]:
        if not isinstance(edit, dict):
            continue
        rel_path = str(edit.get("path") or "").strip()
        find_text = str(edit.get("find") or "")
        replace_text = str(edit.get("replace") or "")
        reason = str(edit.get("reason") or "").strip()

        target = _resolve_code_agent_target(workspace_root, rel_path)
        if target is None:
            skipped.append({"path": rel_path, "reason": "invalid or disallowed target path"})
            continue
        if not target.exists() or not target.is_file():
            skipped.append({"path": rel_path, "reason": "target file does not exist"})
            continue
        if not find_text:
            skipped.append({"path": rel_path, "reason": "missing find text"})
            continue

        record = {
            "path": rel_path,
            "abs_path": str(target),
            "reason": reason,
            "find": find_text[:120],
            "replace": replace_text[:120],
        }

        if not allow_write:
            proposed.append(record)
            continue

        try:
            original = target.read_text(encoding="utf-8")
        except Exception as exc:
            skipped.append({"path": rel_path, "reason": f"read failed: {exc}"})
            continue

        if find_text not in original:
            skipped.append({"path": rel_path, "reason": "find text not present"})
            continue

        updated = original.replace(find_text, replace_text, 1)
        if updated == original:
            skipped.append({"path": rel_path, "reason": "no-op replacement"})
            continue

        try:
            target.write_text(updated, encoding="utf-8")
            validation_error = _validate_changed_file(target)
            if validation_error:
                target.write_text(original, encoding="utf-8")
                skipped.append({"path": rel_path, "reason": validation_error})
                continue
            applied.append(record)
        except Exception as exc:
            try:
                target.write_text(original, encoding="utf-8")
            except Exception:
                pass
            skipped.append({"path": rel_path, "reason": f"write failed: {exc}"})

    return {
        "applied": applied,
        "proposed": proposed,
        "skipped": skipped,
    }


def _agent_post_summary_to_chat(task_id: str) -> None:
    """Post a one-line agent completion summary to the originating chat session."""
    snap = _code_agent_task_snapshot(task_id)
    if not snap:
        return
    session_id = str(snap.get("session_id") or "").strip()
    nb_path = str(snap.get("notebook_path") or "").strip()
    if not session_id or not nb_path:
        return

    status = snap.get("status", "unknown")
    applied = snap.get("applied_edits") or []
    skipped = snap.get("skipped_edits") or []
    goal = str(snap.get("goal") or "")[:50]

    if status == "completed" and applied:
        files = ", ".join(dict.fromkeys(e.get("path", "?") for e in applied))
        msg = f"Agent {task_id}: Applied {len(applied)} edit(s) to {files}. Validated OK."
    elif status == "completed":
        msg = f"Agent {task_id}: No applicable edits found for '{goal}'."
        if skipped:
            msg += f" ({len(skipped)} skipped)"
    else:
        summary = str(snap.get("summary") or "")[:100]
        msg = f"Agent {task_id}: Failed — {summary or 'unknown error'}."

    try:
        nb = LabNotebook(nb_path)
        nb.save_chat_message(session_id=session_id, role="aria", text=msg, label="Aria")
        nb.close()
    except Exception:
        pass


def _run_code_agent_task(
    task_id: str,
    goal: str,
    notebook_path: str,
    allow_write: bool,
) -> None:
    """4-phase code agent pipeline: Index → Understand → Plan → Execute."""
    _code_agent_task_update(task_id, status="running", started_at=time.time(), phase="index")
    timings: Dict[str, float] = {}
    try:
        workspace_root = _chat_workspace_root(notebook_path)
        settings = _get_local_ollama_settings()
        host = settings["host"]
        small_model = settings["small_model"]
        model_7b = settings["model"]

        llm = get_aria()._get_llm()
        if llm:
            try:
                if hasattr(llm, "is_available") and not llm.is_available():
                    llm = None
            except Exception:
                llm = None

        # --- Phase 1: Index ---
        t0 = time.time()
        _code_agent_task_update(task_id, phase="index")
        relevant_files = _query_file_index(goal, workspace_root, max_results=12)
        timings["index"] = time.time() - t0

        # --- Phase 2: Understand ---
        t0 = time.time()
        _code_agent_task_update(task_id, phase="understand")
        understandings: List[Dict[str, Any]] = []
        if relevant_files:
            understandings = _run_understanding_phase(
                goal=goal,
                relevant_files=relevant_files,
                host=host,
                small_model=small_model,
                workspace_root=workspace_root,
                max_files=4,
            )
        timings["understand"] = time.time() - t0

        # --- Phase 3: Plan ---
        t0 = time.time()
        _code_agent_task_update(task_id, phase="plan")
        plan_result = _build_agent_plan(
            goal=goal,
            file_understandings=understandings,
            file_index_hits=relevant_files,
            workspace_root=workspace_root,
            llm=llm,
        )
        timings["plan"] = time.time() - t0

        plan = plan_result.get("plan") or {}
        planner_backend = plan_result.get("planner_backend", "none")
        strategy = plan_result.get("execution_strategy", "none")
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

        # --- Phase 4: Execute ---
        t0 = time.time()
        _code_agent_task_update(task_id, phase="execute")
        edits: List[Dict[str, Any]] = []

        if strategy == "direct":
            # High-confidence plan — convert steps directly to edit dicts
            for step in steps:
                if isinstance(step, dict) and step.get("find"):
                    edits.append({
                        "path": step.get("file", ""),
                        "find": step.get("find", ""),
                        "replace": step.get("replace", ""),
                        "reason": step.get("reason", ""),
                    })
        elif strategy == "delegate_7b":
            # For each step: try direct match first, then delegate to 7b
            for step in steps:
                if not isinstance(step, dict):
                    continue
                edit = _run_execution_agent(
                    step=step,
                    host=host,
                    model=model_7b,
                    workspace_root=workspace_root,
                )
                if edit:
                    edits.append(edit)

        edit_result = _apply_code_agent_edits(
            edits, workspace_root=workspace_root, allow_write=allow_write,
        )
        timings["execute"] = time.time() - t0

        # Force-rebuild index after edits
        if edit_result.get("applied"):
            _build_workspace_file_index(workspace_root, force=True)

        _code_agent_task_update(
            task_id,
            status="completed",
            finished_at=time.time(),
            phase="done",
            planner_backend=planner_backend,
            execution_strategy=strategy,
            main_llm_backend=str(getattr(llm, "name", "") or "none").strip().lower() if llm else "none",
            local_ollama_used=strategy in ("direct", "delegate_7b"),
            summary=str(plan.get("summary") or "").strip(),
            notes=plan.get("notes") if isinstance(plan.get("notes"), list) else [],
            understanding_count=len(understandings),
            index_hits=len(relevant_files),
            plan_steps=len(steps),
            timings=timings,
            search_hits=[
                {"path": f.get("rel_path"), "abs_path": f.get("abs_path")}
                for f in relevant_files[:8]
            ],
            proposed_edits=edit_result.get("proposed", []),
            applied_edits=edit_result.get("applied", []),
            skipped_edits=edit_result.get("skipped", []),
        )
        _agent_post_summary_to_chat(task_id)
    except Exception as exc:
        _code_agent_task_update(
            task_id,
            status="failed",
            finished_at=time.time(),
            phase="error",
            summary="Codebase agent task failed before completion.",
            notes=[str(exc), traceback.format_exc()[:1200]],
            timings=timings,
        )
        _agent_post_summary_to_chat(task_id)


def _spawn_code_agent_task(
    goal: str,
    notebook_path: str,
    allow_write: bool = True,
    session_id: str = "",
) -> Dict[str, Any]:
    task_id = f"agent-{uuid.uuid4().hex[:10]}"
    now = time.time()
    task = {
        "task_id": task_id,
        "goal": goal,
        "allow_write": bool(allow_write),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "session_id": session_id,
        "notebook_path": notebook_path,
        "summary": "",
        "notes": [],
        "search_hits": [],
        "proposed_edits": [],
        "applied_edits": [],
        "skipped_edits": [],
    }
    with _CODE_AGENT_TASKS_LOCK:
        _CODE_AGENT_TASKS[task_id] = task

    worker = threading.Thread(
        target=_run_code_agent_task,
        args=(task_id, goal, notebook_path, bool(allow_write)),
        daemon=True,
        name=f"aria-code-agent-{task_id}",
    )
    worker.start()

    snap = _code_agent_task_snapshot(task_id)
    return snap or task


def _compute_compression_opportunities(coverage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive actionable compression opportunities from coverage aggregates."""
    coverage = coverage or {}
    totals = coverage.get("totals") or {}
    techniques = coverage.get("techniques") or []

    n_tested = int(totals.get("n_tested") or 0)
    n_survived = int(totals.get("n_survived") or 0)
    n_compressed_tested = int(totals.get("n_compressed_tested") or 0)
    n_compressed_survived = int(totals.get("n_compressed_survived") or 0)

    compressed_test_share = (
        n_compressed_tested / n_tested if n_tested > 0 else 0.0
    )
    compressed_survival_rate = (
        n_compressed_survived / n_compressed_tested
        if n_compressed_tested > 0
        else 0.0
    )
    overall_survival_rate = n_survived / n_tested if n_tested > 0 else 0.0

    dense_bucket = next(
        (t for t in techniques if str(t.get("technique") or "").lower() in {"dense", "dense_matrix", "standard_float"}),
        None,
    )
    dense_survival_rate = float(dense_bucket.get("survival_rate") or 0.0) if dense_bucket else 0.0

    ranked = sorted(
        techniques,
        key=lambda item: (
            float(item.get("survival_rate") or 0.0),
            float(item.get("avg_quality_retention") or 0.0),
            float(item.get("n_tested") or 0.0),
        ),
        reverse=True,
    )

    recommendations: List[Dict[str, Any]] = []
    if compressed_test_share < 0.2:
        recommendations.append({
            "title": "Expand compression exploration",
            "rationale": (
                f"Only {compressed_test_share * 100:.1f}% of tested programs use "
                "compressed parameterization. Increase compactness-focused synthesis runs."
            ),
            "suggested_config": {
                "mode": "synthesis",
                "model_source": "mixed",
                "morph_ratio": 0.85,
                "max_depth": 5,
                "max_ops": 8,
                "math_space_weight": 1.8,
                "residual_prob": 0.85,
                "n_programs": 80,
            },
        })

    if n_compressed_tested > 0 and compressed_survival_rate < max(dense_survival_rate, overall_survival_rate):
        recommendations.append({
            "title": "Stabilize compressed candidates",
            "rationale": (
                "Compressed candidates survive less often than the current baseline. "
                "Favor gradient-safe compact architectures before increasing novelty pressure."
            ),
            "suggested_config": {
                "mode": "synthesis",
                "max_depth": 4,
                "max_ops": 7,
                "residual_prob": 0.9,
                "math_space_weight": 1.5,
                "n_programs": 70,
            },
        })

    if ranked:
        top = ranked[0]
        recommendations.append({
            "title": f"Scale proven compact technique: {top.get('technique')}",
            "rationale": (
                f"Technique '{top.get('technique')}' currently has the strongest "
                f"survival/quality profile ({float(top.get('survival_rate') or 0.0) * 100:.1f}% survival)."
            ),
            "suggested_config": {
                "mode": "continuous",
                "model_source": "mixed",
                "morph_ratio": 0.8,
                "n_programs": 100,
            },
        })

    return {
        "summary": {
            "n_tested": n_tested,
            "n_survived": n_survived,
            "n_compressed_tested": n_compressed_tested,
            "n_compressed_survived": n_compressed_survived,
            "compressed_test_share": round(compressed_test_share, 4),
            "compressed_survival_rate": round(compressed_survival_rate, 4),
            "overall_survival_rate": round(overall_survival_rate, 4),
            "dense_survival_rate": round(dense_survival_rate, 4),
        },
        "top_techniques": ranked[:5],
        "recommendations": recommendations,
    }


def _compute_sparse_evidence(nb: LabNotebook) -> Dict[str, Any]:
    """Aggregate sparse execution telemetry for briefing/evidence payloads."""
    try:
        summary_row = nb.conn.execute(
            """
            SELECT
                COUNT(*) AS n_sparse_programs,
                AVG(sparse_density_mean) AS avg_density_mean,
                AVG(sparse_density_last) AS avg_density_last,
                AVG(sparse_nm_compliance) AS avg_nm_compliance,
                SUM(COALESCE(sparse_fallback_calls, 0)) AS total_fallback_calls,
                SUM(COALESCE(sparse_kernel_fallback_calls, 0)) AS total_kernel_fallback_calls,
                AVG(sparse_active_params_estimate) AS avg_active_params_estimate
            FROM program_results
            WHERE sparse_density_mean IS NOT NULL
            """
        ).fetchone()
    except Exception:
        return {
            "n_sparse_programs": 0,
            "top_sparse_ops": [],
        }

    n_sparse_programs = int(summary_row["n_sparse_programs"] or 0)
    if n_sparse_programs <= 0:
        return {
            "n_sparse_programs": 0,
            "top_sparse_ops": [],
        }

    recent_rows = nb.conn.execute(
        """
        SELECT sparse_density_mean
        FROM program_results
        WHERE sparse_density_mean IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 30
        """
    ).fetchall()
    recent_densities = [
        float(r["sparse_density_mean"])
        for r in recent_rows
        if r["sparse_density_mean"] is not None
    ]

    op_aggregates: Dict[str, Dict[str, float]] = {}
    telemetry_rows = nb.conn.execute(
        """
        SELECT sparse_telemetry_json
        FROM program_results
        WHERE sparse_telemetry_json IS NOT NULL
          AND sparse_telemetry_json != ''
        ORDER BY timestamp DESC
        LIMIT 200
        """
    ).fetchall()
    for row in telemetry_rows:
        payload = row["sparse_telemetry_json"]
        if not payload:
            continue
        try:
            entries = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            op_name = str(item.get("op_name") or "").strip()
            if not op_name:
                continue
            stats = op_aggregates.setdefault(op_name, {
                "calls": 0.0,
                "fallback_calls": 0.0,
                "density_sum": 0.0,
            })
            calls = float(item.get("calls") or 0.0)
            stats["calls"] += calls
            stats["fallback_calls"] += float(item.get("fallback_calls") or 0.0)
            stats["density_sum"] += calls * float(item.get("last_density") or 1.0)

    top_sparse_ops = []
    for op_name, stats in op_aggregates.items():
        calls = stats["calls"]
        if calls <= 0:
            continue
        top_sparse_ops.append({
            "op_name": op_name,
            "calls": int(calls),
            "fallback_calls": int(stats["fallback_calls"]),
            "avg_density": max(0.0, min(1.0, stats["density_sum"] / calls)),
        })
    top_sparse_ops.sort(key=lambda item: (item["calls"], -item["fallback_calls"]), reverse=True)

    avg_density_mean = float(summary_row["avg_density_mean"] or 0.0)
    avg_nm_compliance = summary_row["avg_nm_compliance"]
    total_fallback_calls = int(summary_row["total_fallback_calls"] or 0)
    total_kernel_fallback_calls = int(summary_row["total_kernel_fallback_calls"] or 0)
    kernel_fallback_rate = (
        total_kernel_fallback_calls / total_fallback_calls
        if total_fallback_calls > 0
        else 0.0
    )

    return {
        "n_sparse_programs": n_sparse_programs,
        "avg_density_mean": round(avg_density_mean, 4),
        "avg_density_last": round(float(summary_row["avg_density_last"] or avg_density_mean), 4),
        "avg_nm_compliance": round(float(avg_nm_compliance), 4) if avg_nm_compliance is not None else None,
        "total_fallback_calls": total_fallback_calls,
        "total_kernel_fallback_calls": total_kernel_fallback_calls,
        "kernel_fallback_rate": round(kernel_fallback_rate, 4),
        "avg_active_params_estimate": int(float(summary_row["avg_active_params_estimate"] or 0.0)),
        "recent_density": [round(d, 4) for d in recent_densities[:10]],
        "top_sparse_ops": top_sparse_ops[:5],
    }


def _normalize_start_mode(mode: Any) -> str:
    raw = str(mode or "single").strip().lower()
    aliases = {
        "synthesis": "single",
        "evolution": "evolve",
        "novelty_search": "novelty",
        "compact": "compact_synthesis",
        "sparse": "sparse_morph",
        "sparse_morphology": "sparse_morph",
        "sparse_morphological": "sparse_morph",
        "refine": "refine_fingerprint",
        "fingerprint_refine": "refine_fingerprint",
        "refine_recommended": "refine_fingerprint",
    }
    return aliases.get(raw, raw)


def _apply_compact_synthesis_bias(config: RunConfig) -> Dict[str, Any]:
    """Apply conservative compactness defaults and report changed fields."""
    changes: Dict[str, Any] = {}

    def _set_if_diff(field_name: str, new_value: Any) -> None:
        old_value = getattr(config, field_name)
        if old_value == new_value:
            return
        setattr(config, field_name, new_value)
        changes[field_name] = {"from": old_value, "to": new_value}

    _set_if_diff("model_source", "mixed")
    _set_if_diff("morph_ratio", max(float(config.morph_ratio), 0.75))
    _set_if_diff("n_layers", max(1, min(int(config.n_layers), 3)))
    _set_if_diff("model_dim", max(16, min(int(config.model_dim), 192)))
    _set_if_diff("max_depth", max(2, min(int(config.max_depth), 6)))
    _set_if_diff("max_ops", max(4, min(int(config.max_ops), 10)))
    _set_if_diff("residual_prob", max(float(config.residual_prob), 0.8))
    _set_if_diff("math_space_weight", min(float(config.math_space_weight), 1.8))
    _set_if_diff("n_programs", max(1, min(int(config.n_programs), 80)))

    return changes


def _apply_sparse_morph_bias(config: RunConfig) -> Dict[str, Any]:
    """Apply sparse-focused morphological defaults and report changed fields."""
    changes: Dict[str, Any] = {}

    def _set_if_diff(field_name: str, new_value: Any) -> None:
        old_value = getattr(config, field_name)
        if old_value == new_value:
            return
        setattr(config, field_name, new_value)
        changes[field_name] = {"from": old_value, "to": new_value}

    _set_if_diff("model_source", "morphological_box")
    _set_if_diff("morph_focus_sparse", True)
    _set_if_diff("n_layers", max(1, min(int(config.n_layers), 4)))
    _set_if_diff("max_depth", max(2, min(int(config.max_depth), 6)))
    _set_if_diff("max_ops", max(4, min(int(config.max_ops), 10)))
    _set_if_diff("n_programs", max(120, int(config.n_programs)))

    return changes


def _normalize_hypotheses(hypotheses: list) -> list:
    return [_normalize_hypothesis(hypothesis) for hypothesis in hypotheses]


def _knowledge_title_exists(nb: LabNotebook, title: str) -> bool:
    """Return True if an active knowledge entry already has this title."""
    if not title:
        return False
    title_norm = str(title).strip().lower()
    for row in nb.get_knowledge():
        row_title = str(row.get("title") or "").strip().lower()
        if row_title == title_norm:
            return True
    return False


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    """Small dependency-free Pearson correlation for numeric lists."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    den = (den_x * den_y) ** 0.5
    if den <= 1e-12:
        return None
    return num / den


def _backfill_knowledge_from_real_data(nb: LabNotebook) -> Dict[str, Any]:
    """Create missing knowledge categories using measured experiment data."""
    categories = ["anti_pattern", "sweet_spot", "correlation", "tool_insight"]
    existing_by_category: Dict[str, int] = {}
    for category in categories:
        existing_by_category[category] = len(nb.get_knowledge(category=category))

    created: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Shared data pulls
    completed = nb.conn.execute(
        """SELECT experiment_id, experiment_type, n_programs_generated,
                  n_stage1_passed, best_loss_ratio, best_novelty_score, timestamp
           FROM experiments
           WHERE status = 'completed'
           ORDER BY timestamp DESC
           LIMIT 300"""
    ).fetchall()

    # 1) Anti-pattern
    if existing_by_category["anti_pattern"] == 0:
        zero_survivor_runs = [
            r for r in completed
            if (r["n_programs_generated"] or 0) > 0 and (r["n_stage1_passed"] or 0) == 0
        ]
        by_mode: Dict[str, List[Any]] = {}
        for row in zero_survivor_runs:
            mode = str(row["experiment_type"] or "unknown")
            by_mode.setdefault(mode, []).append(row)
        if by_mode:
            worst_mode, rows = max(by_mode.items(), key=lambda it: len(it[1]))
            count = len(rows)
            avg_generated = sum((r["n_programs_generated"] or 0) for r in rows) / max(count, 1)
            title = f"Anti-Pattern: {worst_mode} often yields zero S1 survivors"
            content = (
                f"Observed {count} completed {worst_mode} runs with zero stage-1 survivors "
                f"(average generated programs: {avg_generated:.1f}). "
                "This region currently underperforms and should be deprioritized or re-parameterized."
            )
            evidence = [str(r["experiment_id"]) for r in rows[:5]]
            if not _knowledge_title_exists(nb, title):
                nb.add_knowledge("anti_pattern", title, content, evidence=evidence, confidence=min(0.9, 0.55 + 0.05 * count))
                created.append({"category": "anti_pattern", "title": title, "evidence_count": len(evidence)})
            else:
                skipped.append({"category": "anti_pattern", "reason": "duplicate_title"})
        else:
            skipped.append({"category": "anti_pattern", "reason": "insufficient_zero_survivor_runs"})
    else:
        skipped.append({"category": "anti_pattern", "reason": "already_populated"})

    # 2) Sweet spot
    if existing_by_category["sweet_spot"] == 0:
        candidates = [
            r for r in completed
            if (r["n_programs_generated"] or 0) > 0 and (r["n_stage1_passed"] or 0) > 0 and r["best_loss_ratio"] is not None
        ]
        if candidates:
            def _score(row: Any) -> float:
                gen = float(row["n_programs_generated"] or 1)
                s1_rate = float(row["n_stage1_passed"] or 0) / max(gen, 1.0)
                loss = float(row["best_loss_ratio"] or 1.0)
                return s1_rate - 0.15 * loss

            top = sorted(candidates, key=_score, reverse=True)[:5]
            best = top[0]
            best_rate = (best["n_stage1_passed"] or 0) / max(best["n_programs_generated"] or 1, 1)
            title = f"Sweet Spot: {best['experiment_type']} settings with high S1 yield"
            content = (
                f"Top recent runs show {best_rate * 100:.1f}% S1 pass rate with best loss "
                f"{float(best['best_loss_ratio']):.3f} in {best['experiment_type']} mode. "
                "These conditions represent a productive search region worth repeating."
            )
            evidence = [str(r["experiment_id"]) for r in top]
            if not _knowledge_title_exists(nb, title):
                nb.add_knowledge("sweet_spot", title, content, evidence=evidence, confidence=0.72)
                created.append({"category": "sweet_spot", "title": title, "evidence_count": len(evidence)})
            else:
                skipped.append({"category": "sweet_spot", "reason": "duplicate_title"})
        else:
            skipped.append({"category": "sweet_spot", "reason": "insufficient_successful_runs"})
    else:
        skipped.append({"category": "sweet_spot", "reason": "already_populated"})

    # 3) Correlation
    if existing_by_category["correlation"] == 0:
        xs: List[float] = []
        ys: List[float] = []
        corr_evidence: List[str] = []
        for row in completed:
            gen = row["n_programs_generated"] or 0
            nov = row["best_novelty_score"]
            if gen <= 0 or nov is None:
                continue
            xs.append(float(nov))
            ys.append(float(row["n_stage1_passed"] or 0) / float(gen))
            corr_evidence.append(str(row["experiment_id"]))
        corr = _pearson_corr(xs, ys)
        if corr is not None and len(xs) >= 5:
            relation = "positive" if corr >= 0.15 else "negative" if corr <= -0.15 else "weak"
            title = f"Correlation: novelty vs S1 pass rate is {relation}"
            content = (
                f"Computed Pearson correlation r={corr:.3f} from {len(xs)} completed runs between "
                "best novelty score and S1 pass rate. "
                "Use this relationship to calibrate novelty-vs-fitness trade-offs."
            )
            evidence = corr_evidence[:8]
            if not _knowledge_title_exists(nb, title):
                nb.add_knowledge("correlation", title, content, evidence=evidence, confidence=0.66)
                created.append({"category": "correlation", "title": title, "evidence_count": len(evidence)})
            else:
                skipped.append({"category": "correlation", "reason": "duplicate_title"})
        else:
            skipped.append({"category": "correlation", "reason": "insufficient_variance_or_samples"})
    else:
        skipped.append({"category": "correlation", "reason": "already_populated"})

    # 4) Tool insight
    if existing_by_category["tool_insight"] == 0:
        errors = nb.conn.execute(
            """SELECT error_type, COUNT(*) AS n
               FROM program_results
               WHERE error_type IS NOT NULL AND error_type != ''
               GROUP BY error_type
               ORDER BY n DESC
               LIMIT 1"""
        ).fetchone()
        if errors and errors["error_type"]:
            total_with_error = nb.conn.execute(
                "SELECT COUNT(*) AS n FROM program_results WHERE error_type IS NOT NULL AND error_type != ''"
            ).fetchone()["n"]
            err_type = str(errors["error_type"])
            count = int(errors["n"] or 0)
            share = (count / max(total_with_error, 1)) * 100.0
            title = f"Tool Insight: dominant failure type is {err_type}"
            content = (
                f"{err_type} accounts for {count}/{total_with_error} ({share:.1f}%) of logged program failures. "
                "Prioritizing guardrails and diagnostics around this failure class should improve throughput."
            )
            if not _knowledge_title_exists(nb, title):
                nb.add_knowledge("tool_insight", title, content, evidence=None, confidence=0.69)
                created.append({"category": "tool_insight", "title": title, "evidence_count": 0})
            else:
                skipped.append({"category": "tool_insight", "reason": "duplicate_title"})
        else:
            skipped.append({"category": "tool_insight", "reason": "no_error_telemetry"})
    else:
        skipped.append({"category": "tool_insight", "reason": "already_populated"})

    after_counts = {category: len(nb.get_knowledge(category=category)) for category in categories}
    return {
        "created": created,
        "skipped": skipped,
        "counts_before": existing_by_category,
        "counts_after": after_counts,
    }


def _rank_label(delta: Optional[int], seen_runs: int) -> str:
    if seen_runs <= 1:
        return "new"
    if delta is None:
        return "unknown"
    if delta <= -2:
        return "up"
    if delta >= 2:
        return "down"
    return "stable"


def _parse_bool_query(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_report_date(value: Optional[str], end_of_day: bool = False) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.timestamp()
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _report_program_matches_theme(program: Dict[str, Any], theme: str) -> bool:
    normalized = str(theme or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    graph_json = str(program.get("graph_json") or "").lower()
    arch_spec = str(program.get("arch_spec_json") or "").lower()
    pruning_method = str(program.get("pruning_method") or "").lower()
    if normalized == "sparsity":
        return (
            program.get("sparse_density_mean") is not None
            or "sparse" in graph_json
            or "sparse" in arch_spec
            or bool(pruning_method)
        )
    if normalized == "compression":
        compression_markers = (
            "low_rank", "shared_basis", "tied_proj", "grouped_linear", "bottleneck", "quant", "compressed"
        )
        return any(marker in graph_json or marker in arch_spec for marker in compression_markers)
    if normalized == "routing":
        return (
            program.get("routing_confidence_mean") is not None
            or "routing" in graph_json
            or "moe" in graph_json
            or "gate" in graph_json
        )
    if normalized == "mathspace":
        return (
            bool(program.get("graph_uses_math_spaces"))
            or "mathspace" in graph_json
            or "clifford" in graph_json
            or "hyperbolic" in graph_json
            or "padic" in graph_json
            or "tropical" in graph_json
        )
    if normalized == "failure_modes":
        return (program.get("stage1_passed") or 0) == 0 or bool(program.get("error_type"))
    return True


def _experiment_s1_rate(exp: Dict[str, Any]) -> Optional[float]:
    generated = exp.get("n_programs_generated")
    if generated is None:
        generated = exp.get("n_programs")
    passed = exp.get("n_stage1_passed")
    if passed is None:
        passed = exp.get("s1_passed")
    try:
        gen = float(generated or 0)
        s1 = float(passed or 0)
    except Exception:
        return None
    if gen <= 0:
        return None
    return s1 / gen


def _report_experiment_matches_trend(exp: Dict[str, Any], trend: str) -> bool:
    normalized = str(trend or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    rate = _experiment_s1_rate(exp)
    novelty = exp.get("best_novelty_score")
    if normalized == "high_novelty":
        return isinstance(novelty, (int, float)) and float(novelty) >= 0.5
    if rate is None:
        return False
    if normalized in {"improving", "high_survival"}:
        return rate >= 0.08
    if normalized == "declining":
        return rate < 0.03
    if normalized == "plateaued":
        return 0.03 <= rate < 0.08
    return True


def _build_filtered_report_summary(
    base_summary: Dict[str, Any],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not experiments:
        return dict(base_summary or {})
    total_programs = 0
    total_survivors = 0
    for exp in experiments:
        total_programs += int(exp.get("n_programs_generated") or 0)
        total_survivors += int(exp.get("n_stage1_passed") or 0)
    out = dict(base_summary or {})
    out["total_experiments"] = len(experiments)
    out["total_programs_evaluated"] = total_programs
    out["stage1_survivors"] = total_survivors
    return out


def _build_report_snapshot_key(scope: str, query_payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        {"scope": scope, "query": query_payload or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _infer_tier_for_program(nb: LabNotebook, program: dict) -> str:
    """Infer tier for a raw program_results row by checking the leaderboard."""
    result_id = program.get("result_id")
    if not result_id:
        return "screening"
    row = nb.conn.execute(
        "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
    ).fetchone()
    return row["tier"] if row else "screening"


def _count_discovery_tiers(nb: LabNotebook) -> dict:
    """Count unique fingerprints per tier + total S1 survivors."""
    rows = nb.conn.execute(
        "SELECT tier, COUNT(*) AS cnt FROM leaderboard GROUP BY tier"
    ).fetchall()
    counts = {r["tier"]: r["cnt"] for r in rows}
    total_s1 = nb.conn.execute(
        "SELECT COUNT(*) AS cnt FROM program_results WHERE stage1_passed = 1"
    ).fetchone()
    counts["total_survivors"] = total_s1["cnt"] if total_s1 else 0
    return counts


def _compute_cross_run_stability(nb: LabNotebook, top_programs: list) -> dict:
    """Compute rank movement for top candidates across recent experiments.

    Uses graph fingerprint as the architecture key and tracks its rank
    among stage-1-passing programs for each completed experiment.
    """
    experiments = [
        exp for exp in nb.get_recent_experiments(40)
        if exp.get("status") == "completed"
    ]
    if not top_programs or not experiments:
        return {
            "summary": {"stable": 0, "up": 0, "down": 0, "new": 0},
            "candidates": [],
            "window_size": len(experiments),
        }

    fingerprint_ranks_by_experiment: dict[str, dict[str, int]] = {}
    for exp in experiments:
        experiment_id = exp.get("experiment_id")
        if not experiment_id:
            continue
        programs = nb.get_program_results(experiment_id)
        ranked = sorted(
            [
                p for p in programs
                if p.get("stage1_passed") and p.get("loss_ratio") is not None
            ],
            key=lambda p: p.get("loss_ratio", float("inf")),
        )
        ranks = {}
        for idx, program in enumerate(ranked, start=1):
            fp = program.get("graph_fingerprint")
            if fp and fp not in ranks:
                ranks[fp] = idx
        fingerprint_ranks_by_experiment[experiment_id] = ranks

    candidates = []
    summary = {"stable": 0, "up": 0, "down": 0, "new": 0}
    for index, program in enumerate(top_programs[:20], start=1):
        fp = program.get("graph_fingerprint")
        if not fp:
            continue

        history = []
        for exp in experiments:
            experiment_id = exp.get("experiment_id")
            if not experiment_id:
                continue
            rank = fingerprint_ranks_by_experiment.get(experiment_id, {}).get(fp)
            if rank is None:
                continue
            history.append({
                "experiment_id": experiment_id,
                "timestamp": exp.get("timestamp"),
                "rank": rank,
            })

        seen_runs = len(history)
        latest_rank = history[0]["rank"] if history else None
        previous_rank = history[1]["rank"] if len(history) > 1 else None
        delta = None
        if latest_rank is not None and previous_rank is not None:
            delta = latest_rank - previous_rank
        trend = _rank_label(delta, seen_runs)
        summary[trend] = summary.get(trend, 0) + 1

        candidates.append({
            "result_id": program.get("result_id"),
            "graph_fingerprint": fp,
            "current_overall_rank": index,
            "seen_runs": seen_runs,
            "latest_rank": latest_rank,
            "previous_rank": previous_rank,
            "rank_delta": delta,
            "trend": trend,
        })

    return {
        "summary": summary,
        "candidates": candidates,
        "window_size": len(experiments),
    }


def _get_sse_timeout_seconds() -> float:
    """Get SSE stream polling timeout from env with safe fallback."""
    raw = os.environ.get("ARIA_SSE_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    if timeout <= 0:
        logger.warning("Non-positive ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    return timeout


def _get_runner(notebook_path: str) -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(notebook_path)
    return _runner


def _compute_recommendation(program: dict, leaderboard_entry: Optional[dict]) -> dict:
    """Deterministic next-action recommendation based on tier and pass/fail."""
    tier = (leaderboard_entry or {}).get("tier", "screening")
    s1 = program.get("stage1_passed", False)

    if not s1:
        return {
            "action": "archive",
            "rationale": "Program did not pass Stage 1 learning evaluation.",
            "confidence": "high",
        }

    if tier == "breakthrough":
        return {
            "action": "publish",
            "rationale": "Breakthrough-tier architecture with validated performance.",
            "confidence": "high",
        }

    if tier == "validation":
        passed = (leaderboard_entry or {}).get("validation_passed", False)
        if passed:
            return {
                "action": "scale up or publish",
                "rationale": "Validation passed with multi-seed stability confirmed.",
                "confidence": "high",
                "bias_check": "grammar_independence_verified",
            }
        return {
            "action": "re-validate",
            "rationale": "Validation tier but not yet passed; may need more seeds or longer training.",
            "confidence": "medium",
        }

    if tier == "investigation":
        passed = (leaderboard_entry or {}).get("investigation_passed", False)
        if passed:
            return {
                "action": "validate",
                "rationale": "Investigation passed; promote to validation for multi-seed confirmation.",
                "confidence": "high",
            }
        return {
            "action": "re-investigate or archive",
            "rationale": "Investigation tier but not yet passed; re-run or archive if stale.",
            "confidence": "medium",
        }

    # screening (default)
    return {
        "action": "investigate",
        "rationale": "Screening-tier candidate; needs deeper investigation to confirm potential.",
        "confidence": "medium",
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _promotion_evidence_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    seen_runs = int(((entry.get("cross_run_stability") or {}).get("seen_runs") or 0))
    baseline_ratio = _safe_float(entry.get("validation_baseline_ratio"))
    std = _safe_float(entry.get("validation_multi_seed_std"))

    checks = {
        "baselineEvidence": baseline_ratio is not None,
        "baselineBeat": baseline_ratio is not None and baseline_ratio < 1.0,
        "multiSeedStd": std is not None,
        "boundedStd": std is not None and std <= 0.12,
        "ckaArtifactBacked": entry.get("cka_source") == "artifact",
        "repeatObserved": seen_runs >= 3,
    }
    evidence_count = sum(1 for ok in checks.values() if ok)
    total_checks = len(checks)
    completeness = evidence_count / total_checks if total_checks else 0.0

    std_signal = 0.0
    if std is not None:
        if std <= 0.05:
            std_signal = 1.0
        elif std <= 0.12:
            std_signal = 0.65
        elif std <= 0.2:
            std_signal = 0.35
        else:
            std_signal = 0.1

    if seen_runs >= 5:
        repeat_signal = 1.0
    elif seen_runs >= 3:
        repeat_signal = 0.65
    elif seen_runs >= 2:
        repeat_signal = 0.4
    elif seen_runs >= 1:
        repeat_signal = 0.2
    else:
        repeat_signal = 0.0

    margin_signal = 0.0
    if baseline_ratio is not None:
        margin = 1.0 - baseline_ratio
        if margin >= 0.1:
            margin_signal = 1.0
        elif margin > 0:
            margin_signal = 0.7
        else:
            margin_signal = 0.15

    score = round((completeness * 0.5 + std_signal * 0.2 + repeat_signal * 0.2 + margin_signal * 0.1) * 100)
    missing = [name for name, ok in checks.items() if not ok]

    return {
        "score": score,
        "seen_runs": seen_runs,
        "std": std,
        "evidence_count": evidence_count,
        "total_checks": total_checks,
        "missing": missing,
    }


def _decision_gate_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "screeningEvidence": entry.get("screening_loss_ratio") is not None and entry.get("screening_novelty") is not None,
        "investigationEvidence": entry.get("investigation_loss_ratio") is not None and entry.get("investigation_robustness") is not None,
        "robustnessFloor": entry.get("investigation_robustness") is not None and float(entry.get("investigation_robustness") or 0.0) >= 0.5,
        "validationEvidence": (
            entry.get("validation_loss_ratio") is not None
            and entry.get("validation_baseline_ratio") is not None
            and entry.get("validation_multi_seed_std") is not None
        ),
        "baselineBeatsReference": entry.get("validation_baseline_ratio") is not None and float(entry.get("validation_baseline_ratio") or 0.0) < 1.0,
        "consistencyBounded": entry.get("validation_multi_seed_std") is not None and float(entry.get("validation_multi_seed_std") or 1.0) <= 0.12,
    }
    decision_ready = all(checks.values())
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "decision_ready": decision_ready,
        "missing": missing,
    }


def _build_scale_up_templates_for_result(result_id: Optional[str]) -> List[Dict[str, Any]]:
    normalized = str(result_id or "").strip()
    if not normalized:
        return []

    return [
        {
            "template_id": "multi_seed_stress",
            "title": "Multi-seed stress validation",
            "description": "Run deeper multi-seed validation to confirm consistency and variance bounds.",
            "start_payload": {
                "mode": "validation",
                "result_ids": [normalized],
                "validation_steps": 12000,
                "validation_n_seeds": 7,
                "validation_batch_size": 8,
                "validation_seq_len": 512,
            },
        },
        {
            "template_id": "robustness_recheck",
            "title": "Robustness re-check",
            "description": "Re-run investigation-level robustness checks before heavier scale-up spend.",
            "start_payload": {
                "mode": "investigation",
                "result_ids": [normalized],
                "investigation_steps": 3500,
                "investigation_batch_size": 4,
                "n_training_programs": 4,
            },
        },
        {
            "template_id": "efficiency_scale_up",
            "title": "Scale-up + efficiency profile",
            "description": "Run scale-up training with one-shot pruning baseline to profile efficiency/quality trade-offs.",
            "start_payload": {
                "mode": "scale_up",
                "result_ids": [normalized],
                "scale_up_steps": 8000,
                "scale_up_batch_size": 8,
                "scale_up_seq_len": 512,
                "one_shot_pruning_baseline": True,
                "one_shot_pruning_method": "wanda",
                "one_shot_pruning_sparsity": 0.5,
            },
        },
    ]


def _build_reproducibility_workflow(
    repro_packet: Dict[str, Any],
    scale_up_templates: List[Dict[str, Any]],
    result_id: Optional[str] = None,
    graph_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    packet = repro_packet or {}
    ready_count = int(packet.get("ready_count") or 0)
    total_checks = int(packet.get("total_checks") or 6)
    missing = {str(item) for item in (packet.get("missing") or []) if item}

    template_by_id = {
        str(template.get("template_id")): template
        for template in (scale_up_templates or [])
        if isinstance(template, dict) and template.get("template_id")
    }

    def _payload(template_id: str) -> Optional[Dict[str, Any]]:
        template = template_by_id.get(template_id)
        if not template:
            return None
        payload = template.get("start_payload")
        return dict(payload) if isinstance(payload, dict) else None

    checks = [
        ("result_id", "Result identifier captured", None, "Program result must be persisted before repro closure."),
        ("graph_fingerprint", "Fingerprint captured", None, "Graph fingerprint is required for cross-run traceability."),
        ("arch_spec", "Architecture spec recorded", "robustness_recheck", "Re-run robustness check if architecture metadata is incomplete."),
        ("baseline_ratio", "Baseline ratio measured", "multi_seed_stress", "Run multi-seed validation to compute baseline ratio."),
        ("multi_seed_std", "Multi-seed variance measured", "multi_seed_stress", "Run multi-seed validation to measure stability variance."),
        ("cka_artifact", "Artifact-backed CKA recorded", "efficiency_scale_up", "After run completion, stamp CKA artifact integrity in artifact references."),
    ]

    steps: List[Dict[str, Any]] = []
    for check_id, label, template_id, guidance in checks:
        is_complete = check_id not in missing
        step: Dict[str, Any] = {
            "check_id": check_id,
            "label": label,
            "status": "complete" if is_complete else "missing",
            "guidance": guidance,
        }
        if result_id:
            step["result_id"] = result_id
        if graph_fingerprint:
            step["graph_fingerprint"] = graph_fingerprint
        if not is_complete and template_id:
            payload = _payload(template_id)
            if payload:
                step["action_label"] = "Run template"
                step["start_payload"] = payload
        steps.append(step)

    next_actions = [
        {
            "check_id": step.get("check_id"),
            "label": step.get("label"),
            "action_label": step.get("action_label"),
            "start_payload": step.get("start_payload"),
            "guidance": step.get("guidance"),
        }
        for step in steps
        if step.get("status") == "missing"
    ][:3]

    return {
        "status": "ready" if ready_count >= total_checks else "in_progress",
        "ready_count": ready_count,
        "total_checks": total_checks,
        "progress_label": f"{ready_count}/{total_checks}",
        "remaining": max(0, total_checks - ready_count),
        "steps": steps,
        "next_actions": next_actions,
        "result_id": result_id,
        "graph_fingerprint": graph_fingerprint,
    }


def _compute_breakthrough_production_readiness(nb: LabNotebook, analytics: Any) -> Dict[str, Any]:
    leaderboard_entries = nb.get_leaderboard(tier="breakthrough", limit=20, sort_by="composite_score")
    if not leaderboard_entries:
        return {
            "breakthrough_count": 0,
            "decision_ready_count": 0,
            "high_confidence_count": 0,
            "full_repro_packet_count": 0,
            "artifact_cka_count": 0,
            "epic_switch_recommendation": {
                "action": "stay_current_epic",
                "reason": "No breakthrough-tier candidates are available yet.",
            },
            "top_candidates": [],
            "scale_up_templates": [],
            "reproducibility_workflow": None,
        }

    stability = _compute_cross_run_stability(nb, nb.get_top_programs(20, sort_by="loss_ratio"))
    stability_by_result = {
        c.get("result_id"): c
        for c in stability.get("candidates", [])
        if c.get("result_id")
    }

    evaluated: List[Dict[str, Any]] = []
    for entry in leaderboard_entries:
        row = dict(entry)
        row["cross_run_stability"] = stability_by_result.get(
            row.get("result_id"),
            {
                "trend": "unknown",
                "seen_runs": 0,
                "latest_rank": None,
                "previous_rank": None,
                "rank_delta": None,
            },
        )
        row["reproducibility_packet"] = analytics.reproducibility_packet_status(row)
        promotion = _promotion_evidence_for_entry(row)
        gate = _decision_gate_for_entry(row)
        scale_up_templates = _build_scale_up_templates_for_result(row.get("result_id"))
        reproducibility_workflow = _build_reproducibility_workflow(
            row["reproducibility_packet"],
            scale_up_templates,
            result_id=row.get("result_id"),
            graph_fingerprint=row.get("graph_fingerprint"),
        )
        evaluated.append({
            "result_id": row.get("result_id"),
            "architecture_family": row.get("architecture_family"),
            "composite_score": row.get("composite_score"),
            "promotion_confidence_score": promotion["score"],
            "seen_runs": promotion["seen_runs"],
            "decision_ready": gate["decision_ready"],
            "decision_missing": gate["missing"],
            "repro_packet": row["reproducibility_packet"],
            "cka_source": row.get("cka_source"),
            "scale_up_templates": scale_up_templates,
            "reproducibility_workflow": reproducibility_workflow,
        })

    breakthrough_count = len(evaluated)
    decision_ready_count = sum(1 for row in evaluated if row.get("decision_ready"))
    high_confidence_count = sum(1 for row in evaluated if int(row.get("promotion_confidence_score") or 0) >= 75)
    full_repro_packet_count = sum(1 for row in evaluated if (row.get("repro_packet") or {}).get("status") == "ready")
    artifact_cka_count = sum(1 for row in evaluated if row.get("cka_source") == "artifact")

    switch_ready = any(
        row.get("decision_ready")
        and int(row.get("promotion_confidence_score") or 0) >= 75
        and (row.get("repro_packet") or {}).get("status") == "ready"
        and row.get("cka_source") == "artifact"
        for row in evaluated
    )

    if switch_ready:
        recommendation = {
            "action": "switch_to_scale_up_epic",
            "reason": "At least one breakthrough candidate meets decision, confidence, repro, and artifact-backed CKA gates.",
        }
    else:
        recommendation = {
            "action": "stay_current_epic",
            "reason": "Breakthrough evidence is still incomplete; continue hardening reproducibility and validation before switching epics.",
        }

    top_candidates = sorted(
        evaluated,
        key=lambda row: (
            int(bool(row.get("decision_ready"))),
            int(row.get("promotion_confidence_score") or 0),
            float(row.get("composite_score") or 0.0),
        ),
        reverse=True,
    )[:3]
    scale_up_templates = top_candidates[0].get("scale_up_templates", []) if top_candidates else []

    return {
        "breakthrough_count": breakthrough_count,
        "decision_ready_count": decision_ready_count,
        "high_confidence_count": high_confidence_count,
        "full_repro_packet_count": full_repro_packet_count,
        "artifact_cka_count": artifact_cka_count,
        "epic_switch_recommendation": recommendation,
        "top_candidates": top_candidates,
        "scale_up_templates": scale_up_templates,
        "reproducibility_workflow": (
            top_candidates[0].get("reproducibility_workflow")
            if top_candidates
            else None
        ),
    }


def _annotate_qkv_usage(programs: list, analytics) -> None:
    for program in programs:
        if not isinstance(program, dict):
            continue
        qkv_usage = analytics.qkv_usage_enum(program)
        program["qkv_usage"] = qkv_usage
        program["uses_qkv"] = qkv_usage != "qkv_free"
        program["compression_metrics"] = analytics.canonical_compression_metrics(program)
        program["reproducibility_packet"] = analytics.reproducibility_packet_status(program)


def _normalize_result_ids(raw_ids: Any) -> List[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for value in raw_ids:
        if value is None:
            continue
        result_id = str(value).strip()
        if not result_id or result_id in seen:
            continue
        seen.add(result_id)
        normalized.append(result_id)
    return normalized


def _resolve_scale_up_result_ids(
    nb: LabNotebook,
    result_ids: List[str],
    graph_fingerprints: List[str],
) -> Dict[str, Any]:
    """Resolve explicit result IDs and/or fingerprint prefixes for scale-up."""
    merged_result_ids: List[str] = []
    seen: set[str] = set()
    for result_id in result_ids:
        if result_id in seen:
            continue
        seen.add(result_id)
        merged_result_ids.append(result_id)

    resolved: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for fingerprint in graph_fingerprints:
        rows = nb.conn.execute(
            """
            SELECT result_id, graph_fingerprint, experiment_id, stage1_passed,
                   loss_ratio, timestamp
            FROM program_results
            WHERE graph_fingerprint LIKE ?
            ORDER BY stage1_passed DESC,
                     (loss_ratio IS NULL) ASC,
                     loss_ratio ASC,
                     timestamp DESC
            LIMIT 5
            """,
            (f"{fingerprint}%",),
        ).fetchall()

        if not rows:
            unresolved.append(fingerprint)
            continue

        chosen = dict(rows[0])
        chosen_result_id = str(chosen.get("result_id") or "")
        if chosen_result_id and chosen_result_id not in seen:
            seen.add(chosen_result_id)
            merged_result_ids.append(chosen_result_id)

        candidates = [
            {
                "result_id": row["result_id"],
                "graph_fingerprint": row["graph_fingerprint"],
                "experiment_id": row["experiment_id"],
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": row["loss_ratio"],
            }
            for row in rows
        ]
        resolved.append({
            "requested_fingerprint": fingerprint,
            "selected_result_id": chosen.get("result_id"),
            "selected_graph_fingerprint": chosen.get("graph_fingerprint"),
            "selected_experiment_id": chosen.get("experiment_id"),
            "candidate_count": len(rows),
            "candidates": candidates,
        })

    return {
        "result_ids": merged_result_ids,
        "resolved_fingerprints": resolved,
        "unresolved_fingerprints": unresolved,
    }


def _build_start_mode_eligibility(
    nb: LabNotebook,
    mode: str,
    result_ids: List[str],
) -> Dict[str, Any]:
    """Validate candidate progression eligibility for start modes.

    Returns a structured payload containing per-candidate reasons.
    """
    payload: Dict[str, Any] = {
        "mode": mode,
        "requested_result_ids": list(result_ids),
        "eligible_result_ids": [],
        "ineligible": [],
        "all_eligible": False,
    }
    if not result_ids:
        return payload

    placeholders = ",".join("?" for _ in result_ids)
    leaderboard_rows = nb.conn.execute(
        f"""
        SELECT result_id, tier, investigation_passed, validation_passed,
               investigation_loss_ratio, validation_loss_ratio
        FROM leaderboard
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()
    program_rows = nb.conn.execute(
        f"""
        SELECT result_id, stage1_passed
        FROM program_results
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()

    leaderboard_by_id = {row["result_id"]: dict(row) for row in leaderboard_rows}
    program_by_id = {row["result_id"]: dict(row) for row in program_rows}

    for result_id in result_ids:
        lb = leaderboard_by_id.get(result_id)
        program = program_by_id.get(result_id)

        if lb is None:
            if program is None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "result_not_found",
                    "detail": "Result ID was not found in program results.",
                })
            elif not bool(program.get("stage1_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_stage1_survivor",
                    "detail": "Result exists but is not a Stage-1 survivor.",
                })
            else:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_in_leaderboard",
                    "detail": "Result exists but has no leaderboard progression record.",
                })
            continue

        tier = str(lb.get("tier") or "").lower()

        if mode == "investigation":
            if tier != "screening":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_screening_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; only screening tier can be investigated.",
                    "tier": tier or None,
                })
                continue
            if lb.get("investigation_loss_ratio") is not None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "already_investigated_unchanged",
                    "detail": "Candidate already has investigation evidence; provide a changed-condition trigger before re-investigating.",
                    "tier": tier,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        if mode == "validation":
            if tier != "investigation":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; validation requires investigation tier.",
                    "tier": tier or None,
                })
                continue
            if not bool(lb.get("investigation_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_passed",
                    "detail": "Investigation evidence did not pass robustness gate.",
                    "tier": tier,
                })
                continue
            if bool(lb.get("validation_passed")) or lb.get("validation_loss_ratio") is not None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "already_validated",
                    "detail": "Candidate already has validation evidence.",
                    "tier": tier,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        payload["ineligible"].append({
            "result_id": result_id,
            "reason": "unsupported_mode",
            "detail": f"Eligibility checks are not implemented for mode '{mode}'.",
        })

    payload["all_eligible"] = len(payload["ineligible"]) == 0 and len(payload["eligible_result_ids"]) > 0
    payload["summary"] = {
        "requested": len(result_ids),
        "eligible": len(payload["eligible_result_ids"]),
        "ineligible": len(payload["ineligible"]),
    }
    return payload


def _build_report_action_eligibility(
    nb: LabNotebook,
    result_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build per-result report action eligibility aligned with start guardrails."""
    normalized_ids = _normalize_result_ids(result_ids)
    if not normalized_ids:
        return {}

    inv = _build_start_mode_eligibility(nb, "investigation", normalized_ids)
    val = _build_start_mode_eligibility(nb, "validation", normalized_ids)

    inv_eligible = set(inv.get("eligible_result_ids") or [])
    val_eligible = set(val.get("eligible_result_ids") or [])
    inv_reason = {
        row.get("result_id"): row.get("reason")
        for row in (inv.get("ineligible") or [])
        if row.get("result_id")
    }
    val_reason = {
        row.get("result_id"): row.get("reason")
        for row in (val.get("ineligible") or [])
        if row.get("result_id")
    }

    eligibility_by_id: Dict[str, Dict[str, Any]] = {}
    for result_id in normalized_ids:
        investigation_eligible = result_id in inv_eligible
        validation_eligible = result_id in val_eligible
        queue_eligible = investigation_eligible or validation_eligible
        queue_reason = None
        if not queue_eligible:
            queue_reason = inv_reason.get(result_id) or val_reason.get(result_id) or "not_progression_eligible"

        eligibility_by_id[result_id] = {
            "investigationEligible": investigation_eligible,
            "validationEligible": validation_eligible,
            "queueEligible": queue_eligible,
            "queueReason": queue_reason,
            "investigationReason": inv_reason.get(result_id),
            "validationReason": val_reason.get(result_id),
        }

    return eligibility_by_id


def _llm_config_path(notebook_path: str) -> Path:
    """Path for persisted LLM configuration, next to the notebook DB."""
    return Path(notebook_path).parent / "llm_config.json"


def _load_persisted_llm_config(notebook_path: str):
    """Auto-load LLM config from disk if present."""
    config_path = _llm_config_path(notebook_path)
    if not config_path.exists():
        return
    try:
        import json as _json
        data = _json.loads(config_path.read_text())
        backend = str(data.get("backend", "")).strip()
        if not backend:
            return
        aria = get_aria()
        aria.configure_llm(
            backend_name=backend,
            api_key=str(data.get("api_key", "")).strip(),
            model=str(data.get("model", "")).strip(),
            host=str(data.get("host", "")).strip(),
        )
        logger.info(f"Loaded persisted LLM config: {backend}")
    except Exception as e:
        logger.warning(f"Failed to load persisted LLM config: {e}")


def _save_llm_config(notebook_path: str, config: Dict):
    """Persist LLM config to disk so it survives restarts."""
    config_path = _llm_config_path(notebook_path)
    try:
        import json as _json
        config_path.write_text(_json.dumps(config, indent=2))
        logger.info(f"Saved LLM config to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save LLM config: {e}")


_DISMISSED_ACTIONS: set = set()

# Singleton autonomy engine (created lazily on first API call)
_aria_autonomy = None
_aria_action_store = None


def _get_autonomy(notebook_path: str):
    """Get or create the singleton AriaAutonomy instance."""
    global _aria_autonomy, _aria_action_store
    if _aria_autonomy is None:
        from .autonomy import AriaAutonomy
        from .actions import ActionStore
        nb = LabNotebook(notebook_path)
        _aria_autonomy = AriaAutonomy(notebook=nb)
        _aria_action_store = ActionStore(nb.conn)
    return _aria_autonomy, _aria_action_store


def _compute_action_queue(nb, analytics=None) -> List[Dict[str, Any]]:
    """Aggregate prioritized actions from existing data sources."""
    actions: List[Dict[str, Any]] = []

    # 1. Breakthrough candidates from leaderboard
    try:
        breakthroughs = nb.get_leaderboard(tier="breakthrough", limit=5, sort_by="composite_score")
        for entry in breakthroughs:
            rid = entry.get("result_id", "")
            actions.append({
                "id": f"breakthrough_{rid[:12]}",
                "type": "breakthrough",
                "priority": 1,
                "icon": "trophy",
                "title": f"Architecture {rid[:8]} — Breakthrough",
                "summary": f"Composite score {entry.get('composite_score', 0):.3f}. Tier: breakthrough.",
                "detail": {
                    "result_id": rid,
                    "composite_score": entry.get("composite_score"),
                    "screening_loss_ratio": entry.get("screening_loss_ratio"),
                    "tier": "breakthrough",
                },
                "actions": [
                    {"label": "View Details", "action": "navigate", "payload": {"tab": "discoveries", "result_id": rid}},
                ],
                "dismissable": True,
                "source": "leaderboard",
            })
    except Exception:
        pass

    # 2. Stalled run warning — last 3+ completed experiments with 0 S1 survivors
    try:
        recent = nb.get_recent_experiments(5)
        completed = [e for e in recent if e.get("status") == "completed"]
        if len(completed) >= 3 and all(
            (e.get("n_stage1_passed") or 0) == 0 for e in completed[:3]
        ):
            actions.append({
                "id": "warning_stalled_runs",
                "type": "warning",
                "priority": 2,
                "icon": "warning",
                "title": "Pipeline stalled — zero S1 survivors",
                "summary": f"Last {len(completed[:3])} completed runs produced no Stage 1 survivors.",
                "detail": {
                    "recent_experiments": [
                        {"id": e.get("experiment_id", "")[:12], "s1": e.get("n_stage1_passed", 0)}
                        for e in completed[:3]
                    ],
                },
                "actions": [
                    {"label": "Run Novelty Search", "action": "start", "payload": {"mode": "novelty"}},
                ],
                "dismissable": True,
                "source": "experiments",
            })
    except Exception:
        pass

    # 3. Healer fixes from recent tasks
    try:
        healer_tasks = nb.get_recent_healer_tasks(limit=5)
        active = [t for t in healer_tasks if t.get("state") not in ("completed", "failed")]
        for task in active[:2]:
            tid = task.get("task_id", "")
            actions.append({
                "id": f"healer_{tid[:12]}",
                "type": "healer",
                "priority": 4,
                "icon": "wrench",
                "title": f"Code healer: {task.get('trigger_type', 'repair')}",
                "summary": f"Task {tid[:12]} — {task.get('state', 'active')}. {task.get('scope', '')[:80]}",
                "detail": {
                    "task_id": tid,
                    "state": task.get("state"),
                    "trigger_type": task.get("trigger_type"),
                    "experiment_id": task.get("experiment_id"),
                },
                "actions": [],
                "dismissable": True,
                "source": "healer",
            })
    except Exception:
        pass

    # 4. Diagnosis issues from analytics
    try:
        if analytics:
            analytics_data = analytics.get_analytics_data() if hasattr(analytics, "get_analytics_data") else {}
        else:
            from .analytics import ExperimentAnalytics
            analytics_obj = ExperimentAnalytics(nb)
            analytics_data = analytics_obj.get_analytics_data() if hasattr(analytics_obj, "get_analytics_data") else {}
        issues = _diagnose_research_issues(analytics_data, nb)
        for i, issue in enumerate(issues[:3]):
            actions.append({
                "id": f"diagnosis_{i}",
                "type": "diagnosis",
                "priority": 3,
                "icon": "stethoscope",
                "title": "Diagnosis",
                "summary": issue.get("issue", "Unknown issue"),
                "detail": {"config_fix": issue.get("config_fix")},
                "actions": (
                    [{"label": "Apply Fix", "action": "config_fix", "payload": issue.get("config_fix", {})}]
                    if issue.get("action_type") in ("config_fix", "grammar_fix")
                    else []
                ),
                "dismissable": True,
                "source": "diagnostics",
            })
    except Exception:
        pass

    # 5. Strategy suggestion (lightweight — just presence indicator)
    try:
        summary = nb.get_dashboard_summary()
        total_exp = summary.get("total_experiments", 0)
        if total_exp == 0:
            actions.append({
                "id": "strategy_first_run",
                "type": "strategy",
                "priority": 5,
                "icon": "lightbulb",
                "title": "Get started",
                "summary": "No experiments yet. Start your first continuous run to begin exploring architectures.",
                "detail": {},
                "actions": [
                    {"label": "Start Continuous", "action": "start", "payload": {"mode": "continuous"}},
                ],
                "dismissable": False,
                "source": "strategy",
            })
    except Exception:
        pass

    # Filter out dismissed actions
    actions = [a for a in actions if a["id"] not in _DISMISSED_ACTIONS]

    # Sort by priority
    actions.sort(key=lambda a: a.get("priority", 10))

    return actions[:8]


def create_app(
    notebook_path: str = "research/lab_notebook.db",
    static_folder: Optional[str] = None,
) -> Flask:
    """Create the Flask API app."""

    if static_folder is None:
        static_folder = str(Path(__file__).parent.parent / "dashboard" / "build")

    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    CORS(app)

    def _dashboard_index_path() -> Optional[Path]:
        if not app.static_folder:
            return None
        candidate = Path(app.static_folder) / "index.html"
        return candidate if candidate.is_file() else None

    def _dashboard_missing_response():
        expected = str((Path(__file__).parent.parent / "dashboard" / "build" / "index.html"))
        body = (
            "<html><body><h2>Dashboard frontend build is missing.</h2>"
            f"<p>Expected index file at: {expected}</p>"
            "<p>Build dashboard assets (dashboard/build) and retry.</p>"
            "</body></html>"
        )
        return body, 503, {"Content-Type": "text/html; charset=utf-8"}

    def _is_asset_path(path: str) -> bool:
        name = Path(path or "").name
        return "." in name

    # Auto-load persisted LLM config
    _load_persisted_llm_config(notebook_path)

    # ── Global error handlers ──

    @app.errorhandler(404)
    def not_found(e):
        # Only return JSON for API routes; let static files 404 naturally
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        index_path = _dashboard_index_path()
        if index_path and not _is_asset_path(request.path):
            return send_from_directory(app.static_folder, "index.html")
        if _is_asset_path(request.path):
            return "Not found", 404
        return _dashboard_missing_response()

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"500 error on {request.method} {request.path}: {e}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.error(f"Unhandled exception on {request.method} {request.path}: "
                     f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    @app.after_request
    def log_response(response):
        if request.path.startswith("/api/") and response.status_code >= 400:
            logger.warning(f"{request.method} {request.path} -> {response.status_code}")
        return response

    # ── Dashboard routes ──

    @app.route("/")
    def index():
        if not _dashboard_index_path():
            return _dashboard_missing_response()
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/favicon.ico")
    def favicon():
        if app.static_folder:
            icon = Path(app.static_folder) / "favicon.ico"
            if icon.is_file():
                return send_from_directory(app.static_folder, "favicon.ico")
        return "", 204

    @app.route("/<path:path>")
    def static_files(path):
        if app.static_folder:
            static_path = Path(app.static_folder) / path
            if static_path.is_file():
                return send_from_directory(app.static_folder, path)
        index_path = _dashboard_index_path()
        if index_path and not _is_asset_path(path):
            return send_from_directory(app.static_folder, "index.html")
        return "Not found", 404

    # ── Read-only API routes ──

    @app.route("/api/status")
    def api_status():
        """Get Aria's current status and dashboard summary."""
        nb = LabNotebook(notebook_path)
        runner = _get_runner(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()
            progress_payload = runner.progress.to_dict()
            trigger = _get_run_trigger_snapshot(progress_payload.get("experiment_id"))
            progress_payload["run_trigger_source"] = trigger.get("source")
            progress_payload["run_trigger"] = trigger
            return jsonify({
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "is_running": runner.is_running,
                "progress": progress_payload,
                "run_trigger_source": trigger.get("source"),
                "run_trigger": trigger,
            })
        except Exception as e:
            logger.error(f"Error in /api/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments")
    def api_experiments():
        """List recent experiments."""
        n = request.args.get("n", 20, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_recent_experiments(n))
        except Exception as e:
            logger.error(f"Error in /api/experiments: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>")
    def api_experiment_detail(experiment_id):
        """Get experiment details with entries and per-experiment programs."""
        nb = LabNotebook(notebook_path)
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404
            entries = nb.get_entries(experiment_id=experiment_id)
            programs = nb.get_program_results(experiment_id)
            prereg = nb.get_preregistration_for_experiment(experiment_id)
            deviations = nb.get_preregistration_deviations(experiment_id)
            payload = {
                "experiment": exp,
                "entries": entries,
                "programs": programs,
                "preregistration": prereg,
                "preregistration_deviations": deviations,
            }
            return jsonify(_json_safe(payload))
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/programs")
    def api_experiment_programs(experiment_id):
        """All programs for an experiment (not just S1 survivors)."""
        nb = LabNotebook(notebook_path)
        try:
            programs = nb.get_program_results(experiment_id)
            return jsonify(_json_safe(programs))
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>")
    def api_program_detail(result_id):
        """Full program detail with parsed graph JSON + fingerprint + all metrics."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            # Include training curve availability flag
            try:
                curve = nb.get_training_curve(result_id)
                program["has_training_curve"] = len(curve) > 0
            except Exception:
                program["has_training_curve"] = False

            # Try LLM explanation of fingerprint (non-critical)
            try:
                ctx = build_program_context(program)
                explanation = aria.explain_fingerprint(ctx)
                if explanation:
                    program["llm_explanation"] = explanation
            except Exception as e:
                logger.debug(f"LLM fingerprint explanation failed for {result_id}: {e}")

            program = _enrich_program_detail(nb, program)

            try:
                program["lineage_chain"] = _program_lineage_chain(nb, result_id)
            except Exception:
                program["lineage_chain"] = []

            return jsonify(_json_safe(program))
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>/lineage")
    def api_program_lineage(result_id: str):
        """Program lineage chain for refinement traceability."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404
            chain = _program_lineage_chain(nb, result_id)
            return jsonify(_json_safe({
                "result_id": result_id,
                "lineage_chain": chain,
                "depth": len(chain),
            }))
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/lineage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>/refine-analysis")
    def api_program_refine_analysis(result_id):
        """Data-driven refinement analysis for a program."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics, RefinementAnalyzer

            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            analytics = ExperimentAnalytics(nb)
            analyzer = RefinementAnalyzer(analytics)
            analysis = analyzer.analyze_program_for_refinement(result_id, program)
            return jsonify(_json_safe(analysis))
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/refine-analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>/external-benchmarks", methods=["POST"])
    def api_program_external_benchmarks(result_id):
        """Attach external benchmark scores to a program result."""
        nb = LabNotebook(notebook_path)
        try:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, (dict, list)):
                return jsonify({"error": "Payload must be a JSON object or list."}), 400
            ok = nb.set_external_benchmarks(result_id, payload)
            if not ok:
                return jsonify({"error": "Program result not found or payload invalid."}), 404
            return jsonify({"status": "ok", "result_id": result_id})
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}/external-benchmarks: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/healer/tasks")
    def api_healer_tasks():
        """List recent Code Healer tasks."""
        nb = LabNotebook(notebook_path)
        try:
            limit = request.args.get("limit", 20, type=int)
            return jsonify(nb.get_recent_healer_tasks(limit=max(1, min(limit, 200))))
        except Exception as e:
            logger.error(f"Error in /api/healer/tasks: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/healer/tasks/<task_id>")
    def api_healer_task_detail(task_id: str):
        """Get one healer task with state history."""
        nb = LabNotebook(notebook_path)
        try:
            task = nb.get_healer_task(task_id)
            if task is None:
                return jsonify({"error": "Not found"}), 404
            return jsonify({
                "task": task,
                "events": nb.get_healer_events(task_id, limit=200),
            })
        except Exception as e:
            logger.error(f"Error in /api/healer/tasks/{task_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/failures")
    def api_failure_analysis(experiment_id):
        """Failure analysis: error distribution, stage funnel."""
        nb = LabNotebook(notebook_path)
        try:
            analysis = nb.get_failure_analysis(experiment_id)
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Error in failure analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/analysis")
    def api_experiment_analysis(experiment_id):
        """LLM-generated analysis (stored or on-demand)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404

            # Return stored analysis if available
            stored = exp.get("llm_analysis")
            if stored:
                return jsonify({"analysis": stored, "source": "stored"})

            # Try generating on-demand
            results = exp.get("results") or {}
            from .llm.context import build_experiment_context
            ctx = build_experiment_context(results)
            analysis = aria.analyze_results(results, context=ctx)

            if analysis:
                # Cache it
                try:
                    nb.conn.execute(
                        "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                        (analysis, experiment_id),
                    )
                    nb.conn.commit()
                except Exception as e:
                    logger.warning("Failed caching llm_analysis for %s: %s",
                                   experiment_id, e)
                return jsonify({"analysis": analysis, "source": "generated"})

            return jsonify({"analysis": None, "source": "unavailable",
                            "reason": "No LLM backend configured"})
        except Exception as e:
            logger.error(f"Error in experiment analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends")
    def api_trends():
        """Cross-experiment trend data for charts."""
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_experiment_trends())
        except Exception as e:
            logger.error(f"Error in /api/trends: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends/context")
    def api_trends_context():
        """Trend data plus adaptation-event deltas for inline linkage UI."""
        nb = LabNotebook(notebook_path)

        def _event_delta_payload(trends: List[Dict[str, Any]], event: Dict[str, Any]) -> Dict[str, Any]:
            timestamp = float(event.get("timestamp") or 0.0)
            previous = [row for row in trends if float(row.get("timestamp") or 0.0) < timestamp]
            following = [row for row in trends if float(row.get("timestamp") or 0.0) >= timestamp]

            before = previous[-3:]
            after = following[:3]

            before_ids = [str(row.get("experiment_id")) for row in before if row.get("experiment_id")]
            after_ids = [str(row.get("experiment_id")) for row in after if row.get("experiment_id")]

            def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if not values:
                    return None
                return sum(values) / len(values)

            before_adj_s1 = _avg(before, "adjusted_s1_pass_rate")
            after_adj_s1 = _avg(after, "adjusted_s1_pass_rate")
            before_novelty = _avg(before, "best_novelty_score")
            after_novelty = _avg(after, "best_novelty_score")
            before_loss = _avg(before, "best_loss_ratio")
            after_loss = _avg(after, "best_loss_ratio")

            return {
                "timestamp": timestamp,
                "event_type": event.get("event_type"),
                "description": event.get("description") or "Grammar weights adjusted",
                "before_window": {
                    "n_experiments": len(before),
                    "experiment_ids": before_ids,
                    "adjusted_s1_rate": before_adj_s1,
                    "best_novelty": before_novelty,
                    "best_loss_ratio": before_loss,
                },
                "after_window": {
                    "n_experiments": len(after),
                    "experiment_ids": after_ids,
                    "adjusted_s1_rate": after_adj_s1,
                    "best_novelty": after_novelty,
                    "best_loss_ratio": after_loss,
                },
                "delta": {
                    "adjusted_s1_rate": (
                        after_adj_s1 - before_adj_s1
                        if after_adj_s1 is not None and before_adj_s1 is not None
                        else None
                    ),
                    "best_novelty": (
                        after_novelty - before_novelty
                        if after_novelty is not None and before_novelty is not None
                        else None
                    ),
                    "best_loss_ratio": (
                        after_loss - before_loss
                        if after_loss is not None and before_loss is not None
                        else None
                    ),
                },
            }

        try:
            trends = nb.get_experiment_trends()
            learning_log = nb.get_learning_log(limit=300)
            adaptation_events = [
                _event_delta_payload(trends, event)
                for event in learning_log
                if event.get("event_type") == "grammar_weights_applied"
            ]
            return jsonify({
                "trends": trends,
                "adaptation_events": adaptation_events,
                "generated_at": time.time(),
            })
        except Exception as e:
            logger.error(f"Error in /api/trends/context: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs")
    def api_programs():
        """List top programs."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort", "novelty_score")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            programs = nb.get_top_programs(n, sort_by)
            _annotate_qkv_usage(programs, analytics)
            return jsonify(_json_safe(programs))
        except Exception as e:
            logger.error(f"Error in /api/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights")
    def api_insights():
        """List active insights, deduplicated by content (keeps latest)."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            raw = nb.get_insights(category=category, limit=200)
            return jsonify(_deduplicate_insights(raw))
        except Exception as e:
            logger.error(f"Error in /api/insights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights/boost", methods=["POST"])
    def api_insights_boost():
        """Record a request to boost an insight in future experiment selection."""
        payload = request.get_json(silent=True) or {}
        insight_id = str(payload.get("insight_id") or "").strip()
        content = str(payload.get("content") or "").strip()
        category = str(payload.get("category") or "").strip()
        confidence = payload.get("confidence")
        if not insight_id:
            return jsonify({"error": "insight_id required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            evidence = json.dumps({
                "insight_id": insight_id,
                "category": category or None,
                "confidence": confidence,
                "content": content[:400] if content else None,
            }, sort_keys=True)
            desc = f"Boost requested for insight {insight_id}"
            if category:
                desc += f" ({category})"
            nb.log_learning_event(
                "insight_boost",
                desc,
                evidence=evidence,
            )
            return jsonify({"status": "ok", "insight_id": insight_id})
        except Exception as e:
            logger.error(f"Error in /api/insights/boost: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/entries")
    def api_entries():
        """List notebook entries."""
        exp_id = request.args.get("experiment_id")
        entry_type = request.args.get("type")
        n = request.args.get("n", 50, type=int)
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type=entry_type, limit=n
            )
            return jsonify(_normalize_entries(entries))
        except Exception as e:
            logger.error(f"Error in /api/entries: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/live-feed")
    def api_live_feed():
        """List persisted live-feed events for replay in the dashboard."""
        exp_id = request.args.get("experiment_id")
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            query_limit = max(n, 1000)
            entries = nb.get_entries(
                experiment_id=exp_id,
                entry_type="live_feed",
                limit=query_limit,
            )

            # Default behavior should show a coherent experiment stream.
            # Without this, mixed cross-experiment rows can look like broken
            # generation timelines (e.g., Gen 3 -> Gen 13 with unrelated runs).
            if not exp_id:
                latest_exp_id = next(
                    (
                        entry.get("experiment_id")
                        for entry in entries
                        if entry.get("experiment_id")
                    ),
                    None,
                )
                if latest_exp_id:
                    entries = [
                        entry
                        for entry in entries
                        if entry.get("experiment_id") == latest_exp_id
                    ]

            events = []
            for entry in reversed(entries):
                evt = _entry_to_live_feed_event(entry)
                if evt is not None:
                    events.append(evt)
            if len(events) > n:
                events = events[-n:]
            return jsonify(events)
        except Exception as e:
            logger.error(f"Error in /api/live-feed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/metrics/<metric_name>")
    def api_metrics(metric_name):
        """Get time-series metrics."""
        exp_id = request.args.get("experiment_id")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))
        except Exception as e:
            logger.error(f"Error in /api/metrics/{metric_name}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/dashboard")
    def api_dashboard():
        """Get all dashboard data in one call."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()

            # Add campaign/hypothesis/knowledge counts
            try:
                active_campaigns = nb.get_active_campaigns()
                total_hypotheses = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses"
                ).fetchone()[0]
                knowledge_entries = nb.conn.execute(
                    "SELECT COUNT(*) FROM knowledge_base WHERE status = 'active'"
                ).fetchone()[0]
                summary["active_campaigns"] = len(active_campaigns)
                summary["total_hypotheses"] = total_hypotheses
                summary["knowledge_entries"] = knowledge_entries
            except Exception as e:
                logger.warning("Failed enriching dashboard campaign metadata: %s", e)

            recent_experiments = nb.get_recent_experiments(10)
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            top_programs = nb.get_top_programs(10)
            _annotate_qkv_usage(top_programs, analytics)
            production_readiness = _compute_breakthrough_production_readiness(nb, analytics)

            data = {
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "recent_experiments": recent_experiments,
                "top_programs": top_programs,
                "production_readiness": production_readiness,
                "insights": _deduplicate_insights(nb.get_insights(limit=50)),
                "recent_entries": _normalize_entries(nb.get_entries(limit=20)),
                "is_running": runner.is_running,
                "progress": runner.progress.to_dict(),
            }

            # Compute deltas from latest completed experiment
            try:
                completed = [e for e in recent_experiments
                             if e.get("status") == "completed"]
                if len(completed) >= 2:
                    latest = completed[0]
                    previous = completed[1]
                    data["deltas"] = {
                        "experiment_id": latest.get("experiment_id"),
                        "programs": (latest.get("n_programs_generated") or 0)
                                    - (previous.get("n_programs_generated") or 0),
                        "stage1": (latest.get("n_stage1_passed") or 0)
                                  - (previous.get("n_stage1_passed") or 0),
                        "best_loss": round(
                            (latest.get("best_loss_ratio") or 1)
                            - (previous.get("best_loss_ratio") or 1), 4
                        ) if latest.get("best_loss_ratio") else None,
                        "best_novelty": round(
                            (latest.get("best_novelty_score") or 0)
                            - (previous.get("best_novelty_score") or 0), 4
                        ) if latest.get("best_novelty_score") else None,
                    }
            except Exception:
                pass

            # Include learning trajectory trend in summary
            try:
                trajectory = analytics.learning_trajectory()
                if trajectory and trajectory.get("trend") != "insufficient_data":
                    summary["learning_trend"] = trajectory.get("trend")
                    summary["learning_slope"] = trajectory.get("slope")
                    summary["recent_s1_rate"] = trajectory.get("recent_s1_rate")
            except Exception:
                pass

            # Include latest auto-recommendation if experiment just completed
            last_rec = runner.last_recommendation
            if last_rec:
                data["last_recommendation"] = last_rec

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/dashboard: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Report endpoint ──

    @app.route("/api/report")
    def api_report():
        """Consolidated research report with all data."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            fast_mode = _parse_bool_query(request.args.get("fast"), default=False)
            include_heavy = _parse_bool_query(
                request.args.get("include_heavy"),
                default=not fast_mode,
            )
            include_narrative = _parse_bool_query(
                request.args.get("include_narrative"),
                default=not fast_mode,
            )

            top_limit = 20 if not fast_mode else 12
            expanded_limit = 80 if include_heavy else 0
            recent_limit = 100 if include_heavy else 30

            data = {
                "summary": nb.get_dashboard_summary(),
                "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(top_limit, sort_by="loss_ratio"),
                "top_programs_expanded": nb.get_top_programs(expanded_limit, sort_by="loss_ratio") if include_heavy else [],
                "recent_experiments": nb.get_recent_experiments(recent_limit),
                "op_success_rates": analytics.op_success_rates(),
                "failure_patterns": analytics.failure_patterns(),
                "grammar_weights": {
                    "learned": analytics.compute_grammar_weights(),
                    "default": analytics.get_current_grammar_weights(),
                    "control_comparison": analytics.control_experiment_comparison(),
                    "holdout_validation": analytics.holdout_validation(),
                    "learning_diagnostics": analytics.grammar_weight_learning_diagnostics(),
                },
                "learning_log": nb.get_learning_log(limit=20 if fast_mode else 50),
                "insights": nb.get_insights(),
                "report_mode": {
                    "fast": fast_mode,
                    "include_heavy": include_heavy,
                    "include_narrative": include_narrative,
                },
            }
            if include_heavy:
                data.update({
                    "math_family_coverage": analytics.math_family_coverage(),
                    "mathspace_operator_impact": analytics.mathspace_operator_impact(),
                    "routing_mode_comparison": analytics.routing_mode_comparison(),
                    "gating_behavior_diagnostics": analytics.gating_behavior_diagnostics(),
                    "structural_correlations": analytics.structural_correlations(),
                    "top_op_combinations": analytics.top_op_combinations(10),
                    "efficiency_frontier": analytics.efficiency_frontier(),
                    "experiment_clusters": analytics.experiment_clusters(),
                })
            learning_diagnostics = data["grammar_weights"].get("learning_diagnostics") or {}
            data["architecture_rerun_telemetry"] = {
                "unique_fingerprint_count": int(learning_diagnostics.get("unique_fingerprints") or 0),
                "total_result_rows": int(learning_diagnostics.get("total_rows") or 0),
                "repeat_result_rows": int(learning_diagnostics.get("repeat_rows") or 0),
                "rerun_ratio": float(learning_diagnostics.get("rerun_ratio") or 0.0),
                "top_fingerprint_concentration": float(learning_diagnostics.get("top_fingerprint_concentration") or 0.0),
                "weighting_mode": str(learning_diagnostics.get("mode") or "unknown"),
            }
            data["action_eligibility"] = _build_report_action_eligibility(
                nb,
                [
                    row.get("result_id")
                    for row in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]
                    if row.get("result_id")
                ],
            )
            _annotate_qkv_usage(data["top_programs"], analytics)
            _annotate_qkv_usage(data["top_programs_expanded"], analytics)

            expanded_by_fingerprint: Dict[str, List[Dict[str, Any]]] = {}
            for row in data["top_programs_expanded"]:
                fp = row.get("graph_fingerprint")
                if not fp:
                    continue
                expanded_by_fingerprint.setdefault(fp, []).append(row)

            grouped_rank_by_fingerprint = {
                row.get("graph_fingerprint"): index
                for index, row in enumerate(data["top_programs"], start=1)
                if row.get("graph_fingerprint")
            }
            for fp, rows in expanded_by_fingerprint.items():
                repeat_count = len(rows)
                grouped_rank = grouped_rank_by_fingerprint.get(fp)
                for repeat_index, row in enumerate(rows, start=1):
                    row["group_repeat_count"] = repeat_count
                    row["group_repeat_index"] = repeat_index
                    row["grouped_fingerprint_rank"] = grouped_rank

            data["cross_run_stability"] = _compute_cross_run_stability(
                nb, data["top_programs"]
            )
            stability_by_result = {
                candidate.get("result_id"): candidate
                for candidate in data["cross_run_stability"].get("candidates", [])
                if candidate.get("result_id")
            }
            stability_by_fingerprint = {
                candidate.get("graph_fingerprint"): candidate
                for candidate in data["cross_run_stability"].get("candidates", [])
                if candidate.get("graph_fingerprint")
            }

            fallback_stability = {
                "trend": "unknown",
                "seen_runs": 0,
                "latest_rank": None,
                "previous_rank": None,
                "rank_delta": None,
            }
            for program in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]:
                by_result = stability_by_result.get(program.get("result_id"))
                by_fingerprint = stability_by_fingerprint.get(program.get("graph_fingerprint"))
                program["cross_run_stability"] = by_result or by_fingerprint or fallback_stability

            # Generate narrative only when explicitly enabled
            data["narrative"] = None
            if include_narrative:
                try:
                    narrative = aria.generate_report_narrative(data)
                    data["narrative"] = narrative
                except Exception as e:
                    logger.debug(f"Report narrative generation failed: {e}")
                    data["narrative"] = None

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/report: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/report/query")
    def api_report_query():
        """Scoped report payload for date/theme/trend report generation."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            start_ts = _parse_report_date(request.args.get("start_date"), end_of_day=False)
            end_ts = _parse_report_date(request.args.get("end_date"), end_of_day=True)
            theme = str(request.args.get("theme") or "all").strip().lower()
            trend = str(request.args.get("trend") or "all").strip().lower()
            include_narrative = _parse_bool_query(
                request.args.get("include_narrative"),
                default=False,
            )
            try:
                limit = int(request.args.get("limit") or 20)
            except Exception:
                limit = 20
            limit = max(5, min(120, limit))

            snapshot_query = {
                "start_date": request.args.get("start_date"),
                "end_date": request.args.get("end_date"),
                "theme": theme,
                "trend": trend,
                "limit": limit,
                "include_narrative": bool(include_narrative),
            }
            latest_completed_ts = nb.get_latest_completed_experiment_timestamp()
            snapshot_key = _build_report_snapshot_key("report_query", snapshot_query)

            if not include_narrative:
                cached = nb.get_report_snapshot(
                    snapshot_key=snapshot_key,
                    scope="report_query",
                    min_latest_completed_ts=latest_completed_ts,
                )
                if isinstance(cached, dict):
                    cached["snapshot_cache"] = {
                        "enabled": True,
                        "hit": True,
                        "key": snapshot_key,
                        "latest_completed_ts": latest_completed_ts,
                    }
                    return jsonify(cached)

            experiments = nb.get_recent_experiments(500)
            filtered_experiments = []
            for exp in experiments:
                ts = exp.get("timestamp")
                if isinstance(ts, (int, float)):
                    if start_ts is not None and ts < start_ts:
                        continue
                    if end_ts is not None and ts > end_ts:
                        continue
                if not _report_experiment_matches_trend(exp, trend):
                    continue
                filtered_experiments.append(exp)

            sort_by = "novelty_score" if trend == "high_novelty" else "loss_ratio"
            expanded = nb.get_top_programs(max(limit * 3, 120), sort_by=sort_by)
            filtered_programs: List[Dict[str, Any]] = []
            for program in expanded:
                ts = program.get("timestamp")
                if isinstance(ts, (int, float)):
                    if start_ts is not None and ts < start_ts:
                        continue
                    if end_ts is not None and ts > end_ts:
                        continue
                if not _report_program_matches_theme(program, theme):
                    continue
                filtered_programs.append(program)

            grouped = []
            seen = set()
            for row in filtered_programs:
                fp = row.get("graph_fingerprint")
                if fp and fp in seen:
                    continue
                if fp:
                    seen.add(fp)
                grouped.append(row)
                if len(grouped) >= limit:
                    break

            base_summary = nb.get_dashboard_summary()
            summary = _build_filtered_report_summary(base_summary, filtered_experiments)

            data = {
                "summary": summary,
                "top_programs": grouped,
                "top_programs_expanded": filtered_programs[: max(limit * 2, 40)],
                "recent_experiments": filtered_experiments[: max(limit * 5, 40)],
                "op_success_rates": analytics.op_success_rates(),
                "failure_patterns": analytics.failure_patterns(),
                "insights": nb.get_insights(),
                "learning_log": nb.get_learning_log(limit=30),
                "narrative": None,
                "query": {
                    "start_date": request.args.get("start_date"),
                    "end_date": request.args.get("end_date"),
                    "theme": theme,
                    "trend": trend,
                    "limit": limit,
                    "matched_experiments": len(filtered_experiments),
                    "matched_programs": len(filtered_programs),
                },
                "snapshot_cache": {
                    "enabled": True,
                    "hit": False,
                    "key": snapshot_key,
                    "latest_completed_ts": latest_completed_ts,
                },
            }

            if include_narrative:
                try:
                    data["narrative"] = aria.generate_report_narrative(data)
                except Exception as e:
                    logger.debug(f"Scoped report narrative generation failed: {e}")
                    data["narrative"] = None

            if not include_narrative:
                try:
                    nb.save_report_snapshot(
                        snapshot_key=snapshot_key,
                        scope="report_query",
                        query=snapshot_query,
                        payload=data,
                        latest_completed_ts=latest_completed_ts,
                    )
                except Exception as e:
                    logger.debug(f"Scoped report snapshot save failed: {e}")

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/report/query: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Analytics endpoints ──

    @app.route("/api/analytics/op-success")
    def api_op_success():
        """Op success rate table."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.op_success_rates())
        except Exception as e:
            logger.error(f"Error in op-success: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/failure-patterns")
    def api_failure_patterns():
        """Failure analysis by error type and stage."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.failure_patterns())
        except Exception as e:
            logger.error(f"Error in failure-patterns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/grammar-weights")
    def api_grammar_weights():
        """Current vs learned grammar weights."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            defaults = analytics.get_current_grammar_weights()
            learned = analytics.compute_grammar_weights()
            control_comparison = analytics.control_experiment_comparison()
            holdout = analytics.holdout_validation()
            explanation = aria.explain_grammar_weights(defaults, learned)
            diagnostics = analytics.grammar_weight_learning_diagnostics()
            return jsonify({
                "default": defaults,
                "learned": learned,
                "control_comparison": control_comparison,
                "holdout_validation": holdout,
                "learning_diagnostics": diagnostics,
                "architecture_rerun_telemetry": {
                    "unique_fingerprint_count": int(diagnostics.get("unique_fingerprints") or 0),
                    "total_result_rows": int(diagnostics.get("total_rows") or 0),
                    "repeat_result_rows": int(diagnostics.get("repeat_rows") or 0),
                    "rerun_ratio": float(diagnostics.get("rerun_ratio") or 0.0),
                    "top_fingerprint_concentration": float(diagnostics.get("top_fingerprint_concentration") or 0.0),
                    "weighting_mode": str(diagnostics.get("mode") or "unknown"),
                },
                "explanation": explanation,
            })
        except Exception as e:
            logger.error(f"Error in grammar-weights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier")
    def api_efficiency_frontier():
        """Pareto-optimal programs on loss vs FLOPs."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/regression-vs-baseline")
    def api_regression_vs_baseline():
        """Accuracy/speed tradeoff view based on baseline ratio vs throughput."""
        limit = request.args.get("limit", 200, type=int)
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                """
                SELECT
                    result_id,
                    experiment_id,
                    timestamp,
                    loss_ratio,
                    baseline_loss_ratio,
                    throughput_tok_s,
                    flops_per_token,
                    novelty_score
                FROM program_results
                WHERE stage1_passed = 1
                  AND baseline_loss_ratio IS NOT NULL
                  AND throughput_tok_s IS NOT NULL
                  AND throughput_tok_s > 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(20, int(limit)),),
            ).fetchall()

            points = []
            for row in rows:
                item = dict(row)
                item["baseline_beats_reference"] = float(item.get("baseline_loss_ratio") or 0.0) < 1.0
                points.append(item)

            # Pareto frontier for (maximize throughput, minimize baseline ratio)
            frontier = []
            best_ratio = float("inf")
            for item in sorted(points, key=lambda p: float(p.get("throughput_tok_s") or 0.0), reverse=True):
                ratio = float(item.get("baseline_loss_ratio") or float("inf"))
                if ratio <= best_ratio:
                    frontier.append(item)
                    best_ratio = ratio

            summary = {
                "n_points": len(points),
                "n_beating_baseline": sum(1 for p in points if p["baseline_beats_reference"]),
                "best_baseline_ratio": min(
                    (float(p.get("baseline_loss_ratio") or float("inf")) for p in points),
                    default=None,
                ),
                "best_throughput_tok_s": max(
                    (float(p.get("throughput_tok_s") or 0.0) for p in points),
                    default=0.0,
                ),
                "frontier_count": len(frontier),
            }

            return jsonify({
                "points": points,
                "pareto_frontier": frontier,
                "summary": summary,
            })
        except Exception as e:
            logger.error(f"Error in regression-vs-baseline: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/experiment-clusters")
    def api_experiment_clusters():
        """Deterministic experiment clustering summary and stability signal."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.experiment_clusters())
        except Exception as e:
            logger.error(f"Error in experiment-clusters: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-health")
    def api_routing_health():
        """Routing telemetry health summary grouped by routing mode."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_health() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("explanation", "Routing telemetry is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-comparison")
    def api_routing_comparison():
        """Consolidated routing-mode comparison with confidence/sample labels."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_mode_comparison() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("n_modes", 0)
            payload.setdefault("total_programs", 0)
            payload.setdefault("routed_programs", 0)
            payload.setdefault("uniform_programs", 0)
            payload.setdefault("explanation", "Routing comparison data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/gating-diagnostics")
    def api_gating_diagnostics():
        """Canonical gating behavior diagnostics (entropy/collapse/retention)."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.gating_behavior_diagnostics() or {}
            payload.setdefault("available", False)
            payload.setdefault("total_routed_programs", 0)
            payload.setdefault("avg_gate_entropy", None)
            payload.setdefault("collapse_risk_counts", {"low": 0, "medium": 0, "high": 0, "unknown": 0})
            payload.setdefault("by_mode", [])
            payload.setdefault("token_retention_curve_overall", [])
            payload.setdefault("explanation", "Gating diagnostics are unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in gating-diagnostics: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/math-family-coverage")
    def api_math_family_coverage():
        """Coverage of evaluated/surviving programs by mathematical family."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.math_family_coverage())
        except Exception as e:
            logger.error(f"Error in math-family-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/mathspace-impact")
    def api_mathspace_impact():
        """Impact of math-space operators/families on S1/validation/novelty."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.mathspace_operator_impact() or {}
            payload.setdefault("available", False)
            payload.setdefault("totals", {
                "n_programs_with_graph": 0,
                "n_programs_with_mathspace": 0,
                "n_mathspace_ops_observed": 0,
            })
            payload.setdefault("by_operator", [])
            payload.setdefault("by_family", [])
            payload.setdefault("top_trustworthy_operators", [])
            payload.setdefault("explanation", "Math-space impact data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in mathspace-impact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-coverage")
    def api_compression_coverage():
        """Coverage of compression techniques across tested and surviving programs."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.compression_coverage())
        except Exception as e:
            logger.error(f"Error in compression-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-opportunities")
    def api_compression_opportunities():
        """Ranked compactness opportunities with actionable next-run suggestions."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            coverage = analytics.compression_coverage() or {}
            return jsonify(_compute_compression_opportunities(coverage))
        except Exception as e:
            logger.error(f"Error in compression-opportunities: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/negative-results")
    def api_negative_results():
        """Aggregated negative results: failed ops, error types, anti-patterns."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.negative_results_synthesis())
        except Exception as e:
            logger.error(f"Error in negative-results: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-trajectory")
    def api_learning_trajectory():
        """S1 rate trend over time with regression analysis."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.learning_trajectory())
        except Exception as e:
            logger.error(f"Error in learning-trajectory: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/control-comparison")
    def api_control_comparison():
        """Compare control (default weights) vs learned-weight experiments."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            result = analytics.control_experiment_comparison()
            if result is None:
                return jsonify({"status": "insufficient_data",
                                "message": "Need at least 2 control and 2 learned experiments"})
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in control-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-summary")
    def api_learning_summary():
        """Aria-generated 3-5 bullet summary of what the system has learned."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            payload = aria.summarize_learning_bullets({
                "summary": nb.get_dashboard_summary(),
                "grammar_default": analytics.get_current_grammar_weights(),
                "grammar_learned": analytics.compute_grammar_weights(),
                "frontier": analytics.efficiency_frontier(),
                "clusters": analytics.experiment_clusters(),
                "recent_experiments": nb.get_recent_experiments(10),
                "trajectory": analytics.learning_trajectory(),
            })
            payload.setdefault("bullets", [])
            payload.setdefault("source", "rule-based")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in learning-summary: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-log")
    def api_learning_log():
        """Audit trail of grammar weight changes."""
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_learning_log(limit=n))
        except Exception as e:
            logger.error(f"Error in learning-log: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/insight-interactions")
    def api_insight_interactions():
        """Pairwise insight synergy/antagonism learned from selection outcomes."""
        nb = LabNotebook(notebook_path)
        try:
            limit = request.args.get("limit", 80, type=int)
            min_trials = request.args.get("min_trials", 1, type=int)
            rows = nb.get_selection_insight_interactions(limit=max(1, min(limit, 500)))
            rows = [
                row for row in rows
                if int(row.get("n_trials") or 0) >= max(1, int(min_trials))
            ]

            insight_rows = nb.get_insights(limit=500)
            insight_by_id = {
                str(row.get("insight_id")): row for row in insight_rows if row.get("insight_id")
            }

            enriched: List[Dict[str, Any]] = []
            for row in rows:
                a_id = str(row.get("insight_a") or "")
                b_id = str(row.get("insight_b") or "")
                a = insight_by_id.get(a_id, {})
                b = insight_by_id.get(b_id, {})
                mean_reward = _to_safe_float(row.get("mean_reward"), 0.0)
                n_trials = int(row.get("n_trials") or 0)
                supported = int(row.get("n_supported") or 0)
                not_supported = int(row.get("n_not_supported") or 0)
                support_rate = (supported / n_trials) if n_trials > 0 else 0.0
                label = "synergistic" if mean_reward >= 0.55 else ("antagonistic" if mean_reward <= 0.45 else "mixed")
                confidence = "high" if n_trials >= 8 else ("medium" if n_trials >= 4 else "low")
                enriched.append({
                    **row,
                    "support_rate": round(support_rate, 6),
                    "interaction_label": label,
                    "confidence_label": confidence,
                    "insight_a_content": a.get("content"),
                    "insight_b_content": b.get("content"),
                    "insight_a_category": a.get("category"),
                    "insight_b_category": b.get("category"),
                    "is_singleton": a_id == b_id,
                })

            synergistic = [
                row for row in enriched
                if not row.get("is_singleton") and row.get("interaction_label") == "synergistic"
            ][:10]
            antagonistic = [
                row for row in enriched
                if not row.get("is_singleton") and row.get("interaction_label") == "antagonistic"
            ][:10]
            singleton = [
                row for row in enriched
                if row.get("is_singleton")
            ][:10]
            return jsonify({
                "available": len(enriched) > 0,
                "total_interactions": len(enriched),
                "synergistic_pairs": synergistic,
                "antagonistic_pairs": antagonistic,
                "singleton_insights": singleton,
                "interactions": enriched,
                "explanation": (
                    "Interaction score is learned from downstream outcomes of selection decisions "
                    "(supported/not_supported with reward aggregation)."
                ),
            })
        except Exception as e:
            logger.error(f"Error in insight-interactions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/decision-packet/<result_id>")
    def api_decision_packet(result_id):
        """One-click evidence bundle for promotion decisions."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            fingerprint = program.get("graph_fingerprint", "")
            experiment_id = program.get("experiment_id")

            # Leaderboard entry (targeted)
            leaderboard_entry = None
            try:
                leaderboard_entry = nb.get_leaderboard_entry(result_id)
            except Exception:
                leaderboard_entry = None

            # Experiment data + failure analysis
            experiment = None
            failure_analysis = {"funnel": {}, "errors": {}, "stage_deaths": {}}
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass
                try:
                    failure_analysis = nb.get_failure_analysis(experiment_id)
                except Exception:
                    pass

            # Hypothesis chain — find hypothesis linked to this experiment
            hypothesis_chain = []
            if experiment_id:
                try:
                    hyp_row = nb.conn.execute(
                        "SELECT hypothesis_id FROM hypotheses WHERE experiment_id = ?",
                        (experiment_id,),
                    ).fetchone()
                    if hyp_row:
                        hypothesis_chain = nb.get_hypothesis_chain(
                            hyp_row["hypothesis_id"] if isinstance(hyp_row, dict)
                            else hyp_row[0]
                        )
                except Exception:
                    pass

            # Cross-run stability for this specific result
            cross_run = {"trend": "unknown", "seen_runs": 0}
            try:
                top = nb.get_top_programs(20, sort_by="loss_ratio")
                stability = _compute_cross_run_stability(nb, top)
                for c in stability.get("candidates", []):
                    if c.get("result_id") == result_id:
                        cross_run = {
                            "trend": c.get("trend", "unknown"),
                            "seen_runs": c.get("seen_runs", 0),
                        }
                        break
            except Exception:
                pass

            # Build outcomes by phase
            tier = (leaderboard_entry or {}).get("tier", "screening")
            outcomes = {
                "screening": {
                    "loss_ratio": program.get("loss_ratio"),
                    "novelty": program.get("novelty_score"),
                },
                "investigation": None,
                "validation": None,
            }
            if leaderboard_entry:
                inv_lr = leaderboard_entry.get("investigation_loss_ratio")
                if inv_lr is not None:
                    outcomes["investigation"] = {
                        "loss_ratio": inv_lr,
                        "robustness": leaderboard_entry.get("investigation_robustness"),
                        "passed": bool(leaderboard_entry.get("investigation_passed")),
                    }
                val_lr = leaderboard_entry.get("validation_loss_ratio")
                if val_lr is not None:
                    outcomes["validation"] = {
                        "loss_ratio": val_lr,
                        "baseline_ratio": leaderboard_entry.get("validation_baseline_ratio"),
                        "multi_seed_std": leaderboard_entry.get("validation_multi_seed_std"),
                        "passed": bool(leaderboard_entry.get("validation_passed")),
                    }

            # Baseline comparison
            bl_ratio = program.get("baseline_loss_ratio")
            baseline_comparison = {"ratio": bl_ratio, "interpretation": "unknown"}
            if bl_ratio is not None:
                if bl_ratio < 0.95:
                    baseline_comparison["interpretation"] = "outperforms"
                elif bl_ratio <= 1.05:
                    baseline_comparison["interpretation"] = "comparable"
                else:
                    baseline_comparison["interpretation"] = "underperforms"

            # Failure context
            failure_context = {
                "stage_at_death": program.get("stage_at_death"),
                "error_type": program.get("error_type"),
                "experiment_errors": failure_analysis.get("errors", {}),
                "experiment_funnel": failure_analysis.get("funnel", {}),
            }

            # Recommendation
            recommendation = _compute_recommendation(program, leaderboard_entry)

            # Evidence flags
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            packet_status = analytics.reproducibility_packet_status(
                leaderboard_entry if leaderboard_entry else program
            )
            evidence_flags = {
                "has_baseline": bl_ratio is not None,
                "has_cka_artifact": program.get("cka_source") == "artifact",
                "has_multi_seed": outcomes["validation"] is not None,
                "has_hypothesis": len(hypothesis_chain) > 0,
                "repro_packet_ready": packet_status.get("status") == "ready",
            }

            return jsonify({
                "result_id": result_id,
                "fingerprint": fingerprint,
                "experiment_id": experiment_id,
                "hypothesis_chain": hypothesis_chain,
                "outcomes": outcomes,
                "baseline_comparison": baseline_comparison,
                "failure_context": failure_context,
                "cross_run_stability": cross_run,
                "recommendation": recommendation,
                "evidence_flags": evidence_flags,
                "compression_metrics": analytics.canonical_compression_metrics(
                    leaderboard_entry if leaderboard_entry else program
                ),
                "reproducibility_packet": packet_status,
            })
        except Exception as e:
            logger.error(f"Error in /api/decision-packet/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/reproducibility-manifest/<result_id>")
    def api_reproducibility_manifest(result_id):
        """Exportable reproducibility manifest for a program result."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            experiment_id = program.get("experiment_id")
            experiment = None
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass

            config = (experiment or {}).get("config", {}) or {}
            training = {}
            try:
                tp = json.loads(program.get("training_program_json") or "{}")
                training = tp
            except (json.JSONDecodeError, TypeError):
                pass

            # Grammar weights snapshot from experiment config
            grammar_weights = config.get("applied_grammar_weights") or config.get("grammar_weights")
            grammar_config = config.get("grammar_config", {})

            manifest = {
                "result_id": result_id,
                "graph_fingerprint": program.get("graph_fingerprint"),
                "experiment_id": experiment_id,
                "experiment_type": (experiment or {}).get("experiment_type"),
                "timestamp": program.get("timestamp"),
                "code_version": config.get("code_version"),
                "seeds": {
                    "experiment_seed": config.get("seed"),
                    "training_seed": training.get("seed"),
                },
                "data": {
                    "data_mode": config.get("data_mode"),
                    "dataset": config.get("dataset"),
                    "seq_len": training.get("seq_len") or config.get("seq_len"),
                    "batch_size": training.get("batch_size") or config.get("batch_size"),
                    "vocab_size": training.get("vocab_size") or config.get("vocab_size"),
                },
                "grammar": {
                    "max_ops": grammar_config.get("max_ops"),
                    "max_depth": grammar_config.get("max_depth"),
                    "weights_snapshot": grammar_weights,
                },
                "training": {
                    "learning_rate": training.get("learning_rate") or training.get("lr"),
                    "steps": training.get("steps") or training.get("n_steps"),
                    "warmup_steps": training.get("warmup_steps"),
                },
                "architecture": {
                    "param_count": program.get("param_count"),
                    "graph_json": program.get("graph_json"),
                },
                "outcomes": {
                    "stage0_passed": bool(program.get("stage0_passed")),
                    "stage05_passed": bool(program.get("stage05_passed")),
                    "stage1_passed": bool(program.get("stage1_passed")),
                    "loss_ratio": program.get("loss_ratio"),
                    "novelty_score": program.get("novelty_score"),
                    "baseline_loss_ratio": program.get("baseline_loss_ratio"),
                },
                "canonical_metrics": {
                    "compression": analytics.canonical_compression_metrics(program),
                },
                "packet_status": analytics.reproducibility_packet_status(program),
            }
            return jsonify(manifest)
        except Exception as e:
            logger.error(f"Error in /api/reproducibility-manifest/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>/training-curve")
    def api_training_curve(result_id):
        """Per-step training data for a program."""
        nb = LabNotebook(notebook_path)
        try:
            curve = nb.get_training_curve(result_id)
            return jsonify(curve)
        except Exception as e:
            logger.error(f"Error in training-curve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Leaderboard endpoints ──

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            stability = _compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {
                        "trend": "unknown",
                        "seen_runs": 0,
                        "latest_rank": None,
                        "previous_rank": None,
                        "rank_delta": None,
                    },
                )
            _annotate_qkv_usage(entries, analytics)
            # Group by tier for the dashboard
            tiers = {}
            for entry in entries:
                t = entry.get("tier", "screening")
                if t not in tiers:
                    tiers[t] = []
                tiers[t].append(entry)
            return jsonify({
                "entries": entries,
                "by_tier": tiers,
                "total": len(entries),
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/discoveries")
    def api_discoveries():
        """Unified discoveries endpoint merging leaderboard + raw candidates.

        Query params:
          tier: filter by tier (screening/investigation/validation/breakthrough)
          limit: max results (default 100)
          sort: sort key (default composite_score)
          view: 'all' for raw candidates, 'ranked' for leaderboard (default ranked)
        """
        from .naming import annotate_display_names

        tier = request.args.get("tier")
        limit = request.args.get("limit", 100, type=int)
        sort_by = request.args.get("sort", "composite_score")
        view = request.args.get("view", "ranked")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            if view == "all":
                # Raw S1 survivors from program_results
                programs = nb.get_top_programs(limit, sort_by="loss_ratio")
                _annotate_qkv_usage(programs, analytics)
                # Add family classification + display names
                for p in programs:
                    p["architecture_family"] = nb._classify_architecture_family(
                        graph_json=p.get("graph_json"),
                        routing_mode=p.get("routing_mode"),
                    )
                    p["tier"] = _infer_tier_for_program(nb, p)
                annotate_display_names(programs)
                # Strip large fields from response
                for p in programs:
                    p.pop("graph_json", None)
                    p.pop("_graph_json", None)
                    p.pop("loss_curve", None)

                # Compute tier counts from all S1 survivors
                tier_counts = _count_discovery_tiers(nb)

                return jsonify({
                    "entries": _json_safe(programs),
                    "total": len(programs),
                    "tier_counts": tier_counts,
                    "view": "all",
                })

            # Default: ranked leaderboard view
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            stability = _compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {"trend": "unknown", "seen_runs": 0,
                     "latest_rank": None, "previous_rank": None, "rank_delta": None},
                )
            _annotate_qkv_usage(entries, analytics)
            annotate_display_names(entries)

            # Summary counts
            tier_counts = _count_discovery_tiers(nb)

            return jsonify({
                "entries": _json_safe(entries),
                "total": len(entries),
                "tier_counts": tier_counts,
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
                "view": "ranked",
            })
        except Exception as e:
            logger.error(f"Error in /api/discoveries: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/fingerprint/resolve")
    def api_fingerprint_resolve():
        """Resolve a result_id or fingerprint prefix to a concrete program result."""
        value = str(request.args.get("value") or "").strip()
        if not value:
            return jsonify({"error": "value query param required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            direct = nb.conn.execute(
                "SELECT result_id, graph_fingerprint FROM program_results WHERE result_id = ?",
                (value,),
            ).fetchone()
            if direct:
                return jsonify({
                    "result_id": direct["result_id"],
                    "graph_fingerprint": direct.get("graph_fingerprint"),
                    "resolved_from": "result_id",
                    "candidates": [],
                })
            resolved = _resolve_scale_up_result_ids(nb, [], [value])
            if resolved.get("resolved_fingerprints"):
                chosen = resolved["resolved_fingerprints"][0]
                return jsonify({
                    "result_id": chosen.get("selected_result_id"),
                    "graph_fingerprint": chosen.get("selected_graph_fingerprint"),
                    "resolved_from": "graph_fingerprint",
                    "candidates": chosen.get("candidates", []),
                })
            return jsonify({"error": "No matching fingerprint or result_id found."}), 404
        except Exception as e:
            logger.error(f"Error in /api/fingerprint/resolve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Control endpoints ──

    @app.route("/api/experiments/start", methods=["POST"])
    def api_start_experiment():
        """Start a new experiment. Accepts RunConfig fields + optional hypothesis."""
        runner = _get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        body = request.get_json(silent=True) or {}
        auto_harden = bool(body.pop("auto_harden", True))
        hypothesis = body.pop("hypothesis", None)
        preregistration = body.pop("preregistration", None)
        exploratory = bool(body.pop("exploratory", False))
        refine_analysis_json = body.pop("refine_analysis_json", "")
        mode = _normalize_start_mode(body.pop("mode", "single"))

        config = RunConfig.from_dict(body) if body else RunConfig()
        if refine_analysis_json:
            config.refine_analysis_json = (
                refine_analysis_json if isinstance(refine_analysis_json, str)
                else json.dumps(refine_analysis_json)
            )
        compact_changes: Dict[str, Any] = {}
        sparse_morph_changes: Dict[str, Any] = {}
        if mode == "compact_synthesis":
            compact_changes = _apply_compact_synthesis_bias(config)
            mode = "single"
        if mode == "sparse_morph":
            sparse_morph_changes = _apply_sparse_morph_bias(config)
            mode = "single"

        config, prescreen = runner.prescreen_run_config(
            config,
            mode=mode,
            auto_harden=auto_harden,
        )

        eligibility: Optional[Dict[str, Any]] = None
        scale_up_resolution: Optional[Dict[str, Any]] = None
        refine_resolution: Optional[Dict[str, Any]] = None

        try:
            if mode == "continuous":
                config.continuous = True
                exp_id = runner.start_continuous(config)
            elif mode == "evolve":
                exp_id = runner.start_evolution(
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "novelty":
                exp_id = runner.start_novelty_search(
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "investigation":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for investigation mode"}), 400
                nb = LabNotebook(notebook_path)
                try:
                    eligibility = _build_start_mode_eligibility(nb, "investigation", result_ids)
                finally:
                    nb.close()
                if not eligibility.get("all_eligible"):
                    return jsonify({
                        "error": "Ineligible result_ids for investigation mode",
                        "eligibility": eligibility,
                    }), 409
                exp_id = runner.start_investigation(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "validation":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                if not result_ids:
                    return jsonify({"error": "result_ids required for validation mode"}), 400
                nb = LabNotebook(notebook_path)
                try:
                    eligibility = _build_start_mode_eligibility(nb, "validation", result_ids)
                finally:
                    nb.close()
                if not eligibility.get("all_eligible"):
                    return jsonify({
                        "error": "Ineligible result_ids for validation mode",
                        "eligibility": eligibility,
                    }), 409
                exp_id = runner.start_validation(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "scale_up":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                graph_fingerprints = _normalize_result_ids(
                    body.get("graph_fingerprints", body.get("fingerprints", [])),
                )
                nb = LabNotebook(notebook_path)
                try:
                    scale_up_resolution = _resolve_scale_up_result_ids(
                        nb,
                        result_ids=result_ids,
                        graph_fingerprints=graph_fingerprints,
                    )
                finally:
                    nb.close()
                result_ids = scale_up_resolution.get("result_ids", [])
                if not result_ids:
                    return jsonify({
                        "error": "result_ids or graph_fingerprints required for scale_up mode",
                        "scale_up_resolution": scale_up_resolution,
                    }), 400
                config.scale_up = True
                config.scale_up_result_ids = ",".join(result_ids)
                exp_id = runner.start_scale_up(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )
            elif mode == "refine_fingerprint":
                result_ids = _normalize_result_ids(body.get("result_ids", []))
                graph_fingerprints = _normalize_result_ids(
                    body.get("graph_fingerprints", body.get("fingerprints", [])),
                )
                nb = LabNotebook(notebook_path)
                try:
                    refine_resolution = _resolve_scale_up_result_ids(
                        nb,
                        result_ids=result_ids,
                        graph_fingerprints=graph_fingerprints,
                    )
                finally:
                    nb.close()

                result_ids = refine_resolution.get("result_ids", [])
                if not result_ids:
                    return jsonify({
                        "error": "result_ids or graph_fingerprints required for refine_fingerprint mode",
                        "refine_resolution": refine_resolution,
                    }), 400

                exp_id = runner.start_fingerprint_refinement(
                    result_ids,
                    config,
                    hypothesis=hypothesis,
                )
            else:
                exp_id = runner.start_experiment(
                    config,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    exploratory=exploratory,
                )

            _record_run_trigger(
                experiment_id=exp_id,
                source="ui_start",
                mode=mode,
                details={
                    "endpoint": "/api/experiments/start",
                    "auto_harden": auto_harden,
                },
            )
            critique = (
                runner.progress.hypothesis_critique
                if isinstance(runner.progress.hypothesis_critique, dict)
                else None
            )
            missing_fields = _extract_hypothesis_missing_fields(critique)

            return jsonify({
                "experiment_id": exp_id,
                "status": "started",
                "config": config.to_dict(),
                "prescreen": prescreen,
                "compact_synthesis_bias": compact_changes,
                "sparse_morph_bias": sparse_morph_changes,
                "scale_up_resolution": scale_up_resolution,
                "refine_resolution": refine_resolution,
                "aria_message": runner.progress.aria_message,
                "hypothesis_critique": critique,
                "hypothesis_review_gate": critique.get("gate") if critique else None,
                "hypothesis_missing_fields": missing_fields,
                "eligibility": eligibility,
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error starting experiment: {e}\n{traceback.format_exc()}")
            error_text = str(e)
            auto_repair_task: Optional[Dict[str, Any]] = None
            if _should_autospawn_self_repair(error_text):
                try:
                    auto_repair_task = _spawn_code_agent_task(
                        goal=(
                            "Experiment start failed with runtime/code error. "
                            f"mode={mode}, error={error_text}. "
                            "Identify root cause, apply safe code/config fixes, and report validation."
                        ),
                        notebook_path=notebook_path,
                        allow_write=True,
                        session_id="",
                    )
                except Exception as spawn_err:
                    logger.warning("Auto self-repair spawn failed: %s", spawn_err)
            return jsonify({
                "error": error_text,
                "auto_repair_started": bool(auto_repair_task),
                "auto_repair_task": auto_repair_task,
            }), 500

    @app.route("/api/experiments/stop", methods=["POST"])
    def api_stop_experiment():
        """Stop the currently running experiment."""
        runner = _get_runner(notebook_path)
        if not runner.is_running:
            return jsonify({"error": "No experiment is running"}), 409

        runner.stop()
        return jsonify({
            "status": "stopping",
            "aria_message": runner.progress.aria_message,
        })

    @app.route("/api/experiments/<experiment_id>/cancel", methods=["POST"])
    def api_cancel_experiment(experiment_id):
        """Cancel a stuck/running experiment by marking it as failed."""
        nb = LabNotebook(notebook_path)
        try:
            cancelled = nb.cancel_experiment(experiment_id)
            if not cancelled:
                return jsonify({
                    "error": "Experiment not found or not in running state",
                }), 404
            return jsonify({"status": "cancelled", "experiment_id": experiment_id})
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/rerun", methods=["POST"])
    def api_rerun_experiment(experiment_id):
        """Relaunch an experiment using its stored config and mode."""
        runner = _get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        nb = LabNotebook(notebook_path)
        try:
            source = nb.get_resumable_experiment(experiment_id)
            if source is None:
                source = nb.get_experiment(experiment_id)
            if source is None:
                return jsonify({"error": "Experiment not found"}), 404

            try:
                config_dict = json.loads(source.get("config_json") or "{}")
            except Exception:
                config_dict = {}
            config = RunConfig.from_dict(config_dict)
            hypothesis = source.get("hypothesis")
            exp_type = str(source.get("experiment_type") or "synthesis").strip().lower()

            # If it is still marked running from a stale reboot state, mark it cancelled first.
            if str(source.get("status") or "").strip().lower() == "running":
                nb.cancel_experiment(experiment_id)

            if exp_type == "continuous":
                config.continuous = True
                new_id = runner.start_continuous(config)
                mode = "continuous"
            elif exp_type == "evolution":
                new_id = runner.start_evolution(config, hypothesis=hypothesis)
                mode = "evolve"
            elif exp_type == "novelty":
                new_id = runner.start_novelty_search(config, hypothesis=hypothesis)
                mode = "novelty"
            else:
                # Fallback to single synthesis-style rerun.
                new_id = runner.start_experiment(config, hypothesis=hypothesis)
                mode = "single"

            _record_run_trigger(
                experiment_id=new_id,
                source="ui_rerun",
                mode=mode,
                details={
                    "endpoint": f"/api/experiments/{experiment_id}/rerun",
                    "source_experiment_id": experiment_id,
                },
            )

            return jsonify({
                "status": "started",
                "source_experiment_id": experiment_id,
                "experiment_id": new_id,
                "mode": mode,
                "config": config.to_dict(),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error rerunning experiment {experiment_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/cleanup-stale", methods=["POST"])
    def api_cleanup_stale():
        """Clean up stale running experiments that are no longer active."""
        nb = LabNotebook(notebook_path)
        try:
            count = nb.cleanup_stale_experiments()
            return jsonify({"cleaned": count})
        finally:
            nb.close()

    @app.route("/api/progress")
    def api_progress():
        """Get current experiment progress (poll-based alternative to SSE)."""
        runner = _get_runner(notebook_path)
        progress_payload = runner.progress.to_dict()
        trigger = _get_run_trigger_snapshot(progress_payload.get("experiment_id"))
        progress_payload["run_trigger_source"] = trigger.get("source")
        progress_payload["run_trigger"] = trigger
        return jsonify({
            "is_running": runner.is_running,
            "progress": progress_payload,
            "run_trigger_source": trigger.get("source"),
            "run_trigger": trigger,
        })

    @app.route("/api/events")
    def api_events():
        """SSE endpoint for real-time experiment events."""
        runner = _get_runner(notebook_path)
        sse_timeout = _get_sse_timeout_seconds()

        def event_stream():
            while True:
                for event in runner.get_events(timeout=sse_timeout):
                    data = json.dumps(
                        _json_safe(event.get("data", {})),
                        allow_nan=False,
                    )
                    yield f"event: {event['type']}\ndata: {data}\n\n"
                # After timeout, check if client is still connected
                yield f"event: keepalive\ndata: {{}}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/diagnostics/fingerprint")
    def api_fingerprint_diagnostics():
        """Expose lightweight runtime diagnostics for fingerprint analysis."""
        reset = str(request.args.get("reset", "0")).strip().lower() in {"1", "true", "yes"}
        try:
            from research.eval.fingerprint import get_sensitivity_skip_stats

            stats = get_sensitivity_skip_stats(reset=reset)
            return jsonify({
                "sensitivity_skips": stats,
            })
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/fingerprint: {e}")
            return jsonify({
                "sensitivity_skips": {
                    "total": 0,
                    "by_reason": {},
                },
                "error": str(e),
            }), 500

    @app.route("/api/diagnostics/report-cache")
    def api_report_cache_diagnostics():
        """Expose report snapshot cache usage and retention diagnostics."""
        nb = LabNotebook(notebook_path)
        try:
            cleanup = str(request.args.get("cleanup", "0")).strip().lower() in {"1", "true", "yes"}
            try:
                ttl_seconds = int(os.environ.get("ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600)))
            except Exception:
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400"))
            except Exception:
                max_rows_per_scope = 400

            cleanup_stats = None
            if cleanup:
                cleanup_stats = nb.cleanup_report_snapshots(
                    ttl_seconds=max(60, ttl_seconds),
                    max_rows_per_scope=max(20, max_rows_per_scope),
                )

            snapshot_stats = nb.get_report_snapshot_stats()
            return jsonify({
                "snapshot_cache": snapshot_stats,
                "retention": {
                    "ttl_seconds": max(60, int(ttl_seconds or 0)),
                    "max_rows_per_scope": max(20, int(max_rows_per_scope or 0)),
                },
                "cleanup_triggered": bool(cleanup),
                "cleanup": cleanup_stats,
            })
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/report-cache: {e}")
            return jsonify({
                "snapshot_cache": {
                    "total_snapshots": 0,
                    "n_scopes": 0,
                    "oldest_age_seconds": None,
                    "newest_age_seconds": None,
                    "scopes": [],
                },
                "error": str(e),
            }), 500
        finally:
            nb.close()

    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        """Get the default RunConfig."""
        return jsonify(RunConfig().to_dict())

    # ── LLM Configuration endpoints ──

    @app.route("/api/llm/config")
    def api_llm_config():
        """Get current LLM backend configuration."""
        aria = get_aria()
        return jsonify(aria.get_llm_config())

    @app.route("/api/llm/config", methods=["POST"])
    def api_llm_configure():
        """Configure the LLM backend at runtime and persist to disk."""
        aria = get_aria()
        body = request.get_json(silent=True) or {}

        backend_name = str(body.get("backend", "")).strip()
        if not backend_name:
            return jsonify({"error": "backend is required (anthropic, openai, ollama)"}), 400

        api_key = str(body.get("api_key", "")).strip()
        model = str(body.get("model", "")).strip()
        host = str(body.get("host", "")).strip()

        success = aria.configure_llm(
            backend_name=backend_name,
            api_key=api_key,
            model=model,
            host=host,
        )

        if success:
            # Quick health check: try a minimal LLM call to verify the key works
            health_ok = True
            health_error = None
            llm = aria._get_llm()
            if llm:
                try:
                    test_resp = llm.generate(
                        "Respond with exactly: OK",
                        max_tokens=10, temperature=0,
                    )
                    if not (test_resp and test_resp.text):
                        health_ok = False
                        health_error = "LLM returned empty response"
                except Exception as e:
                    health_ok = False
                    health_error = f"{type(e).__name__}: {str(e)[:150]}"
                    logger.warning(f"LLM health check failed: {health_error}")

            # Persist config so it survives server restarts
            _save_llm_config(notebook_path, {
                "backend": backend_name,
                "api_key": api_key,
                "model": model,
                "host": host,
            })

            # Clear any cached deterministic briefing so AI takes over
            if hasattr(aria, "_briefing_cache"):
                aria._briefing_cache = None

            result = {
                "status": "configured",
                "config": aria.get_llm_config(),
            }
            if not health_ok:
                result["status"] = "configured_with_warning"
                result["warning"] = health_error
            return jsonify(result)
        else:
            return jsonify({"error": "Failed to configure LLM backend"}), 500

    # ── Strategy Briefing endpoint ──

    def _normalize_briefing_mode(mode: Optional[str]) -> Optional[str]:
        if not mode:
            return None
        normalized = str(mode).strip().lower()
        aliases = {
            "evolution": "evolve",
            "evolve": "evolve",
            "novelty_search": "novelty",
            "novelty": "novelty",
            "investigate": "investigation",
            "investigation": "investigation",
            "validate": "validation",
            "validation": "validation",
            "scale-up": "scale_up",
            "scale_up": "scale_up",
            "continuous": "continuous",
            "single": "single",
        }
        return aliases.get(normalized, normalized)

    def _briefing_action_from_mode(mode: Optional[str]) -> Optional[str]:
        if not mode:
            return None
        actions = {
            "investigation": "investigate",
            "validation": "validate",
            "continuous": "continuous",
            "novelty": "novelty_search",
            "evolve": "evolve",
            "scale_up": "scale_up",
        }
        return actions.get(mode)

    def _briefing_action_label(mode: Optional[str], hypothesis: Optional[str] = None) -> str:
        """Human-readable label for an LLM-suggested action."""
        labels = {
            "continuous": "Run Continuous Research",
            "evolve": "Run Evolution Search",
            "novelty": "Run Novelty Search",
            "investigation": "Investigate Candidates",
            "validation": "Run Validation",
            "scale_up": "Scale Up Training",
        }
        return labels.get(mode, f"Run {mode or 'experiment'}")

    def _sparse_coverage_summary(sparse_coverage_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        summary = sparse_coverage_data or {}
        sparse_share = summary.get("sparse_share")
        sparse_survival_rate = summary.get("sparse_survival_rate")
        target_share = 0.15
        sparse_share_value = float(sparse_share) if isinstance(sparse_share, (int, float)) else None
        sparse_survival_value = float(sparse_survival_rate) if isinstance(sparse_survival_rate, (int, float)) else None
        below_target = bool(sparse_share_value is not None and sparse_share_value < target_share)
        return {
            "sparse_share": sparse_share_value,
            "sparse_survival_rate": sparse_survival_value,
            "target_share": target_share,
            "below_target": below_target,
        }

    def _augment_sparse_action_config(
        suggested_config: Optional[Dict[str, Any]],
        normalized_mode: Optional[str],
        sparse_coverage_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        config = dict(suggested_config or {})
        sparse_summary = _sparse_coverage_summary(sparse_coverage_data)
        if not sparse_summary.get("below_target"):
            return config

        mode = str(normalized_mode or config.get("mode") or "").strip().lower()
        if mode not in {"novelty", "evolve", "continuous", "single", "synthesis"}:
            return config

        config.setdefault("model_source", "mixed")
        config.setdefault("morph_focus_sparse", True)
        config.setdefault("morph_ratio", 0.8)
        config.setdefault("use_synthesized_training", True)
        config.setdefault("morph_sparse_weight_storage", "semi_structured_2_4")
        config.setdefault("math_space_weight", 2.2)
        config.setdefault("n_programs", 120)
        if mode in {"novelty", "evolve"}:
            config.setdefault("max_depth", 6)
            config.setdefault("max_ops", 10)
        return config

    @app.route("/api/strategy/briefing")
    def api_strategy_briefing():
        """Data-driven strategy briefing for the overview page.

        Tries LLM-powered briefing first (via Aria), falls back to
        deterministic rules.  Always returns a valid response.
        """
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            summary = nb.get_dashboard_summary()
            recent = nb.get_recent_experiments(10)
            trajectory = analytics.learning_trajectory() or {}
            compression_coverage = analytics.compression_coverage() or {}
            compression_opportunities = _compute_compression_opportunities(compression_coverage)
            primitive_effectiveness = analytics.compression_primitive_effectiveness() or {}
            sparse_evidence = _compute_sparse_evidence(nb)
            sparse_coverage_data = analytics.sparse_coverage() or {}
            sparse_coverage_summary = _sparse_coverage_summary(sparse_coverage_data)

            # Optional: highlight a just-completed experiment
            just_completed_id = request.args.get("just_completed")
            just_completed_exp = None
            if just_completed_id:
                for e in recent:
                    if (e.get("experiment_id") or "").startswith(just_completed_id):
                        just_completed_exp = e
                        break
                # Clear briefing cache so LLM sees the new context
                aria_inst = get_aria()
                if hasattr(aria_inst, "_briefing_cache"):
                    aria_inst._briefing_cache = None

            # --- Pipeline counts ---
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}
            screening = tiers.get("screening", 0)
            investigation = tiers.get("investigation", 0)
            validation = tiers.get("validation", 0)
            breakthrough = tiers.get("breakthrough", 0)

            # --- Recent outcomes ---
            completed = [e for e in recent if e.get("status") == "completed"]
            recent_s1_rates = []
            for e in completed[:5]:
                gen = e.get("n_programs_generated") or 0
                passed = e.get("n_stage1_passed") or 0
                if gen > 0:
                    recent_s1_rates.append(passed / gen)

            avg_recent_s1 = (
                sum(recent_s1_rates) / len(recent_s1_rates)
                if recent_s1_rates
                else None
            )

            # --- Learning trend ---
            trend = trajectory.get("trend", "insufficient_data")
            slope = trajectory.get("slope")

            # --- Common data block (used by both LLM and deterministic) ---
            total_exp = summary.get("total_experiments", 0)
            total_progs = summary.get("total_programs_evaluated", 0)
            s1_survivors = summary.get("stage1_survivors", 0)

            pipeline_data = {
                "screening": screening,
                "investigation": investigation,
                "validation": validation,
                "breakthrough": breakthrough,
            }
            compression_summary = (compression_opportunities.get("summary") or {})
            data_block = {
                "total_experiments": total_exp,
                "total_programs": total_progs,
                "s1_survivors": s1_survivors,
                "avg_recent_s1_rate": avg_recent_s1,
                "learning_trend": trend,
                "learning_slope": slope,
                "pipeline": pipeline_data,
                "compression": compression_summary,
                "compression_primitives": primitive_effectiveness.get("primitives", []),
                "sparse": sparse_evidence,
            }

            recent_window = recent[:10]
            recent_cancelled = 0
            recent_failed = 0
            for exp in recent_window:
                status = str(exp.get("status") or "").strip().lower()
                if status in {"cancelled", "canceled"}:
                    recent_cancelled += 1
                elif status == "failed":
                    recent_failed += 1

            recent_completed_window = completed[:5]
            recent_zero_s1_runs = 0
            for exp in recent_completed_window:
                gen = exp.get("n_programs_generated") or 0
                passed = exp.get("n_stage1_passed") or 0
                if gen > 0 and passed == 0:
                    recent_zero_s1_runs += 1

            recommendation_evidence = {
                "learning_trend": trend,
                "learning_slope": slope,
                "avg_recent_s1_rate": avg_recent_s1,
                "recent_completed_runs": len(recent_completed_window),
                "recent_zero_s1_runs": recent_zero_s1_runs,
                "recent_cancelled_runs": recent_cancelled,
                "recent_failed_runs": recent_failed,
                "pipeline": pipeline_data,
                "compression": compression_summary,
                "compression_primitives": primitive_effectiveness.get("primitives", []),
                "sparse": sparse_evidence,
                "sparse_coverage": sparse_coverage_summary,
            }

            # --- Try LLM-powered briefing first ---
            aria = get_aria()
            fallback_reason: Optional[str] = None
            llm = aria._get_llm()
            llm_reachable = False
            if llm is None:
                fallback_reason = "llm_not_configured"
            else:
                try:
                    llm_reachable = bool(llm.is_available()) if hasattr(llm, "is_available") else True
                except Exception:
                    llm_reachable = False
                if not llm_reachable:
                    fallback_reason = "llm_unreachable"
            try:
                from .llm.context import build_briefing_context

                # Gather extra context for LLM
                try:
                    active_campaigns = nb.get_active_campaigns()
                    campaign = active_campaigns[0] if active_campaigns else None
                except Exception:
                    campaign = None

                try:
                    dw = analytics.get_current_grammar_weights() or {}
                except Exception:
                    dw = {}

                try:
                    gw = analytics.compute_grammar_weights() or {}
                except Exception:
                    gw = {}

                try:
                    top_programs = nb.conn.execute(
                        "SELECT graph_fingerprint, loss_ratio, novelty_score, tier "
                        "FROM leaderboard ORDER BY composite_score DESC LIMIT 3"
                    ).fetchall()
                    top_progs = [dict(r) for r in top_programs] if top_programs else None
                except Exception:
                    top_progs = None

                try:
                    briefing_context = build_briefing_context(
                        recent_experiments=recent,
                        pipeline_tiers=tiers,
                        learning_trajectory=trajectory,
                        campaign=campaign,
                        grammar_weights=gw,
                        default_weights=dw,
                        top_programs=top_progs,
                        just_completed=just_completed_exp,
                        sparse_coverage=sparse_coverage_data,
                    )
                except Exception:
                    briefing_context = {
                        "pipeline": pipeline_data,
                        "learning": {
                            "trend": trend,
                            "slope": slope,
                            "avg_recent_s1_rate": avg_recent_s1,
                        },
                        "recent_experiments": recent[:5],
                        "campaign": campaign,
                    }

                ai_briefing = aria.generate_briefing(context=briefing_context)
                if ai_briefing and ai_briefing.get("briefing_text"):
                    suggested = ai_briefing.get("suggested_action") or {}
                    normalized_mode = _normalize_briefing_mode(suggested.get("mode"))
                    action_key = _briefing_action_from_mode(normalized_mode)
                    suggested_config = dict(suggested.get("config") or {})
                    hypothesis = suggested.get("hypothesis")
                    if normalized_mode:
                        suggested_config["mode"] = normalized_mode
                    if hypothesis:
                        suggested_config["hypothesis"] = hypothesis
                    # Modes that require result_ids — resolve them automatically
                    if normalized_mode in ("investigation", "validation") and not suggested_config.get("result_ids"):
                        _tier = "screening" if normalized_mode == "investigation" else "investigation"
                        _tier_rows = nb.conn.execute(
                            f"SELECT result_id FROM leaderboard "
                            f"WHERE tier = ? AND {_tier}_passed = 1 "
                            f"ORDER BY {_tier}_loss_ratio ASC LIMIT 20",
                            (_tier,),
                        ).fetchall()
                        _rids = [r["result_id"] for r in _tier_rows if r["result_id"]]
                        suggested_config["result_ids"] = _rids

                    if normalized_mode in ("investigation", "validation"):
                        _requested = _normalize_result_ids(suggested_config.get("result_ids", []))
                        _eligibility = _build_start_mode_eligibility(nb, normalized_mode, _requested)
                        _eligible = _eligibility.get("eligible_result_ids") or []
                        if _eligible:
                            suggested_config["result_ids"] = _eligible
                        else:
                            # No actionable candidates under start-mode guardrails — downgrade to continuous
                            normalized_mode = "continuous"
                            action_key = "continuous"
                            _hypothesis = suggested_config.get("hypothesis")
                            suggested_config = {
                                "mode": "continuous",
                                "model_source": "mixed",
                            }
                            if _hypothesis:
                                suggested_config["hypothesis"] = _hypothesis

                    suggested_config = _augment_sparse_action_config(
                        suggested_config,
                        normalized_mode,
                        sparse_coverage_data,
                    )
                    return jsonify({
                        "briefing": ai_briefing["briefing_text"],
                        "action": action_key or normalized_mode or "continuous",
                        "action_label": _briefing_action_label(
                            normalized_mode, hypothesis),
                        "action_rationale": suggested.get("reasoning", ""),
                        "ai_powered": True,
                        "confidence": ai_briefing.get("confidence", 0.5),
                        "suggested_config": suggested_config or None,
                        "evidence": recommendation_evidence,
                        "data": data_block,
                        "compression_opportunities": compression_opportunities,
                    })
                if fallback_reason is None:
                    fallback_reason = "llm_empty_response"
            except Exception as e:
                logger.warning(f"LLM briefing unavailable, using deterministic: {e}")
                err_msg = str(e)[:120]
                fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"

            # --- Deterministic fallback: build briefing sentences ---
            sentences = []
            if total_exp > 0:
                sentences.append(
                    f"Across {total_exp} experiments, {total_progs:,} architectures "
                    f"have been evaluated with {s1_survivors} stage-1 survivors "
                    f"({s1_survivors / max(total_progs, 1) * 100:.1f}% overall pass rate)."
                )

            # 2. Recent performance
            if avg_recent_s1 is not None:
                n_recent = len(recent_s1_rates)
                sentences.append(
                    f"The last {n_recent} completed experiment{'s' if n_recent != 1 else ''} "
                    f"averaged a {avg_recent_s1 * 100:.1f}% S1 pass rate."
                )

            # 3. Learning trajectory
            if trend == "improving" and slope is not None:
                sentences.append(
                    f"The system is learning — S1 rate is improving at "
                    f"+{abs(slope) * 100:.2f} percentage points per experiment."
                )
            elif trend == "declining" and slope is not None:
                sentences.append(
                    f"S1 rate is declining ({slope * 100:.2f} pp/experiment). "
                    f"Consider switching search strategy or trying evolution mode."
                )
            elif trend == "plateaued":
                sentences.append(
                    "S1 rate has plateaued — a novelty search or evolution run "
                    "could help escape the current local optimum."
                )

            # 4. Pipeline state
            pipeline_parts = []
            if screening > 0:
                pipeline_parts.append(f"{screening} at screening")
            if investigation > 0:
                pipeline_parts.append(f"{investigation} under investigation")
            if validation > 0:
                pipeline_parts.append(f"{validation} in validation")
            if breakthrough > 0:
                pipeline_parts.append(
                    f"{breakthrough} breakthrough{'s' if breakthrough != 1 else ''}"
                )
            if pipeline_parts:
                sentences.append(
                    f"Candidate pipeline: {', '.join(pipeline_parts)}."
                )

            compressed_share = float(compression_summary.get("compressed_test_share") or 0.0)
            compressed_survival = float(compression_summary.get("compressed_survival_rate") or 0.0)
            if compression_summary:
                sentences.append(
                    "Compression coverage: "
                    f"{compressed_share * 100:.1f}% of tested candidates use compact techniques; "
                    f"compressed survival is {compressed_survival * 100:.1f}%."
                )

            sparse_n = int(sparse_evidence.get("n_sparse_programs") or 0)
            if sparse_n > 0:
                sparse_density = float(sparse_evidence.get("avg_density_mean") or 0.0)
                sparse_nm = sparse_evidence.get("avg_nm_compliance")
                sparse_fragment = (
                    f"Sparse telemetry: {sparse_n} runs with mean density {sparse_density * 100:.1f}%"
                )
                if sparse_nm is not None:
                    sparse_fragment += f", N:M compliance {float(sparse_nm) * 100:.1f}%"
                sparse_fragment += "."
                sentences.append(sparse_fragment)

            # 5. Last experiment outcome
            if completed:
                last = completed[0]
                last_s1 = last.get("n_stage1_passed") or 0
                last_gen = last.get("n_programs_generated") or 0
                last_loss = last.get("best_loss_ratio")
                last_id = last.get("experiment_id", "")[:8]
                parts = [
                    f"Last experiment ({last_id}): "
                    f"{last_s1}/{last_gen} passed S1"
                ]
                if last_loss is not None:
                    parts.append(f"best loss {last_loss:.4f}")
                aria_sum = last.get("aria_summary")
                if aria_sum:
                    parts.append(f"— {aria_sum}")
                sentences.append(". ".join(parts) + ".")

            # 6. Data-driven diversity analysis
            try:
                # Op category distribution from learning log
                op_rows = nb.conn.execute(
                    "SELECT op_name, s1_passes, total_uses FROM op_success_rates "
                    "WHERE total_uses >= 5 ORDER BY "
                    "CAST(s1_passes AS REAL) / CAST(total_uses AS REAL) DESC LIMIT 3"
                ).fetchall()
                if op_rows:
                    top_ops = [f"{r['op_name']} ({r['s1_passes']}/{r['total_uses']})"
                               for r in op_rows]
                    sentences.append(
                        f"Top-performing operators: {', '.join(top_ops)}."
                    )

                # Failure mode analysis
                failure_rows = nb.conn.execute(
                    "SELECT stage_at_death, COUNT(*) as cnt FROM program_results "
                    "WHERE stage1_passed = 0 AND stage_at_death IS NOT NULL "
                    "GROUP BY stage_at_death ORDER BY cnt DESC LIMIT 2"
                ).fetchall()
                if failure_rows:
                    failure_parts = [f"{r['stage_at_death']} ({r['cnt']})"
                                     for r in failure_rows]
                    sentences.append(
                        f"Dominant failure stages: {', '.join(failure_parts)}."
                    )

                # Architecture diversity check
                unique_fps = nb.conn.execute(
                    "SELECT COUNT(DISTINCT SUBSTR(graph_fingerprint, 1, 8)) "
                    "FROM leaderboard"
                ).fetchone()[0]
                total_leaderboard = screening + investigation + validation + breakthrough
                if unique_fps is not None and total_leaderboard > 0:
                    diversity_ratio = unique_fps / total_leaderboard
                    if diversity_ratio < 0.5:
                        sentences.append(
                            f"Warning: only {unique_fps} unique architecture "
                            f"families in {total_leaderboard} "
                            f"leaderboard entries — search may be converging."
                        )
            except Exception:
                pass  # Analytics are optional enhancements

            briefing = " ".join(sentences)

            # --- Determine recommended action ---
            action = None
            action_label = None
            action_rationale = None
            screening_result_ids = []

            if breakthrough > 0:
                action = "export_breakthrough"
                action_label = "Export Breakthrough Report"
                action_rationale = (
                    f"{breakthrough} candidate{'s have' if breakthrough != 1 else ' has'} "
                    f"reached breakthrough tier — ready for publication review."
                )
            elif compressed_share < 0.2 and total_exp >= 3:
                action = "compact_synthesis"
                action_label = "Run Compactness-Focused Synthesis"
                action_rationale = (
                    "Compression techniques are underexplored in this campaign. "
                    "Run a compactness-focused synthesis batch to improve model efficiency coverage."
                )
            elif sparse_coverage_summary.get("below_target") and total_exp >= 3:
                sparse_share = float(sparse_coverage_summary.get("sparse_share") or 0.0)
                sparse_survival = float(sparse_coverage_summary.get("sparse_survival_rate") or 0.0)
                target_share = float(sparse_coverage_summary.get("target_share") or 0.15)
                action = "novelty_search"
                action_label = "Run Sparse-Focused Novelty Search"
                action_rationale = (
                    f"Sparse coverage is below target ({sparse_share * 100:.1f}% < {target_share * 100:.0f}%) "
                    f"with {sparse_survival * 100:.1f}% sparse survival. "
                    "Run novelty search with sparse-focused morphological sampling to explore high-upside sparse candidates."
                )
            elif validation > 0 and screening == 0 and investigation == 0:
                action = "monitor_validation"
                action_label = "Review Validation Progress"
                action_rationale = (
                    f"{validation} candidate{'s are' if validation != 1 else ' is'} "
                    f"in validation. Monitor results before starting new experiments."
                )
            elif screening > 0:
                inv_failed = nb.conn.execute(
                    "SELECT COUNT(*) FROM leaderboard "
                    "WHERE tier = 'investigation' AND investigation_passed = 0"
                ).fetchone()[0]
                # Fetch actual result_ids for screening survivors
                screening_rows = nb.conn.execute(
                    "SELECT result_id FROM leaderboard "
                    "WHERE tier = 'screening' AND screening_passed = 1 "
                    "ORDER BY screening_loss_ratio ASC LIMIT 20"
                ).fetchall()
                screening_candidate_ids = [r["result_id"] for r in screening_rows if r["result_id"]]
                screening_result_ids = []
                if screening_candidate_ids:
                    screening_eligibility = _build_start_mode_eligibility(
                        nb,
                        "investigation",
                        screening_candidate_ids,
                    )
                    screening_result_ids = screening_eligibility.get("eligible_result_ids") or []
                if not screening_result_ids:
                    # No actionable screening survivors — fall through to default
                    action = "continuous"
                    action_label = "Continue Research"
                    action_rationale = (
                        "Screening survivors exist but are not currently eligible for investigation reruns. "
                        "Continue generating new architectures."
                    )
                else:
                    action = "investigate"
                    action_label = (
                        f"Investigate {len(screening_result_ids)} Screening "
                        f"Survivor{'s' if len(screening_result_ids) != 1 else ''}"
                    )
                    rationale_parts = [
                        f"{len(screening_result_ids)} candidate{'s' if len(screening_result_ids) != 1 else ''} passed "
                        f"screening and "
                        f"{'are' if len(screening_result_ids) != 1 else 'is'} awaiting deeper investigation"
                    ]
                    if inv_failed > 0:
                        rationale_parts.append(
                            f"({inv_failed} prior investigation"
                            f"{'s' if inv_failed != 1 else ''} "
                            f"failed — fresh candidates may outperform)"
                        )
                    if avg_recent_s1 is not None:
                        rationale_parts.append(
                            f"with recent {avg_recent_s1 * 100:.0f}% hit rate"
                        )
                    action_rationale = ", ".join(rationale_parts) + "."
            elif total_exp == 0:
                action = "start_first"
                action_label = "Run First Experiment"
                action_rationale = (
                    "No experiments yet. Start a mixed continuous run to begin "
                    "exploring the architecture space."
                )
            elif trend == "declining" or (
                len(recent_s1_rates) >= 3
                and all(r == 0 for r in recent_s1_rates[:3])
            ):
                action = "novelty_search"
                action_label = "Try Evolution / Novelty Search"
                action_rationale = (
                    "Recent experiments are underperforming. An evolution or "
                    "novelty-driven search can escape the current local minimum."
                )
            else:
                action = "continuous"
                action_label = "Continue Research"
                action_rationale = (
                    "The pipeline is active and the system is "
                    + ("learning" if trend == "improving" else "exploring")
                    + ". Continue generating and evaluating new architectures."
                )

            # Build deterministic suggested_config from action
            det_mode_map = {
                "investigate": "investigation",
                "continuous": "continuous",
                "start_first": "continuous",
                "novelty_search": "novelty",
                "compact_synthesis": "synthesis",
                "export_breakthrough": None,
                "monitor_validation": None,
            }
            det_mode = det_mode_map.get(action, "continuous")
            if action == "compact_synthesis":
                det_config = {
                    "mode": "synthesis",
                    "model_source": "mixed",
                    "morph_ratio": 0.85,
                    "max_depth": 5,
                    "max_ops": 8,
                    "math_space_weight": 1.8,
                    "residual_prob": 0.85,
                    "n_programs": 80,
                }
            elif action == "novelty_search" and sparse_coverage_summary.get("below_target"):
                det_config = {
                    "mode": "novelty",
                    "model_source": "mixed",
                    "morph_ratio": 0.8,
                    "morph_focus_sparse": True,
                    "morph_sparse_weight_storage": "semi_structured_2_4",
                    "use_synthesized_training": True,
                    "math_space_weight": 2.2,
                    "max_depth": 6,
                    "max_ops": 10,
                    "n_programs": 120,
                }
            elif action == "investigate" and screening_result_ids:
                det_config = {
                    "mode": "investigation",
                    "model_source": "mixed",
                    "result_ids": screening_result_ids,
                }
            else:
                det_config = (
                    {"mode": det_mode, "model_source": "mixed"}
                    if det_mode
                    else None
                )

            det_config = _augment_sparse_action_config(
                det_config,
                det_config.get("mode") if isinstance(det_config, dict) else det_mode,
                sparse_coverage_data,
            ) if isinstance(det_config, dict) else det_config

            return jsonify({
                "briefing": briefing,
                "action": action,
                "action_label": action_label,
                "action_rationale": action_rationale,
                "ai_powered": False,
                "fallback_reason": fallback_reason,
                "suggested_config": det_config,
                "evidence": recommendation_evidence,
                "data": data_block,
                "compression_opportunities": compression_opportunities,
            })
        except Exception as e:
            logger.error(f"Error in /api/strategy/briefing: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Aria Intelligence endpoints ──

    @app.route("/api/aria/cycle-status")
    def api_aria_cycle_status():
        """Get Aria continuous-cycle status (planning/running/analyzing)."""
        runner = _get_runner(notebook_path)
        try:
            return jsonify(runner.get_aria_cycle_status())
        except Exception as e:
            logger.error(f"Error in /api/aria/cycle-status: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/aria/cycle-history")
    def api_aria_cycle_history():
        """Get persisted Aria cycle summaries from notebook live-feed entries."""
        n = request.args.get("n", 100, type=int)
        mode_filter = str(request.args.get("mode") or "").strip().lower()
        status_filter = str(request.args.get("status") or "").strip().lower()
        query_text = str(request.args.get("q") or "").strip().lower()
        output_format = str(request.args.get("format") or "json").strip().lower()
        nb = LabNotebook(notebook_path)
        try:
            entries = _normalize_entries(nb.get_entries(entry_type="live_feed", limit=n * 4))
            history: List[Dict[str, Any]] = []
            for entry in reversed(entries):
                metadata = entry.get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("live_feed_type") != "aria_cycle":
                    continue
                payload = metadata.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                row = dict(payload)
                row["entry_id"] = entry.get("entry_id")
                row["experiment_id"] = entry.get("experiment_id")
                row["entry_timestamp"] = entry.get("timestamp")

                row_mode = str(row.get("mode") or "").strip().lower()
                row_status = str(row.get("status") or "").strip().lower()
                if mode_filter and row_mode != mode_filter:
                    continue
                if status_filter and row_status != status_filter:
                    continue
                if query_text:
                    searchable = " ".join([
                        str(row.get("mode") or ""),
                        str(row.get("status") or ""),
                        str(row.get("reasoning") or ""),
                        str(row.get("error") or ""),
                    ]).lower()
                    if query_text not in searchable:
                        continue

                history.append(row)
                if len(history) >= n:
                    break

            if output_format == "csv":
                fieldnames = [
                    "cycle_index",
                    "mode",
                    "status",
                    "timestamp",
                    "delta_programs",
                    "delta_stage1_survivors",
                    "stage1_survivors",
                    "confidence",
                    "experiment_id",
                    "reasoning",
                    "error",
                ]
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                writer.writeheader()
                for row in history:
                    writer.writerow({k: row.get(k) for k in fieldnames})
                csv_payload = buffer.getvalue()
                return Response(
                    csv_payload,
                    mimetype="text/csv",
                    headers={
                        "Content-Disposition": "attachment; filename=aria_cycle_history.csv",
                    },
                )

            return jsonify(history)
        except Exception as e:
            logger.error(f"Error in /api/aria/cycle-history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/cycle-control", methods=["POST"])
    def api_aria_cycle_control():
        """Control Aria cycle policy: start, pause, resume."""
        runner = _get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        action = str(body.get("action") or "").strip().lower()

        if action == "pause":
            status = runner.pause_aria_cycle()
            return jsonify({"ok": True, "action": "pause", "cycle": status})

        if action == "resume":
            status = runner.resume_aria_cycle()
            return jsonify({"ok": True, "action": "resume", "cycle": status})

        if action == "start":
            if runner.is_running:
                return jsonify({"error": "An experiment is already running"}), 409

            auto_harden = bool(body.get("auto_harden", True))
            config_payload = body.get("config") if isinstance(body.get("config"), dict) else body
            config_payload = dict(config_payload or {})
            config_payload.pop("action", None)
            config_payload.pop("auto_harden", None)
            config_payload["continuous"] = True

            try:
                config = RunConfig.from_dict(config_payload)
                config, prescreen = runner.prescreen_run_config(
                    config,
                    mode="continuous",
                    auto_harden=auto_harden,
                )
                exp_id = runner.start_continuous(config)
                _record_run_trigger(
                    experiment_id=exp_id,
                    source="cycle_control",
                    mode="continuous",
                    details={
                        "endpoint": "/api/aria/cycle-control",
                        "action": "start",
                        "auto_harden": auto_harden,
                    },
                )
                return jsonify({
                    "ok": True,
                    "action": "start",
                    "experiment_id": exp_id,
                    "config": config.to_dict(),
                    "prescreen": prescreen,
                    "cycle": runner.get_aria_cycle_status(),
                })
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error(f"Error starting cycle control: {e}")
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "action must be one of: start, pause, resume"}), 400

    @app.route("/api/aria/recommendation")
    def api_aria_recommendation():
        """Get Aria's experiment recommendation based on all data."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from .llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            suggestion = aria.suggest_experiment(context)
            if suggestion:
                suggestion["evidence_pack"] = build_evidence_pack(
                    nb,
                    analytics=None,
                    recommendation=suggestion,
                    decision_type="api_recommendation",
                    recent_experiments=history,
                )
            return jsonify(suggestion)
        except Exception as e:
            logger.error(f"Error in /api/aria/recommendation: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/strategy")
    def api_aria_strategy():
        """Get Aria's research strategy recommendation."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from .llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            strategy = aria.plan_strategy(context)
            return jsonify({
                "strategy": strategy,
                "available": strategy is not None,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/strategy: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/tools")
    def api_aria_tools():
        """Report Aria tool capabilities and current operational readiness."""
        runner = _get_runner(notebook_path)
        aria = get_aria()
        llm = aria._get_llm()
        llm_available = False
        llm_reason = "not_configured"
        if llm:
            try:
                llm_available = bool(getattr(llm, "is_available", lambda: True)())
                llm_reason = "ok" if llm_available else "unreachable"
            except Exception:
                llm_available = False
                llm_reason = "unreachable"

        cycle_status = runner.get_aria_cycle_status()
        ollama_helper = _local_ollama_helper_status(llm)
        return jsonify({
            "codebase_agent": {
                "spawn_endpoint": True,
                "status_endpoint": True,
                "workspace_scoped": True,
                "allow_write_default": True,
                "execution_first_for_fix_requests": True,
                "small_model_swarm_enabled": True,
                "small_model_swarm_max_workers": _get_local_ollama_settings().get("max_small_workers", 3),
                "simple_task_policy": "prefer_3b_swarm_then_7b",
                "complex_task_policy": "prefer_7b_single",
            },
            "local_ollama_helper": ollama_helper,
            "chat_actions": ["adjust_config", "adjust_grammar", "start_experiment", "edit_file", "spawn_agent"],
            "chat_guardrails": _chat_guardrail_snapshot(window=200),
            "local_context_tools": ["runner.progress", "notebook.get_recent_experiments", "workspace.search"],
            "llm": {
                "available": llm_available,
                "reason": llm_reason,
            },
            "runner": {
                "is_running": bool(runner.is_running),
                "progress_status": (runner.progress.to_dict() or {}).get("status"),
            },
            "run_trigger": _get_run_trigger_snapshot((runner.progress.to_dict() or {}).get("experiment_id")),
            "continuous": {
                "active": bool(cycle_status.get("continuous_active")),
                "phase": cycle_status.get("phase"),
            },
        })

    @app.route("/api/aria/chat/guardrails")
    def api_aria_chat_guardrails():
        """Expose chat action/summarization guardrail metrics."""
        try:
            window = int(request.args.get("window", 200))
        except Exception:
            window = 200
        return jsonify(_chat_guardrail_snapshot(window=window))

    @app.route("/api/aria/agent/spawn", methods=["POST"])
    def api_aria_agent_spawn():
        """Spawn a background Aria codebase agent task for autonomous repair/refactor."""
        body = request.get_json(silent=True) or {}
        goal = str(body.get("goal") or "").strip()
        allow_write = bool(body.get("allow_write", True))

        if not goal:
            return jsonify({"error": "goal is required"}), 400

        spawn_session_id = str(body.get("session_id") or "").strip()
        task = _spawn_code_agent_task(
            goal=goal,
            notebook_path=notebook_path,
            allow_write=allow_write,
            session_id=spawn_session_id,
        )
        return jsonify({"ok": True, "task": task}), 202

    @app.route("/api/aria/agent/status/<task_id>")
    def api_aria_agent_status(task_id: str):
        """Get status/result for a background Aria codebase agent task."""
        task = _code_agent_task_snapshot(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        detail = str(request.args.get("detail") or "").strip().lower()
        if detail != "full":
            task = {
                **task,
                **_summarize_agent_task(task),
            }
        return jsonify({"ok": True, "task": task})

    @app.route("/api/aria/agent/status/<task_id>/summary")
    def api_aria_agent_status_summary(task_id: str):
        """Get concise milestone summary for a background Aria codebase agent task."""
        task = _code_agent_task_snapshot(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        return jsonify({"ok": True, "task": _summarize_agent_task(task)})

    @app.route("/api/aria/diagnose", methods=["POST"])
    def api_aria_diagnose():
        """Run Aria's self-diagnosis: gather analytics, identify issues, apply fixes."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        try:
            analytics_data = {}
            try:
                analytics_data = runner._gather_analytics_data(nb)
            except Exception as exc:
                logger.debug(f"Analytics gather failed during diagnosis: {exc}")

            diagnosed_issues = _diagnose_research_issues(analytics_data, nb)
            actions_applied: List[Dict[str, Any]] = []

            for issue in diagnosed_issues:
                cfg_fix = issue.get("config_fix")
                if cfg_fix and issue.get("action_type") in ("config_fix", "grammar_fix"):
                    try:
                        result = runner.execute_chat_action(cfg_fix, nb)
                        if result.get("status") == "applied":
                            applied_keys = list((result.get("changes") or result.get("weights") or {}).keys())
                            actions_applied.append({
                                "issue": issue["issue"],
                                "action_type": issue["action_type"],
                                "keys_applied": applied_keys,
                            })
                    except Exception as exc:
                        logger.debug(f"Diagnosis config fix failed: {exc}")

            return jsonify({
                "ok": True,
                "issues_found": len(diagnosed_issues),
                "issues": [
                    {
                        "issue": i["issue"],
                        "action_type": i.get("action_type", "info"),
                        "fixed": i["issue"] in [a["issue"] for a in actions_applied],
                    }
                    for i in diagnosed_issues
                ],
                "actions_applied": actions_applied,
                "summary": (
                    f"Found {len(diagnosed_issues)} issue(s), applied {len(actions_applied)} fix(es)."
                    if diagnosed_issues
                    else "No issues found in current analytics."
                ),
            })
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/chat", methods=["POST"])
    def api_aria_chat():
        """Interactive Aria chat response grounded in current research context."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()

        try:
            body = request.get_json(silent=True) or {}
            question = str(body.get("message") or "").strip()
            history_raw = body.get("history") or []
            session_id = str(body.get("session_id") or "").strip()
            spawn_agent = bool(body.get("spawn_agent", False))
            allow_code_writes = bool(body.get("allow_code_writes", True))
            explicit_detailed = _chat_requests_detailed_response(question)
            summary_requested = _chat_requests_summary_response(question)
            brief_response_requested = (
                bool(body.get("brief_response", False))
                or _chat_requests_brief_response(question)
            )
            concise_default_mode = not explicit_detailed and not summary_requested
            brief_response = bool(brief_response_requested or concise_default_mode)
            self_fix_now = _chat_requests_self_fix_now(question)
            fix_request = spawn_agent or _chat_requests_codebase_fix(question) or self_fix_now
            execution_first_mode = bool(fix_request)
            fallback_reason: Optional[str] = None
            local_agent_result: Dict[str, Any] = {"tools_used": [], "summary": "", "code_hits": []}
            code_agent_task: Optional[Dict[str, Any]] = None

            if not question:
                return jsonify({"error": "message is required"}), 400

            if execution_first_mode:
                # Diagnose → Act → Report instead of blindly spawning agents
                analytics_data = {}
                try:
                    analytics_data = runner._gather_analytics_data(nb)
                except Exception as exc:
                    logger.debug(f"Analytics gather failed during diagnosis: {exc}")

                diagnosed_issues = _diagnose_research_issues(analytics_data, nb)
                actions_taken: List[str] = []
                config_keys_applied: List[str] = []

                # Apply config/grammar fixes directly
                for issue in diagnosed_issues:
                    cfg_fix = issue.get("config_fix")
                    if cfg_fix and issue.get("action_type") in ("config_fix", "grammar_fix"):
                        try:
                            result = runner.execute_chat_action(cfg_fix, nb)
                            if result.get("status") == "applied":
                                applied = result.get("changes") or result.get("weights") or {}
                                config_keys_applied.extend(applied.keys())
                                actions_taken.append(issue["issue"])
                        except Exception as exc:
                            logger.debug(f"Config fix failed: {exc}")

                # Decide whether to spawn an agent
                is_vague = self_fix_now  # "fix yourself", "fix what's wrong", etc.
                if not is_vague and fix_request:
                    # Specific fix request — spawn agent with enriched goal
                    diag_context = "; ".join(i["issue"] for i in diagnosed_issues) if diagnosed_issues else "No issues diagnosed"
                    enriched_goal = f"{question}\n\nDiagnosis context: {diag_context}"
                    try:
                        code_agent_task = _spawn_code_agent_task(
                            goal=enriched_goal,
                            notebook_path=notebook_path,
                            allow_write=allow_code_writes,
                            session_id=session_id,
                        )
                    except Exception as exc:
                        logger.warning(f"Unable to spawn codebase agent from chat: {exc}")

                # Build reply
                if diagnosed_issues:
                    reply_parts = []
                    for issue in diagnosed_issues:
                        if issue.get("action_type") == "info":
                            reply_parts.append(issue["issue"] + ".")
                        elif issue["issue"] in actions_taken:
                            reply_parts.append(f"Diagnosed: {issue['issue']}. Applied config fix ({', '.join(issue.get('config_fix', {}).get('changes', issue.get('config_fix', {}).get('weights', {})).keys())}).")
                        else:
                            reply_parts.append(f"Diagnosed: {issue['issue']}.")
                    if code_agent_task:
                        task_id = code_agent_task.get("task_id")
                        reply_parts.append(f"Agent `{task_id}` working on the code-level fix.")
                    concise_reply = " ".join(reply_parts)
                elif code_agent_task:
                    task_id = code_agent_task.get("task_id")
                    concise_reply = f"No config issues found. Spawned agent `{task_id}` to investigate."
                else:
                    concise_reply = "Ran diagnostics — no actionable issues found in current analytics."

                if session_id:
                    try:
                        nb.save_chat_message(
                            session_id=session_id,
                            role="aria",
                            text=concise_reply,
                            label="Aria",
                        )
                    except Exception:
                        pass
                _record_chat_guardrail_event(
                    actionable=bool(actions_taken or code_agent_task),
                    advice_only=not bool(actions_taken or code_agent_task),
                    summary_text=concise_reply,
                )
                return jsonify({
                    "reply": concise_reply,
                    "ai_powered": False,
                    "used_context": True,
                    "fallback_reason": None,
                    "brief_mode": True,
                    "execution_first_mode": True,
                    "advice_only": not bool(actions_taken or code_agent_task),
                    "agent_task": code_agent_task,
                    "actions_taken": actions_taken,
                    "local_tools_used": [],
                    "local_code_hits": [],
                })

            # Persist user message to DB if session_id provided
            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id, role="user", text=question,
                        label="You",
                    )
                except Exception:
                    pass  # Non-fatal — don't block chat on persistence failure

            # Build history lines: prefer DB history when session_id given
            history_lines: List[str] = []
            if session_id:
                try:
                    db_messages = nb.get_chat_history(session_id, limit=12)
                    for msg in db_messages:
                        role = str(msg.get("role") or "user").strip().lower()
                        text = str(msg.get("text") or "").strip()
                        if not text:
                            continue
                        label = "ARIA" if role in {"aria", "assistant"} else role.upper()
                        history_lines.append(f"{label}: {text}")
                except Exception:
                    pass  # Fall through to request-body history
            if not history_lines and isinstance(history_raw, list):
                for entry in history_raw[-8:]:
                    if not isinstance(entry, dict):
                        continue
                    role = str(entry.get("role") or "user").strip().lower()
                    if role not in {"user", "aria", "assistant", "system"}:
                        role = "user"
                    text = str(entry.get("text") or "").strip()
                    if not text:
                        continue
                    label = "ARIA" if role in {"aria", "assistant"} else role.upper()
                    history_lines.append(f"{label}: {text}")

            try:
                analytics_data = runner._gather_analytics_data(nb)
            except Exception:
                analytics_data = {}

            try:
                history = nb.get_recent_experiments(10)
            except Exception:
                history = []

            try:
                past_hypotheses = runner._get_past_hypotheses(nb)
            except Exception:
                past_hypotheses = []

            try:
                from .llm.context import build_rich_context
                context = build_rich_context(
                    results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                             "stage1_passed": 0, "novel_count": 0},
                    analytics_data=analytics_data,
                    history=history,
                    past_hypotheses=past_hypotheses,
                )
            except Exception:
                context = (
                    "Context fallback:\n"
                    f"- Recent experiments: {len(history)}\n"
                    f"- Analytics keys: {len(analytics_data) if isinstance(analytics_data, dict) else 0}\n"
                    f"- Past hypotheses: {len(past_hypotheses) if isinstance(past_hypotheses, list) else 0}"
                )

            local_agent_result = _run_local_chat_agent(
                question=question,
                runner=runner,
                nb=nb,
                notebook_path=notebook_path,
                enable_code_tools=True,
            )
            if local_agent_result.get("summary"):
                context = f"{context}\n\n{local_agent_result['summary']}"
            # Cap context to ~2000 chars to prevent LLM from echoing data back
            if len(context) > 2000:
                context = context[:2000] + "\n[context truncated]"
            if code_agent_task:
                task_id = code_agent_task.get("task_id")
                context = (
                    f"{context}\n\n"
                    "Autonomous codebase agent was spawned for this request:\n"
                    f"- task_id={task_id}\n"
                    f"- allow_write={bool(code_agent_task.get('allow_write'))}\n"
                    "- can inspect and patch any workspace file with safety checks"
                )

            llm = aria._get_llm()
            if llm:
                try:
                    if hasattr(llm, "is_available") and not llm.is_available():
                        fallback_reason = "llm_unreachable"
                except Exception:
                    fallback_reason = "llm_unreachable"
                try:
                    from .llm.prompts import SYSTEM_PROMPT, CHAT_PROMPT
                    prompt_question = question
                    prompt_question = (
                        f"{prompt_question}\n\n"
                        "STRICT CONTRACT:\n"
                        "1) Return only typed actions using ```action JSON blocks.\n"
                        "2) Allowed type values: adjust_config, adjust_grammar, start_experiment, edit_file, spawn_agent.\n"
                        "3) Do not output execution plans, pseudo-code, or non-action code blocks.\n"
                        "4) If no action is appropriate, return one short plain sentence only."
                    )
                    # Keep only last 5 history lines, each capped at 100 chars
                    trimmed_history = [
                        (line[:100] + "..." if len(line) > 100 else line)
                        for line in history_lines[-5:]
                    ]
                    prompt = CHAT_PROMPT.format(
                        context=context,
                        history="\n".join(trimmed_history) if trimmed_history else "(none)",
                        question=prompt_question,
                    )
                    max_tokens = 200 if brief_response else 384
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens)
                    aria._track_cost(resp)
                    text = (resp.text or "").strip()
                    if text:
                        parsed = _parse_action_contract_response(text)
                        actions = parsed.get("actions") or []
                        advice_only = bool(parsed.get("advice_only"))
                        actions_taken = []
                        for action in actions:
                            try:
                                if str(action.get("type") or "") == "spawn_agent":
                                    goal = str(action.get("goal") or "").strip() or question
                                    if goal:
                                        # Route technical planning details to local planner context
                                        context_lines = [f"Original request: {question}"]
                                        local_summary = str(local_agent_result.get("summary") or "").strip()
                                        if local_summary:
                                            context_lines.append(f"Local evidence summary: {local_summary}")
                                        hits = local_agent_result.get("code_hits") or []
                                        if hits:
                                            top_hits = ", ".join(
                                                f"{str(h.get('path') or '?')}:{int(h.get('line') or 0)}"
                                                for h in hits[:5]
                                            )
                                            context_lines.append(f"Relevant code hits: {top_hits}")
                                        try:
                                            ws = _chat_workspace_root(notebook_path)
                                            idx_hits = _query_file_index(goal, ws, max_results=6)
                                            if idx_hits:
                                                files_hint = ", ".join(h["rel_path"] for h in idx_hits[:6])
                                                context_lines.append(f"Indexed files: {files_hint}")
                                        except Exception:
                                            pass
                                        history_tail = " | ".join(history_lines[-3:]) if history_lines else ""
                                        if history_tail:
                                            context_lines.append(f"Chat context: {history_tail}")
                                        goal = f"{goal}\n\nTechnical plan context:\n- " + "\n- ".join(context_lines)
                                        agent_task = _spawn_code_agent_task(
                                            goal=goal,
                                            notebook_path=notebook_path,
                                            allow_write=allow_code_writes,
                                            session_id=session_id,
                                        )
                                        result = {
                                            "status": "spawned",
                                            "task_id": agent_task.get("task_id"),
                                            "goal": _truncate_summary(str(action.get("goal") or question), 120),
                                        }
                                        if not code_agent_task:
                                            code_agent_task = agent_task
                                    else:
                                        result = {"status": "error", "error": "No goal provided"}
                                else:
                                    result = runner.execute_chat_action(action, nb)
                                if (
                                    str(action.get("type") or "").strip() == "start_experiment"
                                    and str(result.get("status") or "").strip() == "started"
                                    and result.get("experiment_id")
                                ):
                                    _record_run_trigger(
                                        experiment_id=str(result.get("experiment_id")),
                                        source="chat_action",
                                        mode=str(result.get("mode") or "single").strip() or "single",
                                        details={
                                            "endpoint": "/api/aria/chat",
                                            "session_id": session_id or None,
                                        },
                                    )
                                actions_taken.append({
                                    "type": action.get("type"),
                                    "status": result.get("status", "unknown"),
                                    "detail": result,
                                })
                            except Exception as action_err:
                                actions_taken.append({
                                    "type": action.get("type"),
                                    "status": "error",
                                    "detail": {"error": str(action_err)},
                                })
                        actionable = any(
                            str(a.get("status") or "").lower() in {"applied", "started", "spawned"}
                            for a in actions_taken
                        )
                        if actionable:
                            action_types = ", ".join(
                                sorted({str(a.get("type") or "?") for a in actions_taken})
                            )
                            status_bits = []
                            for item in actions_taken:
                                t = str(item.get("type") or "?")
                                s = str(item.get("status") or "unknown")
                                status_bits.append(f"{t}:{s}")
                            reply_text = _truncate_summary(
                                f"Action started: {action_types}. "
                                f"Status: {'; '.join(status_bits[:4])}. "
                                f"Next checkpoint: monitor task progress and report completion.",
                                240,
                            )
                        else:
                            summary = str(parsed.get("summary") or "").strip()
                            reply_text = _truncate_summary(
                                summary or "advice_only: no valid executable actions were produced.",
                                220,
                            )
                            advice_only = True
                        if code_agent_task and code_agent_task.get("task_id"):
                            snap = _summarize_agent_task(code_agent_task)
                            reply_text = _truncate_summary(
                                f"{reply_text} Task {snap.get('task_id')} queued ({snap.get('milestone_summary')}).",
                                260,
                            )
                        _record_chat_guardrail_event(
                            actionable=actionable,
                            advice_only=advice_only,
                            summary_text=reply_text,
                        )
                        if session_id:
                            try:
                                nb.save_chat_message(
                                    session_id=session_id, role="aria",
                                    text=reply_text, label="Aria",
                                )
                            except Exception:
                                pass
                        return jsonify({
                            "reply": reply_text,
                            "ai_powered": True,
                            "used_context": True,
                            "fallback_reason": None,
                            "brief_mode": brief_response,
                            "agent_task": code_agent_task,
                            "actions_taken": actions_taken,
                            "advice_only": advice_only,
                            "local_tools_used": local_agent_result.get("tools_used", []),
                            "local_code_hits": [
                                {
                                    "path": hit.get("path"),
                                    "abs_path": hit.get("abs_path"),
                                    "line": hit.get("line"),
                                    "score": hit.get("score"),
                                }
                                for hit in local_agent_result.get("code_hits", [])
                            ],
                        })
                    fallback_reason = fallback_reason or "llm_empty_response"
                except Exception as e:
                    logger.warning(f"Aria chat LLM failed, using fallback: {e}")
                    err_msg = str(e)[:120]
                    fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"
            else:
                fallback_reason = "llm_not_configured"

            # Fallback: no LLM available. Keep it short.
            if code_agent_task:
                task_id = code_agent_task.get("task_id")
                fallback_reply = f"Agent `{task_id}` is working on it. No LLM available for chat right now."
            elif summary_requested:
                fallback_reply = "LLM unavailable. Check Strategy Advisor for current recommendations."
            else:
                fallback_reply = "LLM unavailable. Try a fix-intent request (e.g. 'fix X') to spawn an agent."
            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id, role="aria",
                        text=fallback_reply,
                        label=f"Aria (fallback: {fallback_reason})",
                    )
                except Exception:
                    pass
            _record_chat_guardrail_event(
                actionable=False,
                advice_only=True,
                summary_text=fallback_reply,
            )
            return jsonify({
                "reply": fallback_reply,
                "ai_powered": False,
                "used_context": True,
                "fallback_reason": fallback_reason,
                "brief_mode": brief_response,
                "advice_only": True,
                "agent_task": code_agent_task,
                "local_tools_used": local_agent_result.get("tools_used", []),
                "local_code_hits": [
                    {
                        "path": hit.get("path"),
                        "abs_path": hit.get("abs_path"),
                        "line": hit.get("line"),
                        "score": hit.get("score"),
                    }
                    for hit in local_agent_result.get("code_hits", [])
                ],
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/chat: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/chat/history")
    def api_aria_chat_history():
        """Load chat history from the database."""
        nb = LabNotebook(notebook_path)
        try:
            session_id = request.args.get("session_id", "default")
            limit = min(int(request.args.get("limit", 50)), 200)
            messages = nb.get_chat_history(session_id, limit=limit)
            return jsonify({"messages": messages, "session_id": session_id})
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/chat/message", methods=["POST"])
    def api_aria_chat_message():
        """Save a single chat message to the database."""
        nb = LabNotebook(notebook_path)
        try:
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id", "default")
            role = body.get("role", "user")
            text = body.get("text", "")
            label = body.get("label")
            message_id = body.get("message_id")
            metadata = body.get("metadata")
            if not text:
                return jsonify({"error": "text is required"}), 400
            mid = nb.save_chat_message(
                session_id=session_id, role=role, text=text,
                label=label, message_id=message_id, metadata=metadata,
            )
            return jsonify({"message_id": mid, "saved": True})
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/message: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    def _estimate_tokens(text: str) -> int:
        """Rough token count: ~4 chars per token."""
        return len(text or "") // 4

    @app.route("/api/aria/chat/compact", methods=["POST"])
    def api_aria_chat_compact():
        """Compact older chat messages into a summary when token budget exceeded."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id", "default")
            token_budget = int(body.get("token_budget", 4000))

            messages = nb.get_chat_history(session_id, limit=200)
            if not messages:
                return jsonify({"compacted": False, "reason": "no messages"})

            # Calculate tokens for active messages
            total_tokens = sum(_estimate_tokens(m.get("text", "")) for m in messages)
            if total_tokens <= token_budget:
                return jsonify({"compacted": False, "reason": "within budget",
                                "total_tokens": total_tokens})

            # Find oldest messages that exceed the budget
            # Keep recent messages within budget, compact the rest
            keep_tokens = 0
            keep_from = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = _estimate_tokens(messages[i].get("text", ""))
                if keep_tokens + msg_tokens > token_budget * 0.7:  # Keep 70% budget for recent
                    keep_from = i + 1
                    break
                keep_tokens += msg_tokens

            to_compact = messages[:keep_from]
            if not to_compact:
                return jsonify({"compacted": False, "reason": "nothing to compact"})

            # Build text for summarization
            compact_text = "\n".join(
                f"{m.get('role', 'unknown').upper()}: {m.get('text', '')}"
                for m in to_compact
            )

            # Try LLM summarization, fall back to first-sentence extraction
            summary_text = None
            llm = aria._get_llm()
            if llm:
                try:
                    from .llm.prompts import SYSTEM_PROMPT, CHAT_COMPACTION_PROMPT
                    prompt = CHAT_COMPACTION_PROMPT.format(messages=compact_text[:3000])
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=300)
                    aria._track_cost(resp)
                    summary_text = (resp.text or "").strip()
                except Exception as e:
                    logger.warning(f"Chat compaction LLM failed: {e}")

            if not summary_text:
                # Fallback: extract first sentence from each message
                lines = []
                for m in to_compact:
                    text = (m.get("text") or "").strip()
                    first_sentence = text.split(".")[0].strip()
                    if first_sentence and len(first_sentence) > 10:
                        role = m.get("role", "?").upper()
                        lines.append(f"- [{role}] {first_sentence}.")
                    if len(lines) >= 5:
                        break
                summary_text = "\n".join(lines) if lines else "Previous conversation summarized."

            # Save summary message
            import uuid as _uuid
            summary_id = f"summary-{_uuid.uuid4().hex[:8]}"
            compact_ids = [m["message_id"] for m in to_compact if m.get("message_id")]

            nb.save_chat_message(
                session_id=session_id, role="system",
                text=summary_text, label="Summary",
                message_id=summary_id,
                metadata={"compaction": True, "summarized_count": len(compact_ids)},
            )
            nb.mark_messages_compacted(compact_ids, summary_id)

            return jsonify({
                "compacted": True,
                "messages_compacted": len(compact_ids),
                "summary_id": summary_id,
                "summary_tokens": _estimate_tokens(summary_text),
                "original_tokens": total_tokens,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/compact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/system/status")
    def api_system_status():
        """Report system status: CUDA, LLM, database, runner state."""
        import torch
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            # CUDA info
            cuda_available = torch.cuda.is_available()
            cuda_info = {}
            if cuda_available:
                try:
                    cuda_info = {
                        "device_name": torch.cuda.get_device_name(0),
                        "device_count": torch.cuda.device_count(),
                    }
                    mem = torch.cuda.mem_get_info(0)
                    cuda_info["memory_free_gb"] = round(mem[0] / 1e9, 1)
                    cuda_info["memory_total_gb"] = round(mem[1] / 1e9, 1)
                except Exception as e:
                    logger.warning("Failed collecting CUDA details: %s", e)

            # LLM backend
            llm = aria._get_llm()
            llm_reachable = False
            if llm is not None:
                try:
                    llm_reachable = bool(llm.is_available()) if hasattr(llm, "is_available") else True
                except Exception:
                    llm_reachable = False
            llm_info = {
                "available": llm_reachable,
                "configured": llm is not None,
                "backend": llm.name if llm else None,
            }

            # Database stats
            summary = nb.get_dashboard_summary()
            db_info = {
                "path": notebook_path,
                "total_experiments": summary.get("total_experiments", 0),
                "total_programs": summary.get("total_programs_evaluated", 0),
            }

            return jsonify({
                "cuda": {"available": cuda_available, **cuda_info},
                "llm": llm_info,
                "database": db_info,
                "is_running": runner.is_running,
            })
        except Exception as e:
            logger.error(f"Error in /api/system/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/validate", methods=["POST"])
    def api_validate_pipeline():
        """Validate the synthesis pipeline by generating and testing programs."""
        body = request.get_json(silent=True) or {}
        n = body.get("n", 5)
        n = min(n, 20)  # cap at 20

        try:
            from ..synthesis.grammar import GrammarConfig, batch_generate
            from ..synthesis.compiler import compile_model
            from ..synthesis.validator import validate_graph
            from ..eval.sandbox import safe_eval

            grammar = GrammarConfig(model_dim=256, max_depth=8, max_ops=12)
            graphs = batch_generate(n, grammar)

            generated = len(graphs)
            compiled = 0
            passed_s0 = 0
            errors = []

            for graph in graphs:
                val = validate_graph(graph)
                if not val.valid:
                    errors.append(f"validation: {val.errors[0] if val.errors else 'unknown'}")
                    continue

                try:
                    model = compile_model(
                        [graph] * 2,
                        vocab_size=1000,
                        max_seq_len=128,
                    )
                    compiled += 1

                    result = safe_eval(model, batch_size=1, seq_len=64,
                                       vocab_size=1000, device="cpu")
                    if result.passed:
                        passed_s0 += 1
                    else:
                        errors.append(f"sandbox: {result.error or 'failed'}")
                    del model
                except Exception as e:
                    errors.append(f"compile: {str(e)[:60]}")

            healthy = compiled > 0 and passed_s0 > 0
            return jsonify({
                "generated": generated,
                "compiled": compiled,
                "passed_s0": passed_s0,
                "errors": errors[:5],
                "healthy": healthy,
            })
        except Exception as e:
            logger.error(f"Error in pipeline validation: {e}")
            return jsonify({
                "generated": 0,
                "compiled": 0,
                "passed_s0": 0,
                "errors": [str(e)],
                "healthy": False,
            })

    # ── Campaign endpoints ──

    @app.route("/api/campaigns")
    def api_campaigns():
        """List all campaigns with summary stats."""
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC"
            ).fetchall()
            campaigns = []
            for r in rows:
                d = dict(r)
                # Add summary stats
                d["n_experiments"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_hypotheses"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_decisions"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                campaigns.append(d)
            return jsonify(campaigns)
        except Exception as e:
            logger.error(f"Error in /api/campaigns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>")
    def api_campaign_detail(campaign_id):
        """Full campaign detail with experiments, hypotheses, decisions."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404
            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = _normalize_hypotheses(nb.get_campaign_hypotheses(campaign_id))
            decisions = nb.get_campaign_decisions(campaign_id)
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=campaign,
                experiments=experiments,
                hypotheses=hypotheses,
                decisions=decisions,
            )
            return jsonify({
                "campaign": campaign,
                "experiments": experiments,
                "hypotheses": hypotheses,
                "decisions": decisions,
                "success_criteria_tracker": success_criteria_tracker,
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/report")
    def api_campaign_report(campaign_id):
        """Compiled campaign report (LLM-generated narrative)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404

            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = _normalize_hypotheses(nb.get_campaign_hypotheses(campaign_id))
            decisions = nb.get_campaign_decisions(campaign_id)
            knowledge = nb.get_knowledge()
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=campaign,
                experiments=experiments,
                hypotheses=hypotheses,
                decisions=decisions,
            )

            from .llm.context import build_campaign_report_context
            context = build_campaign_report_context(
                campaign, experiments, hypotheses, decisions, knowledge)
            report = aria.compile_campaign_report(
                campaign, experiments, hypotheses, decisions, knowledge,
                context=context)

            return jsonify({
                "campaign": campaign,
                "report": report,
                "stats": {
                    "n_experiments": len(experiments),
                    "n_hypotheses": len(hypotheses),
                    "n_confirmed": sum(1 for h in hypotheses if h.get("status") == "confirmed"),
                    "n_refuted": sum(1 for h in hypotheses if h.get("status") == "refuted"),
                    "n_decisions": len(decisions),
                },
                "success_criteria_tracker": success_criteria_tracker,
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}/report: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/hypotheses")
    def api_campaign_hypotheses(campaign_id):
        """Hypothesis chain for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            return jsonify(hypotheses)
        except Exception as e:
            logger.error(f"Error in campaign hypotheses: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/decisions")
    def api_campaign_decisions(campaign_id):
        """Decision log for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            decisions = nb.get_campaign_decisions(campaign_id)
            return jsonify(decisions)
        except Exception as e:
            logger.error(f"Error in campaign decisions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns", methods=["POST"])
    def api_create_campaign():
        """Create a new campaign manually."""
        body = request.get_json(silent=True) or {}
        title = body.get("title", "")
        objective = body.get("objective", "")
        success_criteria = body.get("success_criteria", "")

        if not title or not objective or not success_criteria:
            return jsonify({"error": "title, objective, and success_criteria required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            campaign_id = nb.create_campaign(
                title=title, objective=objective,
                success_criteria=success_criteria,
                parent_id=body.get("parent_campaign_id"),
            )
            return jsonify({
                "campaign_id": campaign_id,
                "status": "created",
            })
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/pause", methods=["POST"])
    def api_pause_campaign(campaign_id):
        """Pause a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            nb.update_campaign(campaign_id, status="paused")
            return jsonify({"status": "paused"})
        except Exception as e:
            logger.error(f"Error pausing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/complete", methods=["POST"])
    def api_complete_campaign(campaign_id):
        """Complete a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            nb.update_campaign(campaign_id, status="completed",
                               completed_at=time.time())
            runner = _get_runner(notebook_path)
            runner._emit_event("campaign_completed", {
                "campaign_id": campaign_id,
                "title": (campaign or {}).get("title", ""),
            })
            return jsonify({"status": "completed"})
        except Exception as e:
            logger.error(f"Error completing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Hypothesis endpoints ──

    @app.route("/api/hypotheses/<hypothesis_id>/chain")
    def api_hypothesis_chain(hypothesis_id):
        """Hypothesis lineage chain."""
        nb = LabNotebook(notebook_path)
        try:
            chain = nb.get_hypothesis_chain(hypothesis_id)
            return jsonify(chain)
        except Exception as e:
            logger.error(f"Error in hypothesis chain: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Knowledge base endpoints ──

    @app.route("/api/knowledge")
    def api_knowledge():
        """Knowledge base entries, optionally filtered by category."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_knowledge(category=category)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in /api/knowledge: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/knowledge/search")
    def api_knowledge_search():
        """Search knowledge base."""
        q = request.args.get("q", "")
        if not q:
            return jsonify([])
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.search_knowledge(q)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in knowledge search: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/knowledge/backfill", methods=["POST"])
    def api_knowledge_backfill():
        """Backfill missing knowledge categories from measured experiment data."""
        nb = LabNotebook(notebook_path)
        try:
            result = _backfill_knowledge_from_real_data(nb)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in /api/knowledge/backfill: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Action Queue endpoints ──

    @app.route("/api/actions")
    def api_actions():
        """Aggregated prioritized action list for the dashboard."""
        nb = LabNotebook(notebook_path)
        try:
            actions = _compute_action_queue(nb)
            return jsonify(actions)
        except Exception as e:
            logger.error(f"Error in /api/actions: {e}")
            return jsonify([]), 500
        finally:
            nb.close()

    @app.route("/api/actions/<action_id>/dismiss", methods=["POST"])
    def api_action_dismiss(action_id):
        """Dismiss an action card (ephemeral, resets on server restart)."""
        clean_id = str(action_id or "").strip()[:64]
        if not clean_id:
            return jsonify({"error": "Missing action_id"}), 400
        _DISMISSED_ACTIONS.add(clean_id)
        return jsonify({"dismissed": clean_id, "total_dismissed": len(_DISMISSED_ACTIONS)})

    @app.route("/api/actions/<action_id>/approve", methods=["POST"])
    def api_action_approve(action_id):
        """User approves a pending autonomous action."""
        try:
            autonomy, store = _get_autonomy(notebook_path)
            action = autonomy.approve(action_id)
            if not action:
                return jsonify({"error": "Action not found or not pending"}), 404
            store.update_status(
                action_id, action.status,
                executed_at=action.executed_at,
                undo_snapshot=action.undo_snapshot,
            )
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error approving action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/actions/<action_id>/undo", methods=["POST"])
    def api_action_undo(action_id):
        """Undo a recently executed autonomous action (within 5 min window)."""
        try:
            autonomy, store = _get_autonomy(notebook_path)
            action = autonomy.undo(action_id)
            if not action:
                return jsonify({"error": "Action not found or undo window expired"}), 404
            store.update_status(action_id, action.status)
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error undoing action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    # ── Aria Autonomy endpoints ─────────────────────────────────────

    @app.route("/api/aria/autonomy")
    def api_aria_autonomy_get():
        """Get current autonomy trust level and per-decision-type settings."""
        try:
            autonomy, _ = _get_autonomy(notebook_path)
            return jsonify(autonomy.get_config())
        except Exception as e:
            logger.error(f"Error in GET /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/aria/autonomy", methods=["PUT"])
    def api_aria_autonomy_put():
        """Update autonomy trust level or per-decision-type overrides."""
        try:
            autonomy, _ = _get_autonomy(notebook_path)
            body = request.get_json(force=True, silent=True) or {}
            config = autonomy.update_config(body)
            return jsonify(config)
        except Exception as e:
            logger.error(f"Error in PUT /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/aria/activity")
    def api_aria_activity():
        """Get Aria's recent autonomous decisions and their outcomes."""
        try:
            autonomy, store = _get_autonomy(notebook_path)
            limit = request.args.get("limit", 20, type=int)
            # Combine in-memory actions with persisted ones
            memory_actions = autonomy.get_recent_activity(limit)
            stored_actions = store.get_recent(limit)

            # Merge: prefer in-memory (fresher), fill with stored
            seen_ids = {a["action_id"] for a in memory_actions}
            merged = list(memory_actions)
            for sa in stored_actions:
                if sa["action_id"] not in seen_ids:
                    merged.append(sa)
                    seen_ids.add(sa["action_id"])

            merged.sort(key=lambda a: a.get("created_at", 0), reverse=True)
            return jsonify(merged[:limit])
        except Exception as e:
            logger.error(f"Error in /api/aria/activity: {e}")
            return jsonify([]), 500

    return app


class _SseLogFilter(logging.Filter):
    """Suppress noisy werkzeug logs for frequently-polled endpoints."""

    _SUPPRESSED = (
        "GET /api/events",
        "GET /api/aria/cycle-status",
        "GET /api/dashboard",
        "GET /api/analytics/learning-trajectory",
        "GET /api/leaderboard",
        "GET /api/analytics/math-family-coverage",
        "GET /static/",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(s in msg for s in self._SUPPRESSED):
            return False
        return True


def _setup_logging(log_dir: Optional[str] = None):
    """Configure logging with console and file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Suppress SSE endpoint spam from werkzeug
    logging.getLogger("werkzeug").addFilter(_SseLogFilter())

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = str(Path(__file__).parent.parent)
    log_path = Path(log_dir) / "aria_dashboard.log"
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024,  # 2MB
            backupCount=1,
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")
    except Exception as e:
        logger.warning(f"Could not create log file at {log_path}: {e}")


def run_server(
    notebook_path: str = "research/lab_notebook.db",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Run the API server."""
    _setup_logging()
    app = create_app(notebook_path)
    logger.info(f"Starting Aria's Dashboard API on http://{host}:{port}")
    print(f"Starting Aria's Dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
