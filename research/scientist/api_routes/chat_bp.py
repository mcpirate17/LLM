"""Aria chat and agent route registration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from flask import jsonify, request
from .deps import ApiRouteContext
from ..persona import get_aria
from ..code_agent import _spawn_code_agent_task
from ._helpers import (
    get_runner,
    record_run_trigger,
)
from ._utils import with_notebook_context
from ._strategy_diagnostics import diagnose_research_issues
from ._chat import (
    chat_requests_detailed_response,
    chat_requests_summary_response,
    chat_requests_brief_response,
    chat_requests_self_fix_now,
    chat_requests_codebase_fix,
    record_chat_guardrail_event,
    chat_guardrail_snapshot,
    code_agent_task_snapshot,
    summarize_agent_task,
    run_local_chat_agent,
    chat_workspace_root,
    query_file_index,
    parse_action_contract_response,
    truncate_summary,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


def register_chat_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/aria/chat/guardrails")
    def api_aria_chat_guardrails():
        """Expose chat action/summarization guardrail metrics."""
        try:
            window = int(request.args.get("window", 200))
        except ValueError:
            logger.debug(
                "Invalid chat guardrails window=%r; defaulting to 200",
                request.args.get("window"),
            )
            window = 200
        return jsonify(chat_guardrail_snapshot(window=window))

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
        task = code_agent_task_snapshot(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        detail = str(request.args.get("detail") or "").strip().lower()
        if detail != "full":
            task = {
                **task,
                **summarize_agent_task(task),
            }
        return jsonify({"ok": True, "task": task})

    @app.route("/api/aria/agent/status/<task_id>/summary")
    def api_aria_agent_status_summary(task_id: str):
        """Get concise milestone summary for a background Aria codebase agent task."""
        task = code_agent_task_snapshot(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        return jsonify({"ok": True, "task": summarize_agent_task(task)})

    @app.route("/api/aria/diagnose", methods=["POST"])
    @wnb
    def api_aria_diagnose(nb=None):
        """Run Aria's self-diagnosis: gather analytics, identify issues, apply fixes."""
        runner = get_runner(notebook_path)
        analytics_data = {}
        try:
            analytics_data = runner._gather_analytics_data(nb)
        except Exception as exc:
            logger.debug(f"Analytics gather failed during diagnosis: {exc}")

        diagnosed_issues = diagnose_research_issues(analytics_data, nb)
        actions_applied: List[Dict[str, Any]] = []

        for issue in diagnosed_issues:
            cfg_fix = issue.get("config_fix")
            if cfg_fix and issue.get("action_type") in (
                "config_fix",
                "grammar_fix",
            ):
                try:
                    result = runner.execute_chat_action(cfg_fix, nb)
                    if result.get("status") == "applied":
                        applied_keys = list(
                            (
                                result.get("changes") or result.get("weights") or {}
                            ).keys()
                        )
                        actions_applied.append(
                            {
                                "issue": issue["issue"],
                                "action_type": issue["action_type"],
                                "keys_applied": applied_keys,
                            }
                        )
                except Exception as exc:
                    logger.debug(f"Diagnosis config fix failed: {exc}")

        return jsonify(
            {
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
            }
        )

    @app.route("/api/aria/chat", methods=["POST"])
    @wnb
    def api_aria_chat(nb=None):
        """Interactive Aria chat response grounded in current research context."""
        runner = get_runner(notebook_path)
        aria = get_aria()

        body = request.get_json(silent=True) or {}
        question = str(body.get("message") or "").strip()
        history_raw = body.get("history") or []
        session_id = str(body.get("session_id") or "").strip()
        spawn_agent = bool(body.get("spawn_agent", False))
        allow_code_writes = bool(body.get("allow_code_writes", True))
        explicit_detailed = chat_requests_detailed_response(question)
        summary_requested = chat_requests_summary_response(question)
        brief_response_requested = bool(
            body.get("brief_response", False)
        ) or chat_requests_brief_response(question)
        concise_default_mode = not explicit_detailed and not summary_requested
        brief_response = bool(brief_response_requested or concise_default_mode)
        self_fix_now = chat_requests_self_fix_now(question)
        fix_request = (
            spawn_agent or chat_requests_codebase_fix(question) or self_fix_now
        )
        execution_first_mode = bool(fix_request)
        fallback_reason: Optional[str] = None
        local_agent_result: Dict[str, Any] = {
            "tools_used": [],
            "summary": "",
            "code_hits": [],
        }
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

            diagnosed_issues = diagnose_research_issues(analytics_data, nb)
            actions_taken: List[str] = []
            config_keys_applied: List[str] = []

            # Apply config/grammar fixes directly
            for issue in diagnosed_issues:
                cfg_fix = issue.get("config_fix")
                if cfg_fix and issue.get("action_type") in (
                    "config_fix",
                    "grammar_fix",
                ):
                    try:
                        result = runner.execute_chat_action(cfg_fix, nb)
                        if result.get("status") == "applied":
                            applied = (
                                result.get("changes") or result.get("weights") or {}
                            )
                            config_keys_applied.extend(applied.keys())
                            actions_taken.append(issue["issue"])
                    except Exception as exc:
                        logger.debug(f"Config fix failed: {exc}")

            # Decide whether to spawn an agent
            is_vague = self_fix_now  # "fix yourself", "fix what's wrong", etc.
            if not is_vague and fix_request:
                # Specific fix request — spawn agent with enriched goal
                diag_context = (
                    "; ".join(i["issue"] for i in diagnosed_issues)
                    if diagnosed_issues
                    else "No issues diagnosed"
                )
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
                        reply_parts.append(
                            f"Diagnosed: {issue['issue']}. Applied config fix ({', '.join(issue.get('config_fix', {}).get('changes', issue.get('config_fix', {}).get('weights', {})).keys())})."
                        )
                    else:
                        reply_parts.append(f"Diagnosed: {issue['issue']}.")
                if code_agent_task:
                    task_id = code_agent_task.get("task_id")
                    reply_parts.append(
                        f"Agent `{task_id}` working on the code-level fix."
                    )
                concise_reply = " ".join(reply_parts)
            elif code_agent_task:
                task_id = code_agent_task.get("task_id")
                concise_reply = (
                    f"No config issues found. Spawned agent `{task_id}` to investigate."
                )
            else:
                concise_reply = (
                    "Ran diagnostics — no actionable issues found in current analytics."
                )

            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id,
                        role="aria",
                        text=concise_reply,
                        label="Aria",
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to persist concise Aria reply for session_id=%s: %s",
                        session_id,
                        exc,
                    )
            record_chat_guardrail_event(
                actionable=bool(actions_taken or code_agent_task),
                advice_only=not bool(actions_taken or code_agent_task),
                summary_text=concise_reply,
            )
            return jsonify(
                {
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
                }
            )

        # Persist user message to DB if session_id provided
        if session_id:
            try:
                nb.save_chat_message(
                    session_id=session_id,
                    role="user",
                    text=question,
                    label="You",
                )
            except Exception as exc:
                logger.debug(
                    "Failed to persist user chat message for session_id=%s: %s",
                    session_id,
                    exc,
                )

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
            except Exception as exc:
                logger.debug(
                    "Failed to load DB chat history for session_id=%s; falling back to request history: %s",
                    session_id,
                    exc,
                )
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
            from ..llm.context_experiment import build_rich_context

            context = build_rich_context(
                results={
                    "total": 0,
                    "stage0_passed": 0,
                    "stage05_passed": 0,
                    "stage1_passed": 0,
                    "novel_count": 0,
                },
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

        local_agent_result = run_local_chat_agent(
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
                from ..llm.prompts import SYSTEM_PROMPT, CHAT_PROMPT

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
                    parsed = parse_action_contract_response(text)
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
                                    local_summary = str(
                                        local_agent_result.get("summary") or ""
                                    ).strip()
                                    if local_summary:
                                        context_lines.append(
                                            f"Local evidence summary: {local_summary}"
                                        )
                                    hits = local_agent_result.get("code_hits") or []
                                    if hits:
                                        top_hits = ", ".join(
                                            f"{str(h.get('path') or '?')}:{int(h.get('line') or 0)}"
                                            for h in hits[:5]
                                        )
                                        context_lines.append(
                                            f"Relevant code hits: {top_hits}"
                                        )
                                    try:
                                        ws = chat_workspace_root(notebook_path)
                                        idx_hits = query_file_index(
                                            goal, ws, max_results=6
                                        )
                                        if idx_hits:
                                            files_hint = ", ".join(
                                                h["rel_path"] for h in idx_hits[:6]
                                            )
                                            context_lines.append(
                                                f"Indexed files: {files_hint}"
                                            )
                                    except Exception as exc:
                                        logger.debug(
                                            "Failed to query file index for chat goal=%r: %s",
                                            goal[:120],
                                            exc,
                                        )
                                    history_tail = (
                                        " | ".join(history_lines[-3:])
                                        if history_lines
                                        else ""
                                    )
                                    if history_tail:
                                        context_lines.append(
                                            f"Chat context: {history_tail}"
                                        )
                                    goal = (
                                        f"{goal}\n\nTechnical plan context:\n- "
                                        + "\n- ".join(context_lines)
                                    )
                                    agent_task = _spawn_code_agent_task(
                                        goal=goal,
                                        notebook_path=notebook_path,
                                        allow_write=allow_code_writes,
                                        session_id=session_id,
                                    )
                                    result = {
                                        "status": "spawned",
                                        "task_id": agent_task.get("task_id"),
                                        "goal": truncate_summary(
                                            str(action.get("goal") or question), 120
                                        ),
                                    }
                                    if not code_agent_task:
                                        code_agent_task = agent_task
                                else:
                                    result = {
                                        "status": "error",
                                        "error": "No goal provided",
                                    }
                            else:
                                result = runner.execute_chat_action(action, nb)
                            if (
                                str(action.get("type") or "").strip()
                                == "start_experiment"
                                and str(result.get("status") or "").strip() == "started"
                                and result.get("experiment_id")
                            ):
                                record_run_trigger(
                                    experiment_id=str(result.get("experiment_id")),
                                    source="chat_action",
                                    mode=str(result.get("mode") or "single").strip()
                                    or "single",
                                    details={
                                        "endpoint": "/api/aria/chat",
                                        "session_id": session_id or None,
                                    },
                                )
                            actions_taken.append(
                                {
                                    "type": action.get("type"),
                                    "status": result.get("status", "unknown"),
                                    "detail": result,
                                }
                            )
                        except Exception as action_err:
                            actions_taken.append(
                                {
                                    "type": action.get("type"),
                                    "status": "error",
                                    "detail": {"error": str(action_err)},
                                }
                            )
                    actionable = any(
                        str(a.get("status") or "").lower()
                        in {"applied", "started", "spawned"}
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
                        reply_text = truncate_summary(
                            f"Action started: {action_types}. "
                            f"Status: {'; '.join(status_bits[:4])}. "
                            f"Next checkpoint: monitor task progress and report completion.",
                            240,
                        )
                    else:
                        summary = str(parsed.get("summary") or "").strip()
                        reply_text = truncate_summary(
                            summary
                            or "advice_only: no valid executable actions were produced.",
                            220,
                        )
                        advice_only = True
                    if code_agent_task and code_agent_task.get("task_id"):
                        snap = summarize_agent_task(code_agent_task)
                        reply_text = truncate_summary(
                            f"{reply_text} Task {snap.get('task_id')} queued ({snap.get('milestone_summary')}).",
                            260,
                        )
                    record_chat_guardrail_event(
                        actionable=actionable,
                        advice_only=advice_only,
                        summary_text=reply_text,
                    )
                    if session_id:
                        try:
                            nb.save_chat_message(
                                session_id=session_id,
                                role="aria",
                                text=reply_text,
                                label="Aria",
                            )
                        except Exception as exc:
                            logger.debug(
                                "Failed to persist Aria chat reply for session_id=%s: %s",
                                session_id,
                                exc,
                            )
                    return jsonify(
                        {
                            "reply": reply_text,
                            "ai_powered": True,
                            "used_context": True,
                            "fallback_reason": None,
                            "brief_mode": brief_response,
                            "agent_task": code_agent_task,
                            "actions_taken": actions_taken,
                            "advice_only": advice_only,
                            "local_tools_used": local_agent_result.get(
                                "tools_used", []
                            ),
                            "local_code_hits": [
                                {
                                    "path": hit.get("path"),
                                    "abs_path": hit.get("abs_path"),
                                    "line": hit.get("line"),
                                    "score": hit.get("score"),
                                }
                                for hit in local_agent_result.get("code_hits", [])
                            ],
                        }
                    )
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
            fallback_reply = (
                "LLM unavailable. Check Strategy Advisor for current recommendations."
            )
        else:
            fallback_reply = "LLM unavailable. Try a fix-intent request (e.g. 'fix X') to spawn an agent."
        if session_id:
            try:
                nb.save_chat_message(
                    session_id=session_id,
                    role="aria",
                    text=fallback_reply,
                    label=f"Aria (fallback: {fallback_reason})",
                )
            except Exception as exc:
                logger.debug(
                    "Failed to persist fallback Aria chat reply for session_id=%s: %s",
                    session_id,
                    exc,
                )
        record_chat_guardrail_event(
            actionable=False,
            advice_only=True,
            summary_text=fallback_reply,
        )
        return jsonify(
            {
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
            }
        )

    @app.route("/api/aria/chat/history")
    @wnb
    def api_aria_chat_history(nb=None):
        """Load chat history from the database."""
        session_id = request.args.get("session_id", "default")
        limit = min(int(request.args.get("limit", 50)), 200)
        messages = nb.get_chat_history(session_id, limit=limit)
        return jsonify({"messages": messages, "session_id": session_id})

    @app.route("/api/aria/chat/message", methods=["POST"])
    @wnb
    def api_aria_chat_message(nb=None):
        """Save a single chat message to the database."""
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
            session_id=session_id,
            role=role,
            text=text,
            label=label,
            message_id=message_id,
            metadata=metadata,
        )
        return jsonify({"message_id": mid, "saved": True})

    @app.route("/api/aria/chat/compact", methods=["POST"])
    @wnb
    def api_aria_chat_compact(nb=None):
        """Compact older chat messages into a summary when token budget exceeded."""
        aria = get_aria()
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id", "default")
        token_budget = int(body.get("token_budget", 4000))

        messages = nb.get_chat_history(session_id, limit=200)
        if not messages:
            return jsonify({"compacted": False, "reason": "no messages"})

        # Calculate tokens for active messages
        total_tokens = sum(estimate_tokens(m.get("text", "")) for m in messages)
        if total_tokens <= token_budget:
            return jsonify(
                {
                    "compacted": False,
                    "reason": "within budget",
                    "total_tokens": total_tokens,
                }
            )

        # Find oldest messages that exceed the budget
        # Keep recent messages within budget, compact the rest
        keep_tokens = 0
        keep_from = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens(messages[i].get("text", ""))
            if (
                keep_tokens + msg_tokens > token_budget * 0.7
            ):  # Keep 70% budget for recent
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
                from ..llm.prompts import SYSTEM_PROMPT, CHAT_COMPACTION_PROMPT

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
            summary_text = (
                "\n".join(lines) if lines else "Previous conversation summarized."
            )

        # Save summary message
        import uuid as _uuid

        summary_id = f"summary-{_uuid.uuid4().hex[:8]}"
        compact_ids = [m["message_id"] for m in to_compact if m.get("message_id")]

        nb.save_chat_message(
            session_id=session_id,
            role="system",
            text=summary_text,
            label="Summary",
            message_id=summary_id,
            metadata={"compaction": True, "summarized_count": len(compact_ids)},
        )
        nb.mark_messages_compacted(compact_ids, summary_id)

        return jsonify(
            {
                "compacted": True,
                "messages_compacted": len(compact_ids),
                "summary_id": summary_id,
                "summary_tokens": estimate_tokens(summary_text),
                "original_tokens": total_tokens,
            }
        )
