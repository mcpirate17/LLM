import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useEventBus } from '../hooks/useEventBus';
import useRenderPerf from '../hooks/useRenderPerf';

const API_BASE = process.env.REACT_APP_API_URL || '';
const SESSION_ID_KEY = 'aria_chat_session_id_v2';
const TOKEN_BUDGET = 4000; // ~4K tokens before compaction

function getSessionId() {
  try {
    let sid = window.sessionStorage.getItem(SESSION_ID_KEY);
    if (!sid) {
      sid = `chat-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      window.sessionStorage.setItem(SESSION_ID_KEY, sid);
    }
    return sid;
  } catch {
    return `chat-${Date.now()}`;
  }
}

function estimateTokens(text) {
  return Math.ceil((text || '').length / 4);
}

function buildMessage(id, role, text, meta = {}) {
  return {
    id,
    role,
    text,
    timestamp: Date.now(),
    ...meta,
  };
}

function summarizeForChat(text, maxChars = 280) {
  const cleaned = String(text || '')
    .replace(/```(?!action\b)[\s\S]*?```/gi, '[details sent to local agent]')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return '';
  if (cleaned.length <= maxChars) return cleaned;
  return `${cleaned.slice(0, maxChars - 1).trimEnd()}…`;
}

function formatTimestamp(ts) {
  // Handle both seconds (from DB) and milliseconds (from JS)
  const ms = ts > 1e12 ? ts : ts * 1000;
  const d = new Date(ms);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function fallbackReasonLabel(reason) {
  if (!reason) return 'unknown';
  if (reason === 'llm_not_configured') return 'LLM not configured';
  if (reason === 'llm_unreachable') return 'LLM configured but unreachable';
  if (reason === 'llm_empty_response') return 'LLM returned empty response';
  if (String(reason).startsWith('llm_error:')) return `LLM error (${String(reason).split(':', 2)[1] || 'unknown'})`;
  return String(reason);
}

function localHelperReasonLabel(reason) {
  const key = String(reason || '').toLowerCase();
  if (!key) return 'unknown';
  if (key === 'ok') return 'ready';
  if (key === 'primary_llm_not_remote') return 'primary LLM is not remote';
  if (key === 'model_not_configured') return 'model not configured';
  if (key === 'ollama_cli_missing') return 'ollama CLI missing';
  if (key === 'model_not_found_or_unreachable') return 'model unavailable';
  if (key === 'vram_limit_exceeded') return 'over 10GB limit';
  return key.replace(/_/g, ' ');
}

function dbMessageToLocal(msg) {
  // Truncate old verbose messages from DB to keep chat readable
  let text = msg.text || '';
  if (msg.role !== 'user' && text.length > 300) {
    text = text.slice(0, 297).trimEnd() + '...';
  }
  return {
    id: msg.message_id || `db-${msg.timestamp}`,
    role: msg.role,
    text,
    timestamp: msg.timestamp > 1e12 ? msg.timestamp : msg.timestamp * 1000,
    label: msg.label || undefined,
    isSummary: Boolean(msg.summary_of),
  };
}

function renderLocalEvidence(message) {
  const toolsUsed = Array.isArray(message.localToolsUsed) ? message.localToolsUsed : [];
  const codeHits = Array.isArray(message.localCodeHits) ? message.localCodeHits : [];
  if (toolsUsed.length === 0 && codeHits.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: 6,
        paddingTop: 6,
        borderTop: '1px solid var(--border)',
        fontSize: 10,
        color: 'var(--text-muted)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      {toolsUsed.length > 0 && (
        <div>
          <span style={{ fontWeight: 700 }}>Local tools:</span>{' '}
          <span>{toolsUsed.join(', ')}</span>
        </div>
      )}
      {codeHits.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontWeight: 700 }}>Code evidence:</span>
          {codeHits.slice(0, 3).map((hit, idx) => {
            const line = hit.line || '?';
            const label = `${hit.path || 'unknown'}:${line}`;
            const absPath = hit.abs_path || '';
            const vscodeHref = absPath
              ? `vscode://file${encodeURI(absPath)}${Number.isFinite(Number(hit.line)) ? `:${Number(hit.line)}` : ''}`
              : null;

            if (!vscodeHref) {
              return (
                <span key={`${hit.path || 'hit'}-${hit.line || idx}-${idx}`}>
                  {label}
                </span>
              );
            }

            return (
              <a
                key={`${hit.path || 'hit'}-${hit.line || idx}-${idx}`}
                href={vscodeHref}
                style={{ color: 'var(--accent-blue)', textDecoration: 'underline' }}
                title={`Open ${label} in VS Code`}
              >
                {label}
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

function actionSummary(action) {
  const type = action.type || 'unknown';
  const detail = action.detail || {};
  const fmtVal = (v) => typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v);
  if (type === 'adjust_config') {
    const changes = detail.changes || {};
    const preview = Object.entries(changes).slice(0, 4).map(([k, v]) => `${k}=${fmtVal(v)}`).join(', ');
    return summarizeForChat(`Adjusted config: ${preview}${Object.keys(changes).length > 4 ? ', …' : ''}`, 160);
  }
  if (type === 'adjust_grammar') {
    const weights = detail.weights || {};
    const preview = Object.entries(weights).slice(0, 4).map(([k, v]) => `${k}=${fmtVal(v)}`).join(', ');
    return summarizeForChat(`Adjusted grammar: ${preview}${Object.keys(weights).length > 4 ? ', …' : ''}`, 160);
  }
  if (type === 'start_experiment') {
    return detail.experiment_id ? `Started experiment ${detail.experiment_id.slice(0, 8)}` : (detail.error || 'Experiment busy');
  }
  if (type === 'edit_file') {
    return detail.path ? `Edited ${detail.path}: ${detail.description || ''}` : (detail.error || 'Edit failed');
  }
  if (type === 'spawn_agent') {
    return detail.task_id ? summarizeForChat(`Spawned agent ${detail.task_id}: ${detail.goal || ''}`, 160) : (detail.error || 'Agent spawn failed');
  }
  return summarizeForChat(`${type}: ${action.status}`, 160);
}

function renderActions(message) {
  const actions = Array.isArray(message.actionsTaken) ? message.actionsTaken : [];
  if (actions.length === 0) return null;

  return (
    <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
      {actions.map((action, idx) => {
        const ok = action.status === 'applied' || action.status === 'started' || action.status === 'spawned';
        return (
          <div
            key={`action-${idx}`}
            style={{
              fontSize: 11,
              padding: '4px 8px',
              borderLeft: `3px solid ${ok ? 'var(--accent-green, #4caf50)' : 'var(--accent-red, #f44336)'}`,
              background: ok ? 'rgba(76, 175, 80, 0.08)' : 'rgba(244, 67, 54, 0.08)',
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span style={{ fontSize: 12 }}>{ok ? '\u{1F527}' : '\u26A0\uFE0F'}</span>
            <span>{actionSummary(action)}</span>
          </div>
        );
      })}
    </div>
  );
}

function renderAgentTask(message) {
  const task = message.agentTask;
  if (!task || !task.task_id) return null;

  const status = String(task.status || 'queued').toLowerCase();
  const isFailed = status === 'failed';
  const statusColor = isFailed
    ? 'var(--accent-red, #f44336)'
    : status === 'completed'
      ? 'var(--accent-green, #4caf50)'
      : 'var(--accent-yellow, #ffb300)';

  const milestone = String(task.milestone_summary || task.summary || '').trim();
  const detailUrl = task.full_status_url
    ? `${API_BASE}${task.full_status_url}`
    : `${API_BASE}/api/aria/agent/status/${encodeURIComponent(task.task_id)}?detail=full`;

  return (
    <div
      style={{
        marginTop: 6,
        paddingTop: 6,
        borderTop: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
        <span style={{ fontWeight: 700 }}>Codebase Agent:</span>{' '}
        <span>{task.task_id}</span>{' '}
        <span style={{ color: statusColor, fontWeight: 700 }}>
          ({status.toUpperCase()})
        </span>
      </div>
      {milestone && (
        <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
          {summarizeForChat(milestone, 220)}
        </div>
      )}
      <a
        href={detailUrl}
        target="_blank"
        rel="noreferrer"
        style={{ alignSelf: 'flex-start', fontSize: 10, color: 'var(--accent-blue)' }}
      >
        Open full task details
      </a>
    </div>
  );
}

function messageDedupKey(message) {
  return [
    String(message?.role || ''),
    String(message?.label || ''),
    String(message?.text || '').trim(),
  ].join('::');
}

function AriaChatPanel({ isRunning, autonomousMode, onAutonomousEnd }) {
  useRenderPerf('AriaChatPanel');

  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');
  const [localHelper, setLocalHelper] = useState(null);
  const [chatGuardrails, setChatGuardrails] = useState(null);
  const completedAgentTasksRef = useRef(new Set());
  const lastEvidenceSnapshotRef = useRef('');
  const staleNoticeKeyRef = useRef('');
  const sessionId = useRef(getSessionId()).current;

  // Persist a system message to DB (fire-and-forget)
  const persistSystemMessage = useCallback((text, label) => {
    fetch(`${API_BASE}/api/aria/chat/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        role: 'system',
        text,
        label: label || 'System',
      }),
    }).catch(() => {});
  }, [sessionId]);

  const addSystemMessage = useCallback((text) => {
    setMessages((prev) => {
      const next = [...prev, buildMessage(`sys-${Date.now()}`, 'system', text)];
      return next;
    });
    persistSystemMessage(text);
  }, [persistSystemMessage]);

  // Load chat history from DB
  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/aria/chat/history?session_id=${encodeURIComponent(sessionId)}&limit=50`);
      if (res.ok) {
        const data = await res.json();
        const dbMessages = (data.messages || []).map(dbMessageToLocal);
        if (dbMessages.length > 0) {
          setMessages(dbMessages);
          setLoading(false);
          return true;
        }
      }
    } catch { /* ignore */ }
    return false;
  }, [sessionId]);

  // Auto-compact when token budget exceeded
  const compactingRef = useRef(false);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  const checkCompaction = useCallback(async () => {
    if (compactingRef.current) return;
    const msgs = messagesRef.current;
    const totalTokens = msgs.reduce((sum, m) => sum + estimateTokens(m.text), 0);
    if (totalTokens > TOKEN_BUDGET) {
      compactingRef.current = true;
      try {
        const res = await fetch(`${API_BASE}/api/aria/chat/compact`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (res.ok) {
          await loadHistory();
        }
      } catch { /* ignore */ }
      compactingRef.current = false;
    }
  }, [sessionId, loadHistory]);

  // Trigger compaction check after messages change
  const messagesLength = messages.length;
  useEffect(() => {
    if (messagesLength > 8) {
      checkCompaction();
    }
  }, [messagesLength, checkCompaction]);

  useEffect(() => {
    const interval = setInterval(async () => {
      const currentMessages = messagesRef.current || [];
      const taskById = new Map();
      for (const message of currentMessages) {
        const task = message?.agentTask;
        const taskId = task?.task_id;
        if (!taskId || taskById.has(taskId)) continue;
        taskById.set(taskId, task);
      }

      const pendingTaskIds = [];
      for (const [taskId, task] of taskById.entries()) {
        const status = String(task?.status || '').toLowerCase();
        if (status !== 'completed' && status !== 'failed') {
          pendingTaskIds.push(taskId);
        }
      }

      if (pendingTaskIds.length === 0) return;

      await Promise.all(pendingTaskIds.map(async (taskId) => {
        try {
          const res = await fetch(`${API_BASE}/api/aria/agent/status/${encodeURIComponent(taskId)}/summary`);
          const data = await res.json();
          if (!res.ok || !data?.task) return;
          const task = data.task;

          setMessages((prev) => {
            let changed = false;
            const next = prev.map((msg) => {
              if (msg?.agentTask?.task_id !== taskId) return msg;
              const previousTask = msg.agentTask || {};
              const mergedTask = {
                ...previousTask,
                ...task,
              };

              const unchanged =
                previousTask.status === mergedTask.status
                && previousTask.updated_at === mergedTask.updated_at
                && previousTask.summary === mergedTask.summary
                && previousTask.error === mergedTask.error
                && previousTask.current_step === mergedTask.current_step
                && previousTask.total_steps === mergedTask.total_steps
                && JSON.stringify(previousTask.applied_edits || []) === JSON.stringify(mergedTask.applied_edits || [])
                && JSON.stringify(previousTask.proposed_edits || []) === JSON.stringify(mergedTask.proposed_edits || []);

              if (unchanged) {
                return msg;
              }

              changed = true;
              return {
                ...msg,
                agentTask: mergedTask,
              };
            });
            return changed ? next : prev;
          });

          const status = String(task.status || '').toLowerCase();
          if ((status === 'completed' || status === 'failed') && !completedAgentTasksRef.current.has(taskId)) {
            completedAgentTasksRef.current.add(taskId);
            addSystemMessage(
              summarizeForChat(`Codebase agent update: ${task.milestone_summary || `${taskId} ${status}`}`, 170),
            );
          }
        } catch {
          // ignore polling errors
        }
      }));
    }, 2500);

    return () => clearInterval(interval);
  }, [addSystemMessage]);

  const refreshAnalysis = useCallback(async () => {
    setError('');
    try {
      let evidenceKey = '';
      try {
        const progressRes = await fetch(`${API_BASE}/api/progress`);
        if (progressRes.ok) {
          const progressPayload = await progressRes.json();
          const p = progressPayload?.progress || {};
          evidenceKey = JSON.stringify({
            experiment_id: p.experiment_id || null,
            status: p.status || null,
            current_generation: p.current_generation ?? null,
            total_generations: p.total_generations ?? null,
            best_fitness: p.best_fitness ?? null,
            avg_fitness: p.avg_fitness ?? null,
            archive_size: p.archive_size ?? null,
            run_trigger_source: progressPayload?.run_trigger_source || p.run_trigger_source || null,
          });
        }
      } catch {
        // proceed without stale-evidence guard if progress probe fails
      }

      const shouldGuardStale = !isRunning && !autonomousMode;
      if (shouldGuardStale && evidenceKey && evidenceKey === lastEvidenceSnapshotRef.current) {
        if (staleNoticeKeyRef.current !== evidenceKey) {
          staleNoticeKeyRef.current = evidenceKey;
          addSystemMessage('No new evidence since last analysis snapshot.');
        }
        setLoading(false);
        return;
      }

      const briefingRes = await fetch(`${API_BASE}/api/strategy/briefing`);

      const newMessages = [];

      if (briefingRes.ok) {
        const briefing = await briefingRes.json();
        if (briefing && !briefing.error && briefing.briefing) {
          const consolidatedText = briefing.action_label
            ? `${briefing.briefing}\n\nRecommended Action: ${briefing.action_label}${briefing.action_rationale ? ` — ${briefing.action_rationale}` : ''}`
            : briefing.briefing;
          newMessages.push(buildMessage(
            `briefing-${Date.now()}`,
            'aria',
            summarizeForChat(consolidatedText, 300),
            { label: briefing.ai_powered ? 'AI Briefing' : 'Fallback Briefing' },
          ));
        }
      }

      if (newMessages.length > 0) {
        let messagesToPersist = [];
        setMessages((prev) => {
          const seen = new Set(prev.slice(-80).map((m) => messageDedupKey(m)));
          const uniqueMessages = [];
          for (const msg of newMessages) {
            const key = messageDedupKey(msg);
            if (seen.has(key)) {
              continue;
            }
            seen.add(key);
            uniqueMessages.push(msg);
          }
          messagesToPersist = uniqueMessages;
          if (uniqueMessages.length === 0) {
            return prev;
          }
          return [...prev, ...uniqueMessages];
        });
        // Persist analysis messages to DB
        for (const msg of messagesToPersist) {
          fetch(`${API_BASE}/api/aria/chat/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              role: msg.role,
              text: msg.text,
              label: msg.label,
              message_id: msg.id,
            }),
          }).catch(() => {});
        }
      }

      if (evidenceKey) {
        lastEvidenceSnapshotRef.current = evidenceKey;
        staleNoticeKeyRef.current = '';
      }
    } catch {
      setError('Aria analysis is temporarily unavailable.');
    } finally {
      setLoading(false);
    }
  }, [sessionId, addSystemMessage, isRunning, autonomousMode]);

  const refreshToolingStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/aria/tools`);
      if (!res.ok) return;
      const data = await res.json();
      const helper = data?.local_ollama_helper;
      if (!helper || typeof helper !== 'object') return;
      const guardrails = data?.chat_guardrails;
      setLocalHelper({
        enabled: Boolean(helper.enabled),
        reason: String(helper.reason || ''),
        model: String(helper.model || ''),
        estimatedVramGb: helper.estimated_vram_gb,
        maxVramGb: helper.max_vram_gb,
      });
      if (guardrails && typeof guardrails === 'object') {
        setChatGuardrails({
          actionableRate: Number(guardrails.actionable_response_rate || 0),
          adviceOnlyRate: Number(guardrails.advice_only_rate || 0),
          avgSummaryLength: Number(guardrails.summary_length?.avg || 0),
          window: Number(guardrails.window || 0),
        });
      }
    } catch {
      // ignore polling failures
    }
  }, []);

  const sendMessage = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;

    const userMessage = buildMessage(`user-${Date.now()}`, 'user', text, { label: 'You' });

    setMessages((prev) => [...prev, userMessage]);
    setDraft('');
    setSending(true);
    setError('');

    try {
      const res = await fetch(`${API_BASE}/api/aria/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
        }),
      });
      const data = await res.json();
      if (!res.ok || data?.error) {
        throw new Error(data?.error || 'Chat request failed');
      }
      if (data?.reply) {
        const ariaMessage = buildMessage(
          `aria-${Date.now()}`,
          'aria',
          summarizeForChat(data.reply, 300),
          {
            label: data.ai_powered
              ? 'Aria'
              : `Aria (fallback: ${fallbackReasonLabel(data.fallback_reason)})`,
            localToolsUsed: Array.isArray(data.local_tools_used) ? data.local_tools_used : [],
            localCodeHits: Array.isArray(data.local_code_hits) ? data.local_code_hits : [],
            actionsTaken: Array.isArray(data.actions_taken) ? data.actions_taken : [],
            agentTask: data?.agent_task && typeof data.agent_task === 'object' ? data.agent_task : null,
          },
        );
        setMessages((prev) => [...prev, ariaMessage]);
      }
    } catch {
      setError('Aria could not respond to that message right now.');
    } finally {
      setSending(false);
    }
  }, [draft, sending, sessionId]);

  const initializedRef = useRef(false);
  useEffect(() => {
    if (!initializedRef.current) {
      initializedRef.current = true;
      // Try loading from DB first
      loadHistory().then((hadMessages) => {
        if (!hadMessages) {
          // Manual-only mode: do not auto-analyze on init.
        }
      });
    }
    return undefined;
  }, [loadHistory]);

  useEffect(() => {
    refreshToolingStatus();
    const interval = setInterval(() => {
      refreshToolingStatus();
    }, 15000);
    return () => clearInterval(interval);
  }, [refreshToolingStatus]);

  useEffect(() => {
    const handler = (event) => {
      const detail = event?.detail || {};
      const task = detail?.task || {};
      const taskId = String(task.task_id || '').trim();
      if (!taskId) return;

      const source = String(detail?.source || 'start').replace(/_/g, ' ');
      const helperState = localHelper
        ? (localHelper.enabled
          ? `local helper active (${localHelper.model || 'configured'})`
          : `local helper blocked (${localHelperReasonLabel(localHelper.reason)})`)
        : 'local helper status unknown';

      addSystemMessage(
        summarizeForChat(`Auto-repair agent started (${taskId}) after ${source} failure — ${helperState}.`, 170),
      );
    };

    window.addEventListener('aria-auto-repair-started', handler);
    return () => window.removeEventListener('aria-auto-repair-started', handler);
  }, [addSystemMessage, localHelper]);

  // Subscribe to SSE events via shared EventBus (instead of opening a dedicated EventSource)
  useEventBus('experiment_completed', useCallback((data) => {
    let resultLine = '';
    try {
      const mode = data.mode || data.results?.experiment_type || '';
      const expId = (data.experiment_id || '').slice(0, 8);
      const results = data.results || {};
      const s1 = results.stage1_passed ?? results.n_stage1_passed ?? 0;
      const gen = results.total_generated ?? results.n_programs_generated ?? 0;
      const loss = results.best_loss_ratio;
      const parts = [];
      if (expId) parts.push(`[${expId}]`);
      if (mode) parts.push(mode);
      if (gen > 0) parts.push(`${s1}/${gen} S1 survivors (${(s1/gen*100).toFixed(1)}%)`);
      if (loss != null) parts.push(`best loss ${loss.toFixed(4)}`);
      if (parts.length > 0) resultLine = ` — ${parts.join(' · ')}`;
    } catch { /* ignore parse errors */ }
    addSystemMessage(`Experiment completed${resultLine}. Ask Aria for the next action when ready.`);
  }, [addSystemMessage]));

  useEventBus('experiment_started', useCallback(() => {
    addSystemMessage('Experiment started. Ask Aria if you want immediate action guidance.');
  }, [addSystemMessage]));

  useEventBus('aria_cycle_completed', useCallback((data) => {
    try {
      const cycle = data.cycle_index || 0;
      const mode = data.mode || 'synthesis';
      const status = data.status || 'completed';
      const s1 = data.stage1_survivors;
      const deltaS1 = data.delta_stage1_survivors;
      const reasoning = data.reasoning || '';
      const parts = [`Cycle ${cycle}`, mode, status];
      if (typeof s1 === 'number') parts.push(`S1 total ${s1}`);
      if (typeof deltaS1 === 'number') parts.push(`ΔS1 +${deltaS1}`);
      if (reasoning) parts.push(reasoning);
      addSystemMessage(`Aria cycle summary: ${parts.join(' · ')}`);
    } catch {
      addSystemMessage('Aria completed a research cycle and is preparing the next step.');
    }
  }, [addSystemMessage]));

  useEventBus('mode_selected', useCallback((data) => {
    try {
      const mode = data.mode || 'unknown';
      const reasoning = data.reasoning || '';
      addSystemMessage(`Aria selected mode: ${mode}${reasoning ? ` — ${reasoning}` : ''}`);
    } catch {
      addSystemMessage('Aria selected the next experiment mode.');
    }
  }, [addSystemMessage]));

  useEventBus('continuous_limit_reached', useCallback((data) => {
    try {
      const reason = data.reason || 'limit reached';
      addSystemMessage(`Autonomous session complete: ${reason}`);
    } catch {
      addSystemMessage('Autonomous session complete.');
    }
    if (onAutonomousEnd) onAutonomousEnd();
  }, [addSystemMessage, onAutonomousEnd]));

  useEventBus('knowledge_extracted', useCallback((data) => {
    try {
      const count = data.count || data.n_insights || 0;
      if (count > 0) {
        addSystemMessage(`Extracted ${count} insight${count === 1 ? '' : 's'} from latest results.`);
      }
    } catch { /* ignore */ }
  }, [addSystemMessage]));

  // Re-fetch when LLM is configured (dispatched by ControlPanel)
  useEffect(() => {
    const handler = () => {
      addSystemMessage('LLM configured. Ask Aria when you want an action-oriented review.');
    };
    window.addEventListener('llm-configured', handler);
    return () => window.removeEventListener('llm-configured', handler);
  }, [addSystemMessage]);

  const handleClear = useCallback(async () => {
    setMessages([]);
    // Start a new session
    try {
      const newSid = `chat-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      window.sessionStorage.setItem(SESSION_ID_KEY, newSid);
      // Force page-level re-mount would be heavy; just clear local state
    } catch { /* ignore */ }
  }, []);

  return (
    <div className="card" style={{ marginTop: 12, marginBottom: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
          Aria Chat
          {messages.length > 0 && (
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400 }}>
              ({messages.length})
            </span>
          )}
          {autonomousMode && (
            <span style={{
              fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
              color: 'var(--accent-purple)',
              background: 'rgba(137, 87, 229, 0.12)',
              border: '1px solid var(--accent-purple)',
              borderRadius: 4, padding: '1px 5px',
            }}>
              Autonomous
            </span>
          )}
          {localHelper && (
            <span
              title={localHelper.model
                ? `model=${localHelper.model}${localHelper.estimatedVramGb != null ? `, est=${localHelper.estimatedVramGb}GB` : ''}${localHelper.maxVramGb != null ? `, limit=${localHelper.maxVramGb}GB` : ''}`
                : undefined}
              style={{
                fontSize: 9,
                fontWeight: 700,
                textTransform: 'uppercase',
                color: localHelper.enabled ? 'var(--accent-green, #4caf50)' : 'var(--text-muted)',
                background: localHelper.enabled ? 'rgba(76, 175, 80, 0.12)' : 'var(--bg-primary)',
                border: `1px solid ${localHelper.enabled ? 'var(--accent-green, #4caf50)' : 'var(--border)'}`,
                borderRadius: 4,
                padding: '1px 5px',
              }}
            >
              Local LM: {localHelper.enabled ? 'ready' : localHelperReasonLabel(localHelper.reason)}
            </span>
          )}
          <span
            title="Aria code agent can directly patch Python and JavaScript files"
            style={{
              fontSize: 9,
              fontWeight: 700,
              textTransform: 'uppercase',
              color: 'var(--accent-blue)',
              background: 'rgba(31, 111, 235, 0.12)',
              border: '1px solid var(--accent-blue)',
              borderRadius: 4,
              padding: '1px 5px',
            }}
          >
            Self-fix: .py/.js
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <span
            title="Aria analyzes only on explicit request"
            style={{
              fontSize: 10,
              padding: '2px 6px',
              background: 'var(--bg-primary)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              color: 'var(--text-secondary)',
            }}
          >
            Auto: Off (Manual only)
          </span>
          {messages.length > 0 && (
            <button
              className="refresh-btn"
              style={{ fontSize: 10, padding: '2px 6px' }}
              onClick={handleClear}
            >
              Clear
            </button>
          )}
          <button className="refresh-btn" onClick={refreshAnalysis} disabled={loading}>
            {loading ? 'Loading...' : 'Ask for Action'}
          </button>
        </div>
      </div>

      {isRunning && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
          Run active — Aria will only respond when you explicitly ask.
        </div>
      )}

      {error && (
        <div style={{ fontSize: 11, color: 'var(--accent-yellow)', marginBottom: 8 }}>
          {error}
        </div>
      )}
      {chatGuardrails && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8 }}>
          Guardrails ({chatGuardrails.window}): actionable {(chatGuardrails.actionableRate * 100).toFixed(0)}% ·
          advice-only {(chatGuardrails.adviceOnlyRate * 100).toFixed(0)}% ·
          avg summary {Math.round(chatGuardrails.avgSummaryLength)} chars
        </div>
      )}

      <div style={{ maxHeight: 400, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {messages.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Aria has not posted analysis yet.
          </div>
        ) : (
          messages.map((m) => (
            <div
              key={m.id}
              style={{
                padding: '8px 10px',
                borderRadius: 6,
                ...(m.role === 'user' ? {
                  background: 'var(--bg-tertiary)',
                  marginLeft: 32,
                  borderRight: '2px solid var(--accent-blue)',
                } : m.role === 'system' ? {
                  background: 'var(--bg-primary)',
                  textAlign: 'center',
                  fontStyle: 'italic',
                  borderLeft: 'none',
                } : {
                  background: m.isSummary ? 'rgba(137, 87, 229, 0.08)' : 'var(--bg-tertiary)',
                  marginRight: 32,
                  borderLeft: `2px solid ${m.isSummary ? 'var(--accent-purple)' : 'var(--accent-purple)'}`,
                }),
              }}
            >
              <div style={{ display: 'flex', justifyContent: m.role === 'system' ? 'center' : 'space-between', gap: 8, marginBottom: 3 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  {m.isSummary ? 'Summary' : m.label || (m.role === 'user' ? 'You' : m.role === 'system' ? 'System' : 'Aria')}
                </span>
                {m.role !== 'system' && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{formatTimestamp(m.timestamp)}</span>}
              </div>
              <div style={{ fontSize: m.role === 'system' ? 11 : 12, lineHeight: 1.45, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>{m.text}</div>
              {renderAgentTask(m)}
              {renderActions(m)}
              {renderLocalEvidence(m)}
            </div>
          ))
        )}
      </div>

      <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Ask Aria about the latest results..."
          aria-label="Message to Aria"
          style={{
            flex: 1,
            background: 'var(--bg-primary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            color: 'var(--text-primary)',
            fontSize: 12,
            padding: '8px 10px',
          }}
        />
        <button
          className="refresh-btn"
          onClick={sendMessage}
          disabled={sending || !draft.trim()}
          aria-label={sending ? 'Sending message, please wait' : 'Send message'}
          aria-busy={sending}
        >
          {sending ? 'Sending…' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default AriaChatPanel;
