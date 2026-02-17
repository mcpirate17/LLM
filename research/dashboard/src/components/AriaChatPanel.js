import React, { useEffect, useRef, useState, useCallback } from 'react';

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

function dbMessageToLocal(msg) {
  return {
    id: msg.message_id || `db-${msg.timestamp}`,
    role: msg.role,
    text: msg.text,
    timestamp: msg.timestamp > 1e12 ? msg.timestamp : msg.timestamp * 1000,
    label: msg.label || undefined,
    isSummary: Boolean(msg.summary_of),
  };
}

function AriaChatPanel({ isRunning, autonomousMode, onAutonomousEnd }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');
  const eventSourceRef = useRef(null);
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

  const refreshAnalysis = useCallback(async () => {
    setError('');
    try {
      const [briefingRes, strategyRes, recRes] = await Promise.all([
        fetch(`${API_BASE}/api/strategy/briefing`),
        fetch(`${API_BASE}/api/aria/strategy`),
        fetch(`${API_BASE}/api/aria/recommendation`),
      ]);

      const newMessages = [];

      if (briefingRes.ok) {
        const briefing = await briefingRes.json();
        if (briefing && !briefing.error && briefing.briefing) {
          newMessages.push(buildMessage(
            `briefing-${Date.now()}`,
            'aria',
            briefing.briefing,
            { label: briefing.ai_powered ? 'AI Briefing' : 'Fallback Briefing' },
          ));
          if (briefing.action_label) {
            newMessages.push(buildMessage(
              `action-${Date.now()}`,
              'aria',
              `${briefing.action_label}${briefing.action_rationale ? ` — ${briefing.action_rationale}` : ''}`,
              { label: 'Recommended Action' },
            ));
          }
        }
      }

      if (strategyRes.ok) {
        const strategy = await strategyRes.json();
        if (strategy?.strategy) {
          newMessages.push(buildMessage(
            `strategy-${Date.now()}`,
            'aria',
            strategy.strategy,
            { label: 'Strategy' },
          ));
        }
      }

      if (recRes.ok) {
        const rec = await recRes.json();
        if (rec?.reasoning) {
          newMessages.push(buildMessage(
            `rec-${Date.now()}`,
            'aria',
            rec.reasoning,
            { label: 'Experiment Recommendation' },
          ));
        }
      }

      if (newMessages.length > 0) {
        setMessages((prev) => [...prev, ...newMessages]);
        // Persist analysis messages to DB
        for (const msg of newMessages) {
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
    } catch {
      setError('Aria analysis is temporarily unavailable.');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

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
          data.reply,
          {
            label: data.ai_powered
              ? 'Aria'
              : `Aria (fallback: ${fallbackReasonLabel(data.fallback_reason)})`,
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
          refreshAnalysis();
        }
      });
    }
    const interval = setInterval(refreshAnalysis, 30000);
    return () => clearInterval(interval);
  }, [refreshAnalysis, loadHistory]);

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/events`);
    eventSourceRef.current = es;

    es.addEventListener('experiment_completed', (event) => {
      let resultLine = '';
      try {
        const data = JSON.parse(event.data || '{}');
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
      addSystemMessage(`Experiment completed${resultLine}. Aria is reviewing the latest results.`);
      setTimeout(() => refreshAnalysis(), 1500);
    });

    es.addEventListener('experiment_started', () => {
      addSystemMessage('Experiment started. Aria will post analysis when results are ready.');
    });

    es.addEventListener('aria_cycle_completed', (event) => {
      try {
        const data = JSON.parse(event.data || '{}');
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
    });

    es.addEventListener('mode_selected', (event) => {
      try {
        const data = JSON.parse(event.data || '{}');
        const mode = data.mode || 'unknown';
        const reasoning = data.reasoning || '';
        addSystemMessage(`Aria selected mode: ${mode}${reasoning ? ` — ${reasoning}` : ''}`);
      } catch {
        addSystemMessage('Aria selected the next experiment mode.');
      }
    });

    es.addEventListener('continuous_limit_reached', (event) => {
      try {
        const data = JSON.parse(event.data || '{}');
        const reason = data.reason || 'limit reached';
        addSystemMessage(`Autonomous session complete: ${reason}`);
      } catch {
        addSystemMessage('Autonomous session complete.');
      }
      if (onAutonomousEnd) onAutonomousEnd();
    });

    es.addEventListener('knowledge_extracted', (event) => {
      try {
        const data = JSON.parse(event.data || '{}');
        const count = data.count || data.n_insights || 0;
        if (count > 0) {
          addSystemMessage(`Extracted ${count} insight${count === 1 ? '' : 's'} from latest results.`);
        }
      } catch { /* ignore */ }
    });

    es.onerror = () => {
      // keep silent; browser will retry
    };

    return () => es.close();
  }, [addSystemMessage, refreshAnalysis, onAutonomousEnd]);

  // Re-fetch when LLM is configured (dispatched by ControlPanel)
  useEffect(() => {
    const handler = () => {
      addSystemMessage('LLM configured. Aria is generating her first AI-powered analysis...');
      setTimeout(() => refreshAnalysis(), 500);
    };
    window.addEventListener('llm-configured', handler);
    return () => window.removeEventListener('llm-configured', handler);
  }, [addSystemMessage, refreshAnalysis]);

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
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
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
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {isRunning && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
          Run active — Aria will append analysis after completion.
        </div>
      )}

      {error && (
        <div style={{ fontSize: 11, color: 'var(--accent-yellow)', marginBottom: 8 }}>
          {error}
        </div>
      )}

      <div style={{ maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
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
                background: m.isSummary ? 'rgba(137, 87, 229, 0.08)' : m.role === 'system' ? 'var(--bg-primary)' : 'var(--bg-tertiary)',
                borderRadius: 6,
                borderLeft: `2px solid ${m.isSummary ? 'var(--accent-purple)' : m.role === 'system' ? 'var(--text-muted)' : 'var(--accent-purple)'}`,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 3 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  {m.isSummary ? 'Summary' : m.label || (m.role === 'system' ? 'System' : 'Aria')}
                </span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{formatTimestamp(m.timestamp)}</span>
              </div>
              <div style={{ fontSize: 12, lineHeight: 1.45, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>{m.text}</div>
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
          style={{
            flex: 1,
            background: 'var(--bg-primary)',
            border: '1px solid var(--border-color)',
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
        >
          {sending ? 'Sending…' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default AriaChatPanel;
