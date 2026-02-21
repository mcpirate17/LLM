import React, { useState, useEffect, useRef, useCallback } from 'react';

const DESIGNER_BASE = process.env.REACT_APP_DESIGNER_URL || 'http://127.0.0.1:5174';
const API_BASE = process.env.REACT_APP_API_URL || '';

const INTENTS = [
  { key: 'balanced', label: 'Balanced', color: 'var(--text-secondary)' },
  { key: 'quality', label: 'Quality', color: 'var(--accent-green)' },
  { key: 'compression', label: 'Compression', color: 'var(--accent-blue)' },
  { key: 'sparsity', label: 'Sparsity', color: 'var(--accent-purple)' },
  { key: 'novelty', label: 'Novelty', color: 'var(--accent-orange, #e88d3f)' },
];

/**
 * MorphPanel — Smart Morph UI for generating intent-driven mutations.
 *
 * Calls POST /api/programs/{resultId}/morph to generate scored candidates,
 * then displays them as selectable cards with op diffs and score breakdowns.
 */
function MorphPanel({ resultId, onSelectCandidate }) {
  const [intent, setIntent] = useState('balanced');
  const [candidates, setCandidates] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sourceOps, setSourceOps] = useState([]);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCandidates(null);
    try {
      const res = await fetch(`${API_BASE}/api/programs/${resultId}/morph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent, n_candidates: 6, use_analysis: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setCandidates(data.candidates || []);
      setSourceOps(data.source_ops || []);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [resultId, intent]);

  const intentColor = INTENTS.find(i => i.key === intent)?.color || 'var(--text-secondary)';

  return (
    <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)', maxHeight: 320, overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>Smart Morph</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {INTENTS.map(i => (
            <button
              key={i.key}
              onClick={() => setIntent(i.key)}
              style={{
                fontSize: 10,
                padding: '2px 8px',
                borderRadius: 10,
                border: intent === i.key ? `1.5px solid ${i.color}` : '1px solid var(--border)',
                background: intent === i.key ? 'var(--bg-tertiary)' : 'none',
                color: intent === i.key ? i.color : 'var(--text-muted)',
                cursor: 'pointer',
                fontWeight: intent === i.key ? 600 : 400,
              }}
            >
              {i.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            marginLeft: 'auto',
            fontSize: 11,
            padding: '3px 12px',
            borderRadius: 4,
            border: `1px solid ${intentColor}`,
            background: 'none',
            color: intentColor,
            cursor: loading ? 'wait' : 'pointer',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Generating\u2026' : 'Generate'}
        </button>
      </div>

      {error && (
        <div style={{ fontSize: 11, color: 'var(--accent-red)', marginBottom: 6 }}>{error}</div>
      )}

      {candidates && candidates.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: 12 }}>
          No valid mutations generated. Try a different intent.
        </div>
      )}

      {candidates && candidates.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {candidates.map((c, idx) => (
            <div
              key={c.fingerprint}
              onClick={() => onSelectCandidate && onSelectCandidate(c)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '6px 10px',
                borderRadius: 6,
                border: '1px solid var(--border)',
                background: 'var(--bg-secondary)',
                cursor: 'pointer',
                transition: 'border-color 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = intentColor}
              onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
            >
              {/* Rank */}
              <span style={{
                fontSize: 14, fontWeight: 700, color: intentColor,
                minWidth: 22, textAlign: 'center',
              }}>
                #{idx + 1}
              </span>

              {/* Score */}
              <div style={{ minWidth: 48, textAlign: 'center' }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {(c.score * 100).toFixed(0)}
                </div>
                <div style={{ fontSize: 8, color: 'var(--text-muted)', textTransform: 'uppercase' }}>score</div>
              </div>

              {/* Stats */}
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', minWidth: 70 }}>
                <div>{c.n_ops} ops, d={c.depth}</div>
                <div>{(c.params_estimate / 1000).toFixed(0)}K params</div>
              </div>

              {/* Op diff */}
              <div style={{ flex: 1, display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                {c.added_ops.map(op => (
                  <span key={`+${op}`} style={{
                    fontSize: 9, padding: '1px 5px', borderRadius: 3,
                    background: 'rgba(80,200,120,0.15)', color: 'var(--accent-green)',
                  }}>+{op}</span>
                ))}
                {c.removed_ops.map(op => (
                  <span key={`-${op}`} style={{
                    fontSize: 9, padding: '1px 5px', borderRadius: 3,
                    background: 'rgba(255,100,100,0.15)', color: 'var(--accent-red)',
                  }}>-{op}</span>
                ))}
                {c.added_ops.length === 0 && c.removed_ops.length === 0 && (
                  <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>same ops, different wiring</span>
                )}
              </div>

              {/* Score breakdown mini-bars */}
              <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                {Object.entries(c.score_breakdown || {}).map(([k, v]) => (
                  <div key={k} title={`${k}: ${(v * 100).toFixed(0)}%`} style={{
                    width: 4, height: Math.max(4, v * 28),
                    background: intentColor, borderRadius: 2, opacity: 0.5 + v * 0.5,
                  }} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LineagePanel({ resultId }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadLineage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/designer/lineage?limit=20`);
      const data = await res.json().catch(() => []);
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setRows(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err?.message || String(err));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadLineage();
  }, [loadLineage, resultId]);

  return (
    <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)', maxHeight: 250, overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
          Run Lineage
        </span>
        <button
          onClick={loadLineage}
          disabled={loading}
          style={{
            marginLeft: 'auto',
            fontSize: 10,
            padding: '2px 8px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
          }}
        >
          Refresh
        </button>
      </div>
      {loading && <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Loading lineage…</div>}
      {error && <div style={{ fontSize: 11, color: 'var(--accent-red)' }}>{error}</div>}
      {!loading && !error && rows.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>No designer run lineage yet.</div>
      )}
      {!loading && !error && rows.slice(0, 8).map((row) => (
        <div
          key={row.run_id}
          style={{
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '6px 8px',
            marginBottom: 6,
            background: 'var(--bg-secondary)',
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-primary)', fontWeight: 600 }}>{row.run_id}</div>
          <div style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
            workflow: {row.workflow_id || '-'} • status: {row.status || 'unknown'}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            fingerprint: {row.graph_fingerprint || '-'}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * ArchitectureDrawer — Slide-out panel embedding aria-designer in an iframe.
 *
 * Opens from the right side (70% width) to show the visual graph editor
 * for a specific result_id from the research pipeline.
 *
 * PostMessage bridge:
 *   Receives from iframe: graph-loaded, graph-changed, graph-data
 *   Sends to iframe: get-graph
 *
 * Props:
 *   resultId  — research pipeline result_id to import and display
 *   onClose   — callback to close the drawer
 *   readOnly  — if true, loads designer in readonly mode (default: true)
 *   onGraphLoaded — callback when graph finishes loading in designer
 */
function ArchitectureDrawer({ resultId, onClose, readOnly = true, onGraphLoaded }) {
  const [loading, setLoading] = useState(true);
  const [booting, setBooting] = useState(true);
  const [designerReady, setDesignerReady] = useState(false);
  const [error, setError] = useState(null);
  const [graphInfo, setGraphInfo] = useState(null);
  const [showMorph, setShowMorph] = useState(false);
  const [showLineage, setShowLineage] = useState(false);
  const iframeRef = useRef(null);

  const iframeSrc = resultId
    ? `${DESIGNER_BASE}?import_result_id=${encodeURIComponent(resultId)}&embedded=1${readOnly ? '&readonly=1' : ''}`
    : null;

  // Auto-start designer backend
  useEffect(() => {
    setLoading(true);
    setBooting(true);
    setDesignerReady(false);
    setError(null);
    setGraphInfo(null);

    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/designer/ensure-running`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force_restart: false }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data?.ok === false) {
          throw new Error(data?.error || `HTTP ${res.status}`);
        }
        if (!cancelled) {
          setDesignerReady(true);
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Could not auto-start Aria Designer: ${err?.message || err}`);
          setLoading(false);
        }
      } finally {
        if (!cancelled) {
          setBooting(false);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [resultId]);

  // Keep designer alive while drawer is open; idle policy can stop it after close.
  useEffect(() => {
    if (!designerReady || !resultId) return undefined;
    let cancelled = false;

    const touch = async () => {
      try {
        await fetch(`${API_BASE}/api/designer/touch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'architecture-drawer-open' }),
        });
      } catch (e) {
        if (!cancelled) {
          // Non-blocking keepalive; ignore transient failures.
        }
      }
    };

    touch();
    const interval = setInterval(touch, 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [designerReady, resultId]);

  // PostMessage listener — receive events from aria-designer iframe
  useEffect(() => {
    const handler = (e) => {
      if (e.data?.source !== 'aria-designer') return;

      switch (e.data.type) {
        case 'graph-loaded':
          setLoading(false);
          setGraphInfo({
            name: e.data.name,
            nodeCount: e.data.nodeCount,
            edgeCount: e.data.edgeCount,
          });
          if (onGraphLoaded) onGraphLoaded(e.data);
          break;
        case 'graph-changed':
          setGraphInfo(prev => prev ? {
            ...prev,
            nodeCount: e.data.nodeCount,
            edgeCount: e.data.edgeCount,
          } : null);
          break;
        case 'graph-data':
          // Future: handle graph data responses
          break;
        default:
          break;
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [onGraphLoaded]);

  // Send command to iframe
  const sendToDesigner = useCallback((type, payload = {}) => {
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage(
        { target: 'aria-designer', type, ...payload },
        '*'
      );
    }
  }, []);

  const handleIframeLoad = () => {
    // Loading spinner stays until we get graph-loaded postMessage
  };

  const handleIframeError = () => {
    setLoading(false);
    setError('Could not connect to Aria Designer after auto-start.');
  };

  const handleOpenFull = () => {
    if (iframeSrc) {
      window.open(
        `${DESIGNER_BASE}?import_result_id=${encodeURIComponent(resultId)}`,
        '_blank'
      );
    }
  };

  const handleGetGraph = () => {
    sendToDesigner('get-graph');
  };

  const handleSelectMorphCandidate = useCallback((candidate) => {
    // Load the morph candidate's workflow into the designer via postMessage
    const wf = candidate.workflow_json || candidate.graph_json;
    if (wf) {
      sendToDesigner('load-graph', { graphJson: wf });
    }
  }, [sendToDesigner]);

  return (
    <div className="arch-drawer-backdrop" onClick={onClose}>
      <div className="arch-drawer" onClick={e => e.stopPropagation()}>
        <div className="arch-drawer-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 16 }}>Architecture Viewer</span>
            <span style={{
              fontSize: 10,
              color: 'var(--text-muted)',
              background: 'var(--bg-tertiary)',
              padding: '2px 6px',
              borderRadius: 4,
            }}>
              {resultId}
            </span>
            {graphInfo && (
              <span style={{
                fontSize: 10,
                color: 'var(--text-secondary)',
              }}>
                {graphInfo.nodeCount} nodes, {graphInfo.edgeCount} edges
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button
              onClick={() => setShowLineage(prev => !prev)}
              title="Run lineage and graph fingerprint history"
              style={{
                background: showLineage ? 'var(--bg-tertiary)' : 'none',
                border: `1px solid ${showLineage ? 'var(--accent-blue)' : 'var(--border)'}`,
                color: showLineage ? 'var(--accent-blue)' : 'var(--text-secondary)',
                fontSize: 11,
                padding: '3px 8px',
                borderRadius: 4,
                cursor: 'pointer',
                fontWeight: showLineage ? 600 : 400,
              }}
            >
              Lineage
            </button>
            <button
              onClick={() => setShowMorph(prev => !prev)}
              title="Smart Morph: generate intent-driven mutations"
              style={{
                background: showMorph ? 'var(--bg-tertiary)' : 'none',
                border: `1px solid ${showMorph ? 'var(--accent-purple)' : 'var(--border)'}`,
                color: showMorph ? 'var(--accent-purple)' : 'var(--text-secondary)',
                fontSize: 11,
                padding: '3px 8px',
                borderRadius: 4,
                cursor: 'pointer',
                fontWeight: showMorph ? 600 : 400,
              }}
            >
              Morph
            </button>
            <button
              onClick={handleGetGraph}
              title="Get current graph JSON"
              style={{
                background: 'none',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
                fontSize: 11,
                padding: '3px 8px',
                borderRadius: 4,
                cursor: 'pointer',
              }}
            >
              Export Graph
            </button>
            {/* Window control buttons — minimize (hide panel), maximize (open full), close */}
            <div style={{ display: 'flex', gap: 2, marginLeft: 8 }}>
              <button
                onClick={handleOpenFull}
                title="Open in full Aria Designer"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: 14,
                  width: 28,
                  height: 28,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  borderRadius: 4,
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-tertiary)'}
                onMouseLeave={e => e.currentTarget.style.background = 'none'}
              >
                {/* Maximize icon — two overlapping squares */}
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.3">
                  <rect x="2.5" y="2.5" width="7" height="7" rx="1" />
                  <path d="M4.5 2.5V1.5C4.5 1 5 .5 5.5 .5H10.5C11 .5 11.5 1 11.5 1.5V6.5C11.5 7 11 7.5 10.5 7.5H9.5" />
                </svg>
              </button>
              <button
                onClick={onClose}
                title="Close"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: 14,
                  width: 28,
                  height: 28,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  borderRadius: 4,
                }}
                onMouseEnter={e => { e.currentTarget.style.background = '#c42b1c'; e.currentTarget.style.color = '#fff'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--text-muted)'; }}
              >
                {/* Close icon — X */}
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 2L10 10M10 2L2 10" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {(booting || loading) && (
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', justifyContent: 'center', alignItems: 'center',
              background: 'var(--bg-primary)',
              color: 'var(--text-secondary)',
              fontSize: 13,
              zIndex: 1,
            }}>
              {booting ? 'Starting Aria Designer\u2026' : 'Loading architecture...'}
            </div>
          )}

          {error && (
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', flexDirection: 'column',
              justifyContent: 'center', alignItems: 'center',
              background: 'var(--bg-primary)',
              color: 'var(--accent-red)',
              fontSize: 13,
              gap: 8,
              zIndex: 1,
            }}>
              <div>{error}</div>
            </div>
          )}

          {iframeSrc && designerReady && (
            <iframe
              ref={iframeRef}
              src={iframeSrc}
              onLoad={handleIframeLoad}
              onError={handleIframeError}
              title="Aria Designer"
              style={{
                width: '100%',
                height: '100%',
                border: 'none',
                background: 'var(--bg-primary)',
              }}
            />
          )}
        </div>

        {showMorph && (
          <MorphPanel
            resultId={resultId}
            onSelectCandidate={handleSelectMorphCandidate}
          />
        )}
        {showLineage && <LineagePanel resultId={resultId} />}
      </div>
    </div>
  );
}

export default ArchitectureDrawer;
