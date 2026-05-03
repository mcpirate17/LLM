import { apiCall, postJson } from "../services/apiService";
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { parseDesignerBridgeMessage } from '../utils/designerBridge';

import {
  analyzeResearchGraph,
  analyzeWorkflowGraph,
  buildIntegrityWarning,
} from './architecture/architectureUtils';

import MorphPanel from './architecture/MorphPanel';
import LineagePanel from './architecture/LineagePanel';

function GraphHealthCard({ label, check, status, detail }) {
  const ok = check ? check.hasInputPath && check.deadNodeCount === 0 : status === 'ready';
  const color = check == null && status !== 'ready'
    ? 'var(--text-muted)'
    : ok
      ? 'var(--accent-green)'
      : 'var(--accent-yellow)';
  return (
    <div style={{
      minWidth: 0,
      padding: '8px 10px',
      borderRadius: 6,
      border: '1px solid var(--border)',
      background: 'var(--bg-tertiary)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 650 }}>{label}</span>
        <span style={{ fontSize: 10, color, fontWeight: 700, textTransform: 'uppercase' }}>
          {check ? (ok ? 'connected' : 'check') : (status || 'pending')}
        </span>
      </div>
      {check ? (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-secondary)' }}>
          <span>{check.nodeCount} nodes</span>
          <span>{check.edgeCount} edges</span>
          <span style={{ color: check.deadNodeCount === 0 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
            {check.deadNodeCount} unreachable
          </span>
        </div>
      ) : (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {detail || 'Waiting for graph data'}
        </div>
      )}
    </div>
  );
}

/**
 * ArchitectureDrawer — Slide-out panel embedding aria_designer in an iframe.
 */
function ArchitectureDrawer({ resultId, onClose, readOnly = true, onGraphLoaded, onActionComplete }) {
  const [loading, setLoading] = useState(true);
  const [booting, setBooting] = useState(true);
  const [designerReady, setDesignerReady] = useState(false);
  const [bridgeReady, setBridgeReady] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [graphInfo, setGraphInfo] = useState(null);
  const [sourceGraphCheck, setSourceGraphCheck] = useState(null);
  const [designerGraphCheck, setDesignerGraphCheck] = useState(null);
  const [integrityWarning, setIntegrityWarning] = useState(null);
  const [showMorph, setShowMorph] = useState(false);
  const [showLineage, setShowLineage] = useState(false);
  const [bridgeStep, setBridgeStep] = useState('booting');
  const [committing, setCommitting] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [designerBaseUrl, setDesignerBaseUrl] = useState(null);
  const iframeRef = useRef(null);
  const pendingGraphRequestRef = useRef({ reason: null, requestId: null });
  const loadResultSentRef = useRef(false);
  const prevResultIdRef = useRef(resultId);
  const [startingDesigner, setStartingDesigner] = useState(true);
  const [drawerWidth, setDrawerWidth] = useState(70);
  const resizeRef = useRef({ startX: 0, startWidth: 70 });
  const resizingRef = useRef(false);
  const [resizing, setResizing] = useState(false);

  // Drag-to-resize from left edge
  useEffect(() => {
    if (!resizing) return undefined;
    const onMove = (e) => {
      const delta = resizeRef.current.startX - e.clientX;
      const nextVw = resizeRef.current.startWidth + (delta / window.innerWidth) * 100;
      setDrawerWidth(Math.max(35, Math.min(95, nextVw)));
    };
    const onUp = () => {
      resizingRef.current = false;
      setResizing(false);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [resizing]);

  // Reset load guard when resultId changes so a new graph gets loaded
  if (prevResultIdRef.current !== resultId) {
    prevResultIdRef.current = resultId;
    loadResultSentRef.current = false;
  }

  // Send command to iframe
  const sendToDesigner = useCallback((type, payload = {}) => {
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage(
        { target: 'aria_designer', type, ...payload },
        '*'
      );
    }
  }, []);

  const requestGraph = useCallback((reason) => {
    const requestId = `${reason}-${Date.now()}`;
    pendingGraphRequestRef.current = { reason, requestId };
    sendToDesigner('get-graph', { reason, requestId });
  }, [sendToDesigner]);

  const commitToResearch = useCallback(async (graphJson) => {
    setCommitting(true);
    setNotice('Committing changes to research pipeline...');
    try {
      const res = await postJson('/api/designer/commit', { result_id: resultId, graph_json: graphJson });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Commit failed');
      setNotice('Graph committed successfully.');
      if (onActionComplete) onActionComplete('commit', data);
      setTimeout(() => setNotice(null), 3000);
    } catch (err) {
      setError(`Commit failed: ${err.message}`);
    } finally {
      setCommitting(false);
    }
  }, [resultId, onActionComplete]);

  // Handle incoming messages from iframe
  useEffect(() => {
    const handleMessage = (event) => {
      const msg = parseDesignerBridgeMessage(event.data);
      if (!msg || msg.kind === 'ignore') return;

      switch (msg.kind) {
        case 'embedded-ready':
          setDesignerReady(true);
          setBridgeStep('ready');
          // Tell the designer to load the graph for this result (once only).
          // In blank-canvas mode there is no source result to load.
          if (resultId && !loadResultSentRef.current) {
            loadResultSentRef.current = true;
            sendToDesigner('load-result', { resultId });
          }
          break;
        case 'graph-loaded':
          setLoading(false);
          setDesignerGraphCheck(analyzeWorkflowGraph(msg.payload));
          if (onGraphLoaded) onGraphLoaded(msg.payload);
          break;
        case 'graph-load-error':
          setLoading(false);
          setError(msg.error || 'Designer failed to load graph');
          break;
        case 'graph-data':
          if (msg.payload?.requestId === pendingGraphRequestRef.current.requestId) {
            if (pendingGraphRequestRef.current.reason === 'commit') {
              commitToResearch(msg.payload);
            }
            pendingGraphRequestRef.current = { reason: null, requestId: null };
          }
          break;
        default:
          break;
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [commitToResearch, onGraphLoaded, sendToDesigner, resultId]);

  // Ensure designer services and fetch source graph in parallel.
  useEffect(() => {
    const abortController = new AbortController();

    setBooting(true);
    setError(null);
    setBridgeStep(readOnly ? 'loading-iframe' : 'starting-services');
    setLoading(true);

    const checkDesigner = postJson('/api/designer/ensure-running', { force_restart: false }, {
      signal: abortController.signal,
      timeoutMs: 60000,
    }).then(res => res.json().then(payload => {
      if (!res.ok || payload?.ok === false) {
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      return payload;
    }));

    const fetchSource = resultId
      ? apiCall(`/api/programs/${resultId}`, { signal: abortController.signal }).then(r => r.json())
      : Promise.resolve(null);

    Promise.all([checkDesigner, fetchSource])
      .then(([designerPayload, sourceData]) => {
        if (abortController.signal.aborted) return;
        const lifecycleUrl = designerPayload?.status?.ui_health_url;
        setDesignerBaseUrl(lifecycleUrl || null);
        setStartingDesigner(false);
        if (sourceData) {
          setGraphInfo(sourceData);
          setSourceGraphCheck(analyzeResearchGraph(sourceData.graph_json_parsed));
        } else {
          setGraphInfo(null);
          setSourceGraphCheck(null);
        }
        setBridgeStep(sourceData ? 'loading-iframe' : 'ready');
        setBooting(false);
        if (!sourceData) setLoading(false);
      })
      .catch((err) => {
        if (abortController.signal.aborted) return;
        setStartingDesigner(false);
        setError(`Failed to initialize: ${err.message}`);
        setLoading(false);
        setBooting(false);
      });

    return () => { abortController.abort(); };
  }, [readOnly, resultId]);

  useEffect(() => {
    setIntegrityWarning(buildIntegrityWarning(sourceGraphCheck, designerGraphCheck));
  }, [sourceGraphCheck, designerGraphCheck]);

  const designerUrl = useMemo(() => {
    const params = new URLSearchParams({
      mode: readOnly ? 'readonly' : 'edit',
      readonly: readOnly ? '1' : '0',
      embedded: '1',
      origin: window.location.origin,
    });
    if (resultId) params.set('result_id', resultId);

    const base = designerBaseUrl || new URL('/designer-proxy/', window.location.origin).toString();

    return `${base.replace(/\/?$/, '/')}?${params.toString()}`;
  }, [designerBaseUrl, resultId, readOnly]);

  return (
    <>
    <div className="arch-drawer-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }} />
    <div
      className={`arch-drawer${fullscreen ? ' arch-drawer-fullscreen' : ''}`}
      style={fullscreen ? undefined : { width: `${drawerWidth}vw` }}
    >
      {!fullscreen && (
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            resizeRef.current = { startX: e.clientX, startWidth: drawerWidth };
            resizingRef.current = true;
            setResizing(true);
          }}
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: 0,
            width: 8,
            cursor: 'col-resize',
            background: resizing ? 'rgba(88, 166, 255, 0.2)' : 'transparent',
            borderLeft: '1px solid rgba(88, 166, 255, 0.35)',
            zIndex: 2,
          }}
          title="Drag to resize"
          aria-hidden="true"
        />
      )}
      <div className="arch-drawer-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="ux-stack">
            <span className="ux-state-title">Architecture Viewer</span>
            <span className="ux-state-subtle">
              {resultId ? `result: ${resultId.slice(0, 12)}` : 'blank canvas'}
            </span>
          </div>
        </div>
        <div className="header-actions">
          {notice && <span className="notice-text">{notice}</span>}
          {!readOnly && (
            <button
              className="start-btn"
              disabled={committing || !designerReady || loading}
              onClick={() => requestGraph('commit')}
            >
              {committing ? 'Saving...' : 'Commit to Research'}
            </button>
          )}
          <button
            className="refresh-btn"
            onClick={() => setShowMorph(!showMorph)}
            style={{
              background: showMorph ? 'var(--bg-tertiary)' : 'none',
              border: `1px solid ${showMorph ? 'var(--accent-blue)' : 'var(--border)'}`,
              color: showMorph ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            Smart Morph
          </button>
          <button
            className="refresh-btn"
            onClick={() => setShowLineage(!showLineage)}
            style={{
              background: showLineage ? 'var(--bg-tertiary)' : 'none',
              border: `1px solid ${showLineage ? 'var(--accent-blue)' : 'var(--border)'}`,
              color: showLineage ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            Lineage
          </button>
          <button
            className="refresh-btn"
            onClick={() => {
              const next = !fullscreen;
              setFullscreen(next);
              sendToDesigner('set-embedded', { embedded: !next });
            }}
            title={fullscreen ? 'Exit fullscreen' : 'Expand to fullscreen'}
            style={{ fontSize: 16, padding: '4px 8px' }}
          >
            {fullscreen ? '\u2750' : '\u2922'}
          </button>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
      </div>

      <div className="arch-drawer-body">
        {(booting || startingDesigner) ? (
          <div className="ux-state ux-state-loading">
            <span className="ux-spinner" />
            <div className="ux-stack">
              <span className="ux-state-title">Booting bridge</span>
              <span className="ux-state-subtle">{bridgeStep}...</span>
            </div>
          </div>
        ) : (
          <>
            {error && <div className="error-banner">{error}</div>}
            {integrityWarning && <div className="warn-banner">{integrityWarning}</div>}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: 8,
              marginBottom: 10,
            }}>
              <GraphHealthCard label="Backend Graph" check={sourceGraphCheck} status={graphInfo ? 'loaded' : 'pending'} />
              <GraphHealthCard label="Viewer Graph" check={designerGraphCheck} status={loading ? 'loading' : designerReady ? 'ready' : bridgeStep} />
              <GraphHealthCard
                label="Bridge"
                status={bridgeReady || designerReady ? 'ready' : bridgeStep}
                detail={readOnly ? 'Read-only embedded viewer' : 'Editable designer session'}
              />
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', flex: 1 }}>
              <div style={{ flex: 1, position: 'relative', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                <iframe
                  ref={iframeRef}
                  src={designerUrl}
                  title="Aria Designer"
                  className="designer-iframe"
                  style={{ opacity: loading ? 0.5 : 1, flex: 1 }}
                />
                {loading && (
                  <div className="iframe-loader">
                    <span className="ux-spinner" />
                  </div>
                )}
              </div>

              {showMorph && (
                <MorphPanel 
                  resultId={resultId} 
                  onSelectCandidate={(c) => {
                    sendToDesigner('load-graph', { graphJson: c.workflow_json });
                    setNotice(`Applied mutation #${c.fingerprint?.slice(0, 6)} to canvas.`);
                    setTimeout(() => setNotice(null), 3000);
                  }} 
                />
              )}
              
              {showLineage && <LineagePanel resultId={resultId} />}
            </div>
          </>
        )}
      </div>
    </div>
    </>
  );
}

export default ArchitectureDrawer;
