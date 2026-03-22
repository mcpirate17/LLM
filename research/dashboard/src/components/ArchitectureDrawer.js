import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { parseDesignerBridgeMessage } from '../utils/designerBridge';

import {
  analyzeResearchGraph,
  analyzeWorkflowGraph,
  buildIntegrityWarning,
} from './architecture/architectureUtils';

import MorphPanel from './architecture/MorphPanel';
import LineagePanel from './architecture/LineagePanel';

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
      const res = await apiCall(`/api/designer/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_id: resultId, graph_json: graphJson }),
      });
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
    setBridgeStep('starting-services');
    setLoading(true);

    const checkDesigner = apiCall('/api/designer/ensure-running', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force_restart: false }),
      signal: abortController.signal,
      timeoutMs: 60000,
    }).then(res => res.json().then(payload => {
      if (!res.ok || payload?.ok === false) {
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
    }));

    const fetchSource = resultId
      ? apiCall(`/api/programs/${resultId}`, { signal: abortController.signal }).then(r => r.json())
      : Promise.resolve(null);

    Promise.all([checkDesigner, fetchSource])
      .then(([_, sourceData]) => {
        if (abortController.signal.aborted) return;
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
  }, [resultId]);

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

    // Only use the standalone Vite server when the dashboard itself is running
    // from CRA dev-server (port 3000). Otherwise, force same-origin proxy.
    const onDashboardDevServer = window.location.port === '3000';
    // Chrome 128+ blocks 0.0.0.0 for iframe/subresource requests (PNA).
    const safeOrigin = window.location.hostname === '0.0.0.0'
      ? `${window.location.protocol}//localhost:${window.location.port}`
      : window.location.origin;
    const base = onDashboardDevServer
      ? 'http://localhost:5174/'
      : new URL('/designer-proxy/', safeOrigin).toString();

    return `${base.replace(/\/?$/, '/')}?${params.toString()}`;
  }, [resultId, readOnly]);

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
