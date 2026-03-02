import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { parseDesignerBridgeMessage } from '../utils/designerBridge';

import {
  DESIGNER_BASE,
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
  const iframeRef = useRef(null);
  const pendingGraphRequestRef = useRef({ reason: null, requestId: null });

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
          // Tell the designer to load the graph for this result
          sendToDesigner('load-result', { resultId });
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

  // Initial load
  useEffect(() => {
    if (!resultId) return;
    setLoading(true);
    setBridgeStep('fetching-source');
    apiCall(`/api/programs/${resultId}`)
      .then((r) => r.json())
      .then((data) => {
        setGraphInfo(data);
        setSourceGraphCheck(analyzeResearchGraph(data.graph_json_parsed));
        setBridgeStep('loading-iframe');
        setBooting(false);
      })
      .catch((err) => {
        setError(`Failed to fetch source graph: ${err.message}`);
        setLoading(false);
        setBooting(false);
      });
  }, [resultId]);

  useEffect(() => {
    setIntegrityWarning(buildIntegrityWarning(sourceGraphCheck, designerGraphCheck));
  }, [sourceGraphCheck, designerGraphCheck]);

  const designerUrl = useMemo(() => {
    const params = new URLSearchParams({
      result_id: resultId,
      mode: readOnly ? 'readonly' : 'edit',
      embedded: '1',
      origin: window.location.origin,
    });
    
    // In development (when dashboard is on port 3000), 
    // try to talk to the Vite dev server directly.
    // Otherwise, use the production proxy served by the Python backend.
    const isDev = window.location.port === '3000' || window.location.hostname === 'localhost';
    const base = isDev ? 'http://localhost:5174' : DESIGNER_BASE;
    
    return `${base}/?${params.toString()}`;
  }, [resultId, readOnly]);

  return (
    <div className="arch-drawer">
      <div className="arch-drawer-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="close-btn" onClick={onClose}>&times;</button>
          <div className="ux-stack">
            <span className="ux-state-title">Architecture Viewer</span>
            <span className="ux-state-subtle">result: {resultId?.slice(0, 12)}</span>
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
        </div>
      </div>

      <div className="arch-drawer-body">
        {booting ? (
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
  );
}

export default ArchitectureDrawer;
