import React, { Suspense, useState } from 'react';
import HelpPanel from '../HelpPanel';
import ProgramDetail from '../ProgramDetail';
import { postJson } from '../../services/apiService';

function OverlayFrame({ onClose, maxWidth, children, closeLabel, top = 60 }) {
  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0,0,0,0.5)',
        zIndex: 1000,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'flex-start',
        paddingTop: top,
        overflow: 'auto',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--bg-primary)',
          borderRadius: 12,
          maxWidth,
          width: '90%',
          maxHeight: `calc(100vh - ${top * 2}px)`,
          overflow: 'auto',
          padding: 24,
          position: 'relative',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: 12,
            right: 12,
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            fontSize: 20,
            cursor: 'pointer',
            lineHeight: 1,
          }}
          aria-label={closeLabel}
        >
          &times;
        </button>
        {children}
      </div>
    </div>
  );
}

export function ChatDrawer({
  open,
  onClose,
  isRunning,
  autonomousMode,
  onAutonomousEnd,
  fallback,
  AriaChatPanelComponent,
}) {
  if (!open || !AriaChatPanelComponent) return null;

  return (
    <div className="chat-drawer-backdrop" onClick={onClose}>
      <div className="chat-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="chat-drawer-header">
          <span>Aria Chat</span>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 20,
              cursor: 'pointer',
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>
        <div style={{ flex: 1, overflow: 'auto' }}>
          <Suspense fallback={fallback}>
            <AriaChatPanelComponent
              isRunning={isRunning}
              autonomousMode={autonomousMode}
              onAutonomousEnd={onAutonomousEnd}
            />
          </Suspense>
        </div>
      </div>
    </div>
  );
}

export function SettingsOverlay({
  open,
  onClose,
  overrideIneligibleAlways,
  setOverrideIneligibleAlways,
  strategyBlocksAdvancedStart,
  strategyLockReason,
  onAllowAdvancedStartOverride,
  controlPanelProps,
  ControlPanelComponent,
}) {
  const [ablationStarting, setAblationStarting] = useState(false);
  const [ablationMessage, setAblationMessage] = useState('');

  if (!open || !ControlPanelComponent) return null;

  return (
    <OverlayFrame onClose={onClose} maxWidth={800} closeLabel="Close settings">
      <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Experiment Settings</div>
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 14,
          fontSize: 12,
          color: 'var(--text-secondary)',
        }}
      >
        <input
          type="checkbox"
          checked={overrideIneligibleAlways}
          onChange={(event) => setOverrideIneligibleAlways(Boolean(event.target.checked))}
        />
        Always allow override for ineligible fingerprints (Investigate/Validate)
      </label>
      {strategyBlocksAdvancedStart && (
        <div
          style={{
            marginBottom: 16,
            padding: '8px 10px',
            borderRadius: 6,
            border: '1px solid var(--accent-yellow)',
            background: 'rgba(210, 153, 34, 0.12)',
            fontSize: 12,
            color: 'var(--text-secondary)',
            lineHeight: 1.5,
          }}
        >
          <div style={{ marginBottom: 6 }}>{strategyLockReason}</div>
          <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 8px' }} onClick={onAllowAdvancedStartOverride}>
            Use advanced setup anyway
          </button>
        </div>
      )}
      <div
        style={{
          marginBottom: 16,
          padding: '10px 12px',
          borderRadius: 6,
          border: '1px solid var(--border)',
          background: 'var(--bg-tertiary)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>Continuous Ablations</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
              cycles=5 · interval=2 · top=1 · variants=4
            </div>
          </div>
          <button
            className="start-btn"
            disabled={ablationStarting}
            onClick={async () => {
              setAblationStarting(true);
              setAblationMessage('');
              try {
                const res = await postJson('/api/ablations/bulk/start', {
                  max_experiments: 5,
                  n_programs: 40,
                  interval: 2,
                  top_k: 1,
                  max_signals: 2,
                  max_graphs: 4,
                  model_source: 'mixed',
                }, { timeoutMs: 120000 });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                  setAblationMessage(data.error || 'Failed to start continuous ablations');
                } else {
                  setAblationMessage(`Started ${data.run_id || 'continuous'} with causal ablations`);
                }
              } catch (error) {
                setAblationMessage(`Error: ${error.message}`);
              } finally {
                setAblationStarting(false);
              }
            }}
            style={{ padding: '6px 12px', fontSize: 12, whiteSpace: 'nowrap' }}
            title="Start a bounded continuous run that periodically launches causal ablation suites"
          >
            {ablationStarting ? 'Starting...' : 'Start Ablation Run'}
          </button>
        </div>
        {ablationMessage && (
          <div style={{ marginTop: 8, fontSize: 11, color: ablationMessage.startsWith('Error') || ablationMessage.startsWith('Failed') ? 'var(--accent-red)' : 'var(--accent-green)' }}>
            {ablationMessage}
          </div>
        )}
      </div>
      <Suspense fallback={<div style={{ padding: 20, color: 'var(--text-muted)', fontSize: 13 }}>Loading settings...</div>}>
        <ControlPanelComponent {...controlPanelProps} />
      </Suspense>
    </OverlayFrame>
  );
}

export function HelpOverlay({ open, onClose }) {
  if (!open) return null;

  return (
    <OverlayFrame onClose={onClose} maxWidth={720} closeLabel="Close help">
      <HelpPanel />
    </OverlayFrame>
  );
}

export function ProgramDetailOverlay({ resultId, fallback, ...props }) {
  if (!resultId) return null;

  return (
    <Suspense fallback={fallback}>
      <ProgramDetail resultId={resultId} {...props} />
    </Suspense>
  );
}

export function DesignerDrawerOverlay({
  open,
  resultId,
  onClose,
  fallback,
  ArchitectureDrawerComponent,
  readOnly = true,
}) {
  if (!open || !ArchitectureDrawerComponent) return null;

  return (
    <Suspense fallback={fallback}>
      <ArchitectureDrawerComponent resultId={resultId} onClose={onClose} readOnly={readOnly} />
    </Suspense>
  );
}
