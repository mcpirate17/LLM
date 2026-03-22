import React from 'react';
import AriaRecommendationPanel from './AriaRecommendationPanel';

/**
 * "Ask Aria" button, recommendation panel, and start/force-start buttons.
 */
function ExperimentActions({
  mode,
  loadingRec,
  onAskAria,
  recommendation,
  onApplyRecommendation,
  onStart,
  startLocked,
  blockedConfig,
  onForceStart,
}) {
  return (
    <>
      <div style={{ marginTop: 16 }}>
        <button
          className="refresh-btn"
          onClick={onAskAria}
          disabled={loadingRec}
          style={{ width: '100%', justifyContent: 'center', background: 'rgba(137, 87, 229, 0.1)', color: 'var(--accent-purple)', borderColor: 'rgba(137, 87, 229, 0.3)' }}
        >
          {loadingRec ? 'Aria is thinking...' : 'Ask Aria for Experiment Strategy'}
        </button>
      </div>

      <AriaRecommendationPanel recommendation={recommendation} onApply={onApplyRecommendation} />

      <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
        <button className="start-btn" onClick={onStart} disabled={startLocked} style={{ flex: 1 }}>
          {mode === 'continuous' ? 'Start Continuous Research' : 'Run Experiment'}
        </button>
        {blockedConfig && (
          <button className="start-btn" onClick={onForceStart} style={{ background: 'rgba(248, 81, 73, 0.1)', color: 'var(--accent-red)' }}>
            Force Start
          </button>
        )}
      </div>
    </>
  );
}

export default React.memo(ExperimentActions);
