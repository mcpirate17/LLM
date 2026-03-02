import React from 'react';

export function DataPopulateBar({ learningTrajectory, controlComparison, onStartExperiment }) {
  if (!onStartExperiment) return null;

  const nExperiments = learningTrajectory?.n_experiments || 0;
  const hasEnoughData = nExperiments >= 10
    && controlComparison?.status !== 'insufficient_data';
  if (hasEnoughData) return null;

  const controlNeeded = controlComparison?.status === 'insufficient_data';
  const nControlExps = controlComparison?.control?.experiments || 0;
  const nLearnedExps = controlComparison?.learned?.experiments || 0;

  return (
    <div className="card" style={{
      padding: '14px 16px',
      borderLeft: '3px solid var(--accent-blue)',
      background: 'var(--bg-secondary)',
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
        More experiments needed to populate learning analytics
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 10 }}>
        {nExperiments < 5
          ? `You have ${nExperiments} experiment${nExperiments === 1 ? '' : 's'}. At least 5 are needed for trajectory analysis, and control experiments run automatically every 5th continuous experiment.`
          : nExperiments < 10
            ? `${nExperiments} experiments completed. More data will improve trajectory analysis and statistical significance.`
            : ''}
        {controlNeeded && nExperiments >= 5 && (
          <span> Control comparison needs {'\u2265'}2 control + {'\u2265'}2 learned experiments (currently {nControlExps} control, {nLearnedExps} learned). Controls run automatically every 5th continuous experiment.</span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <button
          className="refresh-btn"
          style={{ fontSize: 12, padding: '5px 14px', fontWeight: 600 }}
          onClick={() => onStartExperiment({
            mode: 'continuous', n_cycles: 5, source: 'learning_panel',
            auto_harden: true, preflight_override: true, enforce_preflight: true,
          })}
        >
          Run 5 Continuous
        </button>
        <button
          className="refresh-btn"
          style={{ fontSize: 12, padding: '5px 14px', fontWeight: 600 }}
          onClick={() => onStartExperiment({
            mode: 'continuous', n_cycles: 10, source: 'learning_panel',
            auto_harden: true, preflight_override: true, enforce_preflight: true,
          })}
        >
          Run 10 Continuous
        </button>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {nExperiments} experiment{nExperiments === 1 ? '' : 's'} so far
        </span>
      </div>
    </div>
  );
}

export default DataPopulateBar;
