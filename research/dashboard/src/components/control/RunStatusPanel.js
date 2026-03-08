import React from 'react';

export function RunStatusPanel({ isRunning, progress, onStop, programProgressText, pct, isGenerationProgress, mode }) {
  if (!isRunning) return null;

  const isNovelty = mode === 'novelty';
  const isEvolve = mode === 'evolve';

  return (
    <div className="card" style={{ marginBottom: 16, border: '1px solid var(--accent-blue)', background: 'rgba(88, 166, 255, 0.05)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="status-dot running" />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <strong style={{ fontSize: 14, color: 'var(--accent-blue)' }}>
              Experiment Running
            </strong>
            <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
              Mode: {mode?.replace('_', ' ') || 'Single'}
            </span>
          </div>
        </div>
        <button className="start-btn" onClick={onStop} style={{ background: 'var(--accent-red)', borderColor: 'var(--accent-red)', padding: '4px 12px', fontSize: 12 }}>
          Stop
        </button>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>
          <span>{programProgressText}</span>
          <span>{pct}%</span>
        </div>
        <div style={{ height: 8, background: 'var(--bg-tertiary)', borderRadius: 4, overflow: 'hidden' }}>
          <div 
            style={{ 
              height: '100%', 
              width: `${pct}%`, 
              background: 'var(--accent-blue)',
              transition: 'width 0.3s ease'
            }} 
          />
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {isGenerationProgress && (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>
            GEN: {progress.current_generation} / {progress.total_generations}
          </div>
        )}
        {(isNovelty || isEvolve) && progress.best_fitness != null && (
          <div style={{ fontSize: 11, color: 'var(--accent-green)', fontWeight: 600 }}>
            BEST FIT: {progress.best_fitness.toFixed(3)}
          </div>
        )}
        {isNovelty && progress.archive_size != null && (
          <div style={{ fontSize: 11, color: 'var(--accent-purple)', fontWeight: 600 }}>
            ARCHIVE: {progress.archive_size}
          </div>
        )}
      </div>
    </div>
  );
}

export default RunStatusPanel;
