import React, { useState, useEffect, useRef } from 'react';

function formatElapsed(seconds) {
  if (!seconds || seconds <= 0) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m > 60) {
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  }
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function Stat({ label, value, color }) {
  if (value == null || value === '') return null;
  return (
    <div style={{ fontSize: 11, fontFamily: 'monospace' }}>
      <span style={{ color: 'var(--text-muted)', marginRight: 4 }}>{label}</span>
      <span style={{ color: color || 'var(--text-primary)', fontWeight: 700 }}>{value}</span>
    </div>
  );
}

export function RunStatusPanel({ isRunning, progress, onStop, programProgressText, pct, isGenerationProgress, mode }) {
  // Track last known progress so we can show it after run ends
  const lastProgressRef = useRef(null);
  const [lastCompleted, setLastCompleted] = useState(null);

  useEffect(() => {
    if (isRunning && progress?.experiment_id) {
      lastProgressRef.current = { ...progress, _snapshot_ts: Date.now() };
    }
  }, [isRunning, progress]);

  // When a run finishes, capture the final state
  useEffect(() => {
    if (!isRunning && lastProgressRef.current?.experiment_id) {
      setLastCompleted({ ...lastProgressRef.current, _ended_ts: Date.now() });
      lastProgressRef.current = null;
    }
  }, [isRunning]);

  const isNovelty = mode === 'novelty';
  const isEvolve = mode === 'evolve';

  if (isRunning) {
    const elapsed = formatElapsed(progress?.elapsed_seconds);
    const stage = progress?.current_stage || progress?.status || '';
    const s0 = progress?.stage0_passed || 0;
    const s1 = progress?.stage1_passed || 0;
    const bestLr = progress?.best_loss_ratio;
    const msg = progress?.aria_message || '';
    const expId = progress?.experiment_id?.slice(0, 8) || '';

    return (
      <div className="card" style={{ marginBottom: 16, border: '1px solid var(--accent-blue)', background: 'rgba(88, 166, 255, 0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="status-dot running" />
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <strong style={{ fontSize: 14, color: 'var(--accent-blue)' }}>
                Experiment Running
              </strong>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                {mode?.replace('_', ' ') || 'Single'} {expId && `\u2022 ${expId}`} {elapsed && `\u2022 ${elapsed}`}
              </span>
            </div>
          </div>
          <button className="start-btn" onClick={onStop} style={{ background: 'var(--accent-red)', borderColor: 'var(--accent-red)', padding: '4px 12px', fontSize: 12 }}>
            Stop
          </button>
        </div>

        {/* Progress bar */}
        <div style={{ marginBottom: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
            <span>{programProgressText}</span>
            <span>{pct}%</span>
          </div>
          <div style={{ height: 8, background: 'var(--bg-tertiary)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${pct}%`, background: 'var(--accent-blue)', transition: 'width 0.3s ease' }} />
          </div>
        </div>

        {/* Live stats grid */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: msg ? 8 : 0 }}>
          {stage && <Stat label="STAGE" value={stage.replace(/_/g, ' ')} />}
          <Stat label="S0" value={s0} color={s0 > 0 ? 'var(--accent-green)' : undefined} />
          <Stat label="S1" value={s1} color={s1 > 0 ? 'var(--accent-green)' : 'var(--text-muted)'} />
          {bestLr != null && <Stat label="BEST LR" value={bestLr.toFixed(4)} color="var(--accent-green)" />}
          {isGenerationProgress && (
            <Stat label="GEN" value={`${progress.current_generation}/${progress.total_generations}`} />
          )}
          {(isNovelty || isEvolve) && progress.best_fitness != null && (
            <Stat label="BEST FIT" value={progress.best_fitness.toFixed(3)} color="var(--accent-green)" />
          )}
          {isNovelty && progress.archive_size != null && (
            <Stat label="ARCHIVE" value={progress.archive_size} color="var(--accent-purple)" />
          )}
        </div>

        {/* Aria message */}
        {msg && (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, marginTop: 4, borderTop: '1px solid var(--border)', paddingTop: 6 }}>
            {msg.length > 200 ? msg.slice(0, 200) + '...' : msg}
          </div>
        )}
      </div>
    );
  }

  // ── Idle state: show last completed run summary ──
  if (lastCompleted) {
    const s0 = lastCompleted.stage0_passed || 0;
    const s1 = lastCompleted.stage1_passed || 0;
    const total = lastCompleted.total_programs || 0;
    const elapsed = formatElapsed(lastCompleted.elapsed_seconds);
    const expId = lastCompleted.experiment_id?.slice(0, 8) || '';
    const status = lastCompleted.status || 'completed';
    const failed = status === 'failed';
    const stopped = status === 'stopped';
    const ago = lastCompleted._ended_ts
      ? formatElapsed((Date.now() - lastCompleted._ended_ts) / 1000) + ' ago'
      : '';

    return (
      <div className="card" style={{
        marginBottom: 16,
        border: `1px solid ${failed ? 'var(--accent-red)' : stopped ? 'var(--accent-yellow)' : 'var(--border)'}`,
        background: failed ? 'rgba(248, 81, 73, 0.05)' : 'rgba(255,255,255,0.02)',
        opacity: 0.85,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="status-dot" style={{ background: failed ? 'var(--accent-red)' : stopped ? 'var(--accent-yellow)' : 'var(--accent-green)' }} />
            <strong style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Last Run: {failed ? 'Failed' : stopped ? 'Stopped' : 'Completed'}
            </strong>
          </div>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {expId} {ago && `\u2022 ${ago}`}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 11 }}>
          <Stat label="PROGRAMS" value={total} />
          <Stat label="S0" value={s0} />
          <Stat label="S1" value={s1} color={s1 > 0 ? 'var(--accent-green)' : 'var(--accent-red)'} />
          {elapsed && <Stat label="ELAPSED" value={elapsed} />}
          {lastCompleted.best_loss_ratio != null && (
            <Stat label="BEST LR" value={lastCompleted.best_loss_ratio.toFixed(4)} color="var(--accent-green)" />
          )}
        </div>
        {lastCompleted.error && (
          <div style={{ fontSize: 11, color: 'var(--accent-red)', marginTop: 6 }}>
            {lastCompleted.error.slice(0, 200)}
          </div>
        )}
      </div>
    );
  }

  return null;
}

export default RunStatusPanel;
