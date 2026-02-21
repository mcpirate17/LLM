import React from 'react';
import AriaAvatar from './AriaAvatar';

function StatusBar({
  aria,
  isRunning,
  progress,
  ariaCycle,
  onCycleControl,
  cycleControlBusy,
  learningTrajectory,
  productionReadiness,
}) {
  const mood = aria?.mood || 'curious';

  // Activity text
  let activityText;
  if (isRunning && progress) {
    const expId = progress.experiment_id ? String(progress.experiment_id).slice(0, 12) : '';
    const status = progress.status || 'running';
    const current = progress.current ?? '';
    const total = progress.total ?? '';
    activityText = `Running experiment ${expId} — ${status}${current !== '' && total !== '' ? `, ${current}/${total} programs` : ''}`;
  } else if (ariaCycle && ariaCycle.continuous_active) {
    const cycleIdx = ariaCycle.cycle_index || 0;
    const mode = (ariaCycle.selected_mode || ariaCycle.last_completed_mode || 'idle');
    const phase = ariaCycle.phase_label || ariaCycle.phase || 'Idle';
    activityText = `Cycle ${cycleIdx} · ${mode} · ${phase}`;
  } else {
    const hyp = aria?.current_hypothesis;
    activityText = hyp ? `Idle — ${String(hyp).slice(0, 100)}` : 'Idle — ready for next run';
  }

  // Pipeline funnel from production_readiness
  const pr = productionReadiness;
  const screening = pr?.screening_count ?? 0;
  const investigation = pr?.investigation_count ?? 0;
  const validation = pr?.validation_count ?? pr?.decision_ready_count ?? 0;
  const breakthrough = pr?.breakthrough_count ?? 0;

  // Trend arrow from learningTrajectory
  let trendLabel = null;
  if (learningTrajectory?.trend && learningTrajectory.trend !== 'insufficient_data') {
    if (learningTrajectory.trend === 'improving') trendLabel = { arrow: '\u2197', text: 'Improving', color: 'var(--accent-green)' };
    else if (learningTrajectory.trend === 'declining') trendLabel = { arrow: '\u2198', text: 'Declining', color: 'var(--accent-red)' };
    else trendLabel = { arrow: '\u2192', text: 'Plateau', color: 'var(--accent-yellow)' };
  }

  // Cycle control button
  let cycleButton = null;
  if (ariaCycle) {
    if (!ariaCycle.continuous_active) {
      cycleButton = (
        <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 8px' }} onClick={() => onCycleControl('start')} disabled={cycleControlBusy || isRunning}>
          {cycleControlBusy ? '...' : 'Start'}
        </button>
      );
    } else if (ariaCycle.cycle_paused) {
      cycleButton = (
        <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 8px' }} onClick={() => onCycleControl('resume')} disabled={cycleControlBusy}>
          {cycleControlBusy ? '...' : 'Resume'}
        </button>
      );
    } else {
      cycleButton = (
        <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 8px' }} onClick={() => onCycleControl('pause')} disabled={cycleControlBusy}>
          {cycleControlBusy ? '...' : 'Pause'}
        </button>
      );
    }
  }

  const badgeStyle = (color) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: 12,
    fontSize: 11,
    fontWeight: 600,
    background: `${color}20`,
    color,
    border: `1px solid ${color}40`,
  });

  return (
    <div className="card status-bar" style={{ padding: '10px 14px', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        {/* Avatar + mood */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <AriaAvatar mood={mood} size={28} />
          <span style={{ fontSize: 12, color: 'var(--text-secondary)', textTransform: 'capitalize', fontWeight: 600 }}>{mood}</span>
        </div>

        {/* Activity text */}
        <div style={{ fontSize: 12, color: 'var(--text-primary)', flex: 1, minWidth: 0 }}>
          {isRunning && <span className="pulse-dot" style={{ display: 'inline-block', width: 6, height: 6, marginRight: 6, verticalAlign: 'middle' }} />}
          {activityText}
        </div>

        {/* Pipeline funnel */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0, fontSize: 11 }}>
          <span style={badgeStyle('var(--accent-blue)')}>{screening} Screening</span>
          <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
          <span style={badgeStyle('var(--accent-yellow)')}>{investigation} Investigating</span>
          <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
          <span style={badgeStyle('var(--accent-purple)')}>{validation} Validated</span>
          <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
          <span style={badgeStyle('var(--accent-green)')}>{breakthrough} Breakthrough</span>
        </div>

        {/* Cycle control */}
        {cycleButton}

        {/* Trend arrow */}
        {trendLabel && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
            <span style={{ color: trendLabel.color, fontWeight: 700, fontSize: 14 }}>{trendLabel.arrow}</span>
            <span style={{ color: trendLabel.color, fontSize: 11 }}>{trendLabel.text}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default StatusBar;
