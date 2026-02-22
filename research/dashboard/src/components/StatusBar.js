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

  const nativeRunner = progress?.native_runner || null;
  const nativeFallbackRate = Number(nativeRunner?.fallback_metrics?.fallback_rate);
  const nativeFallbackLimitRaw = nativeRunner?.fallback_metrics?.max_allowed_fallback_rate;
  const nativeFallbackLimit = nativeFallbackLimitRaw != null ? Number(nativeFallbackLimitRaw) : null;
  const nativeLegacyUsed = Number(nativeRunner?.fallback_metrics?.legacy_compile_invocations || 0);
  const nativeLegacyLimitRaw = nativeRunner?.fallback_metrics?.max_allowed_legacy_compile_invocations;
  const nativeLegacyLimit = nativeLegacyLimitRaw != null ? Number(nativeLegacyLimitRaw) : null;
  const nativeExecPath = nativeRunner?.execution_path;
  const selectiveLayerBuild = nativeRunner?.selective_execution?.layer_build || {};
  const selectiveApplied = Number(selectiveLayerBuild.applied_layers || 0);
  const selectiveSkipped = Number(selectiveLayerBuild.skipped_layers || 0);
  const abiLastProbe = nativeRunner?.abi_last_probe || null;
  const abiParityAttempted = Boolean(abiLastProbe?.parity_attempted);
  const abiParityPass = abiLastProbe?.parity_pass;
  const abiParityMaxAbs = Number(abiLastProbe?.parity_max_abs_diff);
  const abiParityMaxAbsText = Number.isFinite(abiParityMaxAbs) ? abiParityMaxAbs.toExponential(2) : null;
  const abiParitySampleRate = Number(abiLastProbe?.parity_sample_rate);
  const abiParitySampleRateText = Number.isFinite(abiParitySampleRate) ? `${Math.round(abiParitySampleRate * 100)}%` : null;
  const abiParityThreshold = Number(abiLastProbe?.parity_max_abs_threshold);
  const abiParityThresholdText = Number.isFinite(abiParityThreshold) ? abiParityThreshold.toExponential(2) : null;
  const abiParityStrict = Boolean(abiLastProbe?.parity_strict);
  const abiBadgeColor = abiParityAttempted
    ? (abiParityPass ? 'var(--accent-green)' : 'var(--accent-red)')
    : 'var(--accent-yellow)';
  const backendCutover = nativeRunner?.cutover_gate || null;
  const backendCutoverStatus = String(backendCutover?.status || '').toLowerCase();
  const backendCutoverReady = typeof backendCutover?.ready === 'boolean' ? backendCutover.ready : null;
  const cutoverChecks = [];
  if (Number.isFinite(nativeFallbackLimit) && Number.isFinite(nativeFallbackRate)) {
    cutoverChecks.push(nativeFallbackRate <= nativeFallbackLimit);
  }
  if (Number.isFinite(nativeLegacyLimit)) {
    cutoverChecks.push(nativeLegacyUsed <= nativeLegacyLimit);
  }
  if (abiParityAttempted && abiParityPass != null) {
    cutoverChecks.push(Boolean(abiParityPass));
  }
  const localCutoverReady = cutoverChecks.length > 0 ? cutoverChecks.every(Boolean) : null;
  const cutoverReady = backendCutoverReady != null ? backendCutoverReady : localCutoverReady;
  const cutoverState = backendCutoverStatus || (
    cutoverReady == null ? 'pending' : (cutoverReady ? 'ready' : 'blocked')
  );
  const nativeSummary = nativeRunner
    ? `Native ${nativeRunner.enabled ? 'on' : 'off'}${nativeRunner.strict ? ' (strict)' : ''}${
      Number.isFinite(nativeFallbackRate) ? ` · fb ${(nativeFallbackRate * 100).toFixed(1)}%` : ''
    }`
    : null;

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

        {isRunning && nativeSummary && (
          <span
            style={badgeStyle(nativeRunner.enabled ? 'var(--accent-blue)' : 'var(--accent-red)')}
            title={nativeRunner ? JSON.stringify(nativeRunner) : ''}
          >
            {nativeSummary}
            {nativeExecPath ? ` · ${nativeExecPath}` : ''}
            {nativeRunner?.selective_execution?.layer_exec_enabled ? ` · L ${selectiveApplied}/${selectiveSkipped}` : ''}
          </span>
        )}
        {isRunning && nativeRunner && (
          <span
            style={badgeStyle(abiBadgeColor)}
            title={
              abiLastProbe
                ? `ABI parity ${abiParityStrict ? 'strict' : 'observe'}${abiParitySampleRateText ? ` | sample ${abiParitySampleRateText}` : ''}${abiParityThresholdText ? ` | max_abs<=${abiParityThresholdText}` : ''}`
                : 'No ABI probe telemetry yet.'
            }
          >
            ABI {abiParityAttempted ? (abiParityPass ? 'parity-pass' : 'parity-fail') : 'pending'}
            {abiParityMaxAbsText ? ` · ${abiParityMaxAbsText}` : ''}
          </span>
        )}
        {isRunning && nativeRunner && (
          <span style={badgeStyle(
            cutoverState === 'pending'
              ? 'var(--accent-blue)'
              : (cutoverState === 'ready' ? 'var(--accent-green)' : 'var(--accent-red)')
          )}>
            Cutover {cutoverState}
          </span>
        )}
      </div>
    </div>
  );
}

export default StatusBar;
