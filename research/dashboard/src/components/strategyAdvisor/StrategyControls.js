import React from 'react';

const EXECUTE_LABEL = 'Execute Recommended Action';

export default function StrategyControls({
  isRunning,
  autonomousMode,
  onStopAutonomous,
  isNavigateAction,
  isActionable,
  actionLabel,
  navigateLabel,
  starting,
  startingAutonomous,
  showLimits,
  setShowLimits,
  autoMaxExperiments,
  setAutoMaxExperiments,
  autoMaxMinutes,
  setAutoMaxMinutes,
  onStart,
  handleStartClick,
  handleStartAutonomous,
  handleNavigateClick,
  // Diagnose
  diagnosing,
  diagResult,
  handleDiagnose,
  // Fallback mode
  isAiPowered,
  briefing,
  onOpenAdvancedPanel,
}) {
  return (
    <>
      {/* Action buttons section */}
      <div className="strategy-actions">
        {isRunning && autonomousMode ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start' }}>
            <div style={{ fontSize: 12, color: 'var(--accent-purple)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="pulse-dot" style={{ background: 'var(--accent-purple)' }}></span>
              Autonomous mode active — Aria is running experiments automatically.
            </div>
            <button
              className="strategy-apply-btn"
              onClick={() => onStopAutonomous && onStopAutonomous()}
              style={{
                background: 'var(--accent-red, #e74c3c)', color: '#fff', fontWeight: 600,
                fontSize: 13, padding: '6px 16px',
              }}
            >
              Stop Autonomous Mode
            </button>
          </div>
        ) : isRunning ? (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Experiment running — Aria will analyze results when complete.
          </div>
        ) : isNavigateAction ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start' }}>
            <button
              className="strategy-apply-btn"
              onClick={handleNavigateClick}
              style={{ background: 'var(--accent-green)', color: '#000', fontWeight: 600 }}
            >
              {EXECUTE_LABEL}
            </button>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Action: {navigateLabel}
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className="strategy-apply-btn"
                onClick={handleStartClick}
                disabled={starting || !onStart}
                style={{
                  background: 'var(--accent-green)', color: '#000', fontWeight: 600,
                  opacity: starting ? 0.7 : 1,
                  fontSize: 14,
                  padding: '8px 20px',
                }}
              >
                {starting ? 'Executing...' : EXECUTE_LABEL}
              </button>
              <button
                className="strategy-apply-btn"
                onClick={handleStartAutonomous}
                disabled={startingAutonomous}
                style={{
                  background: 'var(--accent-purple)', color: '#fff', fontWeight: 600,
                  opacity: startingAutonomous ? 0.7 : 1,
                  fontSize: 13,
                  padding: '8px 16px',
                }}
              >
                {startingAutonomous ? 'Starting...' : 'Start Autonomous Mode'}
              </button>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Action: {actionLabel}
            </div>
            <div>
              <button
                type="button"
                onClick={() => setShowLimits(v => !v)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontSize: 11, color: 'var(--text-muted)', padding: 0,
                  textDecoration: 'underline',
                }}
              >
                {showLimits ? 'Hide limits' : 'Autonomous limits...'}
              </button>
              {showLimits && (
                <div style={{ display: 'flex', gap: 12, marginTop: 6, alignItems: 'center' }}>
                  <label style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    Max experiments:
                    <input
                      type="number" min={1} max={100} value={autoMaxExperiments}
                      onChange={e => setAutoMaxExperiments(Math.max(1, Math.min(100, parseInt(e.target.value) || 20)))}
                      style={{
                        width: 48, marginLeft: 4, background: 'var(--bg-primary)',
                        border: '1px solid var(--border)', borderRadius: 4,
                        color: 'var(--text-primary)', fontSize: 11, padding: '2px 4px',
                      }}
                    />
                  </label>
                  <label style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    Max minutes:
                    <input
                      type="number" min={5} max={480} value={autoMaxMinutes}
                      onChange={e => setAutoMaxMinutes(Math.max(5, Math.min(480, parseInt(e.target.value) || 60)))}
                      style={{
                        width: 48, marginLeft: 4, background: 'var(--bg-primary)',
                        border: '1px solid var(--border)', borderRadius: 4,
                        color: 'var(--text-primary)', fontSize: 11, padding: '2px 4px',
                      }}
                    />
                  </label>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Diagnose & Fix */}
      <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <button
          className="strategy-apply-btn"
          onClick={handleDiagnose}
          disabled={diagnosing}
          style={{
            background: 'var(--accent-yellow)',
            color: '#000',
            fontWeight: 600,
            fontSize: 12,
            padding: '5px 12px',
            opacity: diagnosing ? 0.7 : 1,
          }}
        >
          {diagnosing ? 'Diagnosing...' : 'Diagnose & Fix'}
        </button>
        {diagResult && !diagResult.error && (
          <span style={{ fontSize: 11, color: diagResult.actions_applied?.length > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
            {diagResult.summary}
          </span>
        )}
        {diagResult?.error && (
          <span style={{ fontSize: 11, color: 'var(--accent-red, #e74c3c)' }}>
            {diagResult.error}
          </span>
        )}
      </div>
      {diagResult && diagResult.issues && diagResult.issues.length > 0 && (
        <div style={{
          marginTop: 6,
          padding: '6px 10px',
          borderRadius: 6,
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          fontSize: 11,
          lineHeight: 1.6,
        }}>
          {diagResult.issues.map((issue, i) => (
            <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
              <span style={{ color: issue.fixed ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                {issue.fixed ? '\u2713' : '\u2022'}
              </span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {issue.issue}
                {issue.fixed && <span style={{ color: 'var(--accent-green)', marginLeft: 4 }}>(fixed)</span>}
                {issue.action_type === 'info' && <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>(info)</span>}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Fallback mode notice */}
      {!isAiPowered && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span>Aria is in rule-based fallback mode ({fallbackReasonLabel(briefing?.fallback_reason)}).</span>
          <button
            className="refresh-btn"
            style={{ fontSize: 10, padding: '2px 8px' }}
            onClick={() => { if (onOpenAdvancedPanel) onOpenAdvancedPanel(); }}
          >
            Configure LLM
          </button>
        </div>
      )}
    </>
  );
}

function fallbackReasonLabel(reason) {
  if (!reason) return 'unknown';
  if (reason === 'llm_not_configured') return 'LLM not configured';
  if (reason === 'llm_unreachable') return 'LLM configured but unreachable';
  if (reason === 'llm_empty_response') return 'LLM returned no briefing text';
  if (String(reason).startsWith('llm_error:')) {
    const detail = String(reason).slice('llm_error:'.length).trim();
    return `LLM error: ${detail || 'unknown'}`;
  }
  return String(reason);
}
