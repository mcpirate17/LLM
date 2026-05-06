import React, { useCallback, useEffect, useState } from 'react';
import { apiService } from '../../services/apiService';

const PANEL_STYLE = {
  padding: 12,
  background: 'var(--bg-tertiary)',
  borderRadius: 6,
  border: '1px solid var(--border)',
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};

const SECTION_LABEL_STYLE = {
  fontSize: 12,
  fontWeight: 600,
  color: 'var(--text-secondary)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
};

const STAT_GRID_STYLE = {
  display: 'grid',
  gridTemplateColumns: 'repeat(5, 1fr)',
  gap: 8,
  fontSize: 12,
};

const STAT_BOX_STYLE = {
  padding: '6px 8px',
  background: 'var(--bg-secondary)',
  borderRadius: 4,
  border: '1px solid var(--border)',
};

const STAT_LABEL = { fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 };
const STAT_VALUE = { fontWeight: 600, color: 'var(--text-primary)' };

const fmt = (v, d = 3) => (v == null ? '--' : Number(v).toFixed(d));

function CountBadge({ children, color = 'var(--accent-blue)' }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 6px', fontSize: 10,
      borderRadius: 8, border: `1px solid ${color}`, color,
      background: 'transparent', marginLeft: 6, fontWeight: 600,
    }}>
      {children}
    </span>
  );
}

// Default budgets per stage — match research/defaults.py.
const STAGE_DEFAULTS = {
  screening: { steps: 750, seeds: 1, label: 'S1 confirmation', color: 'rgba(255, 184, 108, 0.45)', textColor: 'var(--score-elite)' },
  investigation: { steps: 2500, seeds: 1, label: 'Investigation', color: 'rgba(88, 166, 255, 0.45)', textColor: 'var(--accent-blue)' },
  validation: { steps: 10000, seeds: 1, label: 'Validation', color: 'rgba(63, 185, 80, 0.45)', textColor: 'var(--score-good)' },
};

const STAGE_DETAILS = {
  screening: 'exact replay: S0, S0.5, S0.75, rapid, S1',
  investigation: 'investigation tier',
  validation: 'validation tier',
};

function formatQueueError(error) {
  const message = String(error?.message || error || '');
  if (message.includes('Database temporarily busy')) {
    return 'Database is busy with another write. Retry shortly; queued tasks are still queued.';
  }
  return message;
}

function RerunQueuePanel({ resultId, leaderboardEntry }) {
  const [pending, setPending] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastSubmit, setLastSubmit] = useState(null);
  // Per-stage step inputs, each pre-populated from STAGE_DEFAULTS.  User
  // can edit any one before clicking the corresponding queue button.
  const [stageSteps, setStageSteps] = useState({
    screening: STAGE_DEFAULTS.screening.steps,
    investigation: STAGE_DEFAULTS.investigation.steps,
    validation: STAGE_DEFAULTS.validation.steps,
  });
  const [valSeeds, setValSeeds] = useState(1);

  const refresh = useCallback(async () => {
    if (!resultId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getPendingReruns(resultId);
      setPending(Array.isArray(data?.tasks) ? data.tasks : []);
    } catch (e) {
      setError(formatQueueError(e));
    } finally {
      setLoading(false);
    }
  }, [resultId]);

  useEffect(() => { refresh(); }, [refresh]);

  const queueStage = useCallback(async (stage) => {
    if (!resultId || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const resp = await apiService.queueValidationRerun(resultId, {
        stage,
        n: 1,
        reason: 'program_detail_panel',
        n_steps: stageSteps[stage],
        n_seeds: stage === 'validation' ? valSeeds : 1,
        candidate_confirmation: stage === 'screening',
      });
      setLastSubmit(resp);
      await refresh();
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setSubmitting(false);
    }
  }, [resultId, submitting, refresh, stageSteps, valSeeds]);

  const cancel = useCallback(async (taskId) => {
    if (!resultId || submitting) return;
    setSubmitting(true);
    try {
      await apiService.cancelPendingRerun(resultId, taskId);
      await refresh();
    } catch (e) {
      setError(formatQueueError(e));
    } finally {
      setSubmitting(false);
    }
  }, [resultId, submitting, refresh]);

  const drainNow = useCallback(async () => {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const resp = await apiService.drainPendingValidationRerun(resultId);
      setLastSubmit(resp);
      await refresh();
    } catch (e) {
      await refresh();
      setError(formatQueueError(e));
    } finally {
      setSubmitting(false);
    }
  }, [submitting, refresh]);

  const queuedCount = pending.filter(t => t.status === 'queued').length;
  const runningCount = pending.filter(t => t.status === 'running').length;

  const lb = leaderboardEntry || {};
  const nRuns = lb.n_runs;
  const cvLoss = lb.cv_loss;
  const cvUnd = lb.cv_understanding;
  const cvCap = lb.cv_capability;
  const stab = lb.score_stability_penalty;

  return (
    <div style={PANEL_STYLE}>
      <div style={SECTION_LABEL_STYLE}>
        <span>
          Score-Stability Reruns
          {queuedCount > 0 && <CountBadge>{queuedCount} queued</CountBadge>}
          {runningCount > 0 && <CountBadge color="var(--accent-yellow)">{runningCount} running</CountBadge>}
        </span>
        <button
          className="refresh-btn"
          onClick={refresh}
          disabled={loading}
          style={{ fontSize: 11, padding: '2px 8px' }}
          title="Refresh queue status"
        >
          {loading ? '…' : '↻'}
        </button>
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Adds independent rerun rows for this fingerprint. S1 confirmation uses
        exact replay through S0/S0.5/S0.75/rapid/S1; investigation and validation
        use their own tier pipelines. Completed rows are aggregated by mean ± std
        on the next leaderboard write; consistent metrics raise <code>n_runs</code>{' '}
        and reduce the CV penalty.
      </div>

      <div style={STAT_GRID_STYLE}>
        <div style={STAT_BOX_STYLE}>
          <div style={STAT_LABEL}>n_runs</div>
          <div style={STAT_VALUE}>{nRuns ?? '--'}</div>
        </div>
        <div style={STAT_BOX_STYLE}>
          <div style={STAT_LABEL}>CV(loss)</div>
          <div style={STAT_VALUE}>{fmt(cvLoss)}</div>
        </div>
        <div style={STAT_BOX_STYLE}>
          <div style={STAT_LABEL}>CV(und.)</div>
          <div style={STAT_VALUE}>{fmt(cvUnd)}</div>
        </div>
        <div style={STAT_BOX_STYLE}>
          <div style={STAT_LABEL}>CV(cap.)</div>
          <div style={STAT_VALUE}>{fmt(cvCap)}</div>
        </div>
        <div style={STAT_BOX_STYLE}>
          <div style={STAT_LABEL}>stability×</div>
          <div style={STAT_VALUE}>{fmt(stab)}</div>
        </div>
      </div>

      {/* Per-stage queue rows: each stage has its own steps input
          (pre-populated with the stage default) and one button.  Click
          repeatedly to enqueue multiple — each click queues one task. */}
      <div style={{
        display: 'flex', flexDirection: 'column', gap: 6,
        padding: 8, background: 'var(--bg-secondary)', borderRadius: 4,
        border: '1px solid var(--border)',
      }}>
        {['screening', 'investigation', 'validation'].map(stage => {
          const cfg = STAGE_DEFAULTS[stage];
          return (
            <div key={stage} style={{
              display: 'grid',
              gridTemplateColumns: stage === 'validation' ? '1fr 110px 70px 1fr' : '1fr 110px 1fr',
              gap: 8, alignItems: 'end',
            }}>
              <button
                onClick={() => queueStage(stage)}
                disabled={submitting}
                className="start-btn"
                style={{
                  fontSize: 12, padding: '6px 12px',
                  background: 'transparent',
                  border: `1px solid ${cfg.color}`,
                  color: cfg.textColor,
                  opacity: submitting ? 0.5 : 1,
                  borderRadius: 4, cursor: submitting ? 'not-allowed' : 'pointer',
                  textAlign: 'left',
                }}
                title={`Queue 1 ${cfg.label} run at the configured step budget — click again to queue another`}
              >
                Queue {cfg.label}
              </button>
              <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                steps
                <input
                  type="number" min={50} max={50000} step={250}
                  value={stageSteps[stage]}
                  onChange={e => {
                    const v = Math.max(50, Math.min(50000, parseInt(e.target.value) || cfg.steps));
                    setStageSteps(s => ({ ...s, [stage]: v }));
                  }}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12, marginTop: 2 }}
                />
              </label>
              {stage === 'validation' && (
                <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  seeds
                  <input
                    type="number" min={1} max={5}
                    value={valSeeds}
                    onChange={e => setValSeeds(Math.max(1, Math.min(5, parseInt(e.target.value) || 1)))}
                    style={{ width: '100%', padding: '4px 6px', fontSize: 12, marginTop: 2 }}
                  />
                </label>
              )}
              <span style={{ fontSize: 10, color: 'var(--text-muted)', alignSelf: 'center', textAlign: 'right' }}>
                {STAGE_DETAILS[stage]} · default {cfg.steps}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        {lastSubmit && lastSubmit.queued_count > 0 && (
          <span style={{ fontSize: 11, color: 'var(--score-good)' }}>
            ✓ queued {lastSubmit.stage} ({lastSubmit.n_steps} steps)
          </span>
        )}
        <button
          onClick={drainNow}
          disabled={submitting || queuedCount === 0}
          style={{
            fontSize: 12, padding: '6px 12px',
            background: 'rgba(63, 185, 80, 0.18)',
            border: '1px solid rgba(63, 185, 80, 0.55)',
            color: 'var(--score-good)',
            opacity: (submitting || queuedCount === 0) ? 0.5 : 1,
            cursor: queuedCount === 0 ? 'not-allowed' : 'pointer',
            borderRadius: 4, marginLeft: 'auto',
          }}
          title={
            queuedCount === 0
              ? 'No queued reruns to drain'
              : 'Pop the highest-priority queued validation task and start it now (works any fingerprint, not just this one)'
          }
        >
          ▶ Run next pending
        </button>
        {lastSubmit && lastSubmit.status === 'launched' && (
          <span style={{ fontSize: 11, color: 'var(--score-good)' }}>
            ✓ launched {lastSubmit.task_ids?.[0]?.slice(0, 8)}
          </span>
        )}
        {lastSubmit && lastSubmit.status === 'busy' && (
          <span style={{ fontSize: 11, color: 'var(--accent-yellow)' }}>
            runner busy ({String(lastSubmit.running_experiment_id || '?').slice(0, 8)})
          </span>
        )}
        {lastSubmit && lastSubmit.status === 'idle' && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            queue empty
          </span>
        )}
      </div>

      {error && (
        <div style={{
          padding: 6, fontSize: 11, color: 'var(--accent-red)',
          background: 'rgba(248, 81, 73, 0.1)', borderRadius: 4,
        }}>
          {error}
        </div>
      )}

      {pending.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Queue:</div>
          {pending.map(t => (
            <div
              key={t.task_id}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '4px 8px', fontSize: 11, fontFamily: 'monospace',
                background: 'var(--bg-secondary)', borderRadius: 4,
                border: '1px solid var(--border)',
              }}
            >
              <span>
                <span style={{
                  color: t.status === 'running' ? 'var(--accent-yellow)' : 'var(--text-muted)',
                  marginRight: 8,
                }}>
                  [{t.status}]
                </span>
                {t.task_id?.slice(0, 12)}
                {t.stage && (
                  <span style={{ color: STAGE_DEFAULTS[t.stage]?.textColor || 'var(--text-muted)', marginLeft: 8 }}>
                    {t.stage}{t.n_steps ? ` ${t.n_steps}` : ''}
                  </span>
                )}
                {t.source_context && (
                  <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
                    via {t.source_context}
                  </span>
                )}
              </span>
              {t.status === 'queued' && (
                <button
                  onClick={() => cancel(t.task_id)}
                  disabled={submitting}
                  style={{
                    fontSize: 10, padding: '2px 6px',
                    background: 'transparent',
                    border: '1px solid var(--accent-red)',
                    color: 'var(--accent-red)', borderRadius: 3, cursor: 'pointer',
                  }}
                  title="Cancel queued task"
                >
                  cancel
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(RerunQueuePanel);
