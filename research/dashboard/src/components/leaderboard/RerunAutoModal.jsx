import React, { useCallback, useEffect, useState } from 'react';
import { apiService } from '../../services/apiService';

const BACKDROP = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  zIndex: 9999,
};

const PANEL = {
  width: 720, maxWidth: '92vw', maxHeight: '88vh',
  background: 'var(--bg-primary)', borderRadius: 8,
  border: '1px solid var(--border)', boxShadow: '0 12px 40px rgba(0,0,0,0.45)',
  display: 'flex', flexDirection: 'column',
};

const HEADER = {
  padding: '12px 16px', borderBottom: '1px solid var(--border)',
  fontWeight: 600, fontSize: 13, display: 'flex',
  alignItems: 'center', justifyContent: 'space-between',
};

const BODY = { flex: 1, overflowY: 'auto', padding: 16 };
const FOOTER = {
  padding: '10px 16px', borderTop: '1px solid var(--border)',
  display: 'flex', gap: 8, justifyContent: 'flex-end',
};

const CONTROL_ROW = {
  display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12,
};

const TABLE_STYLE = {
  width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: 'monospace',
};
const TH = {
  textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)',
  color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', fontSize: 10,
};
const TD = { padding: '4px 6px', borderBottom: '1px solid var(--border)' };

const fmt = (v, d = 1) => (v == null ? '--' : Number(v).toFixed(d));
const short = (s, n = 12) => String(s || '').slice(0, n);

function RerunAutoModal({ open, onClose }) {
  const [topN, setTopN] = useState(15);
  const [n, setN] = useState(2);
  const [nCap, setNCap] = useState(4);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [applied, setApplied] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    setApplied(null);
    try {
      const params = `?top_n=${topN}&n=${n}&n_runs_cap=${nCap}`;
      const data = await apiService.previewQueueRerunAuto(params);
      setReport(data);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [topN, n, nCap]);

  useEffect(() => {
    if (open) {
      refresh();
    } else {
      setReport(null);
      setApplied(null);
      setError(null);
    }
  }, [open, refresh]);

  const apply = useCallback(async () => {
    if (!report || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const data = await apiService.applyQueueRerunAuto({
        top_n: topN,
        n,
        n_runs_cap: nCap,
      });
      setApplied(data);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setSubmitting(false);
    }
  }, [report, submitting, topN, n, nCap]);

  if (!open) return null;

  const eligible = report?.eligible || [];
  const totalTasksToQueue = eligible.length * n;

  return (
    <div style={BACKDROP} onClick={onClose}>
      <div style={PANEL} onClick={e => e.stopPropagation()}>
        <div style={HEADER}>
          <span>Auto Queue Score-Stability Reruns</span>
          <button onClick={onClose} className="close-btn">&times;</button>
        </div>
        <div style={BODY}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.55 }}>
            Flags fingerprints whose 95% upper-bound CI on composite score
            reaches the score at rank <code>top_n</code> — i.e. they are in
            "striking distance" of being top <code>top_n</code> after a
            confirmation rerun. Excludes fingerprints already at
            <code> n_runs ≥ n_runs_cap</code>. Sigma is the per-row
            tier-weighted CV(composite) when observed (n≥2), or a cohort
            prior CV otherwise.
          </div>

          <div style={CONTROL_ROW}>
            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Boundary rank (top_n)
              <input
                type="number" min={1} max={50} value={topN}
                onChange={e => setTopN(parseInt(e.target.value) || 15)}
                style={{ width: '100%', padding: '4px 6px', fontSize: 12, marginTop: 2 }}
              />
            </label>
            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Reruns per fp (n)
              <input
                type="number" min={1} max={5} value={n}
                onChange={e => setN(parseInt(e.target.value) || 1)}
                style={{ width: '100%', padding: '4px 6px', fontSize: 12, marginTop: 2 }}
              />
            </label>
            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Total runs cap
              <input
                type="number" min={2} max={10} value={nCap}
                onChange={e => setNCap(parseInt(e.target.value) || 4)}
                style={{ width: '100%', padding: '4px 6px', fontSize: 12, marginTop: 2 }}
              />
            </label>
          </div>

          {loading && <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading…</div>}
          {error && (
            <div style={{
              padding: 8, fontSize: 12, color: 'var(--accent-red)',
              background: 'rgba(248, 81, 73, 0.1)', borderRadius: 4, marginBottom: 12,
            }}>
              {error}
            </div>
          )}

          {report && !loading && (
            <>
              <div style={{
                padding: 8, fontSize: 11, marginBottom: 8,
                background: 'var(--bg-tertiary)', borderRadius: 4,
                border: '1px solid var(--border)',
                display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8,
              }}>
                <span>
                  <span style={{ color: 'var(--text-muted)' }}>Boundary @ rank {topN}: </span>
                  <strong>{fmt(report.boundary_top_n)}</strong>
                </span>
                <span>
                  <span style={{ color: 'var(--text-muted)' }}>Prior CV: </span>
                  <strong>{fmt(report.prior_cv, 3)}</strong>
                  <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>
                    ({report.prior_cv_source?.source})
                  </span>
                </span>
                <span>
                  <span style={{ color: 'var(--text-muted)' }}>Eligible: </span>
                  <strong>{eligible.length}</strong>
                  <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>
                    → {totalTasksToQueue} tasks if applied
                  </span>
                </span>
              </div>

              {applied && (
                <div style={{
                  padding: 8, fontSize: 11, marginBottom: 8,
                  background: 'rgba(63, 185, 80, 0.10)', borderRadius: 4,
                  border: '1px solid var(--score-good)',
                  color: 'var(--score-good)',
                }}>
                  ✓ Queued {applied.queued?.length || 0} fingerprint(s)
                  ({applied.queued?.reduce((acc, q) => acc + (q.task_ids?.length || 0), 0)} task(s))
                </div>
              )}

              <table style={TABLE_STYLE}>
                <thead>
                  <tr>
                    <th style={TH}>Fingerprint</th>
                    <th style={TH}>Tier</th>
                    <th style={TH}>n</th>
                    <th style={TH}>Composite</th>
                    <th style={TH}>± σ</th>
                    <th style={TH}>Upper95</th>
                    <th style={TH}>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {eligible.slice(0, 200).map(e => (
                    <tr key={e.graph_fingerprint}>
                      <td style={TD}>{short(e.graph_fingerprint)}</td>
                      <td style={TD}>{e.tier}</td>
                      <td style={TD}>{e.n_runs}</td>
                      <td style={TD}>{fmt(e.composite)}</td>
                      <td style={TD}>{fmt(e.sigma)}</td>
                      <td style={{ ...TD, fontWeight: 600 }}>{fmt(e.upper_bound_95)}</td>
                      <td style={{ ...TD, color: 'var(--text-muted)' }}>
                        {e.sigma_source}
                      </td>
                    </tr>
                  ))}
                  {eligible.length > 200 && (
                    <tr>
                      <td style={TD} colSpan={7}>
                        … {eligible.length - 200} more (will all be queued on apply)
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </>
          )}
        </div>
        <div style={FOOTER}>
          <button onClick={refresh} disabled={loading || submitting} className="refresh-btn"
            style={{ fontSize: 12, padding: '6px 14px' }}>
            Refresh preview
          </button>
          <button onClick={onClose} className="refresh-btn"
            style={{ fontSize: 12, padding: '6px 14px' }}>
            Close
          </button>
          <button
            onClick={apply}
            disabled={!report || submitting || eligible.length === 0 || !!applied}
            className="start-btn"
            style={{
              fontSize: 12, padding: '6px 14px',
              background: 'rgba(88, 166, 255, 0.18)',
              border: '1px solid rgba(88, 166, 255, 0.55)',
              color: 'var(--accent-blue)',
              opacity: (submitting || eligible.length === 0 || applied) ? 0.5 : 1,
            }}
          >
            {submitting ? 'Queueing…' : applied ? 'Queued' : `Apply (queue ${totalTasksToQueue})`}
          </button>
        </div>
      </div>
    </div>
  );
}

export default RerunAutoModal;
