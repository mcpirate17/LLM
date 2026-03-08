import React, { useState, useMemo, useEffect } from 'react';
import { formatBenchmarkValue } from '../../utils/format';

const BenchmarkTargets = ({ benchmarkMetrics, benchmarkObserved, onBenchmarkObservedChange }) => {
  const [benchmarkInputDraft, setBenchmarkInputDraft] = useState('');
  const [benchmarkInputError, setBenchmarkInputError] = useState('');

  const benchmarkInputCount = useMemo(() => {
    const src = benchmarkObserved && typeof benchmarkObserved === 'object' ? benchmarkObserved : {};
    return Object.entries(src).filter(([, v]) => Number.isFinite(Number(v))).length;
  }, [benchmarkObserved]);

  useEffect(() => {
    const src = benchmarkObserved && typeof benchmarkObserved === 'object' ? benchmarkObserved : {};
    setBenchmarkInputDraft(JSON.stringify(src, null, 2));
  }, [benchmarkObserved]);

  const handleApplyBenchmarkObserved = () => {
    if (typeof onBenchmarkObservedChange !== 'function') return;
    try {
      const parsed = benchmarkInputDraft.trim() ? JSON.parse(benchmarkInputDraft) : {};
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setBenchmarkInputError('Input must be a JSON object of metric keys to numeric values.');
        return;
      }
      const cleaned = {};
      for (const [k, v] of Object.entries(parsed)) {
        const n = Number(v);
        if (Number.isFinite(n)) cleaned[k] = n;
      }
      onBenchmarkObservedChange(cleaned);
      setBenchmarkInputError('');
    } catch (err) {
      setBenchmarkInputError(`Invalid JSON: ${err.message}`);
    }
  };

  const handleResetBenchmarkObserved = () => {
    if (typeof onBenchmarkObservedChange !== 'function') return;
    onBenchmarkObservedChange({});
    setBenchmarkInputError('');
  };

  if (!benchmarkMetrics) return null;

  const notMeasuredTargets = Array.isArray(benchmarkMetrics.targets)
    ? benchmarkMetrics.targets.filter((t) => t.status === 'not_measured')
    : [];

  return (
    <>
      <div style={{ marginBottom: 8, padding: '8px 10px', border: '1px solid #1f3147', borderRadius: 8, background: '#0f1928' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
          <strong style={{ fontSize: 12, color: '#d8e6f5' }}>External Benchmark Inputs</strong>
          <span style={{ fontSize: 11, color: '#8fa8c2' }}>{benchmarkInputCount} loaded</span>
        </div>
        <textarea
          value={benchmarkInputDraft}
          onChange={(e) => setBenchmarkInputDraft(e.target.value)}
          spellCheck={false}
          className="benchmark-textarea"
          placeholder={`{
  "mmlu_5shot": 67.1,
  "humaneval_0shot": 61.2
}`}
        />
        {benchmarkInputError && <div className="field-error-msg">{benchmarkInputError}</div>}
        <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
          <button type="button" className="btn-small" onClick={handleApplyBenchmarkObserved}>Apply Inputs</button>
          <button type="button" className="btn-small btn-secondary" onClick={handleResetBenchmarkObserved}>Clear</button>
        </div>
      </div>

      <div className="metrics-grid" style={{ marginBottom: 8 }}>
        <div className="stat">
          <div className="stat-val">{Math.round((benchmarkMetrics.summary?.score || 0) * 100)}%</div>
          <div className="stat-label">Target Score</div>
        </div>
        <div className="stat">
          <div className="stat-val">{benchmarkMetrics.summary?.on_target ?? 0}</div>
          <div className="stat-label">On Target</div>
        </div>
        <div className="stat">
          <div className="stat-val">{benchmarkMetrics.summary?.off_target ?? 0}</div>
          <div className="stat-label">Off Target</div>
        </div>
      </div>

      {Array.isArray(benchmarkMetrics.targets) && benchmarkMetrics.targets.length > 0 && (
        <div style={{ maxHeight: 220, overflowY: 'auto', border: '1px solid #1f3147', borderRadius: 8 }}>
          <table className="op-profile-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Observed</th>
                <th>Target</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkMetrics.targets.map((target) => (
                <tr key={target.id}>
                  <td>{target.label}</td>
                  <td>{formatBenchmarkValue(target.observed, target.unit)}</td>
                  <td>{formatBenchmarkValue(target.target, target.unit)}</td>
                  <td>
                    <span className={`status-text status-${target.status}`}>
                      {target.status.replace('_', ' ')}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {benchmarkMetrics.scaling_projection && (
        <div className="scaling-projection">
          <div>Projected accuracy: <strong>{benchmarkMetrics.scaling_projection.projected_mamba_avg_accuracy?.toFixed(2) ?? '-'}</strong></div>
          <div>Delta vs Mamba-2.8B: <strong>{benchmarkMetrics.scaling_projection.delta_vs_mamba_2p8b_avg?.toFixed(2) ?? '-'}</strong></div>
        </div>
      )}
    </>
  );
};

export default BenchmarkTargets;
