import React, { useMemo } from 'react';
import { lossColor, noveltyColor } from '../utils/colors';
import { scoreColor, scoreGradient, scoreToneLabel } from '../utils/format';

/**
 * ReferenceArchitectures — Task 3G
 * 
 * Displays GPT-2, Mamba, RWKV, RAG baselines as comparison targets.
 */
export function ReferenceArchitectures({ leaderboardEntries, onSelectProgram }) {
  const references = useMemo(() => {
    return leaderboardEntries
      .filter(e => e.is_reference)
      .sort((a, b) => (a.reference_name || '').localeCompare(b.reference_name || ''));
  }, [leaderboardEntries]);

  const fmt = (value, digits = 4) => {
    if (value == null) return '—';
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(digits) : '—';
  };

  if (references.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Reference Architectures</div>
        <p className="ux-state ux-state-empty">
          No reference architectures found in the database.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Reference Baselines</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
        Post-BPE calibration targets for novelty and performance. Score colors use the same champion scale as Discoveries.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10, marginBottom: 14 }}>
        {references.map((ref) => (
          <button
            key={`card-${ref.result_id}`}
            onClick={() => onSelectProgram && onSelectProgram(ref.result_id)}
            style={{
              textAlign: 'left',
              padding: '10px 12px',
              borderRadius: 6,
              border: `1px solid ${scoreColor(ref.composite_score)}`,
              background: 'rgba(13, 17, 23, 0.42)',
              color: 'var(--text-primary)',
              cursor: 'pointer',
            }}
            title={`${ref.reference_name || 'Reference'}: ${scoreToneLabel(ref.composite_score)}`}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'baseline' }}>
              <strong>{ref.reference_name || 'Unnamed Ref'}</strong>
              <span style={{ color: scoreColor(ref.composite_score), fontFamily: 'monospace', fontWeight: 700 }}>
                {fmt(ref.composite_score, 1)}
              </span>
            </div>
            <div className="champion-strip" style={{ margin: '6px 0' }}>
              <div
                className="champion-strip-fill"
                style={{
                  width: `${Number.isFinite(Number(ref.composite_score)) ? Math.max(4, Math.min(100, (Number(ref.composite_score) / 320) * 100)) : 4}%`,
                  background: scoreGradient(ref.composite_score),
                }}
              />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {ref.architecture_family || 'Unknown'} · {ref.param_count ? `${(ref.param_count / 1e6).toFixed(1)}M params` : 'params unknown'}
            </div>
          </button>
        ))}
      </div>
      
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Family</th>
              <th>Discovery</th>
              <th>Validation</th>
              <th>Score</th>
              <th>Novelty</th>
              <th>Throughput</th>
              <th>Params</th>
              <th>LongCtx</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {references.map((ref) => (
              <tr key={ref.result_id}>
                <td style={{ fontWeight: 600, color: 'var(--accent-purple)' }}>
                  {ref.reference_name || 'Unnamed Ref'}
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                  {ref.architecture_family || 'Unknown'}
                </td>
                <td style={{ color: lossColor(ref.screening_loss_ratio) }}>
                  {fmt(ref.screening_loss_ratio)}
                </td>
                <td style={{ color: lossColor(ref.validation_loss_ratio) }}>
                  {fmt(ref.validation_loss_ratio)}
                </td>
                <td style={{ color: scoreColor(ref.composite_score), fontWeight: 700 }}>
                  {fmt(ref.composite_score, 3)}
                </td>
                <td style={{ color: noveltyColor(ref.screening_novelty) }}>
                  {fmt(ref.screening_novelty, 3)}
                </td>
                <td style={{ color: 'var(--text-secondary)' }}>
                  {ref.throughput_tok_s ? `${Math.round(ref.throughput_tok_s).toLocaleString()} /s` : '—'}
                </td>
                <td style={{ color: 'var(--text-secondary)' }}>
                  {ref.param_count ? `${(ref.param_count / 1e6).toFixed(1)}M` : '—'}
                </td>
                <td style={{ color: 'var(--text-secondary)' }}>
                  {ref.robustness_long_ctx_score != null ? fmt(ref.robustness_long_ctx_score, 3) : '—'}
                </td>
                <td>
                  <button 
                    className="refresh-btn" 
                    style={{ fontSize: 10, padding: '2px 8px' }}
                    onClick={() => onSelectProgram && onSelectProgram(ref.result_id)}
                  >
                    Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ReferenceArchitectures;
