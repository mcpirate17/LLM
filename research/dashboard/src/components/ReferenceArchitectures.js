import React, { useMemo } from 'react';
import { lossColor, noveltyColor } from '../utils/colors';

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
        Industry standard architectures used as calibration targets for novelty and performance.
      </p>
      
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Family</th>
              <th>Loss Ratio</th>
              <th>Throughput</th>
              <th>Params</th>
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
                  {(ref.screening_loss_ratio || 0).toFixed(4)}
                </td>
                <td style={{ color: 'var(--text-secondary)' }}>
                  {ref.throughput_tok_s ? `${Math.round(ref.throughput_tok_s).toLocaleString()} /s` : '—'}
                </td>
                <td style={{ color: 'var(--text-secondary)' }}>
                  {ref.param_count ? `${(ref.param_count / 1e6).toFixed(1)}M` : '—'}
                </td>
                <td>
                  <button 
                    className="refresh-btn" 
                    style={{ fontSize: 10, padding: '2px 8px' }}
                    onClick={() => onSelectProgram && onSelectProgram(ref.result_id)}
                  >
                    View Graph
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
