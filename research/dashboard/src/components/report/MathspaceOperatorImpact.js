import React from 'react';

export default function MathspaceOperatorImpact({ impact }) {
  const rows = Array.isArray(impact?.by_operator) ? impact.by_operator : [];
  const families = Array.isArray(impact?.by_family) ? impact.by_family : [];
  const topTrust = Array.isArray(impact?.top_trustworthy_operators) ? impact.top_trustworthy_operators : [];
  const totals = impact?.totals || {};

  if (!impact || impact.available === false || rows.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-title">Mathspace Operator Impact</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Which mathspace operators are most represented and how they correlate with Stage-1/validation outcomes.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Coverage:</strong>{' '}
        {totals.n_programs_with_mathspace ?? 0}/{totals.n_programs_with_graph ?? 0} programs with graph traces use mathspace ops.
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Trust score = (50% S1 pass + 30% validation pass + 20% baseline wins) × sample reliability,
        where sample reliability scales with tested count up to 25 programs.
      </div>

      {topTrust.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
          {topTrust.map(row => (
            <span
              key={row.op_name}
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 999,
                border: `1px solid ${row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)'}`,
                color: row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)',
                background: 'var(--bg-tertiary)',
              }}
            >
              {row.op_name} · trust {(Number(row.trust_score || 0) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      <div style={{ overflowX: 'auto', marginBottom: 10 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px' }}>Operator</th>
              <th style={{ padding: '6px 8px' }}>N</th>
              <th style={{ padding: '6px 8px' }}>S1 %</th>
              <th style={{ padding: '6px 8px' }}>Validation %</th>
              <th style={{ padding: '6px 8px' }}>Baseline Win %</th>
              <th style={{ padding: '6px 8px' }}>Trust %</th>
              <th style={{ padding: '6px 8px' }}>Avg Novelty</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map(row => (
              <tr key={row.op_name} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', color: 'var(--accent-blue)' }}>{row.op_name}</td>
                <td style={{ padding: '6px 8px' }}>{row.n_tested ?? 0}</td>
                <td style={{ padding: '6px 8px' }}>{((row.stage1_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{((row.validation_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{((row.baseline_win_rate || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{((row.trust_score || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{row.avg_novelty_score != null ? Number(row.avg_novelty_score).toFixed(3) : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {families.length > 0 && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-muted)' }}>
          {families.map(row => (
            <span key={row.family}>
              <strong style={{ color: 'var(--accent-purple)' }}>{row.family}:</strong> S1 {(row.stage1_pass_rate * 100).toFixed(0)}% · V {(row.validation_pass_rate * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      {impact.explanation && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>{impact.explanation}</div>
      )}
    </div>
  );
}
