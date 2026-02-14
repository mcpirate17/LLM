import React from 'react';

function TopPrograms({ programs, compact, onSelectProgram }) {
  if (!programs || programs.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Top Programs {compact ? '(Preview)' : ''}</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          No surviving programs yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">
        Top Programs {compact ? `(${programs.length})` : `— ${programs.length} Survivors`}
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th>Fingerprint</th>
            <th>Novelty</th>
            {!compact && <th>Structural</th>}
            {!compact && <th>Behavioral</th>}
            <th>Loss Ratio</th>
            <th>Params</th>
            {!compact && <th>Similar To</th>}
            {!compact && <th>Throughput</th>}
          </tr>
        </thead>
        <tbody>
          {programs.map((p, i) => (
            <tr key={p.result_id || i}
              style={{ cursor: onSelectProgram ? 'pointer' : 'default' }}
              onClick={() => onSelectProgram && onSelectProgram(p.result_id)}>
              <td style={{ fontFamily: 'monospace', fontSize: 12, color: onSelectProgram ? 'var(--accent-blue)' : 'inherit' }}>
                {p.graph_fingerprint?.slice(0, 10) || '--'}
              </td>
              <td>
                <span className={p.novelty_score > 0.5 ? 'badge novel' : ''}>
                  {p.novelty_score?.toFixed(3) || '--'}
                </span>
              </td>
              {!compact && <td>{p.structural_novelty?.toFixed(3) || '--'}</td>}
              {!compact && <td>{p.behavioral_novelty?.toFixed(3) || '--'}</td>}
              <td>{p.loss_ratio?.toFixed(4) || '--'}</td>
              <td>{p.param_count ? `${(p.param_count / 1e6).toFixed(1)}M` : '--'}</td>
              {!compact && <td>{p.most_similar_to || '--'}</td>}
              {!compact && <td>{p.throughput_tok_s ? `${p.throughput_tok_s.toFixed(0)} tok/s` : '--'}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      {onSelectProgram && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, textAlign: 'right' }}>
          Click a row to view program details
        </div>
      )}
    </div>
  );
}

export default TopPrograms;
