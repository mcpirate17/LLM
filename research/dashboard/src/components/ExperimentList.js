import React from 'react';

function formatTime(timestamp) {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleString();
}

function formatDuration(seconds) {
  if (!seconds) return '--';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function ExperimentList({ experiments, onSelectExperiment }) {
  if (!experiments || experiments.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiments</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          No experiments yet. Run: python -m research --mode=synthesize --n 100
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Experiments</div>
      <table className="data-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Status</th>
            <th>Programs</th>
            <th>S1 Pass</th>
            <th>Best Loss</th>
            <th>Best Novelty</th>
            <th>Mood</th>
            <th>Duration</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map(exp => (
            <tr key={exp.experiment_id}
              style={{ cursor: onSelectExperiment ? 'pointer' : 'default' }}
              onClick={() => onSelectExperiment && onSelectExperiment(exp.experiment_id)}>
              <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
                {exp.experiment_id}
              </td>
              <td>{exp.experiment_type}</td>
              <td>
                <span className={`badge ${exp.status === 'completed' ? 'pass' :
                  exp.status === 'running' ? 'running' : 'fail'}`}>
                  {exp.status}
                </span>
              </td>
              <td>{exp.n_programs_generated || 0}</td>
              <td>{exp.n_stage1_passed || 0}</td>
              <td>{exp.best_loss_ratio?.toFixed(3) || '--'}</td>
              <td>{exp.best_novelty_score?.toFixed(3) || '--'}</td>
              <td>{exp.aria_mood || '--'}</td>
              <td>{formatDuration(exp.duration_seconds)}</td>
              <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {formatTime(exp.timestamp)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {experiments.length > 0 && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, textAlign: 'right' }}>
          {onSelectExperiment ? 'Click a row to view details | ' : ''}
          Showing {experiments.length} most recent experiments
        </div>
      )}
    </div>
  );
}

export default ExperimentList;
