import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

function MetricsChart({ experiments }) {
  if (!experiments || experiments.length === 0) {
    return (
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="card-title">Experiment History</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          No experiment data to chart yet.
        </p>
      </div>
    );
  }

  // Transform experiments into chart data
  const chartData = [...experiments].reverse().map((exp, i) => ({
    name: exp.experiment_id?.slice(0, 6) || `E${i}`,
    programs: exp.n_programs_generated || 0,
    survivors: exp.n_stage1_passed || 0,
    novelty: ((exp.best_novelty_score || 0) * 100).toFixed(0),
  }));

  return (
    <div className="card" style={{ gridColumn: '1 / -1' }}>
      <div className="card-title">Experiment History</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Programs generated vs. survivors per experiment. A survivor is an architecture
        that compiled, stayed numerically stable, and demonstrated learning — a potential
        alternative to transformer attention.
      </p>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#8b949e' }} />
            <YAxis tick={{ fontSize: 11, fill: '#8b949e' }} />
            <Tooltip
              contentStyle={{
                background: '#161b22',
                border: '1px solid #30363d',
                borderRadius: 8,
                fontSize: 13,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="programs" fill="#58a6ff" name="Programs Generated" />
            <Bar dataKey="survivors" fill="#3fb950" name="Stage 1 Survivors" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default MetricsChart;
