import React, { useState, useEffect } from 'react';
import { apiCall } from '../../services/apiService';

export function StrategyBacktest() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    apiCall('/api/analytics/strategy-backtest')
      .then(r => r.json())
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(e => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="card"><div className="ux-state-loading"><span className="ux-spinner" /></div></div>;
  if (error) return <div className="card" style={{ color: 'var(--accent-red)' }}>Error: {error}</div>;
  if (!data || !data.intents || data.intents.length === 0) return null;

  const intents = data.intents;
  const maxS1 = Math.max(...intents.map(i => i.s1_pass_rate), 0.01);

  return (
    <div className="card">
      <div className="card-title">Strategy Backtest</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.5 }}>
        Aggregated outcomes by search intent. This tracks whether "Novelty Search" actually finds more diverse
        architectures than "Balanced" or "Quality" search, and which intent leads to the highest S1 pass rate.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {intents.map(i => (
          <div key={i.intent} style={{ display: 'grid', gridTemplateColumns: '100px 1fr auto', gap: 16, alignItems: 'center' }}>
            <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'capitalize', color: 'var(--text-primary)' }}>
              {i.intent}
            </div>
            <div style={{ position: 'relative', height: 20, background: 'var(--bg-tertiary)', borderRadius: 10, overflow: 'hidden' }}>
              <div 
                style={{ 
                  height: '100%', 
                  width: `${(i.s1_pass_rate / maxS1) * 100}%`,
                  background: 'var(--accent-blue)',
                  opacity: 0.7,
                  transition: 'width 0.6s ease-out'
                }} 
              />
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', paddingLeft: 10, fontSize: 10, fontWeight: 600, color: '#fff', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>
                S1 Pass: {(i.s1_pass_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', textAlign: 'right', minWidth: 120 }}>
              {i.n_experiments} exps · {i.n_programs} progs
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 20, maxHeight: 300, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Intent</th>
              <th>Avg Best Loss</th>
              <th>Avg Novelty</th>
              <th>Throughput</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {intents.map(i => (
              <tr key={i.intent}>
                <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>{i.intent}</td>
                <td style={{ color: i.avg_best_loss < 0.6 ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                  {i.avg_best_loss != null ? i.avg_best_loss.toFixed(4) : '--'}
                </td>
                <td style={{ color: i.avg_best_novelty > 0.5 ? 'var(--accent-purple)' : 'var(--text-primary)' }}>
                  {i.avg_best_novelty != null ? i.avg_best_novelty.toFixed(3) : '--'}
                </td>
                <td style={{ fontSize: 11 }}>
                  {i.avg_throughput != null ? `${Math.round(i.avg_throughput).toLocaleString()} t/s` : '--'}
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {Math.round(i.avg_duration / 60)} min
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default StrategyBacktest;
