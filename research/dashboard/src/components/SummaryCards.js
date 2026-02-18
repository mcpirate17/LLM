import React from 'react';

function SummaryCards({ summary, learningTrend }) {
  if (!summary) return null;

  const survRate = summary.survival_rate || 0;
  const survColor = survRate > 0.05 ? 'green' : survRate > 0 ? 'yellow' : '';
  const novelty = summary.top_novelty_score || 0;
  const noveltyColor = novelty > 0.8 ? 'green' : novelty > 0.5 ? 'yellow' : 'purple';

  // Build pass rate sub-text with trend arrow
  let passRateSub = survRate > 0.05 ? 'Strong throughput' : survRate > 0 ? 'Some candidates pass' : 'No passing candidates yet';
  let trendIndicator = null;
  if (learningTrend && learningTrend.trend && learningTrend.trend !== 'insufficient_data') {
    const arrow = learningTrend.trend === 'improving' ? '\u2191'
      : learningTrend.trend === 'declining' ? '\u2193' : '\u2192';
    const trendColor = learningTrend.trend === 'improving' ? 'var(--accent-green)'
      : learningTrend.trend === 'declining' ? 'var(--accent-red, #e74c3c)' : 'var(--accent-yellow)';
    const slopeStr = learningTrend.slope != null
      ? `${learningTrend.slope > 0 ? '+' : ''}${(learningTrend.slope * 100).toFixed(2)}%/exp`
      : '';
    trendIndicator = { arrow, trendColor, slopeStr };
  }

  const cards = [
    {
      label: 'Runs',
      value: summary.total_experiments,
      sub: `${summary.completed_experiments} completed`,
      color: summary.total_experiments > 5 ? 'green' : '',
    },
    {
      label: 'Candidates Tested',
      value: summary.total_programs_evaluated,
      sub: `${summary.stage1_survivors} survivor${summary.stage1_survivors !== 1 ? 's' : ''}`,
      color: summary.stage1_survivors > 0 ? 'green' : '',
    },
    {
      label: 'Pass Rate',
      value: `${(survRate * 100).toFixed(1)}%`,
      sub: passRateSub,
      color: survColor,
      trend: trendIndicator,
    },
    {
      label: 'Novelty Peak',
      value: summary.top_novelty_score?.toFixed(3) || '—',
      sub: novelty > 0.8 ? 'High structural novelty' : novelty > 0.5 ? 'Moderate novelty' : `${summary.active_insights} active insights`,
      color: noveltyColor,
    },
  ];

  return (
    <div className="card">
      <div className="card-title">Research Summary</div>
      <div className="summary-grid">
        {cards.map((card, i) => (
          <div key={i} className="stat-card">
            <div className={`stat-value ${card.color}`}>{card.value}</div>
            <div className="stat-label">{card.label}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{card.sub}</div>
            {card.trend && (
              <div style={{ fontSize: 11, marginTop: 2, display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ color: card.trend.trendColor, fontWeight: 700, fontSize: 13 }}>{card.trend.arrow}</span>
                <span style={{ color: card.trend.trendColor, fontSize: 10 }}>{card.trend.slopeStr}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default SummaryCards;
