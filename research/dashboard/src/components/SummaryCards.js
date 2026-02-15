import React from 'react';

function SummaryCards({ summary }) {
  if (!summary) return null;

  const survRate = summary.survival_rate || 0;
  const survColor = survRate > 0.05 ? 'green' : survRate > 0 ? 'yellow' : '';
  const novelty = summary.top_novelty_score || 0;
  const noveltyColor = novelty > 0.8 ? 'green' : novelty > 0.5 ? 'yellow' : 'purple';

  const cards = [
    {
      label: 'Experiments',
      value: summary.total_experiments,
      sub: `${summary.completed_experiments} completed`,
      color: summary.total_experiments > 5 ? 'green' : '',
    },
    {
      label: 'Programs Evaluated',
      value: summary.total_programs_evaluated,
      sub: `${summary.stage1_survivors} survivor${summary.stage1_survivors !== 1 ? 's' : ''} (learned from data)`,
      color: summary.stage1_survivors > 0 ? 'green' : '',
    },
    {
      label: 'Survival Rate',
      value: `${(survRate * 100).toFixed(1)}%`,
      sub: survRate > 0.05 ? 'Strong — many architectures learn' : survRate > 0 ? 'Some architectures learn' : 'No learnable architectures yet',
      color: survColor,
    },
    {
      label: 'Top Novelty',
      value: summary.top_novelty_score?.toFixed(3) || '—',
      sub: novelty > 0.8 ? 'Very different from known architectures' : novelty > 0.5 ? 'Moderately novel structure' : `${summary.active_insights} insights`,
      color: noveltyColor,
    },
  ];

  return (
    <div className="card">
      <div className="card-title">Research Summary</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Overall progress searching for novel LLM architectures. Programs are randomly
        synthesized computation graphs tested as alternatives to transformer attention layers.
        Survivors passed all stages and demonstrated learning ability on real data.
      </p>
      <div className="summary-grid">
        {cards.map((card, i) => (
          <div key={i} className="stat-card">
            <div className={`stat-value ${card.color}`}>{card.value}</div>
            <div className="stat-label">{card.label}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{card.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SummaryCards;
