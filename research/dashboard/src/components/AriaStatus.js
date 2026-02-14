import React from 'react';
import AriaAvatar from './AriaAvatar';

function AriaStatus({ aria }) {
  if (!aria) return <div className="card aria-card"><p>Waiting for connection...</p></div>;

  return (
    <div className="card aria-card">
      <div className="card-title">Aria's Status</div>

      <div className="aria-mood">
        <AriaAvatar mood={aria.mood} size={80} />
        <div>
          <span className="mood-label">{aria.mood}</span>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <span className="badge running">{aria.research_focus}</span>
            {aria.llm_enabled && (
              <span className="badge novel" style={{ fontSize: 10 }}>LLM</span>
            )}
          </div>
        </div>
      </div>

      {aria.current_hypothesis && (
        <div className="aria-hypothesis">
          "{aria.current_hypothesis}"
        </div>
      )}

      <div className="aria-stats">
        <div className="aria-stat">
          <span className="aria-stat-label">Experiments today</span>
          <span>{aria.experiments_today}</span>
        </div>
        <div className="aria-stat">
          <span className="aria-stat-label">Discoveries</span>
          <span>{aria.discoveries_today}</span>
        </div>
        <div className="aria-stat">
          <span className="aria-stat-label">Energy</span>
          <span>{(aria.energy * 100).toFixed(0)}%</span>
        </div>
        <div className="aria-stat">
          <span className="aria-stat-label">Focus</span>
          <span>{aria.research_focus}</span>
        </div>
      </div>

      {aria.recent_insights && aria.recent_insights.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Recent Insights:
          </div>
          {aria.recent_insights.slice(0, 3).map((insight, i) => (
            <div key={i} style={{ fontSize: 12, color: 'var(--text-muted)', padding: '2px 0' }}>
              • {insight.length > 80 ? insight.slice(0, 80) + '...' : insight}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default AriaStatus;
