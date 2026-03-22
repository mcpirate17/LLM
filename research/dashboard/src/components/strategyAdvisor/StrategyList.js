import React from 'react';
import { TIER_COLORS } from '../../utils/scoringEngine';

function PipelineBadge({ label, count, color }) {
  return (
    <div className="pipeline-badge" style={{ borderColor: color }}>
      <span className="pipeline-count" style={{ color }}>{count}</span>
      <span className="pipeline-label">{label}</span>
    </div>
  );
}

function LearningTrendBadge({ trend, onNavigate }) {
  const t = trend.trend;
  const color = t === 'improving' ? 'var(--accent-green)'
    : t === 'declining' ? 'var(--accent-red, #e74c3c)'
    : 'var(--accent-yellow)';
  const arrow = t === 'improving' ? '\u2191' : t === 'declining' ? '\u2193' : '\u2192';
  const label = t === 'improving' ? 'Improving' : t === 'declining' ? 'Declining' : 'Plateaued';
  const slopeStr = trend.slope != null ? `${trend.slope > 0 ? '+' : ''}${(trend.slope * 100).toFixed(2)}%/exp` : '';
  const s1Str = trend.recent_s1_rate != null ? `${(trend.recent_s1_rate * 100).toFixed(1)}% recent S1` : '';
  return (
    <div
      className="pipeline-badge"
      onClick={() => onNavigate && onNavigate('learning')}
      style={{
        borderColor: color,
        minWidth: 90,
        cursor: onNavigate ? 'pointer' : 'default'
      }}
      title="View detailed learning trajectory"
    >
      <span style={{ color, fontWeight: 700, fontSize: 13 }}>{arrow} {label}</span>
      <span className="pipeline-label">
        {slopeStr}{slopeStr && s1Str ? ' | ' : ''}{s1Str}
      </span>
    </div>
  );
}

export default function StrategyList({ tierSummary, learningTrajectory, evidenceItems, onNavigateEvidence }) {
  const ts = tierSummary;

  return (
    <>
      {/* Evidence items */}
      {evidenceItems.length > 0 && (
        <div style={{
          marginTop: 8,
          padding: '8px 10px',
          borderRadius: 6,
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 4 }}>
            Why this was chosen
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {evidenceItems.slice(0, 5).map((item, i) => (
              <button
                key={i}
                type="button"
                onClick={() => {
                  if (onNavigateEvidence && item.tab) {
                    onNavigateEvidence(item.tab);
                  }
                }}
                style={{
                  fontSize: 11,
                  padding: '2px 6px',
                  borderRadius: 4,
                  background: 'var(--bg-primary)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                  cursor: onNavigateEvidence && item.tab ? 'pointer' : 'default',
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Pipeline badges */}
      <div className="strategy-pipeline" role="list" aria-label="Research pipeline">
        <PipelineBadge label="Screening" count={ts.screening} color={TIER_COLORS.screening} />
        <span className="pipeline-arrow" aria-hidden="true">&rarr;</span>
        <PipelineBadge label="Investigation" count={ts.investigation} color={TIER_COLORS.investigation} />
        <span className="pipeline-arrow" aria-hidden="true">&rarr;</span>
        <PipelineBadge label="Validation" count={ts.validation} color={TIER_COLORS.validation} />
        <span className="pipeline-arrow" aria-hidden="true">&rarr;</span>
        <PipelineBadge label="Breakthrough" count={ts.breakthrough} color={TIER_COLORS.breakthrough} />
        {learningTrajectory && learningTrajectory.trend && learningTrajectory.trend !== 'insufficient_data' && (
          <span className="pipeline-arrow" style={{ marginLeft: 'auto' }} />
        )}
        {learningTrajectory && learningTrajectory.trend && learningTrajectory.trend !== 'insufficient_data' && (
          <LearningTrendBadge trend={learningTrajectory} onNavigate={onNavigateEvidence} />
        )}
      </div>
    </>
  );
}
