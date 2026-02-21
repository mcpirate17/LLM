import React from 'react';
import AriaAvatar from './AriaAvatar';
import useRenderPerf from '../hooks/useRenderPerf';
import { useAriaData } from '../hooks/useAriaData';

function sanitizeHypothesisText(rawText, maxLength = 220) {
  if (!rawText) return '';

  const text = String(rawText)
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!text) return '';
  if (text.length <= maxLength) return text;

  const clipped = text.slice(0, maxLength).trim();
  const boundary = clipped.lastIndexOf(' ');
  if (boundary > Math.floor(maxLength * 0.55)) {
    return `${clipped.slice(0, boundary).trim()}...`;
  }
  return `${clipped}...`;
}

function AriaStatus({ aria }) {
  useRenderPerf('AriaStatus');
  const { summary } = useAriaData() || {};

  if (!aria) return <div className="card aria-card"><p>Waiting for connection...</p></div>;

  const summarizedHypothesis = sanitizeHypothesisText(aria.current_hypothesis);

  // All-time counters: prefer summary (source of truth from LabNotebook),
  // fall back to aria object (which copies the same values under shorter keys).
  const totalExperiments = summary?.total_experiments ?? aria.total_experiments;
  const totalPrograms = summary?.total_programs_evaluated ?? aria.total_programs;
  const stage1Survivors = summary?.stage1_survivors ?? aria.stage1_survivors;

  return (
    <div className="card aria-card">
      <div className="card-title">Aria's Status</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Aria is the AI scientist running the search. She formulates hypotheses about what
        makes architectures work, designs experiments, analyzes results, and adapts her
        strategy based on what she learns.
      </p>

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

      {summarizedHypothesis && (
        <div className="aria-hypothesis">
          "{summarizedHypothesis}"
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
        <div className="aria-stat" title="Internal activity level — higher when more experiments complete successfully">
          <span className="aria-stat-label">Energy</span>
          <span>{(aria.energy * 100).toFixed(0)}%</span>
        </div>
        <div className="aria-stat">
          <span className="aria-stat-label">Focus</span>
          <span>{aria.research_focus}</span>
        </div>
      </div>

      {(totalExperiments != null || totalPrograms != null) && (
        <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', fontWeight: 600 }}>
            All-time
          </div>
          <div className="aria-stats">
            {totalExperiments != null && (
              <div className="aria-stat">
                <span className="aria-stat-label">Experiments</span>
                <span>{totalExperiments}</span>
              </div>
            )}
            {totalPrograms != null && (
              <div className="aria-stat">
                <span className="aria-stat-label">Programs</span>
                <span>{totalPrograms}</span>
              </div>
            )}
            {stage1Survivors != null && (
              <div className="aria-stat">
                <span className="aria-stat-label">S1 Survivors</span>
                <span>{stage1Survivors}</span>
              </div>
            )}
          </div>
        </div>
      )}

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
