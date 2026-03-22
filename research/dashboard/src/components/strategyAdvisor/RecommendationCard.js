import React from 'react';
import DataSourceBadge from './DataSourceBadge';

function TrendChip({ trend, slope }) {
  const color = trend === 'improving' ? 'var(--accent-green)'
    : trend === 'declining' ? 'var(--accent-red, #e74c3c)'
    : 'var(--accent-yellow)';
  const arrow = trend === 'improving' ? '\u2191' : trend === 'declining' ? '\u2193' : '\u2192';
  const label = trend === 'improving' ? 'Learning' : trend === 'declining' ? 'Declining' : 'Plateaued';
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, color,
      background: `color-mix(in srgb, ${color} 12%, transparent)`,
      borderRadius: 4, padding: '1px 5px',
    }}>
      {arrow} {label}
      {slope != null && ` (${slope > 0 ? '+' : ''}${(slope * 100).toFixed(2)}%/exp)`}
    </span>
  );
}

export default function RecommendationCard({
  briefing,
  hasBriefing,
  isAiPowered,
  briefingSummary,
  analyzing,
  strategy,
  suggestedConfig,
  paramSummary,
  actionLabel,
  isActionable,
  mergedDataSources,
  onNavigateEvidence,
}) {
  return (
    <>
      {/* Aria's Analysis -- the main briefing */}
      <div style={{
        padding: '12px 14px',
        marginBottom: 12,
        background: 'var(--bg-tertiary)',
        borderRadius: 6,
        borderLeft: `3px solid ${isAiPowered ? 'var(--accent-purple)' : 'var(--accent-blue)'}`,
        fontSize: 13,
        lineHeight: 1.6,
        color: 'var(--text-secondary)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: 0.5,
            color: isAiPowered ? 'var(--accent-purple)' : 'var(--accent-blue)',
          }}>
            Aria's Analysis
          </span>
          <span style={{
            fontSize: 9, fontWeight: 600,
            color: isAiPowered ? 'var(--accent-purple)' : 'var(--text-muted)',
            background: isAiPowered
              ? 'rgba(137, 87, 229, 0.12)'
              : 'rgba(128, 128, 128, 0.12)',
            border: `1px solid ${isAiPowered ? 'var(--accent-purple)' : 'var(--text-muted)'}`,
            borderRadius: 4,
            padding: '1px 5px',
          }}>
            {isAiPowered ? 'AI-Powered' : 'Rule-Based'}
          </span>
          {hasBriefing && briefing.data?.learning_trend && briefing.data.learning_trend !== 'insufficient_data' && (
            <TrendChip trend={briefing.data.learning_trend} slope={briefing.data.learning_slope} />
          )}
        </div>
        {analyzing ? (
          <div style={{ color: 'var(--accent-purple)', fontStyle: 'italic' }}>
            Aria is analyzing the latest results...
          </div>
        ) : hasBriefing ? (
          briefingSummary || 'No concise summary available.'
        ) : (
          <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
            No briefing data available. Run an experiment to get started.
          </span>
        )}
        {briefing?.ref_comparison && (
          <div style={{
            marginTop: 6, padding: '6px 10px',
            background: briefing.ref_comparison.beats_all_references
              ? 'rgba(63, 185, 80, 0.12)' : 'rgba(139, 148, 158, 0.08)',
            borderRadius: 6,
            border: briefing.ref_comparison.beats_all_references
              ? '1px solid var(--accent-green)' : '1px solid var(--border)',
            fontSize: 12,
          }}>
            {briefing.ref_comparison.beats_all_references ? (
              <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
                Synthesized model beats all references by {briefing.ref_comparison.margin_pct}%
                {' '}(score {briefing.ref_comparison.best_synthesized_score?.toFixed(1)} vs best ref {briefing.ref_comparison.best_reference_score?.toFixed(1)})
              </span>
            ) : (
              <span style={{ color: 'var(--text-muted)' }}>
                Best reference: {briefing.ref_comparison.best_reference_score?.toFixed(1)}
                {briefing.ref_comparison.references?.map(r =>
                  <span key={r.name}> | {r.name}: {r.score?.toFixed(1)}</span>
                )}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Suggested experiment header */}
      <div className="strategy-content">
        <div className="strategy-header">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
            <DataSourceBadge dataSources={mergedDataSources} onNavigateEvidence={onNavigateEvidence} />
            <span style={{
              fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
              color: isActionable ? 'var(--accent-green)' : 'var(--accent-yellow)',
              background: isActionable ? 'rgba(63, 185, 80, 0.16)' : 'rgba(210, 153, 34, 0.16)',
              border: `1px solid ${isActionable ? 'var(--accent-green)' : 'var(--accent-yellow)'}`,
              borderRadius: 4,
              padding: '1px 5px',
            }}>
              {isActionable ? 'Actionable' : 'Advice only'}
            </span>
            {briefing?.confidence != null && isAiPowered && (
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Confidence: {(briefing.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div className="strategy-title">{actionLabel}</div>
          <div className="strategy-rationale">
            {briefing?.action_rationale || strategy.rationale}
          </div>
          {suggestedConfig?.hypothesis && (
            <div style={{
              marginTop: 6, padding: '6px 10px',
              background: 'var(--bg-primary)',
              borderRadius: 4,
              fontSize: 12,
              color: 'var(--text-secondary)',
              fontStyle: 'italic',
              borderLeft: '2px solid var(--accent-purple)',
            }}>
              {suggestedConfig.hypothesis}
            </div>
          )}
          {paramSummary.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {paramSummary.map((p, i) => (
                <span key={i} style={{
                  fontSize: 11, padding: '2px 6px', borderRadius: 4,
                  background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                }}>{p}</span>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
