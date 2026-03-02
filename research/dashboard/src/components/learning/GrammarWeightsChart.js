import React from 'react';
import { CHART_DEFAULTS, getFixedScale } from '../../utils/chartScales';
import Tooltip from '../shared/Tooltip';
import { CATEGORY_DESCRIPTIONS } from '../../utils/categoryConfig';

export function GrammarWeightsChart({ defaultWeights, learnedWeights, explanation, onStartExperiment }) {
  if (!defaultWeights) return null;

  const categories = Object.keys(defaultWeights).sort();
  const weightValues = categories.map(c => Math.max(defaultWeights[c] || 0, (learnedWeights || {})[c] || 0));
  const weightDefaults = CHART_DEFAULTS.grammar_weight;
  const weightScale = getFixedScale('learning.grammar_weight', weightValues, {
    defaultMin: weightDefaults.min,
    defaultMax: weightDefaults.max,
  });
  const maxWeight = Math.max(weightScale.max, 1);

  return (
    <div className="card">
      <div className="card-title">Grammar Weights (Default vs Learned)</div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        <p style={{ margin: '0 0 8px' }}>
          This chart shows the probability of each operation category being selected during architecture synthesis.
          The system <strong>learns from experience</strong>: categories that consistently appear in "Stage 1 Survivors" (models that learned successfully) have their weights increased.
        </p>
        <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, background: 'rgba(88, 166, 255, 0.3)', border: '1px solid var(--accent-blue)', borderRadius: 2 }} />
            <span>Default Weight</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, background: 'rgba(63, 185, 80, 0.3)', border: '1px solid var(--accent-green)', borderRadius: 2 }} />
            <span>Boosted (Success)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, background: 'rgba(248, 81, 73, 0.3)', border: '1px solid var(--accent-red)', borderRadius: 2 }} />
            <span>Penalized (Failure)</span>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {categories.map(cat => {
          const def = defaultWeights[cat] || 0;
          const learned = (learnedWeights || {})[cat];
          const hasLearned = learned !== undefined && learned !== null;
          return (
            <div key={cat} style={{ fontSize: 13 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <Tooltip content={CATEGORY_DESCRIPTIONS[cat] || "A grouping of related primitive operations."}>
                  <span style={{ color: 'var(--text-secondary)', fontWeight: 500, cursor: 'help', borderBottom: '1px dotted var(--text-muted)' }}>
                    {cat.replace(/_/g, ' ')}
                  </span>
                </Tooltip>
                <span style={{ fontSize: 11 }}>
                  {hasLearned && (
                    <span style={{
                      color: learned > def ? 'var(--accent-green)' : learned < def ? 'var(--accent-red)' : 'var(--text-muted)',
                      marginRight: 8,
                      fontWeight: 600
                    }}>
                      {learned > def ? '+' : ''}{(((learned - def) / (def || 1)) * 100).toFixed(0)}%
                    </span>
                  )}
                  <span style={{ color: 'var(--text-muted)' }}>{def.toFixed(1)} &rarr; </span>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{hasLearned ? learned.toFixed(1) : def.toFixed(1)}</span>
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2, background: 'var(--bg-tertiary)', padding: '4px 6px', borderRadius: 6, border: '1px solid var(--border)' }}>
                {/* Default weight bar (reference) */}
                <Tooltip content={<div><strong>Default Weight: {def.toFixed(2)}</strong><br/>Baseline frequency for this category.<br/><br/>{CATEGORY_DESCRIPTIONS[cat]}</div>}>
                  <div style={{ height: 6, background: 'var(--bg-primary)', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${(def / maxWeight) * 100}%`,
                      background: 'var(--accent-blue)',
                      opacity: 0.4,
                    }} />
                  </div>
                </Tooltip>
                
                {/* Learned weight bar (actual) */}
                {hasLearned ? (
                  <Tooltip content={<div><strong>Learned Weight: {learned.toFixed(2)}</strong><br/>Current frequency after learning.<br/><br/>{learned > def ? 'Boosted due to success correlation.' : 'Penalized due to lack of survivors.'}</div>}>
                    <div style={{ height: 10, background: 'var(--bg-primary)', borderRadius: 5, position: 'relative', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${(learned / maxWeight) * 100}%`,
                        background: learned > def ? 'var(--accent-green)' : 'var(--accent-red)',
                        opacity: 0.8,
                      }} />
                      {/* Vertical line showing where the default was */}
                      <div style={{
                        position: 'absolute',
                        left: `${(def / maxWeight) * 100}%`,
                        top: 0,
                        bottom: 0,
                        width: 2,
                        background: 'var(--text-primary)',
                        opacity: 0.5,
                        zIndex: 1
                      }} />
                    </div>
                  </Tooltip>
                ) : (
                  <div style={{ height: 10, fontSize: 10, color: 'var(--text-muted)', fontStyle: 'italic', display: 'flex', alignItems: 'center' }}>
                    Pending sufficient data...
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)', textAlign: 'right' }}>
        See <strong>Op Success Rates</strong> below for a detailed per-operation breakdown.
      </div>
      {!learnedWeights && (
        <div style={{ marginTop: 10, padding: '10px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', border: '1px solid var(--border)' }}>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 6px', lineHeight: 1.5 }}>
            No learned weights yet — only default weights are shown. The system needs at least 5 distinct
            operation categories with success data to compute learned weights. Run more diverse experiments
            to explore different op categories.
          </p>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Categories discovered: {categories.length} · Need success data across {'\u2265'}5 categories
          </div>
          {onStartExperiment && categories.length < 5 && (
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '4px 10px', marginTop: 8 }}
              onClick={() => onStartExperiment({
                mode: 'continuous', n_cycles: 5,
                source: 'grammar_weights', auto_harden: true,
                preflight_override: true, enforce_preflight: true,
              })}
            >
              Run 5 Continuous
            </button>
          )}
        </div>
      )}
      {explanation && (
        <div style={{ marginTop: 12, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6, borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontSize: 11, color: 'var(--accent-purple)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 4 }}>
            Aria's interpretation
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
            {explanation}
          </div>
        </div>
      )}
    </div>
  );
}

export default GrammarWeightsChart;
