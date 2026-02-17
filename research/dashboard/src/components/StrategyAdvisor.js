import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

const TIER_COLORS = {
  screening: 'var(--accent-blue)',
  investigation: 'var(--accent-yellow)',
  validation: 'var(--accent-purple)',
  breakthrough: 'var(--accent-green)',
};

/**
 * Pure deterministic strategy computation.
 * Returns { id, title, rationale, action, tierSummary }.
 */
export function computeStrategy(dashboard, leaderboard, mathCoverage) {
  // Build tier summary from leaderboard
  const entries = Array.isArray(leaderboard) ? leaderboard : [];
  const tierSummary = {
    screening: 0,
    investigation: 0,
    validation: 0,
    breakthrough: 0,
  };

  const breakthroughCandidates = [];
  const validationPassed = [];
  const investigationPassed = [];
  const screeningSurvivors = [];

  const normalizeTier = (entry) => {
    const tier = typeof entry?.tier === 'string' ? entry.tier.toLowerCase() : '';
    if (tier === 'screening' || tier === 'investigation' || tier === 'validation' || tier === 'breakthrough') {
      return tier;
    }
    return null;
  };

  for (const entry of entries) {
    const tier = normalizeTier(entry);
    const effectiveTier = tier || 'screening';
    tierSummary[effectiveTier] += 1;
    if (effectiveTier === 'breakthrough') {
      breakthroughCandidates.push(entry);
    } else if (effectiveTier === 'validation' && entry.validation_passed) {
      validationPassed.push(entry);
    } else if (effectiveTier === 'investigation' && entry.investigation_passed) {
      investigationPassed.push(entry);
    } else {
      screeningSurvivors.push(entry);
    }
  }

  const totalExperiments = dashboard?.summary?.total_experiments || 0;
  const totalPrograms = dashboard?.summary?.total_programs_evaluated ?? dashboard?.summary?.total_programs ?? 0;
  const stage1Survivors = dashboard?.summary?.stage1_survivors || 0;
  const survivalRate = totalPrograms > 0 ? stage1Survivors / totalPrograms : 0;

  // Check recent experiment history for consecutive zero-survivor runs
  const recentExperiments = Array.isArray(dashboard?.recent_experiments)
    ? dashboard.recent_experiments.slice(0, 3)
    : [];
  const lastThreeZeroSurvivors = recentExperiments.length >= 3 &&
    recentExperiments.every(exp => (exp?.stage1_survivors || exp?.s1_passed || 0) === 0);

  // Math family coverage gaps
  const families = Array.isArray(mathCoverage) ? mathCoverage : [];
  const undertestedFamilies = families.filter(f => (f.tested_share || 0) < 0.05);

  // Priority rules (1-9)

  // 1. No experiments yet
  if (totalExperiments === 0) {
    return {
      id: 1,
      title: 'Start Mixed Continuous Research',
      rationale: 'No experiments have been run yet. Begin with continuous mixed-source research to establish a baseline of architecture candidates across graph synthesis and morphological box sources.',
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
    };
  }

  // 2. Breakthrough candidates exist
  if (breakthroughCandidates.length > 0) {
    return {
      id: 2,
      title: `Export/Publish ${breakthroughCandidates.length} Breakthrough${breakthroughCandidates.length > 1 ? 's' : ''}`,
      rationale: `${breakthroughCandidates.length} candidate${breakthroughCandidates.length > 1 ? 's have' : ' has'} passed validation with high composite scores. Review and export these breakthrough architectures.`,
      action: null, // navigate to leaderboard
      tierSummary,
    };
  }

  // 3. Validation-passed candidates ready for scale-up
  if (validationPassed.length > 0) {
    return {
      id: 3,
      title: `Scale Up ${validationPassed.length} Validated Candidate${validationPassed.length > 1 ? 's' : ''}`,
      rationale: `${validationPassed.length} candidate${validationPassed.length > 1 ? 's have' : ' has'} passed validation. Scale up training to confirm performance at larger dimensions and longer sequences.`,
      action: { suggestedMode: 'scale_up', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
    };
  }

  // 4. Investigation-passed, not yet validated
  if (investigationPassed.length > 0) {
    return {
      id: 4,
      title: `Run Validation on ${investigationPassed.length} Investigated Candidate${investigationPassed.length > 1 ? 's' : ''}`,
      rationale: `${investigationPassed.length} candidate${investigationPassed.length > 1 ? 's have' : ' has'} passed investigation but not yet been validated. Run multi-seed validation to confirm robustness.`,
      action: { suggestedMode: 'validation', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
    };
  }

  // 5. Screening survivors exist, none investigated
  if (screeningSurvivors.length > 0 && tierSummary.investigation === 0 && tierSummary.validation === 0) {
    return {
      id: 5,
      title: `Investigate ${screeningSurvivors.length} Screening Survivor${screeningSurvivors.length > 1 ? 's' : ''}`,
      rationale: `${screeningSurvivors.length} candidate${screeningSurvivors.length > 1 ? 's' : ''} passed initial screening but none have been investigated yet. Run deeper investigation with extended training and multiple training programs.`,
      action: { suggestedMode: 'investigation', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
    };
  }

  // 6. Low survival rate after significant experiments
  if (totalExperiments > 10 && survivalRate < 0.01) {
    return {
      id: 6,
      title: 'Try Evolution/Novelty Search',
      rationale: `Survival rate is only ${(survivalRate * 100).toFixed(1)}% across ${totalExperiments} experiments. Population-based search can breed better candidates by combining successful traits.`,
      action: { suggestedMode: 'evolve', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
    };
  }

  // 7. Under-tested math families
  if (undertestedFamilies.length > 0) {
    const familyNames = undertestedFamilies.slice(0, 3).map(f => f.family || f.name).join(', ');
    return {
      id: 7,
      title: 'Expand Math Space Coverage',
      rationale: `${undertestedFamilies.length} math ${undertestedFamilies.length === 1 ? 'family is' : 'families are'} under-explored (<5% tested): ${familyNames}. Increase math space weight to diversify architecture search.`,
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed', math_space_weight: 4.0 } },
      tierSummary,
    };
  }

  // 8. Last 3 experiments had zero survivors
  if (lastThreeZeroSurvivors) {
    return {
      id: 8,
      title: 'Novelty Search to Escape Local Minimum',
      rationale: 'The last 3 experiments produced zero survivors each. Novelty search can escape the current search region by rewarding architectural diversity over raw fitness.',
      action: { suggestedMode: 'novelty', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
    };
  }

  // 9. Default
  return {
    id: 9,
    title: 'Continue Mixed Continuous Research',
    rationale: 'The pipeline is healthy. Continue exploring the architecture space with mixed-source continuous research to find new candidates.',
    action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
    tierSummary,
  };
}

function StrategyAdvisor({ dashboardData, onApplyStrategy, onStart, isRunning }) {
  const [leaderboard, setLeaderboard] = useState(null);
  const [mathCoverage, setMathCoverage] = useState(null);
  const [aiRec, setAiRec] = useState(null);
  const [aiLoading, setAiLoading] = useState(true);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [lbRes, mcRes] = await Promise.all([
        fetch(`${API_BASE}/api/leaderboard?sort=composite_score&limit=100`),
        fetch(`${API_BASE}/api/analytics/math-family-coverage`),
      ]);
      if (lbRes.ok) {
        const lbData = await lbRes.json();
        setLeaderboard(lbData.entries || []);
      }
      if (mcRes.ok) {
        const mcData = await mcRes.json();
        setMathCoverage(Array.isArray(mcData) ? mcData : mcData.families || []);
      }
    } catch {
      // Silently fail — strategy will use available data
    }
    setLoading(false);
  }, []);

  const fetchAiRec = useCallback(async () => {
    setAiLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/aria/recommendation`);
      if (res.ok) {
        const data = await res.json();
        if (data && !data.error && data.reasoning) {
          setAiRec(data);
        }
      }
    } catch {
      // AI unavailable — fall back to deterministic
    }
    setAiLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
    fetchAiRec();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData, fetchAiRec]);

  if (loading && !leaderboard) {
    return (
      <div className="card" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', padding: 8, textAlign: 'center' }}>
          Loading strategy advisor...
        </div>
      </div>
    );
  }

  const strategy = computeStrategy(dashboardData, leaderboard, mathCoverage);
  const isPublishAction = strategy.action === null;
  const ts = strategy.tierSummary;

  // Merge AI recommendation with deterministic strategy
  const hasAi = aiRec && aiRec.reasoning;
  const displayRationale = hasAi ? aiRec.reasoning : strategy.rationale;

  // Build final config: deterministic mode + AI config overrides
  const buildStartConfig = () => {
    if (!strategy.action) return null;
    const base = {
      mode: strategy.action.suggestedMode || 'continuous',
      model_source: strategy.action.configOverrides?.model_source || 'mixed',
      source: 'strategy_advisor',
      ...strategy.action.configOverrides,
    };
    // Layer AI-suggested config on top
    if (hasAi && aiRec.config && typeof aiRec.config === 'object') {
      Object.assign(base, aiRec.config);
    }
    return base;
  };

  // Summarize key params for display
  const suggestedParams = hasAi && aiRec.config ? aiRec.config : null;
  const paramSummary = suggestedParams
    ? Object.entries(suggestedParams)
        .filter(([k]) => k !== 'mode' && k !== 'source')
        .slice(0, 4)
        .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${typeof v === 'number' ? v.toFixed?.(2) ?? v : v}`)
    : [];

  const handleStartClick = () => {
    const config = buildStartConfig();
    if (config && onStart) {
      onStart(config);
    }
  };

  return (
    <div className="card strategy-advisor" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
      <div className="strategy-content">
        <div className="strategy-header">
          <div className="strategy-title">{strategy.title}</div>
          <div className="strategy-rationale">{displayRationale}</div>
          {hasAi && aiRec.confidence != null && (
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              AI confidence: {(aiRec.confidence * 100).toFixed(0)}%
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
          {!hasAi && !aiLoading && (
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, fontStyle: 'italic' }}>
              Rule-based recommendation (AI not available)
            </div>
          )}
        </div>

        <div className="strategy-actions">
          {isRunning ? (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Experiment running — strategy will update when complete.
            </div>
          ) : isPublishAction ? (
            <button
              className="strategy-apply-btn"
              onClick={() => onApplyStrategy && onApplyStrategy(strategy)}
            >
              Review in Leaderboard
            </button>
          ) : (
            <button
              className="strategy-apply-btn"
              onClick={handleStartClick}
              style={{ background: 'var(--accent-green)', color: '#000', fontWeight: 600 }}
            >
              Start Experiment
            </button>
          )}
        </div>
      </div>

      <div className="strategy-pipeline">
        <PipelineBadge label="Screening" count={ts.screening} color={TIER_COLORS.screening} />
        <span className="pipeline-arrow">&rarr;</span>
        <PipelineBadge label="Investigation" count={ts.investigation} color={TIER_COLORS.investigation} />
        <span className="pipeline-arrow">&rarr;</span>
        <PipelineBadge label="Validation" count={ts.validation} color={TIER_COLORS.validation} />
        <span className="pipeline-arrow">&rarr;</span>
        <PipelineBadge label="Breakthrough" count={ts.breakthrough} color={TIER_COLORS.breakthrough} />
      </div>
    </div>
  );
}

export default StrategyAdvisor;

function PipelineBadge({ label, count, color }) {
  return (
    <div className="pipeline-badge" style={{ borderColor: color }}>
      <span className="pipeline-count" style={{ color }}>{count}</span>
      <span className="pipeline-label">{label}</span>
    </div>
  );
}
