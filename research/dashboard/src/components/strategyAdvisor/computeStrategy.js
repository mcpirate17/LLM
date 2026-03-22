/**
 * Pure deterministic strategy computation.
 * Returns { id, title, rationale, action, tierSummary, dataSources }.
 */
export default function computeStrategy(dashboard, leaderboard, mathCoverage) {
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
  const investigationFailed = [];
  const screeningSurvivors = [];

  const normalizeTier = (entry) => {
    const tier = typeof entry?.tier === 'string' ? entry.tier.toLowerCase() : '';
    if (tier === 'screening' || tier === 'investigation' || tier === 'validation' || tier === 'breakthrough') {
      return tier;
    }
    return null;
  };

  for (const entry of entries) {
    // Skip pinned reference architectures — they are baselines, not discoveries
    const rid = String(entry?.result_id || '').toLowerCase();
    const refName = String(entry?.reference_name || '').trim();
    if (
      entry.is_reference ||
      entry.is_pinned ||
      entry.model_source === 'reference' ||
      refName.length > 0 ||
      rid.startsWith('ref_')
    ) {
      continue;
    }
    const tier = normalizeTier(entry);
    const effectiveTier = tier || 'screening';
    tierSummary[effectiveTier] += 1;
    if (effectiveTier === 'breakthrough') {
      breakthroughCandidates.push(entry);
    } else if (effectiveTier === 'validation' && entry.validation_passed) {
      validationPassed.push(entry);
    } else if (effectiveTier === 'investigation' && entry.investigation_passed) {
      investigationPassed.push(entry);
    } else if (effectiveTier === 'investigation' && !entry.investigation_passed) {
      investigationFailed.push(entry);
    } else if (effectiveTier === 'screening') {
      screeningSurvivors.push(entry);
    }
  }

  const totalExperiments = dashboard?.summary?.total_experiments || 0;
  const totalPrograms = dashboard?.summary?.total_programs_evaluated || 0;
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

  // Priority rules (1-10)

  // 1. No experiments yet
  if (totalExperiments === 0) {
    return {
      id: 1,
      title: 'Start Mixed Continuous Research',
      rationale: 'No experiments have been run yet. Begin with continuous mixed-source research to establish a baseline of architecture candidates across graph synthesis and morphological box sources.',
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'Total Experiments', value: 0, threshold: 1, comparison: '<', tab: 'experiments' },
      ],
    };
  }

  // 2. Breakthrough candidates exist
  if (breakthroughCandidates.length > 0) {
    return {
      id: 2,
      title: `Export/Publish ${breakthroughCandidates.length} Breakthrough${breakthroughCandidates.length > 1 ? 's' : ''}`,
      rationale: `${breakthroughCandidates.length} candidate${breakthroughCandidates.length > 1 ? 's have' : ' has'} passed validation with high composite scores. Review and export these breakthrough architectures.`,
      action: null,
      tierSummary,
      dataSources: [
        { metric: 'Breakthrough Candidates', value: breakthroughCandidates.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
      ],
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
      dataSources: [
        { metric: 'Validation-Passed Candidates', value: validationPassed.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
      ],
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
      dataSources: [
        { metric: 'Investigation-Passed Candidates', value: investigationPassed.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
        { metric: 'Validation-Passed Candidates', value: validationPassed.length, threshold: 0, comparison: '=', tab: 'leaderboard' },
      ],
    };
  }

  // 5. Screening survivors awaiting investigation
  if (screeningSurvivors.length > 0) {
    const sources = [
      { metric: 'Screening Survivors', value: screeningSurvivors.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
    ];
    if (investigationFailed.length > 0) {
      sources.push({ metric: 'Prior Investigation Failures', value: investigationFailed.length, threshold: null, comparison: 'context', tab: 'leaderboard' });
    }
    return {
      id: 5,
      title: `Investigate ${screeningSurvivors.length} Screening Survivor${screeningSurvivors.length > 1 ? 's' : ''}`,
      rationale: `${screeningSurvivors.length} candidate${screeningSurvivors.length > 1 ? 's' : ''} passed screening and ${screeningSurvivors.length > 1 ? 'are' : 'is'} awaiting investigation.${investigationFailed.length > 0 ? ` (${investigationFailed.length} prior investigation${investigationFailed.length > 1 ? 's' : ''} failed — new candidates may perform better.)` : ''} Run deeper investigation with extended training and multiple training programs.`,
      action: { suggestedMode: 'investigation', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
      dataSources: sources,
    };
  }

  // 6. All investigations failed
  if (investigationFailed.length > 0 && screeningSurvivors.length === 0 && investigationPassed.length === 0) {
    return {
      id: 6,
      title: 'Find New Candidates (All Investigations Failed)',
      rationale: `${investigationFailed.length} candidate${investigationFailed.length > 1 ? 's were' : ' was'} investigated but ${investigationFailed.length > 1 ? 'none' : 'it did not'} pass${investigationFailed.length === 1 ? '' : 'ed'}. Run more screening experiments to discover new candidates worth investigating.`,
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'Investigation Failures', value: investigationFailed.length, threshold: 0, comparison: '>', tab: 'leaderboard' },
        { metric: 'Screening Survivors', value: 0, threshold: 0, comparison: '=', tab: 'leaderboard' },
        { metric: 'Investigation-Passed', value: 0, threshold: 0, comparison: '=', tab: 'leaderboard' },
      ],
    };
  }

  // 7. Low survival rate
  if (totalExperiments > 10 && survivalRate < 0.01) {
    return {
      id: 7,
      title: 'Try Evolution/Novelty Search',
      rationale: `Survival rate is only ${(survivalRate * 100).toFixed(1)}% across ${totalExperiments} experiments. Population-based search can breed better candidates by combining successful traits.`,
      action: { suggestedMode: 'evolve', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'S1 Pass Rate', value: `${(survivalRate * 100).toFixed(1)}%`, threshold: '1%', comparison: '<', tab: 'trends' },
        { metric: 'Total Experiments', value: totalExperiments, threshold: 10, comparison: '>', tab: 'experiments' },
      ],
    };
  }

  // 8. Under-tested math families
  if (undertestedFamilies.length > 0) {
    const familyNames = undertestedFamilies.slice(0, 3).map(f => f.family || f.name).join(', ');
    return {
      id: 8,
      title: 'Expand Math Space Coverage',
      rationale: `${undertestedFamilies.length} math ${undertestedFamilies.length === 1 ? 'family is' : 'families are'} under-explored (<5% tested): ${familyNames}. Increase math space weight to diversify architecture search.`,
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed', math_space_weight: 4.0 } },
      tierSummary,
      dataSources: [
        { metric: 'Under-tested Math Families', value: undertestedFamilies.length, threshold: '5% coverage', comparison: '<', tab: 'learning' },
        { metric: 'Families', value: familyNames, threshold: null, comparison: 'context', tab: 'learning' },
      ],
    };
  }

  // 9. Last 3 experiments had zero survivors
  if (lastThreeZeroSurvivors) {
    return {
      id: 9,
      title: 'Novelty Search to Escape Local Minimum',
      rationale: 'The last 3 experiments produced zero survivors each. Novelty search can escape the current search region by rewarding architectural diversity over raw fitness.',
      action: { suggestedMode: 'novelty', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'Consecutive Zero-Survivor Runs', value: 3, threshold: 3, comparison: '>=', tab: 'trends' },
      ],
    };
  }

  // 10. Default
  return {
    id: 10,
    title: 'Continue Mixed Continuous Research',
    rationale: 'The pipeline is healthy. Continue exploring the architecture space with mixed-source continuous research to find new candidates.',
    action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
    tierSummary,
    dataSources: [
      { metric: 'Pipeline Status', value: 'healthy', threshold: null, comparison: 'nominal', tab: 'overview' },
      { metric: 'S1 Pass Rate', value: `${(survivalRate * 100).toFixed(1)}%`, threshold: null, comparison: 'context', tab: 'trends' },
      { metric: 'Total Experiments', value: totalExperiments, threshold: null, comparison: 'context', tab: 'experiments' },
    ],
  };
}
