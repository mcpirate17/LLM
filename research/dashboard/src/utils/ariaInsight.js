/**
 * Generate a narrative insight from LearningTrajectory + GrammarWeights data.
 *
 * Returns a plain string explaining Aria's current thought process — which
 * grammar weight adjustments are planned, why the trajectory looks the way
 * it does, and what the next cycle will prioritise.
 *
 * Returns null when there is insufficient data for a meaningful narrative.
 */
export function generateAriaInsight(trajectory, weights) {
  if (!trajectory || !trajectory.trend || trajectory.trend === 'insufficient_data') {
    return null;
  }

  const trend = trajectory.trend;
  const slope = trajectory.slope || 0;
  const recentS1 = trajectory.recent_s1_rate;
  const overallS1 = trajectory.overall_s1_rate;
  const defaultW = weights?.default;
  const learnedW = weights?.learned;

  // Build weight-delta analysis when learned weights exist
  const weightChanges = [];
  if (defaultW && learnedW) {
    const categories = Object.keys(defaultW).sort();
    for (const cat of categories) {
      const def = defaultW[cat] || 0;
      const learned = learnedW[cat];
      if (learned == null || def === 0) continue;
      const delta = learned - def;
      const pct = (delta / def) * 100;
      if (Math.abs(pct) >= 8) {
        weightChanges.push({
          category: cat.replace(/_/g, ' '),
          delta,
          pct,
          direction: delta > 0 ? 'boosted' : 'reduced',
          learned,
          def,
        });
      }
    }
    weightChanges.sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct));
  }

  const boosted = weightChanges.filter(w => w.direction === 'boosted');
  const reduced = weightChanges.filter(w => w.direction === 'reduced');

  // --- Plateaued ---
  if (trend === 'plateaued') {
    const parts = [];
    const rateStr = recentS1 != null ? `${(recentS1 * 100).toFixed(1)}%` : null;

    if (rateStr) {
      parts.push(
        `The S1 pass rate has plateaued at ${rateStr}. ` +
        `The search is producing survivors at a steady rate, but not accelerating.`
      );
    } else {
      parts.push('The S1 pass rate has plateaued — new experiments are not improving the survival rate.');
    }

    if (weightChanges.length > 0) {
      parts.push('To break through, the grammar is being rebalanced:');
      if (boosted.length > 0) {
        const boostList = boosted.slice(0, 3).map(
          w => `${w.category} (+${w.pct.toFixed(0)}% to ${w.learned.toFixed(1)})`
        ).join(', ');
        parts.push(`Boosting: ${boostList} — these categories have above-average S1 contribution.`);
      }
      if (reduced.length > 0) {
        const reduceList = reduced.slice(0, 3).map(
          w => `${w.category} (${w.pct.toFixed(0)}% to ${w.learned.toFixed(1)})`
        ).join(', ');
        parts.push(`Reducing: ${reduceList} — underperforming categories that consume search budget.`);
      }
    } else if (defaultW && !learnedW) {
      parts.push('No learned weight adjustments yet — the system is still using default grammar weights. More experiments are needed before adaptation kicks in.');
    }

    return parts.join(' ');
  }

  // --- Improving ---
  if (trend === 'improving') {
    const slopeStr = `+${(slope * 100).toFixed(2)}%/exp`;
    const rateStr = recentS1 != null ? ` (recent S1: ${(recentS1 * 100).toFixed(1)}%)` : '';
    const parts = [
      `The learning trajectory is improving at ${slopeStr}${rateStr}. ` +
      `The search strategy is finding better architectures over time.`
    ];

    if (boosted.length > 0) {
      const topBoost = boosted[0];
      parts.push(
        `The strongest contributor is "${topBoost.category}" ` +
        `(weight ${topBoost.def.toFixed(1)} \u2192 ${topBoost.learned.toFixed(1)}), ` +
        `which the grammar will continue to favour.`
      );
    }

    return parts.join(' ');
  }

  // --- Declining ---
  if (trend === 'declining') {
    const slopeStr = `${(slope * 100).toFixed(2)}%/exp`;
    const rateStr = recentS1 != null ? `${(recentS1 * 100).toFixed(1)}%` : 'falling';
    const parts = [
      `The S1 pass rate is declining (${slopeStr}, recent: ${rateStr}). ` +
      `Recent experiments are finding fewer viable architectures.`
    ];

    if (reduced.length > 0) {
      const topReduce = reduced[0];
      parts.push(
        `The biggest drag is "${topReduce.category}" ` +
        `(weight ${topReduce.def.toFixed(1)} \u2192 ${topReduce.learned.toFixed(1)}). ` +
        `The grammar is pulling back on underperformers to free search budget.`
      );
    }

    if (boosted.length > 0) {
      const pivotList = boosted.slice(0, 2).map(w => w.category).join(' and ');
      parts.push(`Pivoting toward ${pivotList} as the most promising search direction.`);
    }

    return parts.join(' ');
  }

  return null;
}
