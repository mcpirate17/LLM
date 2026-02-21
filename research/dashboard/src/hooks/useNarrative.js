import { createContext, useContext, useMemo } from 'react';

const NarrativeContext = createContext(null);

/**
 * Analyse weight deltas between default and learned grammar weights.
 * Returns sorted array of { category, def, learned, delta, pct, direction }.
 */
function analyseWeightDeltas(defaultW, learnedW) {
  if (!defaultW || !learnedW) return [];
  const changes = [];
  for (const cat of Object.keys(defaultW).sort()) {
    const def = defaultW[cat] || 0;
    const learned = learnedW[cat];
    if (learned == null || def === 0) continue;
    const delta = learned - def;
    const pct = (delta / def) * 100;
    if (Math.abs(pct) >= 5) {
      changes.push({
        category: cat,
        label: cat.replace(/_/g, ' '),
        def,
        learned,
        delta,
        pct,
        direction: delta > 0 ? 'boosted' : 'reduced',
      });
    }
  }
  changes.sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct));
  return changes;
}

/**
 * Build a first-person Aria narrative from trajectory trend + weight deltas.
 * Uses switch-case on trend to produce natural-language summaries.
 */
function buildNarrative(trajectory, weightDeltas) {
  if (!trajectory?.trend || trajectory.trend === 'insufficient_data') {
    return null;
  }

  const { trend, slope = 0, recent_s1_rate: recentS1 } = trajectory;
  const rateStr = recentS1 != null ? `${(recentS1 * 100).toFixed(1)}%` : null;
  const boosted = weightDeltas.filter(w => w.direction === 'boosted');
  const reduced = weightDeltas.filter(w => w.direction === 'reduced');

  const parts = [];

  switch (trend) {
    case 'plateaued': {
      if (rateStr) {
        parts.push(
          `I've detected a performance plateau at ${rateStr} S1 pass rate. ` +
          `The search is producing survivors at a steady clip, but progress has stalled.`
        );
      } else {
        parts.push(
          `I've detected a performance plateau \u2014 new experiments aren't improving the survival rate.`
        );
      }

      if (boosted.length > 0 || reduced.length > 0) {
        parts.push('To break through, I\'m rebalancing the grammar:');
        for (const w of boosted.slice(0, 3)) {
          parts.push(
            `Increasing ${w.label} weight to ${w.learned.toFixed(1)} ` +
            `(+${w.pct.toFixed(0)}%) to explore deeper architectures in that space.`
          );
        }
        for (const w of reduced.slice(0, 3)) {
          parts.push(
            `Reducing ${w.label} weight to ${w.learned.toFixed(1)} ` +
            `(${w.pct.toFixed(0)}%) \u2014 it's consuming search budget without producing survivors.`
          );
        }
      } else {
        parts.push(
          'No weight adjustments yet \u2014 still using default grammar weights. ' +
          'I need more experiment data before adaptation can kick in.'
        );
      }
      break;
    }

    case 'improving': {
      const slopeStr = `+${(slope * 100).toFixed(2)}%/exp`;
      parts.push(
        `The learning trajectory is improving at ${slopeStr}` +
        `${rateStr ? ` (recent S1: ${rateStr})` : ''}. ` +
        `I'm finding better architectures over time.`
      );

      if (boosted.length > 0) {
        const top = boosted[0];
        parts.push(
          `My strongest lever is ${top.label} ` +
          `(${top.def.toFixed(1)} \u2192 ${top.learned.toFixed(1)}). ` +
          `I'll keep favouring it in the next cycle.`
        );
      }

      if (reduced.length > 0) {
        const bottom = reduced[0];
        parts.push(
          `I'm pulling back on ${bottom.label} ` +
          `(${bottom.pct.toFixed(0)}%) to redirect budget toward what's working.`
        );
      }
      break;
    }

    case 'declining': {
      const slopeStr = `${(slope * 100).toFixed(2)}%/exp`;
      parts.push(
        `The S1 pass rate is declining (${slopeStr}` +
        `${rateStr ? `, recent: ${rateStr}` : ''}). ` +
        `Recent experiments are finding fewer viable architectures.`
      );

      if (reduced.length > 0) {
        const drag = reduced[0];
        parts.push(
          `The biggest drag is ${drag.label} ` +
          `(${drag.def.toFixed(1)} \u2192 ${drag.learned.toFixed(1)}). ` +
          `I'm pulling back hard to free search budget.`
        );
      }

      if (boosted.length > 0) {
        const pivots = boosted.slice(0, 2).map(w => w.label).join(' and ');
        parts.push(`I'm pivoting toward ${pivots} as the most promising rescue direction.`);
      }
      break;
    }

    default:
      return null;
  }

  return parts.join(' ');
}

/**
 * NarrativeProvider — computes first-person Aria narratives from trajectory
 * and grammar-weight data. Children consume via useNarrative().
 *
 * Props:
 *   trajectoryData — from /api/analytics/learning-trajectory
 *   weightData     — from /api/analytics/grammar-weights ({ default, learned, ... })
 */
export function NarrativeProvider({ trajectoryData, weightData, children }) {
  const value = useMemo(() => {
    const defaultW = weightData?.default;
    const learnedW = weightData?.learned;
    const deltas = analyseWeightDeltas(defaultW, learnedW);
    const narrative = buildNarrative(trajectoryData, deltas);

    return {
      narrative,
      trend: trajectoryData?.trend || null,
      recentS1Rate: trajectoryData?.recent_s1_rate ?? null,
      slope: trajectoryData?.slope ?? null,
      weightDeltas: deltas,
      boosted: deltas.filter(w => w.direction === 'boosted'),
      reduced: deltas.filter(w => w.direction === 'reduced'),
    };
  }, [trajectoryData, weightData]);

  return (
    <NarrativeContext.Provider value={value}>
      {children}
    </NarrativeContext.Provider>
  );
}

export function useNarrative() {
  return useContext(NarrativeContext);
}

export default NarrativeContext;
