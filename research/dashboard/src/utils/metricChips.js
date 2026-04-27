/**
 * Metric chip generators — each produces [{ label, source, reliability }]
 * for the shared MetricChipBadge component.
 */

import { evalMetricQuality } from './backendScore';

function reliabilityFromConfidence(value, high = 0.7, med = 0.4) {
  if (value == null) return 'low';
  return value >= high ? 'high' : value >= med ? 'medium' : 'low';
}

function reliabilityFromCount(count, high = 100, med = 30) {
  return (count || 0) >= high ? 'high' : (count || 0) >= med ? 'medium' : 'low';
}

/** Chips for program/discovery detail views (report context). */
export function programMetricChips(program) {
  if (!program) return [];
  const chips = [];
  const hasValidation = program.validation_loss_ratio != null;
  const lr = hasValidation ? program.validation_loss_ratio : program.loss_ratio;
  chips.push({
    label: 'Loss',
    source: lr != null ? 'validation' : 'measured',
    reliability: lr != null ? 'high' : 'low',
  });
  chips.push({
    label: 'Novelty',
    source: program.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic',
    reliability: reliabilityFromConfidence(program.novelty_confidence),
  });
  chips.push({
    label: 'Baseline',
    source: program.baseline_loss_ratio != null ? 'baseline-run' : 'not-available',
    reliability: program.baseline_loss_ratio != null ? 'medium' : 'low',
  });
  if (program.routing_confidence_mean != null) {
    chips.push({
      label: 'Routing',
      source: 'telemetry',
      reliability: reliabilityFromConfidence(program.routing_confidence_mean),
    });
  }
  return chips;
}

/** Chips for leaderboard entries (may have tiered validation/investigation fields). */
export function leaderboardMetricChips(entry) {
  if (!entry) return [];
  const chips = [];
  const evalQuality = evalMetricQuality(entry);
  chips.push({
    label: evalQuality.label,
    source: evalQuality.version,
    reliability: evalQuality.reliability,
  });
  chips.push({
    label: 'Loss',
    source: 'measured',
    reliability: entry.validation_loss_ratio != null ? 'high'
      : entry.investigation_loss_ratio != null ? 'medium' : 'low',
  });
  chips.push({
    label: 'Novelty',
    source: entry.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic',
    reliability: reliabilityFromConfidence(entry.novelty_confidence),
  });
  chips.push({
    label: 'Baseline',
    source: entry.validation_baseline_ratio != null ? 'baseline-run' : 'not-available',
    reliability: entry.validation_multi_seed_std != null
      ? (entry.validation_multi_seed_std <= 0.12 ? 'high' : 'medium')
      : 'low',
  });
  if (entry.routing_confidence_mean != null) {
    chips.push({
      label: 'Routing',
      source: 'telemetry',
      reliability: reliabilityFromConfidence(entry.routing_confidence_mean),
    });
  }
  return chips;
}

/** Chips for experiment-level summary. */
export function experimentMetricChips(exp) {
  if (!exp) return [];
  const nPrograms = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const evidenceReliability = reliabilityFromCount(nPrograms);
  return [
    {
      label: 'Loss',
      source: exp.best_loss_ratio != null ? 'measured' : 'not-evaluated',
      reliability: exp.best_loss_ratio != null ? evidenceReliability : 'low',
    },
    {
      label: 'Novelty',
      source: exp.best_novelty_score != null ? 'heuristic' : 'insufficient-data',
      reliability: s1 > 0 ? evidenceReliability : 'low',
    },
    {
      label: 'Baseline',
      source: 'not-available',
      reliability: 'low',
    },
  ];
}

/** Chips for per-op stats tables. */
export function opMetricChips(row) {
  if (!row) return [];
  return [
    {
      label: 'S1',
      source: 'measured',
      reliability: reliabilityFromCount(row.n_used, 100, 40),
    },
    {
      label: 'Novelty',
      source: row.avg_novelty_confidence != null && row.avg_novelty_confidence >= 0.5
        ? 'artifact-backed' : 'heuristic',
      reliability: reliabilityFromConfidence(row.avg_novelty_confidence),
    },
  ];
}

/** Chips for routing mode health tables. */
export function routingMetricChips(row) {
  if (!row) return [];
  return [
    {
      label: 'Routing',
      source: 'telemetry',
      reliability: reliabilityFromConfidence(row.avg_confidence_mean),
    },
    {
      label: 'Sample',
      source: 'mode-aggregate',
      reliability: reliabilityFromCount(row.n_programs, 80, 30),
    },
  ];
}
