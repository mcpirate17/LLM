function parseArchSpec(value) {
  if (!value || typeof value !== 'string') return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

export function resolveLossRatio(entry) {
  if (!entry || typeof entry !== 'object') return null;

  const validation = entry.validation_loss_ratio;
  if (validation != null && Number.isFinite(Number(validation))) {
    return Number(validation);
  }

  const investigation = entry.investigation_loss_ratio;
  if (investigation != null && Number.isFinite(Number(investigation))) {
    return Number(investigation);
  }

  const screening = entry.screening_loss_ratio;
  if (screening != null && Number.isFinite(Number(screening))) {
    return Number(screening);
  }

  const loss = entry.loss_ratio;
  return loss != null && Number.isFinite(Number(loss)) ? Number(loss) : null;
}

export const CAPABILITY_RANKER_EVIDENCE_FIELDS = [
  'induction_intermediate_auc',
  'binding_intermediate_auc',
  'ar_intermediate_diagnostic_score',
  'binding_multislot_diagnostic_score',
  'induction_validation_auc',
  'ar_validation_rank_score',
];

export function hasCapabilityRankingEvidence(entry) {
  return CAPABILITY_RANKER_EVIDENCE_FIELDS.some((field) => entry?.[field] != null);
}

export function candidateEligibility(entry) {
  if (!entry || typeof entry !== 'object') {
    return {
      investigationEligible: false,
      capabilityRankingEligible: false,
      validationEligible: false,
      confirmationEligible: false,
      queueEligible: false,
      queueReason: 'missing_candidate_data',
    };
  }

  const tier = typeof entry.tier === 'string' ? entry.tier.toLowerCase() : '';
  const capabilityStatus = String(entry?.capability_quality?.status || '').toLowerCase();
  const hasInvestigationEvidence = entry.investigation_loss_ratio != null || entry.investigation_robustness != null;
  const hasRankerEvidence = hasCapabilityRankingEvidence(entry);
  const hasValidationEvidence = entry.validation_loss_ratio != null || entry.validation_baseline_ratio != null || Boolean(entry.validation_passed);
  const isCapabilityQualified = capabilityStatus === 'qualified' || capabilityStatus === 'breakthrough';
  const stage1KnownFailed = entry.stage1_passed === false || entry.stage1_passed === 0;
  const investigationEligible = !stage1KnownFailed && tier === 'screening' && !hasInvestigationEvidence;
  const capabilityRankingEligible = (
    (tier === 'investigation' || tier === 'capability_ranking')
    && Boolean(entry.investigation_passed)
    && !hasRankerEvidence
  );
  const validationEligible = (
    (tier === 'investigation' && Boolean(entry.investigation_passed))
    || (tier === 'capability_ranking' && Boolean(entry.investigation_passed))
    || (tier === 'validation' && !isCapabilityQualified)
  );
  const confirmationEligible = (
    !entry.is_reference
    && (tier === 'validation' || tier === 'breakthrough' || Boolean(entry.validation_passed))
    && (Boolean(entry.validation_passed) || tier === 'breakthrough')
  );

  let queueReason = null;
  if (!investigationEligible && !capabilityRankingEligible && !validationEligible && !confirmationEligible) {
    if (tier === 'screening' && hasInvestigationEvidence) {
      queueReason = 'already_investigated_unchanged';
    } else if (tier === 'investigation' && !entry.investigation_passed) {
      queueReason = 'not_investigation_passed';
    } else if (tier === 'validation' && hasValidationEvidence && !entry.validation_passed) {
      queueReason = 'not_validation_passed';
    } else if (tier === 'validation' && hasValidationEvidence) {
      queueReason = 'needs_capability_revalidation';
    } else {
      queueReason = 'not_progression_eligible';
    }
  }

  return {
    investigationEligible,
    capabilityRankingEligible,
    validationEligible,
    confirmationEligible,
    queueEligible: investigationEligible || capabilityRankingEligible || validationEligible || confirmationEligible,
    queueReason,
  };
}

export function buildEligibilityByResultId(entries) {
  const map = {};
  for (const entry of Array.isArray(entries) ? entries : []) {
    const resultId = entry?.result_id;
    if (!resultId) continue;
    map[resultId] = candidateEligibility(entry);
  }
  return map;
}

export function reproducibilityPacketStatus(entry) {
  const spec = parseArchSpec(entry?.arch_spec_json);
  const checks = [
    { label: 'result_id', ok: !!entry?.result_id },
    { label: 'graph_fingerprint', ok: !!entry?.graph_fingerprint },
    { label: 'arch_spec', ok: !!spec },
    { label: 'loss_ratio', ok: resolveLossRatio(entry) != null },
    { label: 'baseline_ratio', ok: entry?.validation_baseline_ratio != null || entry?.baseline_loss_ratio != null },
    { label: 'multi_seed_std', ok: entry?.validation_multi_seed_std != null },
    { label: 'cka_artifact', ok: entry?.cka_source === 'artifact' },
  ];
  const readyCount = checks.filter((check) => check.ok).length;
  const totalChecks = checks.length;
  const label = readyCount === totalChecks ? 'Ready' : readyCount >= 4 ? 'Partial' : 'Sparse';
  const color = readyCount === totalChecks
    ? 'var(--accent-green)'
    : readyCount >= 4
      ? 'var(--accent-yellow)'
      : 'var(--accent-red)';
  return {
    label,
    color,
    readyCount,
    totalChecks,
    missing: checks.filter((check) => !check.ok).map((check) => check.label),
  };
}

export function decisionGate(entry) {
  const checks = {
    screeningEvidence: entry?.screening_loss_ratio != null && entry?.screening_novelty != null,
    investigationEvidence: entry?.investigation_loss_ratio != null && entry?.investigation_robustness != null,
    robustnessFloor: entry?.investigation_robustness != null && entry.investigation_robustness >= 0.5,
    validationEvidence: entry?.validation_loss_ratio != null
      && entry?.validation_baseline_ratio != null
      && entry?.validation_multi_seed_std != null,
    baselineBeatsReference: entry?.validation_baseline_ratio != null && entry.validation_baseline_ratio < 1.0,
    consistencyBounded: entry?.validation_multi_seed_std != null && entry.validation_multi_seed_std <= 0.12,
    ckaArtifactBacked: entry?.cka_source === 'artifact',
  };
  const decisionReady = Object.values(checks).every(Boolean);
  const missing = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  return {
    decisionReady,
    label: decisionReady ? 'Decision-Ready' : 'Exploratory',
    color: decisionReady ? 'var(--accent-green)' : 'var(--accent-yellow)',
    missing,
    checks,
  };
}
