export const LONG_ACTION_TIMEOUT_MS = 120000;

export function summarizePreflightBlock(err, fallbackMessage) {
  const preflight = err?.preflight || {};
  const verdict = String(preflight?.verdict || '').toUpperCase();
  const checks = Array.isArray(preflight?.checks) ? preflight.checks : [];
  const failingCheck = checks.find((check) => check?.status === 'fail') || checks.find((check) => check?.status === 'warn');
  const detail = failingCheck?.message || failingCheck?.name || '';
  return [err?.error || fallbackMessage, verdict ? `(${verdict})` : '', detail].filter(Boolean).join(' ');
}

export async function parseErrorPayload(response, fallbackMessage) {
  try {
    return await response.json();
  } catch {
    return { error: fallbackMessage || `HTTP ${response.status}` };
  }
}

export function buildEligibilityFilter(eligibilityByResultId) {
  return (mode, resultIds) => {
    const ids = Array.isArray(resultIds) ? resultIds.filter(Boolean) : [];
    if (!ids.length) {
      return {
        ok: false,
        eligibleIds: [],
        message: `No result ids provided for ${mode} action.`,
      };
    }

    const eligibilityKey = mode === 'confirmation'
      ? 'confirmationEligible'
      : mode === 'capability_ranking'
        ? 'capabilityRankingEligible'
      : mode === 'validation'
        ? 'validationEligible'
        : 'investigationEligible';
    const eligibleIds = [];
    const ineligibleIds = [];
    for (const resultId of ids) {
      (eligibilityByResultId[resultId]?.[eligibilityKey] ? eligibleIds : ineligibleIds).push(resultId);
    }

    if (!eligibleIds.length) {
      const label = ineligibleIds.slice(0, 3).join(', ') || 'unknown';
      return {
        ok: false,
        eligibleIds: [],
        message: `No eligible ${mode} candidates found. Ineligible: ${label}.`,
      };
    }

    if (ineligibleIds.length > 0) {
      const label = ineligibleIds.slice(0, 3).join(', ');
      return {
        ok: true,
        eligibleIds,
        message: `Skipping ${ineligibleIds.length} ineligible ${mode} candidate${ineligibleIds.length === 1 ? '' : 's'} (${label}).`,
      };
    }

    return { ok: true, eligibleIds, message: null };
  };
}
