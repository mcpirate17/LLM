import { useState, useCallback, useEffect, useMemo } from 'react';
import useLocalStorage from './useLocalStorage';

const INVESTIGATION_QUEUE_KEY = 'aria_investigation_queue_v1';

function normalizeQueue(items) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const normalized = [];
  for (const item of items) {
    const resultId = item?.resultId;
    if (!resultId || seen.has(resultId)) continue;
    seen.add(resultId);
    normalized.push({
      resultId,
      fingerprint: item?.fingerprint || null,
      source: item?.source || 'unknown',
      architectureFamily: item?.architectureFamily || null,
      intent: item?.intent === 'validation' ? 'validation' : 'investigation',
    });
  }
  return normalized;
}

function queueReasonLabel(reason) {
  if (reason === 'already_investigated_unchanged') {
    return 'Candidate already has investigation evidence and is unchanged.';
  }
  if (reason === 'not_investigation_passed') {
    return 'Candidate is investigation-tier but has not passed robustness gate.';
  }
  if (reason === 'already_promoted') {
    return 'Candidate is already in validation/breakthrough tier.';
  }
  if (reason === 'not_progression_eligible') {
    return 'Candidate is not currently eligible for investigation/validation progression.';
  }
  return 'Candidate is not eligible for this queue action.';
}

function resolveQueueIntent(candidate, eligibility) {
  if (candidate?.intent === 'investigation' || candidate?.intent === 'validation') {
    return candidate.intent;
  }
  if (candidate?.validationEligible || eligibility?.validationEligible) {
    return 'validation';
  }
  if (candidate?.investigationEligible || eligibility?.investigationEligible) {
    return 'investigation';
  }
  return null;
}

export default function useInvestigationQueue({ eligibilityByResultId, setActionError }) {
  const [investigationQueue, setInvestigationQueue] = useLocalStorage(INVESTIGATION_QUEUE_KEY, []);

  const addToInvestigationQueue = useCallback((candidate) => {
    const resultId = candidate?.resultId;
    if (!resultId) return;
    const eligibility = eligibilityByResultId[resultId] || null;
    if (candidate?.queueEligible === false && !eligibility?.queueEligible) {
      setActionError(queueReasonLabel(candidate?.queueReason));
      return;
    }
    const intent = resolveQueueIntent(candidate, eligibility);
    if (!intent) {
      setActionError(queueReasonLabel(candidate?.queueReason));
      return;
    }
    setInvestigationQueue(prev => normalizeQueue([
      ...prev.filter(item => item.resultId !== resultId),
      {
        resultId,
        fingerprint: candidate?.fingerprint || null,
        source: candidate?.source || 'unknown',
        architectureFamily: candidate?.architectureFamily || null,
        intent,
      },
    ]));
  }, [eligibilityByResultId, setActionError, setInvestigationQueue]);

  const removeFromInvestigationQueue = useCallback((resultId) => {
    setInvestigationQueue(prev => prev.filter(item => item.resultId !== resultId));
  }, [setInvestigationQueue]);

  const clearInvestigationQueue = useCallback(() => {
    setInvestigationQueue([]);
  }, [setInvestigationQueue]);

  // Prune queue items that are no longer eligible for their declared intent
  useEffect(() => {
    setInvestigationQueue(prev => {
      let changed = false;
      const next = [];
      for (const item of prev) {
        const intent = item?.intent === 'validation' ? 'validation' : 'investigation';
        const eligibility = eligibilityByResultId[item.resultId];
        if (!eligibility) {
          if (item.intent !== intent) {
            changed = true;
            next.push({ ...item, intent });
          } else {
            next.push(item);
          }
          continue;
        }
        const stillEligibleForIntent = intent === 'validation'
          ? eligibility.validationEligible
          : eligibility.investigationEligible;
        if (!stillEligibleForIntent) {
          changed = true;
          continue;
        }
        if (item.intent !== intent) {
          changed = true;
          next.push({ ...item, intent });
        } else {
          next.push(item);
        }
      }
      return changed ? next : prev;
    });
  }, [eligibilityByResultId, setInvestigationQueue]);

  const queueBreakdown = useMemo(() => {
    return investigationQueue.reduce((acc, item) => {
      if (item.intent === 'validation') {
        acc.validation += 1;
      } else {
        acc.investigation += 1;
      }
      return acc;
    }, { investigation: 0, validation: 0 });
  }, [investigationQueue]);

  return {
    investigationQueue,
    addToInvestigationQueue,
    removeFromInvestigationQueue,
    clearInvestigationQueue,
    queueBreakdown,
  };
}
