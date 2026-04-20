import { useReducer, useEffect, useRef } from 'react';
import { apiCall, postJson } from '../services/apiService';
import useCopyToClipboard from './useCopyToClipboard';
import apiService from '../services/apiService';
import { summarizeRefineTrace } from '../components/programDetail/refineTraceSummary';

function programDetailReducer(state, action) {
  switch (action.type) {
    case 'FETCH_START':
      return {
        ...state,
        loading: true,
        error: null,
        latestRefineLaunch: null,
        refineLaunchHistory: [],
        refineTrace: null,
        linkedHypothesis: null,
        linkedDecision: null,
        linkedExperiment: null,
        linkedCampaign: null,
      };
    case 'FETCH_SUCCESS':
      return { ...state, loading: false, program: action.payload, error: null };
    case 'FETCH_ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'SET_LINKED_DATA':
      return { ...state, ...action.payload };
    case 'SET_LEADERBOARD_ENTRY':
      return { ...state, leaderboardEntry: action.payload };
    case 'SET_REFINE_ANALYSIS':
      return {
        ...state,
        refineAnalysis: action.payload.data,
        refineAnalysisLoading: action.payload.loading,
        refineAnalysisError: action.payload.error,
      };
    case 'SET_REFINE_TRACE':
      return {
        ...state,
        refineTrace: action.payload.trace,
        refineTraceLoading: action.payload.loading,
        refineLaunchHistory: action.payload.history || state.refineLaunchHistory,
      };
    case 'SET_DRAWER':
      return { ...state, ...action.payload };
    case 'SET_MODAL':
      return { ...state, ...action.payload };
    case 'SET_ACTION':
      return {
        ...state,
        actionStarting: action.payload.starting,
        actionError: action.payload.error,
        latestRefineLaunch: action.payload.latestRefineLaunch || state.latestRefineLaunch,
        refineLaunchHistory: action.payload.refineLaunchHistory || state.refineLaunchHistory,
      };
    case 'SET_BACKFILL':
      return { ...state, ...action.payload };
    case 'SET_DECISION_PACKET':
      return {
        ...state,
        decisionPacket: action.payload.packet,
        decisionPacketLoading: action.payload.loading,
        decisionPacketError: action.payload.error,
      };
    case 'SET_UI':
      return { ...state, ...action.payload };
    default:
      return state;
  }
}

const initialState = {
  program: null,
  loading: true,
  error: null,
  scaleUpOpen: false,
  scaleUpConfig: { steps: 5000, batch_size: 8, seq_len: 512 },
  scaleUpStarting: false,
  manualRunOpen: false,
  manualRunStarting: false,
  manualRunConfig: {
    steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256,
    data_source: 'corpus',
    hf_dataset: 'roneneldan/TinyStories', hf_subset: '',
    tokenizer: 'byte',
  },
  backfillRunning: false,
  backfillResult: null,
  lossBackfillRunning: false,
  lossBackfillResult: null,
  leaderboardEntry: null,
  actionStarting: null,
  actionError: null,
  overrideIneligible: false,
  linkedHypothesis: null,
  linkedDecision: null,
  linkedExperiment: null,
  linkedCampaign: null,
  provenanceOpen: true,
  decisionPacket: null,
  decisionPacketLoading: false,
  decisionPacketError: null,
  decisionPacketOpen: true,
  manifestLoading: false,
  latestRefineLaunch: null,
  refineLaunchHistory: [],
  refineTrace: null,
  refineTraceLoading: false,
  refineAnalysis: null,
  refineAnalysisLoading: false,
  refineAnalysisError: null,
  drawerWidthVw: 45,
  drawerMaximized: false,
  resizingDrawer: false,
};

export default function useProgramData({ resultId, defaultOverrideIneligible, onActionComplete, onClose, eligibilityByResultId }) {
  const [state, dispatch] = useReducer(programDetailReducer, {
    ...initialState,
    overrideIneligible: Boolean(defaultOverrideIneligible),
  });

  const {
    program, loading, error, leaderboardEntry, latestRefineLaunch,
    refineLaunchHistory, refineTrace, refineAnalysis, resizingDrawer,
    drawerWidthVw, overrideIneligible, actionStarting,
  } = state;

  const [manifestCopied, copyManifest] = useCopyToClipboard();
  const drawerResizeRef = useRef({ startX: 0, startVw: 45 });

  const fetchAndCopyManifest = () => {
    if (!resultId) return;
    dispatch({ type: 'SET_UI', payload: { manifestLoading: true } });
    apiService.getReproducibilityManifest(resultId)
      .then(d => {
        copyManifest(JSON.stringify(d, null, 2));
        dispatch({ type: 'SET_UI', payload: { manifestLoading: false } });
      })
      .catch(() => { dispatch({ type: 'SET_UI', payload: { manifestLoading: false } }); });
  };

  const fetchDecisionPacket = () => {
    if (!resultId) return;
    dispatch({ type: 'SET_DECISION_PACKET', payload: { loading: true, error: null } });
    apiService.getDecisionPacket(resultId)
      .then(d => { dispatch({ type: 'SET_DECISION_PACKET', payload: { packet: d, loading: false } }); })
      .catch(e => { dispatch({ type: 'SET_DECISION_PACKET', payload: { error: 'Failed: ' + e.message, loading: false } }); });
  };

  // Sync defaultOverrideIneligible prop
  useEffect(() => {
    dispatch({ type: 'SET_UI', payload: { overrideIneligible: Boolean(defaultOverrideIneligible) } });
  }, [defaultOverrideIneligible, resultId]);

  // Fetch program + leaderboard
  useEffect(() => {
    if (!resultId) return;
    const controller = new AbortController();
    const { signal } = controller;
    dispatch({ type: 'FETCH_START' });

    apiCall(`/api/programs/${encodeURIComponent(resultId)}`, { signal, timeoutMs: 30000 })
      .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error || `HTTP ${r.status}`))))
      .then(d => {
        if (signal.aborted) return;
        dispatch({ type: 'FETCH_SUCCESS', payload: d });
        if (d?.experiment_id) {
          apiCall(`/api/experiments/${encodeURIComponent(d.experiment_id)}`, { signal })
            .then(r => r.ok ? r.json() : null)
            .then(expData => {
              if (signal.aborted || !expData?.experiment) return;
              const linkedData = { linkedExperiment: expData.experiment };
              if (expData.experiment.campaign_id) {
                linkedData.linkedCampaign = { campaign_id: expData.experiment.campaign_id, title: expData.experiment.campaign_title || expData.experiment.campaign_id };

                Promise.all([
                  apiCall(`/api/campaigns/${encodeURIComponent(expData.experiment.campaign_id)}/hypotheses`, { signal }).then(r => r.ok ? r.json() : []),
                  apiCall(`/api/campaigns/${encodeURIComponent(expData.experiment.campaign_id)}/decisions`, { signal }).then(r => r.ok ? r.json() : []),
                ]).then(([hyps, decs]) => {
                  if (signal.aborted) return;
                  const linkedHyp = (Array.isArray(hyps) ? hyps : []).find(
                    h => h.experiment_id === d.experiment_id
                  );
                  const linkedDec = (Array.isArray(decs) ? decs : []).find(dec => {
                    const evidenceIds = dec.evidence_ids || [];
                    return Array.isArray(evidenceIds) && evidenceIds.includes(resultId);
                  });
                  dispatch({
                    type: 'SET_LINKED_DATA',
                    payload: { ...linkedData, linkedHypothesis: linkedHyp, linkedDecision: linkedDec }
                  });
                }).catch(() => {
                  if (!signal.aborted) dispatch({ type: 'SET_LINKED_DATA', payload: linkedData });
                });
              } else {
                dispatch({ type: 'SET_LINKED_DATA', payload: linkedData });
              }
            })
            .catch(() => {});
        }
      })
      .catch(e => {
        if (!signal.aborted) {
          dispatch({ type: 'FETCH_ERROR', payload: 'Failed to load program: ' + e.message });
        }
      });

    apiCall('/api/leaderboard?limit=200&trusted_only=1', { signal })
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!signal.aborted && data?.entries) {
          const entry = data.entries.find(e => e.result_id === resultId);
          dispatch({ type: 'SET_LEADERBOARD_ENTRY', payload: entry || null });
        }
      })
      .catch(() => {});

    return () => controller.abort();
  }, [resultId]);

  // Fetch refine analysis
  useEffect(() => {
    if (!resultId || !program?.stage1_passed || program?.is_reference) return;
    dispatch({ type: 'SET_REFINE_ANALYSIS', payload: { loading: true, error: null } });
    apiCall(`/api/programs/${encodeURIComponent(resultId)}/refine-analysis`)
      .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error || 'Failed'))))
      .then(data => { dispatch({ type: 'SET_REFINE_ANALYSIS', payload: { data, loading: false } }); })
      .catch(e => { dispatch({ type: 'SET_REFINE_ANALYSIS', payload: { error: e.message, loading: false } }); });
  }, [resultId, program?.stage1_passed, program?.is_reference]);

  // Poll refinement trace
  useEffect(() => {
    if (!latestRefineLaunch?.experimentId || !resultId) return;
    let cancelled = false;
    let intervalId = null;
    const pollTrace = async () => {
      if (cancelled) return;
      dispatch({ type: 'SET_REFINE_TRACE', payload: { loading: true } });
      try {
        const response = await apiCall(`/api/experiments/${latestRefineLaunch.experimentId}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        if (cancelled) return;
        const tracePayload = summarizeRefineTrace(payload, resultId, program?.graph_fingerprint);

        const nextHistory = refineLaunchHistory.map(item => (
          item.experimentId === latestRefineLaunch.experimentId
            ? {
                ...item,
                status: tracePayload.status,
                topCandidate: tracePayload.newCandidates?.[0] || null,
              }
            : item
        ));

        dispatch({
          type: 'SET_REFINE_TRACE',
          payload: { trace: tracePayload, loading: false, history: nextHistory }
        });

        if (tracePayload.completed && intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      } catch (e) {
        if (!cancelled) {
          dispatch({
            type: 'SET_REFINE_TRACE',
            payload: { trace: { error: e?.message || 'Failed to load refinement trace' }, loading: false }
          });
        }
      }
    };
    pollTrace();
    intervalId = setInterval(pollTrace, 4000);
    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [latestRefineLaunch, resultId, program?.graph_fingerprint]);

  // Drawer resize
  useEffect(() => {
    if (!resizingDrawer) return undefined;
    const onMouseMove = (event) => {
      const viewportWidth = window.innerWidth || 1;
      const deltaPx = drawerResizeRef.current.startX - event.clientX;
      const deltaVw = (deltaPx / viewportWidth) * 100;
      const nextVw = drawerResizeRef.current.startVw + deltaVw;
      dispatch({ type: 'SET_DRAWER', payload: { drawerWidthVw: Math.max(35, Math.min(90, nextVw)) } });
    };
    const onMouseUp = () => {
      dispatch({ type: 'SET_DRAWER', payload: { resizingDrawer: false } });
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [resizingDrawer]);

  // Derived eligibility
  const entryTier = typeof leaderboardEntry?.tier === 'string' ? leaderboardEntry.tier : (typeof program?.tier === 'string' ? program.tier : '');
  const tier = String(entryTier || '').toLowerCase();
  const capabilityStatus = String(leaderboardEntry?.capability_quality?.status || '').toLowerCase();
  const hasInvestigationEvidence = leaderboardEntry?.investigation_loss_ratio != null;
  const hasValidationEvidence = (
    leaderboardEntry?.validation_loss_ratio != null
    || leaderboardEntry?.validation_baseline_ratio != null
    || Boolean(leaderboardEntry?.validation_passed)
  );
  const isCapabilityQualified = capabilityStatus === 'qualified' || capabilityStatus === 'breakthrough';
  const alreadyInvestigated = Boolean(
    hasInvestigationEvidence || tier === 'investigation' || tier === 'validation' || tier === 'breakthrough'
  );
  const alreadyValidated = Boolean(
    tier === 'breakthrough' || isCapabilityQualified
  );
  const fallbackEligibility = {
    investigationEligible: Boolean(program?.stage1_passed) && (tier === 'screening' && !hasInvestigationEvidence),
    validationEligible: (
      (tier === 'investigation' && Boolean(leaderboardEntry?.investigation_passed ?? program?.investigation_passed))
      || (tier === 'validation' && !isCapabilityQualified)
    ),
  };
  const resolvedEligibility = eligibilityByResultId?.[resultId] || fallbackEligibility;
  const canInvestigate = Boolean(resolvedEligibility.investigationEligible || overrideIneligible);
  const canValidate = Boolean(resolvedEligibility.validationEligible || overrideIneligible);

  // Action handlers
  const handleLaunchRefinement = async (intent, actionKey, failureLabel) => {
    dispatch({ type: 'SET_ACTION', payload: { starting: actionKey, error: null } });
    try {
      const res = await postJson('/api/experiments/start', {
        mode: 'refine_fingerprint',
        graph_fingerprints: [program.graph_fingerprint],
        n_programs: 24,
        model_source: 'fingerprint_refine',
        refine_intent: intent,
        mutation_rate: 0.85,
        preflight_override: true,
        enforce_preflight: true,
        ...(refineAnalysis ? { refine_analysis_json: refineAnalysis } : {}),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        dispatch({ type: 'SET_ACTION', payload: { starting: null, error: payload.error || failureLabel } });
      } else {
        const resolved = payload?.refine_resolution || {};
        const launchData = {
          experimentId: payload?.experiment_id,
          intent,
          startedAt: Date.now(),
          sourceResultId: resultId,
          sourceFingerprint: program?.graph_fingerprint,
          resolvedResultIds: Array.isArray(resolved?.result_ids) ? resolved.result_ids : [],
          resolvedFingerprints: Array.isArray(resolved?.resolved_fingerprints) ? resolved.resolved_fingerprints : [],
          unresolvedFingerprints: Array.isArray(resolved?.unresolved_fingerprints) ? resolved.unresolved_fingerprints : [],
        };

        const nextItem = {
          experimentId: payload?.experiment_id,
          intent,
          startedAt: Date.now(),
          sourceResultId: resultId,
          sourceFingerprint: program?.graph_fingerprint,
          status: 'running',
          topCandidate: null,
        };
        const nextHistory = [nextItem, ...refineLaunchHistory.filter(item => item.experimentId !== nextItem.experimentId)].slice(0, 3);

        dispatch({
          type: 'SET_ACTION',
          payload: { starting: null, error: null, latestRefineLaunch: launchData, refineLaunchHistory: nextHistory }
        });
        if (onActionComplete) onActionComplete();
      }
    } catch (e) {
      dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
    }
  };

  return {
    state,
    dispatch,
    // Derived
    tier,
    alreadyInvestigated,
    alreadyValidated,
    canInvestigate,
    canValidate,
    // Handlers
    handleLaunchRefinement,
    fetchDecisionPacket,
    fetchAndCopyManifest,
    manifestCopied,
    drawerResizeRef,
  };
}
