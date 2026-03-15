import { apiCall } from "../services/apiService";
import React, { useReducer, useEffect, useRef } from 'react';
import { lossColor, noveltyColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import apiService from '../services/apiService';
import FingerprintRadar from './program/FingerprintRadar';
import SparsityDiagnostics from './program/SparsityDiagnostics';
import TrainingCurve from './program/TrainingCurve';
import RobustnessProfile from './program/RobustnessProfile';
import AriaAdvice from './program/AriaAdvice';
import ReferenceComparison from './program/ReferenceComparison';
import MetricRow from './program/MetricRow';
import HypothesisInfo from './program/HypothesisInfo';
import BenchmarkEvidenceSnapshot from './program/BenchmarkEvidenceSnapshot';
import ExternalBenchmarkCard from './program/ExternalBenchmarkCard';
import TokenMixingTaxonomy from './program/TokenMixingTaxonomy';
import GatingDiagnostics from './program/GatingDiagnostics';
import RefinementRationale from './program/RefinementRationale';
import RefinementLineage from './program/RefinementLineage';
import RefinementAdvisor from './program/RefinementAdvisor';
import { DecisionPacketPanel, ProgramHeaderSection, ProvenancePanel, RefinementTracePanel } from './programDetail/InfoPanels';
import { summarizeRefineTrace } from './programDetail/refineTraceSummary';

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

function ProgramDetail({ resultId, onClose, onActionComplete, onSelectExperiment, onViewInLeaderboard, onSelectCampaign, onOpenInDesigner, onAddToComparison, eligibilityByResultId, defaultOverrideIneligible = false }) {
  const [state, dispatch] = useReducer(programDetailReducer, {
    ...initialState,
    overrideIneligible: Boolean(defaultOverrideIneligible)
  });

  const {
    program, loading, error, scaleUpOpen, scaleUpConfig, scaleUpStarting,
    manualRunOpen, manualRunStarting, manualRunConfig, backfillRunning,
    backfillResult, lossBackfillRunning, lossBackfillResult, leaderboardEntry,
    actionStarting, actionError, overrideIneligible, linkedHypothesis,
    linkedDecision, linkedExperiment, linkedCampaign, provenanceOpen,
    decisionPacket, decisionPacketLoading, decisionPacketError, decisionPacketOpen,
    manifestLoading, latestRefineLaunch, refineLaunchHistory, refineTrace,
    refineTraceLoading, refineAnalysis, refineAnalysisLoading, refineAnalysisError,
    drawerWidthVw, drawerMaximized, resizingDrawer
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

  const formatUnixTimestamp = (value) => {
    if (value == null) return null;
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    const ms = n > 1e12 ? n : n * 1000;
    return new Date(ms).toLocaleString();
  };

  useEffect(() => {
    dispatch({ type: 'SET_UI', payload: { overrideIneligible: Boolean(defaultOverrideIneligible) } });
  }, [defaultOverrideIneligible, resultId]);

  const fetchDecisionPacket = () => {
    if (!resultId) return;
    dispatch({ type: 'SET_DECISION_PACKET', payload: { loading: true, error: null } });
    apiService.getDecisionPacket(resultId)
      .then(d => { dispatch({ type: 'SET_DECISION_PACKET', payload: { packet: d, loading: false } }); })
      .catch(e => { dispatch({ type: 'SET_DECISION_PACKET', payload: { error: 'Failed: ' + e.message, loading: false } }); });
  };

  useEffect(() => {
    if (!resultId) return;
    dispatch({ type: 'FETCH_START' });

    apiService.getProgram(resultId)
      .then(d => {
        dispatch({ type: 'FETCH_SUCCESS', payload: d });
        if (d?.experiment_id) {
          apiService.getExperiment(d.experiment_id)
            .then(expData => {
              if (expData?.experiment) {
                const linkedData = { linkedExperiment: expData.experiment };
                if (expData.experiment.campaign_id) {
                  linkedData.linkedCampaign = { campaign_id: expData.experiment.campaign_id, title: expData.experiment.campaign_title || expData.experiment.campaign_id };
                  
                  Promise.all([
                    apiService.getCampaignHypotheses(expData.experiment.campaign_id),
                    apiService.getCampaignDecisions(expData.experiment.campaign_id)
                  ]).then(([hyps, decs]) => {
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
                    dispatch({ type: 'SET_LINKED_DATA', payload: linkedData });
                  });
                } else {
                  dispatch({ type: 'SET_LINKED_DATA', payload: linkedData });
                }
              }
            })
            .catch(() => {});
        }
      })
      .catch(e => { dispatch({ type: 'FETCH_ERROR', payload: 'Failed to load program: ' + e.message }); });

    apiService.getLeaderboard('?limit=200')
      .then(data => {
        if (data?.entries) {
          const entry = data.entries.find(e => e.result_id === resultId);
          dispatch({ type: 'SET_LEADERBOARD_ENTRY', payload: entry || null });
        }
      })
      .catch(() => {});
  }, [resultId]);

  useEffect(() => {
    if (!resultId || !program?.stage1_passed) return;
    dispatch({ type: 'SET_REFINE_ANALYSIS', payload: { loading: true, error: null } });
    apiCall(`/api/programs/${encodeURIComponent(resultId)}/refine-analysis`)
      .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error || 'Failed'))))
      .then(data => { dispatch({ type: 'SET_REFINE_ANALYSIS', payload: { data, loading: false } }); })
      .catch(e => { dispatch({ type: 'SET_REFINE_ANALYSIS', payload: { error: e.message, loading: false } }); });
  }, [resultId, program?.stage1_passed]);

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

  if (!resultId) return null;
  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';
  const fmtMs = v => v != null ? `${Number(v).toFixed(1)}ms` : '--';
  const fmtMem = v => v != null ? `${Number(v).toFixed(1)}MB` : '--';
  const fmtInt = v => v != null ? Number(v).toLocaleString() : '--';
  const shortId = (v, n = 12) => { const s = String(v || '').trim(); return !s ? '--' : (s.length > n ? s.slice(0, n) : s); };
  const entryTier = typeof leaderboardEntry?.tier === 'string' ? leaderboardEntry.tier : (typeof program?.tier === 'string' ? program.tier : '');
  const tier = String(entryTier || '').toLowerCase();
  const hasInvestigationEvidence = leaderboardEntry?.investigation_loss_ratio != null;
  const hasValidationEvidence = (
    leaderboardEntry?.validation_loss_ratio != null
    || leaderboardEntry?.validation_baseline_ratio != null
    || Boolean(leaderboardEntry?.validation_passed)
  );
  const alreadyInvestigated = Boolean(
    hasInvestigationEvidence || tier === 'investigation' || tier === 'validation' || tier === 'breakthrough'
  );
  const alreadyValidated = Boolean(
    tier === 'validation' || tier === 'breakthrough' || hasValidationEvidence
  );
  const fallbackEligibility = {
    investigationEligible: Boolean(program?.stage1_passed) && (tier === 'screening' && !hasInvestigationEvidence),
    validationEligible: tier === 'investigation' && Boolean(leaderboardEntry?.investigation_passed ?? program?.investigation_passed),
  };
  const resolvedEligibility = eligibilityByResultId?.[resultId] || fallbackEligibility;
  const canInvestigate = Boolean(resolvedEligibility.investigationEligible || overrideIneligible);
  const canValidate = Boolean(resolvedEligibility.validationEligible || overrideIneligible);
  const investigateDisabled = actionStarting === 'investigate' || !canInvestigate;
  const validateDisabled = actionStarting === 'validate' || !canValidate;
  const investigateTitle = canInvestigate
    ? 'Deep study with multiple training programs'
    : 'Already investigated for this fingerprint. Enable override to run anyway.';
  const validateTitle = canValidate
    ? 'Publication-grade multi-seed validation'
    : 'Already validated or not eligible. Enable override to run anyway.';
  const refinementTraceLabels = {
    panelTitle: 'Refinement Trace',
    openRefinementRun: 'Open Refinement Run',
    viewTopRefinedResult: 'View Top Refined Result',
    newFingerprints: 'New Fingerprints',
    openFingerprint: 'Open Fingerprint',
    recentRefinementLaunches: 'Recent Refinement Launches',
  };
  // Reducer-owned equivalents of the legacy setters keep the refinement launch
  // contract centralized here: setLatestRefineLaunch, setRefineLaunchHistory,
  // and lastRefinedCandidate all flow through latestRefineLaunch/refineLaunchHistory/refineTrace.

  const handleLaunchRefinement = async (intent, actionKey, failureLabel) => {
    dispatch({ type: 'SET_ACTION', payload: { starting: actionKey, error: null } });
    try {
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'refine_fingerprint',
          graph_fingerprints: [program.graph_fingerprint],
          n_programs: 24,
          model_source: 'fingerprint_refine',
          refine_intent: intent,
          mutation_rate: 0.85,
          preflight_override: true,
          enforce_preflight: true,
          ...(refineAnalysis ? { refine_analysis_json: refineAnalysis } : {}),
        }),
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

  return (
    <div className="program-drawer-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div
        className="program-drawer"
        onClick={e => e.stopPropagation()}
        onMouseDown={e => e.stopPropagation()}
        style={drawerMaximized
          ? { width: '100%', minWidth: 0, maxWidth: '100%' }
          : { width: `${drawerWidthVw}vw`, minWidth: 460, maxWidth: '95vw' }}
      >
        {!drawerMaximized && (
          <div
            onMouseDown={(event) => {
              event.preventDefault();
              event.stopPropagation();
              drawerResizeRef.current = { startX: event.clientX, startVw: drawerWidthVw };
              dispatch({ type: 'SET_DRAWER', payload: { resizingDrawer: true } });
            }}
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: 8,
              cursor: 'col-resize',
              background: resizingDrawer ? 'rgba(88, 166, 255, 0.2)' : 'transparent',
              borderLeft: '1px solid rgba(88, 166, 255, 0.35)',
              zIndex: 2,
            }}
            title="Drag to resize"
            aria-hidden="true"
          />
        )}
        <div className="program-drawer-header">
          <span>Program Detail</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              className="refresh-btn"
              aria-pressed={drawerMaximized}
              onClick={() => dispatch({ type: 'SET_DRAWER', payload: { drawerMaximized: !drawerMaximized } })}
              style={{ fontSize: 16, padding: '4px 8px' }}
              title={drawerMaximized ? 'Exit fullscreen' : 'Expand to fullscreen'}
            >
              {drawerMaximized ? '\u2750' : '\u2922'}
            </button>
            <button className="close-btn" onClick={onClose}>&times;</button>
          </div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: 24 }}>
        {loading ? (
          <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
        ) : error ? (
          <p style={{ color: 'var(--accent-red)' }}>{error}</p>
        ) : !program ? (
          <p style={{ color: 'var(--accent-red)' }}>Program not found</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <ProgramHeaderSection program={program} leaderboardEntry={leaderboardEntry} />
            <ProvenancePanel
              program={program}
              linkedHypothesis={linkedHypothesis}
              leaderboardEntry={leaderboardEntry}
              linkedCampaign={linkedCampaign}
              linkedExperiment={linkedExperiment}
              provenanceOpen={provenanceOpen}
              setProvenanceOpen={(val) => dispatch({ type: 'SET_UI', payload: { provenanceOpen: val } })}
              formatUnixTimestamp={formatUnixTimestamp}
              onClose={onClose}
              onSelectExperiment={onSelectExperiment}
              onAddToComparison={onAddToComparison}
              onOpenInDesigner={onOpenInDesigner}
              onViewInLeaderboard={onViewInLeaderboard}
              onSelectCampaign={onSelectCampaign}
              decisionPacket={decisionPacket}
              decisionPacketLoading={decisionPacketLoading}
              fetchDecisionPacket={fetchDecisionPacket}
              manifestLoading={manifestLoading}
              manifestCopied={manifestCopied}
              fetchAndCopyManifest={fetchAndCopyManifest}
              resultId={resultId}
            />
            <DecisionPacketPanel
              decisionPacket={decisionPacket}
              decisionPacketError={decisionPacketError}
              decisionPacketOpen={decisionPacketOpen}
              setDecisionPacketOpen={(val) => dispatch({ type: 'SET_UI', payload: { decisionPacketOpen: val } })}
            />
            {/* Error if failed */}
            {(program.error_message || program.stage0_error) && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--accent-red)',
              }}>
                {program.error_type && (
                  <span style={{ fontWeight: 600 }}>[{program.error_type}] </span>
                )}
                {program.error_message || program.stage0_error}
              </div>
            )}
            {actionError && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                color: 'var(--accent-red)',
              }}>
                {actionError}
              </div>
            )}
            {/* Refinement Trace */}
            <RefinementTracePanel
              latestRefineLaunch={latestRefineLaunch}
              labels={refinementTraceLabels}
              shortId={shortId}
              onClose={onClose}
              onSelectExperiment={onSelectExperiment}
              onViewInLeaderboard={onViewInLeaderboard}
              refineTraceLoading={refineTraceLoading}
              refineTrace={refineTrace}
              fmtInt={fmtInt}
              fmt={fmt}
              refineLaunchHistory={refineLaunchHistory}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Core Metrics
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Parameters" value={program.param_count ? `${(program.param_count / 1e6).toFixed(2)}M` : null} />
                  <MetricRow label="Loss Ratio" value={program.loss_ratio != null ?
                    <span style={{
                      color: lossColor(program.loss_ratio),
                      fontWeight: program.loss_ratio < 0.5 ? 600 : 'normal',
                    }} title={program.loss_ratio < 0.5 ? 'Learned quickly — strong candidate' : program.loss_ratio < 0.7 ? 'Moderate learning' : 'Slow learning'}>
                      {fmt(program.loss_ratio)}
                    </span> : null} />
                  <MetricRow label="Final Loss" value={fmt(program.final_loss)} />
                  <MetricRow label="Discovery Loss" value={program.discovery_loss != null ? fmt(program.discovery_loss) : null} />
                  <MetricRow label="Discovery LR" value={program.discovery_loss_ratio != null ? fmt(program.discovery_loss_ratio) : null} />
                  <MetricRow label="Validation Loss" value={program.validation_loss != null ? fmt(program.validation_loss) : null} />
                  <MetricRow label="Validation LR" value={program.validation_loss_ratio != null ? fmt(program.validation_loss_ratio) : null} />
                  <MetricRow label="Gen Gap" value={program.generalization_gap != null ? fmt(program.generalization_gap) : null} />
                  <MetricRow label="Baseline Ratio" value={program.baseline_loss_ratio != null ?
                    <span style={{
                      color: program.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)',
                      fontWeight: program.baseline_loss_ratio < 1 ? 600 : 'normal',
                    }} title={program.baseline_loss_ratio < 1 ? 'Beats a standard transformer!' : 'Underperforms a transformer of same size'}>
                      {fmt(program.baseline_loss_ratio)} {program.baseline_loss_ratio < 1 ? '(beats transformer)' : ''}
                    </span> : null} />
                  <MetricRow label="Throughput" value={program.throughput_tok_s != null ? `${Number(program.throughput_tok_s).toFixed(0)} tok/s` : null} />
                  <MetricRow label="Param Efficiency" value={program.param_efficiency != null ? fmt(program.param_efficiency) : (leaderboardEntry?.param_efficiency != null ? fmt(leaderboardEntry.param_efficiency) : null)} />
                  <MetricRow label="Sample Efficiency" value={program.sample_efficiency != null ?
                    <span style={{
                      color: program.sample_efficiency >= 0.8 ? 'var(--accent-green)' : program.sample_efficiency >= 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)',
                    }} title={`Converges to 25% initial loss in ${((1 - program.sample_efficiency) * 100).toFixed(0)}% of training budget`}>
                      {fmt(program.sample_efficiency, 3)}
                    </span> : null} />
                  <MetricRow label="Novelty" value={program.novelty_score != null ?
                    <span style={{
                      color: noveltyColor(program.novelty_score),
                    }} title={program.novelty_score > 0.8 ? 'Very different from known architectures' : program.novelty_score > 0.5 ? 'Moderately novel' : 'Similar to existing architectures'}>
                      {fmt(program.novelty_score, 3)}
                    </span> : null} />
                </div>
                <BenchmarkEvidenceSnapshot program={program} leaderboardEntry={leaderboardEntry} />
                <RobustnessProfile program={program} leaderboardEntry={leaderboardEntry} />
                <AriaAdvice analysis={refineAnalysis} />
                <ReferenceComparison program={program} leaderboardEntry={leaderboardEntry} />
                <ExternalBenchmarkCard program={program} />
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8, marginTop: 12 }}>
                  Sandbox Timing
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Compile" value={fmtMs(program.compile_time_ms)} />
                  <MetricRow label="Forward" value={fmtMs(program.forward_time_ms)} />
                  <MetricRow label="Backward" value={fmtMs(program.backward_time_ms)} />
                  <MetricRow label="Peak Memory" value={fmtMem(program.peak_memory_mb)} />
                  <MetricRow label="FLOPs (fwd)" value={program.flops_forward ? fmtInt(program.flops_forward) : null} />
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: -8 }}>
                  Fingerprint & Similarity
                </div>
                <div style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  gap: 16,
                  padding: '16px 12px',
                  background: 'var(--bg-secondary)',
                  borderRadius: 8,
                  border: '1px solid var(--border)'
                }}>
                  <FingerprintRadar program={program} size={260} />
                  <div style={{ width: '100%', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                      Primary Reference Class
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--accent-purple)', fontWeight: 600 }}>
                      {program.most_similar_to || 'Truly Novel (No match)'}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                      Highest CKA similarity among the known baseline families.
                    </div>
                  </div>
                  {(program.fp_cka_vs_transformer != null || program.fp_cka_vs_ssm != null) && (
                    <div style={{ width: '100%', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
                        CKA Similarity vs Baselines
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                        Higher percentages mean this program behaves more like that baseline family.
                      </div>
                      {[
                        { label: 'Transformer', value: program.fp_cka_vs_transformer, color: 'var(--accent-blue)' },
                        { label: 'SSM', value: program.fp_cka_vs_ssm, color: 'var(--accent-green)' },
                        { label: 'Conv', value: program.fp_cka_vs_conv, color: 'var(--accent-yellow)' },
                      ].map(({ label, value, color }) => value != null && (
                        <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <span style={{ fontSize: 11, color: 'var(--text-secondary)', minWidth: 70 }}>{label}</span>
                          <div style={{ flex: 1, height: 8, background: 'var(--bg-tertiary)', borderRadius: 4 }}>
                            <div style={{
                              width: `${Math.min(value, 1) * 100}%`, height: '100%',
                              background: color, borderRadius: 4, opacity: 0.7,
                            }} />
                          </div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 30, textAlign: 'right' }}>{(Number(value) * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8, marginTop: 12 }}>
                  Sandbox Timing
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Compile" value={fmtMs(program.compile_time_ms)} />
                  <MetricRow label="Forward" value={fmtMs(program.forward_time_ms)} />
                  <MetricRow label="Backward" value={fmtMs(program.backward_time_ms)} />
                  <MetricRow label="Peak Memory" value={fmtMem(program.peak_memory_mb)} />
                  <MetricRow label="FLOPs (fwd)" value={program.flops_forward ? fmtInt(program.flops_forward) : null} />
                </div>
              </div>
            </div>
            {program.initial_loss != null && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Training Metrics
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
                  <MetricRow label="Initial Loss" value={fmt(program.initial_loss)} />
                  <MetricRow label="Min Loss" value={fmt(program.min_loss)} />
                  <MetricRow label="Steps" value={program.n_train_steps} />
                  <MetricRow label="Avg Step Time" value={fmtMs(program.avg_step_time_ms)} />
                  <MetricRow label="Mean Grad Norm" value={fmt(program.mean_grad_norm, 3)} />
                  <MetricRow label="Max Grad Norm" value={fmt(program.max_grad_norm, 3)} />
                </div>
              </div>
            )}
            {program.has_training_curve && (
              <TrainingCurve resultId={resultId} />
            )}
            {program.llm_explanation && (
              <div style={{
                padding: 12,
                background: 'var(--bg-tertiary)',
                borderRadius: 4,
                borderLeft: '2px solid var(--accent-purple)',
                fontSize: 13,
                color: 'var(--text-secondary)',
                fontStyle: 'italic',
              }}>
                <div style={{ fontSize: 11, color: 'var(--accent-purple)', marginBottom: 4, fontWeight: 600, fontStyle: 'normal' }}>
                  ARIA'S ANALYSIS
                </div>
                {program.llm_explanation}
              </div>
            )}
            {linkedHypothesis && (
              <HypothesisInfo hypothesis={linkedHypothesis} />
            )}
            {linkedDecision && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
                borderLeft: `2px solid ${
                  linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                  linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)'
                }`,
                fontSize: 13,
              }}>
                <div style={{
                  fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
                  color: linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                         linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                }}>
                  Decision: {linkedDecision.decision_type?.replace('_', ' ')}
                </div>
                <div>{linkedDecision.rationale}</div>
              </div>
            )}
            <RefinementRationale program={program} />
            <RefinementLineage program={program} onViewInLeaderboard={onViewInLeaderboard} />
            {program.stage1_passed && (
              <RefinementAdvisor
                analysis={refineAnalysis}
                loading={refineAnalysisLoading}
                error={refineAnalysisError}
                onLaunchRefinement={handleLaunchRefinement}
                actionStarting={actionStarting}
              />
            )}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!scaleUpOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => dispatch({ type: 'SET_MODAL', payload: { scaleUpOpen: true } })}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(88, 166, 255, 0.15)', border: '1px solid rgba(88, 166, 255, 0.4)', color: 'var(--accent-blue)' }}
                  >
                    Scale Up This Architecture
                  </button>
                ) : (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                      Scale-Up Configuration
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                        <input type="number" min="1000" max="50000" step="1000"
                          value={scaleUpConfig.steps}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { scaleUpConfig: { ...scaleUpConfig, steps: parseInt(e.target.value) || 5000 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="4" max="16" step="1"
                          value={scaleUpConfig.batch_size}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { scaleUpConfig: { ...scaleUpConfig, batch_size: parseInt(e.target.value) || 8 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="256" max="1024" step="128"
                          value={scaleUpConfig.seq_len}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { scaleUpConfig: { ...scaleUpConfig, seq_len: parseInt(e.target.value) || 512 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={scaleUpStarting}
                        onClick={async () => {
                          dispatch({ type: 'SET_MODAL', payload: { scaleUpStarting: true } });
                          try {
                            dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                            const res = await apiCall(`/api/experiments/start`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({
                                mode: 'scale_up',
                                result_ids: [resultId],
                                scale_up_steps: scaleUpConfig.steps,
                                scale_up_batch_size: scaleUpConfig.batch_size,
                                scale_up_seq_len: scaleUpConfig.seq_len,
                                preflight_override: true,
                                enforce_preflight: true,
                              }),
                            });
                            if (!res.ok) {
                              const err = await res.json();
                              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start scale-up' } });
                            } else {
                              dispatch({ type: 'SET_MODAL', payload: { scaleUpOpen: false } });
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                          }
                          dispatch({ type: 'SET_MODAL', payload: { scaleUpStarting: false } });
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {scaleUpStarting ? 'Starting...' : 'Start Scale-Up'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => dispatch({ type: 'SET_MODAL', payload: { scaleUpOpen: false } })}
                        style={{ padding: '6px 12px', fontSize: 12 }}
                      >
                        Cancel
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                      Trains for {scaleUpConfig.steps} steps with batch={scaleUpConfig.batch_size}, seq={scaleUpConfig.seq_len}
                    </div>
                  </div>
                )}
              </div>
            )}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!manualRunOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunOpen: true } })}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(210, 153, 34, 0.15)', border: '1px solid rgba(210, 153, 34, 0.4)', color: 'var(--accent-yellow)' }}
                  >
                    Manual Training Run
                  </button>
                ) : (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                      Manual Training Configuration
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                        <input type="number" min="500" max="50000" step="500"
                          value={manualRunConfig.steps}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: parseInt(e.target.value) || 2500 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="1" max="32" step="1"
                          value={manualRunConfig.batch_size}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, batch_size: parseInt(e.target.value) || 4 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="64" max="2048" step="64"
                          value={manualRunConfig.seq_len}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, seq_len: parseInt(e.target.value) || 256 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Training Programs</label>
                        <input type="number" min="1" max="10" step="1"
                          value={manualRunConfig.n_training_programs}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, n_training_programs: parseInt(e.target.value) || 3 } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Data Source</label>
                        <select
                          value={manualRunConfig.data_source}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, data_source: e.target.value } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        >
                          <option value="corpus">Corpus</option>
                          <option value="random">Random</option>
                          <option value="huggingface">HuggingFace</option>
                        </select>
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Tokenizer</label>
                        <select
                          value={manualRunConfig.tokenizer}
                          onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, tokenizer: e.target.value } } })}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        >
                          <option value="byte">Byte (1 byte = 1 token)</option>
                          <option value="tiktoken">BPE / tiktoken (GPT-2, ~4x context)</option>
                          <option value="whitespace">Whitespace hash</option>
                        </select>
                      </div>
                      {manualRunConfig.data_source === 'huggingface' && (
                        <>
                          <div>
                            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Dataset</label>
                            <input type="text"
                              value={manualRunConfig.hf_dataset}
                              onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, hf_dataset: e.target.value } } })}
                              placeholder="roneneldan/TinyStories"
                              style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                            />
                          </div>
                          <div>
                            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Subset</label>
                            <input type="text"
                              value={manualRunConfig.hf_subset}
                              onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, hf_subset: e.target.value } } })}
                              placeholder="(optional)"
                              style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                            />
                          </div>
                        </>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: 1000, batch_size: 4, n_training_programs: 1, seq_len: 256 } } })}>
                        Quick
                      </button>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256 } } })}>
                        Standard
                      </button>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: 5000, batch_size: 8, n_training_programs: 5, seq_len: 512 } } })}>
                        Deep
                      </button>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={manualRunStarting}
                        onClick={async () => {
                          dispatch({ type: 'SET_MODAL', payload: { manualRunStarting: true } });
                          try {
                            dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                            const body = {
                              mode: 'investigation',
                              force: true,
                              result_ids: [resultId],
                              n_training_programs: manualRunConfig.n_training_programs,
                              investigation_steps: manualRunConfig.steps,
                              investigation_batch_size: manualRunConfig.batch_size,
                              max_seq_len: manualRunConfig.seq_len,
                              data_mode: manualRunConfig.data_source,
                              preflight_override: true,
                              enforce_preflight: true,
                            };
                            if (manualRunConfig.tokenizer && manualRunConfig.tokenizer !== 'byte') {
                              body.tokenizer_mode = manualRunConfig.tokenizer;
                            }
                            if (manualRunConfig.data_source === 'huggingface') {
                              body.hf_dataset = manualRunConfig.hf_dataset;
                              body.hf_subset = manualRunConfig.hf_subset;
                            }
                            const res = await apiCall(`/api/experiments/start`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify(body),
                            });
                            if (!res.ok) {
                              const err = await res.json();
                              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start manual run' } });
                            } else {
                              dispatch({ type: 'SET_MODAL', payload: { manualRunOpen: false } });
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                          }
                          dispatch({ type: 'SET_MODAL', payload: { manualRunStarting: false } });
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {manualRunStarting ? 'Starting...' : 'Launch Manual Run'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunOpen: false } })}
                        style={{ padding: '6px 12px', fontSize: 12 }}
                      >
                        Cancel
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                      {manualRunConfig.n_training_programs} program(s), {manualRunConfig.steps} steps, batch={manualRunConfig.batch_size}, seq={manualRunConfig.seq_len}, data={manualRunConfig.data_source}, tok={manualRunConfig.tokenizer}
                      {manualRunConfig.data_source === 'huggingface' && manualRunConfig.hf_dataset && ` (${manualRunConfig.hf_dataset})`}
                    </div>
                  </div>
                )}
              </div>
            )}
            {program.stage1_passed && (() => {
              const metrics = [
                { key: 'novelty_score', label: 'Novelty' },
                { key: 'fp_jacobian_spectral_norm', label: 'Spectral Norm' },
                { key: 'fp_interaction_locality', label: 'Locality' },
                { key: 'fp_interaction_sparsity', label: 'Sparsity' },
                { key: 'fp_isotropy', label: 'Isotropy' },
                { key: 'fp_rank_ratio', label: 'Rank Ratio' },
                { key: 'fp_sensitivity_uniformity', label: 'Sensitivity' },
              ];
              const missing = metrics.filter(m => program[m.key] == null);
              const lbMissing = leaderboardEntry ? [
                { key: 'robustness_noise_score', label: 'Noise Robustness' },
                { key: 'quant_int8_retention', label: 'INT8 Quantization' },
                { key: 'init_sensitivity_std', label: 'Init Sensitivity' },
                { key: 'param_efficiency', label: 'Param Efficiency' },
              ].filter(m => leaderboardEntry[m.key] == null) : [];
              const allMissing = [...missing, ...lbMissing];
              if (allMissing.length === 0 && !backfillResult) return null;
              return (
                <div style={{
                  padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                  border: '1px solid var(--border)',
                }}>
                  {allMissing.length > 0 && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                      Missing: {allMissing.map(m => m.label).join(', ')}
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button
                      className="start-btn"
                      disabled={backfillRunning}
                      onClick={async () => {
                        dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: true, backfillResult: null } });
                        try {
                          dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                          const res = await apiCall(`/api/programs/${resultId}/backfill-metrics`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ device: 'cpu' }),
                          });
                          if (!res.ok) {
                            const err = await res.json();
                            dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Backfill failed' } });
                            dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
                          } else {
                            const data = await res.json();
                            dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: data.backfill || { status: 'ok' } } });
                          }
                        } catch (e) {
                          dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                          dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
                        }
                      }}
                      style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.4)', color: '#a78bfa' }}
                    >
                      {backfillRunning ? 'Computing...' : 'Recompute Missing Metrics'}
                    </button>
                    {backfillResult && backfillResult.status === 'ok' && (
                      <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>Done — reload to see updates</span>
                    )}
                    {backfillResult && backfillResult.status === 'error' && (
                      <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>Failed</span>
                    )}
                  </div>
                  {(program.discovery_loss_ratio == null || program.validation_loss_ratio == null) && (
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                        Missing loss:{' '}
                        {[
                          program.discovery_loss_ratio == null && 'Discovery',
                          program.validation_loss_ratio == null && 'Validation',
                        ].filter(Boolean).join(', ')}
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <button
                          className="start-btn"
                          disabled={lossBackfillRunning}
                          onClick={async () => {
                            dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: true, lossBackfillResult: null } });
                            try {
                              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                              const res = await apiCall(`/api/programs/${resultId}/backfill-loss`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ device: 'cpu' }),
                              });
                              if (!res.ok) {
                                const err = await res.json();
                                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Loss backfill failed' } });
                                dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: false, lossBackfillResult: { status: 'error' } } });
                              } else {
                                const data = await res.json();
                                dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: false, lossBackfillResult: data.updates || { status: 'ok' } } });
                              }
                            } catch (e) {
                              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                              dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: false, lossBackfillResult: { status: 'error' } } });
                            }
                          }}
                          style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.4)', color: '#a78bfa' }}
                        >
                          {lossBackfillRunning ? 'Evaluating...' : 'Compute Discovery & Validation Loss'}
                        </button>
                        {lossBackfillResult && !lossBackfillResult.status && (
                          <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>
                            {lossBackfillResult.discovery_loss_ratio != null && `D.LR: ${Number(lossBackfillResult.discovery_loss_ratio).toFixed(4)}`}
                            {lossBackfillResult.discovery_loss_ratio != null && lossBackfillResult.validation_loss_ratio != null && ' | '}
                            {lossBackfillResult.validation_loss_ratio != null && `V.LR: ${Number(lossBackfillResult.validation_loss_ratio).toFixed(4)}`}
                          </span>
                        )}
                        {lossBackfillResult && lossBackfillResult.status === 'error' && (
                          <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>Failed</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
            {program.stage1_passed && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={overrideIneligible}
                    onChange={(e) => dispatch({ type: 'SET_UI', payload: { overrideIneligible: Boolean(e.target.checked) } })}
                  />
                  Override ineligible guardrails
                </label>
                <button
                  className="start-btn"
                  disabled={investigateDisabled}
                  onClick={async () => {
                    dispatch({ type: 'SET_ACTION', payload: { starting: 'investigate', error: null } });
                    try {
                      const res = await apiCall(`/api/experiments/start`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          mode: 'investigation',
                          result_ids: [resultId],
                          force: overrideIneligible,
                          override_ineligible: overrideIneligible,
                          preflight_override: true,
                          enforce_preflight: true,
                        }),
                      });
                      if (!res.ok) {
                        const err = await res.json();
                        dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start investigation' } });
                      } else {
                        dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                        if (onActionComplete) onActionComplete();
                        onClose();
                      }
                    } catch (e) {
                      dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                    }
                  }}
                  style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(63, 185, 80, 0.15)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
                  title={investigateTitle}
                >
                  {actionStarting === 'investigate' ? 'Starting...' : 'Investigate'}
                </button>
                {alreadyInvestigated && !overrideIneligible && (
                  <span style={{
                    fontSize: 11,
                    padding: '4px 8px',
                    borderRadius: 4,
                    background: 'rgba(210,153,34,0.12)',
                    color: 'var(--accent-yellow)',
                  }} title="Candidate already has investigation evidence; wait for changed conditions before re-investigating">
                    Already investigated
                  </span>
                )}
                <button
                  className="start-btn"
                  disabled={validateDisabled}
                  onClick={async () => {
                    dispatch({ type: 'SET_ACTION', payload: { starting: 'validate', error: null } });
                    try {
                      const res = await apiCall(`/api/experiments/start`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          mode: 'validation',
                          result_ids: [resultId],
                          force: overrideIneligible,
                          override_ineligible: overrideIneligible,
                          preflight_override: true,
                          enforce_preflight: true,
                        }),
                      });
                      if (!res.ok) {
                        const err = await res.json();
                        dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start validation' } });
                      } else {
                        dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                        if (onActionComplete) onActionComplete();
                        onClose();
                      }
                    } catch (e) {
                      dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                    }
                  }}
                  style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(188, 140, 255, 0.15)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                  title={validateTitle}
                >
                  {actionStarting === 'validate' ? 'Starting...' : 'Validate'}
                </button>
                {alreadyValidated && !overrideIneligible && (
                  <span style={{
                    fontSize: 11,
                    padding: '4px 8px',
                    borderRadius: 4,
                    background: 'rgba(88,166,255,0.12)',
                    color: 'var(--accent-blue)',
                  }} title="Candidate already has validation evidence. Enable override to rerun validation.">
                    Already validated
                  </span>
                )}
                {actionError && (
                  <span style={{ fontSize: 11, color: 'var(--accent-red)', alignSelf: 'center' }}>
                    {actionError}
                  </span>
                )}
              </div>
            )}
            {leaderboardEntry?.investigation_best_training && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Best Training Program (from Investigation)
                </div>
                <pre style={{
                  fontSize: 11, padding: 8, background: 'var(--bg-tertiary)',
                  borderRadius: 4, overflow: 'auto', maxHeight: 120,
                  color: 'var(--text-secondary)',
                }}>
                  {typeof leaderboardEntry.investigation_best_training === 'string'
                    ? leaderboardEntry.investigation_best_training
                    : JSON.stringify(leaderboardEntry.investigation_best_training, null, 2)}
                </pre>
              </div>
            )}
            <TokenMixingTaxonomy graphJson={program.graph_json_parsed} />
            <GatingDiagnostics program={program} />
            <SparsityDiagnostics program={program} />
            {program.graph_json_parsed && onOpenInDesigner && (
              <div style={{ marginTop: 8 }}>
                <button
                  onClick={() => {
                    onClose();
                    onOpenInDesigner(resultId);
                  }}
                  style={{
                    background: 'rgba(188, 140, 255, 0.15)',
                    border: '1px solid rgba(188, 140, 255, 0.4)',
                    color: 'var(--accent-purple)',
                    fontSize: 12,
                    fontWeight: 600,
                    padding: '8px 20px',
                    borderRadius: 6,
                    cursor: 'pointer',
                    width: '100%',
                  }}
                  title="Open this architecture in the visual graph designer"
                >
                  Open in Designer
                </button>
              </div>
            )}
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

export default ProgramDetail;
