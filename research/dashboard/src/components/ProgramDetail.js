import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useRef } from 'react';
import { lossColor, noveltyColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import apiService from '../services/apiService';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../utils/chartScales';

import StagePipeline from './program/StagePipeline';
import FingerprintRadar from './program/FingerprintRadar';
import RoutingHeatmap from './program/RoutingHeatmap';
import SparsityDiagnostics from './program/SparsityDiagnostics';
import TrainingCurve from './program/TrainingCurve';
import RobustnessProfile from './program/RobustnessProfile';
import AriaAdvice from './program/AriaAdvice';
import ReferenceComparison from './program/ReferenceComparison';
import MetricRow from './program/MetricRow';
import HypothesisInfo from './program/HypothesisInfo';
import BenchmarkEvidenceSnapshot from './program/BenchmarkEvidenceSnapshot';
import ExternalBenchmarkCard from './program/ExternalBenchmarkCard';
import EvidenceFlagChips from './program/EvidenceFlagChips';
import HypothesisLineage from './program/HypothesisLineage';
import OutcomesByPhase from './program/OutcomesByPhase';
import TierBadge from './program/TierBadge';
import FailureContext from './program/FailureContext';
import RecommendationCard from './program/RecommendationCard';
import TokenMixingTaxonomy from './program/TokenMixingTaxonomy';
import GatingDiagnostics from './program/GatingDiagnostics';
import RefinementRationale from './program/RefinementRationale';
import RefinementLineage from './program/RefinementLineage';
import RefinementAdvisor from './program/RefinementAdvisor';

/**
 * ProgramDetail — Modal showing computation graph, stage pipeline,
 * fingerprint radar chart, training metrics, similar architectures,
 * sandbox metrics, FLOPs, baseline comparison, training curve.
 */
// Compatibility alias: some cached bundles still reference RadarChart.
const RadarChart = FingerprintRadar;

function ProgramDetail({ resultId, onClose, onActionComplete, onSelectExperiment, onViewInLeaderboard, onSelectCampaign, onOpenInDesigner, onAddToComparison, eligibilityByResultId, defaultOverrideIneligible = false }) {
  const [program, setProgram] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scaleUpOpen, setScaleUpOpen] = useState(false);
  const [scaleUpConfig, setScaleUpConfig] = useState({ steps: 5000, batch_size: 8, seq_len: 512 });
  const [scaleUpStarting, setScaleUpStarting] = useState(false);
  const [manualRunOpen, setManualRunOpen] = useState(false);
  const [manualRunStarting, setManualRunStarting] = useState(false);
  const [manualRunConfig, setManualRunConfig] = useState({
    steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256,
    data_source: 'corpus',
    hf_dataset: 'roneneldan/TinyStories', hf_subset: '',
  });
  const [backfillRunning, setBackfillRunning] = useState(false);
  const [backfillResult, setBackfillResult] = useState(null);
  const [lossBackfillRunning, setLossBackfillRunning] = useState(false);
  const [lossBackfillResult, setLossBackfillResult] = useState(null);
  const [leaderboardEntry, setLeaderboardEntry] = useState(null);
  const [actionStarting, setActionStarting] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [overrideIneligible, setOverrideIneligible] = useState(Boolean(defaultOverrideIneligible));
  const [linkedHypothesis, setLinkedHypothesis] = useState(null);
  const [linkedDecision, setLinkedDecision] = useState(null);
  const [linkedExperiment, setLinkedExperiment] = useState(null);
  const [linkedCampaign, setLinkedCampaign] = useState(null);
  const [provenanceOpen, setProvenanceOpen] = useState(true);
  const [decisionPacket, setDecisionPacket] = useState(null);
  const [decisionPacketLoading, setDecisionPacketLoading] = useState(false);
  const [decisionPacketError, setDecisionPacketError] = useState(null);
  const [decisionPacketOpen, setDecisionPacketOpen] = useState(true);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [manifestCopied, copyManifest] = useCopyToClipboard();
  const [latestRefineLaunch, setLatestRefineLaunch] = useState(null);
  const [refineLaunchHistory, setRefineLaunchHistory] = useState([]);
  const [refineTrace, setRefineTrace] = useState(null);
  const [refineTraceLoading, setRefineTraceLoading] = useState(false);
  const [refineAnalysis, setRefineAnalysis] = useState(null);
  const [refineAnalysisLoading, setRefineAnalysisLoading] = useState(false);
  const [refineAnalysisError, setRefineAnalysisError] = useState(null);
  const [drawerWidthVw, setDrawerWidthVw] = useState(45);
  const [drawerMaximized, setDrawerMaximized] = useState(false);
  const [resizingDrawer, setResizingDrawer] = useState(false);
  const drawerResizeRef = useRef({ startX: 0, startVw: 45 });

  const fetchAndCopyManifest = () => {
    if (!resultId) return;
    setManifestLoading(true);
    apiService.getReproducibilityManifest(resultId)
      .then(d => {
        copyManifest(JSON.stringify(d, null, 2));
        setManifestLoading(false);
      })
      .catch(() => { setManifestLoading(false); });
  };

  const formatUnixTimestamp = (value) => {
    if (value == null) return null;
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    const ms = n > 1e12 ? n : n * 1000;
    return new Date(ms).toLocaleString();
  };

  useEffect(() => {
    setOverrideIneligible(Boolean(defaultOverrideIneligible));
  }, [defaultOverrideIneligible, resultId]);

  const fetchDecisionPacket = () => {
    if (!resultId) return;
    setDecisionPacketLoading(true);
    setDecisionPacketError(null);
    apiService.getDecisionPacket(resultId)
      .then(d => { setDecisionPacket(d); setDecisionPacketLoading(false); })
      .catch(e => { setDecisionPacketError('Failed: ' + e.message); setDecisionPacketLoading(false); });
  };

  useEffect(() => {
    if (!resultId) return;
    setLoading(true);
    setError(null);
    setLatestRefineLaunch(null);
    setRefineLaunchHistory([]);
    setRefineTrace(null);
    setRefineTraceLoading(false);
    setLinkedHypothesis(null);
    setLinkedDecision(null);
    setLinkedExperiment(null);
    setLinkedCampaign(null);
    
    apiService.getProgram(resultId)
      .then(d => {
        setProgram(d);
        setLoading(false);
        // Fetch linked hypothesis via experiment
        if (d?.experiment_id) {
          apiService.getExperiment(d.experiment_id)
            .then(expData => {
              if (expData?.experiment) {
                setLinkedExperiment(expData.experiment);
                if (expData.experiment.campaign_id) {
                  setLinkedCampaign({ campaign_id: expData.experiment.campaign_id, title: expData.experiment.campaign_title || expData.experiment.campaign_id });
                  // Find hypothesis linked to this experiment
                  apiService.getCampaignHypotheses(expData.experiment.campaign_id)
                    .then(hyps => {
                      const linked = (Array.isArray(hyps) ? hyps : []).find(
                        h => h.experiment_id === d.experiment_id
                      );
                      if (linked) setLinkedHypothesis(linked);
                    })
                    .catch(() => {});
                  // Find decisions mentioning this result
                  apiService.getCampaignDecisions(expData.experiment.campaign_id)
                    .then(decs => {
                      const linked = (Array.isArray(decs) ? decs : []).find(d => {
                        const evidenceIds = d.evidence_ids || [];
                        return Array.isArray(evidenceIds) && evidenceIds.includes(resultId);
                      });
                      if (linked) setLinkedDecision(linked);
                    })
                    .catch(() => {});
                }
              }
            })
            .catch(() => {});
        }
      })
      .catch(e => { setError('Failed to load program: ' + e.message); setLoading(false); });
    // Fetch leaderboard entry for this result
    apiService.getLeaderboard('?limit=200')
      .then(data => {
        if (data?.entries) {
          const entry = data.entries.find(e => e.result_id === resultId);
          setLeaderboardEntry(entry || null);
        }
      })
      .catch(() => {});
  }, [resultId]);

  // Auto-fetch refinement analysis for S1 survivors
  useEffect(() => {
    if (!resultId || !program?.stage1_passed) return;
    setRefineAnalysisLoading(true);
    setRefineAnalysisError(null);
    apiCall(`/api/programs/${encodeURIComponent(resultId)}/refine-analysis`)
      .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error || 'Failed'))))
      .then(data => { setRefineAnalysis(data); setRefineAnalysisLoading(false); })
      .catch(e => { setRefineAnalysisError(e.message); setRefineAnalysisLoading(false); });
  }, [resultId, program?.stage1_passed]);

  useEffect(() => {
    if (!latestRefineLaunch?.experimentId || !resultId) return;

    let cancelled = false;
    let intervalId = null;

    const summarizeTrace = (payload) => {
      const experiment = payload?.experiment || {};
      const programs = Array.isArray(payload?.programs) ? payload.programs : [];

      const withRefinementMeta = programs.map(row => {
        let refinement = null;
        try {
          const raw = row?.graph_json;
          if (raw && typeof raw === 'string') {
            const parsed = JSON.parse(raw);
            refinement = parsed?.metadata?.refinement || null;
          }
        } catch (_) {
          refinement = null;
        }
        return { ...row, _refinement: refinement };
      });

      const lineage = withRefinementMeta.filter(
        row => String(row?._refinement?.source_result_id || '') === String(resultId),
      );
      const scoped = lineage.length > 0 ? lineage : withRefinementMeta;

      const finiteLosses = scoped
        .map(row => Number(row?.loss_ratio))
        .filter(value => Number.isFinite(value));
      const bestLoss = finiteLosses.length > 0 ? Math.min(...finiteLosses) : null;
      const stage1Survivors = scoped.filter(row => Boolean(row?.stage1_passed)).length;

      const uniqueFingerprints = [];
      const uniqueResultIds = [];
      const newCandidates = [];
      for (const row of scoped) {
        const fp = String(row?.graph_fingerprint || '').trim();
        const rid = String(row?.result_id || '').trim();
        if (fp && fp !== String(program?.graph_fingerprint || '') && !uniqueFingerprints.includes(fp)) {
          uniqueFingerprints.push(fp);
        }
        if (rid && rid !== String(resultId) && !uniqueResultIds.includes(rid)) {
          uniqueResultIds.push(rid);
        }
        if (rid && fp && rid !== String(resultId) && !newCandidates.some(c => c.resultId === rid)) {
          newCandidates.push({ resultId: rid, fingerprint: fp });
        }
      }

      const status = String(experiment?.status || '').toLowerCase();
      const completed = Boolean(experiment?.completed_at) || status === 'completed' || status === 'failed' || status === 'cancelled';

      return {
        status: status || 'running',
        completed,
        experiment,
        totals: {
          programs: programs.length,
          scopedPrograms: scoped.length,
          stage1Survivors,
          bestLoss,
        },
        newFingerprints: uniqueFingerprints.slice(0, 6),
        newResultIds: uniqueResultIds.slice(0, 6),
        newCandidates: newCandidates.slice(0, 6),
      };
    };

    const pollTrace = async () => {
      if (cancelled) return;
      setRefineTraceLoading(true);
      try {
        const response = await apiCall(`/api/experiments/${latestRefineLaunch.experimentId}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        if (cancelled) return;
        const tracePayload = summarizeTrace(payload);
        setRefineTrace(tracePayload);
        setRefineLaunchHistory(prev => prev.map(item => (
          item.experimentId === latestRefineLaunch.experimentId
            ? {
                ...item,
                status: tracePayload.status,
                topCandidate: tracePayload.newCandidates?.[0] || null,
              }
            : item
        )));
        if (tracePayload.completed && intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      } catch (e) {
        if (!cancelled) {
          setRefineTrace({ error: e?.message || 'Failed to load refinement trace' });
        }
      } finally {
        if (!cancelled) {
          setRefineTraceLoading(false);
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
      setDrawerWidthVw(Math.max(35, Math.min(90, nextVw)));
    };
    const onMouseUp = () => {
      setResizingDrawer(false);
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
  const shortId = (v, n = 12) => {
    const s = String(v || '').trim();
    if (!s) return '--';
    return s.length > n ? s.slice(0, n) : s;
  };

  const entryTier = typeof leaderboardEntry?.tier === 'string'
    ? leaderboardEntry.tier
    : (typeof program?.tier === 'string' ? program.tier : '');
  const tier = String(entryTier || '').toLowerCase();
  const hasInvestigationEvidence = (leaderboardEntry?.investigation_loss_ratio ?? program?.investigation_loss_ratio) != null;
  const fallbackEligibility = {
    investigationEligible: Boolean(program?.stage1_passed) && (tier === 'screening' && !hasInvestigationEvidence),
    validationEligible: tier === 'investigation' && Boolean(leaderboardEntry?.investigation_passed ?? program?.investigation_passed),
  };
  const resolvedEligibility = eligibilityByResultId?.[resultId] || fallbackEligibility;
  const lastRefinedCandidate =
    refineTrace?.newCandidates?.[0]
    || refineLaunchHistory.find(item => item?.topCandidate)?.topCandidate
    || null;

  const handleLaunchRefinement = async (intent, actionKey, failureLabel) => {
    setActionStarting(actionKey);
    try {
      setActionError(null);
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
          ...(refineAnalysis ? { refine_analysis_json: refineAnalysis } : {}),
        }),
      });

      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        setActionError(payload.error || failureLabel);
      } else {
        const resolved = payload?.refine_resolution || {};
        setLatestRefineLaunch({
          experimentId: payload?.experiment_id,
          intent,
          startedAt: Date.now(),
          sourceResultId: resultId,
          sourceFingerprint: program?.graph_fingerprint,
          resolvedResultIds: Array.isArray(resolved?.result_ids) ? resolved.result_ids : [],
          resolvedFingerprints: Array.isArray(resolved?.resolved_fingerprints) ? resolved.resolved_fingerprints : [],
          unresolvedFingerprints: Array.isArray(resolved?.unresolved_fingerprints) ? resolved.unresolved_fingerprints : [],
        });
        setRefineLaunchHistory(prev => {
          const nextItem = {
            experimentId: payload?.experiment_id,
            intent,
            startedAt: Date.now(),
            sourceResultId: resultId,
            sourceFingerprint: program?.graph_fingerprint,
            status: 'running',
            topCandidate: null,
          };
          const deduped = prev.filter(item => item.experimentId !== nextItem.experimentId);
          return [nextItem, ...deduped].slice(0, 3);
        });
        if (onActionComplete) onActionComplete();
      }
    } catch (e) {
      setActionError('Error: ' + e.message);
    }
    setActionStarting(null);
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
              setResizingDrawer(true);
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
              onClick={() => setDrawerMaximized(v => !v)}
              style={{ fontSize: 12, padding: '5px 10px' }}
              title={drawerMaximized ? 'Restore panel size' : 'Maximize panel'}
            >
              {drawerMaximized ? 'Restore' : 'Maximize'}
            </button>
            <button className="refresh-btn" onClick={onClose} style={{ fontSize: 18, lineHeight: 1, padding: '4px 8px' }}>&times;</button>
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
            {/* Header info */}
            <div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--accent-blue)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>{program.graph_fingerprint}</span>
                {program.stage_at_death && program.stage_at_death !== 'survived' && (
                  <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>
                    died at {program.stage_at_death}
                  </span>
                )}
                {leaderboardEntry && (
                  <>
                    <TierBadge tier={leaderboardEntry.tier} entry={leaderboardEntry} />
                    <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-green)' }}>
                      Score: {Number(leaderboardEntry.composite_score).toFixed(3)}
                    </span>
                  </>
                )}
              </div>
              <StagePipeline program={program} />
            </div>

            {/* Provenance & Context */}
            {(program.experiment_id || linkedHypothesis || leaderboardEntry || linkedCampaign) && (
              <div style={{
                background: 'var(--bg-tertiary)',
                borderRadius: 6,
                border: '1px solid var(--border)',
                overflow: 'hidden',
              }}>
                <div
                  onClick={() => setProvenanceOpen(!provenanceOpen)}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    userSelect: 'none',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                    Provenance & Context
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {provenanceOpen ? '▾ collapse' : '▸ expand'}
                  </span>
                </div>
                {provenanceOpen && (
                  <div style={{ padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {/* Source experiment */}
                    {program.experiment_id && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Experiment:</span>
                        <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{program.experiment_id.slice(0, 12)}</span>
                        {linkedExperiment?.started_at && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            {formatUnixTimestamp(linkedExperiment.started_at)}
                          </span>
                        )}
                        {linkedExperiment?.experiment_type && (
                          <span style={{ fontSize: 10, color: 'var(--accent-blue)', border: '1px solid var(--accent-blue)', borderRadius: 3, padding: '0 4px' }}>
                            {linkedExperiment.experiment_type}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Hypothesis 1-liner */}
                    {linkedHypothesis && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90, flexShrink: 0 }}>Hypothesis:</span>
                        <span style={{
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 400,
                          color: linkedHypothesis.status === 'confirmed' ? 'var(--accent-green)' :
                                 linkedHypothesis.status === 'refuted' ? 'var(--accent-red)' : 'var(--text-secondary)',
                        }} title={linkedHypothesis.prediction}>
                          [{linkedHypothesis.status}] {linkedHypothesis.prediction}
                        </span>
                      </div>
                    )}

                    {/* Campaign */}
                    {linkedCampaign && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Campaign:</span>
                        <span>{linkedCampaign.title || linkedCampaign.campaign_id}</span>
                      </div>
                    )}

                    {/* Leaderboard status */}
                    {leaderboardEntry && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Leaderboard:</span>
                        <TierBadge tier={leaderboardEntry.tier} entry={leaderboardEntry} />
                        <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                          {Number(leaderboardEntry.composite_score).toFixed(3)}
                        </span>
                        {leaderboardEntry.tier === 'screening' && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            needs investigation to advance
                          </span>
                        )}
                        {leaderboardEntry.tier === 'investigation' && !leaderboardEntry.investigation_passed && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            investigation completed (below threshold)
                          </span>
                        )}
                        {leaderboardEntry.tier === 'investigation' && leaderboardEntry.investigation_passed && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            ready for validation
                          </span>
                        )}
                      </div>
                    )}

                    {/* Quick nav links */}
                    <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
                      {program.experiment_id && onSelectExperiment && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onSelectExperiment(program.experiment_id); }}
                        >
                          Open Experiment
                        </button>
                      )}
                      {onAddToComparison && (
                        <button
                          className="refresh-btn"
                          title="Add to side-by-side comparison"
                          onClick={() => onAddToComparison(resultId)}
                        >
                          Compare
                        </button>
                      )}
                      {onOpenInDesigner && (
                        <button
                          className="refresh-btn"
                          title="Open this architecture in Aria Designer"
                          onClick={() => onOpenInDesigner(resultId)}
                        >
                          Open in Designer
                        </button>
                      )}
                      {leaderboardEntry && onViewInLeaderboard && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onViewInLeaderboard(resultId); }}
                        >
                          View in Leaderboard
                        </button>
                      )}
                      {linkedCampaign && onSelectCampaign && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onSelectCampaign(linkedCampaign.campaign_id); }}
                        >
                          Open Campaign
                        </button>
                      )}
                      <button
                        className="refresh-btn"
                        style={{
                          fontSize: 11, padding: '3px 10px',
                          background: decisionPacket ? 'rgba(188, 140, 255, 0.15)' : undefined,
                          borderColor: 'var(--accent-purple)',
                          color: 'var(--accent-purple)',
                        }}
                        disabled={decisionPacketLoading}
                        onClick={fetchDecisionPacket}
                      >
                        {decisionPacketLoading ? 'Loading...' : 'Decision Packet'}
                      </button>
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 11, padding: '3px 10px' }}
                        disabled={manifestLoading}
                        onClick={fetchAndCopyManifest}
                      >
                        {manifestLoading ? 'Loading...' : manifestCopied ? 'Copied!' : 'Copy Manifest'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Decision Packet */}
            {decisionPacketError && (
              <div style={{ padding: 8, background: 'rgba(248, 81, 73, 0.1)', border: '1px solid var(--accent-red)', borderRadius: 4, fontSize: 12, color: 'var(--accent-red)' }}>
                {decisionPacketError}
              </div>
            )}
            {decisionPacket && (
              <div style={{
                background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--accent-purple)', overflow: 'hidden',
              }}>
                <div
                  onClick={() => setDecisionPacketOpen(!decisionPacketOpen)}
                  style={{
                    padding: '8px 12px', cursor: 'pointer',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    userSelect: 'none', background: 'rgba(188, 140, 255, 0.08)',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
                    Decision Packet
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {decisionPacketOpen ? '\u25BE collapse' : '\u25B8 expand'}
                  </span>
                </div>
                {decisionPacketOpen && (
                  <div style={{ padding: '8px 12px 12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {/* Evidence Flags */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Evidence Flags</div>
                      <EvidenceFlagChips flags={decisionPacket.evidence_flags} />
                    </div>
                    {/* Hypothesis Lineage */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Hypothesis Lineage</div>
                      <HypothesisLineage chain={decisionPacket.hypothesis_chain} />
                    </div>
                    {/* Outcomes by Phase */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Outcomes by Phase</div>
                      <OutcomesByPhase outcomes={decisionPacket.outcomes} />
                    </div>
                    {/* Failure Context */}
                    {(decisionPacket.failure_context?.stage_at_death || decisionPacket.failure_context?.error_type) && (
                      <div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Failure Context</div>
                        <FailureContext context={decisionPacket.failure_context} />
                      </div>
                    )}
                    {/* Recommendation */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Recommendation</div>
                      <RecommendationCard recommendation={decisionPacket.recommendation} />
                    </div>
                  </div>
                )}
              </div>
            )}

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
            {latestRefineLaunch && (
              <div style={{
                padding: 10,
                background: 'var(--bg-tertiary)',
                borderRadius: 6,
                border: '1px solid var(--border)',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}>
                <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', color: 'var(--accent-purple)' }}>
                  Refinement Trace
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'grid', gap: 4 }}>
                  <div><strong>Intent:</strong> {latestRefineLaunch.intent}</div>
                  <div><strong>Experiment:</strong> <span style={{ fontFamily: 'monospace' }}>{shortId(latestRefineLaunch.experimentId, 16)}</span></div>
                  <div><strong>Source:</strong> <span style={{ fontFamily: 'monospace' }}>{shortId(latestRefineLaunch.sourceResultId, 12)}</span> · {shortId(latestRefineLaunch.sourceFingerprint, 18)}</div>
                  <div><strong>Resolved IDs:</strong> {latestRefineLaunch.resolvedResultIds.length > 0 ? latestRefineLaunch.resolvedResultIds.map(v => shortId(v, 10)).join(', ') : 'none'}</div>
                  {latestRefineLaunch.unresolvedFingerprints.length > 0 && (
                    <div><strong>Unresolved fingerprints:</strong> {latestRefineLaunch.unresolvedFingerprints.join(', ')}</div>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {latestRefineLaunch.experimentId && onSelectExperiment && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 10px' }}
                      onClick={() => { onClose(); onSelectExperiment(latestRefineLaunch.experimentId); }}
                    >
                      Open Refinement Run
                    </button>
                  )}
                  {refineTrace?.newResultIds?.[0] && onViewInLeaderboard && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 10px' }}
                      onClick={() => { onClose(); onViewInLeaderboard(refineTrace.newResultIds[0]); }}
                    >
                      View Top Refined Result
                    </button>
                  )}
                </div>
                {refineTraceLoading && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Collecting live refinement outcomes…</div>
                )}
                {refineTrace?.error && (
                  <div style={{ fontSize: 11, color: 'var(--accent-red)' }}>{refineTrace.error}</div>
                )}
                {refineTrace && !refineTrace.error && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    <div><strong>Status:</strong> {refineTrace.status}</div>
                    <div>
                      <strong>Outcomes:</strong> {fmtInt(refineTrace.totals?.programs)} programs,
                      {' '}{fmtInt(refineTrace.totals?.stage1Survivors)} S1 survivors,
                      {' '}best loss {fmt(refineTrace.totals?.bestLoss)}
                    </div>
                    {refineTrace.newFingerprints?.length > 0 && (
                      <div>
                        <strong>New Fingerprints:</strong> {refineTrace.newFingerprints.map(fp => shortId(fp, 18)).join(', ')}
                      </div>
                    )}
                    {refineTrace.newCandidates?.length > 0 && (
                      <div>
                        <strong>Open Fingerprint:</strong>{' '}
                        {onViewInLeaderboard ? (
                          <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                            {refineTrace.newCandidates.map(candidate => (
                              <button
                                key={candidate.resultId}
                                className="refresh-btn"
                                style={{ fontSize: 10, padding: '2px 8px', fontFamily: 'monospace' }}
                                onClick={() => { onClose(); onViewInLeaderboard(candidate.resultId); }}
                                title={`Open ${candidate.fingerprint}`}
                              >
                                {shortId(candidate.fingerprint, 18)}
                              </button>
                            ))}
                          </span>
                        ) : (
                          refineTrace.newCandidates.map(candidate => shortId(candidate.fingerprint, 18)).join(', ')
                        )}
                      </div>
                    )}
                    {refineTrace.newResultIds?.length > 0 && (
                      <div>
                        <strong>New Result IDs:</strong> {refineTrace.newResultIds.map(rid => shortId(rid, 10)).join(', ')}
                      </div>
                    )}
                  </div>
                )}
                {refineLaunchHistory.length > 0 && (
                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                      Recent Refinement Launches
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {refineLaunchHistory.map(item => (
                        <div key={item.experimentId} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', fontSize: 11 }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{item.intent}</span>
                          <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>{shortId(item.experimentId, 12)}</span>
                          <span style={{ color: 'var(--text-muted)' }}>{item.status || 'running'}</span>
                          {onSelectExperiment && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 10, padding: '2px 8px' }}
                              onClick={() => { onClose(); onSelectExperiment(item.experimentId); }}
                            >
                              Open Run
                            </button>
                          )}
                          {item.topCandidate?.resultId && onViewInLeaderboard && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 10, padding: '2px 8px', fontFamily: 'monospace' }}
                              onClick={() => { onClose(); onViewInLeaderboard(item.topCandidate.resultId); }}
                              title={`Open ${item.topCandidate.fingerprint}`}
                            >
                              Open Fingerprint
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Metrics + Radar side by side */}
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
                  
                  {/* Similar To Metric */}
                  <div style={{ width: '100%', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                      Primary Reference Class
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--accent-purple)', fontWeight: 600 }}>
                      {program.most_similar_to || 'Truly Novel (No match)'}
                    </div>
                  </div>

                  {/* CKA Similarity bars moved here */}
                  {(program.fp_cka_vs_transformer != null || program.fp_cka_vs_ssm != null) && (
                    <div style={{ width: '100%', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
                        CKA Distance vs Baselines
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

            {/* Training metrics */}
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

            {/* Training Curve */}
            {program.has_training_curve && (
              <TrainingCurve resultId={resultId} />
            )}

            {/* LLM Explanation */}
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

            {/* Linked Hypothesis */}
            {linkedHypothesis && (
              <HypothesisInfo hypothesis={linkedHypothesis} />
            )}

            {/* Linked Decision */}
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

            {/* Scale Up Button (only for S1 survivors) */}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!scaleUpOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => setScaleUpOpen(true)}
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
                          onChange={e => setScaleUpConfig(c => ({ ...c, steps: parseInt(e.target.value) || 5000 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="4" max="16" step="1"
                          value={scaleUpConfig.batch_size}
                          onChange={e => setScaleUpConfig(c => ({ ...c, batch_size: parseInt(e.target.value) || 8 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="256" max="1024" step="128"
                          value={scaleUpConfig.seq_len}
                          onChange={e => setScaleUpConfig(c => ({ ...c, seq_len: parseInt(e.target.value) || 512 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={scaleUpStarting}
                        onClick={async () => {
                          setScaleUpStarting(true);
                          try {
                            setActionError(null);
                            const res = await apiCall(`/api/experiments/start`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({
                                mode: 'scale_up',
                                result_ids: [resultId],
                                scale_up_steps: scaleUpConfig.steps,
                                scale_up_batch_size: scaleUpConfig.batch_size,
                                scale_up_seq_len: scaleUpConfig.seq_len,
                              }),
                            });
                            if (!res.ok) {
                              const err = await res.json();
                              setActionError(err.error || 'Failed to start scale-up');
                            } else {
                              setScaleUpOpen(false);
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            setActionError('Error: ' + e.message);
                          }
                          setScaleUpStarting(false);
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {scaleUpStarting ? 'Starting...' : 'Start Scale-Up'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => setScaleUpOpen(false)}
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

            {/* Manual Training Run (power-user override) */}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!manualRunOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => setManualRunOpen(true)}
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
                          onChange={e => setManualRunConfig(c => ({ ...c, steps: parseInt(e.target.value) || 2500 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="1" max="32" step="1"
                          value={manualRunConfig.batch_size}
                          onChange={e => setManualRunConfig(c => ({ ...c, batch_size: parseInt(e.target.value) || 4 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="64" max="2048" step="64"
                          value={manualRunConfig.seq_len}
                          onChange={e => setManualRunConfig(c => ({ ...c, seq_len: parseInt(e.target.value) || 256 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Training Programs</label>
                        <input type="number" min="1" max="10" step="1"
                          value={manualRunConfig.n_training_programs}
                          onChange={e => setManualRunConfig(c => ({ ...c, n_training_programs: parseInt(e.target.value) || 3 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Data Source</label>
                        <select
                          value={manualRunConfig.data_source}
                          onChange={e => setManualRunConfig(c => ({ ...c, data_source: e.target.value }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        >
                          <option value="corpus">Corpus</option>
                          <option value="random">Random</option>
                          <option value="huggingface">HuggingFace</option>
                        </select>
                      </div>
                      {manualRunConfig.data_source === 'huggingface' && (
                        <>
                          <div>
                            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Dataset</label>
                            <input type="text"
                              value={manualRunConfig.hf_dataset}
                              onChange={e => setManualRunConfig(c => ({ ...c, hf_dataset: e.target.value }))}
                              placeholder="roneneldan/TinyStories"
                              style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                            />
                          </div>
                          <div>
                            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Subset</label>
                            <input type="text"
                              value={manualRunConfig.hf_subset}
                              onChange={e => setManualRunConfig(c => ({ ...c, hf_subset: e.target.value }))}
                              placeholder="(optional)"
                              style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                            />
                          </div>
                        </>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => setManualRunConfig(c => ({ ...c, steps: 1000, batch_size: 4, n_training_programs: 1, seq_len: 256 }))}>
                        Quick
                      </button>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => setManualRunConfig(c => ({ ...c, steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256 }))}>
                        Standard
                      </button>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => setManualRunConfig(c => ({ ...c, steps: 5000, batch_size: 8, n_training_programs: 5, seq_len: 512 }))}>
                        Deep
                      </button>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={manualRunStarting}
                        onClick={async () => {
                          setManualRunStarting(true);
                          try {
                            setActionError(null);
                            const body = {
                              mode: 'investigation',
                              force: true,
                              result_ids: [resultId],
                              n_training_programs: manualRunConfig.n_training_programs,
                              investigation_steps: manualRunConfig.steps,
                              investigation_batch_size: manualRunConfig.batch_size,
                              max_seq_len: manualRunConfig.seq_len,
                              data_mode: manualRunConfig.data_source,
                            };
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
                              setActionError(err.error || 'Failed to start manual run');
                            } else {
                              setManualRunOpen(false);
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            setActionError('Error: ' + e.message);
                          }
                          setManualRunStarting(false);
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {manualRunStarting ? 'Starting...' : 'Launch Manual Run'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => setManualRunOpen(false)}
                        style={{ padding: '6px 12px', fontSize: 12 }}
                      >
                        Cancel
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                      {manualRunConfig.n_training_programs} program(s), {manualRunConfig.steps} steps, batch={manualRunConfig.batch_size}, seq={manualRunConfig.seq_len}, data={manualRunConfig.data_source}
                      {manualRunConfig.data_source === 'huggingface' && manualRunConfig.hf_dataset && ` (${manualRunConfig.hf_dataset})`}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Recompute Missing Metrics */}
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
                        setBackfillRunning(true);
                        setBackfillResult(null);
                        try {
                          setActionError(null);
                          const res = await apiCall(`/api/programs/${resultId}/backfill-metrics`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ device: 'cpu' }),
                          });
                          if (!res.ok) {
                            const err = await res.json();
                            setActionError(err.error || 'Backfill failed');
                            setBackfillResult({ status: 'error' });
                          } else {
                            const data = await res.json();
                            setBackfillResult(data.backfill || { status: 'ok' });
                          }
                        } catch (e) {
                          setActionError('Error: ' + e.message);
                          setBackfillResult({ status: 'error' });
                        }
                        setBackfillRunning(false);
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
                            setLossBackfillRunning(true);
                            setLossBackfillResult(null);
                            try {
                              setActionError(null);
                              const res = await apiCall(`/api/programs/${resultId}/backfill-loss`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ device: 'cpu' }),
                              });
                              if (!res.ok) {
                                const err = await res.json();
                                setActionError(err.error || 'Loss backfill failed');
                                setLossBackfillResult({ status: 'error' });
                              } else {
                                const data = await res.json();
                                setLossBackfillResult(data.updates || { status: 'ok' });
                              }
                            } catch (e) {
                              setActionError('Error: ' + e.message);
                              setLossBackfillResult({ status: 'error' });
                            }
                            setLossBackfillRunning(false);
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

            {/* Investigate / Validate actions */}
            {program.stage1_passed && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={overrideIneligible}
                    onChange={(e) => setOverrideIneligible(Boolean(e.target.checked))}
                  />
                  Override ineligible guardrails
                </label>
                {(resolvedEligibility.investigationEligible || overrideIneligible) && (
                  <button
                    className="start-btn"
                    disabled={actionStarting === 'investigate'}
                    onClick={async () => {
                      setActionStarting('investigate');
                      try {
                        setActionError(null);
                        const res = await apiCall(`/api/experiments/start`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({
                            mode: 'investigation',
                            result_ids: [resultId],
                            force: overrideIneligible,
                            override_ineligible: overrideIneligible,
                          }),
                        });
                        if (!res.ok) {
                          const err = await res.json();
                          setActionError(err.error || 'Failed to start investigation');
                        } else {
                          if (onActionComplete) onActionComplete();
                          onClose();
                        }
                      } catch (e) {
                        setActionError('Error: ' + e.message);
                      }
                      setActionStarting(null);
                    }}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(63, 185, 80, 0.15)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
                    title="Deep study with multiple training programs"
                  >
                    {actionStarting === 'investigate' ? 'Starting...' : 'Investigate'}
                  </button>
                )}
                {!resolvedEligibility.investigationEligible && (leaderboardEntry?.tier === 'screening') && (
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
                {(resolvedEligibility.validationEligible || overrideIneligible) && (
                  <button
                    className="start-btn"
                    disabled={actionStarting === 'validate'}
                    onClick={async () => {
                      setActionStarting('validate');
                      try {
                        setActionError(null);
                        const res = await apiCall(`/api/experiments/start`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({
                            mode: 'validation',
                            result_ids: [resultId],
                            force: overrideIneligible,
                            override_ineligible: overrideIneligible,
                          }),
                        });
                        if (!res.ok) {
                          const err = await res.json();
                          setActionError(err.error || 'Failed to start validation');
                        } else {
                          if (onActionComplete) onActionComplete();
                          onClose();
                        }
                      } catch (e) {
                        setActionError('Error: ' + e.message);
                      }
                      setActionStarting(null);
                    }}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(188, 140, 255, 0.15)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                    title="Publication-grade multi-seed validation"
                  >
                    {actionStarting === 'validate' ? 'Starting...' : 'Validate'}
                  </button>
                )}
                {actionError && (
                  <span style={{ fontSize: 11, color: 'var(--accent-red)', alignSelf: 'center' }}>
                    {actionError}
                  </span>
                )}
              </div>
            )}

            {/* Leaderboard training program details */}
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

            {/* Open in Designer action */}
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
