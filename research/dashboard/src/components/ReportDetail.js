import { apiCall } from "../services/apiService";
import React, { useEffect, useMemo, useState } from 'react';
import { promotionEvidenceView } from '../utils/backendScore';
import useInteractiveTable from './shared/useInteractiveTable';
import SortIndicator from './shared/SortIndicator';
import {
  reliabilityBand, wilsonInterval, decisionGate,
  reproducibilityPacketStatus,
} from './report/reportUtils';
import generateMarkdown from './report/generateMarkdown';
import StatCard from './report/StatCard';
import EfficiencyChart from './report/EfficiencyChart';
import DiscoveryRankings from './report/DiscoveryRankings';
import AlternativesToAttention from './report/AlternativesToAttention';
import FunctionalFamilyEvidence from './report/FunctionalFamilyEvidence';
import MathspaceOperatorImpact from './report/MathspaceOperatorImpact';
import RoutingModeComparison from './report/RoutingModeComparison';
import CompressionTechniqueCoverage from './report/CompressionTechniqueCoverage';
import NegativeResultsSummary from './report/NegativeResultsSummary';
import ConfidenceInfographic from './report/ConfidenceInfographic';
import MirroredOpsChart from './report/MirroredOpsChart';
import TemplatePerformance from './report/TemplatePerformance';


function ReportDetail({
  scope,
  onBack,
  onSelectProgram,
  onSelectExperiment,
  onInvestigate,
  onCapabilityRank,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  queuedResultIds,
  eligibilityByResultId,
  onHypothesisHandoff,
  onOpenInDesigner,
}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [startDate, setStartDate] = useState(scope?.params?.start_date || '');
  const [endDate, setEndDate] = useState(scope?.params?.end_date || '');
  const [theme, setTheme] = useState(scope?.params?.theme || 'all');
  const [trend, setTrend] = useState(scope?.params?.trend || 'all');
  const [queryLimit, setQueryLimit] = useState(20);
  const [declutterMode, setDeclutterMode] = useState(false);
  const [detailsReady, setDetailsReady] = useState(false);
  // stabilityTable hook is called below after stabilityCandidates is derived

  const isAllTime = !scope?.params;

  const fetchReport = async ({ fast = true } = {}) => {
    setError(null);
    if (fast) {
      setLoading(true);
    } else {
      setLoadingDetails(true);
    }
    try {
      const qs = new URLSearchParams({
        fast: fast ? '1' : '0',
        include_heavy: fast ? '0' : '1',
        include_narrative: '0',
      });
      const res = await apiCall(`/api/report?${qs.toString()}`);
      if (!res.ok && fast) {
        // Resilient fallback: keep Reports usable when consolidated fast endpoint errors.
        const fallbackQs = new URLSearchParams({
          theme: 'all',
          trend: 'all',
          limit: '20',
          include_narrative: '0',
        });
        const fallbackRes = await apiCall(`/api/report/query?${fallbackQs.toString()}`);
        if (fallbackRes.ok) {
          const fallbackPayload = await fallbackRes.json();
          setData(fallbackPayload);
          setLastUpdated(new Date());
          return;
        }
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      setData(payload);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e?.message || 'Failed to load report');
    } finally {
      setLoading(false);
      setLoadingDetails(false);
    }
  };

  const fetchScopedReport = async (overrides = {}) => {
    setGenerating(true);
    setError(null);
    try {
      const effectiveTheme = overrides.theme ?? theme;
      const effectiveTrend = overrides.trend ?? trend;
      const effectiveStartDate = overrides.start_date ?? startDate;
      const effectiveEndDate = overrides.end_date ?? endDate;
      const qs = new URLSearchParams({
        theme: effectiveTheme,
        trend: effectiveTrend,
        limit: String(queryLimit || 20),
        include_narrative: '0',
      });
      if (effectiveStartDate) qs.set('start_date', effectiveStartDate);
      if (effectiveEndDate) qs.set('end_date', effectiveEndDate);
      const res = await apiCall(`/api/report/query?${qs.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      setData(payload);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e?.message || 'Failed to generate scoped report');
    } finally {
      setGenerating(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isAllTime) {
      fetchReport({ fast: true });
    } else {
      fetchScopedReport(scope.params);
    }
  }, [isAllTime, scope?.params]); // fetchReport/fetchScopedReport are stable mount-only fetchers

  useEffect(() => {
    setDetailsReady(false);
    if (!data) return undefined;

    let cancelled = false;
    let timeoutId = null;
    let idleId = null;
    const activate = () => {
      if (!cancelled) {
        setDetailsReady(true);
      }
    };

    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
      idleId = window.requestIdleCallback(activate, { timeout: 250 });
    } else {
      timeoutId = window.setTimeout(activate, 120);
    }

    return () => {
      cancelled = true;
      if (idleId !== null && typeof window !== 'undefined' && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
      if (timeoutId !== null && typeof window !== 'undefined') {
        window.clearTimeout(timeoutId);
      }
    };
  }, [data]);

  const handleExport = () => {
    if (!data) return;
    const md = generateMarkdown(data);
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const reportData = data || {};
  const s = reportData.summary || {};
  const top = reportData.top_programs || [];
  const topExpanded = reportData.top_programs_expanded || [];
  const reportActionEligibility = reportData.action_eligibility || {};
  const mergedEligibilityByResultId = useMemo(() => ({
    ...(eligibilityByResultId || {}),
    ...reportActionEligibility,
  }), [eligibilityByResultId, reportActionEligibility]);
  const experiments = reportData.recent_experiments || [];
  const ops = reportData.op_success_rates || [];
  const failures = reportData.failure_patterns || {};
  const frontier = reportData.efficiency_frontier || [];
  const grammarWeights = reportData.grammar_weights || {};
  const insights = reportData.insights || [];
  const learningLog = reportData.learning_log || [];
  const crossRunStability = reportData.cross_run_stability || {};
  const mathFamilyCoverage = reportData.math_family_coverage || { families: [], totals: { n_tested: 0, n_survived: 0 } };
  const mathspaceOperatorImpact = reportData.mathspace_operator_impact || null;
  const routingModeComparison = reportData.routing_mode_comparison || null;
  const architectureRerunTelemetry = reportData.architecture_rerun_telemetry || {};
  const stabilitySummary = crossRunStability.summary || {};
  const stabilityCandidates = crossRunStability.candidates || [];

  const {
    sortKey: stabilitySortKey,
    sortDesc: stabilitySortDesc,
    filterQuery: stabilityFilter,
    setFilterQuery: setStabilityFilter,
    sortedRows: sortedStabilityCandidates,
    handleSort: handleStabilitySort,
  } = useInteractiveTable({
    rows: stabilityCandidates,
    filterFields: ['graph_fingerprint', 'trend', 'latest_rank', 'previous_rank'],
    initialSortKey: 'latest_rank',
    initialSortDesc: true,
  });

  const {
    totalProg,
    s1Survivors,
    s1Rate,
    sortedOps,
    bestOps,
    worstOps,
    confidenceFactors,
    confidenceScore,
    confidenceBand,
    confidenceWarnings,
    confidenceStrengths,
    decisionReadyCount,
    baselineEvidenceCount,
    baselineWinCount,
    baselineWinInterval,
    averagePromotionScore,
    fullReproPacketCount,
    avgReproCompleteness,
  } = useMemo(() => {
    const totalPrograms = s.total_programs_evaluated || 0;
    const totalSurvivors = s.stage1_survivors ?? s.total_s1_passed ?? 0;
    const rate = totalPrograms > 0 ? (totalSurvivors / totalPrograms * 100).toFixed(1) : '0.0';
    const opsSorted = Array.isArray(ops)
      ? [...ops].sort((a, b) => (b.s1_rate || 0) - (a.s1_rate || 0))
      : [];
    const best = opsSorted.filter(op => (op.s1_rate || 0) > 0).slice(0, 10);
    const worst = opsSorted.filter(op => (op.s1_rate || 0) === 0 && (op.total_count || 0) > 5).slice(0, 10);
    const confidence = {
      experiments: Math.min(1, (s.total_experiments || 0) / 5),
      programs: Math.min(1, totalPrograms / 500),
      rankings: Math.min(1, top.length / 10),
      opCoverage: Math.min(1, opsSorted.length / 8),
    };
    const score = Math.round(((confidence.experiments + confidence.programs + confidence.rankings + confidence.opCoverage) / 4) * 100);
    const band = score >= 75
      ? { label: 'High confidence', color: 'var(--accent-green)' }
      : score >= 45
        ? { label: 'Moderate confidence', color: 'var(--accent-yellow)' }
        : { label: 'Low confidence', color: 'var(--accent-red)' };
    const warnings = [
      (s.total_experiments || 0) < 3 ? 'Fewer than 3 experiments: trends can change quickly with one additional run.' : null,
      totalPrograms < 200 ? `Only ${totalPrograms} programs evaluated: ranking order is still volatile.` : null,
      top.length < 5 ? 'Discovery ranking depth is shallow (<5 candidates).' : null,
      opsSorted.length < 4 ? 'Limited op-level coverage: "What Works" and "What Doesn\'t Work" are early signals only.' : null,
    ].filter(Boolean);
    const strengths = [
      (s.total_experiments || 0) >= 5 ? `${s.total_experiments || 0} experiments provide multi-run evidence.` : null,
      totalPrograms >= 500 ? `${totalPrograms.toLocaleString()} programs reduce random ranking swings.` : null,
      top.length >= 10 ? `${top.length} ranked discoveries improve selection confidence.` : null,
      opsSorted.length >= 8 ? `${opsSorted.length} ops observed gives broader operation-level signal.` : null,
    ].filter(Boolean);
    const readyCount = top.filter(program => decisionGate(program).decisionReady).length;
    const evidenceCount = top.filter(program => program.baseline_loss_ratio != null).length;
    const winCount = top.filter(program => program.baseline_loss_ratio != null && program.baseline_loss_ratio < 1.0).length;
    const winInterval = wilsonInterval(winCount, evidenceCount);
    const promotionRows = top.map(program => promotionEvidenceView(program));
    const promotionScore = promotionRows.length > 0
      ? Math.round(promotionRows.reduce((sum, row) => sum + row.score, 0) / promotionRows.length)
      : 0;
    const reproducibilityRows = top.map(program => reproducibilityPacketStatus(program));
    const fullPackets = reproducibilityRows.filter(row => row.readyCount === row.totalChecks).length;
    const avgCompleteness = reproducibilityRows.length > 0
      ? Math.round((reproducibilityRows.reduce((sum, row) => sum + row.readyCount / row.totalChecks, 0) / reproducibilityRows.length) * 100)
      : 0;
    return {
      totalProg: totalPrograms,
      s1Survivors: totalSurvivors,
      s1Rate: rate,
      sortedOps: opsSorted,
      bestOps: best,
      worstOps: worst,
      confidenceFactors: confidence,
      confidenceScore: score,
      confidenceBand: band,
      confidenceWarnings: warnings,
      confidenceStrengths: strengths,
      decisionReadyCount: readyCount,
      baselineEvidenceCount: evidenceCount,
      baselineWinCount: winCount,
      baselineWinInterval: winInterval,
      averagePromotionScore: promotionScore,
      fullReproPacketCount: fullPackets,
      avgReproCompleteness: avgCompleteness,
    };
  }, [ops, s, top]);
  const uniqueFingerprintCount = Number(architectureRerunTelemetry.unique_fingerprint_count || 0);
  const totalResultRows = Number(architectureRerunTelemetry.total_result_rows || 0);
  const repeatResultRows = Number(architectureRerunTelemetry.repeat_result_rows || 0);
  const rerunRatioPercent = Number(architectureRerunTelemetry.rerun_ratio || 0) * 100;
  const topFingerprintConcentrationPercent = Number(architectureRerunTelemetry.top_fingerprint_concentration || 0) * 100;

  const failureByType = failures.by_error_type || failures;
  const failureByStage = failures.by_stage || {};

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <button
        className="refresh-btn"
        onClick={onBack}
        style={{ alignSelf: 'flex-start', fontSize: 12, padding: '4px 10px' }}
      >
        &larr; Back to Reports
      </button>
      <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading report...</p></div>
    </div>
  );
  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <button
        className="refresh-btn"
        onClick={onBack}
        style={{ alignSelf: 'flex-start', fontSize: 12, padding: '4px 10px' }}
      >
        &larr; Back to Reports
      </button>
      <div className="card"><p style={{ color: 'var(--accent-red)' }}>Error loading report: {error}</p></div>
    </div>
  );
  if (!data) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Back button */}
      <button
        className="refresh-btn"
        onClick={onBack}
        style={{ alignSelf: 'flex-start', fontSize: 12, padding: '4px 10px' }}
      >
        &larr; Back to Reports
      </button>

      {/* Header + Export */}
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div className="card-title" style={{ marginBottom: 4 }}>
            {scope?.label || 'Research Report'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Consolidated findings from {s.total_experiments || 0} experiments
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: {isAllTime ? '/api/report' : '/api/report/query'}
          </div>
        </div>
        <button className="start-btn" onClick={handleExport} style={{ padding: '8px 16px', fontSize: 13 }}>
          Export Markdown
        </button>
        <button
          className="refresh-btn"
          onClick={() => setDeclutterMode((v) => !v)}
          style={{ marginLeft: 8, padding: '8px 12px', fontSize: 12 }}
        >
          {declutterMode ? 'Show Detailed Report' : 'Declutter Report'}
        </button>
      </div>

      <div className="card">
        <div className="card-title">Generate Report by Date / Theme / Trend</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Build a scoped report without loading all heavy diagnostics. Use full details only when needed.
        </p>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: 10,
          alignItems: 'end',
        }}>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Start date
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ width: '100%', marginTop: 4, background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-primary)', borderRadius: 4, padding: '6px 8px' }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            End date
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ width: '100%', marginTop: 4, background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-primary)', borderRadius: 4, padding: '6px 8px' }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Theme
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              style={{ width: '100%', marginTop: 4, background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-primary)', borderRadius: 4, padding: '6px 8px' }}
            >
              <option value="all">All</option>
              <option value="sparsity">Sparsity</option>
              <option value="compression">Compression</option>
              <option value="routing">Routing</option>
              <option value="mathspace">Mathspace</option>
              <option value="failure_modes">Failure modes</option>
            </select>
          </label>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Trend
            <select
              value={trend}
              onChange={(e) => setTrend(e.target.value)}
              style={{ width: '100%', marginTop: 4, background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-primary)', borderRadius: 4, padding: '6px 8px' }}
            >
              <option value="all">All</option>
              <option value="improving">Improving</option>
              <option value="declining">Declining</option>
              <option value="plateaued">Plateaued</option>
              <option value="high_novelty">High novelty</option>
              <option value="high_survival">High survival</option>
            </select>
          </label>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Top-K
            <input
              type="number"
              min={5}
              max={120}
              value={queryLimit}
              onChange={(e) => setQueryLimit(Math.max(5, Math.min(120, parseInt(e.target.value || '20', 10))))}
              style={{ width: '100%', marginTop: 4, background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-primary)', borderRadius: 4, padding: '6px 8px' }}
            />
          </label>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
          <button
            className="start-btn"
            onClick={() => fetchScopedReport()}
            disabled={generating}
            style={{ padding: '6px 12px', fontSize: 12 }}
          >
            {generating ? 'Generating...' : 'Generate Scoped Report'}
          </button>
          <button
            className="refresh-btn"
            onClick={() => fetchReport({ fast: true })}
            style={{ padding: '6px 12px', fontSize: 12 }}
          >
            Reset to Fast Overview
          </button>
          <button
            className="refresh-btn"
            onClick={() => fetchReport({ fast: false })}
            disabled={loadingDetails}
            style={{ padding: '6px 12px', fontSize: 12 }}
          >
            {loadingDetails ? 'Loading full details...' : 'Load Full Details'}
          </button>
        </div>
        {data?.query && (
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            Query: theme={data.query.theme || 'all'} · trend={data.query.trend || 'all'} · matches: {data.query.matched_experiments || 0} experiments / {data.query.matched_programs || 0} programs
          </div>
        )}
      </div>

      {declutterMode && (
        <div className="card" style={{ borderLeft: '3px solid var(--accent-yellow)' }}>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Declutter mode: executive signal first. Detailed diagnostics are hidden.
          </div>
        </div>
      )}

      {/* Executive Summary */}
      <div className="card">
        <div className="card-title">Executive Summary</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Fast snapshot of search productivity and quality. Use this first to decide whether to inspect rankings,
          failure patterns, or grammar updates in more detail.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginBottom: 16 }}>
          <StatCard label="Experiments" value={s.total_experiments || 0} color="var(--accent-blue)" />
          <StatCard label="Programs Tested" value={totalProg.toLocaleString()} color="var(--accent-purple)" />
          <StatCard label="S1 Survivors" value={s1Survivors} color="var(--accent-green)" />
          <StatCard label="S1 Pass Rate" value={`${s1Rate}%`} color={parseFloat(s1Rate) > 5 ? 'var(--accent-green)' : 'var(--accent-yellow)'} />
          <StatCard label="Novel" value={s.total_novel || 0} color="var(--accent-yellow)" />
        </div>
        {data.narrative && (
          <div style={{
            padding: 16, background: 'var(--bg-tertiary)', borderRadius: 6,
            borderLeft: '3px solid var(--accent-purple)', fontSize: 13,
            lineHeight: 1.6, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap',
          }}>
            <div style={{ fontSize: 11, color: 'var(--accent-purple)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase' }}>
              Aria's Narrative
            </div>
            {data.narrative}
          </div>
        )}
      </div>

      {!declutterMode && (
        detailsReady ? (
        <>
      <div className="card">
        <div className="card-title">How to Read This Report</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          <div><strong>1. Discovery Rankings:</strong> pick candidates worth follow-up and open their full program details.</div>
          <div><strong>2. Timeline + What Works/Doesn't:</strong> verify whether trends are stable across experiments.</div>
          <div><strong>3. Grammar Evolution + Frontier:</strong> check if learned generation policy is moving toward better efficiency.</div>
          <div><strong>4. Insights:</strong> turn repeated patterns into next experiment hypotheses.</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Metric Glossary</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          <div><strong>Loss Ratio:</strong> lower is better; compares post-training loss scale between candidates.</div>
          <div><strong>Baseline Loss Ratio:</strong> candidate loss versus fixed baseline; below 1.0 means candidate beats baseline.</div>
          <div><strong>Novelty Score:</strong> structural/behavioral difference signal; higher means less similar to prior programs.</div>
          <div><strong>Discovery Score:</strong> triage composite from loss, novelty, baseline comparison, and identity bonus.</div>
          <div><strong>S1 Survivor:</strong> program passed stage-1 learning evaluation and is eligible for deeper review.</div>
          <div><strong>CKA Source:</strong> `artifact` means reference-backed similarity; `fallback` means heuristic fallback path.</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Confidence & Data Sufficiency</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          This callout estimates how stable current conclusions are based on sample size and coverage.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Current confidence:</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: confidenceBand.color }}>
            {confidenceBand.label} ({confidenceScore}%)
          </span>
        </div>
        <ConfidenceInfographic
          factors={confidenceFactors}
          decisionReadyCount={decisionReadyCount}
          totalCandidates={top.length || 0}
          avgPromotionScore={averagePromotionScore}
          avgReproCompleteness={avgReproCompleteness}
        />
        <div style={{ display: 'grid', gap: 6, marginBottom: 10 }}>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Repro packet completeness: {avgReproCompleteness}% ({fullReproPacketCount}/{top.length || 0} fully ready)
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Baseline win rate with evidence: {baselineEvidenceCount > 0 ? `${((baselineWinCount / baselineEvidenceCount) * 100).toFixed(1)}%` : 'n/a'}
            {baselineWinInterval ? ` (95% CI ${(baselineWinInterval.low * 100).toFixed(1)}-${(baselineWinInterval.high * 100).toFixed(1)}%)` : ''}
          </div>
        </div>
        {confidenceWarnings.length > 0 && (
          <div style={{ marginBottom: confidenceStrengths.length > 0 ? 8 : 0 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Cautions
            </div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {confidenceWarnings.map((item, idx) => (
                <li key={`${item}-${idx}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {confidenceStrengths.length > 0 && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Supporting signals
            </div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {confidenceStrengths.map((item, idx) => (
                <li key={`${item}-${idx}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Unique Architectures vs Reruns</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Concentration telemetry clarifies whether current learning signals come from architecture breadth
          or repeated reruns of a few fingerprints.
        </p>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12, color: 'var(--text-secondary)' }}>
          <span><strong style={{ color: 'var(--accent-green)' }}>Unique fingerprints:</strong> {uniqueFingerprintCount}</span>
          <span><strong style={{ color: 'var(--text-muted)' }}>Rows:</strong> {totalResultRows}</span>
          <span><strong style={{ color: rerunRatioPercent >= 60 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Rerun ratio:</strong> {rerunRatioPercent.toFixed(1)}%</span>
          <span><strong style={{ color: topFingerprintConcentrationPercent >= 35 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Top fingerprint concentration:</strong> {topFingerprintConcentrationPercent.toFixed(1)}%</span>
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
          Repeat rows: {repeatResultRows} · Weighting mode: {architectureRerunTelemetry.weighting_mode || 'unknown'}
        </div>
      </div>

      {/* Discovery Rankings */}
      {top.length > 0 && (
        <DiscoveryRankings
          programs={top}
          expandedPrograms={topExpanded}
          onSelectProgram={onSelectProgram}
          onInvestigate={onInvestigate}
          onCapabilityRank={onCapabilityRank}
          onValidate={onValidate}
          onOpenInDesigner={onOpenInDesigner}
          onQueueAdd={onQueueAdd}
          onQueueRemove={onQueueRemove}
          queuedResultIds={queuedResultIds}
          eligibilityByResultId={mergedEligibilityByResultId}
        />
      )}

      {/* Alternatives to Attention */}
      {top.length > 0 && <AlternativesToAttention programs={top} />}

      {/* Functional family coverage evidence */}
      {mathFamilyCoverage.families?.length > 0 && <FunctionalFamilyEvidence coverage={mathFamilyCoverage} />}

      {/* Mathspace operator impact evidence */}
      {mathspaceOperatorImpact?.available && <MathspaceOperatorImpact impact={mathspaceOperatorImpact} />}

      {/* Routing Mode Comparison */}
      {(top.length > 0 || routingModeComparison?.available) && (
        <RoutingModeComparison programs={top} comparison={routingModeComparison} />
      )}

      {/* Compression Technique Coverage */}
      {top.length > 0 && <CompressionTechniqueCoverage programs={top} />}

      {stabilityCandidates.length > 0 && (
        <div className="card">
          <div className="card-title">Cross-Run Stability</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Rank movement for top candidates across recent completed experiments. Use this to avoid overreacting to single-run spikes.
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 10, fontSize: 11, color: 'var(--text-muted)' }}>
            <span>Stable: {stabilitySummary.stable || 0}</span>
            <span>Up: {stabilitySummary.up || 0}</span>
            <span>Down: {stabilitySummary.down || 0}</span>
            <span>New: {stabilitySummary.new || 0}</span>
            <span>Window: {crossRunStability.window_size || 0} runs</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
            <input
              value={stabilityFilter}
              onChange={(e) => setStabilityFilter(e.target.value)}
              placeholder="Filter fingerprints"
              className="filter-input"
            />
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table table-compact">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                  <th scope="col" onClick={() => handleStabilitySort('graph_fingerprint')} style={{ padding: '6px', cursor: 'pointer' }}>
                    Fingerprint<SortIndicator active={stabilitySortKey === 'graph_fingerprint'} desc={stabilitySortDesc} />
                  </th>
                  <th scope="col" onClick={() => handleStabilitySort('trend')} style={{ padding: '6px', cursor: 'pointer' }}>
                    Trend<SortIndicator active={stabilitySortKey === 'trend'} desc={stabilitySortDesc} />
                  </th>
                  <th scope="col" onClick={() => handleStabilitySort('latest_rank')} style={{ padding: '6px', cursor: 'pointer' }}>
                    Latest Rank<SortIndicator active={stabilitySortKey === 'latest_rank'} desc={stabilitySortDesc} />
                  </th>
                  <th scope="col" onClick={() => handleStabilitySort('previous_rank')} style={{ padding: '6px', cursor: 'pointer' }}>
                    Previous Rank<SortIndicator active={stabilitySortKey === 'previous_rank'} desc={stabilitySortDesc} />
                  </th>
                  <th scope="col" onClick={() => handleStabilitySort('seen_runs')} style={{ padding: '6px', cursor: 'pointer' }}>
                    Seen Runs<SortIndicator active={stabilitySortKey === 'seen_runs'} desc={stabilitySortDesc} />
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedStabilityCandidates.slice(0, 12).map(candidate => {
                  const trendColor = candidate.trend === 'up'
                    ? 'var(--accent-green)'
                    : candidate.trend === 'down'
                      ? 'var(--accent-red)'
                      : candidate.trend === 'stable'
                        ? 'var(--accent-yellow)'
                        : 'var(--text-muted)';
                  return (
                    <tr key={candidate.result_id || candidate.graph_fingerprint} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace' }}>
                        {(candidate.graph_fingerprint || '').slice(0, 12)}
                      </td>
                      <td style={{ padding: '6px', color: trendColor, fontWeight: 600, textTransform: 'uppercase' }}>
                        {candidate.trend || 'unknown'}
                      </td>
                      <td style={{ padding: '6px' }}>{candidate.latest_rank ?? '--'}</td>
                      <td style={{ padding: '6px' }}>{candidate.previous_rank ?? '--'}</td>
                      <td style={{ padding: '6px' }}>{candidate.seen_runs ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Experiment Timeline */}
      {experiments.length > 0 && (
        <div className="card">
          <div className="card-title">Experiment Timeline</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Chronological view of experiments showing how pass rates and discovery quality evolved over the search.
          </p>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {experiments.map((exp, i) => {
              const s1 = exp.n_stage1_passed || 0;
              const total = exp.n_programs || 0;
              const confirmed = s1 > 0;
              return (
                <div key={exp.experiment_id || i} style={{
                  padding: '8px 12px', borderBottom: '1px solid var(--border)',
                  display: 'flex', gap: 12, alignItems: 'center',
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: confirmed ? 'var(--accent-green)' : 'var(--accent-red)',
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {exp.hypothesis ? `"${exp.hypothesis.slice(0, 80)}"` : `Experiment ${exp.experiment_id?.slice(0, 8)}`}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {exp.experiment_type || 'synthesis'} | {total} programs | {s1} S1 | {exp.created_at?.slice(0, 16)}
                    </div>
                  </div>
                  <span style={{
                    fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
                    background: confirmed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                    color: confirmed ? 'var(--accent-green)' : 'var(--accent-red)',
                  }}>
                    {confirmed ? 'Confirmed' : 'Refuted'}
                  </span>
                  {onSelectExperiment && exp.experiment_id && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '4px 8px', marginLeft: 8 }}
                      onClick={() => onSelectExperiment(exp.experiment_id)}
                      aria-label={`Open experiment ${exp.experiment_id}`}
                    >
                      Open
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* What Works + What Doesn't Work */}
      {(bestOps.length > 0 || worstOps.length > 0) && (
        <MirroredOpsChart bestOps={bestOps} worstOps={worstOps} rows={6} />
      )}

      {data.template_hit_rates && Object.keys(data.template_hit_rates).length > 0 && (
        <TemplatePerformance hitRates={data.template_hit_rates} />
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-title">What Works</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Operation types and patterns that consistently appear in successful architectures that passed Stage 1 learning evaluation.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Use this section as a whitelist for future hypotheses: prioritize ops/combinations with repeatable S1 success.
          </p>
          {bestOps.length > 0 ? (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Top Performing Ops</div>
              {bestOps.map((op, i) => (
                <div key={op.op_name || i} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '4px 0',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace' }}>{op.op_name}</span>
                  <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ fontSize: 12, color: 'var(--accent-green)', fontWeight: 600 }}>
                      {((op.s1_rate || 0) * 100).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      ({op.s1_count ?? Math.round((op.s1_rate || 0) * (op.total_count || 0))}/{op.total_count || 0})
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: reliabilityBand(op.total_count || 0).color,
                      }}
                      title="Reliability from sample size: high (>=30), medium (12-29), low (<12)."
                    >
                      {reliabilityBand(op.total_count || 0).label}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          ) : <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Insufficient data</p>}

          {data.structural_correlations && Object.keys(data.structural_correlations).length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Structural Correlations</div>
              {Object.entries(data.structural_correlations)
                .filter(([, v]) => Math.abs(v) > 0.1)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 8)
                .map(([key, val]) => (
                  <div key={key} style={{
                    display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                    borderBottom: '1px solid var(--border)',
                  }}>
                    <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{key}</span>
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: val > 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                      {val > 0 ? '+' : ''}{val.toFixed(3)}
                    </span>
                  </div>
                ))}
            </div>
          )}

          {data.top_op_combinations && data.top_op_combinations.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Best Op Combinations</div>
              {data.top_op_combinations.slice(0, 5).map((combo, i) => (
                <div key={i} style={{ fontSize: 11, padding: '3px 0', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                  {combo.ops ? combo.ops.join(' + ') : JSON.stringify(combo)}
                  {combo.s1_rate != null && (
                    <span style={{ marginLeft: 8, color: 'var(--accent-green)', fontWeight: 600 }}>
                      {(combo.s1_rate * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">What Doesn't Work</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Operation types and patterns that consistently lead to failure — compilation errors, numerical instability, or inability to learn.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Use this section as a blacklist: reduce or constrain these patterns in upcoming runs to save search budget.
          </p>
          {Object.keys(failureByType).length > 0 || Object.keys(failureByStage).length > 0 ? (
            <>
              {Object.keys(failureByStage).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Failures by Stage</div>
                  {Object.entries(failureByStage).map(([stage, count]) => (
                    <div key={stage} style={{
                      display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                      borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ fontSize: 12 }}>{stage}</span>
                      <span style={{ fontSize: 12, color: 'var(--accent-red)' }}>{count}</span>
                    </div>
                  ))}
                </div>
              )}
              {typeof failureByType === 'object' && !Array.isArray(failureByType) && Object.keys(failureByType).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Failures by Error Type</div>
                  {Object.entries(failureByType).slice(0, 10).map(([errType, count]) => (
                    <div key={errType} style={{
                      display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                      borderBottom: '1px solid var(--border)', gap: 8,
                    }}>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{errType}</span>
                      <span style={{ fontSize: 11, color: 'var(--accent-red)', flexShrink: 0 }}>{typeof count === 'number' ? count : JSON.stringify(count)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>No failure data yet</p>}

          {worstOps.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Worst Performing Ops (0% S1)</div>
              {worstOps.map((op, i) => (
                <div key={op.op_name || i} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{op.op_name}</span>
                  <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      0/{op.total_count || 0} S1
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      ({op.total_count || 0} uses)
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: reliabilityBand(op.total_count || 0).color,
                      }}
                      title="Reliability from sample size: high (>=30), medium (12-29), low (<12)."
                    >
                      {reliabilityBand(op.total_count || 0).label}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Do Not Pursue */}
      <NegativeResultsSummary />

      {/* Grammar Evolution */}
      {grammarWeights.learned && grammarWeights.default && (
        <div className="card">
          <div className="card-title">Grammar Evolution</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            How the generation weights shifted over time. Rising bars mean the system generates more of that operation; falling bars mean it learned to avoid it.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Treat large weight deltas as policy changes; verify they align with the "What Works" and "What Doesn't Work" evidence above.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Weight Changes</div>
              {Object.keys({ ...grammarWeights.default, ...grammarWeights.learned }).sort().map(cat => {
                const old_w = grammarWeights.default[cat] || 1.0;
                const new_w = grammarWeights.learned ? (grammarWeights.learned[cat] || old_w) : old_w;
                const changed = Math.abs(new_w - old_w) > 0.1;
                return (
                  <div key={cat} style={{
                    display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                    borderBottom: '1px solid var(--border)',
                    opacity: changed ? 1 : 0.5,
                  }}>
                    <span style={{ fontSize: 12 }}>{cat}</span>
                    <span style={{ fontSize: 12 }}>
                      <span style={{ color: 'var(--text-muted)' }}>{old_w.toFixed(1)}</span>
                      {changed && (
                        <>
                          <span style={{ color: 'var(--text-muted)', margin: '0 4px' }}>&rarr;</span>
                          <span style={{
                            fontWeight: 600,
                            color: new_w > old_w ? 'var(--accent-green)' : 'var(--accent-red)',
                          }}>
                            {new_w.toFixed(1)}
                          </span>
                        </>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
            {learningLog.length > 0 && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Recent Weight Changes</div>
                <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                  {learningLog.slice(0, 10).map((entry, i) => (
                    <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: 11 }}>
                      <div style={{ color: 'var(--text-secondary)' }}>{entry.description || entry.event_type}</div>
                      <div style={{ color: 'var(--text-muted)' }}>{entry.created_at?.slice(0, 16)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Efficiency Frontier */}
      {frontier.length > 0 && (
        <div className="card">
          <div className="card-title">Efficiency Frontier</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Trade-off between model size (parameters) and learning speed (loss ratio). Points on the frontier are the best architectures at each size — nothing else learns faster for the same parameter budget.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Choose candidates on this curve when you need better learning with limited compute budget.
          </p>
          <EfficiencyChart frontier={frontier} showLabels labelCount={6} onSelectProgram={onSelectProgram} />
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
            {frontier.length} Pareto-optimal programs (lower loss, fewer FLOPs = better)
          </div>
        </div>
      )}
        </>
        ) : (
        <div className="card" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
          <div className="card-title" style={{ marginBottom: 8 }}>Preparing Interactive Diagnostics</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0, lineHeight: 1.6 }}>
            Executive summary is ready. The heavier report tables and charts are mounting after first paint so the rest of the dashboard remains responsive.
          </p>
        </div>
        )
      )}

      {/* Insights / Recommendations */}
      {insights.length > 0 && (
        <div className="card">
          <div className="card-title">Insights & Recommendations</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Key takeaways and suggested next steps synthesized from all experiments.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Convert high-confidence items into explicit hypotheses so they can be validated in campaign and timeline views.
          </p>
          {insights.slice(0, 15).map((ins, i) => (
            <div key={i} style={{
              padding: '8px 12px', borderBottom: '1px solid var(--border)',
              display: 'flex', gap: 8, alignItems: 'flex-start',
            }}>
              <span style={{
                fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 3,
                background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                textTransform: 'uppercase', flexShrink: 0,
              }}>
                {ins.category || 'insight'}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1 }}>
                {ins.content || (typeof ins === 'string' ? ins : JSON.stringify(ins))}
              </span>
              {onHypothesisHandoff && (ins.category === 'hypothesis' || ins.category === 'success_factor') && (
                <button
                  className="refresh-btn"
                  style={{ fontSize: 10, padding: '1px 6px', flexShrink: 0 }}
                  onClick={() => onHypothesisHandoff({
                    source: 'report-insight',
                    hypothesis: ins.content || (typeof ins === 'string' ? ins : ''),
                    objective: `Test insight: ${(ins.content || '').slice(0, 80)}`,
                    suggestedMode: 'single',
                  })}
                  aria-label="Use this insight as experiment hypothesis"
                >
                  Use as Hypothesis
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(ReportDetail);
