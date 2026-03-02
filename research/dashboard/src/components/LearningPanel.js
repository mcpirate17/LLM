import React, { useState, useEffect, useMemo } from 'react';
import { useAriaData } from '../hooks/useAriaData';
import { NarrativeProvider } from '../hooks/useNarrative';

// Shared
import Section from './shared/Section';

// Learning Components
import TargetBalanceCards from './learning/TargetBalanceCards';
import GrammarWeightsChart from './learning/GrammarWeightsChart';
import OpSuccessTable from './learning/OpSuccessTable';
import AdaptationSummary from './learning/AdaptationSummary';
import LearningLog from './learning/LearningLog';
import EfficiencyFrontier from './learning/EfficiencyFrontier';
import LearningTrajectory from './learning/LearningTrajectory';
import ExperimentClusters from './learning/ExperimentClusters';
import RoutingHealth from './learning/RoutingHealth';
import GatingBehaviorDiagnostics from './learning/GatingBehaviorDiagnostics';
import MathFamilyCoverage from './learning/MathFamilyCoverage';
import MathspaceImpact from './learning/MathspaceImpact';
import CompressionCoverage from './learning/CompressionCoverage';
import WhatIHaveLearned from './learning/WhatIHaveLearned';
import ControlComparison from './learning/ControlComparison';
import ArchitectureRerunTelemetry from './learning/ArchitectureRerunTelemetry';
import FingerprintDiagnosticsCard from './learning/FingerprintDiagnosticsCard';
import DataAccumulation from './learning/DataAccumulation';
import AriaThoughtProcess from './learning/AriaThoughtProcess';
import FeedbackLoopSummary from './learning/FeedbackLoopSummary';
import InsightSynergyMatrix from './learning/InsightSynergyMatrix';
import DataPopulateBar from './learning/DataPopulateBar';
import StrategyBacktest from './learning/StrategyBacktest';

const API_BASE = process.env.REACT_APP_API_URL || '';
const MIN_SAMPLES = 5;

function sampleCount(data, kind) {
  if (!data) return 0;
  if (kind === 'clusters') return data.n_experiments || 0;
  if (kind === 'routing') return data.total_programs || 0;
  if (kind === 'gating') return data.total_routed_programs || 0;
  return 0;
}

function computeWeightedAverage(rows, key) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  let total = 0;
  let weightSum = 0;
  for (const row of rows) {
    const weight = Number(row?.n_programs || 0);
    const raw = row?.[key];
    if (raw == null || weight <= 0) continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) continue;
    total += value * weight;
    weightSum += weight;
  }
  if (weightSum <= 0) return null;
  return total / weightSum;
}

function computeTargetSummary(programs, routingData) {
  const rows = Array.isArray(programs) ? programs : [];
  const takeAvg = (key) => {
    const vals = rows
      .map(r => r?.[key])
      .filter(v => v != null)
      .map(v => Number(v))
      .filter(v => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };
  const takeMedian = (key) => {
    const vals = rows
      .map(r => r?.[key])
      .filter(v => v != null)
      .map(v => Number(v))
      .filter(v => Number.isFinite(v))
      .sort((a, b) => a - b);
    if (!vals.length) return null;
    const mid = Math.floor(vals.length / 2);
    return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  };

  const routingRows = routingData?.by_mode || [];
  const routingHasTelemetry = routingRows.some(r => (
    r.token_retention != null
    || r.avg_drop_rate != null
    || r.avg_utilization_entropy != null
    || r.avg_confidence_mean != null
    || r.avg_tokens_total != null
    || r.avg_tokens_processed != null
  ));
  const routingRetention = computeWeightedAverage(
    routingRows.map(r => ({
      ...r,
      token_retention: r.token_retention != null ? r.token_retention
        : (r.avg_drop_rate != null ? (1 - r.avg_drop_rate) : null),
    })),
    "token_retention"
  );

  let bestMode = null;
  let bestScore = -Infinity;
  for (const row of routingRows) {
    const retention = row.token_retention != null
      ? row.token_retention
      : (row.avg_drop_rate != null ? (1 - row.avg_drop_rate) : null);
    const entropy = Number(row.avg_utilization_entropy);
    const conf = Number(row.avg_confidence_mean);
    const score = (Number.isFinite(retention) ? retention : 0)
      + (Number.isFinite(entropy) ? entropy : 0)
      + (Number.isFinite(conf) ? conf : 0);
    if (score > bestScore) {
      bestScore = score;
      bestMode = row.routing_mode;
    }
  }

  return {
    efficiency: {
      throughputMedian: takeMedian("throughput_tok_s"),
      paramsMedian: takeMedian("param_count"),
      flopsMedian: takeMedian("flops_forward"),
      sampleCount: rows.length,
    },
    routing: {
      retention: routingRetention,
      entropy: computeWeightedAverage(routingRows, "avg_utilization_entropy"),
      confidence: computeWeightedAverage(routingRows, "avg_confidence_mean"),
      overflow: computeWeightedAverage(routingRows, "avg_capacity_overflow_count"),
      bestMode,
      sampleCount: routingHasTelemetry
        ? (routingData?.routed_programs ?? routingData?.total_programs ?? 0)
        : 0,
    },
    adaptive: {
      depthSavings: takeAvg("depth_savings_ratio"),
      effectiveDepth: takeAvg("effective_depth_ratio"),
      recursionSavings: takeAvg("recursion_savings_ratio"),
      recursionDepth: takeAvg("recursion_depth_ratio"),
      sampleCount: rows.filter(r =>
        r.depth_savings_ratio != null
        || r.effective_depth_ratio != null
        || r.recursion_savings_ratio != null
        || r.recursion_depth_ratio != null
      ).length,
    }
  };
}

/**
 * LearningPanel — Shows grammar weight evolution, op success rates,
 * learning log timeline, and efficiency frontier.
 */
function LearningPanel({ onNavigateStrategy, onStartExperiment }) {
  const {
    learningTrajectory,
    fingerprintDiagnostics,
    mathFamilyCoverage,
    lastUpdated: sharedLastUpdated,
  } = useAriaData() || {};

  const [weights, setWeights] = useState(null);
  const [opRates, setOpRates] = useState(null);
  const [log, setLog] = useState(null);
  const [frontier, setFrontier] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [routingHealth, setRoutingHealth] = useState(null);
  const [routingComparison, setRoutingComparison] = useState(null);
  const [gatingDiagnostics, setGatingDiagnostics] = useState(null);
  const [mathspaceImpact, setMathspaceImpact] = useState(null);
  const [compressionCoverage, setCompressionCoverage] = useState(null);
  const [learningSummary, setLearningSummary] = useState(null);
  const [topPrograms, setTopPrograms] = useState(null);
  const [controlComparison, setControlComparison] = useState(null);
  const [insightInteractions, setInsightInteractions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  const targetSummary = useMemo(
    () => computeTargetSummary(topPrograms, routingComparison || routingHealth),
    [topPrograms, routingComparison, routingHealth]
  );

  const [openSections, setOpenSections] = useState({
    core: true,
    quality: false,
    diagnostics: false,
    raw: false
  });

  const toggleSection = (id) => {
    setOpenSections(prev => ({ ...prev, [id]: !prev[id] }));
  };

  useEffect(() => {
    const safeFetch = (url) => fetch(url).then(r => {
      if (!r.ok) throw new Error(`HTTP \${r.status}`);
      return r.json();
    }).catch(() => null);

    Promise.all([
      safeFetch(`\${API_BASE}/api/analytics/grammar-weights`),
      safeFetch(`\${API_BASE}/api/analytics/op-success`),
      safeFetch(`\${API_BASE}/api/analytics/learning-log`),
      safeFetch(`\${API_BASE}/api/analytics/efficiency-frontier`),
      safeFetch(`\${API_BASE}/api/analytics/experiment-clusters`),
      safeFetch(`\${API_BASE}/api/analytics/routing-health`),
      safeFetch(`\${API_BASE}/api/analytics/routing-comparison`),
      safeFetch(`\${API_BASE}/api/analytics/gating-diagnostics`),
      safeFetch(`\${API_BASE}/api/analytics/mathspace-impact`),
      safeFetch(`\${API_BASE}/api/analytics/compression-coverage`),
      safeFetch(`\${API_BASE}/api/analytics/learning-summary`),
      safeFetch(`\${API_BASE}/api/programs?n=100&sort_by=loss_ratio`),
      safeFetch(`\${API_BASE}/api/analytics/control-comparison`),
      safeFetch(`\${API_BASE}/api/analytics/insight-interactions`),
    ]).then(([w, ops, lg, fr, cl, rh, rc, gd, mi, cc, ls, tp, ctrl, si]) => {
      if (!w && !ops && !lg && !fr && !cl && !rh && !rc && !gd && !mi && !cc && !ls && !si) {
        setError('Failed to load analytics data. The API may be unavailable.');
      }
      setWeights(w);
      setOpRates(ops);
      setLog(lg);
      setFrontier(fr);
      setClusters(cl);
      setRoutingHealth(rh);
      setRoutingComparison(rc);
      setGatingDiagnostics(gd);
      setMathspaceImpact(mi);
      setCompressionCoverage(cc);
      setLearningSummary(ls);
      setTopPrograms(Array.isArray(tp) ? tp : null);
      setControlComparison(ctrl);
      setInsightInteractions(si);
      setLastUpdated(new Date());
      setLoading(false);
    }).catch(e => {
      setError('Failed to load analytics: ' + e.message);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading analytics...</p></div>;
  }

  if (error) {
    return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  }

  return (
    <NarrativeProvider trajectoryData={learningTrajectory} weightData={weights}>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card" style={{ padding: '12px 16px' }}>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
          The AI scientist searches for novel neural network layer designs by generating random
          compositions of operations, testing if they compile and learn, and evolving the search
          grammar toward successful patterns. This tab shows what the system has learned so far.
        </p>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '8px 0 0' }}>
          Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Shared data: {sharedLastUpdated ? new Date(sharedLastUpdated).toLocaleTimeString() : 'loading'}
        </p>
      </div>
      <FeedbackLoopSummary
        weights={weights}
        trajectory={learningTrajectory}
        controlComparison={controlComparison}
        title="Aria's Analysis / Feedback Loop Summary"
      />
      <DataPopulateBar
        learningTrajectory={learningTrajectory}
        controlComparison={controlComparison}
        onStartExperiment={onStartExperiment}
      />

      <Section title="Core Learning" id="core" isOpen={openSections.core} onToggle={toggleSection}>
        <AriaThoughtProcess />
        <WhatIHaveLearned summary={learningSummary} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <LearningTrajectory trajectory={learningTrajectory} onNavigateStrategy={onNavigateStrategy} onStartExperiment={onStartExperiment} />
          <ControlComparison data={controlComparison} onStartExperiment={onStartExperiment} />
        </div>
        <GrammarWeightsChart
          defaultWeights={weights?.default}
          learnedWeights={weights?.learned}
          explanation={weights?.explanation}
          onStartExperiment={onStartExperiment}
        />
      </Section>

      <Section title="Search Quality" id="quality" isOpen={openSections.quality} onToggle={toggleSection}>
        <ArchitectureRerunTelemetry telemetry={weights?.architecture_rerun_telemetry} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <TargetBalanceCards summary={targetSummary} />
          <StrategyBacktest />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <DataAccumulation
            title="Efficiency Frontier"
            current={Array.isArray(frontier) ? frontier.length : 0}
            threshold={3}
          >
            <EfficiencyFrontier frontier={frontier} />
          </DataAccumulation>
          <DataAccumulation
            title="Experiment Clusters"
            current={sampleCount(clusters, 'clusters')}
            threshold={MIN_SAMPLES}
          >
            <ExperimentClusters clustersData={clusters} />
          </DataAccumulation>
        </div>
      </Section>

      <Section title="Advanced Diagnostics" id="diagnostics" isOpen={openSections.diagnostics} onToggle={toggleSection}>
        <FingerprintDiagnosticsCard diagnostics={fingerprintDiagnostics} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <DataAccumulation
            title="Routing Health"
            current={sampleCount(routingComparison || routingHealth, 'routing')}
            threshold={MIN_SAMPLES}
          >
            <RoutingHealth data={routingComparison || routingHealth} />
          </DataAccumulation>
          <DataAccumulation
            title="Gating Behavior Diagnostics"
            current={sampleCount(gatingDiagnostics, 'gating')}
            threshold={MIN_SAMPLES}
          >
            <GatingBehaviorDiagnostics data={gatingDiagnostics} />
          </DataAccumulation>
        </div>
        <MathFamilyCoverage data={mathFamilyCoverage} />
        <MathspaceImpact data={mathspaceImpact} />
        <CompressionCoverage data={compressionCoverage} programs={topPrograms} />
      </Section>

      <Section title="Raw Data" id="raw" isOpen={openSections.raw} onToggle={toggleSection}>
        <AdaptationSummary log={log} />
        <DataAccumulation
          title="Insight Synergy Matrix"
          current={insightInteractions?.total_interactions || (insightInteractions?.synergistic_pairs?.length || 0) + (insightInteractions?.antagonistic_pairs?.length || 0)}
          threshold={5}
        >
          <InsightSynergyMatrix data={insightInteractions} />
        </DataAccumulation>
        <OpSuccessTable opRates={opRates} />
        <LearningLog log={log} />
      </Section>
    </div>
    </NarrativeProvider>
  );
}

export default LearningPanel;
